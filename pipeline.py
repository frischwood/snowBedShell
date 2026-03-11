"""
DEM-to-cfMesh processing pipeline.

Self-contained module with all processing functions.
No CLI dependencies, no sys.exit — raises exceptions on errors.
"""

import io
import json
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.interpolate import NearestNDInterpolator
from pyproj import Transformer
from shapely.geometry import box as shapely_box


# ---------------------------------------------------------------------------
# Step 1: Download DEM tiles from swisstopo STAC API
# ---------------------------------------------------------------------------

def download_dem_tiles(bbox_wgs84, resolution, output_dir, log=print):
    """Download and merge DEM tiles from swisstopo STAC API.

    Returns path to merged GeoTIFF.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "dem_tiles"
    tiles_dir.mkdir(exist_ok=True)

    w, s, e, n = bbox_wgs84
    stac_url = (
        f"https://data.geo.admin.ch/api/stac/v0.9/collections/"
        f"ch.swisstopo.swissalti3d/items?bbox={w},{s},{e},{n}&limit=200"
    )

    log(f"Querying STAC API...")
    resp = requests.get(stac_url, timeout=30)
    resp.raise_for_status()
    items = resp.json()

    features = items.get("features", [])
    if not features:
        raise ValueError("No DEM tiles found for this region.")

    log(f"Found {len(features)} tile(s)")

    res_str = f"{resolution:.1f}" if resolution == 0.5 else str(int(resolution))

    tile_paths = []
    for feat in features:
        assets = feat.get("assets", {})
        tif_url = None
        for key, asset in assets.items():
            href = asset.get("href", "")
            asset_type = asset.get("type", "")
            if "tif" in asset_type or href.endswith(".tif"):
                if f"_{res_str}_" in href or f"_{res_str}m_" in href.lower():
                    tif_url = href
                    break
        if tif_url is None:
            for key, asset in assets.items():
                href = asset.get("href", "")
                if href.endswith(".tif"):
                    tif_url = href
                    break
        if tif_url is None:
            continue

        fname = tif_url.split("/")[-1]
        local_path = tiles_dir / fname

        if local_path.exists():
            log(f"  Cached: {fname}")
        else:
            log(f"  Downloading: {fname}")
            r = requests.get(tif_url, timeout=120, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

        tile_paths.append(local_path)

    if not tile_paths:
        raise ValueError("No tiles downloaded.")

    log(f"Merging {len(tile_paths)} tile(s)...")
    datasets = [rasterio.open(p) for p in tile_paths]
    merged, merged_transform = merge(datasets)
    for ds in datasets:
        ds.close()

    merged_path = output_dir / "dem_merged.tif"
    profile = rasterio.open(tile_paths[0]).profile.copy()
    profile.update(
        height=merged.shape[1],
        width=merged.shape[2],
        transform=merged_transform,
        count=1,
    )
    with rasterio.open(merged_path, "w", **profile) as dst:
        dst.write(merged[0], 1)

    log(f"Merged DEM saved: {merged_path.name}")
    return merged_path


# ---------------------------------------------------------------------------
# Step 2: Process DEM
# ---------------------------------------------------------------------------

def process_dem(dem_path, bbox_lv95, gaussian_sigma=2.0, target_resolution=None, log=print):
    """Process DEM: crop, smooth, subsample, translate to local coords.

    Returns (x, y, Z, origin_lv95).
    """
    with rasterio.open(dem_path) as src:
        e_min, n_min, e_max, n_max = bbox_lv95
        geom = shapely_box(e_min, n_min, e_max, n_max)
        out_image, out_transform = rio_mask(src, [geom.__geo_interface__], crop=True)
        Z = out_image[0].astype(np.float64)
        cell_size = abs(out_transform.a)

    # Handle NoData
    nodata_mask = (Z < -9000) | np.isnan(Z)
    if nodata_mask.any():
        rows, cols = np.where(~nodata_mask)
        vals = Z[~nodata_mask]
        interp = NearestNDInterpolator(list(zip(rows, cols)), vals)
        bad_rows, bad_cols = np.where(nodata_mask)
        Z[nodata_mask] = interp(list(zip(bad_rows, bad_cols)))
        log(f"  Interpolated {nodata_mask.sum()} NoData pixels")

    # Gaussian smoothing
    if gaussian_sigma > 0:
        sigma_px = gaussian_sigma / cell_size
        Z = gaussian_filter(Z, sigma=sigma_px)
        log(f"  Gaussian smoothing: sigma={gaussian_sigma}m ({sigma_px:.1f}px)")

    # Optional subsample
    if target_resolution is not None and target_resolution > cell_size * 1.01:
        factor = int(round(target_resolution / cell_size))
        Z_smooth = uniform_filter(Z, size=factor)
        Z = Z_smooth[::factor, ::factor]
        cell_size *= factor
        log(f"  Subsampled to {cell_size}m ({Z.shape[1]}x{Z.shape[0]})")

    ny, nx = Z.shape

    # Flip so row 0 = south (y increases upward)
    Z = Z[::-1, :]
    x = np.arange(nx) * cell_size
    y = np.arange(ny) * cell_size

    origin_lv95 = (e_min, n_min)
    log(f"  Grid: {nx}x{ny}, res={cell_size}m, Z=[{Z.min():.1f}, {Z.max():.1f}]m")

    return x, y, Z, origin_lv95


# ---------------------------------------------------------------------------
# Step 3: Extend domain with smooth buffer zones
# ---------------------------------------------------------------------------

def extend_domain(x, y, Z, buffer_width=200.0, ref_elevation="mean_edge",
                  domain_height=200.0, log=print):
    """Extend domain with cosine-blended flat buffer zones.

    Returns (x_ext, y_ext, Z_ext, z_top).
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ny, nx = Z.shape

    n_buf_x = int(round(buffer_width / dx))
    n_buf_y = int(round(buffer_width / dy))

    # Reference elevation
    edge_vals = np.concatenate([Z[0, :], Z[-1, :], Z[:, 0], Z[:, -1]])
    if ref_elevation == "mean_edge":
        z_ref = float(np.mean(edge_vals))
    elif ref_elevation == "median_edge":
        z_ref = float(np.median(edge_vals))
    elif ref_elevation == "min_edge":
        z_ref = float(np.min(edge_vals))
    else:
        z_ref = float(ref_elevation)

    z_top = float(Z.max()) + domain_height

    log(f"  Buffer: {buffer_width}m ({n_buf_x}x, {n_buf_y}y cells), z_ref={z_ref:.1f}m")

    # Extended coordinates
    nx_ext = nx + 2 * n_buf_x
    ny_ext = ny + 2 * n_buf_y

    x_ext = np.concatenate([
        x[0] - np.arange(n_buf_x, 0, -1) * dx,
        x,
        x[-1] + np.arange(1, n_buf_x + 1) * dx,
    ])
    y_ext = np.concatenate([
        y[0] - np.arange(n_buf_y, 0, -1) * dy,
        y,
        y[-1] + np.arange(1, n_buf_y + 1) * dy,
    ])

    # Build extended elevation grid with edge values
    Z_ext = np.full((ny_ext, nx_ext), z_ref)
    Z_ext[n_buf_y:n_buf_y + ny, n_buf_x:n_buf_x + nx] = Z

    # Fill buffer zones with nearest edge values
    for i in range(n_buf_x):
        Z_ext[n_buf_y:n_buf_y + ny, i] = Z[:, 0]
        Z_ext[n_buf_y:n_buf_y + ny, n_buf_x + nx + i] = Z[:, -1]
    for j in range(n_buf_y):
        Z_ext[j, n_buf_x:n_buf_x + nx] = Z[0, :]
        Z_ext[n_buf_y + ny + j, n_buf_x:n_buf_x + nx] = Z[-1, :]
    for j in range(n_buf_y):
        for i in range(n_buf_x):
            Z_ext[j, i] = Z[0, 0]
            Z_ext[j, n_buf_x + nx + i] = Z[0, -1]
            Z_ext[n_buf_y + ny + j, i] = Z[-1, 0]
            Z_ext[n_buf_y + ny + j, n_buf_x + nx + i] = Z[-1, -1]

    # Cosine blending: alpha(t) = 0.5 * (1 - cos(pi * t))
    def cosine_alpha(t):
        return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))

    alpha_x = np.zeros(nx_ext)
    if n_buf_x > 0:
        alpha_x[:n_buf_x] = cosine_alpha(np.arange(n_buf_x, 0, -1) / n_buf_x)
        alpha_x[n_buf_x + nx:] = cosine_alpha(np.arange(1, n_buf_x + 1) / n_buf_x)

    alpha_y = np.zeros(ny_ext)
    if n_buf_y > 0:
        alpha_y[:n_buf_y] = cosine_alpha(np.arange(n_buf_y, 0, -1) / n_buf_y)
        alpha_y[n_buf_y + ny:] = cosine_alpha(np.arange(1, n_buf_y + 1) / n_buf_y)

    # Tensor product: alpha = 1 - (1-alpha_x)*(1-alpha_y)
    alpha_2d = 1.0 - (1.0 - alpha_x[np.newaxis, :]) * (1.0 - alpha_y[:, np.newaxis])

    Z_edge = Z_ext.copy()
    Z_ext = (1.0 - alpha_2d) * Z_edge + alpha_2d * z_ref

    log(f"  Extended: {nx_ext}x{ny_ext}, z_ref={z_ref:.1f}m, ceiling={z_top:.1f}m")

    return x_ext, y_ext, Z_ext, z_top, z_ref


# ---------------------------------------------------------------------------
# Step 4: Generate cfMesh surface files
# ---------------------------------------------------------------------------

def triangulate_and_write(x_ext, y_ext, Z_ext, z_top, z_ref, output_dir,
                          write_stl=True, log=print):
    """Triangulate terrain + bounding box, write bbox.fms and optionally bbox.stl.

    The box walls start at z_ref (the terrain edge elevation after buffer
    blending), so they sit flush with the snowBed at the domain boundary.

    Returns (fms_path, stl_path_or_None).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ny, nx = Z_ext.shape
    x_min, x_max = float(x_ext[0]), float(x_ext[-1])
    y_min, y_max = float(y_ext[0]), float(y_ext[-1])
    z_floor = float(z_ref)

    log(f"  Terrain grid: {nx}x{ny}")

    # --- Vertices (vectorized) ---
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    terrain_verts = np.column_stack([
        x_ext[ii.ravel()],
        y_ext[jj.ravel()],
        Z_ext.ravel(),
    ])

    box_corners = np.array([
        [x_min, y_min, z_floor],  # 0: SW bottom
        [x_max, y_min, z_floor],  # 1: SE bottom
        [x_max, y_max, z_floor],  # 2: NE bottom
        [x_min, y_max, z_floor],  # 3: NW bottom
        [x_min, y_min, z_top],    # 4: SW top
        [x_max, y_min, z_top],    # 5: SE top
        [x_max, y_max, z_top],    # 6: NE top
        [x_min, y_max, z_top],    # 7: NW top
    ])

    n_tv = len(terrain_verts)
    all_verts = np.vstack([terrain_verts, box_corners])

    # --- Terrain triangles (vectorized) ---
    jj2, ii2 = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    jf = jj2.ravel()
    iff = ii2.ravel()

    v00 = jf * nx + iff
    v10 = jf * nx + (iff + 1)
    v01 = (jf + 1) * nx + iff
    v11 = (jf + 1) * nx + (iff + 1)

    n_cells = len(jf)
    zeros = np.zeros(n_cells, dtype=np.int64)

    tri1 = np.column_stack([v00, v10, v01, zeros])
    tri2 = np.column_stack([v10, v11, v01, zeros])
    terrain_tris = np.vstack([tri1, tri2])

    log(f"  Terrain triangles: {len(terrain_tris)}")

    # --- Box wall/ceiling triangles ---
    b = n_tv
    box_tris = np.array([
        # xMin (patch 1): normal +x
        [b + 0, b + 3, b + 4, 1],
        [b + 3, b + 7, b + 4, 1],
        # xMax (patch 2): normal -x
        [b + 1, b + 5, b + 2, 2],
        [b + 2, b + 5, b + 6, 2],
        # yMin (patch 3): normal +y
        [b + 0, b + 4, b + 1, 3],
        [b + 1, b + 4, b + 5, 3],
        # yMax (patch 4): normal -y
        [b + 3, b + 2, b + 7, 4],
        [b + 2, b + 6, b + 7, 4],
        # zMax (patch 5): normal -z
        [b + 4, b + 7, b + 5, 5],
        [b + 7, b + 6, b + 5, 5],
    ], dtype=np.int64)

    all_tris = np.vstack([terrain_tris, box_tris])
    log(f"  Total triangles: {len(all_tris)}")

    # --- Feature edges ---
    # 12 bounding box edges
    box_edges = np.array([
        [b + 0, b + 1], [b + 1, b + 2], [b + 2, b + 3], [b + 3, b + 0],
        [b + 4, b + 5], [b + 5, b + 6], [b + 6, b + 7], [b + 7, b + 4],
        [b + 0, b + 4], [b + 1, b + 5], [b + 2, b + 6], [b + 3, b + 7],
    ], dtype=np.int64)

    # Terrain perimeter edges
    perim_edges = []
    # South (j=0)
    for i in range(nx - 1):
        perim_edges.append([i, i + 1])
    # North (j=ny-1)
    off_n = (ny - 1) * nx
    for i in range(nx - 1):
        perim_edges.append([off_n + i, off_n + i + 1])
    # West (i=0)
    for j in range(ny - 1):
        perim_edges.append([j * nx, (j + 1) * nx])
    # East (i=nx-1)
    for j in range(ny - 1):
        perim_edges.append([j * nx + nx - 1, (j + 1) * nx + nx - 1])

    perim_edges = np.array(perim_edges, dtype=np.int64)
    all_edges = np.vstack([box_edges, perim_edges])
    log(f"  Feature edges: {len(all_edges)}")

    # --- Write FMS ---
    fms_path = output_dir / "bbox.fms"
    log(f"  Writing bbox.fms...")
    _write_fms(fms_path, all_verts, all_tris, all_edges)
    log(f"  bbox.fms: {fms_path.stat().st_size / 1e6:.1f} MB")

    # --- Write STL ---
    stl_path = None
    if write_stl:
        stl_path = output_dir / "bbox.stl"
        log(f"  Writing bbox.stl...")
        _write_stl(stl_path, all_verts, all_tris)
        log(f"  bbox.stl: {stl_path.stat().st_size / 1e6:.1f} MB")

    return fms_path, stl_path


def _write_fms(path, vertices, triangles, feature_edges):
    """Write cfMesh .fms file."""
    patch_defs = [
        ("snowBed", "empty"), ("xMin", "empty"), ("xMax", "empty"),
        ("yMin", "empty"), ("yMax", "empty"), ("zMax", "empty"),
    ]

    CHUNK = 100_000

    with open(path, "w") as f:
        # Patch header
        f.write(f"\n{len(patch_defs)}\n(\n")
        for name, ptype in patch_defs:
            f.write(f"\n{name} {ptype}\n")
        f.write(")\n\n")

        # Vertices
        f.write(f"\n{len(vertices)}\n(\n")
        for start in range(0, len(vertices), CHUNK):
            end = min(start + CHUNK, len(vertices))
            chunk = vertices[start:end]
            buf = []
            for v in chunk:
                buf.append(f"({v[0]:.6g} {v[1]:.6g} {v[2]:.6g})\n")
            f.write("".join(buf))
        f.write(")\n\n")

        # Triangles
        f.write(f"\n{len(triangles)}\n(\n")
        for start in range(0, len(triangles), CHUNK):
            end = min(start + CHUNK, len(triangles))
            chunk = triangles[start:end]
            buf = []
            for t in chunk:
                buf.append(f"(({t[0]} {t[1]} {t[2]}) {t[3]})\n")
            f.write("".join(buf))
        f.write(")\n\n")

        # Feature edges
        f.write(f"\n{len(feature_edges)}\n(\n")
        buf = []
        for e in feature_edges:
            buf.append(f"({e[0]} {e[1]})\n")
        f.write("".join(buf))
        f.write(")\n\n")

        # Trailing
        f.write("0()\n0()\n0()\n")


def _write_stl(path, vertices, triangles):
    """Write ASCII STL containing only the snowBed (terrain) surface."""
    # Only snowBed (patch 0) — bbox walls/ceiling are only in the FMS
    mask = triangles[:, 3] == 0
    snow_tris = triangles[mask]

    v0 = vertices[snow_tris[:, 0]]
    v1 = vertices[snow_tris[:, 1]]
    v2 = vertices[snow_tris[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms

    CHUNK = 50_000

    with open(path, "w") as f:
        f.write("solid snowBed\n")
        for start in range(0, len(snow_tris), CHUNK):
            end = min(start + CHUNK, len(snow_tris))
            buf = []
            for k in range(start, end):
                n = normals[k]
                a, b_, c = v0[k], v1[k], v2[k]
                buf.append(
                    f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n"
                    f"    outer loop\n"
                    f"      vertex {a[0]:.6g} {a[1]:.6g} {a[2]:.6g}\n"
                    f"      vertex {b_[0]:.6g} {b_[1]:.6g} {b_[2]:.6g}\n"
                    f"      vertex {c[0]:.6g} {c[1]:.6g} {c[2]:.6g}\n"
                    f"    endloop\n"
                    f"  endfacet\n"
                )
            f.write("".join(buf))
        f.write("endsolid snowBed\n")


# ---------------------------------------------------------------------------
# Step 5: Template OpenFOAM dicts
# ---------------------------------------------------------------------------

def write_openfoam_dicts(output_dir, mesh_cell_size=8, log=print):
    """Write meshDict and createPatchDict templates."""
    output_dir = Path(output_dir)
    system_dir = output_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    meshdict_path = system_dir / "meshDict"
    meshdict_path.write_text(f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                |
| \\\\      /  F ield         | cfMesh: A library for mesh generation          |
|  \\\\    /   O peration     |                                                |
|   \\\\  /    A nd           | Author: Franjo Juretic                         |
|    \\\\/     M anipulation  | E-mail: franjo.juretic@c-fields.com            |
\\*---------------------------------------------------------------------------*/

FoamFile
{{
    version   2.0;
    format    ascii;
    class     dictionary;
    location  "system";
    object    meshDict;
}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

surfaceFile "bbox.fms";

maxCellSize {mesh_cell_size};

localRefinement
{{
    "snowBed"
    {{
        additionalRefinementLevels 2;
        refinementThickness 0.5;
    }}
}}

objectRefinements
{{
}}

boundaryLayers
{{
    patchBoundaryLayers
    {{
    }}
}}

renameBoundary
{{
    defaultType patch;
}}

// ************************************************************************* //
""")
    log(f"  Written: meshDict")

    cpd_path = system_dir / "createPatchDict"
    cpd_path.write_text("""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2.1.x                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      createPatchDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

pointSync false;

patches
(
    {
        name inlet;
        patchInfo
        {
            type patch;
        }
        constructFrom patches;
        patches (xMin);
    }

    {
        name outlet;
        patchInfo
        {
            type patch;
        }
        constructFrom patches;
        patches (xMax);
    }

    {
        name front;
        patchInfo
        {
            type patch;
        }
        constructFrom patches;
        patches (yMin);
    }

    {
        name back;
        patchInfo
        {
            type patch;
        }
        constructFrom patches;
        patches (yMax);
    }

    {
        name top;
        patchInfo
        {
            type patch;
        }
        constructFrom patches;
        patches (zMax);
    }
);

// ************************************************************************* //
""")
    log(f"  Written: createPatchDict")


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def lv95_to_wgs84(bbox_lv95):
    """Convert LV95 bbox to WGS84."""
    t = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon_min, lat_min = t.transform(bbox_lv95[0], bbox_lv95[1])
    lon_max, lat_max = t.transform(bbox_lv95[2], bbox_lv95[3])
    return (lon_min, lat_min, lon_max, lat_max)


def wgs84_to_lv95(bbox_wgs84):
    """Convert WGS84 bbox to LV95."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    e_min, n_min = t.transform(bbox_wgs84[0], bbox_wgs84[1])
    e_max, n_max = t.transform(bbox_wgs84[2], bbox_wgs84[3])
    return (e_min, n_min, e_max, n_max)
