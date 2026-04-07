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
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
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
# Step 2b: Rotate grid to align x-axis with wind direction
# ---------------------------------------------------------------------------

def compute_rotated_download_bbox(bbox_lv95, wind_direction):
    """Compute the axis-aligned LV95 bbox that encloses the rotated ROI.

    When the domain is rotated, the download area must cover the enclosing
    axis-aligned rectangle of the rotated ROI.

    Returns enlarged (e_min, n_min, e_max, n_max) in LV95.
    """
    e_min, n_min, e_max, n_max = bbox_lv95
    cx, cy = (e_min + e_max) / 2, (n_min + n_max) / 2
    hw, hh = (e_max - e_min) / 2, (n_max - n_min) / 2

    bearing = np.radians((wind_direction + 180) % 360)
    theta = np.pi / 2 - bearing  # CCW from East

    # rotate the 4 corners around center
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotated_e = [cx + dx * cos_t - dy * sin_t for dx, dy in corners_local]
    rotated_n = [cy + dx * sin_t + dy * cos_t for dx, dy in corners_local]

    return (min(rotated_e), min(rotated_n), max(rotated_e), max(rotated_n))


def resample_rotated_grid(x, y, Z, origin_lv95, bbox_lv95, wind_direction,
                          log=print):
    """Resample DEM onto a grid rotated to align x-axis with wind direction.

    The wind enters from xMin in direction (1,0,0). The domain x-axis bearing
    is (wind_direction + 180) % 360 from North.

    Args:
        x, y, Z: processed DEM in local coords (origin at bbox SW corner)
        origin_lv95: (e_min, n_min) of the DEM download bbox
        bbox_lv95: (e_min, n_min, e_max, n_max) of the original ROI
        wind_direction: degrees from North clockwise (meteo "wind from")

    Returns: (x_rot, y_rot, Z_rot, rot_origin_lv95, theta)
        x_rot, y_rot: 1D coordinate arrays for the rotated regular grid
        Z_rot: 2D elevation array on the rotated grid
        rot_origin_lv95: (E, N) of the rotated grid's (0,0) corner in LV95
        theta: rotation angle in radians (CCW from East)
    """
    cell_size = float(x[1] - x[0])
    e_min_roi, n_min_roi, e_max_roi, n_max_roi = bbox_lv95
    cx = (e_min_roi + e_max_roi) / 2
    cy = (n_min_roi + n_max_roi) / 2
    hw = (e_max_roi - e_min_roi) / 2
    hh = (n_max_roi - n_min_roi) / 2

    bearing = np.radians((wind_direction + 180) % 360)
    theta = np.pi / 2 - bearing  # CCW from East
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    log(f"  Wind from {wind_direction}°, x-axis bearing {(wind_direction+180)%360}°, "
        f"rotation θ={np.degrees(theta):.1f}° from East")

    # Rotate the 4 ROI corners into the rotated frame to find grid extent
    corners_lv95 = [
        (cx - hw, cy - hh), (cx + hw, cy - hh),
        (cx + hw, cy + hh), (cx - hw, cy + hh),
    ]
    # Transform to rotated coords (centered on cx, cy)
    rot_x = [ (e - cx) * cos_t + (n - cy) * sin_t for e, n in corners_lv95]
    rot_y = [-(e - cx) * sin_t + (n - cy) * cos_t for e, n in corners_lv95]

    # Grid extent in rotated coords
    rx_min, rx_max = min(rot_x), max(rot_x)
    ry_min, ry_max = min(rot_y), max(rot_y)

    # Build regular grid in rotated coordinates
    nx_rot = int(round((rx_max - rx_min) / cell_size))
    ny_rot = int(round((ry_max - ry_min) / cell_size))
    x_rot = np.arange(nx_rot) * cell_size
    y_rot = np.arange(ny_rot) * cell_size

    # The (0,0) corner of the rotated grid in LV95
    rot_origin_e = cx + rx_min * cos_t - ry_min * sin_t
    rot_origin_n = cy + rx_min * sin_t + ry_min * cos_t
    rot_origin_lv95 = (rot_origin_e, rot_origin_n)

    # Transform all rotated grid points to LV95 for DEM sampling
    xx, yy = np.meshgrid(x_rot, y_rot)  # (ny_rot, nx_rot)
    # absolute rotated coords (before centering)
    rx_abs = rx_min + xx
    ry_abs = ry_min + yy
    # back to LV95
    E_grid = cx + rx_abs * cos_t - ry_abs * sin_t
    N_grid = cy + rx_abs * sin_t + ry_abs * cos_t

    # Convert LV95 grid points to DEM local coords for interpolation
    e_local = E_grid - origin_lv95[0]
    n_local = N_grid - origin_lv95[1]

    # Interpolate DEM at rotated grid points
    interp = RegularGridInterpolator((y, x), Z, method='linear',
                                     bounds_error=False, fill_value=None)
    pts = np.column_stack([n_local.ravel(), e_local.ravel()])
    Z_rot = interp(pts).reshape(ny_rot, nx_rot)

    log(f"  Rotated grid: {nx_rot}x{ny_rot}, res={cell_size}m, "
        f"Z=[{Z_rot[np.isfinite(Z_rot)].min():.1f}, {Z_rot[np.isfinite(Z_rot)].max():.1f}]m")

    return x_rot, y_rot, Z_rot, rot_origin_lv95, theta


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
# Step 3a: Write padded ROI shapefile
# ---------------------------------------------------------------------------

def write_padded_roi_shp(origin_lv95, x_ext, y_ext, output_dir,
                         theta=0.0, log=print):
    """Write a shapefile of the padded domain extent in EPSG:2056 (LV95).

    If theta != 0, the domain is a rotated rectangle. The 4 corners in local
    rotated coords are transformed back to LV95.

    Args:
        origin_lv95: (E, N) of the grid's (0,0) corner in LV95
        x_ext, y_ext: extended local coordinate arrays
        output_dir: output directory
        theta: rotation angle in radians (CCW from East), 0 = axis-aligned
    """
    import shapefile

    x0, x1 = float(x_ext[0]), float(x_ext[-1])
    y0, y1 = float(y_ext[0]), float(y_ext[-1])
    oe, on = origin_lv95

    # 4 corners in local coords → LV95
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    corners_local = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    corners_lv95 = [
        (oe + lx * cos_t - ly * sin_t, on + lx * sin_t + ly * cos_t)
        for lx, ly in corners_local
    ]
    corners_lv95.append(corners_lv95[0])  # close the ring

    out_path = Path(output_dir) / "ROI_padded"
    w = shapefile.Writer(str(out_path))
    w.shapeType = shapefile.POLYGON
    w.field("name", "C", size=40)
    w.poly([[[e, n] for e, n in corners_lv95]])
    w.record("padded_domain")
    w.close()

    # Write .prj for EPSG:2056
    prj_wkt = (
        'PROJCS["CH1903+_LV95",'
        'GEOGCS["GCS_CH1903+",'
        'DATUM["D_CH1903+",'
        'SPHEROID["Bessel_1841",6377397.155,299.1528128]],'
        'PRIMEM["Greenwich",0.0],'
        'UNIT["Degree",0.0174532925199433]],'
        'PROJECTION["Hotine_Oblique_Mercator_Azimuth_Center"],'
        'PARAMETER["False_Easting",2600000.0],'
        'PARAMETER["False_Northing",1200000.0],'
        'PARAMETER["Scale_Factor",1.0],'
        'PARAMETER["Azimuth",90.0],'
        'PARAMETER["Longitude_Of_Center",7.439583333333333],'
        'PARAMETER["Latitude_Of_Center",46.95240555555556],'
        'UNIT["Meter",1.0]]'
    )
    (Path(output_dir) / "ROI_padded.prj").write_text(prj_wkt)

    log(f"  Padded ROI: E[{e_min:.1f}, {e_max:.1f}] N[{n_min:.1f}, {n_max:.1f}]")
    return (e_min, n_min, e_max, n_max)


# ---------------------------------------------------------------------------
# Step 3b: Optional coordinate transforms
# ---------------------------------------------------------------------------

def shift_z_origin(Z_ext, z_top, z_ref, log=print):
    """Shift all elevations so Z_ext.min() == 0.

    Returns (Z_ext, z_top, z_ref, z_shift) where z_shift is the value subtracted.
    """
    z_min = float(Z_ext.min())
    Z_ext = Z_ext - z_min
    z_top = z_top - z_min
    z_ref = z_ref - z_min
    log(f"  Z-shift: subtracted {z_min:.1f}m (new range [{Z_ext.min():.1f}, {Z_ext.max():.1f}])")
    return Z_ext, z_top, z_ref, z_min


def rotate_axes_north(x_ext, y_ext, Z_ext, log=print):
    """Rotate coordinate system 90 deg CW so x-axis points North.

    Mapping: new_x = old_y, new_y = -old_x (right-handed: x=North, y=West, z=Up).
    Z array is rotated with np.rot90(Z, k=1).

    Returns (x_new, y_new, Z_new).
    """
    x_new = y_ext.copy()
    y_new = -x_ext[::-1].copy()
    Z_new = np.rot90(Z_ext, k=1)
    log(f"  Rotated: x-axis now points North (x=[{x_new[0]:.1f}, {x_new[-1]:.1f}], "
        f"y=[{y_new[0]:.1f}, {y_new[-1]:.1f}])")
    return x_new, y_new, Z_new


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

def write_openfoam_dicts(output_dir, mesh_cell_size=8, wind_direction=None, log=print):
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
    if wind_direction is not None and wind_direction > 0:
        bearing = (wind_direction + 180) % 360
        coord_comment = (
            f"// Wind from {wind_direction}° (meteo), x-axis bearing {bearing}°\n"
            f"// inlet (xMin) = upwind, outlet (xMax) = downwind\n"
            f"// Wind velocity: U = (Umag, 0, 0)\n\n"
        )
    else:
        coord_comment = (
            "// Coordinate system: X = East, Y = North, Z = Up\n\n"
        )
    cpd_path.write_text(coord_comment + """\
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


def read_shp_bbox(shp_file, shx_file=None, dbf_file=None, prj_file=None):
    """Read bounding box from a .shp file, return as LV95 tuple.

    Parameters
    ----------
    shp_file : file-like
        The .shp file.
    shx_file, dbf_file : file-like, optional
        Companion files (pyshp may need them).
    prj_file : file-like or str, optional
        The .prj file for CRS detection. If absent, LV95 (EPSG:2056) is assumed.

    Returns
    -------
    bbox_lv95 : tuple
        (e_min, n_min, e_max, n_max) in EPSG:2056.
    source_crs : str
        Detected or assumed CRS identifier.
    """
    import shapefile

    kwargs = {"shp": shp_file}
    if shx_file is not None:
        kwargs["shx"] = shx_file
    if dbf_file is not None:
        kwargs["dbf"] = dbf_file

    reader = shapefile.Reader(**kwargs)
    x_min, y_min, x_max, y_max = reader.bbox

    # Detect CRS from .prj
    source_crs = "EPSG:2056"
    if prj_file is not None:
        prj_text = prj_file if isinstance(prj_file, str) else prj_file.read()
        if isinstance(prj_text, bytes):
            prj_text = prj_text.decode("utf-8")
        from pyproj import CRS
        try:
            detected = CRS.from_wkt(prj_text)
            epsg = detected.to_epsg()
            source_crs = f"EPSG:{epsg}" if epsg else detected.to_string()
        except Exception:
            pass

    # Transform to LV95 if needed
    if source_crs != "EPSG:2056":
        t = Transformer.from_crs(source_crs, "EPSG:2056", always_xy=True)
        e_min, n_min = t.transform(x_min, y_min)
        e_max, n_max = t.transform(x_max, y_max)
        if e_min > e_max:
            e_min, e_max = e_max, e_min
        if n_min > n_max:
            n_min, n_max = n_max, n_min
        return (e_min, n_min, e_max, n_max), source_crs

    return (x_min, y_min, x_max, y_max), source_crs
