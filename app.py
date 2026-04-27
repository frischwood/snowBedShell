"""
DEM-to-cfMesh Streamlit web app.

Interactive map selection, DEM processing, and cfMesh file generation.
"""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from pyproj import Transformer
from shapely.geometry import Polygon, box as shapely_box

from pipeline import (
    download_dem_tiles,
    process_dem,
    compute_rotated_download_bbox,
    resample_rotated_grid,
    extend_domain,
    write_padded_roi_shp,
    shift_z_origin,
    triangulate_and_write,
    write_openfoam_dicts,
    wgs84_to_lv95,
    lv95_to_wgs84,
    read_shp_bbox,
)

# ---------------------------------------------------------------------------
# ROI validation
# ---------------------------------------------------------------------------

MAX_SIDE_KM = 10.0

# Simplified Swiss boundary in EPSG:2056 (LV95) — from swisstopo
_SWISS_BOUNDARY_LV95 = Polygon([
    (2485000, 1075000),
    (2485000, 1110000),
    (2490000, 1145000),
    (2495000, 1185000),
    (2510000, 1230000),
    (2525000, 1265000),
    (2570000, 1295000),
    (2630000, 1296000),
    (2720000, 1295000),
    (2795000, 1280000),
    (2834000, 1255000),
    (2830000, 1220000),
    (2815000, 1185000),
    (2785000, 1150000),
    (2750000, 1110000),
    (2715000, 1085000),
    (2680000, 1080000),
    (2630000, 1085000),
    (2580000, 1095000),
    (2530000, 1085000),
    (2490000, 1078000),
    (2485000, 1075000),
])


def validate_roi(bbox_lv95):
    """Validate ROI: max 10 km sides, fully inside Switzerland.

    Returns (ok: bool, message: str).
    """
    e_min, n_min, e_max, n_max = bbox_lv95
    dx_m = e_max - e_min
    dy_m = n_max - n_min

    if dx_m / 1000 > MAX_SIDE_KM:
        return False, f"Width {dx_m/1000:.1f} km exceeds {MAX_SIDE_KM} km limit."
    if dy_m / 1000 > MAX_SIDE_KM:
        return False, f"Height {dy_m/1000:.1f} km exceeds {MAX_SIDE_KM} km limit."

    roi_box = shapely_box(e_min, n_min, e_max, n_max)
    if not _SWISS_BOUNDARY_LV95.contains(roi_box):
        if _SWISS_BOUNDARY_LV95.intersects(roi_box):
            return False, "ROI crosses the Swiss border. Please redraw fully inside Switzerland."
        return False, "ROI is outside Switzerland."

    return True, ""


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DEM to cfMesh",
    page_icon="mountain",
    layout="wide",
)

st.title("DEM to cfMesh Preprocessor")
st.caption(
    "Download Swiss DEM, extend with smooth buffer zones, "
    "generate cfMesh-ready surface files."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "bbox_wgs84" not in st.session_state:
    st.session_state.bbox_wgs84 = None
    st.session_state.bbox_lv95 = None
    st.session_state.roi_error = None
    # Bumped whenever the bbox is set externally (draw, manual entry, shapefile,
    # click-to-move, reset). Used as a key suffix on the dimension sliders so they
    # re-initialize to the new bbox; left untouched when the sliders themselves
    # change the bbox so the in-flight drag isn't clobbered.
    st.session_state.bbox_version = 0
    st.session_state.last_processed_click = None
if "result_zip" not in st.session_state:
    st.session_state.result_zip = None
    st.session_state.result_plot = None
    st.session_state.result_meta = None

# ---------------------------------------------------------------------------
# Sidebar: parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Parameters")

    resolution = st.selectbox(
        "DEM resolution",
        options=[2.0, 0.5],
        index=0,
        format_func=lambda x: f"{x}m",
    )
    gaussian_sigma = st.slider(
        "Gaussian smoothing (m)", 0.0, 10.0, 2.0, 0.5,
        help="Sigma of Gaussian filter applied to DEM",
    )
    target_res = st.number_input(
        "Target resolution (m)", min_value=0.0, value=0.0, step=0.5,
        help="Subsample DEM to this resolution. 0 = keep native.",
    )
    buffer_width = st.slider(
        "Buffer zone width (m)", 0, 500, 200, 10,
        help="Width of smooth transition zone around terrain edges",
    )
    ref_elevation = st.selectbox(
        "Reference elevation",
        options=["mean_edge", "median_edge", "min_edge"],
        help="Elevation that buffer zones blend toward",
    )
    domain_height = st.slider(
        "Domain height above terrain (m)", 50, 1000, 200, 10,
        help="Height of bounding box ceiling above max terrain",
    )
    mesh_cell_size = st.slider(
        "cfMesh max cell size (m)", 1, 40, 8, 1,
        help="maxCellSize parameter in meshDict",
    )
    include_stl = st.checkbox("Include STL in output", value=True,
                              help="ASCII STL for ParaView preview (can be large)")

    st.divider()
    st.header("Coordinate transforms")
    shift_z_to_zero = st.checkbox(
        "Shift elevation to zero", value=True,
        help="Subtract minimum elevation so terrain base starts near z=0",
    )
    wind_direction = st.slider(
        "Wind direction (° from N)", min_value=0, max_value=360,
        value=0, step=1,
        help="Meteorological wind-from direction. North edge normal points in this direction. "
             "Set 0 for no rotation (X=East). The ROI rectangle on the map rotates as you drag.",
    )

    st.divider()
    st.header("Region of Interest")
    _has_sel = st.session_state.bbox_lv95 is not None
    if _has_sel:
        _e_min, _n_min, _e_max, _n_max = st.session_state.bbox_lv95
        _init_len = max(100, min(10000, int(round(_e_max - _e_min))))
        _init_wid = max(100, min(10000, int(round(_n_max - _n_min))))
    else:
        _init_len, _init_wid = 2000, 2000
    _v = st.session_state.get("bbox_version", 0)
    length_m = st.slider(
        "E–W extent [m]", 100, 10000, _init_len, step=10,
        disabled=not _has_sel,
        key=f"roi_length_{_v}",
        help="Rectangle size along the East–West axis (before rotation).",
    )
    width_m = st.slider(
        "N–S extent [m]", 100, 10000, _init_wid, step=10,
        disabled=not _has_sel,
        key=f"roi_width_{_v}",
        help="Rectangle size along the North–South axis (before rotation).",
    )
    shift_mode = st.checkbox(
        "Move ROI by clicking on map",
        value=False,
        disabled=not _has_sel,
        help="When ON, clicking on the map shifts the rectangle's center to that point. "
             "Dimensions are preserved.",
    )
    if _has_sel and (length_m != _init_len or width_m != _init_wid):
        _cx = (_e_min + _e_max) / 2
        _cy = (_n_min + _n_max) / 2
        _new_bbox = (
            _cx - length_m / 2, _cy - width_m / 2,
            _cx + length_m / 2, _cy + width_m / 2,
        )
        _ok, _msg = validate_roi(_new_bbox)
        if _ok:
            st.session_state.bbox_lv95 = _new_bbox
            st.session_state.bbox_wgs84 = lv95_to_wgs84(_new_bbox)
            st.session_state.roi_error = None
            st.session_state.result_zip = None
            st.session_state.result_plot = None
            st.session_state.result_meta = None
        else:
            st.session_state.roi_error = _msg

# ---------------------------------------------------------------------------
# Map selection
# ---------------------------------------------------------------------------

st.subheader("1. Select Region of Interest")

col_map, col_info = st.columns([3, 1])

with col_map:
    m = folium.Map(
        location=[46.8, 8.2],
        zoom_start=8,
        tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg",
        attr="swisstopo",
    )

    _t_to_wgs = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    _t_to_lv = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

    has_selection = st.session_state.bbox_lv95 is not None

    if has_selection:
        e_min_r, n_min_r, e_max_r, n_max_r = st.session_state.bbox_lv95
        cx = (e_min_r + e_max_r) / 2
        cy = (n_min_r + n_max_r) / 2
        hw = (e_max_r - e_min_r) / 2
        hh = (n_max_r - n_min_r) / 2
        # CW rotation by wind_direction so the original north edge of the ROI
        # ends up facing the wind (meteorological "wind from" convention).
        theta = -np.radians(wind_direction % 360)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        corners_lv95 = [
            (cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t)
            for dx, dy in corners_local
        ]
        corners_wgs = [_t_to_wgs.transform(e, n) for e, n in corners_lv95]
        corners_latlon = [[lat, lon] for lon, lat in corners_wgs]

        # Render the rotated rectangle as a server-side folium element. The map
        # HTML is rebuilt every Streamlit rerun, so dragging the wind-direction
        # slider produces an instant visual update with no JS race conditions.
        folium.Polygon(
            locations=corners_latlon,
            color="#e74c3c", weight=2, fill=True, fill_opacity=0.15,
            tooltip=(f"ROI ({wind_direction}° rotation)"
                     if wind_direction % 360 > 0 else "ROI"),
        ).add_to(m)

        lats_fit = [c[0] for c in corners_latlon]
        lons_fit = [c[1] for c in corners_latlon]
        m.fit_bounds(
            [[min(lats_fit), min(lons_fit)], [max(lats_fit), max(lons_fit)]],
            padding=[50, 50],
        )

    # Draw plugin: allow rectangle drawing only when no selection exists.
    # Edit/remove is disabled — clearing happens via the "Reset selection" button.
    Draw(
        draw_options={
            "polyline": False, "polygon": False, "circle": False,
            "circlemarker": False, "marker": False,
            "rectangle": False if has_selection else {
                "shapeOptions": {"color": "#e74c3c", "weight": 2, "fillOpacity": 0.15}
            },
        },
        edit_options={"edit": False, "remove": False},
    ).add_to(m)

    map_output = st_folium(
        m, width=None, height=500,
        returned_objects=["all_drawings", "last_clicked"],
    )

# Parse new drawing from map. Only when no selection exists — once a rectangle
# is saved, the Draw plugin is disabled and the displayed polygon is rendered
# server-side (not in drawnItems), so all_drawings is empty for re-renders.
if not has_selection and map_output and map_output.get("all_drawings"):
    drawings = map_output["all_drawings"]
    if drawings:
        last = drawings[-1]
        coords = last["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        verts_lv = [_t_to_lv.transform(lon, lat) for lon, lat in zip(lons, lats)]
        new_lv95 = (
            min(e for e, _ in verts_lv), min(n for _, n in verts_lv),
            max(e for e, _ in verts_lv), max(n for _, n in verts_lv),
        )
        ok, msg = validate_roi(new_lv95)
        if ok:
            st.session_state.bbox_wgs84 = lv95_to_wgs84(new_lv95)
            st.session_state.bbox_lv95 = new_lv95
            st.session_state.bbox_version += 1
            st.session_state.roi_error = None
            st.session_state.result_zip = None
            st.session_state.result_plot = None
            st.session_state.result_meta = None
            st.rerun()
        else:
            st.session_state.roi_error = msg

# Click-to-shift: when shift_mode is ON, treat a fresh map click as the new
# rectangle center. Dimensions are preserved.
if (shift_mode and has_selection and map_output
        and map_output.get("last_clicked")):
    _click = map_output["last_clicked"]
    _click_id = (round(_click["lat"], 6), round(_click["lng"], 6))
    if st.session_state.last_processed_click != _click_id:
        st.session_state.last_processed_click = _click_id
        _new_e, _new_n = _t_to_lv.transform(_click["lng"], _click["lat"])
        _e_min, _n_min, _e_max, _n_max = st.session_state.bbox_lv95
        _length = _e_max - _e_min
        _width = _n_max - _n_min
        _shifted = (
            _new_e - _length / 2, _new_n - _width / 2,
            _new_e + _length / 2, _new_n + _width / 2,
        )
        _ok, _msg = validate_roi(_shifted)
        if _ok:
            st.session_state.bbox_lv95 = _shifted
            st.session_state.bbox_wgs84 = lv95_to_wgs84(_shifted)
            st.session_state.bbox_version += 1
            st.session_state.roi_error = None
            st.session_state.result_zip = None
            st.session_state.result_plot = None
            st.session_state.result_meta = None
            st.rerun()
        else:
            st.session_state.roi_error = _msg

with col_info:
    if st.session_state.bbox_lv95:
        e_min, n_min, e_max, n_max = st.session_state.bbox_lv95
        dx_km = (e_max - e_min) / 1000
        dy_km = (n_max - n_min) / 1000

        st.metric("Width", f"{dx_km:.2f} km")
        st.metric("Height", f"{dy_km:.2f} km")

        # Estimate grid size
        n_cells_x = int((e_max - e_min) / resolution)
        n_cells_y = int((n_max - n_min) / resolution)
        buf_cells = int(buffer_width / resolution) * 2
        total_cells = (n_cells_x + buf_cells) * (n_cells_y + buf_cells)
        st.metric("Grid (est.)", f"{n_cells_x + buf_cells} x {n_cells_y + buf_cells}")
        st.metric("Triangles (est.)", f"{2 * total_cells:,.0f}")

        if total_cells > 5_000_000:
            st.warning("Large grid! Consider increasing resolution or reducing area.")

        st.caption(f"LV95: E [{e_min:.0f}, {e_max:.0f}]")
        st.caption(f"LV95: N [{n_min:.0f}, {n_max:.0f}]")

        if st.button("Reset selection"):
            st.session_state.bbox_wgs84 = None
            st.session_state.bbox_lv95 = None
            st.session_state.bbox_version += 1
            st.session_state.roi_error = None
            st.session_state.result_zip = None
            st.session_state.result_plot = None
            st.session_state.result_meta = None
            st.rerun()
    elif st.session_state.roi_error:
        st.error(st.session_state.roi_error)
    else:
        st.info("Draw a rectangle on the map using the square tool (top-left toolbar).")

# Manual bbox fallback
with st.expander("Or enter LV95 coordinates manually"):
    bbox_str = st.text_input(
        "E_MIN, N_MIN, E_MAX, N_MAX",
        placeholder="2717000, 1205000, 2719000, 1207000",
    )
    if bbox_str:
        try:
            parts = [float(x.strip()) for x in bbox_str.split(",")]
            if len(parts) == 4:
                manual_lv95 = tuple(parts)
                ok, msg = validate_roi(manual_lv95)
                if ok:
                    st.session_state.bbox_lv95 = manual_lv95
                    st.session_state.bbox_wgs84 = lv95_to_wgs84(manual_lv95)
                    st.session_state.bbox_version += 1
                    st.session_state.roi_error = None
                    st.session_state.result_zip = None
                    st.session_state.result_plot = None
                    st.session_state.result_meta = None
                    st.success(f"Set: {manual_lv95}")
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Need exactly 4 values")
        except ValueError:
            st.error("Invalid number format")

# SHP upload
with st.expander("Or upload a shapefile (.shp)"):
    uploaded_files = st.file_uploader(
        "Upload .shp (required) and companion files (.shx, .dbf, .prj), or a .zip",
        type=["shp", "shx", "dbf", "prj", "zip"],
        accept_multiple_files=True,
        key="shp_upload",
    )
    if uploaded_files:
        shp_file = shx_file = dbf_file = prj_file = None

        for uf in uploaded_files:
            name_lower = uf.name.lower()
            if name_lower.endswith(".zip"):
                with zipfile.ZipFile(uf) as zf:
                    for zname in zf.namelist():
                        zname_lower = zname.lower()
                        if zname_lower.endswith(".shp"):
                            shp_file = io.BytesIO(zf.read(zname))
                        elif zname_lower.endswith(".shx"):
                            shx_file = io.BytesIO(zf.read(zname))
                        elif zname_lower.endswith(".dbf"):
                            dbf_file = io.BytesIO(zf.read(zname))
                        elif zname_lower.endswith(".prj"):
                            prj_file = io.BytesIO(zf.read(zname))
            elif name_lower.endswith(".shp"):
                shp_file = uf
            elif name_lower.endswith(".shx"):
                shx_file = uf
            elif name_lower.endswith(".dbf"):
                dbf_file = uf
            elif name_lower.endswith(".prj"):
                prj_file = uf

        if shp_file is None:
            st.error("No .shp file found.")
        else:
            try:
                bbox_from_shp, detected_crs, prj_used = read_shp_bbox(
                    shp_file, shx_file=shx_file, dbf_file=dbf_file, prj_file=prj_file,
                )
                if prj_used:
                    st.info(f"Detected CRS: {detected_crs}")
                else:
                    st.warning(
                        f"No .prj sidecar found (or it was unparseable); "
                        f"assuming {detected_crs}. If your shapefile uses a "
                        f"different CRS, the bounding box will be wrong."
                    )
                st.caption(
                    f"Bounding box (LV95): E [{bbox_from_shp[0]:.0f}, {bbox_from_shp[2]:.0f}], "
                    f"N [{bbox_from_shp[1]:.0f}, {bbox_from_shp[3]:.0f}]"
                )

                ok, msg = validate_roi(bbox_from_shp)
                if ok:
                    if st.button("Use this bounding box", key="use_shp_bbox"):
                        st.session_state.bbox_lv95 = bbox_from_shp
                        st.session_state.bbox_wgs84 = lv95_to_wgs84(bbox_from_shp)
                        st.session_state.bbox_version += 1
                        st.session_state.roi_error = None
                        st.session_state.result_zip = None
                        st.session_state.result_plot = None
                        st.session_state.result_meta = None
                        st.rerun()
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Failed to read shapefile: {e}")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

st.divider()
st.subheader("2. Generate cfMesh Files")

run_disabled = st.session_state.bbox_wgs84 is None
if run_disabled:
    st.info("Select a region above first.")

if st.button("Run Pipeline", type="primary", disabled=run_disabled, use_container_width=True):
    bbox_wgs84 = st.session_state.bbox_wgs84
    bbox_lv95 = st.session_state.bbox_lv95

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "cfmesh_output"
        trisurface_dir = output_dir / "constant" / "triSurface"
        trisurface_dir.mkdir(parents=True, exist_ok=True)

        status = st.status("Processing...", expanded=True)

        try:
            # Step 1: Download DEM (enlarge bbox if rotation needed)
            status.write("Downloading DEM tiles from swisstopo...")
            download_wgs84 = bbox_wgs84
            download_lv95 = bbox_lv95
            if wind_direction % 360 > 0:
                download_lv95 = compute_rotated_download_bbox(bbox_lv95, wind_direction)
                # The rotated AABB can be up to sqrt(2)x larger than the user's ROI.
                # If it spills outside Switzerland, swisstopo has no tiles there —
                # raise a clear error before the long DEM download.
                if not _SWISS_BOUNDARY_LV95.contains(shapely_box(*download_lv95)):
                    raise ValueError(
                        f"The enlarged download bbox for a {wind_direction}° rotation "
                        f"extends outside Switzerland. Move your ROI further from the "
                        f"border or reduce the rotation angle."
                    )
                download_wgs84 = lv95_to_wgs84(download_lv95)
                status.write(f"  Enlarged download bbox for {wind_direction}° rotation")
            dem_path = download_dem_tiles(
                download_wgs84, resolution, output_dir,
                log=lambda msg: status.write(msg),
            )

            # Step 2: Process DEM
            status.write("Processing DEM...")
            t_res = target_res if target_res > 0 else None
            x, y, Z, origin_lv95 = process_dem(
                dem_path, download_lv95, gaussian_sigma, t_res,
                log=lambda msg: status.write(msg),
            )

            # Step 2b: Resample onto rotated grid
            rotation_theta = 0.0
            if wind_direction % 360 > 0:
                status.write(f"Rotating grid to align with {wind_direction}° wind...")
                x, y, Z, origin_lv95, rotation_theta = resample_rotated_grid(
                    x, y, Z, origin_lv95, bbox_lv95, wind_direction,
                    log=lambda msg: status.write(msg),
                )

            # Step 3: Extend domain
            status.write("Extending domain with buffer zones...")
            x_ext, y_ext, Z_ext, z_top, z_ref = extend_domain(
                x, y, Z, buffer_width, ref_elevation, domain_height,
                log=lambda msg: status.write(msg),
            )

            # Step 3a: Write padded ROI shapefile
            status.write("Writing padded ROI shapefile...")
            write_padded_roi_shp(
                origin_lv95, x_ext, y_ext, output_dir,
                theta=rotation_theta,
                log=lambda msg: status.write(msg),
            )

            # Step 3b: Coordinate transforms
            z_shift = 0.0
            if shift_z_to_zero:
                status.write("Shifting elevation origin to zero...")
                Z_ext, z_top, z_ref, z_shift = shift_z_origin(
                    Z_ext, z_top, z_ref,
                    log=lambda msg: status.write(msg),
                )

            # Step 4: Generate surfaces
            status.write("Generating cfMesh surface files (this may take a moment)...")
            fms_path, stl_path = triangulate_and_write(
                x_ext, y_ext, Z_ext, z_top, z_ref, trisurface_dir,
                write_stl=include_stl,
                log=lambda msg: status.write(msg),
            )

            # Step 5: OpenFOAM dicts
            status.write("Writing OpenFOAM dictionaries...")
            write_openfoam_dicts(
                output_dir, mesh_cell_size,
                wind_direction=wind_direction if wind_direction % 360 > 0 else None,
                log=lambda msg: status.write(msg),
            )

            # Save metadata
            # Bearing of the new x-axis from North.
            # wind_direction = 0 → no rotation, x-axis points East (bearing 90°).
            # wind_direction > 0 → rectangle is rotated CW so its original north edge
            # faces the wind; the rotated x-axis bearing is (wind_direction + 180) % 360.
            bearing = (wind_direction + 180) % 360 if wind_direction % 360 > 0 else 90
            metadata = {
                "bbox_lv95": list(bbox_lv95),
                "bbox_wgs84": list(bbox_wgs84),
                "origin_lv95": list(origin_lv95),
                "gaussian_sigma": gaussian_sigma,
                "target_resolution": t_res,
                "buffer_width": buffer_width,
                "ref_elevation": ref_elevation,
                "domain_height": domain_height,
                "mesh_cell_size": mesh_cell_size,
                "terrain_shape": list(Z_ext.shape),
                "x_range": [float(x_ext[0]), float(x_ext[-1])],
                "y_range": [float(y_ext[0]), float(y_ext[-1])],
                "z_range": [float(Z_ext.min()), float(Z_ext.max())],
                "z_top": z_top,
                "z_shift_applied": z_shift,
                "wind_direction": wind_direction,
                "x_axis_bearing": bearing,
                "rotation_angle_rad": rotation_theta,
                "coordinate_system": f"X bearing {bearing}° from N, Y perpendicular, Z=Up",
            }
            meta_path = output_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2))

            # Generate preview plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            im = axes[0].pcolormesh(x_ext, y_ext, Z_ext, cmap="terrain", shading="auto")
            axes[0].set_aspect("equal")
            axes[0].set_xlabel("x [m]")
            axes[0].set_ylabel("y [m]")
            axes[0].set_title("Extended terrain elevation")
            plt.colorbar(im, ax=axes[0], label="z [m]")

            mid_j = len(y_ext) // 2
            mid_i = len(x_ext) // 2
            axes[1].plot(x_ext, Z_ext[mid_j, :], "b-", label=f"y = {y_ext[mid_j]:.0f}m")
            axes[1].plot(y_ext, Z_ext[:, mid_i], "r-", label=f"x = {x_ext[mid_i]:.0f}m")
            axes[1].axhline(z_top, color="gray", ls="--", alpha=0.5, label="ceiling")
            axes[1].set_xlabel("distance [m]")
            axes[1].set_ylabel("z [m]")
            axes[1].set_title("Cross-sections (buffer blending)")
            axes[1].legend()
            plt.tight_layout()

            # Save plot to bytes
            plot_buf = io.BytesIO()
            fig.savefig(plot_buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            plot_buf.seek(0)

            # Create ZIP (exclude raw DEM tiles)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fpath in output_dir.rglob("*"):
                    if fpath.is_file() and "dem_tiles" not in str(fpath) and fpath.name != "dem_merged.tif":
                        arcname = str(fpath.relative_to(output_dir))
                        zf.write(fpath, arcname)
                # Also add the preview plot
                zf.writestr("terrain_preview.png", plot_buf.getvalue())
            zip_buf.seek(0)

            # Store in session state
            st.session_state.result_zip = zip_buf.getvalue()
            st.session_state.result_plot = plot_buf.getvalue()
            st.session_state.result_meta = metadata

            status.update(label="Done!", state="complete")

        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Pipeline failed: {e}")
            raise

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.result_plot:
    st.divider()
    st.subheader("3. Results")
    st.image(st.session_state.result_plot, use_container_width=True)

if st.session_state.result_meta:
    meta = st.session_state.result_meta
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Grid", f"{meta['terrain_shape'][1]} x {meta['terrain_shape'][0]}")
    col2.metric("Z range", f"{meta['z_range'][0]:.0f} - {meta['z_range'][1]:.0f} m")
    col3.metric("Ceiling", f"{meta['z_top']:.0f} m")
    col4.metric("Triangles (est.)", f"{2 * (meta['terrain_shape'][0]-1) * (meta['terrain_shape'][1]-1):,.0f}")

if st.session_state.result_zip:
    zip_size_mb = len(st.session_state.result_zip) / 1e6

    st.download_button(
        f"Download cfMesh output ({zip_size_mb:.1f} MB)",
        data=st.session_state.result_zip,
        file_name="cfmesh_output.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    with zipfile.ZipFile(io.BytesIO(st.session_state.result_zip)) as _zf:
        _entries = sorted(
            (info.filename, info.file_size) for info in _zf.infolist()
        )
    with st.expander(f"ZIP contents ({len(_entries)} files)"):
        _name_w = max((len(n) for n, _ in _entries), default=0)
        _size_w = max((len(f"{s:,}") for _, s in _entries), default=1)
        _lines = ["cfmesh_output/"]
        for _name, _size in _entries:
            _lines.append(f"  {_name:<{_name_w}}  {_size:>{_size_w},} B")
        st.code("\n".join(_lines), language=None)
        st.caption(
            "Extract into your OpenFOAM case directory, then run:\n"
            "`cartesianMesh && createPatch -overwrite && checkMesh`"
        )
