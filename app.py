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

from shapely.geometry import Polygon, box as shapely_box

from pipeline import (
    download_dem_tiles,
    process_dem,
    extend_domain,
    shift_z_origin,
    rotate_axes_north,
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
    x_axis_north = st.checkbox(
        "X-axis positive towards North", value=True,
        help="Rotate 90° CW so X points North (right-handed: Y=West, Z=Up)",
    )

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

    # Show previously selected region
    if st.session_state.bbox_wgs84:
        lon_min, lat_min, lon_max, lat_max = st.session_state.bbox_wgs84
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color="#2196F3", weight=2, fill=True, fill_opacity=0.1,
            tooltip="Current selection",
        ).add_to(m)
        # Center map on selection
        m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]], padding=[50, 50])

    Draw(
        draw_options={
            "polyline": False, "polygon": False, "circle": False,
            "circlemarker": False, "marker": False,
            "rectangle": {
                "shapeOptions": {"color": "#e74c3c", "weight": 2, "fillOpacity": 0.15}
            },
        },
        edit_options={"remove": True},
    ).add_to(m)

    map_output = st_folium(m, width=None, height=500, returned_objects=["all_drawings"])

# Parse new drawings from map
if map_output and map_output.get("all_drawings"):
    drawings = map_output["all_drawings"]
    if drawings:
        last = drawings[-1]
        coords = last["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        new_wgs84 = (min(lons), min(lats), max(lons), max(lats))
        # Only update if actually changed
        if new_wgs84 != st.session_state.bbox_wgs84:
            new_lv95 = wgs84_to_lv95(new_wgs84)
            ok, msg = validate_roi(new_lv95)
            if ok:
                st.session_state.bbox_wgs84 = new_wgs84
                st.session_state.bbox_lv95 = new_lv95
                st.session_state.roi_error = None
                # Clear old results when region changes
                st.session_state.result_zip = None
                st.session_state.result_plot = None
                st.session_state.result_meta = None
            else:
                st.session_state.bbox_wgs84 = None
                st.session_state.bbox_lv95 = None
                st.session_state.roi_error = msg

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
                    st.session_state.roi_error = None
                    st.session_state.result_zip = None
                    st.session_state.result_plot = None
                    st.success(f"Set: {manual_lv95}")
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
                bbox_from_shp, detected_crs = read_shp_bbox(
                    shp_file, shx_file=shx_file, dbf_file=dbf_file, prj_file=prj_file,
                )
                st.info(f"Detected CRS: {detected_crs}")
                st.caption(
                    f"Bounding box (LV95): E [{bbox_from_shp[0]:.0f}, {bbox_from_shp[2]:.0f}], "
                    f"N [{bbox_from_shp[1]:.0f}, {bbox_from_shp[3]:.0f}]"
                )

                ok, msg = validate_roi(bbox_from_shp)
                if ok:
                    if st.button("Use this bounding box", key="use_shp_bbox"):
                        st.session_state.bbox_lv95 = bbox_from_shp
                        st.session_state.bbox_wgs84 = lv95_to_wgs84(bbox_from_shp)
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
            # Step 1: Download
            status.write("Downloading DEM tiles from swisstopo...")
            dem_path = download_dem_tiles(
                bbox_wgs84, resolution, output_dir,
                log=lambda msg: status.write(msg),
            )

            # Step 2: Process DEM
            status.write("Processing DEM...")
            t_res = target_res if target_res > 0 else None
            x, y, Z, origin_lv95 = process_dem(
                dem_path, bbox_lv95, gaussian_sigma, t_res,
                log=lambda msg: status.write(msg),
            )

            # Step 3: Extend domain
            status.write("Extending domain with buffer zones...")
            x_ext, y_ext, Z_ext, z_top, z_ref = extend_domain(
                x, y, Z, buffer_width, ref_elevation, domain_height,
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

            if x_axis_north:
                status.write("Rotating coordinate system (X -> North)...")
                x_ext, y_ext, Z_ext = rotate_axes_north(
                    x_ext, y_ext, Z_ext,
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
                output_dir, mesh_cell_size, x_axis_north=x_axis_north,
                log=lambda msg: status.write(msg),
            )

            # Save metadata
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
                "x_axis_north": x_axis_north,
                "coordinate_system": "X=North, Y=West, Z=Up" if x_axis_north else "X=East, Y=North, Z=Up",
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

    with st.expander("ZIP contents"):
        st.code(
            "cfmesh_output/\n"
            "  constant/triSurface/bbox.fms\n"
            "  constant/triSurface/bbox.stl  (if enabled)\n"
            "  system/meshDict\n"
            "  system/createPatchDict\n"
            "  metadata.json\n"
            "  terrain_preview.png",
            language=None,
        )
        st.caption(
            "Extract into your OpenFOAM case directory, then run:\n"
            "`cartesianMesh && createPatch -overwrite && checkMesh`"
        )
