from __future__ import annotations

import json
import math
import time
from io import BytesIO

import psutil
import streamlit as st
from PIL import Image
from shapely.geometry import Polygon

import cad_engine as ce
from image_tools import (
    binary_preview_png,
    contours_to_polygons,
    contours_to_svg,
    rasterize_svg,
    trace_bitmap,
)
from preview import figure_for_shape


st.set_page_config(
    page_title="PolyForge CAD",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {min-width: 300px; max-width: 340px;}
.pf-card {border:1px solid rgba(128,128,128,.23); border-radius:16px; padding:14px 16px; margin:8px 0;}
.pf-muted {opacity:.68; font-size:.90rem;}
.pf-title {font-size:1.08rem; font-weight:700; margin-bottom:5px;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Session state ----------
def init_state():
    defaults = {
        "current_shape": None,
        "current_name": "Untitled",
        "current_parts": {},
        "current_meta": {},
        "last_build_seconds": None,
        "last_preview_seconds": None,
        "last_triangles": None,
        "trace_contours": None,
        "trace_size": None,
        "trace_polygons": None,
        "trace_svg": None,
        "scene_specs": [],
        "split_parts": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def set_current(shape, name: str, parts=None, meta=None, build_seconds=None):
    st.session_state.current_shape = shape
    st.session_state.current_name = name
    st.session_state.current_parts = parts or {"Model": shape}
    st.session_state.current_meta = meta or {}
    st.session_state.last_build_seconds = build_seconds
    st.session_state.split_parts = {}


def build_timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    shape = fn(*args, **kwargs)
    return shape, time.perf_counter() - t0


def memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def safe_name(name: str):
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "model"))
    return out.strip("_") or "model"


def show_preview(shape=None, quality="Standard"):
    shape = shape or st.session_state.current_shape
    if shape is None:
        st.info("Generate a model to see the 3D preview.")
        return
    tol_map = {"Draft": 0.65, "Standard": 0.28, "High": 0.12, "Ultra": 0.05}
    tol = tol_map.get(quality, 0.28)
    t0 = time.perf_counter()
    try:
        fig = figure_for_shape(shape, tolerance=tol)
        elapsed = time.perf_counter() - t0
        st.session_state.last_preview_seconds = elapsed
        try:
            stats = ce.shape_stats(shape, tessellation_tolerance=tol)
            st.session_state.last_triangles = stats["triangles"]
        except Exception:
            stats = None
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        if stats:
            cols = st.columns(5)
            cols[0].metric("X", f"{stats['x_mm']:.1f} mm")
            cols[1].metric("Y", f"{stats['y_mm']:.1f} mm")
            cols[2].metric("Z", f"{stats['z_mm']:.1f} mm")
            cols[3].metric("Triangles", f"{stats['triangles']:,}")
            cols[4].metric("Preview", f"{elapsed:.2f}s")
    except Exception as e:
        st.error(f"Preview failed: {e}")


def export_panel(shape=None, name=None, key_prefix="main"):
    shape = shape or st.session_state.current_shape
    if shape is None:
        return
    name = safe_name(name or st.session_state.current_name)
    st.markdown("### Export")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    stl_quality = c4.selectbox("STL quality", ["Draft", "Standard", "High", "Ultra"], index=1, key=f"{key_prefix}_stlq")
    tol_map = {"Draft": 0.12, "Standard": 0.04, "High": 0.015, "Ultra": 0.005}
    if c1.button("Prepare STL", key=f"{key_prefix}_prep_stl"):
        try:
            with st.spinner("Meshing STL…"):
                data = ce.export_bytes(shape, "stl", tolerance=tol_map[stl_quality])
            st.session_state[f"{key_prefix}_stl"] = data
        except Exception as e:
            st.error(f"STL export failed: {e}")
    if st.session_state.get(f"{key_prefix}_stl"):
        c1.download_button("Download STL", st.session_state[f"{key_prefix}_stl"], file_name=f"{name}.stl", mime="model/stl", key=f"{key_prefix}_down_stl")

    if c2.button("Prepare STEP", key=f"{key_prefix}_prep_step"):
        try:
            with st.spinner("Creating STEP…"):
                st.session_state[f"{key_prefix}_step"] = ce.export_bytes(shape, "step")
        except Exception as e:
            st.error(f"STEP export failed: {e}")
    if st.session_state.get(f"{key_prefix}_step"):
        c2.download_button("Download STEP", st.session_state[f"{key_prefix}_step"], file_name=f"{name}.step", mime="application/step", key=f"{key_prefix}_down_step")

    if c3.button("Prepare 3MF", key=f"{key_prefix}_prep_3mf"):
        try:
            with st.spinner("Creating 3MF…"):
                st.session_state[f"{key_prefix}_3mf"] = ce.export_bytes(shape, "3mf")
        except Exception as e:
            st.error(f"3MF export failed: {e}")
    if st.session_state.get(f"{key_prefix}_3mf"):
        c3.download_button("Download 3MF", st.session_state[f"{key_prefix}_3mf"], file_name=f"{name}.3mf", mime="model/3mf", key=f"{key_prefix}_down_3mf")


# ---------- Sidebar ----------
st.sidebar.markdown("# 🧩 PolyForge CAD")
st.sidebar.caption("Parametric 3D-print CAD · Streamlit performance build")
page = st.sidebar.radio(
    "Workspace",
    [
        "Home",
        "Quick Builders",
        "Image → SVG → 3D",
        "Free Build",
        "Split + Connect",
        "Parts + Export",
        "Performance Test",
    ],
)

st.sidebar.divider()
quality = st.sidebar.selectbox("3D preview quality", ["Draft", "Standard", "High", "Ultra"], index=1)
st.sidebar.caption("Use Draft while editing very complex models. Export quality is controlled separately.")
if st.session_state.current_shape is not None:
    st.sidebar.success(f"Current: {st.session_state.current_name}")
    if st.session_state.last_build_seconds is not None:
        st.sidebar.caption(f"Last CAD build: {st.session_state.last_build_seconds:.2f}s")
st.sidebar.caption(f"Server process memory: {memory_mb():.0f} MB")


# ---------- Home ----------
if page == "Home":
    st.title("PolyForge CAD")
    st.write("A cloud-first CAD prototype designed to test how far **Streamlit + CadQuery** can go before we need a stronger frontend or server.")
    st.markdown(
        """
<div class="pf-card"><div class="pf-title">What is working in this build</div>
<div class="pf-muted">Accurate mm-based models · clay cutters · trays and organisers · birth/sign plaques · holders · PNG/JPG tracing · SVG tracing · downloadable SVG · 3D from traced artwork · exact transform/boolean Free Build · model splitting · matched pin/socket connectors · STL/STEP/3MF export · performance diagnostics.</div></div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("### ⚡ Quick Builders\nCreate common printable products from measurements instead of modelling from scratch.")
    c2.markdown("### 🖼️ Image Studio\nTurn clean PNG/JPG or SVG artwork into traced vectors, clean the outline, and generate 3D parts/cutters.")
    c3.markdown("### 🔗 Split + Connect\nSplit the current CAD model on X/Y/Z and generate matching pin/socket connections with exact clearance.")
    st.info("This is a broad **functional beta**, not yet a Tinkercad replacement. Streamlit is excellent for testing the CAD engine and templates, but true mouse-drag handles and face selection will require a custom browser component later.")
    if st.session_state.current_shape is not None:
        st.subheader("Current model")
        show_preview(quality=quality)
        export_panel(key_prefix="home")


# ---------- Quick Builders ----------
if page == "Quick Builders":
    st.title("Quick Builders")
    builder = st.selectbox(
        "Builder",
        ["Rounded Tag / Badge", "Tray / Box", "Desk Organiser", "Clay Cutter", "Birth / Name Plaque", "Ring / Holder"],
    )
    left, right = st.columns([0.38, 0.62], gap="large")

    with left:
        if builder == "Rounded Tag / Badge":
            st.subheader("Rounded Tag / Badge")
            name = st.text_input("Design name", "Rounded_Tag")
            c1, c2 = st.columns(2)
            w = c1.number_input("Width (mm)", 5.0, 300.0, 80.0, 1.0)
            h = c2.number_input("Height (mm)", 5.0, 300.0, 30.0, 1.0)
            t = c1.number_input("Thickness (mm)", 0.4, 30.0, 3.0, 0.1)
            r = c2.number_input("Corner radius (mm)", 0.0, 80.0, 5.0, 0.5)
            ef = c1.number_input("Edge fillet (mm)", 0.0, 5.0, 0.4, 0.1)
            hole = st.checkbox("Add through-hole", True)
            hd = c2.number_input("Hole diameter (mm)", 0.5, 50.0, 4.0, 0.1, disabled=not hole)
            hx = c1.number_input("Hole X (mm)", -150.0, 150.0, -33.0, 0.5, disabled=not hole)
            hy = c2.number_input("Hole Y (mm)", -150.0, 150.0, 0.0, 0.5, disabled=not hole)
            if st.button("Generate tag", type="primary", use_container_width=True):
                try:
                    (shape, elapsed) = build_timed(ce.rounded_plate, w, h, t, r, ef, hd if hole else 0, hx, hy)
                    set_current(shape, name, {"Tag": shape}, {"builder": builder}, elapsed)
                except Exception as e:
                    st.error(str(e))

        elif builder == "Tray / Box":
            st.subheader("Tray / Box")
            name = st.text_input("Design name", "Rounded_Tray")
            c1, c2 = st.columns(2)
            w = c1.number_input("Width (mm)", 20.0, 400.0, 120.0, 1.0)
            d = c2.number_input("Depth (mm)", 20.0, 400.0, 85.0, 1.0)
            h = c1.number_input("Height (mm)", 5.0, 250.0, 35.0, 1.0)
            wall = c2.number_input("Wall (mm)", 0.8, 15.0, 2.4, 0.1)
            floor = c1.number_input("Floor (mm)", 0.8, 20.0, 2.4, 0.1)
            rad = c2.number_input("Corner radius (mm)", 0.0, 50.0, 8.0, 0.5)
            if st.button("Generate tray", type="primary", use_container_width=True):
                try:
                    shape, elapsed = build_timed(ce.tray, w, d, h, wall, floor, rad)
                    set_current(shape, name, {"Tray": shape}, {"builder": builder}, elapsed)
                except Exception as e:
                    st.error(str(e))

        elif builder == "Desk Organiser":
            st.subheader("Desk Organiser")
            name = st.text_input("Design name", "Desk_Organiser")
            c1, c2 = st.columns(2)
            w = c1.number_input("Width (mm)", 60.0, 350.0, 190.0, 1.0)
            d = c2.number_input("Depth (mm)", 50.0, 300.0, 130.0, 1.0)
            h = c1.number_input("Body height (mm)", 20.0, 180.0, 70.0, 1.0)
            wall = c2.number_input("Wall (mm)", 1.0, 8.0, 2.4, 0.1)
            floor = c1.number_input("Floor (mm)", 1.0, 10.0, 2.6, 0.1)
            rad = c2.number_input("Corner radius (mm)", 0.0, 35.0, 10.0, 0.5)
            cols = c1.number_input("Columns", 1, 8, 3, 1)
            rows = c2.number_input("Rows", 1, 8, 2, 1)
            div = c1.number_input("Divider thickness (mm)", 1.0, 8.0, 2.2, 0.1)
            handle = st.checkbox("Add carry handle", True)
            hh = c2.number_input("Overall handle height (mm)", h, 260.0, 125.0, 1.0, disabled=not handle)
            hw = c1.number_input("Handle front/back width (mm)", 4.0, 40.0, 14.0, 1.0, disabled=not handle)
            ht = c2.number_input("Handle thickness (mm)", 2.0, 20.0, 6.0, 0.5, disabled=not handle)
            separate_handle = st.checkbox("Keep handle as separate part", True, disabled=not handle)
            ribs = st.checkbox("Add vertical ribs/flutes (heavier calculation)", False)
            rp = c1.number_input("Rib pitch (mm)", 3.0, 25.0, 9.0, 0.5, disabled=not ribs)
            rd = c2.number_input("Rib depth (mm)", 0.4, 5.0, 1.4, 0.1, disabled=not ribs)
            if st.button("Generate organiser", type="primary", use_container_width=True):
                try:
                    t0 = time.perf_counter()
                    shape, parts = ce.organizer(w, d, h, wall, floor, rad, int(cols), int(rows), div, handle, hw, ht, hh, not separate_handle, ribs, rp, rd)
                    elapsed = time.perf_counter() - t0
                    set_current(shape, name, parts, {"builder": builder}, elapsed)
                except Exception as e:
                    st.error(str(e))

        elif builder == "Clay Cutter":
            st.subheader("Clay Cutter")
            name = st.text_input("Design name", "Clay_Cutter")
            shape_name = st.selectbox("Built-in outline", ["Heart", "Circle", "Star", "Rounded Rectangle"])
            width = st.number_input("Overall width (mm)", 8.0, 150.0, 35.0, 1.0)
            use_exact_h = st.checkbox("Set exact height too", False)
            height = st.number_input("Overall height (mm)", 8.0, 150.0, 35.0, 1.0, disabled=not use_exact_h)
            c1, c2 = st.columns(2)
            depth = c1.number_input("Cutter height (mm)", 3.0, 40.0, 12.0, 0.5)
            wall = c2.number_input("Main wall (mm)", 0.6, 4.0, 1.2, 0.1)
            edge_h = c1.number_input("Cutting edge height (mm)", 0.0, 6.0, 2.0, 0.1)
            edge_w = c2.number_input("Cutting edge wall (mm)", 0.35, 2.5, 0.65, 0.05)
            grip_h = c1.number_input("Grip rim height (mm)", 0.0, 6.0, 2.0, 0.1)
            grip_w = c2.number_input("Grip rim extra width (mm)", 0.0, 5.0, 1.8, 0.1)
            if st.button("Generate cutter", type="primary", use_container_width=True):
                try:
                    if shape_name == "Heart":
                        polys = [ce.heart_polygon()]
                    elif shape_name == "Circle":
                        polys = [ce.circle_polygon()]
                    elif shape_name == "Star":
                        polys = [ce.star_polygon()]
                    else:
                        polys = [ce.rounded_rect_polygon()]
                    t0 = time.perf_counter()
                    shape, parts = ce.clay_cutter(polys, width, height if use_exact_h else None, depth, wall, edge_h, edge_w, grip_h, grip_w)
                    set_current(shape, name, parts, {"builder": builder}, time.perf_counter() - t0)
                except Exception as e:
                    st.error(str(e))

        elif builder == "Birth / Name Plaque":
            st.subheader("Birth / Name Plaque")
            name = st.text_input("Design name", "Birth_Plaque")
            c1, c2 = st.columns(2)
            w = c1.number_input("Plaque width (mm)", 70.0, 320.0, 180.0, 1.0)
            h = c2.number_input("Plaque height (mm)", 70.0, 350.0, 180.0, 1.0)
            t = c1.number_input("Plaque thickness (mm)", 2.0, 20.0, 5.0, 0.5)
            ar = c2.number_input("Arch radius (mm)", 10.0, 160.0, 70.0, 1.0)
            bw = c1.number_input("Base width (mm)", 80.0, 400.0, 185.0, 1.0)
            bd = c2.number_input("Base depth (mm)", 20.0, 150.0, 55.0, 1.0)
            bh = c1.number_input("Base height (mm)", 4.0, 40.0, 12.0, 0.5)
            clr = c2.number_input("Slot clearance (mm)", 0.0, 2.0, 0.35, 0.05)
            txt = st.text_input("Raised name", "NOAH")
            ts = c1.number_input("Name size (mm)", 5.0, 80.0, 20.0, 1.0)
            tr = c2.number_input("Name raised height (mm)", 0.0, 5.0, 1.0, 0.1)
            if st.button("Generate plaque", type="primary", use_container_width=True):
                try:
                    t0 = time.perf_counter()
                    shape, parts = ce.birth_plaque(w, h, t, ar, bw, bd, bh, clr, txt, ts, tr)
                    set_current(shape, name, parts, {"builder": builder}, time.perf_counter() - t0)
                except Exception as e:
                    st.error(str(e))

        elif builder == "Ring / Holder":
            st.subheader("Ring / Holder")
            name = st.text_input("Design name", "Cylinder_Holder")
            c1, c2 = st.columns(2)
            od = c1.number_input("Outer diameter (mm)", 5.0, 250.0, 45.0, 0.5)
            id_ = c2.number_input("Inner diameter (mm)", 2.0, 245.0, 38.0, 0.5)
            h = c1.number_input("Height (mm)", 2.0, 250.0, 30.0, 1.0)
            base = c2.number_input("Base thickness (0=open) (mm)", 0.0, 20.0, 2.4, 0.1)
            fil = c1.number_input("Edge fillet (mm)", 0.0, 4.0, 0.4, 0.1)
            if st.button("Generate holder", type="primary", use_container_width=True):
                try:
                    shape, elapsed = build_timed(ce.ring_holder, od, id_, h, base, fil)
                    set_current(shape, name, {"Holder": shape}, {"builder": builder}, elapsed)
                except Exception as e:
                    st.error(str(e))

    with right:
        st.subheader("Live result")
        if st.session_state.current_shape is not None:
            show_preview(quality=quality)
            export_panel(key_prefix="quick")
        else:
            st.info("Choose measurements and press Generate.")


# ---------- Image Studio ----------
if page == "Image → SVG → 3D":
    st.title("Image → SVG → 3D")
    st.write("Upload a clean PNG/JPG or SVG. This workspace traces the artwork, lets you control cleanup, downloads the traced SVG, and can turn the traced regions into a 3D plate or clay cutter.")
    left, right = st.columns([0.38, 0.62], gap="large")
    with left:
        upload = st.file_uploader("PNG, JPG or SVG", type=["png", "jpg", "jpeg", "svg"])
        c1, c2 = st.columns(2)
        threshold = c1.slider("Threshold", 0, 255, 165, 1)
        simplify = c2.slider("Smooth / simplify", 0.02, 2.0, 0.22, 0.02, help="Higher values remove more small nodes.")
        blur = c1.selectbox("Blur before trace", [0, 3, 5, 7], index=0)
        min_area = c2.number_input("Ignore details smaller than (px²)", 1.0, 5000.0, 30.0, 5.0)
        invert = st.checkbox("Invert foreground/background", False)
        external = st.checkbox("Outer regions only", True)
        max_parts = st.slider("Maximum detected parts", 1, 40, 16)
        svg_raster = st.slider("SVG trace resolution", 600, 3000, 1800, 100, help="SVGs are rasterised at high resolution for a robust first-version importer.")

        if upload and st.button("Trace artwork", type="primary", use_container_width=True):
            try:
                raw = upload.getvalue()
                if upload.name.lower().endswith(".svg"):
                    with st.spinner("Rasterising SVG at high resolution…"):
                        raw = rasterize_svg(raw, svg_raster)
                contours, binary, size = trace_bitmap(raw, threshold, invert, int(blur), simplify, min_area, external)
                contours = contours[:max_parts]
                polys = contours_to_polygons(contours, size, max_parts=max_parts)
                svg = contours_to_svg(contours, size)
                st.session_state.trace_contours = contours
                st.session_state.trace_size = size
                st.session_state.trace_polygons = polys
                st.session_state.trace_svg = svg
                st.session_state.trace_binary = binary_preview_png(binary)
                st.success(f"Detected {len(polys)} printable region(s).")
            except Exception as e:
                st.error(f"Trace failed: {e}")

        if st.session_state.trace_svg:
            st.download_button("Download converted SVG", st.session_state.trace_svg.encode("utf-8"), "traced_design.svg", "image/svg+xml", use_container_width=True)
            st.divider()
            st.subheader("Make it 3D")
            mode = st.selectbox("3D type", ["Clay Cutter", "Flat Graphic / Topper"])
            target_w = st.number_input("Target width (mm)", 5.0, 350.0, 45.0, 1.0)
            exact_h = st.checkbox("Set exact height", False, key="img_exact_h")
            target_h = st.number_input("Target height (mm)", 5.0, 350.0, 45.0, 1.0, disabled=not exact_h)
            if mode == "Clay Cutter":
                c1, c2 = st.columns(2)
                depth = c1.number_input("Cutter height (mm)", 3.0, 40.0, 12.0, 0.5, key="img_depth")
                wall = c2.number_input("Main wall (mm)", 0.6, 4.0, 1.2, 0.1, key="img_wall")
                edgeh = c1.number_input("Edge height (mm)", 0.0, 6.0, 2.0, 0.1, key="img_edgeh")
                edgew = c2.number_input("Edge wall (mm)", 0.35, 2.5, 0.65, 0.05, key="img_edgew")
                griph = c1.number_input("Grip height (mm)", 0.0, 6.0, 2.0, 0.1, key="img_griph")
                gripw = c2.number_input("Grip extra width (mm)", 0.0, 5.0, 1.8, 0.1, key="img_gripw")
            else:
                thick = st.number_input("Extrusion thickness (mm)", 0.4, 30.0, 3.0, 0.2)

            if st.button("Generate 3D from trace", type="primary", use_container_width=True):
                try:
                    polys = st.session_state.trace_polygons or []
                    if not polys:
                        raise ValueError("No trace regions are available.")
                    t0 = time.perf_counter()
                    if mode == "Clay Cutter":
                        shape, parts = ce.clay_cutter(polys, target_w, target_h if exact_h else None, depth, wall, edgeh, edgew, griph, gripw)
                    else:
                        norm = ce.normalize_polygons(polys, target_w, target_h if exact_h else None)
                        parts = {f"Graphic_{i+1}": ce.extrude_polygon(p, thick) for i, p in enumerate(norm)}
                        shape = ce.make_compound(parts.values())
                    set_current(shape, "Traced_3D", parts, {"source": "image_trace"}, time.perf_counter() - t0)
                    st.success("3D model generated.")
                except Exception as e:
                    st.error(f"3D generation failed: {e}")

    with right:
        if st.session_state.get("trace_binary"):
            st.subheader("Trace preview")
            st.image(st.session_state.trace_binary, caption="White regions are the detected printable shapes.", use_container_width=True)
        if st.session_state.current_shape is not None and st.session_state.current_meta.get("source") == "image_trace":
            st.subheader("3D result")
            show_preview(quality=quality)
            export_panel(key_prefix="image")


# ---------- Free Build ----------
def new_scene_spec(kind="Rounded Box"):
    base = {
        "kind": kind,
        "operation": "Solid / Join",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
    }
    if kind == "Rounded Box":
        base.update(width=40.0, depth=30.0, height=8.0, radius=3.0)
    elif kind == "Cylinder":
        base.update(diameter=20.0, height=8.0)
    elif kind == "Text":
        base.update(text="NAME", size=12.0, height=1.2, bold=False)
    return base


def shape_from_spec(s):
    kind = s["kind"]
    if kind == "Rounded Box":
        shp = ce.rounded_prism(s["width"], s["depth"], s["height"], s["radius"])
    elif kind == "Cylinder":
        shp = ce.ring_holder(s["diameter"], 0.001, s["height"], s["height"], 0)  # effectively solid cylinder
    else:
        shp = ce.text_solid(s["text"], s["size"], s["height"], bold=s.get("bold", False))
    return ce.transform_shape(shp, s["x"], s["y"], s["z"], s["rx"], s["ry"], s["rz"])


if page == "Free Build":
    st.title("Free Build")
    st.write("A parameter-based Tinkercad-style test: add solids and holes, position them exactly, then rebuild. True mouse-drag handles are a later custom-component feature.")
    add1, add2, add3, add4 = st.columns(4)
    if add1.button("+ Rounded Box", use_container_width=True):
        st.session_state.scene_specs.append(new_scene_spec("Rounded Box")); st.rerun()
    if add2.button("+ Cylinder", use_container_width=True):
        st.session_state.scene_specs.append(new_scene_spec("Cylinder")); st.rerun()
    if add3.button("+ Text", use_container_width=True):
        st.session_state.scene_specs.append(new_scene_spec("Text")); st.rerun()
    if add4.button("Clear scene", use_container_width=True):
        st.session_state.scene_specs = []; st.rerun()

    for i, spec in enumerate(st.session_state.scene_specs):
        with st.expander(f"{i+1}. {spec['kind']} — {spec['operation']}", expanded=(i == len(st.session_state.scene_specs)-1)):
            a, b, c = st.columns(3)
            spec["operation"] = a.selectbox("Operation", ["Solid / Join", "Hole / Cut", "Separate part"], index=["Solid / Join", "Hole / Cut", "Separate part"].index(spec["operation"]), key=f"op{i}")
            if spec["kind"] == "Rounded Box":
                spec["width"] = a.number_input("Width", 0.5, 500.0, spec["width"], 0.5, key=f"w{i}")
                spec["depth"] = b.number_input("Depth", 0.5, 500.0, spec["depth"], 0.5, key=f"d{i}")
                spec["height"] = c.number_input("Height", 0.5, 500.0, spec["height"], 0.5, key=f"h{i}")
                spec["radius"] = a.number_input("Corner radius", 0.0, 100.0, spec["radius"], 0.5, key=f"r{i}")
            elif spec["kind"] == "Cylinder":
                spec["diameter"] = a.number_input("Diameter", 0.5, 500.0, spec["diameter"], 0.5, key=f"dia{i}")
                spec["height"] = b.number_input("Height", 0.5, 500.0, spec["height"], 0.5, key=f"ch{i}")
            else:
                spec["text"] = a.text_input("Text", spec["text"], key=f"txt{i}")
                spec["size"] = b.number_input("Text size", 1.0, 200.0, spec["size"], 1.0, key=f"ts{i}")
                spec["height"] = c.number_input("Text height", 0.2, 30.0, spec["height"], 0.1, key=f"th{i}")
                spec["bold"] = a.checkbox("Bold", spec.get("bold", False), key=f"tb{i}")
            st.caption("Position")
            spec["x"] = a.number_input("X", -500.0, 500.0, spec["x"], 0.5, key=f"x{i}")
            spec["y"] = b.number_input("Y", -500.0, 500.0, spec["y"], 0.5, key=f"y{i}")
            spec["z"] = c.number_input("Z", -500.0, 500.0, spec["z"], 0.5, key=f"z{i}")
            st.caption("Rotation")
            spec["rx"] = a.number_input("Rotate X°", -360.0, 360.0, spec["rx"], 5.0, key=f"rx{i}")
            spec["ry"] = b.number_input("Rotate Y°", -360.0, 360.0, spec["ry"], 5.0, key=f"ry{i}")
            spec["rz"] = c.number_input("Rotate Z°", -360.0, 360.0, spec["rz"], 5.0, key=f"rz{i}")
            if st.button("Delete object", key=f"del{i}"):
                st.session_state.scene_specs.pop(i); st.rerun()

    if st.button("Rebuild scene", type="primary", use_container_width=True, disabled=not st.session_state.scene_specs):
        try:
            t0 = time.perf_counter()
            joined = None
            separate = []
            parts = {}
            for i, spec in enumerate(st.session_state.scene_specs):
                shp = shape_from_spec(spec)
                parts[f"Object_{i+1}_{spec['kind'].replace(' ', '_')}"] = shp
                if spec["operation"] == "Separate part":
                    separate.append(shp)
                elif spec["operation"] == "Solid / Join":
                    joined = shp if joined is None else joined.fuse(shp)
                else:
                    if joined is not None:
                        joined = joined.cut(shp)
            display = ce.make_compound(([joined] if joined is not None else []) + separate)
            if display is None:
                raise ValueError("The scene contains no visible solid.")
            set_current(display, "Free_Build", parts, {"builder": "Free Build", "scene": st.session_state.scene_specs}, time.perf_counter()-t0)
        except Exception as e:
            st.error(f"Scene rebuild failed: {e}")

    if st.session_state.current_shape is not None:
        show_preview(quality=quality)
        scene_json = json.dumps(st.session_state.scene_specs, indent=2)
        st.download_button("Save editable scene JSON", scene_json, "polyforge_scene.json", "application/json")
        export_panel(key_prefix="free")


# ---------- Split + Connect ----------
if page == "Split + Connect":
    st.title("Split + Connect")
    if st.session_state.current_shape is None:
        st.warning("Generate a model first in Quick Builders, Image Studio or Free Build.")
    else:
        st.write(f"Current model: **{st.session_state.current_name}**")
        bb = st.session_state.current_shape.BoundingBox()
        axis = st.selectbox("Split axis", ["X", "Y", "Z"])
        mins = {"X": bb.xmin, "Y": bb.ymin, "Z": bb.zmin}
        maxs = {"X": bb.xmax, "Y": bb.ymax, "Z": bb.zmax}
        mid = (mins[axis] + maxs[axis]) / 2
        pos = st.slider("Split position (mm)", float(mins[axis] + 0.5), float(maxs[axis] - 0.5), float(mid), 0.5)
        c1, c2, c3 = st.columns(3)
        count = c1.number_input("Pin count", 0, 8, 2, 1)
        pd = c2.number_input("Pin diameter (mm)", 1.0, 20.0, 5.0, 0.1)
        pl = c3.number_input("Pin length (mm)", 1.0, 25.0, 6.0, 0.5)
        clearance = c1.number_input("Socket clearance (mm)", 0.0, 1.5, 0.20, 0.05)
        margin = c2.number_input("Edge margin (mm)", 1.0, 60.0, 10.0, 1.0)
        fit = c3.selectbox("Fit preset", ["Custom", "Tight (~0.10)", "Normal (~0.20)", "Easy (~0.30)"])
        if fit != "Custom":
            clearance = {"Tight (~0.10)": 0.10, "Normal (~0.20)": 0.20, "Easy (~0.30)": 0.30}[fit]
            st.caption(f"Using {clearance:.2f} mm radial clearance in the CAD model.")
        if st.button("Split and add matching connectors", type="primary"):
            try:
                t0 = time.perf_counter()
                a, b = ce.split_with_pins(st.session_state.current_shape, axis, pos, int(count), pd, pl, clearance, margin)
                st.session_state.split_parts = {"Part_A_Male": a, "Part_B_Socket": b}
                set_current(ce.make_compound([a, b]), st.session_state.current_name + "_Split", st.session_state.split_parts, {"builder": "Split + Connect"}, time.perf_counter()-t0)
                # set_current clears split_parts, restore it.
                st.session_state.split_parts = {"Part_A_Male": a, "Part_B_Socket": b}
            except Exception as e:
                st.error(f"Split failed: {e}")
        if st.session_state.split_parts:
            show_preview(quality=quality)
            st.subheader("Separate part downloads")
            for j, (pn, shp) in enumerate(st.session_state.split_parts.items()):
                st.markdown(f"**{pn}**")
                export_panel(shp, f"{safe_name(st.session_state.current_name)}_{pn}", key_prefix=f"split{j}")
        else:
            show_preview(quality=quality)


# ---------- Parts ----------
if page == "Parts + Export":
    st.title("Parts + Export")
    if st.session_state.current_shape is None:
        st.info("No current model yet.")
    else:
        st.write(f"### {st.session_state.current_name}")
        show_preview(quality=quality)
        parts = st.session_state.current_parts or {"Model": st.session_state.current_shape}
        st.markdown("### Individual parts")
        st.caption("Useful for multi-colour/manual colour prints, separate handles, bases, or split assemblies.")
        for idx, (name, shape) in enumerate(parts.items()):
            with st.expander(name):
                try:
                    stats = ce.shape_stats(shape, 0.4)
                    st.write(f"Size: {stats['x_mm']:.1f} × {stats['y_mm']:.1f} × {stats['z_mm']:.1f} mm")
                except Exception:
                    pass
                export_panel(shape, f"{safe_name(st.session_state.current_name)}_{safe_name(name)}", key_prefix=f"part{idx}")
        st.divider()
        st.markdown("### Whole model / assembly")
        export_panel(key_prefix="parts_whole")


# ---------- Performance ----------
if page == "Performance Test":
    st.title("Streamlit + CadQuery Performance Test")
    st.write("This page is specifically for deciding whether free Streamlit hosting is strong enough for the models you want to make.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current process RAM", f"{memory_mb():.0f} MB")
    c2.metric("Last CAD build", "—" if st.session_state.last_build_seconds is None else f"{st.session_state.last_build_seconds:.2f}s")
    c3.metric("Last preview", "—" if st.session_state.last_preview_seconds is None else f"{st.session_state.last_preview_seconds:.2f}s")
    c4.metric("Last triangles", "—" if st.session_state.last_triangles is None else f"{st.session_state.last_triangles:,}")

    st.subheader("Stress model")
    st.caption("Generates a ribbed organiser with many booleans. Start modestly on Community Cloud.")
    a, b, c = st.columns(3)
    w = a.number_input("Width", 100.0, 280.0, 180.0, 10.0, key="stressw")
    d = b.number_input("Depth", 80.0, 220.0, 125.0, 5.0, key="stressd")
    cols = c.slider("Columns", 1, 7, 4)
    rows = a.slider("Rows", 1, 6, 3)
    pitch = b.slider("Rib pitch", 4.0, 18.0, 9.0, 0.5)
    if st.button("Run stress build", type="primary"):
        before = memory_mb()
        try:
            with st.spinner("Running CadQuery stress model…"):
                t0 = time.perf_counter()
                shape, parts = ce.organizer(w, d, 70, 2.4, 2.6, 10, cols, rows, 2.2, True, 14, 6, 130, True, True, pitch, 1.4)
                build_s = time.perf_counter() - t0
                set_current(shape, "Stress_Organiser", parts, {"builder": "Stress"}, build_s)
                after_build = memory_mb()
                t1 = time.perf_counter()
                stats = ce.shape_stats(shape, 0.25)
                tess_s = time.perf_counter() - t1
                after_tess = memory_mb()
            st.success("Stress test completed.")
            st.write({
                "CAD build seconds": round(build_s, 3),
                "Tessellation seconds": round(tess_s, 3),
                "RAM before MB": round(before, 1),
                "RAM after build MB": round(after_build, 1),
                "RAM after tessellation MB": round(after_tess, 1),
                "Triangles": stats["triangles"],
                "Model size mm": [round(stats["x_mm"],1), round(stats["y_mm"],1), round(stats["z_mm"],1)],
            })
        except Exception as e:
            st.error(f"Stress test failed: {e}")
    if st.session_state.current_shape is not None and st.session_state.current_meta.get("builder") == "Stress":
        show_preview(quality="Draft")
