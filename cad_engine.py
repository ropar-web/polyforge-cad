from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import cadquery as cq
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


def as_shape(obj):
    """Return a CadQuery Shape from a Workplane/Shape/Assembly-like object."""
    if obj is None:
        return None
    if isinstance(obj, cq.Shape):
        return obj
    if isinstance(obj, cq.Workplane):
        vals = obj.vals()
        if len(vals) == 1:
            return vals[0]
        return cq.Compound.makeCompound(vals)
    if isinstance(obj, (list, tuple)):
        vals = [as_shape(x) for x in obj if x is not None]
        vals = [x for x in vals if x is not None]
        return cq.Compound.makeCompound(vals) if len(vals) > 1 else (vals[0] if vals else None)
    if hasattr(obj, "val"):
        return obj.val()
    raise TypeError(f"Unsupported CAD object: {type(obj)!r}")


def make_compound(parts: Iterable):
    vals = [as_shape(x) for x in parts if x is not None]
    vals = [x for x in vals if x is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return cq.Compound.makeCompound(vals)


def _clamp_fillet(radius: float, width: float, depth: float) -> float:
    return max(0.0, min(float(radius), max(0.0, width / 2 - 0.02), max(0.0, depth / 2 - 0.02)))


def rounded_prism(width: float, depth: float, height: float, radius: float = 0.0):
    """Rounded XY rectangle extruded from Z=0 to height."""
    width, depth, height = map(float, (width, depth, height))
    if min(width, depth, height) <= 0:
        raise ValueError("Width, depth and height must be positive.")
    wp = cq.Workplane("XY").rect(width, depth).extrude(height)
    r = _clamp_fillet(radius, width, depth)
    if r > 0:
        try:
            wp = wp.edges("|Z").fillet(r)
        except Exception:
            pass
    return as_shape(wp)


def rounded_plate(
    width: float,
    height: float,
    thickness: float,
    radius: float = 4.0,
    edge_fillet: float = 0.0,
    hole_diameter: float = 0.0,
    hole_x: float = 0.0,
    hole_y: float = 0.0,
):
    shape = rounded_prism(width, height, thickness, radius)
    if hole_diameter and hole_diameter > 0:
        hole = cq.Solid.makeCylinder(
            hole_diameter / 2,
            thickness + 2,
            cq.Vector(hole_x, hole_y, -1),
            cq.Vector(0, 0, 1),
        )
        shape = shape.cut(hole)
    if edge_fillet and edge_fillet > 0:
        try:
            shape = cq.Workplane(obj=shape).edges().fillet(float(edge_fillet)).val()
        except Exception:
            pass
    return shape


def tray(
    width: float,
    depth: float,
    height: float,
    wall: float = 2.0,
    floor: float = 2.0,
    radius: float = 5.0,
):
    width, depth, height, wall, floor = map(float, (width, depth, height, wall, floor))
    if wall * 2 >= min(width, depth):
        raise ValueError("Wall thickness is too large for the tray size.")
    if floor >= height:
        raise ValueError("Floor thickness must be smaller than tray height.")
    outer = rounded_prism(width, depth, height, radius)
    iw, idp = width - 2 * wall, depth - 2 * wall
    inner_radius = max(0.0, float(radius) - wall)
    inner = rounded_prism(iw, idp, height - floor + 1.0, inner_radius)
    inner = inner.translate(cq.Vector(0, 0, floor))
    return outer.cut(inner)


def _box_centered(x_len: float, y_len: float, z_len: float, center: Tuple[float, float, float]):
    cx, cy, cz = center
    return cq.Solid.makeBox(
        x_len,
        y_len,
        z_len,
        cq.Vector(cx - x_len / 2, cy - y_len / 2, cz - z_len / 2),
    )


def organizer(
    width: float,
    depth: float,
    height: float,
    wall: float = 2.0,
    floor: float = 2.0,
    radius: float = 8.0,
    columns: int = 3,
    rows: int = 2,
    divider: float = 2.0,
    handle: bool = True,
    handle_width: float = 14.0,
    handle_thickness: float = 5.0,
    handle_height: float = 65.0,
    handle_joined: bool = True,
    ribs: bool = False,
    rib_pitch: float = 8.0,
    rib_depth: float = 1.2,
):
    """Parametric organiser with tray, divider grid and optional U handle/ribs.

    Returns (display_shape, parts_dict). If handle_joined=False, handle is a separate part.
    """
    body = tray(width, depth, height, wall, floor, radius)
    parts: Dict[str, cq.Shape] = {"Body": body}
    combined = body

    # Divider grid; each divider starts at floor and reaches body height.
    inner_w = width - 2 * wall
    inner_d = depth - 2 * wall
    divider_h = max(0.1, height - floor)
    if columns > 1:
        cell = inner_w / columns
        for i in range(1, int(columns)):
            x = -inner_w / 2 + cell * i
            d = _box_centered(divider, inner_d, divider_h, (x, 0, floor + divider_h / 2))
            combined = combined.fuse(d)
    if rows > 1:
        cell = inner_d / rows
        for i in range(1, int(rows)):
            y = -inner_d / 2 + cell * i
            d = _box_centered(inner_w, divider, divider_h, (0, y, floor + divider_h / 2))
            combined = combined.fuse(d)

    # Optional decorative ribs. They deliberately use simple geometry to keep cloud compute reasonable.
    if ribs and rib_pitch > 1 and rib_depth > 0:
        rib_r = rib_depth / 2
        rib_h = max(1.0, height - 1.0)
        # Front/back faces: cylinders with vertical Z axis, placed just outside the walls.
        x = -width / 2 + rib_pitch / 2
        ribs_to_add = []
        while x < width / 2:
            for y in (-depth / 2 - rib_r * 0.35, depth / 2 + rib_r * 0.35):
                ribs_to_add.append(cq.Solid.makeCylinder(rib_r, rib_h, cq.Vector(x, y, 0.5), cq.Vector(0, 0, 1)))
            x += rib_pitch
        y = -depth / 2 + rib_pitch / 2
        while y < depth / 2:
            for x2 in (-width / 2 - rib_r * 0.35, width / 2 + rib_r * 0.35):
                ribs_to_add.append(cq.Solid.makeCylinder(rib_r, rib_h, cq.Vector(x2, y, 0.5), cq.Vector(0, 0, 1)))
            y += rib_pitch
        # Fusing all in one go is generally faster than many separate display parts.
        try:
            combined = combined.fuse(*ribs_to_add)
        except Exception:
            for rib in ribs_to_add:
                try:
                    combined = combined.fuse(rib)
                except Exception:
                    pass

    # U-shaped handle in X-Z plane, centered in Y.
    if handle:
        post_sep = max(20.0, min(width - 2 * wall - handle_thickness * 2, width * 0.62))
        post_h = max(handle_thickness, handle_height - height)
        z0 = height
        left = _box_centered(handle_thickness, handle_width, post_h, (-post_sep / 2, 0, z0 + post_h / 2))
        right = _box_centered(handle_thickness, handle_width, post_h, (post_sep / 2, 0, z0 + post_h / 2))
        top = _box_centered(post_sep + handle_thickness, handle_width, handle_thickness, (0, 0, z0 + post_h))
        handle_shape = left.fuse(right, top)
        # Smooth handle edges gently if possible.
        try:
            handle_shape = cq.Workplane(obj=handle_shape).edges().fillet(min(handle_thickness * 0.28, 2.0)).val()
        except Exception:
            pass
        parts["Handle"] = handle_shape
        if handle_joined:
            combined = combined.fuse(handle_shape)
        else:
            # body stays a single solid; display/export can be compound.
            parts["Body"] = combined
            return make_compound([combined, handle_shape]), parts

    parts["Body"] = combined
    return combined, parts


def ring_holder(
    outer_diameter: float,
    inner_diameter: float,
    height: float,
    base_thickness: float = 0.0,
    edge_fillet: float = 0.0,
):
    od, id_, h = map(float, (outer_diameter, inner_diameter, height))
    if id_ <= 0 or od <= id_:
        raise ValueError("Outer diameter must be larger than inner diameter.")
    outer = cq.Solid.makeCylinder(od / 2, h, cq.Vector(0, 0, 0), cq.Vector(0, 0, 1))
    inner_start = max(0.0, float(base_thickness))
    inner = cq.Solid.makeCylinder(id_ / 2, h - inner_start + 1, cq.Vector(0, 0, inner_start), cq.Vector(0, 0, 1))
    result = outer.cut(inner)
    if edge_fillet > 0:
        try:
            result = cq.Workplane(obj=result).edges().fillet(float(edge_fillet)).val()
        except Exception:
            pass
    return result


def heart_polygon(samples: int = 160) -> Polygon:
    pts = []
    for i in range(samples):
        t = 2 * math.pi * i / samples
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((x, y))
    return Polygon(pts).buffer(0)


def star_polygon(points: int = 5, outer: float = 1.0, inner: float = 0.46) -> Polygon:
    pts = []
    for i in range(points * 2):
        a = -math.pi / 2 + i * math.pi / points
        r = outer if i % 2 == 0 else inner
        pts.append((math.cos(a) * r, math.sin(a) * r))
    return Polygon(pts).buffer(0)


def rounded_rect_polygon(width: float = 2.0, height: float = 1.5, radius: float = 0.2) -> Polygon:
    # Shapely buffer gives rounded corners.
    inner_w = max(0.001, width - 2 * radius)
    inner_h = max(0.001, height - 2 * radius)
    p = Polygon([
        (-inner_w / 2, -inner_h / 2),
        (inner_w / 2, -inner_h / 2),
        (inner_w / 2, inner_h / 2),
        (-inner_w / 2, inner_h / 2),
    ])
    return p.buffer(radius, join_style=1)


def circle_polygon(samples: int = 160) -> Polygon:
    return Polygon([(math.cos(2 * math.pi * i / samples), math.sin(2 * math.pi * i / samples)) for i in range(samples)])


def normalize_polygons(polygons: Sequence[Polygon], target_width: float, target_height: Optional[float] = None):
    polys = [p.buffer(0) for p in polygons if p is not None and not p.is_empty and p.area > 1e-8]
    if not polys:
        raise ValueError("No usable closed outline was found.")
    union = unary_union(polys)
    minx, miny, maxx, maxy = union.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        raise ValueError("Outline has zero width or height.")
    sx = float(target_width) / w
    sy = sx if target_height is None else float(target_height) / h
    scaled = [affinity.scale(p, xfact=sx, yfact=sy, origin=(minx, miny)) for p in polys]
    union2 = unary_union(scaled)
    minx2, miny2, maxx2, maxy2 = union2.bounds
    cx, cy = (minx2 + maxx2) / 2, (miny2 + maxy2) / 2
    return [affinity.translate(p, xoff=-cx, yoff=-cy) for p in scaled]


def _wire_from_coords(coords) -> cq.Wire:
    pts = [cq.Vector(float(x), float(y), 0) for x, y in list(coords)[:-1]]
    if len(pts) < 3:
        raise ValueError("Polygon needs at least 3 points.")
    return cq.Wire.makePolygon(pts, close=True)


def polygon_face(poly: Polygon) -> cq.Face:
    p = poly.buffer(0)
    if p.is_empty or not isinstance(p, Polygon):
        raise ValueError("Expected a single polygon.")
    outer = _wire_from_coords(p.exterior.coords)
    inners = [_wire_from_coords(r.coords) for r in p.interiors]
    return cq.Face.makeFromWires(outer, inners)


def extrude_polygon(poly: Polygon, height: float, z: float = 0.0):
    p = poly.buffer(0)
    if isinstance(p, MultiPolygon):
        return make_compound([extrude_polygon(x, height, z) for x in p.geoms])
    face = polygon_face(p)
    solid = cq.Solid.extrudeLinear(face.outerWire(), face.innerWires(), cq.Vector(0, 0, float(height)))
    if z:
        solid = solid.translate(cq.Vector(0, 0, float(z)))
    return solid


def ring_from_polygon(poly: Polygon, wall: float, height: float, z: float = 0.0):
    poly = poly.buffer(0)
    inner = poly.buffer(-float(wall), join_style=1)
    if inner.is_empty:
        raise ValueError("Wall thickness is too large for this outline.")
    ring_geom = poly.difference(inner)
    pieces = []
    geoms = list(ring_geom.geoms) if isinstance(ring_geom, (MultiPolygon, GeometryCollection)) else [ring_geom]
    for g in geoms:
        if isinstance(g, Polygon) and g.area > 1e-7:
            pieces.append(extrude_polygon(g, height, z))
    return make_compound(pieces)


def clay_cutter(
    polygons: Sequence[Polygon],
    target_width: float = 35.0,
    target_height: Optional[float] = None,
    depth: float = 12.0,
    wall: float = 1.2,
    edge_height: float = 2.0,
    edge_wall: float = 0.65,
    grip_height: float = 2.0,
    grip_width: float = 1.8,
):
    polys = normalize_polygons(polygons, target_width, target_height)
    parts = []
    for idx, poly in enumerate(polys, start=1):
        main_h = max(0.2, depth - max(0.0, edge_height))
        main = ring_from_polygon(poly, wall, main_h, 0)
        component_parts = [main]
        if edge_height > 0:
            edge = ring_from_polygon(poly, edge_wall, min(edge_height, depth), main_h)
            component_parts.append(edge)
        if grip_height > 0 and grip_width > 0:
            # A wider top/grip ring at the bottom of the cutter.
            widened = poly.buffer(grip_width, join_style=1)
            inner = poly.buffer(-wall, join_style=1)
            grip_geom = widened.difference(inner)
            geoms = list(grip_geom.geoms) if isinstance(grip_geom, MultiPolygon) else [grip_geom]
            grip_solids = [extrude_polygon(g, min(grip_height, depth), 0) for g in geoms if isinstance(g, Polygon) and g.area > 1e-7]
            component_parts.extend(grip_solids)
        comp = make_compound(component_parts)
        parts.append(comp)
    return make_compound(parts), {f"Cutter_{i+1}": p for i, p in enumerate(parts)}


def text_solid(text: str, size: float, height: float, font: str = "DejaVu Sans", bold: bool = False):
    if not text:
        raise ValueError("Text cannot be empty.")
    kind = "bold" if bold else "regular"
    wp = cq.Workplane("XY").text(text, fontsize=float(size), distance=float(height), font=font, kind=kind, halign="center", valign="center")
    return as_shape(wp)


def birth_plaque(
    width: float = 180.0,
    height: float = 180.0,
    thickness: float = 5.0,
    arch_radius: float = 70.0,
    base_width: float = 185.0,
    base_depth: float = 55.0,
    base_height: float = 12.0,
    slot_clearance: float = 0.35,
    name: str = "NOAH",
    name_size: float = 20.0,
    name_raise: float = 1.0,
):
    """Simple arched birth/sign plaque with separate base and raised name.

    This is deliberately template-driven, intended as a fast starting point rather than a sculpting system.
    """
    w, h, t = map(float, (width, height, thickness))
    r = min(float(arch_radius), w / 2)
    # Build a rectangle plus semicircle top, all in XY then extrude Z.
    rect_h = max(1.0, h - r)
    rect = cq.Workplane("XY").rect(w, rect_h).extrude(t).translate((0, -r / 2, 0))
    circle = cq.Workplane("XY").circle(r).extrude(t).translate((0, rect_h / 2 - r / 2, 0))
    # Clip circle to top half by intersecting with a large box.
    circle_shape = as_shape(circle)
    clip = cq.Solid.makeBox(w + 4, r + 4, t + 2, cq.Vector(-w / 2 - 2, rect_h / 2 - r / 2, -1))
    top = circle_shape.intersect(clip)
    plaque = as_shape(rect).fuse(top)

    # Raised name near upper region.
    name_shape = None
    if name and name_raise > 0:
        try:
            name_shape = text_solid(name, name_size, name_raise)
            name_shape = name_shape.translate(cq.Vector(0, h * 0.20, t))
        except Exception:
            name_shape = None

    # Base with slot cut to accept plaque.
    base = rounded_prism(base_width, base_depth, base_height, radius=min(6.0, base_depth / 4))
    slot_w = w * 0.56
    slot_d = t + slot_clearance
    slot = cq.Solid.makeBox(slot_w, slot_d, base_height + 2, cq.Vector(-slot_w / 2, -slot_d / 2, -1))
    base = base.cut(slot)

    parts = {"Plaque": plaque, "Base": base}
    display_parts = [plaque.translate(cq.Vector(0, 0, base_height)), base]
    if name_shape is not None:
        parts["Name"] = name_shape
        display_parts.append(name_shape.translate(cq.Vector(0, 0, base_height)))
    return make_compound(display_parts), parts


def fuse_shapes(shapes: Sequence):
    vals = [as_shape(s) for s in shapes if s is not None]
    if not vals:
        return None
    result = vals[0]
    for s in vals[1:]:
        result = result.fuse(s)
    return result


def split_with_pins(
    shape,
    axis: str = "X",
    position: float = 0.0,
    connector_count: int = 2,
    pin_diameter: float = 5.0,
    pin_length: float = 6.0,
    clearance: float = 0.2,
    edge_margin: float = 10.0,
):
    """Split a shape into two parts and optionally add matching pin/socket connectors.

    Works best for plate/box-like generated solids. The split is a true solid intersection.
    """
    s = as_shape(shape)
    bb = s.BoundingBox()
    margin = max(bb.xlen, bb.ylen, bb.zlen, 50) * 2
    axis = axis.upper()
    pos = float(position)

    if axis == "X":
        left_len = pos - bb.xmin + margin
        right_len = bb.xmax - pos + margin
        left_box = cq.Solid.makeBox(left_len, bb.ylen + 2 * margin, bb.zlen + 2 * margin,
                                    cq.Vector(bb.xmin - margin, bb.ymin - margin, bb.zmin - margin))
        right_box = cq.Solid.makeBox(right_len, bb.ylen + 2 * margin, bb.zlen + 2 * margin,
                                     cq.Vector(pos, bb.ymin - margin, bb.zmin - margin))
        a = s.intersect(left_box)
        b = s.intersect(right_box)
        span_min, span_max = bb.ymin + edge_margin, bb.ymax - edge_margin
        fixed = (bb.zmin + bb.zmax) / 2
        direction = cq.Vector(1, 0, 0)
        positions = []
        n = max(0, int(connector_count))
        if n == 1:
            positions = [(0.0, fixed)]
        elif n > 1 and span_max > span_min:
            positions = [(span_min + i * (span_max - span_min) / (n - 1), fixed) for i in range(n)]
        for y, z in positions:
            pin = cq.Solid.makeCylinder(pin_diameter / 2, pin_length, cq.Vector(pos - 0.01, y, z), direction)
            socket = cq.Solid.makeCylinder(pin_diameter / 2 + clearance, pin_length + clearance + 0.8, cq.Vector(pos - 0.2, y, z), direction)
            a = a.fuse(pin)
            b = b.cut(socket)
    elif axis == "Y":
        lower_len = pos - bb.ymin + margin
        upper_len = bb.ymax - pos + margin
        box_a = cq.Solid.makeBox(bb.xlen + 2 * margin, lower_len, bb.zlen + 2 * margin,
                                cq.Vector(bb.xmin - margin, bb.ymin - margin, bb.zmin - margin))
        box_b = cq.Solid.makeBox(bb.xlen + 2 * margin, upper_len, bb.zlen + 2 * margin,
                                cq.Vector(bb.xmin - margin, pos, bb.zmin - margin))
        a = s.intersect(box_a)
        b = s.intersect(box_b)
        span_min, span_max = bb.xmin + edge_margin, bb.xmax - edge_margin
        fixed = (bb.zmin + bb.zmax) / 2
        direction = cq.Vector(0, 1, 0)
        n = max(0, int(connector_count))
        xs = [0.0] if n == 1 else ([span_min + i * (span_max - span_min) / (n - 1) for i in range(n)] if n > 1 and span_max > span_min else [])
        for x in xs:
            pin = cq.Solid.makeCylinder(pin_diameter / 2, pin_length, cq.Vector(x, pos - 0.01, fixed), direction)
            socket = cq.Solid.makeCylinder(pin_diameter / 2 + clearance, pin_length + clearance + 0.8, cq.Vector(x, pos - 0.2, fixed), direction)
            a = a.fuse(pin)
            b = b.cut(socket)
    elif axis == "Z":
        bottom_len = pos - bb.zmin + margin
        top_len = bb.zmax - pos + margin
        box_a = cq.Solid.makeBox(bb.xlen + 2 * margin, bb.ylen + 2 * margin, bottom_len,
                                cq.Vector(bb.xmin - margin, bb.ymin - margin, bb.zmin - margin))
        box_b = cq.Solid.makeBox(bb.xlen + 2 * margin, bb.ylen + 2 * margin, top_len,
                                cq.Vector(bb.xmin - margin, bb.ymin - margin, pos))
        a = s.intersect(box_a)
        b = s.intersect(box_b)
        # Place pins along X at center Y.
        span_min, span_max = bb.xmin + edge_margin, bb.xmax - edge_margin
        y = (bb.ymin + bb.ymax) / 2
        direction = cq.Vector(0, 0, 1)
        n = max(0, int(connector_count))
        xs = [0.0] if n == 1 else ([span_min + i * (span_max - span_min) / (n - 1) for i in range(n)] if n > 1 and span_max > span_min else [])
        for x in xs:
            pin = cq.Solid.makeCylinder(pin_diameter / 2, pin_length, cq.Vector(x, y, pos - 0.01), direction)
            socket = cq.Solid.makeCylinder(pin_diameter / 2 + clearance, pin_length + clearance + 0.8, cq.Vector(x, y, pos - 0.2), direction)
            a = a.fuse(pin)
            b = b.cut(socket)
    else:
        raise ValueError("Axis must be X, Y or Z.")

    return a, b


def transform_shape(shape, tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0, scale=1.0):
    s = as_shape(shape)
    if scale != 1.0:
        # CadQuery/OCP general scaling is easiest via transformGeometry.
        m = cq.Matrix([
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0],
        ])
        s = s.transformGeometry(m)
    center = s.Center()
    if rx:
        s = s.rotate(center, center + cq.Vector(1, 0, 0), rx)
    if ry:
        s = s.rotate(center, center + cq.Vector(0, 1, 0), ry)
    if rz:
        s = s.rotate(center, center + cq.Vector(0, 0, 1), rz)
    if tx or ty or tz:
        s = s.translate(cq.Vector(tx, ty, tz))
    return s


def export_bytes(shape, fmt: str = "stl", tolerance: float = 0.02, angular_tolerance: float = 0.1) -> bytes:
    fmt = fmt.lower().lstrip(".")
    if fmt not in {"stl", "step", "stp", "3mf"}:
        raise ValueError("Supported exports: STL, STEP, 3MF.")
    suffix = ".step" if fmt in {"step", "stp"} else f".{fmt}"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model" + suffix)
        obj = as_shape(shape)
        if fmt == "stl":
            cq.exporters.export(obj, p, tolerance=float(tolerance), angularTolerance=float(angular_tolerance))
        else:
            cq.exporters.export(obj, p)
        with open(p, "rb") as f:
            return f.read()


def shape_stats(shape, tessellation_tolerance: float = 0.2):
    s = as_shape(shape)
    bb = s.BoundingBox()
    verts, tris = s.tessellate(float(tessellation_tolerance), 0.15)
    volume = None
    try:
        volume = float(s.Volume())
    except Exception:
        pass
    return {
        "x_mm": bb.xlen,
        "y_mm": bb.ylen,
        "z_mm": bb.zlen,
        "volume_mm3": volume,
        "vertices": len(verts),
        "triangles": len(tris),
    }
