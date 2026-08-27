from __future__ import annotations

from io import BytesIO
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon


def _ensure_gray(data: bytes) -> np.ndarray:
    im = Image.open(BytesIO(data)).convert("L")
    return np.array(im)


def trace_bitmap(
    data: bytes,
    threshold: int = 160,
    invert: bool = False,
    blur: int = 0,
    simplify: float = 0.25,
    min_area: float = 20.0,
    external_only: bool = True,
):
    """Trace a PNG/JPG into closed contours.

    Returns (contours, preview_binary, image_size), where contours is a list of Nx2 float arrays.
    The default assumes a dark design on a light background.
    """
    gray = _ensure_gray(data)
    if blur and blur > 0:
        k = int(blur)
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    mode = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(gray, int(threshold), 255, mode)
    retrieval = cv2.RETR_EXTERNAL if external_only else cv2.RETR_CCOMP
    contours, hierarchy = cv2.findContours(binary, retrieval, cv2.CHAIN_APPROX_NONE)

    out = []
    for c in contours:
        area = abs(cv2.contourArea(c))
        if area < float(min_area):
            continue
        peri = cv2.arcLength(c, True)
        eps = max(0.01, float(simplify) / 100.0 * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = approx[:, 0, :].astype(float)
        if len(pts) >= 3:
            out.append(pts)

    out.sort(key=lambda p: abs(cv2.contourArea(p.astype(np.float32))), reverse=True)
    h, w = gray.shape[:2]
    return out, binary, (w, h)


def contours_to_polygons(contours: Sequence[np.ndarray], image_size: Tuple[int, int], max_parts: int = 24) -> List[Polygon]:
    w, h = image_size
    polys: List[Polygon] = []
    for pts in list(contours)[: int(max_parts)]:
        # Flip Y so the geometry is conventional Cartesian coordinates.
        xy = [(float(x), float(h - y)) for x, y in pts]
        p = Polygon(xy).buffer(0)
        if p.is_empty:
            continue
        if p.geom_type == "Polygon" and p.area > 1e-6:
            polys.append(p)
        elif p.geom_type == "MultiPolygon":
            polys.extend([g for g in p.geoms if g.area > 1e-6])
    return polys


def contours_to_svg(
    contours: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    fill: str = "#000000",
) -> str:
    w, h = image_size
    paths = []
    for pts in contours:
        if len(pts) < 3:
            continue
        d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
        d.extend(f"L {x:.2f} {y:.2f}" for x, y in pts[1:])
        d.append("Z")
        paths.append(f'<path d="{" ".join(d)}" fill="{fill}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">' + "".join(paths) + "</svg>"
    )


def binary_preview_png(binary: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", binary)
    if not ok:
        raise RuntimeError("Could not encode preview image.")
    return encoded.tobytes()


def rasterize_svg(svg_bytes: bytes, output_width: int = 1800) -> bytes:
    """Rasterize arbitrary SVG for robust tracing.

    We intentionally rasterize at high resolution, then convert the result back to vector contours.
    This supports most SVG transforms/path types without a bespoke SVG parser while preserving
    more than enough outline resolution for FDM-sized designs.
    """
    import cairosvg

    return cairosvg.svg2png(bytestring=svg_bytes, output_width=int(output_width))
