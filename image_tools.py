from __future__ import annotations

from io import BytesIO
from typing import List, Sequence, Tuple, Optional
import math
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


# ---------------- Raster / PNG tracing ----------------
def _ensure_gray(data: bytes) -> np.ndarray:
    # Composite transparency over white so transparent pixels are not traced as a black box.
    im = Image.open(BytesIO(data)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.array(bg.convert("L"))


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
    """Rasterize SVG for visual preview only. Direct SVG->CAD does not use raster tracing."""
    import cairosvg
    return cairosvg.svg2png(bytestring=svg_bytes, output_width=int(output_width), background_color="#ffffff")


# ---------------- Direct SVG vector import ----------------
# This parser intentionally converts SVG vectors directly to polygons instead of rasterising.
# It supports common SVG primitives, paths, nested transforms and Bezier/arc commands.

_num_re = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_token_re = re.compile(rf"[AaCcHhLlMmQqSsTtVvZz]|{_num_re}")


def _mat_mul(a, b):
    # SVG affine matrices (a,b,c,d,e,f): x'=a*x+c*y+e, y'=b*x+d*y+f
    a1,b1,c1,d1,e1,f1 = a; a2,b2,c2,d2,e2,f2 = b
    return (
        a1*a2 + c1*b2,
        b1*a2 + d1*b2,
        a1*c2 + c1*d2,
        b1*c2 + d1*d2,
        a1*e2 + c1*f2 + e1,
        b1*e2 + d1*f2 + f1,
    )


def _apply_mat(p, m):
    x,y = p; a,b,c,d,e,f = m
    return (a*x + c*y + e, b*x + d*y + f)


def _parse_transform(s: Optional[str]):
    m = (1.0,0.0,0.0,1.0,0.0,0.0)
    if not s:
        return m
    for name, args_s in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", s):
        vals = [float(x) for x in re.findall(_num_re, args_s)]
        name = name.lower()
        t = (1.0,0.0,0.0,1.0,0.0,0.0)
        if name == "matrix" and len(vals) >= 6:
            t = tuple(vals[:6])
        elif name == "translate" and vals:
            t = (1,0,0,1,vals[0], vals[1] if len(vals)>1 else 0)
        elif name == "scale" and vals:
            sx=vals[0]; sy=vals[1] if len(vals)>1 else sx
            t = (sx,0,0,sy,0,0)
        elif name == "rotate" and vals:
            ang=math.radians(vals[0]); ca,sa=math.cos(ang),math.sin(ang)
            r=(ca,sa,-sa,ca,0,0)
            if len(vals)>=3:
                cx,cy=vals[1],vals[2]
                t=_mat_mul(_mat_mul((1,0,0,1,cx,cy),r),(1,0,0,1,-cx,-cy))
            else: t=r
        elif name == "skewx" and vals:
            t=(1,0,math.tan(math.radians(vals[0])),1,0,0)
        elif name == "skewy" and vals:
            t=(1,math.tan(math.radians(vals[0])),0,1,0,0)
        # SVG applies transform functions in listed order.
        m = _mat_mul(m, t)
    return m


def _dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _sample_cubic(p0,p1,p2,p3,tol):
    est=_dist(p0,p1)+_dist(p1,p2)+_dist(p2,p3)
    n=max(4,min(1024,int(math.ceil(est/max(tol,1e-4)))))
    out=[]
    for i in range(1,n+1):
        t=i/n; u=1-t
        out.append((u**3*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t**3*p3[0],
                    u**3*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t**3*p3[1]))
    return out


def _sample_quad(p0,p1,p2,tol):
    est=_dist(p0,p1)+_dist(p1,p2)
    n=max(3,min(1024,int(math.ceil(est/max(tol,1e-4)))))
    out=[]
    for i in range(1,n+1):
        t=i/n; u=1-t
        out.append((u*u*p0[0]+2*u*t*p1[0]+t*t*p2[0], u*u*p0[1]+2*u*t*p1[1]+t*t*p2[1]))
    return out


def _vector_angle(ux,uy,vx,vy):
    dot=ux*vx+uy*vy
    l=max(1e-15, math.hypot(ux,uy)*math.hypot(vx,vy))
    ang=math.acos(max(-1,min(1,dot/l)))
    if ux*vy-uy*vx < 0: ang=-ang
    return ang


def _sample_arc(p0, rx, ry, phi_deg, large, sweep, p1, tol):
    rx,ry=abs(rx),abs(ry)
    if rx < 1e-12 or ry < 1e-12 or _dist(p0,p1)<1e-12:
        return [p1]
    phi=math.radians(phi_deg%360); cp,sp=math.cos(phi),math.sin(phi)
    dx=(p0[0]-p1[0])/2; dy=(p0[1]-p1[1])/2
    x1p=cp*dx+sp*dy; y1p=-sp*dx+cp*dy
    lam=(x1p*x1p)/(rx*rx)+(y1p*y1p)/(ry*ry)
    if lam>1:
        s=math.sqrt(lam); rx*=s; ry*=s
    num=max(0.0, rx*rx*ry*ry-rx*rx*y1p*y1p-ry*ry*x1p*x1p)
    den=max(1e-30, rx*rx*y1p*y1p+ry*ry*x1p*x1p)
    coef=math.sqrt(num/den)
    if bool(large)==bool(sweep): coef=-coef
    cxp=coef*(rx*y1p/ry); cyp=coef*(-ry*x1p/rx)
    cx=cp*cxp-sp*cyp+(p0[0]+p1[0])/2
    cy=sp*cxp+cp*cyp+(p0[1]+p1[1])/2
    ux=(x1p-cxp)/rx; uy=(y1p-cyp)/ry
    vx=(-x1p-cxp)/rx; vy=(-y1p-cyp)/ry
    theta1=_vector_angle(1,0,ux,uy)
    dtheta=_vector_angle(ux,uy,vx,vy)
    if not sweep and dtheta>0: dtheta-=2*math.pi
    elif sweep and dtheta<0: dtheta+=2*math.pi
    arc_len=max(rx,ry)*abs(dtheta)
    n=max(4,min(2048,int(math.ceil(arc_len/max(tol,1e-4)))))
    out=[]
    for i in range(1,n+1):
        th=theta1+dtheta*(i/n)
        x=cx+cp*rx*math.cos(th)-sp*ry*math.sin(th)
        y=cy+sp*rx*math.cos(th)+cp*ry*math.sin(th)
        out.append((x,y))
    return out


def _path_loops(d: str, tol: float):
    toks=_token_re.findall(d or "")
    i=0; cmd=None; cur=(0.0,0.0); start=(0.0,0.0); prev_ctrl=None
    loops=[]; pts=[]
    nargs={"M":2,"L":2,"H":1,"V":1,"C":6,"S":4,"Q":4,"T":2,"A":7,"Z":0}
    while i < len(toks):
        if re.match(r"^[A-Za-z]$", toks[i]):
            cmd=toks[i]; i+=1
        if cmd is None: break
        up=cmd.upper(); rel=cmd.islower()
        if up=="Z":
            if pts:
                if _dist(pts[-1],start)>1e-9: pts.append(start)
                if len(pts)>=4: loops.append(pts)
            pts=[]; cur=start; prev_ctrl=None; cmd=None; continue
        need=nargs[up]
        if i+need>len(toks): break
        vals=list(map(float,toks[i:i+need])); i+=need
        def xy(x,y): return (cur[0]+x,cur[1]+y) if rel else (x,y)
        if up=="M":
            p=xy(vals[0],vals[1]); cur=p; start=p
            if pts and len(pts)>=3: loops.append(pts)
            pts=[p]; prev_ctrl=None
            cmd="l" if rel else "L"
        elif up=="L":
            p=xy(vals[0],vals[1]); pts.append(p); cur=p; prev_ctrl=None
        elif up=="H":
            p=(cur[0]+vals[0],cur[1]) if rel else (vals[0],cur[1]); pts.append(p); cur=p; prev_ctrl=None
        elif up=="V":
            p=(cur[0],cur[1]+vals[0]) if rel else (cur[0],vals[0]); pts.append(p); cur=p; prev_ctrl=None
        elif up=="C":
            p1=xy(vals[0],vals[1]); p2=xy(vals[2],vals[3]); p3=xy(vals[4],vals[5])
            pts.extend(_sample_cubic(cur,p1,p2,p3,tol)); cur=p3; prev_ctrl=p2
        elif up=="S":
            p1=(2*cur[0]-prev_ctrl[0],2*cur[1]-prev_ctrl[1]) if prev_ctrl else cur
            p2=xy(vals[0],vals[1]); p3=xy(vals[2],vals[3])
            pts.extend(_sample_cubic(cur,p1,p2,p3,tol)); cur=p3; prev_ctrl=p2
        elif up=="Q":
            p1=xy(vals[0],vals[1]); p2=xy(vals[2],vals[3])
            pts.extend(_sample_quad(cur,p1,p2,tol)); cur=p2; prev_ctrl=p1
        elif up=="T":
            p1=(2*cur[0]-prev_ctrl[0],2*cur[1]-prev_ctrl[1]) if prev_ctrl else cur
            p2=xy(vals[0],vals[1]); pts.extend(_sample_quad(cur,p1,p2,tol)); cur=p2; prev_ctrl=p1
        elif up=="A":
            p=xy(vals[5],vals[6]); pts.extend(_sample_arc(cur,vals[0],vals[1],vals[2],int(vals[3]),int(vals[4]),p,tol)); cur=p; prev_ctrl=None
        # Commands can repeat without being restated; M handled above turns into L.
    # Only closed geometry is printable. For an explicitly closed-looking final loop, accept it.
    if pts and len(pts)>=4 and _dist(pts[0],pts[-1]) <= max(tol,1e-6)*2:
        pts[-1]=pts[0]; loops.append(pts)
    return loops


def _parse_points(s: str):
    v=[float(x) for x in re.findall(_num_re,s or "")]
    return list(zip(v[0::2],v[1::2]))


def _primitive_loops(tag, el, tol):
    a=el.attrib
    def f(k,default=0):
        try:return float(re.findall(_num_re,a.get(k,str(default)))[0])
        except:return float(default)
    if tag=="path": return _path_loops(a.get("d",""),tol)
    if tag=="polygon":
        p=_parse_points(a.get("points",""));
        if len(p)>=3: return [p+[p[0]]]
    if tag=="polyline":
        p=_parse_points(a.get("points",""));
        if len(p)>=3 and _dist(p[0],p[-1])<tol*2: return [p[:-1]+[p[0]]]
    if tag=="rect":
        x,y,w,h=f("x"),f("y"),f("width"),f("height"); rx=f("rx",0); ry=f("ry",rx)
        if w<=0 or h<=0:return []
        rx=min(max(rx,0),w/2); ry=min(max(ry,0),h/2)
        if rx<=0 and ry<=0:return [[(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)]]
        # approximate rounded rectangle with quarter-ellipses
        n=max(4,int(math.ceil((math.pi/2)*max(rx,ry)/max(tol,1e-4))))
        pts=[]
        for cx,cy,a0 in [(x+w-rx,y+ry,-math.pi/2),(x+w-rx,y+h-ry,0),(x+rx,y+h-ry,math.pi/2),(x+rx,y+ry,math.pi)]:
            for j in range(n+1):
                t=a0+(math.pi/2)*(j/n); pts.append((cx+rx*math.cos(t),cy+ry*math.sin(t)))
        pts.append(pts[0]); return [pts]
    if tag in ("circle","ellipse"):
        cx,cy=f("cx"),f("cy"); rx=f("r") if tag=="circle" else f("rx"); ry=rx if tag=="circle" else f("ry")
        if rx<=0 or ry<=0:return []
        n=max(24,min(4096,int(math.ceil(2*math.pi*max(rx,ry)/max(tol,1e-4)))))
        p=[(cx+rx*math.cos(2*math.pi*i/n),cy+ry*math.sin(2*math.pi*i/n)) for i in range(n)]
        return [p+[p[0]]]
    return []


def _style_hidden(el):
    style=(el.attrib.get("style","")+";"+";".join(f"{k}:{v}" for k,v in el.attrib.items() if k in ("display","visibility","fill","fill-opacity","opacity"))).lower()
    if "display:none" in style or "visibility:hidden" in style: return True
    if re.search(r"(?:^|;)\s*(?:opacity|fill-opacity)\s*:\s*0(?:\.0*)?(?:;|$)", style): return True
    if re.search(r"(?:^|;)\s*fill\s*:\s*none(?:;|$)", style): return True
    return False


def _rings_to_filled_polygons(rings: List[List[Tuple[float,float]]], flip_y=True):
    candidates=[]
    for r in rings:
        if len(r)<4: continue
        rr=[(x,-y if flip_y else y) for x,y in r]
        p=Polygon(rr).buffer(0)
        if p.is_empty: continue
        geoms=list(p.geoms) if isinstance(p,MultiPolygon) else [p]
        candidates.extend([g for g in geoms if isinstance(g,Polygon) and g.area>1e-9])
    if not candidates:return []
    # Even-odd nesting: shells at even depth, holes at odd depth.
    ordered=sorted(candidates,key=lambda p:p.area,reverse=True)
    depth=[]
    for i,p in enumerate(ordered):
        rp=p.representative_point(); d=0
        for q in ordered[:i]:
            if q.contains(rp): d+=1
        depth.append(d)
    shells=[]
    for i,p in enumerate(ordered):
        if depth[i]%2==0:
            holes=[]
            for j,h in enumerate(ordered):
                if depth[j]==depth[i]+1 and p.contains(h.representative_point()):
                    # ensure immediate parent is p
                    containing=[q for k,q in enumerate(ordered) if depth[k]==depth[i] and q.contains(h.representative_point())]
                    if containing and min(containing,key=lambda q:q.area).equals(p): holes.append(list(h.exterior.coords))
            shells.append(Polygon(p.exterior.coords, holes).buffer(0))
    merged=unary_union(shells).buffer(0)
    geoms=list(merged.geoms) if isinstance(merged,MultiPolygon) else [merged]
    return [g for g in geoms if isinstance(g,Polygon) and g.area>1e-8]


def svg_to_polygons(svg_bytes: bytes, curve_tolerance: float = 0.15, max_parts: int = 64):
    """Convert SVG vectors directly to Shapely polygons without raster tracing.

    curve_tolerance is in SVG coordinate units. Smaller values preserve curves with more points.
    Supports path M/L/H/V/C/S/Q/T/A/Z, rect, circle, ellipse, polygon and nested transforms.
    Returns (polygons, metadata).
    """
    root=ET.fromstring(svg_bytes)
    rings=[]

    def walk(el,parent_m=(1,0,0,1,0,0),hidden=False):
        tag=el.tag.split("}")[-1].lower()
        hidden=hidden or _style_hidden(el)
        local=_parse_transform(el.attrib.get("transform"))
        m=_mat_mul(parent_m,local)
        if not hidden and tag in {"path","rect","circle","ellipse","polygon","polyline"}:
            for loop in _primitive_loops(tag,el,float(curve_tolerance)):
                rings.append([_apply_mat(p,m) for p in loop])
        for ch in list(el): walk(ch,m,hidden)
    walk(root)
    polys=_rings_to_filled_polygons(rings,flip_y=True)
    polys=sorted(polys,key=lambda p:p.area,reverse=True)[:int(max_parts)]
    if not polys:
        raise ValueError("No closed filled vector regions were found in this SVG.")
    union=unary_union(polys)
    minx,miny,maxx,maxy=union.bounds
    meta={
        "regions":len(polys),"width_units":maxx-minx,"height_units":maxy-miny,
        "viewBox":root.attrib.get("viewBox"),"source":"direct_svg_vector",
        "curve_tolerance":float(curve_tolerance),"rings":len(rings),
    }
    return polys,meta


def polygons_to_svg(polygons: Sequence[Polygon], padding: float = 0.0) -> str:
    """Export polygon geometry as an SVG using even-odd fill, mainly for inspection/download."""
    ps=[p for p in polygons if p is not None and not p.is_empty]
    if not ps:return '<svg xmlns="http://www.w3.org/2000/svg"/>'
    u=unary_union(ps); minx,miny,maxx,maxy=u.bounds
    w=maxx-minx; h=maxy-miny; pad=float(padding)
    def ring_d(coords):
        c=list(coords); return "M "+" L ".join(f"{x-minx+pad:.6f} {maxy-y+pad:.6f}" for x,y in c)+" Z"
    ds=[]
    for p in ps:
        ds.append(ring_d(p.exterior.coords))
        ds.extend(ring_d(r.coords) for r in p.interiors)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w+2*pad:.6f} {h+2*pad:.6f}"><path d="{" ".join(ds)}" fill="#000" fill-rule="evenodd"/></svg>'

# ---------------- Detailed colour-layer image tracing ----------------
def _rgba_over_white(data: bytes):
    im = Image.open(BytesIO(data)).convert('RGBA')
    arr = np.array(im)
    alpha = arr[..., 3]
    rgb = arr[..., :3].astype(np.float32)
    a = alpha[..., None].astype(np.float32) / 255.0
    comp = rgb * a + 255.0 * (1.0 - a)
    return comp.astype(np.uint8), alpha


def _cluster_color_hex(bgr_or_rgb):
    r, g, b = [int(round(float(x))) for x in bgr_or_rgb]
    return f'#{r:02x}{g:02x}{b:02x}'


def _mask_to_polygons(mask: np.ndarray, simplify: float = 0.15, min_area: float = 20.0):
    """Convert one binary colour mask to polygons while preserving direct child holes."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    h_img = mask.shape[0]
    result = []
    for i, c in enumerate(contours):
        # parent == -1 => exterior contour
        if hierarchy[i][3] != -1:
            continue
        if abs(cv2.contourArea(c)) < float(min_area):
            continue
        peri = cv2.arcLength(c, True)
        eps = max(0.02, float(simplify) / 100.0 * peri)
        outer = cv2.approxPolyDP(c, eps, True)[:, 0, :]
        if len(outer) < 3:
            continue
        shell = [(float(x), float(h_img-y)) for x, y in outer]
        holes = []
        child = hierarchy[i][2]
        while child != -1:
            hc = contours[child]
            if abs(cv2.contourArea(hc)) >= float(min_area):
                hp = cv2.approxPolyDP(hc, max(0.02, float(simplify)/100.0*cv2.arcLength(hc, True)), True)[:,0,:]
                if len(hp) >= 3:
                    holes.append([(float(x), float(h_img-y)) for x,y in hp])
            child = hierarchy[child][0]
        p = Polygon(shell, holes).buffer(0)
        if p.is_empty:
            continue
        if isinstance(p, MultiPolygon):
            result.extend([g for g in p.geoms if g.area >= float(min_area)])
        elif p.area >= float(min_area):
            result.append(p)
    return result


def trace_color_layers(
    data: bytes,
    colors: int = 6,
    simplify: float = 0.15,
    min_area: float = 30.0,
    background_tolerance: float = 24.0,
    max_parts: int = 64,
):
    """Trace a raster image into separate filled colour regions.

    Background is estimated from the image border and automatically excluded. Transparent
    pixels are always background. Returns (layers, preview_png, svg, metadata).
    Each layer is {'name','color','polygons','area'}.
    """
    rgb, alpha = _rgba_over_white(data)
    oh, ow = rgb.shape[:2]
    scale = min(1.0, 900.0 / max(oh, ow))
    if scale < 1.0:
        rgb_small = cv2.resize(rgb, (max(2,int(ow*scale)), max(2,int(oh*scale))), interpolation=cv2.INTER_AREA)
        alpha_small = cv2.resize(alpha, (rgb_small.shape[1], rgb_small.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        rgb_small, alpha_small = rgb, alpha
    h, w = rgb_small.shape[:2]

    # Lab gives more perceptual colour grouping than raw RGB.
    lab = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape((-1,3)).astype(np.float32)
    k = int(max(2, min(int(colors), 12, len(pixels))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.8)
    _compact, labels, centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape((h,w))

    # Border pixels define likely background. Use majority cluster among opaque-ish border.
    border_mask = np.zeros((h,w), np.uint8)
    bw = max(1, int(round(min(h,w)*0.035)))
    border_mask[:bw,:]=1; border_mask[-bw:,:]=1; border_mask[:,:bw]=1; border_mask[:,-bw:]=1
    opaque_border = (border_mask.astype(bool) & (alpha_small > 20))
    border_labels = labels[opaque_border]
    if border_labels.size:
        counts = np.bincount(border_labels, minlength=k)
        bg_cluster = int(np.argmax(counts))
    else:
        bg_cluster = 0

    # Also identify clusters visually close to the dominant border colour as background.
    bg_center = centers[bg_cluster]
    bg_clusters = {bg_cluster}
    for ci, cen in enumerate(centers):
        if float(np.linalg.norm(cen-bg_center)) <= float(background_tolerance):
            bg_clusters.add(ci)

    # Convert center colours back to RGB for layer display.
    center_lab_u8 = np.clip(centers,0,255).astype(np.uint8).reshape((-1,1,3))
    center_rgb = cv2.cvtColor(center_lab_u8, cv2.COLOR_LAB2RGB).reshape((-1,3))

    layers = []
    preview = np.full((h,w,3), 255, np.uint8)
    kernel = np.ones((3,3), np.uint8)
    for ci in range(k):
        if ci in bg_clusters:
            continue
        mask = ((labels == ci) & (alpha_small > 20)).astype(np.uint8)*255
        # Remove isolated k-means speckle, then close 1-pixel gaps.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        if cv2.countNonZero(mask) < min_area:
            continue
        polys = _mask_to_polygons(mask, simplify=simplify, min_area=min_area)
        if not polys:
            continue
        # Scale polygons back to original image coordinate scale.
        invs = 1.0/scale
        if abs(invs-1.0) > 1e-9:
            from shapely.affinity import scale as shp_scale
            polys = [shp_scale(p, xfact=invs, yfact=invs, origin=(0,0)) for p in polys]
        col = tuple(int(x) for x in center_rgb[ci])
        area = sum(p.area for p in polys)
        layers.append({'name':f'Layer {len(layers)+1}', 'color':_cluster_color_hex(col), 'rgb':col, 'polygons':polys, 'area':area})
        preview[mask>0] = np.array(col, np.uint8)

    layers.sort(key=lambda x:x['area'], reverse=True)
    # Limit individual geometry count globally, keeping largest regions first.
    total = 0
    trimmed=[]
    for layer in layers:
        ps=sorted(layer['polygons'], key=lambda p:p.area, reverse=True)
        keep=ps[:max(0,int(max_parts)-total)]
        if keep:
            nl=dict(layer); nl['polygons']=keep; nl['area']=sum(p.area for p in keep); trimmed.append(nl); total += len(keep)
        if total >= int(max_parts): break
    layers=trimmed

    # SVG preserves each layer colour and its holes.
    all_polys=[p for layer in layers for p in layer['polygons']]
    if not all_polys:
        raise ValueError('No non-background artwork regions were detected. Try fewer colours or reduce background tolerance.')
    union=unary_union(all_polys); minx,miny,maxx,maxy=union.bounds
    vw=maxx-minx; vh=maxy-miny
    def pd(p):
        def ring(coords):
            return 'M '+' L '.join(f'{x-minx:.3f} {maxy-y:.3f}' for x,y in coords)+' Z'
        return ' '.join([ring(p.exterior.coords)] + [ring(r.coords) for r in p.interiors])
    elems=[]
    for li,layer in enumerate(layers):
        for p in layer['polygons']:
            elems.append(f'<path d="{pd(p)}" fill="{layer["color"]}" fill-rule="evenodd" data-layer="{li}"/>')
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:.3f} {vh:.3f}">' + ''.join(elems) + '</svg>'

    ok, enc = cv2.imencode('.png', cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    preview_png = enc.tobytes() if ok else b''
    return layers, preview_png, svg, {
        'source':'detailed_color_layers', 'layers':len(layers), 'regions':sum(len(x['polygons']) for x in layers),
        'background_cluster':bg_cluster, 'background_clusters':sorted(bg_clusters), 'original_size':(ow,oh)
    }
