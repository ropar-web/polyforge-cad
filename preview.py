from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from cad_engine import as_shape


def shape_to_mesh_data(shape, tolerance: float = 0.25):
    s = as_shape(shape)
    verts, tris = s.tessellate(float(tolerance), 0.15)
    xyz = np.array([v.toTuple() for v in verts], dtype=float)
    tri = np.array(tris, dtype=int)
    return xyz, tri


def figure_for_shape(shape, tolerance: float = 0.25, height: int = 620):
    xyz, tri = shape_to_mesh_data(shape, tolerance=tolerance)
    if len(xyz) == 0 or len(tri) == 0:
        return go.Figure()
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                i=tri[:, 0],
                j=tri[:, 1],
                k=tri[:, 2],
                flatshading=False,
                lighting=dict(ambient=0.65, diffuse=0.75, specular=0.08, roughness=0.85),
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="data",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.15)),
        ),
        showlegend=False,
    )
    return fig
