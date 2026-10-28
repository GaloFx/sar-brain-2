"""Plotly 3D brain graph visualization helpers."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pandas as pd


def edge_xyz(ii, jj, positions):
    xx, yy, zz = [], [], []
    for a, b in zip(ii, jj):
        xx.extend([positions[a, 0], positions[b, 0], None])
        yy.extend([positions[a, 1], positions[b, 1], None])
        zz.extend([positions[a, 2], positions[b, 2], None])
    return xx, yy, zz


def make_functional_trace(C, threshold_count, positions, opacity=0.75):
    iu, ju = np.triu_indices_from(C, k=1)
    vals = C[iu, ju]
    n_keep = min(threshold_count, len(vals))
    valid = np.argsort(np.abs(vals))[-n_keep:]
    ii, jj = iu[valid], ju[valid]
    xx, yy, zz = edge_xyz(ii, jj, positions)
    return go.Scatter3d(
        x=xx,
        y=yy,
        z=zz,
        mode="lines",
        line=dict(color=f"rgba(230,50,40,{opacity})", width=3.5),
        name="Conectividad funcional",
        hoverinfo="skip",
    )


def build_brain_figure(
    state: dict,
    frame_idx: int = 0,
    color_by: str = "activity",
    n_functional_edges: int = 80,
    show_surface: bool = True,
    show_spatial: bool = True,
    show_functional: bool = True,
    height: int = 680,
):
    """Build interactive Plotly 3D figure from pipeline state."""
    coords = state["coords"]
    coords_hemi = state["coords_hemi"]
    faces_hemi = state["faces_hemi"]
    y = state["y"]
    cortical_thickness = state["cortical_thickness"]
    activity_frames = state["activity_frames"]
    corr_frames = state["corr_frames"]
    frame_times = state["frame_times"]
    nodes_df = state["nodes_df"]
    sp_i, sp_j = state["sp_i"], state["sp_j"]

    fig = go.Figure()

    if show_surface:
        for hemi in ("left", "right"):
            xyz = coords_hemi[hemi]
            faces = faces_hemi[hemi][::4]  # subsample for performance
            fig.add_trace(
                go.Mesh3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color="#E2E7EC",
                    opacity=0.25,
                    flatshading=False,
                    name=f"Superficie {hemi}",
                    hoverinfo="skip",
                    lighting=dict(ambient=0.6, diffuse=0.5, specular=0.15),
                )
            )

    if show_spatial:
        sx, sy, sz = edge_xyz(sp_i, sp_j, coords)
        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="lines",
                line=dict(color="rgba(70,100,140,0.22)", width=1.5),
                name="Capa espacial",
                hoverinfo="skip",
            )
        )

    if color_by == "DD":
        node_color = y
        cmin, cmax = float(np.percentile(y, 2)), float(np.percentile(y, 98))
        cbar_title = "DD (HCP-scaled)"
    elif color_by == "thickness":
        node_color = cortical_thickness
        cmin, cmax = 1.8, 3.2
        cbar_title = "Grosor (mm)"
    else:
        node_color = activity_frames[frame_idx]
        cmin, cmax = -1.0, 1.0
        cbar_title = "Actividad"

    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=6.5,
                color=node_color,
                colorscale="RdYlBu_r",
                cmin=cmin,
                cmax=cmax,
                opacity=0.95,
                colorbar=dict(
                    title=cbar_title,
                    x=1.02,
                    len=0.65,
                    titlefont=dict(color="#222222"),
                    tickfont=dict(color="#222222"),
                ),
            ),
            text=[
                f"Nodo {r.node} | {r.hemisphere}<br>Parcela {r.parcel_id}<br>"
                f"DD={y[r.node]:.1f} | Grosor={cortical_thickness[r.node]:.2f}"
                for r in nodes_df.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Regiones",
        )
    )

    if show_functional:
        fig.add_trace(
            make_functional_trace(
                corr_frames[frame_idx], n_functional_edges, coords
            )
        )

    t_val = int(frame_times[frame_idx])
    fig.update_layout(
        title=dict(
            text=f"Grafo cerebral 3D | t={t_val} | color={color_by}",
            font=dict(color="#222222"),
        ),
        template="plotly_white",
        height=height,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            backgroundcolor="#FFFFFF",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.05)),
        ),
        legend=dict(x=0.01, y=0.99, font=dict(color="#222222")),
        showlegend=True,
    )
    return fig
