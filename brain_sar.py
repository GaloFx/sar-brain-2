"""
Core logic: cortical graph (Destrieux + fsaverage5), synthetic node features
calibrated to real HCP Delay Discounting distribution, spatial weight matrices,
SAR (ML_Lag) estimation, and dynamic activity simulation.

Audit notes (v1.1 – GitHub / Streamlit ready):
- Removed all Colab / Google Drive dependencies.
- CSV path resolved relative to package / data/ folder.
- Islands in DistanceBand silenced; sparsity reported.
- Explicit synthetic blocks marked for replacement with real HCP/BALSA thickness & FC.
- Reproducible with fixed SEED.
- Geometry is real (nilearn); phenotype distribution is real HCP; thickness/FC remain synthetic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler

from nilearn import datasets, surface
from libpysal.weights import W, KNN, DistanceBand
from spreg import ML_Lag
from esda.moran import Moran

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 20261011
SURFACE_MESH = "fsaverage5"
N_SPATIAL_NEIGHBORS = 5
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_CANDIDATES = [
    DATA_DIR / "Delay_Discounting_40K_1_Month.csv",
    Path("data/Delay_Discounting_40K_1_Month.csv"),
    Path("Delay Discounting 40K 1 Month.csv"),
    Path("/home/workdir/attachments/Delay Discounting 40K 1 Month.csv"),
]


def _find_csv() -> Path | None:
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def load_dd_distribution(rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    """Load real HCP DD vector or fall back to synthetic distribution of similar shape."""
    csv_path = _find_csv()
    if csv_path is None:
        n_sim = 1198
        vals = rng.exponential(scale=15000, size=n_sim)
        vals = np.clip(vals, 100, 40000)
        return vals, "synthetic_fallback"
    df = pd.read_csv(csv_path)
    col = "DDisc_SV_1mo_40K"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Columns: {list(df.columns)}")
    vals = df[col].dropna().astype(float).values
    return vals, str(csv_path)


def build_cortical_nodes() -> Tuple[pd.DataFrame, np.ndarray, dict, dict, dict]:
    """
    Real geometry: fsaverage5 pial + Destrieux atlas.
    Returns nodes_df, coords (N,3), coords_hemi, faces_hemi, labels_hemi.
    """
    fsavg = datasets.fetch_surf_fsaverage(mesh=SURFACE_MESH)
    atlas = datasets.fetch_atlas_surf_destrieux()

    coords_hemi, faces_hemi, labels_hemi = {}, {}, {}
    for hemi in ("left", "right"):
        mesh = surface.load_surf_mesh(fsavg[f"pial_{hemi}"])
        coords_hemi[hemi] = np.asarray(mesh.coordinates, dtype=float)
        faces_hemi[hemi] = np.asarray(mesh.faces, dtype=int)
        labels_hemi[hemi] = np.asarray(atlas[f"map_{hemi}"]).astype(int).ravel()

    nodes = []
    for hemi in ("left", "right"):
        xyz = coords_hemi[hemi]
        lab = labels_hemi[hemi]
        for parcel_id in sorted(np.unique(lab)):
            if parcel_id == 0:  # medial wall
                continue
            idx = np.flatnonzero(lab == parcel_id)
            if len(idx) < 5:
                continue
            centroid = xyz[idx].mean(axis=0)
            nodes.append(
                {
                    "node": len(nodes),
                    "hemisphere": hemi,
                    "parcel_id": int(parcel_id),
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "z": float(centroid[2]),
                    "n_vertices": int(len(idx)),
                }
            )
    nodes_df = pd.DataFrame(nodes)
    coords = nodes_df[["x", "y", "z"]].to_numpy()
    return nodes_df, coords, coords_hemi, faces_hemi, labels_hemi


def generate_node_features(
    coords: np.ndarray,
    nodes_df: pd.DataFrame,
    dd_real: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    ============================================================
    SUBSTITUTABLE BLOCK – replace with real HCP/BALSA thickness
    and functional connectivity when available.
    ============================================================
    """
    N = len(nodes_df)

    # --- SYNTHETIC cortical thickness (mm) ---
    cortical_thickness = rng.normal(2.5, 0.35, N)
    cortical_thickness = np.clip(cortical_thickness, 1.5, 3.8)

    # --- SYNTHETIC local functional connectivity density ---
    feat = np.column_stack([cortical_thickness, coords[:, 0], coords[:, 1]])
    feat_std = StandardScaler().fit_transform(feat)
    sim = feat_std @ feat_std.T / feat_std.shape[1]
    functional_conn = sim.mean(axis=1)
    functional_conn = (functional_conn - functional_conn.min()) / (
        functional_conn.max() - functional_conn.min() + 1e-12
    )

    # True spatial process for DD generation
    W_true = KNN.from_array(coords, k=6)
    W_true.transform = "r"
    W_mat = W_true.full()[0]

    rho_true = 0.55
    beta_true = np.array([-0.40, 0.65])  # thickness ↓ DD, FC ↑ DD (example)
    X = np.column_stack([cortical_thickness, functional_conn])
    X_std = StandardScaler().fit_transform(X)
    eps = rng.normal(0, 0.35, N)
    y_raw = np.linalg.solve(np.eye(N) - rho_true * W_mat, X_std @ beta_true + eps)

    # Rescale to match empirical HCP mean / SD
    y = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-12) * dd_real.std() + dd_real.mean()
    y = np.clip(y, dd_real.min() * 0.9, dd_real.max() * 1.05)

    df = pd.DataFrame(
        {
            "node": nodes_df["node"],
            "hemisphere": nodes_df["hemisphere"],
            "parcel_id": nodes_df["parcel_id"],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "cortical_thickness": cortical_thickness,
            "functional_conn": functional_conn,
            "DD": y,
        }
    )
    return df, y, cortical_thickness, functional_conn, rho_true, sim


def build_weight_matrices(
    coords: np.ndarray, sim: np.ndarray, N: int
) -> Dict[str, W]:
    """Four row-standardized spatial weight matrices."""
    dist = squareform(pdist(coords, metric="euclidean"))
    np.fill_diagonal(dist, np.inf)

    # W1 – Inverse distance, keep 8 nearest
    W1_raw = 1.0 / dist
    np.fill_diagonal(W1_raw, 0.0)
    k_keep = 8
    for i in range(N):
        idx = np.argsort(W1_raw[i])[::-1][k_keep:]
        W1_raw[i, idx] = 0.0
    neighbors_W1 = {i: list(np.where(W1_raw[i] > 0)[0]) for i in range(N)}
    weights_W1 = {i: list(W1_raw[i, neighbors_W1[i]]) for i in range(N)}
    W1 = W(neighbors_W1, weights_W1)
    W1.transform = "r"

    # W2 – k-NN
    W2 = KNN.from_array(coords, k=6)
    W2.transform = "r"

    # W3 – Feature / functional similarity (75th percentile)
    thresh = np.percentile(sim[np.triu_indices(N, 1)], 75)
    W3_raw = (sim > thresh).astype(float)
    np.fill_diagonal(W3_raw, 0.0)
    neighbors_W3 = {i: list(np.where(W3_raw[i] > 0)[0]) for i in range(N)}
    weights_W3 = {i: [1.0] * len(neighbors_W3[i]) for i in range(N)}
    W3 = W(neighbors_W3, weights_W3)
    W3.transform = "r"

    # W4 – Contiguity (DistanceBand). Silence island warnings.
    thresh_dist = np.percentile(dist[np.isfinite(dist)], 15)  # slightly higher than original 12
    W4 = DistanceBand.from_array(
        coords, threshold=thresh_dist, binary=True, silence_warnings=True
    )
    W4.transform = "r"

    return {
        "W1_InvDist": W1,
        "W2_kNN": W2,
        "W3_FuncConn": W3,
        "W4_Contiguity": W4,
    }


def run_spatial_diagnostics(
    y: np.ndarray, W_dict: Dict[str, W]
) -> pd.DataFrame:
    rows = []
    for name, w in W_dict.items():
        mi = Moran(y, w, permutations=499)
        rows.append(
            {
                "W": name,
                "Moran_I": mi.I,
                "p_sim": mi.p_sim,
                "sparsity_pct": w.pct_nonzero,
                "max_neighbors": w.max_neighbors,
            }
        )
    return pd.DataFrame(rows).set_index("W")


def estimate_sar(
    y: np.ndarray,
    cortical_thickness: np.ndarray,
    functional_conn: np.ndarray,
    W_dict: Dict[str, W],
) -> Tuple[pd.DataFrame, Dict]:
    X_mat = np.column_stack([cortical_thickness, functional_conn])
    name_x = ["CorticalThickness", "FunctionalConn"]
    results = {}
    rows = []
    for name, w in W_dict.items():
        model = ML_Lag(
            y,
            X_mat,
            w=w,
            name_y="DD",
            name_x=name_x,
            name_w=name,
            name_ds="Destrieux_Nodes",
        )
        results[name] = model
        try:
            p_rho = model.z_stat[-1][1]
        except Exception:
            p_rho = np.nan
        rows.append(
            {
                "W": name,
                "rho": float(model.rho),
                "p_rho": p_rho,
                "beta_thickness": float(model.betas[1][0]),
                "beta_funcconn": float(model.betas[2][0]),
                "LogLik": float(model.logll),
                "AIC": float(model.aic),
                "PseudoR2": float(model.pr2),
            }
        )
    summary = pd.DataFrame(rows).set_index("W")
    return summary, results


def simulate_dynamics(
    coords: np.ndarray, rng: np.random.Generator, N: int
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic temporal activity + sliding-window functional correlations."""
    N_STEPS = 180
    WINDOW = 35
    N_FRAMES = 24

    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)
    W_space = np.zeros((N, N))
    for i in range(N):
        js = np.argsort(D[i])[:N_SPATIAL_NEIGHBORS]
        for j in js:
            W_space[i, j] = np.exp(-D[i, j] / 18.0)
    W_space = np.maximum(W_space, W_space.T)
    np.fill_diagonal(W_space, 0.0)
    W_space_norm = W_space / np.maximum(W_space.sum(axis=1, keepdims=True), 1e-12)

    n_latent = 5
    latent = np.zeros((N_STEPS, n_latent))
    for k in range(n_latent):
        phi = rng.uniform(0.72, 0.96)
        latent[0, k] = rng.normal()
        for t in range(1, N_STEPS):
            latent[t, k] = phi * latent[t - 1, k] + np.sqrt(1 - phi**2) * rng.normal()

    loadings = rng.normal(size=(N, n_latent))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=1, keepdims=True), 1e-12)
    drive = latent @ loadings.T
    drive = (drive - drive.mean(axis=0)) / (drive.std(axis=0) + 1e-8)

    X_act = np.zeros((N_STEPS, N))
    X_act[0] = rng.normal(scale=0.25, size=N)
    alpha, beta, gamma = 0.58, 0.20, 0.24
    for t in range(1, N_STEPS):
        spatial_influence = W_space_norm @ X_act[t - 1]
        regime = 0.5 + 0.5 * np.sin(2 * np.pi * t / 75.0)
        X_act[t] = np.tanh(
            alpha * X_act[t - 1]
            + beta * regime * spatial_influence
            + gamma * drive[t]
            + rng.normal(0, 0.22, N)
        )

    Xz = (X_act - X_act.mean(axis=0)) / (X_act.std(axis=0) + 1e-8)
    frame_times = np.unique(
        np.linspace(max(WINDOW, 1), N_STEPS - 1, N_FRAMES, dtype=int)
    )
    corr_frames, activity_frames = [], []
    for t in frame_times:
        segment = Xz[max(0, t - WINDOW + 1) : t + 1]
        C = np.corrcoef(segment, rowvar=False)
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 0.0)
        corr_frames.append(C)
        activity_frames.append(X_act[t].copy())

    sp_i, sp_j = np.where(np.triu(W_space, 1) > 0)
    return (
        X_act,
        corr_frames,
        activity_frames,
        frame_times,
        sp_i,
        sp_j,
    )


def run_full_pipeline(seed: int = SEED):
    """Execute the complete pipeline and return a state dict for the UI."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    dd_real, dd_source = load_dd_distribution(rng)
    nodes_df, coords, coords_hemi, faces_hemi, labels_hemi = build_cortical_nodes()
    N = len(nodes_df)

    df, y, cortical_thickness, functional_conn, rho_true, sim = generate_node_features(
        coords, nodes_df, dd_real, rng
    )
    W_dict = build_weight_matrices(coords, sim, N)
    moran_df = run_spatial_diagnostics(y, W_dict)
    summary_df, sar_results = estimate_sar(
        y, cortical_thickness, functional_conn, W_dict
    )
    best_w = summary_df["AIC"].idxmin()

    X_act, corr_frames, activity_frames, frame_times, sp_i, sp_j = simulate_dynamics(
        coords, rng, N
    )

    return {
        "nodes_df": nodes_df,
        "coords": coords,
        "coords_hemi": coords_hemi,
        "faces_hemi": faces_hemi,
        "df": df,
        "y": y,
        "cortical_thickness": cortical_thickness,
        "functional_conn": functional_conn,
        "rho_true": rho_true,
        "W_dict": W_dict,
        "moran_df": moran_df,
        "summary_df": summary_df,
        "best_w": best_w,
        "dd_real": dd_real,
        "dd_source": dd_source,
        "corr_frames": corr_frames,
        "activity_frames": activity_frames,
        "frame_times": frame_times,
        "sp_i": sp_i,
        "sp_j": sp_j,
        "N": N,
        "seed": seed,
    }
