import math
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import anndata
import scipy
import os, argparse

import tempfile
from pathlib import Path

import sys
import warnings
import time
from datetime import datetime
import json

import torch
from tqdm import tqdm
from scipy.stats import rankdata, norm

import ot

# style.use('seaborn-dark')
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14)

### Simulate spacial noise
# Takes a layer, rotates by angle. Next, we map all coordinates to closest integer coordinates.
# This removes some points as two points who share the same closest integer pair will map to same coordinate.

def rotate_spots(grid,spots,theta=0,translation=0,center_correction=0,figsize=(5,5),plot=False):
    grid = grid.copy() + center_correction
    spots = spots.copy() + center_correction
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    rotated_spots = np.array([R.dot(spots[i]) for i in range(len(spots))])
    rotated_spots += translation

    new_spots = grid[np.argmin(scipy.spatial.distance.cdist(rotated_spots,grid),axis=1)]

    grid -= center_correction
    spots -= center_correction
    rotated_spots -= center_correction
    new_spots -= center_correction

    seen = {}
    mapping = []
    for i in range(len(new_spots)):
        if tuple(new_spots[i]) in seen: continue
        seen[tuple(new_spots[i])] = 1
        mapping.append(i)

    if plot:
        fig = plt.figure(figsize=figsize)
        sns.scatterplot(x = grid[:,0],y = grid[:,1],linewidth=0,s=100, marker=".",alpha=0.2,color='blue')
        sns.scatterplot(x = rotated_spots[:,0],y = rotated_spots[:,1],linewidth=0,s=100, marker=".",color='red')
        sns.scatterplot(x = new_spots[:,0],y = new_spots[:,1],linewidth=0,s=100, marker=".",color='green')
        # plt.show()

    return new_spots,mapping

def simulate_spatial(adata, rotation_angle):
    adata_sim = adata.copy()
    grid_size = 40
    layer_grid = np.array([[x,y] for x in range(grid_size) for y in range(grid_size)])
    new_spots, mappings = rotate_spots(layer_grid, adata.obsm['spatial'], center_correction=-15, theta= rotation_angle)
    adata_sim.obsm['spatial'] = new_spots
    return adata_sim[mappings, :], mappings

### Simulate Gene Expression

def simulate_gene_exp(adata, pc = 0.25, factor = 1):
    """
    Adds noise to gene expression data. The rows are simulated according to a Multinomial distribution, 
    with the total counts per spot drawn from a Negative Binomial Distribution.
    param: pc- Pseudocount to be added to dataframe
    param: factor - amount by which we scale the variance (to increase noise)
    """
    adata_sim = adata.copy()
    # Ensure dense float array
    X_df = pd.DataFrame(adata_sim.X if not isinstance(adata_sim.X, pd.DataFrame) else adata_sim.X)
    df = X_df.copy()
    # add pseudocounts 
    alpha = df.copy().to_numpy() + pc

    # get vector of total counts per spot
    n = df.sum(axis=1).to_numpy()

    # Simulate total counts using negative binomial
    mean = np.mean(n)
    var = np.var(n)*factor
    # Guard against var <= mean -> invalid NB parameters
    if var <= mean:
        warnings.warn(f"[simulate_gene_exp] var (=\u003d{var:.3e}) <= mean (=\u003d{mean:.3e}); inflating var to 1.1*mean to avoid invalid NB.")
        var = 1.1 * max(mean, 1.0)
    n = sample_nb(mean, var, len(n)).astype(int)

    # Reassign zero counts so we don't divide by 0 in future calcuation
    n[n == 0] = 1

    # convert to float
    alpha = np.array(alpha, dtype=np.float64)
    n = np.array(n, dtype=np.float64)

    # convert rows to unit vectors
    alpha = alpha/alpha.sum(axis=1)[:, None]

    dist = np.empty(df.shape)
    for i in range(alpha.shape[0]):
        dist[i] = np.random.multinomial(n[i], alpha[i])
    new_df = pd.DataFrame(dist, index= df.index, columns= df.columns)
    adata_sim.X = new_df
    return adata_sim

def sample_nb(m, v, n = 1):
    """
    param: m - mean
    param: v - variance
    param: n - number of samples
    return: random sample from negative binomial distribution
    """
    r = m**2/(v - m)
    p = m/v
    r = max(r, 1e-8)
    p = min(max(p, 1e-8), 1 - 1e-8)
    samples = np.random.negative_binomial(r, p, n)
    return samples

### Define Simulation Accuracy

# ======================= QPAlign =======================

def qpalign(adata1, adata2, args=None, alpha_override=None, seed=42):
    # ===== CPU Setting =====
    try:
        req = int(getattr(args, 'workers', 0)) if args is not None else 0
    except Exception:
        req = 0
    if req and req > 0:
        try:
            cpu_total = os.cpu_count() or 1
        except Exception:
            cpu_total = 1
        n_workers = max(1, min(req, cpu_total))

        try:
            import torch
            torch.set_num_threads(n_workers)
            torch.set_num_interop_threads(max(1, n_workers))
        except Exception as e:
            warnings.warn(f"[qpalign] set_num_threads failed: {e}")

        os.environ['OMP_NUM_THREADS'] = str(n_workers)
        os.environ['MKL_NUM_THREADS'] = str(n_workers)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_workers)
        os.environ['NUMEXPR_NUM_THREADS'] = str(n_workers)

        if getattr(args, 'progress', True):
            print(f"[qpalign] Using CPU parallelism with {n_workers} worker threads.")

    # Graph building 
    A = build_graph(adata1)
    A2 = build_graph(adata2)

    # Feature building
    X = feature_gaussianize(adata1)
    Y = feature_gaussianize(adata2)

    A_t = torch.as_tensor(A, dtype=torch.float64)
    A2_t = torch.as_tensor(A2, dtype=torch.float64)
    X_t = torch.as_tensor(X, dtype=torch.float64)
    Y_t = torch.as_tensor(Y, dtype=torch.float64)

    if alpha_override is not None:
        lam_edge = float(1.0 - alpha_override)
        lam_feat = float(alpha_override)
    else:
        s = float(args.lam_edge + args.lam_feat) if args is not None else 2.0
        if s > 0:
            lam_edge, lam_feat = args.lam_edge / s, args.lam_feat / s
        else:
            lam_edge, lam_feat = args.lam_edge, args.lam_feat

    Pi = optimize(
        A_t, A2_t, X_t, Y_t,
        max_iter=args.max_iter, step_size=args.step_size,
        lam_edge=lam_edge, lam_feat=lam_feat,
        reg_lambda=args.reg_lambda, sinkhorn_iters=args.sinkhorn_iters,
        bb=args.bb, seed=seed, progress=args.progress, log_every=getattr(args, 'log_every', 20),
        name=getattr(args, 'run_name', 'ours')
    )

    G = permutation_matrix_from_Pi(Pi)
    return Pi, G


def get_simulation_accuracy(adata_layer, adata_layer_sim, mapping, alpha, args=None, seed=42):
    l = adata_layer.copy()
    siml = adata_layer_sim.copy()
    print(f"[get_simulation_accuracy] alpha={alpha}")
    _, G = qpalign(l, siml, args=args, alpha_override=alpha, seed=42)

    s = 0
    for i in range(len(mapping)):
        s += G[mapping[i], i]
    return  float(s) / len(mapping)

### Simulation

def simulate_once(adata_layer, pseudocounts, args=None, seed=42): 
    print(f"[simulate_once] start; pseudocounts={[float(x) for x in pseudocounts]}")
    adata_layer_sim_spatial, mappings = simulate_spatial(adata_layer, math.pi/3)
    
    # because we are varying pseudocounts, want to resimulate gene expression
    for idx, p in enumerate(pseudocounts):
        print(f"  [simulate_once] pseudocount={p}")
        if idx == 0 and p == 0:
            # add the factor = 0 case
            max_accuracy = adata_layer_sim_spatial.shape[0]/adata_layer.shape[0]
            baseline = [max_accuracy]
            mixed_0_01 = [max_accuracy]
            mixed_0_1 = [max_accuracy]
            mixed_0_2 = [max_accuracy]
            mixed_0_5 = [max_accuracy]
            mixed_0_9 = [max_accuracy]

            s = get_simulation_accuracy(adata_layer, adata_layer_sim_spatial, mappings, 1, args=args, seed=seed)
            spatial = [s]
            print(f"    [simulate_once] baseline/max_acc={max_accuracy:.4f} spatial={s:.4f}")
        else:
            adata_layer_sim_both = adata_layer_sim_spatial.copy()
            adata_layer_sim_both = simulate_gene_exp(adata_layer_sim_both, pc = p)
            b = get_simulation_accuracy(adata_layer, adata_layer_sim_both, mappings, 0, args=args)
            print(f"    [simulate_once] baseline (alpha=0)={b:.4f}")

            baseline.append(b)
            mixed_0_01.append(get_simulation_accuracy(adata_layer, adata_layer_sim_both, mappings, 0.01, args=args, seed=seed))
            mixed_0_1.append(get_simulation_accuracy(adata_layer, adata_layer_sim_both, mappings, 0.1, args=args, seed=seed))
            mixed_0_2.append(get_simulation_accuracy(adata_layer, adata_layer_sim_both, mappings, 0.2, args=args, seed=seed))
            mixed_0_5.append(get_simulation_accuracy(adata_layer, adata_layer_sim_both, mappings, 0.5, args=args, seed=seed))
            mixed_0_9.append(get_simulation_accuracy(adata_layer, adata_layer_sim_both, mappings, 0.9, args=args, seed=seed))
            spatial.append(s)
    return baseline, mixed_0_01, mixed_0_1, mixed_0_2, mixed_0_5, mixed_0_9, spatial

#### Read the data

def load_breast_layer(path):
    X = pd.read_csv(path, index_col=0)
    obs = pd.DataFrame(index=X.index.astype(str))
    var = pd.DataFrame(index=X.columns.astype(str))
    adata = anndata.AnnData(X=X.to_numpy(dtype=float), obs=obs, var=var)

    coor = []
    for c in X.index.astype(str):
        parts = str(c).replace(' ', '').split('x')
        if len(parts) != 2:
            raise ValueError(f"Row index '{c}' does not look like 'ixj'.")
        coor.append([float(parts[0]), float(parts[1])])
    adata.obsm['spatial'] = np.array(coor, dtype=float)
    return adata



######################## QPAlign Helpers ########################

# ---------------- build graph ----------------

def build_graph(adata, normalization=True) -> np.ndarray:
    """
    - adata.obsm['spatial'] 形状为 (n, 2)
    - 返回 (n, n) 的邻接矩阵
    """
    coor = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    dist = scipy.spatial.distance.cdist(coor, coor, metric='euclidean')

    n = dist.shape[0]
    A = np.zeros_like(dist, dtype=np.float64)

    mask = ~np.eye(n, dtype=bool)
    d = dist[mask]
    mu = d.mean()
    sd = d.std()
    if sd < 1e-12:
        np.fill_diagonal(A, 0.0)
        return A
    z = (d - mu) / (sd + 1e-12)

    A[mask] = -z if normalization else -d
    np.fill_diagonal(A, 0.0)
    return A

# ---------------- feature gaussianize ----------------

#### feature is in adata (except the index column), make it into a multidimensional guassian vector

def feature_gaussianize(adata) -> np.ndarray:
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    X = np.log1p(X)

    # # z-score
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd_safe = np.where(sd < 1e-12, 1.0, sd)  
    Z = (X - mu) / sd_safe
    Z[:, (sd < 1e-12).ravel()] = 0.0

    row_norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    Z = Z / row_norm
    return Z

    
# ---------------- objective & optimization ----------------

def feature_D_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    if X.shape[1] == 0:
        return torch.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype)
    x2 = (X**2).sum(dim=1, keepdim=True)
    y2 = (Y**2).sum(dim=1, keepdim=True).T
    XY = X @ Y.T
    D = x2 + y2 - 2.0 * XY
    return torch.clamp(D, min=0.0)


def objective_and_grad(Pi, A, A2, Dfeat, lam_edge=1.0, lam_feat=1.0, reg_lambda=0.01):
    E = A @ Pi - Pi @ A2
    f_edge = (E*E).sum()
    G_edge = 2.0 * (A.T @ E - E @ A2.T)

    f_feat = (Dfeat * (Pi*Pi)).sum()
    G_feat = 2.0 * (Dfeat * Pi)

    J = torch.ones_like(Pi)
    f_reg = reg_lambda * (Pi.T @ (J - Pi)).trace()
    G_reg = reg_lambda * (J - 2 * Pi)

    f = lam_edge * f_edge + lam_feat * f_feat + f_reg
    G = lam_edge * G_edge + lam_feat * G_feat + G_reg
    return f, G


@torch.no_grad()
def sinkhorn_projection(P: torch.Tensor, iters: int = 60, eps: float = 1e-8) -> torch.Tensor:
    P.clamp_(min=0.0)
    for _ in range(iters):
        rs = P.sum(dim=1, keepdim=True)
        rs = rs + eps
        P /= rs
        cs = P.sum(dim=0, keepdim=True)
        cs = cs + eps
        P /= cs
    return P


def optimize(A: torch.Tensor, A2: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
             max_iter: int = 300, step_size: float = 1e-2,
             lam_edge: float = 1.0, lam_feat: float = 1.0, reg_lambda: float = 0.01,
             sinkhorn_iters: int = 60, bb: bool = True, seed: int = 42,
             progress: bool = True, log_every: int = 20, name: str = "") -> torch.Tensor:
    n = A.shape[0]
    with torch.no_grad():
        sim_feat = X @ Y.T  
        sim_feat = torch.clamp(sim_feat, min=0)  
        degA = A.sum(dim=1, keepdim=True)    
        degA2 = A2.sum(dim=1, keepdim=True).T
        sim_deg = 1.0 / (1.0 + (degA - degA2).abs())  
        sim = sim_feat + 0.1 * sim_deg  
    
    Pi = sim.clone()
    Pi = sinkhorn_projection(Pi, iters=sinkhorn_iters)
    Dfeat = feature_D_matrix(X, Y)

    prev_f, prev_Pi, prev_G = None, None, None
    tol = 1e-7

    for it in range(max_iter):
        f, G = objective_and_grad(Pi, A, A2, Dfeat, lam_edge, lam_feat, reg_lambda)

        gnorm = float(torch.linalg.norm(G).item())
        # BB 
        alpha = step_size
        if bb and prev_Pi is not None and prev_G is not None:
            S = (Pi - prev_Pi).reshape(-1)
            Yg = (G - prev_G).reshape(-1)
            denom = torch.dot(Yg, S)
            num = torch.dot(S, S)
            denom_abs = float(denom.abs().item())
            if denom_abs > 1e-12:
                alpha = float((num / denom.clamp(min=1e-12)).item())
                alpha = float(np.clip(alpha, 1e-5, 5e-1))

        prev_Pi, prev_G = Pi.clone(), G.clone()
        Pi.add_(G, alpha=-alpha)
        sinkhorn_projection(Pi, iters=sinkhorn_iters)

        if not torch.isfinite(Pi).all():
            warnings.warn("[optimize] Pi has non-finite values; resetting to similarity initialization.")
            Pi = sim.clone()
            sinkhorn_projection(Pi, iters=sinkhorn_iters)

        f_post, _ = objective_and_grad(Pi, A, A2, Dfeat, lam_edge, lam_feat, reg_lambda)
        f_val = float(f_post.item())

        if progress and (it == 0 or (log_every and (it % log_every == 0 or it == max_iter-1))):
            with torch.no_grad():
                Pi_min = float(Pi.min().item())
                Pi_max = float(Pi.max().item())
                rs = Pi.sum(dim=1).mean().item()
                cs = Pi.sum(dim=0).mean().item()
            print(f"[{name}] it={it:04d} f={f_val:.6e} step={alpha:.2e} ||G||={gnorm:.3e} Pi[min,max]=({Pi_min:.3e},{Pi_max:.3e}) rowSum~{rs:.3f} colSum~{cs:.3f}")
            sys.stdout.flush()

        if prev_f is not None:
            rel = abs(f_val - prev_f) / (1.0 + abs(prev_f))
            if rel < tol:
                if progress:
                    print(f"[{name}] early stop at it={it} with Δf={rel:.3e} < tol={tol}.")
                break
        prev_f = f_val

    return Pi


def permutation_matrix_from_Pi(Pi: torch.Tensor) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    P = Pi if isinstance(Pi, np.ndarray) else Pi.detach().cpu().numpy()
    if not np.isfinite(P).all():
        raise ValueError("permutation_matrix_from_Pi: matrix contains NaN/Inf; cannot discretize.")
    n, m = P.shape
    r, c = linear_sum_assignment(-P)
    G = np.zeros((n, m), dtype=np.float64)
    for i in range(len(r)):
        G[r[i], c[i]] = 1.0
    return G


# ===============================
#  Refactored plotting utilities (now supports 1-4 slices)
# ===============================

def _grid_for_n(n: int):
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    return 2, 2  # for 3 or 4


def plot_accuracy_panels(mean, sd, pseudocounts, slice_titles, out_dir,
                         fname_base="panel"):
    os.makedirs(out_dir, exist_ok=True)

    # Consistent plotting order and labels
    alphas_order = [0.1, 0, 1, 0.01, 0.2, 0.5, 0.9]

    alpha_labels = {
        0: r"$\lambda = 0$ (Gene Exp Only)",
        1: r"$\lambda = 1$ (Spatial Only)",
        0.01: r"$\lambda = 0.01$ (Mixed)",
        0.1: r"$\lambda = 0.1$ (Mixed)",
        0.2: r"$\lambda = 0.2$ (Mixed)",
        0.5: r"$\lambda = 0.5$ (Mixed)",
        0.9: r"$\lambda = 0.9$ (Mixed)",
    }

    n = len(slice_titles)
    nrows, ncols = _grid_for_n(n)
    figsize = (6.5*ncols, 5*nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    x = np.asarray([float(p) for p in pseudocounts], dtype=float)

    for i in range(n):
        ax = axes[i]
        for a in alphas_order:
            if a not in mean:
                continue
            y = mean[a][i].values.astype(float)
            yerr = sd[a][i].values.astype(float)
            ax.errorbar(x, y, yerr=yerr, marker='o', linewidth=1, capsize=3,
                        label=alpha_labels.get(a, str(a)))

        # Max accuracy dashed line (from alpha=0, first column)
        max_acc = float(mean[0][i].iloc[0]) if 0 in mean else 1.0
        ax.plot(x, np.full_like(x, max_acc, dtype=float), linestyle='--',
                linewidth=1, label='Max Accuracy')

        ax.set_title(slice_titles[i], fontsize=14)
        ax.set_xlim(min(x) - 0.1, max(x) + 0.1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_facecolor('white')
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1)
        ax.set_xlabel('Pseudocount (δ)', fontsize=14)
        ax.set_ylabel('Overlap', fontsize=14)
        ax.set_xticks(x)

    # hide unused axes (when n=3)
    for j in range(n, len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3,
               frameon=True)

    png_path = os.path.join(out_dir, f"{fname_base}.png")
    pdf_path = os.path.join(out_dir, f"{fname_base}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)


def _parse_slices_arg(s: str):
    valid = ['a', 'b', 'c', 'd']
    if not s:
        return valid
    vals = [t.strip().lower() for t in s.split(',')]
    if any(v in ('all', '*') for v in vals):
        return valid
    out = []
    for v in vals:
        if v not in valid:
            raise argparse.ArgumentTypeError(f"Invalid slice '{v}'. Use one or more of a,b,c,d or 'all'.")
        if v not in out:
            out.append(v)
    # keep canonical order
    order = {k:i for i,k in enumerate(valid)}
    out.sort(key=lambda k: order[k])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="datasets/cancer",
                    help="Directory containing slice CSV files (slice1.csv, slice2.csv, ...)")
    ap.add_argument("--outdir", type=str, default="outputs/")
    ap.add_argument("--limit_n", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    ap.add_argument("--standardize_feat", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)
    ap.add_argument("--max_iter", type=int, default=400)
    ap.add_argument("--sinkhorn_iters", type=int, default=80)
    ap.add_argument("--step_size", type=float, default=1e-2)
    ap.add_argument("--lam_edge", type=float, default=1.0)
    ap.add_argument("--lam_feat", type=float, default=1.0)
    ap.add_argument("--reg_lambda", type=float, default=0.01,
                    help="Regularization strength for the term tr(P^T(J-P))")
    ap.add_argument("--bb", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)
    ap.add_argument("--progress", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)

    # Logging frequency and a run name to identify logs
    ap.add_argument("--log_every", type=int, default=20, help="How often (in iters) to print optimizer status; 0 to disable")
    ap.add_argument("--run_name", type=str, default="ours", help="Name prefix for optimizer logs")

    ap.add_argument("--workers", type=int, default=0, help="CPU cores used for math kernels; 0=auto (all available / SLURM allocation)")

    # Pseudocount list
    ap.add_argument("--pc", type=lambda s: [float(item) for item in s.split(',')],
                    default=[0, 1, 2, 3, 4, 5],
                    help="Comma-separated list of pseudocounts to test (e.g. '0,0.01,0.1,0.2,0.5,0.9')")
    # choose which slices to compute & plot
    ap.add_argument("--slices", type=_parse_slices_arg, default=['a','b','c','d'],
                    help="Comma-separated subset of {a,b,c,d} or 'all' (e.g. 'a,c' or 'all')")
    # NEW: only plot (skip recompute)
    ap.add_argument("--plot_only", action="store_true",
                    help="Only aggregate existing CSVs and plot; skip simulations and alignment computation")
    # NEW: runs per slice
    ap.add_argument("--runs", type=int, default=10,
                    help="Number of repeated simulations per selected slice when not using --plot_only")

    args = ap.parse_args()



    key_to_idx   = {'a':1, 'b':2, 'c':3, 'd':4}
    key_to_name  = {'a':'SliceA', 'b':'SliceB', 'c':'SliceC', 'd':'SliceD'}
    key_to_title = {'a':'Slice A', 'b':'Slice B', 'c':'Slice C', 'd':'Slice D'}

    selected_keys = args.slices  # already normalized order

    # read only selected data (when computing)
    slices = {}
    if not args.plot_only:
        for k in selected_keys:
            path = os.path.join(args.dataset_dir, f"slice{key_to_idx[k]}.csv")
            slices[key_to_name[k]] = load_breast_layer(path)

    ## set a subdir with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path_to_output_dir = os.path.join(args.outdir, f"{timestamp}")

    if not os.path.exists(path_to_output_dir):
        os.makedirs(path_to_output_dir)
    for k in selected_keys:
        slice_name = key_to_name[k]
        sub_dir = os.path.join(path_to_output_dir, slice_name)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    
    ######## Save meta information #########
    meta = {
        "dtype": args.dtype, "workers": args.workers, "pc": args.pc,
        "runs": args.runs, "max_iter": args.max_iter, "slices": args.slices, 
        "sinkhorn_iters": args.sinkhorn_iters, "step_size": args.step_size,
        "lam_edge": float(args.lam_edge), "lam_feat": float(args.lam_feat), "reg_lambda": float(args.reg_lambda),
        "bb": args.bb,
    }
    with open(os.path.join(path_to_output_dir, "run_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    # Number of runs per experiment
    N_RUNS = int(args.runs)
    pseudocounts = args.pc

    # ===================
    #   Run simulations (unless --plot_only)
    # ===================
    if not args.plot_only:
        print(f"[main] N_RUNS={N_RUNS} pseudocounts={pseudocounts} slices={selected_keys}")
        for slice_name, adata in slices.items():
            print(f"[main] === Start slice: {slice_name} ===")
            baseline_all = []
            mixed_0_01_all = []
            mixed_0_1_all = []
            mixed_0_2_all = []
            mixed_0_5_all = []
            mixed_0_9_all = []
            spatial_all = []
            for i in range(N_RUNS):
                seed = 42 + i * 1000
                print(f"[main]  -- Run {i+1}/{N_RUNS} --")
                b, m_0_01,  m_0_1,  m_0_2,  m_0_5, m_0_9, s  = simulate_once(adata, pseudocounts, args=args, seed=seed)
                baseline_all.append(b)
                mixed_0_01_all.append(m_0_01)
                mixed_0_1_all.append(m_0_1)
                mixed_0_2_all.append(m_0_2)
                mixed_0_5_all.append(m_0_5)
                mixed_0_9_all.append(m_0_9)
                spatial_all.append(s)
            
            pd.DataFrame(baseline_all, columns = pseudocounts).to_csv(os.path.join(path_to_output_dir, slice_name,'baseline.csv'))
            pd.DataFrame(mixed_0_01_all, columns = pseudocounts).to_csv(os.path.join(path_to_output_dir, slice_name,'mixed_0.01.csv'))
            pd.DataFrame(mixed_0_1_all, columns = pseudocounts).to_csv(os.path.join(path_to_output_dir, slice_name,'mixed_0.1.csv'))
            pd.DataFrame(mixed_0_2_all, columns = pseudocounts).to_csv(os.path.join(path_to_output_dir, slice_name,'mixed_0.2.csv'))
            pd.DataFrame(mixed_0_5_all, columns = pseudocounts).to_csv(os.path.join(path_to_output_dir, slice_name,'mixed_0.5.csv'))
            pd.DataFrame(mixed_0_9_all, columns = pseudocounts).to_csv(os.path.join(path_to_output_dir, slice_name,'mixed_0.9.csv'))
            pd.DataFrame(spatial_all, columns = pseudocounts).to_csv(os.path.join(path_to_output_dir, slice_name,'spatial.csv'))
            print(f"[main] Saved CSVs for {slice_name}.")

    else:
        print(f"[main] plot_only=True -> will skip computation and only plot. Checking CSV files...")
        missing = []
        alpha_to_filename = {
            0 : 'baseline.csv',
            0.01 : 'mixed_0.01.csv',
            0.1 : 'mixed_0.1.csv',
            0.2 : 'mixed_0.2.csv',
            0.5 : 'mixed_0.5.csv',
            0.9 : 'mixed_0.9.csv',
            1 : 'spatial.csv',
        }
        for k in selected_keys:
            s_name = key_to_name[k]
            for fname in alpha_to_filename.values():
                fpath = os.path.join(path_to_output_dir, s_name, fname)
                if not os.path.exists(fpath):
                    missing.append(fpath)
        if missing:
            print("[main][ERROR] The following files are missing; cannot plot-only. Please run without --plot_only first:")
            for f in missing:
                print("  ", f)
            sys.exit(2)

    alpha_to_filename = {
        0 : 'baseline.csv',
        0.01 : 'mixed_0.01.csv',
        0.1 : 'mixed_0.1.csv',
        0.2 : 'mixed_0.2.csv',
        0.5 : 'mixed_0.5.csv',
        0.9 : 'mixed_0.9.csv',
        1 : 'spatial.csv',    
    }

    mean = {a: [] for a in alpha_to_filename.keys()}
    sd   = {a: [] for a in alpha_to_filename.keys()}

    # only aggregate for selected slices
    for k in selected_keys:
        s_name = key_to_name[k]
        for alpha, fname in alpha_to_filename.items():
            path = os.path.join(path_to_output_dir, s_name, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required CSV: {path}")
            df = pd.read_csv(path, index_col=0)
            mean[alpha].append(df.mean())
            sd[alpha].append(df.std())

    # Pseudocounts (column order) — cast to float for plotting
    # choose a reference present key
    ref_key = 0 if 0 in mean and len(mean[0]) else next(iter(mean.keys()))
    pseudocounts = [float(pc) for pc in list(mean[ref_key][0].index)]

    # Plotting
    fig_dir = os.path.join(path_to_output_dir, "figures")
    slice_titles = [key_to_title[k] for k in selected_keys]
    plot_accuracy_panels(mean, sd, pseudocounts, slice_titles, fig_dir,
                        fname_base="alignment_accuracy_panel")

    print(f"[main] Done. Figures written to {fig_dir}/")
