import os, argparse, json, ast, math, re
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings

# ---------------- utilities ----------------

def standardize_features(X: torch.Tensor) -> torch.Tensor:
    if X.numel() == 0:
        return X
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True) + 1e-8
    return (X - mu) / sd


# ========= Vector parsing =========
_NUM_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

def _clean_vec_str(s: str) -> str:
    s = s.replace('Ellipsis', ' ').replace('...', ' ').replace('…', ' ')
    s = s.replace('[', ' ').replace(']', ' ').repunlace(';', ' ')
    s = s.replace('\t', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(' ', ',')
    s = re.sub(r',,+', ',', s).strip(',')
    return s



def _parse_vector_cell(cell) -> np.ndarray:
    # Fast path: list/ndarray
    if isinstance(cell, (list, np.ndarray)):
        try:
            return np.asarray([x for x in cell if x is not Ellipsis], dtype=float)
        except Exception:
            pass

    s = str(cell).strip()
    # Try literal_eval → Python list / JSON-like
    try:
        v = ast.literal_eval(s)
        if v is Ellipsis:
            return np.array([], dtype=float)
        if isinstance(v, (list, tuple, np.ndarray)):
            v = [x for x in v if x is not Ellipsis]
            return np.asarray(v, dtype=float)
    except Exception:
        pass

    # Fallback: clean & split, keep only numeric tokens
    s2 = _clean_vec_str(s)
    if not s2:
        return np.array([], dtype=float)
    toks = [t for t in s2.split(',') if t]
    nums = []
    for t in toks:
        if _NUM_RE.match(t):
            try:
                nums.append(float(t))
            except Exception:
                pass
    return np.asarray(nums, dtype=float)


def _parse_vectors_series(vec_series: pd.Series, desc: str = "Parsing features") -> np.ndarray:
    # Pass 1: detect dimension
    dim = None
    for v in vec_series:
        arr = _parse_vector_cell(v)
        if arr.size > 0:
            dim = int(arr.size)
            break
    if dim is None:
        # degenerate: all empty → (N,0)
        return np.zeros((len(vec_series), 0), dtype=float)

    # Pass 2: build matrix with padding/truncation
    out = []
    for v in tqdm(vec_series, desc=desc):
        arr = _parse_vector_cell(v)
        if arr.size == dim:
            out.append(arr)
        elif arr.size == 0:
            out.append(np.zeros(dim, dtype=float))
        elif arr.size > dim:
            out.append(arr[:dim])
        else:  # shorter
            pad = np.zeros(dim, dtype=float)
            pad[:arr.size] = arr
            out.append(pad)
    return np.vstack(out)


def load_feature_csv(path: str,
                     id_col_hint=("id",),
                     vec_col_hint=("gaussian_vector",),
                     desc: str = None) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[k.lower()] for k in id_col_hint if k.lower() in cols), df.columns[0])
    vec_col = next((cols[k.lower()] for k in vec_col_hint if k.lower() in cols), df.columns[-1])
    ids = df[id_col].astype(str).tolist()
    X = _parse_vectors_series(df[vec_col], desc or f"Parsing {Path(path).name}")
    return ids, X


def load_edges_csv_vectorized(path: str, id_to_idx: Dict[str, int]) -> np.ndarray:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    src_col = cols.get("source", None) or df.columns[0]
    dst_col = cols.get("target", None) or df.columns[1]
    w_col = cols.get("weight", cols.get("weight_raw", None))
    n = len(id_to_idx)
    A = np.zeros((n, n), dtype=np.float32)
    s_idx = df[src_col].astype(str).map(id_to_idx)
    t_idx = df[dst_col].astype(str).map(id_to_idx)
    mask = s_idx.notna() & t_idx.notna() & (s_idx != t_idx)
    s = s_idx[mask].astype(np.int64).to_numpy()
    t = t_idx[mask].astype(np.int64).to_numpy()
    if w_col is not None:
        w = df.loc[mask, w_col].astype(float).to_numpy()
    else:
        w = np.ones_like(s, dtype=np.float64)
    np.add.at(A, (s, t), w)
    np.add.at(A, (t, s), w)
    np.fill_diagonal(A, 0.0)
    return A


def load_mapping_csv(path: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    acm_like = [i for i, c in enumerate(cols) if "acm" in c]
    dblp_like = [i for i, c in enumerate(cols) if "dblp" in c]
    if len(acm_like) == 1 and len(dblp_like) == 1:
        acm_col = df.columns[acm_like[0]]
        dblp_col = df.columns[dblp_like[0]]
    else:
        acm_col, dblp_col = df.columns[:2]
    acm_ids = df[acm_col].astype(str).tolist()
    dblp_ids = df[dblp_col].astype(str).tolist()
    return list(zip(acm_ids, dblp_ids))

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

    # Regularization
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
        P /= (P.sum(dim=1, keepdim=True) + eps)
        P /= (P.sum(dim=0, keepdim=True) + eps)
    return P

def optimize(A: torch.Tensor, A2: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
             max_iter: int = 300, step_size: float = 1e-2,
             lam_edge: float = 1.0, lam_feat: float = 1.0, reg_lambda: float = 0.01,
             sinkhorn_iters: int = 60, bb: bool = True, seed: int = 42,
             progress: bool = False, log_every: int = 20, name: str = "") -> torch.Tensor:
    n = A.shape[0]
    g = torch.Generator(device=A.device)
    g.manual_seed(seed)
    Pi = torch.rand((n, n), dtype=A.dtype, device=A.device, generator=g)

    Pi = sinkhorn_projection(Pi, iters=sinkhorn_iters)
    
    Dfeat = feature_D_matrix(X, Y)

    prev_f, prev_Pi, prev_G = None, None, None
    tol = 1e-7

    for it in tqdm(range(max_iter), desc="Optimizing", disable=not progress):
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

        if prev_f is not None:
            rel = abs(f_val - prev_f) / (1.0 + abs(prev_f))
            if rel < tol:
                if progress:
                    print(f"[{name}] early stop at it={it} with Δf={rel:.3e} < tol={tol}.")
                break
        prev_f = f_val
    return Pi


def round_to_permutation(Pi: torch.Tensor) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    Pi_np = Pi.detach().cpu().numpy()
    r, c = linear_sum_assignment(-Pi_np)
    return c


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acm_features", type=str, default=None)
    ap.add_argument("--dblp_features", type=str, default=None)
    ap.add_argument("--acm_edges", type=str, default=None)
    ap.add_argument("--dblp_edges", type=str, default=None)
    ap.add_argument("--mapping", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs_cpu_integrated")
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

    ap.add_argument("--workers", type=int, default=0, help="CPU cores used for math kernels; 0=auto (all available / SLURM allocation)")
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")] if s else [], default=[999, 3407, 9999, 12345, 15268],
                    help="Random seeds for multiple runs, e.g. 1,2,3")
    args = ap.parse_args()

    # ---- CPU worker setup ----
    def _auto_workers():
        try:
            if "SLURM_CPUS_PER_TASK" in os.environ:
                return int(os.environ["SLURM_CPUS_PER_TASK"])
            if "SLURM_CPUS_ON_NODE" in os.environ:
                return int(os.environ["SLURM_CPUS_ON_NODE"])
            if hasattr(os, "sched_getaffinity"):
                return len(os.sched_getaffinity(0))
            import multiprocessing as mp
            return mp.cpu_count()
        except Exception:
            return 1

    workers = int(getattr(args, "workers", 0) or 0)
    if workers <= 0:
        workers = _auto_workers()

    os.environ.setdefault("OMP_NUM_THREADS", str(workers))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(workers))
    os.environ.setdefault("MKL_NUM_THREADS", str(workers))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(workers))

    try:
        torch.set_num_threads(max(1, workers))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(4, workers)))
    except Exception:
        pass

    print(f"[CPU] workers = {workers}")

    os.makedirs(args.outdir, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    acm_feat_path, dblp_feat_path = args.acm_features, args.dblp_features
    acm_edges_path, dblp_edges_path = args.acm_edges, args.dblp_edges

    acm_ids_all, X_all = load_feature_csv(acm_feat_path, desc="Parsing ACM features")
    dblp_ids_all, Y_all = load_feature_csv(dblp_feat_path, desc="Parsing DBLP features")
    dX, dY = X_all.shape[1], Y_all.shape[1]
    if dX != dY:
        raise ValueError(f"Feature dim mismatch: ACM d={dX} vs DBLP d={dY}")

    # loading mapping
    mapping_pairs = load_mapping_csv(args.mapping)
    acm_set, dblp_set = set(acm_ids_all), set(dblp_ids_all)
    filtered = [(a, d) for (a, d) in mapping_pairs if a in acm_set and d in dblp_set]
    if len(filtered) == 0:
        raise RuntimeError("After filtering, no mapping pairs have both features available.")
    if args.limit_n and args.limit_n > 0:
        filtered = filtered[:args.limit_n]

    acm_index_full = {nid: i for i, nid in enumerate(acm_ids_all)}
    dblp_index_full = {nid: i for i, nid in enumerate(dblp_ids_all)}

    keep_acm = {a for (a, _) in filtered}
    keep_dblp = {d for (_, d) in filtered}

    acm_ids = [nid for nid in acm_ids_all if nid in keep_acm]
    dblp_ids = [nid for nid in dblp_ids_all if nid in keep_dblp]
    if len(acm_ids) != len(dblp_ids):
        raise RuntimeError(f"ACM/DBLP counts after selection differ: {len(acm_ids)} vs {len(dblp_ids)}")
    n = len(acm_ids)
    print(f"[Info] n={n}, d={dX}")

    X = np.vstack([X_all[acm_index_full[nid]] for nid in acm_ids])
    Y = np.vstack([Y_all[dblp_index_full[nid]] for nid in dblp_ids])

    acm_local = {nid: i for i, nid in enumerate(acm_ids)}
    dblp_local = {nid: i for i, nid in enumerate(dblp_ids)}
    A = load_edges_csv_vectorized(acm_edges_path, acm_local)
    A2 = load_edges_csv_vectorized(dblp_edges_path, dblp_local)

    A_t = torch.tensor(A, dtype=dtype)
    A2_t = torch.tensor(A2, dtype=dtype)
    X_t = torch.tensor(X, dtype=dtype)
    Y_t = torch.tensor(Y, dtype=dtype)
    if args.standardize_feat:
        X_t = standardize_features(X_t)
        Y_t = standardize_features(Y_t)

    if (args.lam_edge + args.lam_feat) > 0:
        s = args.lam_edge + args.lam_feat
        lam_edge = args.lam_edge / s
        lam_feat = args.lam_feat / s
    else:
        lam_edge, lam_feat = args.lam_edge, args.lam_feat

    # Loading ground truth
    dblp_index_map = {nid: i for i, nid in enumerate(dblp_ids)}
    acm_to_dblp = dict(filtered)  
    p_true = np.array([dblp_index_map[acm_to_dblp[a]] for a in acm_ids], dtype=int)
    print("p_true:", p_true)
    print(len(acm_ids), len(dblp_ids), X.shape, Y.shape, A.shape, A2.shape)

    n_runs = max(1, len(args.seeds))
    overlaps = []
    for run_id in range(n_runs):
        seed = args.seeds[run_id] if run_id < len(args.seeds) else 42
        print(f"[Run {run_id+1}/{n_runs}], seed={seed}")
        Pi = optimize(A_t, A2_t, X_t, Y_t,
                    max_iter=args.max_iter, step_size=args.step_size,
                    lam_edge=lam_edge, lam_feat=lam_feat, reg_lambda=args.reg_lambda,
                    sinkhorn_iters=args.sinkhorn_iters, bb=args.bb, seed=seed,
                    progress=args.progress)
        col = round_to_permutation(Pi)
        correct_matches = (col == p_true)
        overlaps.append(correct_matches.sum() / n)
        print(f" → overlap in run {run_id+1}: {(correct_matches.sum() / n)}")
    overlap = sum(overlaps) / n_runs 
    print(f"[Result] Overlaps: {overlaps}")
    print(f"[Result] n={n}, overlap={overlap:.4f}")
    # print prameters
    print(f"[Params] lam_edge={lam_edge:.4f}, lam_feat={lam_feat:.4f}, reg_lambda={args.reg_lambda:.4f}")
    print(f"[Params] max_iter={args.max_iter}, sinkhorn_iters={args.sinkhorn_iters}, step_size={args.step_size}, bb={args.bb}")

    # saving outputs
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.outdir) / stamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Saving outputs to {save_dir}")

    np.save(save_dir / "Pi.npy", Pi.detach().cpu().numpy())
    with open(save_dir / "overlap.txt", "w") as f:
        f.write(f"{overlap:.6f}\n")

    pred_rows = []
    for i, j in enumerate(col):
        pred_rows.append({
            "acm_id": acm_ids[i],
            "pred_dblp_id": dblp_ids[j],
            "correct": bool(j == p_true[i]),
            "score": float(Pi[i, j].item()),
        })
    pd.DataFrame(pred_rows).to_csv(save_dir / "predicted_matching.csv", index=False)

    meta = {
        "n": n, "d": int(X.shape[1]), "dtype": args.dtype,
        "standardize_feat": args.standardize_feat, "max_iter": args.max_iter,
        "sinkhorn_iters": args.sinkhorn_iters, "step_size": args.step_size,
        "lam_edge": float(lam_edge), "lam_feat": float(lam_feat), "reg_lambda": float(args.reg_lambda),
        "bb": args.bb,
        "mode": "raw" if use_raw and not use_precomputed else "precomputed",
    }
    with open(save_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
