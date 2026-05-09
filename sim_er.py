#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, time, math
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------- utils -------------------

def parse_list(spec: str, kind=float):
    if not spec: return []
    items = []
    for tok in spec.split(","):
        tok = tok.strip()
        if ":" in tok:
            a,b,c = tok.split(":")
            items.extend(np.linspace(kind(a), kind(b), int(c)).tolist())
        else:
            items.append(kind(tok))
    return items

def torch_dtype(s: str):
    return torch.float32 if s == "float32" else torch.float64

def standardize_features(X):
    if X.numel() == 0: return X
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True) + 1e-8
    return (X - mu) / sd


# ------------------- data generation (ER graphs + Gaussian features) -------------------

def sample_correlated_graphs(n, d, rho, r, device, dtype, gen=None, standardize=True, p_edge=0.5):
    """
    Generate a pair of correlated graphs (A, A2_obs) and Gaussian node features (X, Y_obs).

    Edge model: correlated Erdos-Renyi (upper-triangular sampling with probabilities)
      - p_edge: marginal edge probability (same for both graphs)
      - rho: edge-wise correlation between A and A2_true
    Feature model: Gaussian as before; correlation between features controlled by r.

    Returns:
      A, A2_obs, X, Y_obs, perm (ground-truth permutation as numpy array)
    """
    if gen is None:
        gen = torch.Generator(device=device); gen.manual_seed(torch.seed())

    # ----------------- correlated ER graphs (upper triangular sampling) -----------------
    # Use the standard construction for correlated Bernoulli pairs
    # See user's provided logic: sample u ~ Uniform(0,1) on upper-triangle and map to (0/0),(1/0),(0/1),(1/1)
    p = float(p_edge)
    # safety clamp
    p = min(max(p, 0.0), 1.0)

    # draw uniform upper-triangular matrix
    u = torch.rand((n, n), device=device, generator=gen)
    u = torch.triu(u, diagonal=1)

    p11 = p * p + rho * p * (1.0 - p)
    p10 = (1.0 - rho) * p * (1.0 - p)
    p01 = p10
    # p00 implicitly = 1 - (p11 + p10 + p01)

    A1 = torch.zeros((n, n), device=device, dtype=dtype)
    A2_true = torch.zeros((n, n), device=device, dtype=dtype)

    mask_11 = (u < p11)
    mask_10 = (u >= p11) & (u < p11 + p10)
    mask_01 = (u >= p11 + p10) & (u < p11 + p10 + p01)

    A1[mask_11 | mask_10] = 1.0
    A2_true[mask_11 | mask_01] = 1.0

    # symmetrize
    A = A1 + A1.T
    A2_true = A2_true + A2_true.T
    A.fill_diagonal_(0.0); A2_true.fill_diagonal_(0.0)

    # ----------------- features (Gaussian) -----------------
    if d > 0:
        X = torch.randn((n, d), device=device, dtype=dtype, generator=gen)
        Zf = torch.randn((n, d), device=device, dtype=dtype, generator=gen)
        scale_r = torch.sqrt(torch.clamp(torch.tensor(1 - r**2, device=device, dtype=dtype), min=0.0))
        Y_true = r * X + scale_r * Zf
        if standardize:
            X = standardize_features(X)
            Y_true = standardize_features(Y_true)
    else:
        X = torch.zeros((n,0), device=device, dtype=dtype)
        Y_true = torch.zeros((n,0), device=device, dtype=dtype)

    # ----------------- random permutation (ground truth) and observed A2/Y -----------------
    perm = torch.randperm(n, device=device, generator=gen)
    P = torch.zeros((n, n), device=device, dtype=dtype)
    P[torch.arange(n, device=device), perm] = 1.0
    A2_obs = P.T @ A2_true @ P
    Y_obs = P.T @ Y_true

    return A, A2_obs, X, Y_obs, perm


# ------------------- objective & gradient -------------------

def feature_D_matrix(X, Y):
    # D_{kj} = sum_i (x_{k,i} - y_{j,i})^2
    if X.shape[1] == 0:
        return torch.zeros((X.shape[0], Y.shape[0]), device=X.device, dtype=X.dtype)
    x2 = (X**2).sum(dim=1, keepdim=True)      # (n,1)
    y2 = (Y**2).sum(dim=1, keepdim=True).T    # (1,n)
    XY = X @ Y.T                               # (n,n)
    D = x2 + y2 - 2.0 * XY
    return torch.clamp(D, min=0.0)


def objective_and_grad(Pi, A, A2, Dfeat, lam_edge=1.0, lam_feat=1.0, reg_lambda=0.01):
    E = A @ Pi - Pi @ A2
    f_edge = (E*E).sum()
    G_edge = 2.0 * (A.T @ E - E @ A2.T)

    f_feat = (Dfeat * (Pi*Pi)).sum()
    G_feat = 2.0 * (Dfeat * Pi)

    # Regularization tr(P^T(J-P))
    J = torch.ones_like(Pi)
    f_reg = reg_lambda * (Pi.T @ (J - Pi)).trace()
    G_reg = reg_lambda * (J - 2 * Pi)

    f = lam_edge * f_edge + lam_feat * f_feat + f_reg
    G = lam_edge * G_edge + lam_feat * G_feat + G_reg
    return f, G, f_edge, f_feat


@torch.no_grad()
def sinkhorn_projection(P, iters=60, eps=1e-8):
    P.clamp_(min=0.0)
    for _ in range(iters):
        P /= (P.sum(dim=1, keepdim=True) + eps)
        P /= (P.sum(dim=0, keepdim=True) + eps)
    return P


def optimize(A, A2, X, Y, max_iter=300, step_size=1e-2,
             lam_edge=1.0, lam_feat=1.0, reg_lambda=0.01, sinkhorn_iters=60, bb=True, tol=1e-7):
    """
    PGD + Sinkhorn. Optional Barzilai–Borwein step-size update (bb=True).
    """
    n = A.shape[0]
    # Initialization
    with torch.no_grad():
        sim_feat = X @ Y.T  
        sim_feat = torch.clamp(sim_feat, min=0) 

        degA = A.sum(dim=1, keepdim=True)    
        degA2 = A2.sum(dim=1, keepdim=True).T
        sim_deg = 1.0 / (1.0 + (degA - degA2).abs())  

        sim = sim_feat + 0.1 * sim_deg  
    # Sinkhorn 
    Pi = sinkhorn_projection(sim, iters=sinkhorn_iters)
    
    Dfeat = feature_D_matrix(X, Y)

    prev_f = None
    prev_Pi = None
    prev_G = None

    for it in range(max_iter):
        f, G, fE, fF = objective_and_grad(Pi, A, A2, Dfeat, lam_edge, lam_feat, reg_lambda)

        # BB step (diagonal-free)
        alpha = step_size
        if bb and prev_Pi is not None and prev_G is not None:
            S = (Pi - prev_Pi).reshape(-1)
            Yg = (G - prev_G).reshape(-1)
            denom = torch.dot(Yg, S) + 1e-12
            num = torch.dot(S, S)
            if denom.abs() > 0:
                alpha = float(num / denom.clamp(min=1e-12))

                # clamp step to a safe range
                alpha = float(np.clip(alpha, 1e-5, 5e-1))

        prev_Pi = Pi.clone()
        prev_G  = G.clone()

        Pi.add_(G, alpha=-alpha)          # gradient step
        sinkhorn_projection(Pi, iters=sinkhorn_iters)

        fval = float(f.detach().cpu())
        if prev_f is not None and abs(fval - prev_f) <= tol*(1.0+prev_f):
            break
        prev_f = fval

    return Pi


# ------------------- rounding -------------------

def round_to_permutation(Pi):
    Pi_np = Pi.detach().cpu().numpy()
    r, c = linear_sum_assignment(-Pi_np)
    return c


# ------------------- trial & worker -------------------

def single_trial(n, d, rho, r, device, dtype, args, seed=None, p_edge=0.5):
    gen = torch.Generator(device=device)
    if seed is None:
        seed = int(time.time()*1e6) % (2**31-1)
    gen.manual_seed(seed)

    A, A2, X, Y, p_true = sample_correlated_graphs(
        n, d, rho, r, device, dtype, gen, standardize=True, p_edge=p_edge
    )

    if (args.lam_edge + args.lam_feat) > 0:
        s = args.lam_edge + args.lam_feat
        lam_edge = args.lam_edge / s
        lam_feat = args.lam_feat / s
    else:
        lam_edge, lam_feat = args.lam_edge, args.lam_feat
    

    Pi = optimize(A, A2, X, Y,
                  max_iter=args.max_iter, step_size=args.step_size,
                  lam_edge=args.lam_edge, lam_feat=args.lam_feat, reg_lambda=args.reg_lambda,
                  sinkhorn_iters=args.sinkhorn_iters, bb=args.bb)

    col = round_to_permutation(Pi)
    p_true_cpu = p_true.detach().cpu().numpy()
    overlap = float(np.mean(col == p_true_cpu))
    return overlap


def distribute_tasks(rho_list, r_list, reps, ngpus):
    grid = [(rho, r, rep) for rho in rho_list for r in r_list for rep in range(reps)]
    shards = [[] for _ in range(ngpus)]
    for i, item in enumerate(grid):
        shards[i % ngpus].append(item)
    return shards


def worker(rank, device, tasks, args, ret_dict, p_edge, mode=None):
    torch.set_default_dtype(torch_dtype(args.dtype))
    dtype = torch_dtype(args.dtype)
    results = []
    pbar = tqdm(total=len(tasks), position=rank, desc=f"GPU {device}", leave=False)
    for (rho, r, rep) in tasks:
        try:
            ov = single_trial(args.n, args.d, float(rho), float(r),
                              device, dtype, args,
                              seed=(args.seed + 10007*rep + 7919*rank), p_edge=p_edge)
            results.append({"rho": float(rho), "r": float(r), "overlap": ov})
        except Exception as e:
            results.append({"rho": float(rho), "r": float(r), "overlap": np.nan})
            print(f"[Worker {rank}] Error at (rho={rho}, r={r}): {e}")
        pbar.update(1)
    pbar.close()
    ret_dict[rank] = results


# ------------------- plotting -------------------

def plot_heatmap(df, n, d, outdir, name="heatmap", mode="mixed"):
    pivot = df.pivot(index="r", columns="rho", values="overlap").sort_index().sort_index(axis=1)
    plt.figure(figsize=(7,6))
    im = plt.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    plt.colorbar(im, label="Overlap")
    plt.xlabel(r"$\rho$"); plt.ylabel(r"$r$")
    plt.title(f"Overlap Heatmap (n={n}, d={d})")
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_n{n}_d{d}.png")
    plt.savefig(path, dpi=250); plt.close()



# ------------------- main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=str, default="auto",
                    help='e.g., "cuda:0,cuda:1" or "auto" or "cpu"')
    ap.add_argument("--outdir", type=str, default="gpu_qap_outputs")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--rho", type=str, default="0.0:0.95:13,0.97,0.99")
    ap.add_argument("--r",   type=str, default="0.0:0.95:13,0.97,0.99")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--sinkhorn_iters", type=int, default=60)
    ap.add_argument("--step_size", type=float, default=1e-2)
    ap.add_argument("--lam_edge", type=float, default=1.0)
    ap.add_argument("--lam_feat", type=float, default=1.0)
    ap.add_argument("--reg_lambda", type=float, default=0.01, help="L1 regularization weight")
    ap.add_argument("--bb", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # devices
    if args.devices == "auto":
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
    else:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    ngpus = len(devices)

    rho_list = parse_list(args.rho, float)
    r_list   = parse_list(args.r, float)

    print(f"[Info] devices={devices}, n={args.n}, d={args.d}, dtype={args.dtype}, reps={args.reps}")
    print(f"[Info] |rho|={len(rho_list)}, |r|={len(r_list)}, total trials={len(rho_list)*len(r_list)*args.reps}")

    mp.set_start_method("spawn", force=True)

    p_edge = 0.5
    
    outdir_mode = os.path.join(args.outdir)
    os.makedirs(outdir_mode, exist_ok=True)

    print(f"[Run] p_edge={p_edge:.6f}, outdir={outdir_mode}")

    # distribute tasks and spawn worker processes
    shards = distribute_tasks(rho_list, r_list, args.reps, ngpus)
    manager = mp.Manager(); ret = manager.dict()
    procs = []

    t0 = time.time()
    for rank, device in enumerate(devices):
        proc = mp.Process(target=worker, args=(rank, device, shards[rank], args, ret, p_edge, None))
        proc.start(); procs.append(proc)
    for proc in procs: proc.join()

    # gather
    records = []
    for rank in range(len(devices)):
        records.extend(ret.get(rank, []))
    df = pd.DataFrame.from_records(records)

    if df.empty or "rho" not in df.columns:
        print(f"[Warning] No results collected. Check worker errors.")

    df = df.dropna(subset=["overlap"])
    df = df.groupby(["rho","r"], as_index=False)["overlap"].mean().sort_values(["rho","r"])
    csv_path = os.path.join(outdir_mode, f"results_n{args.n}_d{args.d}_{args.dtype}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Done] Saved CSV -> {csv_path}. Elapsed {time.time()-t0:.1f}s")

    plot_heatmap(df, args.n, args.d, outdir_mode, name=f"heatmap")
    print(f"[Done] Plots saved to {outdir_mode}")


if __name__ == "__main__":
    main()
