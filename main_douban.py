import json
from argparse import ArgumentParser
from douban import load_data
import os, json, ast, math, re
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
from scipy.optimize import linear_sum_assignment

from main_acmdblp import optimize

def round_to_permutation(Pi: torch.Tensor) -> np.ndarray:
    Pi_np = Pi if isinstance(Pi, np.ndarray) else Pi.detach().cpu().numpy()
    r, c = linear_sum_assignment(-Pi_np)
    return c

def _as_numpy(x):
    import numpy as np
    # 既支持 torch.Tensor 也支持 numpy/scipy
    try:
        return x.detach().cpu().numpy()     # torch.Tensor
    except AttributeError:
        return np.asarray(x)    

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu', help='use GPU')
    parser.add_argument("--device", type=str, default="auto", help='"auto", "cuda:0", or "cpu"')
    # parser.add_argument("--mapping", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--limit_n", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    parser.add_argument("--standardize_feat", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)
    parser.add_argument("--max_iter", type=int, default=400)
    parser.add_argument("--sinkhorn_iters", type=int, default=80)
    parser.add_argument("--step_size", type=float, default=1e-2)
    parser.add_argument("--lam_edge", type=float, default=1.0)
    parser.add_argument("--lam_feat", type=float, default=1.0)
    parser.add_argument("--reg_lambda", type=float, default=0.01,
                    help="Regularization strength for the term tr(P^T(J-P))")
    parser.add_argument("--bb", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)
    parser.add_argument("--progress", type=lambda s: s.lower() in ["true","1","yes","y"], default=True)
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")] if s else [], default=[123,445,980,1301,2692], help="Random seeds for multiple runs, e.g. 1,2,3")
    parser.add_argument("--workers", type=int, default=0, help="CPU cores used for math kernels; 0=auto (all available / SLURM allocation)")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)

    
    if device == "cpu":
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
    else:
        print(f"Using {device}")

    graph1, graph2 = "online", "offline"
    adj1, adj2, x1, x2, gnd, _ = load_data(f"datasets/douban/Douban.mat", graph1, graph2, True)
    print(f"Graph 1: {adj1.shape}, {x1.shape if x1 is not None else None}")
    print(f"Graph 2: {adj2.shape}, {x2.shape if x2 is not None else None}")
    print(f"Ground truth: {gnd.shape}")

    gnd = gnd.detach().cpu().numpy()

    n = gnd.shape[0]

    adj1, adj2, x1, x2 = adj1.float(), adj2.float(), x1.float(), x2.float()
    adj1, adj2, x1, x2 = adj1.to(device), adj2.to(device), x1.to(device), x2.to(device)

    
    if (args.lam_edge + args.lam_feat) > 0:
        s = args.lam_edge + args.lam_feat
        lam_edge = args.lam_edge / s
        lam_feat = args.lam_feat / s
    else:
        lam_edge, lam_feat = args.lam_edge, args.lam_feat

    n_runs = max(1, len(args.seeds))
    overlaps = []
    for run_id in range(n_runs):
        seed = args.seeds[run_id] if run_id < len(args.seeds) else 42
        # seed = 17 + 107 * run_id
        print(f"[Run {run_id+1}/{n_runs}], seed={seed}")
        S = optimize(adj1, adj2, x1, x2,
                    max_iter=args.max_iter, step_size=args.step_size,
                    lam_edge=lam_edge, lam_feat=lam_feat, reg_lambda=args.reg_lambda,
                    sinkhorn_iters=args.sinkhorn_iters, bb=args.bb, seed=seed,
                    progress=args.progress)
        # —— 舍入与准确率 ——
        col = round_to_permutation(S)
        correct_matches = (col == gnd)
        ov = float(correct_matches.sum() / n)
        overlaps.append(ov)
        print(f" → overlap in run {run_id+1}: {ov:.4f}")
    print(f"[Result] Overlaps={overlaps}")
    overlap = sum(overlaps) / n_runs 
    print(f"[Result] n={n}, overlap={overlap:.4f}")
    # print prameters
    print(f"[Params] lam_edge={lam_edge:.4f}, lam_feat={lam_feat:.4f}, reg_lambda={args.reg_lambda:.4f}")
    print(f"[Params] max_iter={args.max_iter}, sinkhorn_iters={args.sinkhorn_iters}, step_size={args.step_size}, bb={args.bb}")