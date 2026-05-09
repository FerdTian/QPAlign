#!/bin/bash

python sim_er.py --outdir outputs/ER --n 100 --d 16 \
    --rho 0.0:0.9:19,0.95,0.97,0.99 \
    --r   0.0:0.9:19,0.95,0.97,0.99 \
    --reps 3 \
    --dtype float32 \
    --reg_lambda 0.1 \
    --lam_edge 0.1 --lam_feat 0.9 \
    --max_iter 400 --sinkhorn_iters 80 --step_size 1e-4 \
