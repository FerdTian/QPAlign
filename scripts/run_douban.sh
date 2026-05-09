#!/bin/bash



python main_douban.py \
  --outdir outputs/douban \
  --workers 112 \
  --progress true \
  --dtype float64 --max_iter 1000 --sinkhorn_iters 200 --step_size 5e-3 \
  --lam_edge 0.2 --lam_feat 0.8 --reg_lambda 0.0 --bb true --seeds 123,445,980,1301,2692 \
