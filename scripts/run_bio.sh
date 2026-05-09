#!/bin/bash


python main_biology.py \
  --dataset_dir datasets/cancer \
  --outdir       outputs/biology \
  --slices a \
  --workers 56 \
  --runs 5 \
  --progress true \
  --dtype float64  --max_iter 400 --sinkhorn_iters 80 --step_size 1e-5 \
  --reg_lambda 0.1 --bb true --log_every 0  
