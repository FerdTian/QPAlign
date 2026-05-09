#!/bin/bash



python main_acmdblp.py \
  --acm_features datasets/ACM-DBLP/ACM_gaussian_vectors.csv \
  --dblp_features datasets/ACM-DBLP/DBLP2_gaussian_vectors.csv \
  --acm_edges datasets/ACM-DBLP/ACM_graph_gaussian.csv \
  --dblp_edges datasets/ACM-DBLP/DBLP2_graph_gaussian.csv \
  --mapping datasets/ACM-DBLP/DBLP-ACM_perfectMapping.csv \
  --outdir       outputs/ACM-DBLP \
  --dim 256 \
  --workers 112 \
  --progress true \
  --dtype float64 --max_iter 1000 --sinkhorn_iters 200 --step_size 1e-5 \
  --lam_edge 0.4 --lam_feat 0.6 --reg_lambda 0.01 --bb true 
