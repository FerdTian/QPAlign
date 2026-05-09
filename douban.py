import scipy.io as sio
import numpy as np
import torch
import os
from scipy import sparse

def _to_ndarray_or_csr(x):
    # Convert the input to np.ndarray or sparse CSR, keeping it numeric.
    if sparse.issparse(x):
        return x.tocsr()
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def load_data(file_path, G1_name, G2_name, use_attr):
    """
    Load dataset from .mat file, then restrict both graphs and attributes
    to the nodes mentioned in gnd only, preserving original relative order.
    Return gnd as permutation vector (column indices).
    """
    assert os.path.exists(file_path), f"{file_path} does not exist"

    print(f"Loading {file_path}...")
    data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # adjacent matrix
    adj_mat1, adj_mat2 = data[G1_name], data[G2_name]
    adj_mat1 = _to_ndarray_or_csr(adj_mat1)
    adj_mat2 = _to_ndarray_or_csr(adj_mat2)

    # attributes
    if use_attr:
        x1 = data.get(f"{G1_name}_node_feat", None)
        x2 = data.get(f"{G2_name}_node_feat", None)
        if x1 is not None:
            x1 = _to_ndarray_or_csr(x1)
            if sparse.issparse(x1):  
                pass
        if x2 is not None:
            x2 = _to_ndarray_or_csr(x2)
            if sparse.issparse(x2):
                pass
    else:
        x1, x2 = None, None

    # Read the ground truth (assumed to be N x 2, 1-based indexing) and convert it to 0-based.
    gnd_raw = np.asarray(data['gnd'])
    if gnd_raw.ndim == 1:
        gnd_raw = gnd_raw.reshape(-1, 1)
    if gnd_raw.shape[1] >= 2:
        pairs = gnd_raw[:, :2].astype(np.int64) - 1
    else:
        raise ValueError(f"gnd should contain two columns (G1 index, G2 index), actual shape: {gnd_raw.shape}")

    # Reading H
    H = data.get('H', None)
    if H is not None:
        H = _to_ndarray_or_csr(H)

    # Extract the common part based on gnd (only remove vertices not appearing in gnd; preserve the original order)
    idx1_all = pairs[:, 0]
    idx2_all = pairs[:, 1]

    idx1_keep = np.unique(idx1_all)  
    idx2_keep = np.unique(idx2_all)

    def _slice_adj(A, row_idx, col_idx):
        if sparse.issparse(A):
            return A[row_idx][:, col_idx].tocsr()
        else:
            return A[np.ix_(row_idx, col_idx)]

    adj_mat1 = _slice_adj(adj_mat1, idx1_keep, idx1_keep)
    adj_mat2 = _slice_adj(adj_mat2, idx2_keep, idx2_keep)

    def _slice_attr(X, row_idx):
        if X is None:
            return None
        if sparse.issparse(X):
            return X[row_idx, :].tocsr()
        else:
            X = np.asarray(X)
            return X[row_idx, :]

    x1 = _slice_attr(x1, idx1_keep)
    x2 = _slice_attr(x2, idx2_keep)

    if H is not None:
        try:
            H = _slice_adj(H, idx1_keep, idx2_keep)
        except Exception:
            pass

    # Generate the permutation vector gnd_perm (in column index form)
    pos2 = {orig_j: new_j for new_j, orig_j in enumerate(idx2_keep)}
    map12 = {}
    for i_orig, j_orig in pairs:
        if i_orig not in map12:
            map12[i_orig] = j_orig

    gnd_perm = np.empty(len(idx1_keep), dtype=np.int64)
    for new_i, i_orig in enumerate(idx1_keep):
        j_orig = map12.get(i_orig, None)
        if j_orig is None:
            raise ValueError(f"The G1 node {i_orig} does not have a corresponding G2 index in gnd (it should have been filtered out)")
        j_new = pos2[j_orig]
        gnd_perm[new_i] = j_new

    def _to_torch_int(A):
        if sparse.issparse(A):
            A = A.toarray()
        return torch.from_numpy(np.asarray(A)).int()

    adj_mat1 = _to_torch_int(adj_mat1)
    adj_mat2 = _to_torch_int(adj_mat2)

    def _to_torch_float(X):
        if X is None:
            return None
        if sparse.issparse(X):
            X = X.toarray()
        return torch.from_numpy(np.asarray(X)).to(torch.float64)

    x1 = _to_torch_float(x1)
    x2 = _to_torch_float(x2)

    gnd = torch.from_numpy(gnd_perm).long()

    if H is not None:
        H = _to_torch_int(H)

    return adj_mat1, adj_mat2, x1, x2, gnd, H
