import numpy as np

nside = 64
lmax = 3 * nside - 1
delta_ell = 10
bpw_edges = delta_ell * np.arange((lmax + 1) // delta_ell + 1)
bpw_edges[-1] = lmax + 1
bpw_edges = bpw_edges.astype(int)
np.savez("bpw_edges.npz", bpw_edges=bpw_edges)
