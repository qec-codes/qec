import numpy as np
import scipy.sparse

np.set_printoptions(
    edgeitems=3,
    infstr="inf",
    linewidth=75,
    nanstr="nan",
    precision=8,
    suppress=False,
    threshold=1000,
    formatter=None,
)

H = np.loadtxt("src/qec/pcms/hgp_seed_matrices/24_6_10.txt").astype(np.uint8)
print(repr(H))

print(scipy.sparse.csr_matrix(H).__repr__())
