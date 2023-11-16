import scipy.sparse
import numpy as np
import ldpc.protograph
import qec.lifted_hgp

def test_lifted_product():

    a1=ldpc.protograph.array([
            [(0), (11), (7), (12)],
            [(1), (8), (1), (8)],
            [(11), (0), (4), (8)],
            [(6), (2), (4), (12)]])
    
    qcode = qec.lifted_hgp.LiftedHypergraphProduct(13,a1,a1)
    qcode.estimate_min_distance(reduce_logicals=True)
    qcode.test_logical_basis()

    x_weights = []
    z_weights = []
    for i in range(qcode.K):
        x_weights.append(qcode.lx[i].nnz)
        z_weights.append(qcode.lz[i].nnz)

    print(qcode)

    print(x_weights)
    print(z_weights)