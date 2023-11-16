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
    for _ in range(1):
        qcode.estimate_min_distance(reduce_logicals=True)
        qcode.test_logical_basis()

        lx, lz = qcode.logical_operator_weights
        print(lx)
        print(lz)

if __name__ == "__main__":
    test_lifted_product()