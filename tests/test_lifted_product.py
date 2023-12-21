import scipy.sparse
import numpy as np
import ldpc.protograph
import qec.lifted_hgp
import qec.css

def test_lifted_product():

    a1=ldpc.protograph.array([
            [(0), (11), (7), (12)],
            [(1), (8), (1), (8)],
            [(11), (0), (4), (8)],
            [(6), (2), (4), (12)]])
    
    qcode = qec.lifted_hgp.LiftedHypergraphProduct(13,a1,a1.T)
    for _ in range(1):
        qcode.estimate_min_distance(reduce_logical_basis=True)
        qcode.test_logical_basis()

        lx, lz = qcode.logical_operator_weights
        print(lx)
        print(lz)
    print(qcode)

    print()
    de = qec.css.CssCodeDistanceEstimator(qcode)
    de.monte_carlo_basis_reduction(1, silent = False)
    qcode.lx = de.lx
    qcode.lz = de.lz
    qcode.test_logical_basis()


    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)

    



if __name__ == "__main__":
    test_lifted_product()
    # test_mc_distance_estimation()