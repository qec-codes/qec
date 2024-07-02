import scipy.sparse
import numpy as np
import ldpc.protograph
import qec.lifted_hgp
import qec.css


def test_lifted_product():

    a1 = ldpc.protograph.array(
        [
            [(0), (11), (7), (12)],
            [(1), (8), (1), (8)],
            [(11), (0), (4), (8)],
            [(6), (2), (4), (12)],
        ]
    )

    qcode = qec.lifted_hgp.LiftedHypergraphProduct(11, a1, a1)
    for _ in range(1):
        qcode.estimate_min_distance(reduce_logical_basis=True, timeout_seconds=5, p=0.25)
        qcode.test_logical_basis()

        lx, lz = qcode.logical_operator_weights
        print(lx)
        print(lz)
    print(qcode)

    # print()
    # qcode.fix_logical_operators(fix_logical="Z")
    # qcode.test_logical_basis()

    # temp = qcode.lx@qcode.lz.T
    # temp.data = temp.data%2
    # temp.eliminate_zeros()
    # # print(temp)

    # lx, lz = qcode.logical_operator_weights

    # temp = qcode.lx@qcode.lz.T
    # temp.data = temp.data % 2

    # print(temp.toarray())

    # print(lx)
    # print(lz)

    # print()


if __name__ == "__main__":
    test_lifted_product()
    # test_mc_distance_estimation()
