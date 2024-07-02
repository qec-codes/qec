import pytest
import numpy as np
import scipy.sparse

import ldpc.codes
import ldpc.mod2
import qec.hgp
import qec.codes


def test_hgp_16_4_6():
    seedH = np.loadtxt("tests/pcms/16_4_6.txt").astype(np.uint8)
    qcode = qec.hgp.HyperGraphProductCode(seedH, seedH)
    d = qcode.estimate_min_distance(reduce_logical_basis=True, timeout_seconds=3)
    qcode.test_logical_basis()
    print(qcode)
    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)
    assert d == 6







    seedH = np.loadtxt("tests/pcms/20_5_8.txt").astype(np.uint8)
    qcode = qec.hgp.HyperGraphProductCode(seedH, seedH)
    d = qcode.estimate_min_distance(reduce_logical_basis=True, timeout_seconds=3)
    qcode.test_logical_basis()
    print(qcode)
    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)
    assert d == 8

    seedH = np.loadtxt("tests/pcms/24_6_10.txt").astype(np.uint8)
    qcode = qec.hgp.HyperGraphProductCode(seedH, seedH)
    d = qcode.estimate_min_distance(reduce_logical_basis=True, timeout_seconds=3)
    qcode.test_logical_basis()
    print(qcode)
    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)
    assert d == 10


    # print()
    # qcode.fix_logical_operators(fix_logical="X")
    # qcode.test_logical_basis()

    # temp = qcode.lx@qcode.lz.T
    # temp.data = temp.data%2
    # temp.eliminate_zeros()
    # print(temp)

    # lx, lz = qcode.logical_operator_weights

    # # print((qcode.lx@qcode.lz.T).toarray())

    # print(lx)
    # print(lz)

    # print()

    # exit(22)


def test_saved_hgp_codes():
    qcode = qec.codes.HGP6Code()
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)

    qcode.save_npz(path="tests/pcms/HGP/",code_label=f"HGP_{qcode.N}_{qcode.K}_{qcode.d}")


    qcode = qec.codes.HGP8Code()
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)

    qcode.save_npz(path="tests/pcms/HGP/",code_label=f"HGP_{qcode.N}_{qcode.K}_{qcode.d}")


    qcode = qec.codes.HGP10Code()
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)

    qcode.save_npz(path="tests/pcms/HGP/",code_label=f"HGP_{qcode.N}_{qcode.K}_{qcode.d}")

def test_hamming_hgp():

    H = ldpc.codes.hamming_code(3)
    qcode = qec.hgp.HyperGraphProductCode(H, H,name = "HammingHGP")
    print(qcode)
    d = qcode.estimate_min_distance(reduce_logical_basis=True,timeout_seconds=5)
    qcode.test_logical_basis()
    print(qcode)
    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)
    print()
    qcode.fix_logical_operators(fix_logical="Z")
    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)

    # import galois

    # logicalgf4 = galois.GF(4).Zeros((2*qcode.K,qcode.N))
    # for i in range(qcode.K):
    #     for j in range(qcode.N):
    #         if(qcode.lx[i,j]==1):
    #             logicalgf4[i,j] = 1
    #         else:
    #             logicalgf4[i,j] = 0

    #     for i in range(qcode.K):
    #         for j in range(qcode.N):
    #             if(qcode.lz[i,j]==1):
    #                 logicalgf4[i+qcode.K,j] = 3
    #             else:
    #                 logicalgf4[i+qcode.K,j] = 0

    # # for row in logicalgf4:
    # #     print(' '.join(str(elem) for elem in row))

    # _,_,U = logicalgf4.plu_decompose()

    # for row in U:
    #     print(' '.join(str(elem) for elem in row))

    # lx2 = np.zeros((2*qcode.K,qcode.N),dtype=np.uint8)
    # lz2 = np.zeros((2*qcode.K,qcode.N),dtype=np.uint8)

    # ## turn back into pair of logical matrices
    # for i in range(2*qcode.K):
    #     if np.any(U[i]==1):
    #         for j in range(qcode.N):
    #             if U[i,j] not in [0,1]:
    #                 raise ValueError("Not a valid value")
    #             if(U[i,j]==1):
    #                 lx2[i,j] = 1

    #     elif np.any(U[i]==3):
    #         for j in range(qcode.N):
    #             if U[i,j] not in [0,3]:
    #                 raise ValueError("Not a valid value")
    #             if(U[i,j]==3):
    #                 lz2[i,j] = 1

    # print()
    # for row in lx2:
    #     print(' '.join(str(elem) for elem in row))

    # test = lx2@lz2.T %2
    # print()
    # for row in test:
    #     print(' '.join(str(elem) for elem in row))     
            

if __name__ == "__main__":
    # test_hamming_hgp()
    test_hgp_16_4_6()
