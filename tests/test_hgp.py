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
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)
    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)
    assert d == 6

    seedH = np.loadtxt("tests/pcms/20_5_8.txt").astype(np.uint8)
    qcode = qec.hgp.HyperGraphProductCode(seedH, seedH)
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)
    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)
    assert d == 8

    seedH = np.loadtxt("tests/pcms/24_6_10.txt").astype(np.uint8)
    qcode = qec.hgp.HyperGraphProductCode(seedH, seedH)
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)
    lx, lz = qcode.logical_operator_weights
    print(lx)
    print(lz)

    assert d == 10


def test_saved_hgp_codes():
    qcode = qec.codes.HGP6Code()
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)

    qcode = qec.codes.HGP8Code()
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)

    qcode = qec.codes.HGP10Code()
    d = qcode.estimate_min_distance(reduce_logical_basis=True)
    qcode.test_logical_basis()
    print(qcode)

    qcode.save_npz(path="src/qec/pcms/hgp_codes/")


if __name__ == "__main__":
    test_hgp_16_4_6()
