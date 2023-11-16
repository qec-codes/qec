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
    d= qcode.estimate_min_distance()
    qcode.test_logical_basis()
    assert d == 6

    seedH = np.loadtxt("tests/pcms/20_5_8.txt").astype(np.uint8)
    qcode = qec.hgp.HyperGraphProductCode(seedH, seedH)
    d= qcode.estimate_min_distance()
    qcode.test_logical_basis()
    assert d == 8

    seedH = np.loadtxt("tests/pcms/24_6_10.txt").astype(np.uint8)
    qcode = qec.hgp.HyperGraphProductCode(seedH, seedH)
    d= qcode.estimate_min_distance()
    qcode.test_logical_basis()
    assert d == 10


if __name__ == "__main__":

    test_hgp_16_4_6()