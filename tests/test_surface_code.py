import pytest
import numpy as np
import scipy.sparse

import ldpc.codes
import ldpc.mod2
import qec.hgp
import qec.codes


def test_surface_code():
    qcode = qec.codes.SurfaceCode(2, 10)
    qcode.estimate_min_distance()
    qcode.test_logical_basis()

    assert qcode.dx == 10
    assert qcode.dz == 2

    print(qcode)


def test_toric_code():
    qcode = qec.codes.ToricCode(4)
    qcode.estimate_min_distance()
    qcode.test_logical_basis()

    print(qcode)


if __name__ == "__main__":
    test_surface_code()
    test_toric_code()
