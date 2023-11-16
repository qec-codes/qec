import numpy as np
import scipy.sparse
import ldpc.codes
import qec.css
import qec.hgp
import qec.codes

def test_steane_code():

    H = ldpc.codes.hamming_code(3)
    qcode = qec.css.CssCode(H, H)
    qcode.N == 7
    qcode.K == 1
    assert qcode.lx.shape[0] ==  qcode.lz.shape[0] == 1
    assert qcode.test_logical_basis()

    qcode = qec.codes.SteaneCode(3)

    qcode.N == 7
    qcode.K == 1
    assert qcode.lx.shape[0] ==  qcode.lz.shape[0] == 1
    assert qcode.test_logical_basis()

    d = qcode.estimate_min_distance()
    print(d)


if __name__ == "__main__":

    test_steane_code()
    # test_surface_code()
    # test_hgp_distance_estimation()







