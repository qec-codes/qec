import numpy as np
import scipy.sparse
import ldpc.codes
import qec.css
import qec.hgp
import qec.codes

def test_steane_code():

    H = ldpc.codes.hamming_code(3)
    qcode = qec.css.CssCode(H, H)
    qcode.test_logical_basis()

    d = qcode.estimate_min_distance(10)
    assert d == 3

def test_surface_code():

    qcode = qec.codes.SurfaceCode(3, 3)
    qcode.estimate_min_distance()

def test_hgp_distance_estimation():

    seedH = np.loadtxt("tests/pcms/16_4_6.txt").astype(np.uint8)

    # seedH = ldpc.codes.hamming_code(2)
    qcode = qec.hgp.HyperGraphProductCode(seedH, seedH)
    # qcode.test_logical_basis()
    # qcode.temp()
    # print(ldpc.mod2.rank(qcode.lz))

    qcode = qec.css.CssCode(qcode.hx, qcode.hz)
    qcode.test_logical_basis()
    print(ldpc.mod2.rank(qcode.lx))

    print()


    kerHx = ldpc.mod2.kernel(qcode.hx)
    logical_stack = scipy.sparse.vstack([qcode.hz, kerHx]).tocsr()
    rankHz = ldpc.mod2.rank(qcode.hz)
    pivots = ldpc.mod2.pivot_rows(logical_stack)
    log_pivots = pivots[rankHz:] 
    subset = log_pivots - qcode.hx.shape[0]
    print(log_pivots)
    assert len(subset) == ldpc.mod2.rank(kerHx[subset,:]) == len(ldpc.mod2.pivot_rows(kerHx[subset,:])) == 16
    logicals = logical_stack[log_pivots,:]
    assert len(subset) == ldpc.mod2.rank(logicals) == len(ldpc.mod2.pivot_rows(logicals)) == 16

    kerHz = ldpc.mod2.kernel(qcode.hz)
    # print(ldpc.mod2.rank(kerHz))
    assert kerHz.shape[0] == ldpc.mod2.rank(kerHz)
    subset = [0,149,55,66]
    assert len(subset) == ldpc.mod2.rank(kerHz[subset,:]) == len(ldpc.mod2.pivot_rows(kerHz[subset,:]))
    logical_stack = scipy.sparse.vstack([qcode.hx, kerHz]).tocsr()
    rankHx = ldpc.mod2.rank(qcode.hx)
    pivots = ldpc.mod2.pivot_rows(logical_stack)
    log_pivots = pivots[rankHx:] 
    subset = log_pivots - qcode.hz.shape[0]
    print(log_pivots)
    assert len(subset) == ldpc.mod2.rank(kerHz[subset,:]) == len(ldpc.mod2.pivot_rows(kerHz[subset,:])) == 16
    logicals = logical_stack[log_pivots,:]
    assert len(subset) == ldpc.mod2.rank(logicals) == len(ldpc.mod2.pivot_rows(logicals)) == 16




if __name__ == "__main__":

    # test_steane_code()
    # test_surface_code()
    test_hgp_distance_estimation()







