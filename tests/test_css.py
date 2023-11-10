import ldpc.codes
import qec.css

def test_steane_code():

    H = ldpc.codes.hamming_code(3)
    qcode = qec.css.CssCode(H, H)
    qcode.test_logical_basis()



