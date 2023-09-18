import numpy as np
from ldpc2.codes import rep_code, hamming_code
from qec.css import CssCode

qcode = CssCode(np.array([[1,1,1,1]]), np.array([[1,1,1,1]]))

code = hamming_code(3)

qcode = CssCode(code, code)


print(qcode)
qcode.test_logical_basis()
