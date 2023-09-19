import numpy as np
from ldpc2.codes import rep_code, hamming_code, ring_code
from ldpc2 import gf2sparse
from qec.css import CssCode
from qec.hgp import HyperGraphProduct

qcode = CssCode(np.array([[1,1,1,1]]), np.array([[1,1,1,1]]))
print(qcode)


code = hamming_code(3)

qcode = CssCode(code, code)


print(qcode)
qcode.test_logical_basis()

for i in range(2,5):
    code = ring_code(i)
    qcode = HyperGraphProduct(code, code)
    print(qcode)

# a= gf2sparse.row_complement_basis(np.array([[1,1,1,1]])).toarray()

# print(a)