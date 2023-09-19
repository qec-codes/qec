import numpy as np
from ldpc2.codes import rep_code, hamming_code, ring_code
from ldpc2 import gf2sparse
from qec.css import CssCode
from qec.hgp import HyperGraphProductCode
from qec.codes import *

qcode = FourTwoTwoCode()
print(qcode)


code = hamming_code(3)

qcode = CssCode(code, code)


print(qcode)
qcode.test_logical_basis()

print()

for i in range(3,6):
    qcode = SurfaceCode(i)
    print(qcode)

    qcode = ToricCode(i)
    print(qcode)

    qcode = SteaneCode(i)
    print(qcode)

# a= gf2sparse.row_complement_basis(np.array([[1,1,1,1]])).toarray()

# print(a)