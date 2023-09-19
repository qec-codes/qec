import numpy as np
from ldpc2.codes import rep_code, hamming_code, ring_code
from ldpc2 import gf2sparse
from qec.css import CssCode
from qec.hgp import HyperGraphProductCode
from qec.codes import *
from qec.stabiliser_code import StabiliserCode

qcode = FourTwoTwoCode()
print(qcode)


code = hamming_code(3)

qcode = CssCode(code, code)


print(qcode)
qcode.test_logical_basis()

print()

for i in range(3, 6):
    qcode = SurfaceCode(i)
    print(qcode)

    qcode = ToricCode(i)
    print(qcode)

    qcode = SteaneCode(i)
    print(qcode)

# a= gf2sparse.row_complement_basis(np.array([[1,1,1,1]])).toarray()

# print(a)


#

print()

stabs = np.array([["X", "X", "X", "X"], ["Z", "Z", "Z", "Z"]])

code = StabiliserCode(pauli_stabs=stabs)

print(code.logical_basis.toarray())

code.test_logical_basis()

print(code)
