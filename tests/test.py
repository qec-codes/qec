import numpy as np
from ldpc.codes import rep_code, hamming_code, ring_code
from ldpc import mod2
from qec.css import CssCode
from qec.hgp import HyperGraphProductCode
from qec.codes import *
from qec.stab_code import StabCode
from qec.lifted_hgp import LiftedHypergraphProduct
import qec.protograph as pt
import scipy.sparse

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

# a= mod2.row_complement_basis(np.array([[1,1,1,1]])).toarray()

# print(a)


#

# print()

stabs = np.array([["X", "X", "X", "X"], ["Z", "Z", "Z", "Z"]])

code = StabCode(pauli_stabs=stabs)

print("Logical basis:")
print(code.logical_basis.toarray())

code.test_logical_basis()

print(code)

# print()

# exit(22)

# from qec.protograph import RingOfCirculantsF2, permutation_matrix

# r = RingOfCirculantsF2([0,1,1])
# c = r.to_binary(10)
# c= permutation_matrix(10,1)

# print(type(c+c))

# import qec.protograph as pt

# pa = pt.array([[(0,1)]])

# pa = pt.identity(10)
# # print(pa)

# # print(pa.to_binary(100))

proto_a=pt.array([
        [(0), (11), (7)],
        [(1), (8), (1)],
        [(11), (0), (4)]])

H = proto_a.to_binary(5).toarray()

from qec.mod2 import nullspace, row_span

print(row_span(nullspace(H)))

# proto_a=pt.array([[(0), (1)]])

qcode=LiftedHypergraphProduct(lift_parameter=20,a=proto_a)

print(qcode)

print(qcode.lx.nnz/np.prod(qcode.lx.shape))

print(qcode)

import scipy.sparse
scipy.sparse.save_npz("test.npz", qcode.hz)

from qec.lifted_hgp_3d import LiftedHGP3D

qcode = LiftedHGP3D(proto_a,proto_a,proto_a,30)
# qcode.test()

print(qcode.N, qcode.K)


# scipy.sparse.save_npz("3d_ldpc_hx.npz", scipy.sparse.csr_matrix(qcode.hx))
# scipy.sparse.save_npz("3d_ldpc_hz.npz", scipy.sparse.csr_matrix(qcode.hz))
# scipy.sparse.save_npz("3d_ldpc_lx.npz", scipy.sparse.csr_matrix(qcode.lx))
# scipy.sparse.save_npz("3d_ldpc_lz.npz", scipy.sparse.csr_matrix(qcode.lz))
# scipy.sparse.save_npz("3d_ldpc_mx.npz", scipy.sparse.csr_matrix(qcode.mx))


# print(rep_code(9).toarray().__repr__())