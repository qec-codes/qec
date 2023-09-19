import numpy as np
import scipy
from typing import Union
from ldpc2 import gf2sparse

from qec.css import CssCode
import qec.util

class HyperGraphProductCode(CssCode):

    def __init__(self, h1: Union[np.ndarray,scipy.sparse.spmatrix], h2: Union[np.ndarray,scipy.sparse.spmatrix], name = None):

        if name is None:
            self.name = "Hypergraph Product"

        #check input types
        self.h1 = qec.util.check_binary_matrix_type(h1)
        self.h2 = qec.util.check_binary_matrix_type(h2)


        self.n1 = h1.shape[1]
        self.n2 = h2.shape[1]

        self.m1 = h1.shape[0]
        self.m2 = h2.shape[0]

        Id = lambda n: scipy.sparse.eye(n, dtype=np.uint8)

        hx = scipy.sparse.hstack([ scipy.sparse.kron(h1,Id(self.n2)) , scipy.sparse.kron(Id(self.m1), h2.T)])
        hz = scipy.sparse.hstack([scipy.sparse.kron(Id(self.n1), h2), scipy.sparse.kron(h1.T, Id(self.m2))])

        CssCode.__init__(self, hx, hz, name = name)

    def compute_logical_basis(self):

        ker_h1 = gf2sparse.kernel(self.h1)
        ker_h2 = gf2sparse.kernel(self.h2)
        ker_h1T = gf2sparse.kernel(self.h1.T)
        ker_h2T = gf2sparse.kernel(self.h2.T)

        row_comp_h1 = gf2sparse.row_complement_basis(self.h1)
        row_comp_h2 = gf2sparse.row_complement_basis(self.h2)
        row_comp_h1T = gf2sparse.row_complement_basis(self.h1.T)
        row_comp_h2T = gf2sparse.row_complement_basis(self.h2.T)

        temp = scipy.sparse.kron(ker_h1, row_comp_h2)
        lz1 = scipy.sparse.hstack([temp, scipy.sparse.csr_matrix((temp.shape[0], self.m1*self.m2), dtype=np.uint8)])

        temp = scipy.sparse.kron(row_comp_h1T, ker_h2T)
        lz2 = scipy.sparse.hstack([scipy.sparse.csr_matrix((temp.shape[0], self.n1*self.n2), dtype=np.uint8), temp])

        lz = scipy.sparse.vstack([lz1, lz2])


        temp = scipy.sparse.kron(row_comp_h1, ker_h2)
        lx1 = scipy.sparse.hstack([temp, scipy.sparse.csr_matrix((temp.shape[0], self.m1*self.m2), dtype=np.uint8)])

        temp = scipy.sparse.kron(ker_h1T, row_comp_h2T)
        lx2 = scipy.sparse.hstack([scipy.sparse.csr_matrix((temp.shape[0], self.n1*self.n2), dtype=np.uint8), temp])

        lx = scipy.sparse.vstack([lx1, lx2])

        return (lx, lz)

        