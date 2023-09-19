from typing import Union
import numpy as np
from ldpc2 import gf2sparse
import scipy
import qec.util

class CssCode(object):

    def __init__(self, hx: Union[np.ndarray,scipy.sparse.spmatrix], hz: Union[np.ndarray,scipy.sparse.spmatrix], name: str = None):
        
        if name is None:
            self.name = "CSS"
        else:
            self.name = name

        self.lx = None
        self.lz = None

        #check matrix input and convert to sparse representation
        self.hx = qec.util.check_binary_matrix_type(hx)
        self.hz = qec.util.check_binary_matrix_type(hz)
        
        #set the number of qubits
        self.N = self.hx.shape[1]
        
        #check that the number of qubits is the same for both matrices
        try:
            assert self.N == self.hz.shape[1]
        except AssertionError:
            raise ValueError(f"Input matrices hx and hz must have the same number of columns.\
                             Current column count, hx: {hx.shape[1]}; hz: {hz.shape[1]}")
        
        #check that matrices commute
        try:
            assert not np.any((self.hx @ self.hz.T).data % 2)
        except AssertionError:
            raise ValueError("Input matrices hx and hz do not commute. I.e. they do not satisfy\
                             the requirement that hx@hz.T = 0.")

        #compute a basis of the logical operators
        self.lx, self.lz = self.compute_logical_basis()
        
        #caculate the dimension of the code using the Rank-Nullity Theorem
        self.K = self.lx.shape[0]

        #
        assert self.K == self.lz.shape[0]

        self.d = np.nan

    def compute_logical_basis(self):
    
        kernel_hx = gf2sparse.kernel(self.hx)
        rank_hx = kernel_hx.shape[1] - kernel_hx.shape[0]
        kernel_hz = gf2sparse.kernel(self.hz)
        rank_hz = kernel_hx.shape[1] - kernel_hz.shape[0]
    
        logical_stack = scipy.sparse.hstack([self.hz.T,kernel_hx.T])
        plu_z = gf2sparse.PluDecomposition(logical_stack)
        kernel_rows= plu_z.pivots[rank_hz:] - rank_hz
        lz = kernel_hx[kernel_rows]

        logical_stack = scipy.sparse.hstack([self.hx.T,kernel_hz.T])
        plu_x = gf2sparse.PluDecomposition(logical_stack)
        kernel_rows= plu_x.pivots[rank_hx:] - rank_hx
        lx = kernel_hz[kernel_rows]

        return (lx, lz)
    
    def test_logical_basis(self):

        if self.lx is None or self.lz is None:
            self.lx, self.lz = self.compute_logical_basis(self.hx, self.hz)

        assert not np.any((self.lx @ self.hz.T).data % 2)
        test = self.lx @ self.lz.T
        test.data = test.data % 2
        test_plu = gf2sparse.PluDecomposition(test)
        assert test_plu.rank == self.K

        assert not np.any((self.lz @ self.hx.T).data % 2)
        test = self.lz @ self.lx.T
        test.data = test.data % 2
        test_plu = gf2sparse.PluDecomposition(test)
        assert test_plu.rank == self.K

    
    def __str__(self):
        return f"{self.name} Code: [[N={self.N}, K={self.K}, dmin={self.d}]]"




        