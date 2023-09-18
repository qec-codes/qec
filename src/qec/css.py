from typing import Union
import numpy as np
from ldpc2 import gf2sparse
import scipy

class CssCode(object):

    def __init__(self, hx: Union[np.ndarray,scipy.sparse.spmatrix], hz: Union[np.ndarray,scipy.sparse.spmatrix]) -> None:
        
        self.lx = None
        self.lz = None

        #check input types
        try:
            assert isinstance(hx, (np.ndarray,scipy.sparse.spmatrix))
        except AssertionError:
            raise TypeError(f"hx must be a numpy array or scipy sparse matrix, not {type(hx)}")
        
        try:
            assert isinstance(hz, (np.ndarray,scipy.sparse.spmatrix))
        except AssertionError:
            raise TypeError(f"hz must be a numpy array or scipy sparse matrix, not {type(hz)}")
        
        #set the number of qubits
        self.N = hx.shape[1]
        
        #check that the number of qubits is the same for both matrices
        try:
            assert self.N == hz.shape[1]
        except AssertionError:
            raise ValueError(f"Input matrices hx and hz must have the same number of columns.\
                             Current column count, hx: {hx.shape[1]}; hz: {hz.shape[1]}")
        
        #convert pcms to sparse representation
        if isinstance(hx, np.ndarray):
            self.hx = scipy.sparse.csr_matrix(hx, dtype=np.uint8)
        else:
            self.hx = hx.astype(np.uint8)

        if isinstance(hz, np.ndarray):
            self.hz = scipy.sparse.csr_matrix(hz, dtype=np.uint8)
        else:
            self.hz = hz.astype(np.uint8)

        #check that the matrices are binary
        try:
            assert np.all(np.isin(self.hz.data, [1, 0]))
        except AssertionError:
            raise ValueError("Input matrix hz must be a binary matrix.")
        
        try:
            assert np.all(np.isin(self.hx.data, [1, 0]))
        except AssertionError:
            raise ValueError("Input matrix hx must be a binary matrix.")

        #check that matrices commute
        try:
            assert not np.any((self.hx @ self.hz.T).data % 2)
        except AssertionError:
            raise ValueError("Input matrices hx and hz do not commute. I.e. they do not satisfy\
                             the requirement that hx@hz.T = 0.")

        #compute a basis of the logical operators
        self.lx, self.lz = self.compute_logical_basis(self.hx, self.hz)
        
        #caculate the dimension of the code using the Rank-Nullity Theorem
        self.K = self.lx.shape[0]

        #
        assert self.K == self.lz.shape[0]

        self.d = np.nan

    def compute_logical_basis(self, hx: Union[np.ndarray,scipy.sparse.spmatrix], hz: Union[np.ndarray,scipy.sparse.spmatrix]):
    
        kernel_hx = gf2sparse.kernel(hx)
        rank_hx = kernel_hx.shape[1] - kernel_hx.shape[0]
        kernel_hz = gf2sparse.kernel(hz)
        rank_hz = kernel_hx.shape[1] - kernel_hz.shape[0]
    
        logical_stack = scipy.sparse.hstack([hz.T,kernel_hx.T])
        plu_z = gf2sparse.PluDecomposition(logical_stack)
        kernel_rows= plu_z.pivots[rank_hz:] - rank_hz
        lz = kernel_hx[kernel_rows]

        logical_stack = scipy.sparse.hstack([hx.T,kernel_hz.T])
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
        return f"CSS Code: [[N={self.N}, K={self.K}, dmin={self.d}]]"




        