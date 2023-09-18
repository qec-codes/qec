from typing import Union
import numpy as np
from ldpc2 import gf2sparse
import scipy

class CssCode(object):

    def __init__(self, hx: Union[np.ndarray,scipy.sparse.spmatrix], hz: Union[np.ndarray,scipy.sparse.spmatrix]) -> None:
        
        try:
            assert isinstance(hx, (np.ndarray,scipy.sparse.spmatrix))
        except AssertionError:
            raise TypeError(f"hx must be a numpy array or scipy sparse matrix, not {type(hx)}")
        
        try:
            assert isinstance(hz, (np.ndarray,scipy.sparse.spmatrix))
        except AssertionError:
            raise TypeError(f"hz must be a numpy array or scipy sparse matrix, not {type(hz)}")
        
        self.N = hx.shape[1]
        
        try:
            assert self.N == hz.shape[1]
        except AssertionError:
            raise ValueError(f"hx and hz must have the same number of columns, not {hx.shape[1]} and {hz.shape[1]}")
        
        #convert pcms to sparse representation
        if isinstance(hx, np.ndarray):
            self.hx = scipy.sparse.csr_matrix(hx)
        else:
            self.hx = hx

        if isinstance(hz, np.ndarray):
            self.hz = scipy.sparse.csr_matrix(hz)
        else:
            self.hz = hz

        self.lx, self.lz = self.compute_logical_basis()
        self.K = self.lx.shape[0]
        assert self.K == self.lz.shape[0]

        self.d = np.nan

    def compute_logical_basis(self):
    
        kernel_hx = gf2sparse.kernel(self.hx)
        self.rank_hx = self.N - kernel_hx.shape[0]
        kernel_hz = gf2sparse.kernel(self.hz)
        self.rank_hz = self.N - kernel_hz.shape[0]
    
        logical_stack = scipy.sparse.hstack([self.hz.T,kernel_hx.T])
        ludz = gf2sparse.PluDecomposition(logical_stack)
        kernel_rows= ludz.pivots[self.rank_hz:] - self.rank_hz
        lz = kernel_hx[kernel_rows]

        logical_stack = scipy.sparse.hstack([self.hx.T,kernel_hz.T])
        ludx = gf2sparse.PluDecomposition(logical_stack)
        kernel_rows= ludx.pivots[self.rank_hx:] - self.rank_hx
        lx = kernel_hz[kernel_rows]

        return lx, lz
    
    def __str__(self):
        return f"CSS Code: [[N={self.N}, K={self.K}, dmin={self.d}]]"




        