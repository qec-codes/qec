import numpy as np
import scipy 
from typing import Union
import ldpc.mod2 

from qec.stabilizer_code.css_code import CSSCode
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse

class HypergraphProductCode(CSSCode):
    def __init__(
        self,
        seed_matrix_1 : Union[np.ndarray, scipy.sparse.spmatrix],
        seed_matrix_2 : Union[np.ndarray, scipy.sparse.spmatrix],
        name : str = None
    ):

        self.name = name if name else "Hypergraph product code"

        if not all(isinstance(seed_m, (np.ndarray, scipy.sparse.spmatrix)) for seed_m in (seed_matrix_1, seed_matrix_2)):
            raise TypeError("The seed matrices must be either numpy arrays or scipy sparse matrices.")

        self.seed_matrix_1 = convert_to_binary_scipy_sparse(seed_matrix_1)
        self.seed_matrix_2 = convert_to_binary_scipy_sparse(seed_matrix_2)

        # maybe move the below to a private _construct_stabilizer_matrices function?
        # --------------------------------------------------------------------------
        self._n1 = seed_matrix_1.shape[1]
        self._n2 = seed_matrix_2.shape[1]

        self._m1 = seed_matrix_1.shape[0]
        self._m2 = seed_matrix_2.shape[0]

        x_left = scipy.sparse.kron(seed_matrix_1, scipy.sparse.eye(self._n2))
        x_right = scipy.sparse.kron(scipy.sparse.eye(self._m1), seed_matrix_2.T)
        self.x_stabilizer_matrix = scipy.sparse.hstack([x_left, x_right])

        z_left = scipy.sparse.kron(scipy.sparse.eye(self._n1), seed_matrix_2)
        z_right = scipy.sparse.kron(seed_matrix_1.T, scipy.sparse.eye(self._m2))
        self.z_stabilizer_matrix = scipy.sparse.hstack([z_left, z_right])
        # --------------------------------------------------------------------------

        super().__init__(self.x_stabilizer_matrix, self.z_stabilizer_matrix, self.name)
    

    def compute_exact_code_distance(self):
    
        rank_seed_m1 = ldpc.mod2.rank(self.seed_matrix_1)
        rank_seed_m2 = ldpc.mod2.rank(self.seed_matrix_2)
        
        if self.seed_matrix_1.shape[1] != rank_seed_m1:
            self.d1 = ldpc.mod2.compute_exact_code_distance(self.seed_matrix_1)
        else:
            self.d1 = np.inf

        if self.seed_matrix_2.shape[1] != rank_seed_m2:
            self.d2 = ldpc.mod2.compute_exact_code_distance(self.seed_matrix_2)
        else:
            self.d2 = np.inf
        
        # note: rank(A) = rank(A^T):
        if self.seed_matrix_1.shape[0] != rank_seed_m1:
            self.d1T = ldpc.mod2.compute_exact_code_distance(self.seed_matrix_1)
        else:
            self.d1T = np.inf

        if self.seed_matrix_2.shape[0] != rank_seed_m2:
            self.d2T = ldpc.mod2.compute_exact_code_distance(self.seed_matrix_1)
        else:
            self.d2T = np.inf

        self.code_distance = min(self.d1, self.d2 , self.d1T, self.d2T)

        return self.code_distance
       


    def estimate_min_distance(self):
        NotImplemented


    def compute_logical_basis(self):

        ker_h1 = ldpc.mod2.kernel(self.seed_matrix_1)
        ker_h2 = ldpc.mod2.kernel(self.seed_matrix_2)
        ker_h1T = ldpc.mod2.kernel(self.seed_matrix_1.T)
        ker_h2T = ldpc.mod2.kernel(self.seed_matrix_2.T)

        row_comp_h1 = ldpc.mod2.row_complement_basis(self.seed_matrix_1)
        row_comp_h2 = ldpc.mod2.row_complement_basis(self.seed_matrix_2)
        row_comp_h1T = ldpc.mod2.row_complement_basis(self.seed_matrix_1.T)
        row_comp_h2T = ldpc.mod2.row_complement_basis(self.seed_matrix_2.T)

        temp = scipy.sparse.kron(ker_h1, row_comp_h2)
        lz1 = scipy.sparse.hstack([temp, scipy.sparse.csr_matrix((temp.shape[0], self._m1*self._m2), dtype=np.uint8)])

        temp = scipy.sparse.kron(row_comp_h1T, ker_h2T)
        lz2 = scipy.sparse.hstack([scipy.sparse.csr_matrix((temp.shape[0], self._n1*self._n2), dtype=np.uint8), temp])

        self.z_logical_operator_basis = scipy.sparse.vstack([lz1, lz2])


        temp = scipy.sparse.kron(row_comp_h1, ker_h2)
        lx1 = scipy.sparse.hstack([temp, scipy.sparse.csr_matrix((temp.shape[0], self._m1*self._m2), dtype=np.uint8)])

        temp = scipy.sparse.kron(ker_h1T, row_comp_h2T)
        lx2 = scipy.sparse.hstack([scipy.sparse.csr_matrix((temp.shape[0], self._n1*self._n2), dtype=np.uint8), temp])

        self.x_logical_operator_basis= scipy.sparse.vstack([lx1, lx2])
        
        # Follows the way it is done in CSSCode -> move it into __init__? 
        #---------------------------------------------------------------- 
        self.logical_qubit_count = self.x_logical_operator_basis.shape[0]
        #---------------------------------------------------------------- 

        return (self.x_logical_operator_basis, self.z_logical_operator_basis)


    def __str__(self):
         return f"{self.name} Hypergraphproduct Code: [[N={self.physical_qubit_count}, K={self.logical_qubit_count}, dx<={self.x_code_distance}, dz<={self.z_code_distance}]]"
