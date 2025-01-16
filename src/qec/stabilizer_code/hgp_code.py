import numpy as np
import scipy 
from typing import Union

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
        NotImplemented
    
    def estimate_min_distance(self):
        NotImplemented

    def compute_logical_basis(self):
        NotImplemented

    def __str__(self):
         return f"{self.name} Hypergraphproduct Code: [[N={self.physical_qubit_count}, K={self.logical_qubit_count}, dx<={self.x_code_distance}, dz<={self.z_code_distance}]]"
