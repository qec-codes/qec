from typing import Union
import numpy as np
import scipy.sparse

def check_binary_matrix_type(matrix: Union[np.ndarray, scipy.sparse.spmatrix]):
    if not isinstance(matrix, (np.ndarray, scipy.sparse.spmatrix)):
        raise TypeError(f"Input must be a binary numpy array or scipy sparse matrix, not {type(matrix)}")
    
    if matrix.dtype not in [np.uint8, np.int8, int]:
        raise TypeError(f"Input matrix must have dtype uint8, int8, or int, not {matrix.dtype}")

    matrix = scipy.sparse.csr_matrix(matrix, dtype=np.uint8) if isinstance(matrix, np.ndarray) else matrix

    if not np.all(np.isin(matrix.data, [1, 0])):
        raise ValueError("Input matrix must be a binary matrix.")
    
    return matrix

