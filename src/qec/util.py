from typing import Union
import numpy as np
import scipy.sparse


def validate_matrix_input(matrix):
    if not isinstance(matrix, (np.ndarray, scipy.sparse.spmatrix)):
        raise TypeError(
            f"Input must be a binary numpy array or scipy sparse matrix, not {type(matrix)}"
        )


def validate_binary_matrix(matrix: Union[np.ndarray, scipy.sparse.spmatrix]):
    if matrix.dtype not in [np.uint8, np.int8, int]:
        raise TypeError(
            f"Input matrix must have dtype uint8, int8, or int, not {matrix.dtype}"
        )

    if isinstance(matrix, scipy.sparse.spmatrix):
        if not np.all(np.isin(matrix.data, [1, 0])):
            raise ValueError("Input matrix must be a binary matrix.")
        else:
            NotImplementedError("Not implemented for numpy arrays yet.")


def convert_to_sparse(matrix: Union[np.ndarray, scipy.sparse.spmatrix]):
    validate_matrix_input(matrix)

    matrix = (
        scipy.sparse.csr_matrix(matrix, dtype=np.uint8)
        if isinstance(matrix, np.ndarray)
        else matrix
    )

    validate_binary_matrix(matrix)

    return matrix
