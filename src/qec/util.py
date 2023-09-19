from typing import Union
import numpy as np
import scipy.sparse


def convert_to_sparse(matrix: Union[np.ndarray, scipy.sparse.spmatrix]):
    """
    Convert a given binary matrix to scipy's compressed sparse row (CSR) format.

    This function takes either a numpy ndarray or a scipy sparse matrix as input
    and returns a matrix in scipy's CSR format. If the input matrix is already in CSR format,
    the original matrix is returned. The function also checks that the input matrix is binary
    and that its data type is either uint8, int8, or int.

    Parameters
    ----------
    matrix : Union[np.ndarray, scipy.sparse.spmatrix]
        The input matrix to convert to CSR format. Must be a binary matrix with dtype uint8, int8, or int.

    Returns
    -------
    scipy.sparse.spmatrix
        The input matrix converted to CSR format.

    Raises
    ------
    TypeError
        If the input is not a numpy ndarray or scipy sparse matrix.
        If the dtype of the matrix is not uint8, int8, or int.

    ValueError
        If the input matrix is not a binary matrix.

    """

    if not isinstance(matrix, (np.ndarray, scipy.sparse.spmatrix)):
        raise TypeError(
            f"Input must be a binary numpy array or scipy sparse matrix, not {type(matrix)}"
        )

    if matrix.dtype not in [np.uint8, np.int8, int]:
        raise TypeError(
            f"Input matrix must have dtype uint8, int8, or int, not {matrix.dtype}"
        )

    matrix = (
        scipy.sparse.csr_matrix(matrix, dtype=np.uint8)
        if isinstance(matrix, np.ndarray)
        else matrix
    )

    if not np.all(np.isin(matrix.data, [1, 0])):
        raise ValueError("Input matrix must be a binary matrix.")

    return matrix
