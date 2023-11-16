from typing import Union, Tuple
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
    # Check input type
    if not isinstance(matrix, (np.ndarray, scipy.sparse.spmatrix)):
        raise TypeError(
            f"Input must be a binary numpy array or scipy sparse matrix, not {type(matrix)}"
        )

    # Check dtype
    if matrix.dtype not in [np.uint8, np.int8, int]:
        raise TypeError(
            f"Input matrix must have dtype uint8, int8, or int, not {matrix.dtype}"
        )

    # Convert numpy array to sparse matrix
    matrix = (
        scipy.sparse.csr_matrix(matrix, dtype=np.uint8)
        if isinstance(matrix, np.ndarray)
        else matrix
    )

    # Eliminate any zero elements
    matrix.eliminate_zeros()

    # Check if the matrix is binary
    if isinstance(matrix, scipy.sparse.spmatrix):
        if not np.all(np.isin(matrix.data, [1, 0])):
            raise ValueError("Input matrix must be a binary matrix.")
        else:
            NotImplementedError("Not implemented for numpy arrays yet.")

    return matrix


def get_row_col_data_indices_binary_nonzero(
    sparse_mat: scipy.sparse.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the row and column indices, along with the data array, of non-zero elements from a CSR sparse matrix.
    The function checks if the data are binary (0 or 1).

    Parameters
    ----------
    sparse_mat : scipy.sparse.csr_matrix
        The input sparse matrix in CSR format.

    Returns
    -------
    row_indices : np.ndarray
        Array of row indices corresponding to the non-zero elements.
    col_indices : np.ndarray
        Array of column indices corresponding to the non-zero elements.
    data : np.ndarray
        Array of non-zero elements in the matrix.

    Raises
    ------
    ValueError
        If the input matrix is not in CSR format or if the data is not binary.

    Example
    -------
    >>> from scipy.sparse import scipy.sparse.csr_matrix
    >>> import numpy as np
    >>> data = np.array([1, 0, 1])
    >>> indices = np.array([0, 1, 3])
    >>> indptr = np.array([0, 2, 3])
    >>> mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=(2, 4))
    >>> get_row_col_data_indices_binary_nonzero(mat)
    (array([0, 1]), array([0, 3]), array([1, 1]))
    """

    # Check if the input matrix is in CSR format
    if not isinstance(sparse_mat, scipy.sparse.csr_matrix):
        raise ValueError("Input matrix must be in CSR format.")

    # Check if all data are binary (0 or 1)
    unique_data = np.unique(sparse_mat.data)
    if not np.all(np.isin(unique_data, [0, 1])):
        raise ValueError("Data in the matrix must be binary (0 or 1).")

    # Get data array directly from the 'data' attribute
    data = sparse_mat.data

    # Filter out zero elements
    nonzero_indices = np.where(data != 0)[0]

    # Get corresponding column indices
    col_indices = sparse_mat.indices[nonzero_indices]

    # Preallocate an array to hold the row indices
    row_indices = np.empty_like(nonzero_indices)

    # Loop through the 'indptr' to efficiently compute row indices
    for i in range(len(sparse_mat.indptr) - 1):
        mask = (sparse_mat.indptr[i] <= nonzero_indices) & (
            nonzero_indices < sparse_mat.indptr[i + 1]
        )
        row_indices[mask] = i

    return row_indices, col_indices, data[nonzero_indices]
