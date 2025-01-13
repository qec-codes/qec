import numpy as np
import scipy.sparse


def convert_to_binary_scipy_sparse(
    matrix: np.typing.ArrayLike,
) -> scipy.sparse.csr_matrix:
    """
    Convert and validate a matrix as a sparse binary matrix in CSR format.

    This function checks whether all elements of the input matrix are binary (0 or 1).
    If the input is not already a sparse matrix, it converts it to a CSR (Compressed Sparse Row) matrix.

    Parameters
    ----------
    matrix : array-like
        Input matrix of shape (M, N). Can be a dense array-like or any SciPy sparse matrix format.

    Returns
    -------
    scipy.sparse.csr_matrix
        Binary sparse matrix in CSR format, of shape (M, N).

    Raises
    ------
    ValueError
        If the input matrix has elements outside {0, 1}.
    TypeError
        If the input is not array-like.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse
    >>> mat = np.array([[0, 1], [1, 0]])
    >>> convert_to_binary_scipy_sparse(mat).toarray()
    array([[0, 1],
           [1, 0]])

    >>> mat = csr_matrix([[0, 1], [1, 0]])
    >>> convert_to_binary_scipy_sparse(mat).toarray()
    array([[0, 1],
           [1, 0]])

    >>> mat = np.array([[0, 2], [1, 0]])
    >>> convert_to_binary_scipy_sparse(mat)
    Traceback (most recent call last):
        ...
    ValueError: All elements of the input matrix must be binary.
    """
    if not isinstance(matrix, (np.ndarray, list, scipy.sparse.spmatrix)):
        raise TypeError("Input must be array-like.")

    if not isinstance(matrix, scipy.sparse.spmatrix):
        matrix = scipy.sparse.csr_matrix(matrix, dtype=np.uint8)

    if not matrix.dtype == np.uint8:
        matrix = matrix.astype(np.uint8)

    if not np.all(np.isin(matrix.data, [0, 1])):
        raise ValueError("All elements of the input matrix must be binary.")

    return matrix
