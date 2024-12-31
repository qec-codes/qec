import numpy as np
import scipy
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse

def symplectic_product(a: np.typing.ArrayLike, b: np.typing.ArrayLike) -> scipy.sparse.csr_matrix:
    """
    Compute the symplectic product of two binary matrices in CSR format.

    The input matrices are first converted to binary sparse format (modulo 2)
    and then partitioned into `x` and `z` components. The symplectic product
    is computed as (a_x * b_z^T + a_z * b_x^T) mod 2. This function is
    particularly useful for calculating commutation between Pauli operators,
    where a result of 0 indicates commuting operators, and 1 indicates
    anti-commuting operators.

    Parameters
    ----------
    a : array_like
        A 2D array-like object with shape (M, 2N), which will be converted to
        a binary sparse matrix (mod 2).
    b : array_like
        A 2D array-like object with shape (M, 2N), which will be converted to
        a binary sparse matrix (mod 2). Must have the same shape as `a`.

    Returns
    -------
    scipy.sparse.csr_matrix
        The symplectic product of the two input matrices, stored in CSR format.

    Raises
    ------
    AssertionError
        If the shapes of `a` and `b` do not match.
    AssertionError
        If the number of columns of `a` (and `b`) is not even.

    Examples
    --------
    >>> import numpy as np
    >>> from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse
    >>> # Create dummy binary data
    >>> a_data = np.array([[1, 0, 0, 1],
    ...                    [0, 1, 1, 0],
    ...                    [1, 1, 0, 0]], dtype=int)
    >>> b_data = np.array([[0, 1, 1, 0],
    ...                    [1, 0, 0, 1],
    ...                    [0, 1, 1, 0]], dtype=int)
    >>> # Compute symplectic product
    >>> sp = symplectic_product(a_data, b_data)
    >>> sp.toarray()
    array([[1, 0, 0],
           [0, 1, 0],
           [1, 0, 0]], dtype=int8)
    """
    # Convert the input arrays to binary sparse format (mod 2).
    a = convert_to_binary_scipy_sparse(a)
    b = convert_to_binary_scipy_sparse(b)

    # Ensure both matrices have the same shape.
    assert a.shape[1] == b.shape[1], "Input matrices must have the same number of columns."
    # Ensure the number of columns is even (we split them into x and z parts).
    assert a.shape[1] % 2 == 0, "Input matrices must have an even number of columns."

    # Determine the half-size (number of x/z columns).
    n = a.shape[1] // 2

    # Partition each matrix into x and z components.
    ax = a[:, :n]
    az = a[:, n:]
    bx = b[:, :n]
    bz = b[:, n:]

    # Compute partial products (mod 2).
    sp = ax @ bz.T + az @ bx.T
    sp.data %= 2

    return sp

def check_binary_pauli_matrices_commute(mat1: scipy.sparse.spmatrix, mat2: scipy.sparse.spmatrix)->bool:
    """
    Check if two binary Pauli matrices commute.
    """
    symplectic_product_result = symplectic_product(mat1, mat2)
    symplectic_product_result.eliminate_zeros()
    return not np.any(symplectic_product_result.data)