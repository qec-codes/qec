import numpy as np
import scipy
import scipy.sparse
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse


def GF4_to_binary(GF4_matrix: np.typing.ArrayLike) -> scipy.sparse.csr_matrix:
    """
    Convert a matrix over GF4 (elements {0,1,2,3}) to a binary sparse matrix in CSR format.

    Each entry (row i, column j) is mapped as follows:
      - 0 => no 1's (row has [0, 0])
      - 1 => one 1 in column 2*j ([1, 0])
      - 2 => two 1's in columns 2*j and 2*j + 1 ([1, 1])
      - 3 => one 1 in column 2*j + 1 ([0, 1])

    Parameters
    ----------
    GF4_matrix : ArrayLike
        Input matrix of shape (M, N) containing only elements from {0, 1, 2, 3}.
        Can be a dense array-like or any SciPy sparse matrix format.

    Returns
    -------
    scipy.sparse.csr_matrix
        Binary sparse matrix in CSR format, of shape (M, 2*N).

    Raises
    ------
    ValueError
        If the input matrix has elements outside {0, 1, 2, 3}.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import scipy.sparse.csr_matrix
    >>> mat = np.array([[0, 1],
    ...                 [2, 3]])
    >>> GF4_to_binary(mat).toarray()
    array([[0, 1, 1, 0],
           [1, 1, 0, 1]], dtype=uint8)
    """
    if scipy.sparse.issparse(GF4_matrix):
        mat_coo = GF4_matrix.tocoo(copy=False)

        if not np.all(np.isin(mat_coo.data, [1, 2, 3])):
            raise ValueError(
                "Input matrix must contain only elements from GF4: {0, 1, 2, 3}"
            )

        row_ids = []
        col_ids = []
        rows, cols = mat_coo.shape

        for r, c, val in zip(mat_coo.row, mat_coo.col, mat_coo.data):
            if val == 1:
                row_ids.append(r)
                col_ids.append(c)
            elif val == 2:
                row_ids.extend([r, r])
                col_ids.extend([c, cols + c])
            elif val == 3:
                row_ids.append(r)
                col_ids.append(cols + c)

        data = np.ones(len(row_ids), dtype=np.uint8)
        return scipy.sparse.csr_matrix(
            (data, (row_ids, col_ids)), shape=(rows, 2 * cols)
        )

    GF4_matrix = np.asanyarray(GF4_matrix, dtype=int)
    if not np.all(np.isin(GF4_matrix, [0, 1, 2, 3])):
        raise ValueError(
            "Input matrix must contain only elements from GF4: {0, 1, 2, 3}"
        )

    row_ids = []
    col_ids = []
    rows, cols = GF4_matrix.shape

    for i in range(rows):
        for j in range(cols):
            val = GF4_matrix[i, j]
            if val == 1:
                row_ids.append(i)
                col_ids.append(j)
            elif val == 2:
                row_ids.extend([i, i])
                col_ids.extend([j, j + cols])
            elif val == 3:
                row_ids.append(i)
                col_ids.append(j + cols)

    data = np.ones(len(row_ids), dtype=np.uint8)
    return scipy.sparse.csr_matrix((data, (row_ids, col_ids)), shape=(rows, 2 * cols))


def pauli_str_to_binary_pcm(
    pauli_strings: np.typing.ArrayLike,
) -> scipy.sparse.csr_matrix:
    """
    Convert an (M x 1) array of Pauli strings, where each string has length N, corresponding to the number of physical qubits, into a binary parity-check matrix (PCM) with dimensions (M x 2*N).

    The mapping for each qubit j in the string is:
      - 'I' => (0|0)
      - 'X' => (1|0)
      - 'Z' => (0|1)
      - 'Y' => (1|1)
    where the first element (a),  in (a|b) is at column j and the second element (b) is at column j + N.

    Parameters
    ----------
    pauli_strings : ArrayLike
        Array of shape (M, 1), where each element is a string of Pauli operators
        ('I', 'X', 'Y', 'Z'). Can be dense or any SciPy sparse matrix format with
        an object/string dtype.

    Returns
    -------
    scipy.sparse.csr_matrix
        Binary parity-check matrix of shape (M, 2*N) in CSR format, where M is the number of stabilisers and
        N is the number of physical qubits.
    Raises
    ------
    ValueError
        If any character in the Pauli strings is not one of {'I', 'X', 'Y', 'Z'}.

    Examples
    --------
    >>> import numpy as np
    >>> paulis = np.array([["XIZ"], ["YYI"]], dtype=object)
    >>> pcm = pauli_str_to_binary_pcm(paulis)
    >>> pcm.toarray()
    array([[1, 0, 0, 0, 0, 1],
           [1, 1, 0, 1, 1, 0]], dtype=uint8)
    """

    if scipy.sparse.issparse(pauli_strings):
        if pauli_strings.dtype == object:
            mat_coo = pauli_strings.tocoo(copy=False)
            dense = np.full(pauli_strings.shape, "I", dtype=str)
            for r, c, val in zip(mat_coo.row, mat_coo.col, mat_coo.data):
                dense[r, c] = val
            pauli_strings = dense
        else:
            pauli_strings = pauli_strings.toarray()

    pauli_strings = np.asanyarray(pauli_strings, dtype=str)

    if pauli_strings.size == 0:
        return scipy.sparse.csr_matrix((0, 0))

    row_ids = []
    col_ids = []

    m_stabilisers = pauli_strings.shape[0]
    n_qubits = len(pauli_strings[0, 0])

    for i, string in enumerate(pauli_strings):
        if len(string[0]) != n_qubits:
            raise ValueError("The Pauli strings do not have equal length.")
        for j, char in enumerate(string[0]):
            if char == "I":
                continue
            elif char == "X":
                row_ids.append(i)
                col_ids.append(j)
            elif char == "Z":
                row_ids.append(i)
                col_ids.append(j + n_qubits)
            elif char == "Y":
                row_ids.extend([i, i])
                col_ids.extend([j, j + n_qubits])
            else:
                raise ValueError(f"Invalid Pauli character '{char}' encountered.")

    data = np.ones(len(row_ids), dtype=np.uint8)

    return scipy.sparse.csr_matrix(
        (data, (row_ids, col_ids)), shape=(m_stabilisers, 2 * n_qubits), dtype=np.uint8
    )


def binary_pcm_to_pauli_str(binary_pcm: np.typing.ArrayLike) -> np.ndarray:
    """
    Convert a binary (M x 2*N) PCM corresponding to M stabilisers acting on N physical qubits,
    back into an array (M x 1) of Pauli strings that have length N.

    For each qubit j, columns (j | j + N) of the PCM encode:
      - (0|0) => 'I'
      - (1|0) => 'X'
      - (0|1) => 'Z'
      - (1|1) => 'Y'

    Parameters
    ----------
    binary_pcm : ArrayLike
        Binary matrix of shape (M, 2*N), in dense or any SciPy sparse matrix format.

    Returns
    -------
    np.ndarray
        Array of shape (M, 1), where each element is a string of Pauli operators with length N.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import scipy.sparse.csr_matrix
    >>> pcm = np.array([[1, 0, 0, 0, 0, 1],
    ...                 [1, 1, 0, 1, 1, 0]], dtype=np.uint8)
    >>> pauli_str_to_return = binary_pcm_to_pauli_str(pcm)
    >>> pauli_str_to_return
    array([['XIZ'],
           ['YYI']], dtype='<U3')
    """
    if scipy.sparse.issparse(binary_pcm):
        binary_pcm = binary_pcm.toarray()

    binary_pcm = np.asanyarray(binary_pcm, dtype=int)
    n_rows, n_cols = binary_pcm.shape
    n_qubits = n_cols // 2
    pauli_strings = [""] * n_rows

    for i in range(n_rows):
        row = binary_pcm[i]
        x_bits = row[:n_qubits]
        z_bits = row[n_qubits:]
        for x_bit, z_bit in zip(x_bits, z_bits):
            if x_bit == 0 and z_bit == 0:
                pauli_strings[i] += "I"
            elif x_bit == 1 and z_bit == 0:
                pauli_strings[i] += "X"
            elif x_bit == 0 and z_bit == 1:
                pauli_strings[i] += "Z"
            else:
                pauli_strings[i] += "Y"

    return np.array(pauli_strings, dtype=str).reshape(-1, 1)


def symplectic_product(
    a: np.typing.ArrayLike, b: np.typing.ArrayLike
) -> scipy.sparse.csr_matrix:
    """
    Compute the symplectic product of two binary matrices in CSR format.

    The input matrices (A,B) are first converted to binary sparse format (modulo 2)
    and then partitioned into `x` and `z` components, where x and z have the same shape:

        A = (a_x|a_z)
        B = (b_x|b_z)

    Then the symplectic product is computed as: (a_x * b_z^T + a_z * b_x^T) mod 2.

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

    Notes
    -----
    This function is particularly useful for calculating commutation between Pauli operators,
    where a result of 0 indicates commuting operators, and 1 indicates anti-commuting operators.

    Examples
    --------
    >>> import numpy as np
    >>> from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse
    >>> a_data = np.array([[1, 0, 0, 1],
    ...                    [0, 1, 1, 0],
    ...                    [1, 1, 0, 0]], dtype=int)
    >>> b_data = np.array([[0, 1, 1, 0],
    ...                    [1, 0, 0, 1],
    ...                    [0, 1, 1, 0]], dtype=int)
    >>> # Compute symplectic product
    >>> sp = symplectic_product(a_data, b_data)
    >>> sp.toarray()
    array([[0, 0, 0],
           [0, 0, 0],
           [1, 1, 1]], dtype=int8)
    """

    a = convert_to_binary_scipy_sparse(a)
    b = convert_to_binary_scipy_sparse(b)

    assert (a.shape[1] == b.shape[1]), "Input matrices must have the same number of columns."
    assert a.shape[1] % 2 == 0, "Input matrices must have an even number of columns."

    n = a.shape[1] // 2

    ax = a[:, :n]
    az = a[:, n:]
    bx = b[:, :n]
    bz = b[:, n:]

    sp = ax @ bz.T + az @ bx.T
    sp.data %= 2

    return sp


def check_binary_pauli_matrices_commute(
    mat1: scipy.sparse.spmatrix, mat2: scipy.sparse.spmatrix
) -> bool:
    """
    Check if two binary Pauli matrices commute.
    """
    symplectic_product_result = symplectic_product(mat1, mat2)
    symplectic_product_result.eliminate_zeros()
    return not np.any(symplectic_product_result.data)


def binary_pauli_hamming_weight(
    mat: scipy.sparse.spmatrix,
) -> np.ndarray:
    """
    Compute the row-wise Hamming weight of a binary Pauli matrix.

    A binary Pauli matrix has 2*n columns, where the first n columns encode
    the X part and the second n columns encode the Z part. The Hamming weight
    for each row is the number of qubits that are acted upon by a non-identity
    Pauli operator (X, Y, or Z). In other words, for each row, we count the
    number of columns where either the X part or the Z part has a 1.

    Parameters
    ----------
    mat : scipy.sparse.spmatrix
        A binary Pauli matrix with an even number of columns (2*n). Each entry
        must be 0 or 1, indicating whether the row has an X or Z component
        for the corresponding qubit.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of length `mat.shape[0]`, where the i-th entry is
        the Hamming weight of the i-th row in `mat`.

    Raises
    ------
    AssertionError
        If the matrix does not have an even number of columns.

    Notes
    -----
    Internally, this function:
      1. Splits the matrix into the X and Z parts.
      2. Computes an elementwise OR of the X and Z parts.
      3. Counts the non-zero entries per row (i.e., columns where the row has a 1).

    Because the bitwise OR operator `|` is not directly supported for CSR
    matrices, we achieve the OR operation by adding the two sparse matrices
    and capping the sum at 1. Any entries with a value >= 1 in the sum
    are set to 1, which corresponds to OR semantics for binary data.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> # Create a 2-row matrix, each row having 6 columns (for n=3 qubits).
    >>> # Row 0: columns [0,2] are set -> X on qubits 0 and 2.
    >>> # Row 1: columns [3,4,5] are set -> Z on qubit 1, Y on qubit 2.
    >>> mat_data = np.array([[1,0,1,0,0,0],
    ...                      [0,0,0,1,1,1]], dtype=np.uint8)
    >>> mat_sparse = csr_matrix(mat_data)
    >>> binary_pauli_hamming_weight(mat_sparse)
    array([2, 2], dtype=int32)
    """
    assert mat.shape[1] % 2 == 0, "Input matrix must have an even number of columns."

    # Determine the number of qubits from the total columns.
    n = mat.shape[1] // 2

    # Partition the matrix into X and Z parts.
    x_part = mat[:, :n]
    z_part = mat[:, n:]

    # We want a bitwise OR. Since CSR matrices do not support a direct OR,
    # we add and then cap at 1: (x_part + z_part >= 1) -> 1
    xz_or = x_part.copy()
    xz_or += z_part
    # Clip values greater than 1 to 1.
    xz_or.data[xz_or.data > 1] = 1

    # The row-wise Hamming weight is the number of non-zero columns in each row.
    return xz_or.getnnz(axis=1)
