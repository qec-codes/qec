import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, issparse


def GF4_to_binary(GF4_matrix: ArrayLike) -> csr_matrix:
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
    csr_matrix
        Binary sparse matrix in CSR format, of shape (M, 2*N).

    Raises
    ------
    ValueError
        If the input matrix has elements outside {0, 1, 2, 3}.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> mat = np.array([[0, 1],
    ...                 [2, 3]])
    >>> GF4_to_binary(mat).toarray()
    array([[0, 1, 1, 0],
           [1, 1, 0, 1]], dtype=uint8)
    """
    if issparse(GF4_matrix):
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
        return csr_matrix((data, (row_ids, col_ids)), shape=(rows, 2 * cols))

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
    return csr_matrix((data, (row_ids, col_ids)), shape=(rows, 2 * cols))


def pauli_str_to_binary_pcm(pauli_strings: ArrayLike) -> csr_matrix:
    """
    Convert an array of Pauli strings (N x 1) into a binary parity-check matrix (PCM).

    The mapping for each qubit j in the string is:
      - 'I' => no bits
      - 'X' => column j
      - 'Z' => column j + n_qubits
      - 'Y' => columns j and j + n_qubits

    Parameters
    ----------
    pauli_strings : ArrayLike
        Array of shape (N, 1), where each element is a string of Pauli operators
        ('I', 'X', 'Y', 'Z'). Can be dense or any SciPy sparse matrix format with
        an object/string dtype.

    Returns
    -------
    csr_matrix
        Binary parity-check matrix of shape (N, 2*n_qubits) in CSR format, where
        n_qubits is the length of each Pauli string.

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
    array([[1, 0, 0, 1, 0, 1],
           [1, 1, 0, 0, 1, 0]], dtype=uint8)
    """
    if issparse(pauli_strings):
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
        return csr_matrix((0, 0))

    row_ids = []
    col_ids = []
    n_rows = pauli_strings.shape[0]
    n_qubits = len(pauli_strings[0, 0])

    for i in range(n_rows):
        row_string = pauli_strings[i, 0]
        for j, char in enumerate(row_string):
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
    return csr_matrix(
        (data, (row_ids, col_ids)), shape=(n_rows, 2 * n_qubits), dtype=np.uint8
    )


def binary_pcm_to_pauli_str(binary_pcm: ArrayLike) -> np.ndarray:
    """
    Convert a binary PCM (with 2*n_qubits columns) back into an array of Pauli strings (N x 1).

    For each qubit j, columns [j, j + n_qubits] encode:
      - (0, 0) => 'I'
      - (1, 0) => 'X'
      - (0, 1) => 'Z'
      - (1, 1) => 'Y'

    Parameters
    ----------
    binary_pcm : ArrayLike
        Binary matrix of shape (N, 2*n_qubits) containing 0/1 values, in dense
        or any SciPy sparse matrix format.

    Returns
    -------
    np.ndarray
        Array of shape (N, 1), where each element is a string of Pauli operators.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> pcm = np.array([[1, 0, 0, 1, 0, 1],
    ...                 [1, 1, 0, 0, 1, 0]], dtype=np.uint8)
    >>> pauli_str_to_return = binary_pcm_to_pauli_str(pcm)
    >>> pauli_str_to_return
    array([['XIZ'],
           ['YYI']], dtype='<U3')
    """
    if issparse(binary_pcm):
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
