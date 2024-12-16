import numpy as np
from scipy.sparse import csr_matrix


def GF4_to_binary(GF4_matrix : np.ndarray[np.integer]) -> csr_matrix:

    row_ids = []
    col_ids = []

    for i, row in enumerate(GF4_matrix):
        for j, value in enumerate(row):
            if value:
                row_ids.extend([i] if value != 2 else [i] * 2)
                col_ids.extend([j] if value == 1 else ([j + GF4_matrix.shape[1]] if value == 3 else [j, j + GF4_matrix.shape[1]]))

    return csr_matrix(((np.ones(len(row_ids))), (row_ids, col_ids)), shape = (GF4_matrix.shape[0], 2 *  GF4_matrix.shape[1]))


def pauli_str_to_binary_pcm(pauli_strings : np.ndarray[str]) -> csr_matrix:

    row_ids = []
    col_ids = []

    for i, row in enumerate(pauli_strings):
        for j, value in enumerate(list(row[0])):
            if value != "I":
                row_ids.extend([i] if value != "Y" else [i] * 2)
                col_ids.extend([j] if value == "X" else ([j + len(pauli_strings[0][0])] if value == "Z" else [j, j + len(pauli_strings[0][0])]))

    return csr_matrix(((np.ones(len(row_ids))), (row_ids, col_ids)), shape = (pauli_strings.shape[0], 2 * len(pauli_strings[0][0])))

def binary_pcm_to_pauli_str(binary_pcm : np.ndarray[np.integer]) -> np.ndarray[str]:

    pauli_strings = ["" for _ in range(binary_pcm.shape[0])] 
    n = binary_pcm.shape[1] // 2

    for i, row in enumerate(binary_pcm):
        for  x_pauli, z_pauli in zip(row[: n], row[n :]):
            if x_pauli == 0 and z_pauli == 0:
                pauli_strings[i] += "I"
            elif x_pauli == 1 and z_pauli == 0:
                pauli_strings[i] += "X"
            elif x_pauli == 0 and z_pauli == 1:
                pauli_strings[i] += "Z"
            else:
                pauli_strings[i] += "Y"

    return np.array(pauli_strings).reshape(-1, 1)

