import numpy as np
from scipy.sparse import csr_matrix


def load_codetables_de_matrix(filename):
    """
    Loads a matrix from a file and converts it into a scipy sparse CSR matrix.

    Parameters:
    - filename: str, path to the file containing the matrix.

    Returns:
    - csr_matrix representing the input matrix.
    """
    row_indices = []
    col_indices = []
    data = []

    with open(filename, "r") as file:
        for row, line in enumerate(file):
            # Remove leading/trailing whitespace and square brackets
            line = line.strip().strip("[]")
            # Replace '|' with space and split into elements
            elements = line.replace("|", " ").split()

            # Enumerate over elements to find indices of '1's
            ones = [col for col, val in enumerate(elements) if val == "1"]

            # Extend the lists with the current row and column indices
            row_indices.extend([row] * len(ones))
            col_indices.extend(ones)
            data.extend([1] * len(ones))

    if not row_indices:
        # Handle the case of an empty matrix
        return csr_matrix((0, 0), dtype=int)

    num_rows = row + 1  # Since row indexing starts at 0
    num_cols = len(elements)

    # Create the CSR matrix
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=int
    )

    return sparse_matrix
