import pytest
import scipy.sparse
import numpy as np
from qec.protograph import permutation_matrix


def test_output_shape():
    n = 4
    t = 1
    mat = permutation_matrix(n, t)
    assert mat.shape == (n, n)


def test_is_permutation_matrix():
    n = 5
    t = 2
    mat = permutation_matrix(n, t)
    row_sums = mat.sum(axis=1).A1  # Convert to 1-D array
    col_sums = mat.sum(axis=0).A1  # Convert to 1-D array
    assert all(row_sums == 1)
    assert all(col_sums == 1)


def test_shifts():
    n = 3
    t = 1
    expected_output = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)

    mat = permutation_matrix(n, t).toarray()
    assert np.array_equal(mat, expected_output)


def test_large_shifts():
    n = 3
    t = 4  # Equivalent to a single right shift for a 3x3 matrix
    expected_output = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)

    mat = permutation_matrix(n, t).toarray()
    assert np.array_equal(mat, expected_output)
