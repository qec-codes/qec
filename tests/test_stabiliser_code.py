import pytest
import numpy as np
import scipy.sparse
from qec.util import validate_matrix_input, validate_binary_matrix, convert_to_sparse
from qec.stab_code import StabCode


def check_sparse_matrix(sparse_matrix, expected_dense_matrix):
    dense_matrix = sparse_matrix.toarray().astype(np.uint8)
    print(dense_matrix)
    print(expected_dense_matrix)
    assert np.array_equal(dense_matrix, expected_dense_matrix.astype(np.uint8))


def test_constructor_with_check_matrix():
    stabs = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]])
    code = StabCode(h=stabs)
    assert code.h is not None
    check_sparse_matrix(
        code.h, np.array([[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]])
    )


def test_constructor_with_pauli_stabs():
    pauli_stabs = np.array([["X", "X", "X", "X"], ["Z", "Z", "Z", "Z"]])
    code = StabCode(pauli_stabs=pauli_stabs)
    assert code.h is not None
    check_sparse_matrix(
        code.h, np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]])
    )


def test_constructor_with_pauli_stabs_integer():
    pauli_stabs = np.array([[1, 1, 1, 1], [3, 3, 3, 3]])
    code = StabCode(pauli_stabs=pauli_stabs)
    assert code.h is not None
    check_sparse_matrix(
        code.h, np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]])
    )
