import pytest
import numpy as np
import scipy.sparse
from qec.util import validate_matrix_input, validate_binary_matrix, convert_to_sparse


def test_convert_to_sparse():
    a = np.array([[1, 0, 1], [1, 1, 1]])
    b = convert_to_sparse(a)
    assert isinstance(b, scipy.sparse.spmatrix)


def test_convert_to_sparse2():
    with pytest.raises(ValueError):
        a = np.array([[1, 0, 88], [1, 1, 1]])
        b = convert_to_sparse(a)


def convert_to_sparse3():
    with pytest.raises(TypeError):
        a = np.array([[1, 0, 88.0], [1, 1, 1]])
        b = convert_to_sparse(a)


def convert_to_sparse4():
    with pytest.raises(TypeError):
        a = np.array([[1, 0, 1.0], [1, 1, 1]])
        b = convert_to_sparse(a)


def convert_to_sparse5():
    with pytest.raises(TypeError):
        a = [[1, 0, 1.0], [1, 1, 1]]
        b = convert_to_sparse(a)


# Tests for validate_matrix_input
def test_validate_matrix_input_numpy():
    matrix = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    validate_matrix_input(matrix)  # Should not raise any errors


def test_validate_matrix_input_scipy_sparse():
    matrix = scipy.sparse.csr_matrix([[1, 0], [0, 1]])
    validate_matrix_input(matrix)  # Should not raise any errors


def test_validate_matrix_input_invalid_type():
    matrix = "invalid"
    with pytest.raises(TypeError):
        validate_matrix_input(matrix)


# Tests for validate_binary_matrix
def test_validate_binary_matrix_numpy_dtype():
    matrix = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    validate_binary_matrix(matrix)  # Should not raise any errors


def test_validate_binary_matrix_scipy_sparse():
    matrix = scipy.sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.uint8)
    validate_binary_matrix(matrix)  # Should not raise any errors


def test_validate_binary_matrix_invalid_dtype():
    matrix = np.array([[1, 0], [0, 1]], dtype=np.float64)
    with pytest.raises(TypeError):
        validate_binary_matrix(matrix)


def test_validate_binary_matrix_nonbinary_data():
    matrix = scipy.sparse.csr_matrix([[1, 0], [0, 2]], dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_binary_matrix(matrix)
