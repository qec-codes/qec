import pytest
import numpy as np
import scipy.sparse
from qec.util import validate_matrix_input, validate_binary_matrix, convert_to_sparse
from qec.stabiliser_code import StabiliserCode

# Tests for the constructor of StabiliserCode class
def test_constructor_with_pauli_stabs():
    pauli_stabs = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    code = StabiliserCode(pauli_stabs=pauli_stabs)
    assert code.h is not None

def test_constructor_with_h():
    h = scipy.sparse.csr_matrix([[1, 0, 0, 1], [0, 1, 1, 0]])
    code = StabiliserCode(h=h)
    assert code.h is not None

def test_constructor_with_neither_pauli_stabs_nor_h():
    with pytest.raises(ValueError):
        StabiliserCode()

def test_constructor_with_invalid_pauli_stabs_dtype():
    pauli_stabs = np.array([[1.5, 0], [0, 1.5]])
    with pytest.raises(TypeError):
        StabiliserCode(pauli_stabs=pauli_stabs)

def test_constructor_with_invalid_pauli_stabs_value():
    pauli_stabs = np.array([[4, 0], [0, 4]], dtype=np.uint8)
    with pytest.raises(ValueError):
        StabiliserCode(pauli_stabs=pauli_stabs)

def test_constructor_with_invalid_h_shape():
    h = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError):
        StabiliserCode(h=h)

def test_init_with_integer_dtype():
    stabs = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=int)
    code = StabiliserCode(pauli_stabs=stabs)
    assert code.h is not None

def test_init_with_string_dtype():
    stabs = np.array([["I", "X", "Y", "Z"], ["X", "I", "Z", "Y"]], dtype=str)
    code = StabiliserCode(pauli_stabs=stabs)
    assert code.h is not None

def test_init_with_h_matrix():
    h = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
    code = StabiliserCode(h=h)
    assert code.h is not None

def test_invalid_input_type():
    with pytest.raises(TypeError):
        StabiliserCode(pauli_stabs="invalid")

def test_invalid_dtype_for_pauli_stabs():
    with pytest.raises(TypeError):
        stabs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
        StabiliserCode(pauli_stabs=stabs)

def test_invalid_integer_value_for_pauli_stabs():
    with pytest.raises(ValueError):
        stabs = np.array([[0, 4], [5, 6]], dtype=int)
        StabiliserCode(pauli_stabs=stabs)

def test_invalid_string_value_for_pauli_stabs():
    with pytest.raises(ValueError):
        stabs = np.array([["I", "X"], ["A", "B"]], dtype=str)
        StabiliserCode(pauli_stabs=stabs)

def test_h_matrix_with_odd_number_of_columns():
    with pytest.raises(ValueError):
        h = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        StabiliserCode(h=h)

def test_no_input_provided():
    with pytest.raises(ValueError):
        StabiliserCode()


def check_sparse_matrix(sparse_matrix, expected_dense_matrix):
    dense_matrix = sparse_matrix.toarray()
    print(dense_matrix)
    assert np.array_equal(dense_matrix, expected_dense_matrix)

def test_h_with_pauli_integers():
    pauli_stabs = np.array([[0, 1, 2], [1, 2, 3]])
    sc = StabiliserCode(pauli_stabs=pauli_stabs)
    expected_h = np.array([
        [0, 1, 1, 0, 0 ,1],
        [1, 1, 0, 0, 1, 1]
    ])
    check_sparse_matrix(sc.h, expected_h)

def test_h_with_pauli_strings():
    pauli_stabs = np.array([["I", "X", "Y"], ["X", "Y", "Z"]])
    sc = StabiliserCode(pauli_stabs=pauli_stabs)
    expected_h = np.array([
        [0, 1, 1, 0, 0 ,1],
        [1, 1, 0, 0, 1, 1]
    ])
    check_sparse_matrix(sc.h, expected_h)

def test_h_with_h_matrix():
    h_matrix = np.array([
        [0, 1, 1, 0, 0 ,1],
        [1, 1, 0, 0, 1, 1]
    ])
    sc = StabiliserCode(h=h_matrix)
    expected_h = np.array([
        [0, 1, 1, 0, 0 ,1],
        [1, 1, 0, 0, 1, 1]
    ])
    check_sparse_matrix(sc.h, expected_h)



