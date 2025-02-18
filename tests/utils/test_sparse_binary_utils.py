import numpy as np
import pytest
import scipy.sparse
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse


def test_valid_dense_numpy_array():
    # Test that a valid numpy dense binary matrix is converted correctly.
    dense = np.array([[0, 1], [1, 0]])
    sparse = convert_to_binary_scipy_sparse(dense)
    assert isinstance(sparse, scipy.sparse.csr_matrix)
    assert sparse.dtype == np.uint8
    np.testing.assert_array_equal(sparse.toarray(), dense)


def test_valid_list_input():
    # Test that a list-of-lists input is properly converted.
    matrix_list = [[0, 1, 0], [1, 0, 1]]
    expected = np.array(matrix_list, dtype=np.uint8)
    sparse = convert_to_binary_scipy_sparse(matrix_list)
    np.testing.assert_array_equal(sparse.toarray(), expected)


def test_valid_already_sparse_matrix():
    # Test that an already sparse matrix (even with non-uint8 dtype) is converted.
    csr = scipy.sparse.csr_matrix([[0, 0, 1], [1, 0, 0]], dtype=np.int32)
    sparse = convert_to_binary_scipy_sparse(csr)
    # Ensure type is converted to uint8
    assert sparse.dtype == np.uint8
    expected = csr.astype(np.uint8).toarray()
    np.testing.assert_array_equal(sparse.toarray(), expected)


def test_invalid_elements():
    # Test that a matrix with non-binary element(s) raises ValueError.
    matrix = np.array([[0, 2], [1, 0]])
    with pytest.raises(
        ValueError, match="All elements of the input matrix must be binary."
    ):
        convert_to_binary_scipy_sparse(matrix)


def test_invalid_input_type():
    # Test that a non-array-like input (e.g. an integer) raises TypeError.
    with pytest.raises(TypeError, match="Input must be array-like."):
        convert_to_binary_scipy_sparse(42)
