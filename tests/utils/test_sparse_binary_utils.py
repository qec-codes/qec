import numpy as np
import scipy
import pytest

from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse
import scipy.sparse


def convert_to_binary_scipy_sparse():
    with pytest.raises(
        ValueError, match="Input matrix is not a `scipy.sparse.spmatrix` matrix."
    ):
        convert_to_binary_scipy_sparse(np.array([1, 0]))

    with pytest.raises(
        ValueError, match="Input matrix is not a `scipy.sparse.spmatrix` matrix."
    ):
        convert_to_binary_scipy_sparse([1, 0])

    with pytest.raises(
        ValueError, match="All elements of the input matrix must be binary."
    ):
        convert_to_binary_scipy_sparse(scipy.sparse.csr_matrix([[2, 0], [1, 1]]))

    matrix = convert_to_binary_scipy_sparse([[0, 0], [1, 1.0]])

    assert isinstance(matrix, scipy.sparse.csr_matrix)
    assert matrix.dtype == np.uint8
    assert matrix.shape == (2, 2)
    assert matrix.toarray().tolist() == [[0, 0], [1, 1]]
