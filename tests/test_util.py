import pytest
import numpy as np
import scipy.sparse
from qec.util import convert_to_sparse


def test_convert_to_sparse():
    a = np.array([[1, 0, 1], [1, 1, 1]])
    b = convert_to_sparse(a)
    assert isinstance(b, scipy.sparse.spmatrix)

    with pytest.raises(ValueError):
        a = np.array([[1, 0, 88], [1, 1, 1]])
        b = convert_to_sparse(a)

    with pytest.raises(TypeError):
        a = np.array([[1, 0, 88.0], [1, 1, 1]])
        b = convert_to_sparse(a)

    with pytest.raises(TypeError):
        a = np.array([[1, 0, 1.0], [1, 1, 1]])
        b = convert_to_sparse(a)

    with pytest.raises(TypeError):
        a = [[1, 0, 1.0], [1, 1, 1]]
        b = convert_to_sparse(a)
