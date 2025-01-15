import pytest
import logging
import numpy as np

from qec.stabilizer_code.hgp_code import HypergraphProductCode
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse


three_repetition = np.array([[1, 1, 0],
                             [0, 1, 1]])

def test_hgp_initilastion():
    temp_code = HypergraphProductCode(
        seed_matrix_1 = three_repetition, seed_matrix_2 = three_repetition 
    )

    assert (
        temp_code.physical_qubit_count == 13
    ), f"Expected N=13, but got N={temp_code.physical_qubit_count}"
    assert (
        temp_code.logical_qubit_count == 1
    ), f"Expected K=1, but got K={temp_code.logical_qubit_count}"

    # Uncomment the following line if you want to test the distance (d)
