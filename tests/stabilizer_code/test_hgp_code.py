import pytest
import numpy as np

from qec.stabilizer_code.hgp_code import HypergraphProductCode
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse

three_repetition = np.array([[1, 1, 0],
                             [0, 1, 1]])

two_repetition = np.array([[1, 1]])

hamming_7_4_3 = np.array([[1, 0, 1, 0, 1, 0, 1],
                          [0, 1, 1, 0, 0, 1, 1],
                          [0, 0, 0, 1, 1, 1, 1]])

def test_hgp_initilastion():
    """Test that the [[n, k, d]] code parameters and the name is set correctly at initialization"""

    temp_code = HypergraphProductCode(
        seed_matrix_1 = three_repetition, seed_matrix_2 = three_repetition 
    )
    assert (
        temp_code.physical_qubit_count == 13
    ), f"Expected N=13, but got N={temp_code.physical_qubit_count}"
    assert (
        temp_code.logical_qubit_count == 1
    ), f"Expected K=1, but got K={temp_code.logical_qubit_count}"
    assert (
        temp_code.name == "Hypergraph product code"
    ) 
    assert (
        np.all(temp_code.x_stabilizer_matrix.toarray()  ==  np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                                      [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                                                                      [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                                                                      [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                                                      [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
                                                                      [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]])))

    assert (
        np.all(temp_code.z_stabilizer_matrix.toarray() ==   np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                                      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                      [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                                                                      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
                                                                      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]]) )
    )

def test_hgp_initilastion_wrong_input_type():
    """Negative test to see that wrong input types are handled correctly."""

    with pytest.raises(TypeError) as e:
        test_code = HypergraphProductCode(seed_matrix_1 = "wrong input", seed_matrix_2 = three_repetition)
    assert e.match("The seed matrices must be either numpy arrays or scipy sparse matrices.")


@pytest.mark.parametrize('seed_matrix_1, seed_matrix_2, expected_distance', [(three_repetition, three_repetition, 3),
                                                                             (three_repetition, two_repetition, 2),
                                                                             (two_repetition, three_repetition, 2),
                                                                             (three_repetition, hamming_7_4_3, 3),
                                                                             (hamming_7_4_3, three_repetition, 3)])
def test_hgp_exact_code_distance(seed_matrix_1, seed_matrix_2, expected_distance):
    """Test exact code distance calculation for known constructions."""

    temp_code = HypergraphProductCode(seed_matrix_1, seed_matrix_2)
    assert temp_code.compute_exact_code_distance() == expected_distance


@pytest.mark.parametrize('seed_matrix_1, seed_matrix_2, expected_xd, expected_zd', [(three_repetition, three_repetition, 3, 3),
                                                                                    (three_repetition, two_repetition, 2, 3),
                                                                                    (two_repetition, three_repetition, 3, 2),
                                                                                    (three_repetition, hamming_7_4_3, 3, 3),
                                                                                    (hamming_7_4_3, three_repetition, 3, 3)])
def test_hgp_exact_X_Z_code_distance(seed_matrix_1, seed_matrix_2, expected_xd, expected_zd):
    """Test exact code distance calculation, separated for x and z distance, for known constructions."""
    temp_code = HypergraphProductCode(seed_matrix_1, seed_matrix_2)
    temp_code.compute_exact_code_distance() 

    assert temp_code.x_code_distance == expected_xd and temp_code.z_code_distance == expected_zd


def test_hgp_compute_logical_basis():
    """Test that the correct X and Z logical basis are returned for know constructions."""
    temp_code = HypergraphProductCode(three_repetition, three_repetition)
    temp_code.compute_logical_basis()

    assert np.all(temp_code.x_logical_operator_basis.toarray() == np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert np.all(temp_code.z_logical_operator_basis.toarray() == np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
