import pytest
import json
import numpy as np

from qec.code_constructions import HypergraphProductCode
from qec.utils.sparse_binary_utils import csr_matrix_to_dict

three_repetition = np.array([[1, 1, 0], [0, 1, 1]])

two_repetition = np.array([[1, 1]])

hamming_7_4_3 = np.array(
    [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]]
)


def test_hgp_initilastion():
    """Test that the [[n, k, d]] code parameters and the name is set correctly at initialization"""

    temp_code = HypergraphProductCode(
        seed_matrix_1=three_repetition, seed_matrix_2=three_repetition
    )
    assert (
        temp_code.physical_qubit_count == 13
    ), f"Expected N=13, but got N={temp_code.physical_qubit_count}"
    assert (
        temp_code.logical_qubit_count == 1
    ), f"Expected K=1, but got K={temp_code.logical_qubit_count}"
    assert temp_code.name == "Hypergraph product code"
    assert np.all(
        temp_code.x_stabilizer_matrix.toarray()
        == np.array(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            ]
        )
    )

    assert np.all(
        temp_code.z_stabilizer_matrix.toarray()
        == np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            ]
        )
    )


def test_hgp_initilastion_wrong_input_type():
    """Negative test to see that wrong input types are handled correctly."""

    with pytest.raises(TypeError) as e:
        HypergraphProductCode(
            seed_matrix_1="wrong input", seed_matrix_2=three_repetition
        )
    assert e.match(
        "The seed matrices must be either numpy arrays or scipy sparse matrices."
    )


@pytest.mark.parametrize(
    "seed_matrix_1, seed_matrix_2, expected_distance",
    [
        (three_repetition, three_repetition, 3),
        (three_repetition, two_repetition, 2),
        (two_repetition, three_repetition, 2),
        (three_repetition, hamming_7_4_3, 3),
        (hamming_7_4_3, three_repetition, 3),
    ],
)
def test_hgp_exact_code_distance(seed_matrix_1, seed_matrix_2, expected_distance):
    """Test exact code distance calculation for known constructions."""

    temp_code = HypergraphProductCode(seed_matrix_1, seed_matrix_2)
    assert temp_code.compute_exact_code_distance() == expected_distance


@pytest.mark.parametrize(
    "seed_matrix_1, seed_matrix_2, expected_xd, expected_zd",
    [
        (three_repetition, three_repetition, 3, 3),
        (three_repetition, two_repetition, 2, 3),
        (two_repetition, three_repetition, 3, 2),
        (three_repetition, hamming_7_4_3, 3, 3),
        (hamming_7_4_3, three_repetition, 3, 3),
    ],
)
def test_hgp_exact_X_Z_code_distance(
    seed_matrix_1, seed_matrix_2, expected_xd, expected_zd
):
    """Test exact code distance calculation, separated for x and z distance, for known constructions."""
    temp_code = HypergraphProductCode(seed_matrix_1, seed_matrix_2)
    temp_code.compute_exact_code_distance()

    assert (
        temp_code.x_code_distance == expected_xd
        and temp_code.z_code_distance == expected_zd
    )


def test_hgp_compute_logical_basis():
    """Test that the correct X and Z logical basis are returned for know constructions."""
    temp_code = HypergraphProductCode(three_repetition, three_repetition)
    temp_code.compute_logical_basis()

    assert np.all(
        temp_code.x_logical_operator_basis.toarray()
        == np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    assert np.all(
        temp_code.z_logical_operator_basis.toarray()
        == np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    )


#----------------------------------------
# Tests for saving HGP codes
#----------------------------------------

test_hgp_code = HypergraphProductCode(three_repetition, three_repetition, name = 'test')

def test_hgp_save_code_correct_content(tmp_path):
    """Test the content of the saved JSON file."""
    filepath = tmp_path / 'test_code.json'
    notes = "Test notes"
    test_hgp_code.save_code(filepath, notes)
    
    with open(filepath, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data['class_name'] == 'HypergraphProductCode'
    assert saved_data['name'] == 'test'
    assert saved_data['parameters']['physical_qubit_count'] == 13 
    assert saved_data['parameters']['logical_qubit_count'] == 1
    assert saved_data['parameters']['code_distance'] == '?'
    assert saved_data['parameters']['x_code_distance'] == '?'
    assert saved_data['parameters']['z_code_distance'] == '?'
    assert saved_data['seed_matrix_1']['data'] == csr_matrix_to_dict(test_hgp_code.seed_matrix_1)['data']
    assert saved_data['seed_matrix_1']['indices'] == csr_matrix_to_dict(test_hgp_code.seed_matrix_1)['indices']
    assert saved_data['seed_matrix_1']['indptr'] == csr_matrix_to_dict(test_hgp_code.seed_matrix_1)['indptr']
    assert saved_data['seed_matrix_1']['shape'] == list(csr_matrix_to_dict(test_hgp_code.seed_matrix_1)['shape'])
    assert saved_data['seed_matrix_2']['data'] == csr_matrix_to_dict(test_hgp_code.seed_matrix_2)['data']
    assert saved_data['seed_matrix_2']['indices'] == csr_matrix_to_dict(test_hgp_code.seed_matrix_2)['indices']
    assert saved_data['seed_matrix_2']['indptr'] == csr_matrix_to_dict(test_hgp_code.seed_matrix_2)['indptr']
    assert saved_data['seed_matrix_2']['shape'] == list(csr_matrix_to_dict(test_hgp_code.seed_matrix_2)['shape'])
    assert saved_data['x_logical_operator_basis']['data'] == csr_matrix_to_dict(test_hgp_code.x_logical_operator_basis)['data']
    assert saved_data['x_logical_operator_basis']['indices'] == csr_matrix_to_dict(test_hgp_code.x_logical_operator_basis)['indices']
    assert saved_data['x_logical_operator_basis']['indptr'] == csr_matrix_to_dict(test_hgp_code.x_logical_operator_basis)['indptr']
    assert saved_data['x_logical_operator_basis']['shape'] == list(csr_matrix_to_dict(test_hgp_code.x_logical_operator_basis)['shape'])
    assert saved_data['z_logical_operator_basis']['data'] == csr_matrix_to_dict(test_hgp_code.z_logical_operator_basis)['data']
    assert saved_data['z_logical_operator_basis']['indices'] == csr_matrix_to_dict(test_hgp_code.z_logical_operator_basis)['indices']
    assert saved_data['z_logical_operator_basis']['indptr'] == csr_matrix_to_dict(test_hgp_code.z_logical_operator_basis)['indptr']
    assert saved_data['z_logical_operator_basis']['shape'] == list(csr_matrix_to_dict(test_hgp_code.z_logical_operator_basis)['shape'])
    assert saved_data['notes'] == notes


