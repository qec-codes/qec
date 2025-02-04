import pytest
import json
import numpy as np
import importlib.resources

from qec.code_constructions import StabilizerCode, CSSCode, HypergraphProductCode

from qec.utils import load_code


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_code("nonexistent.json")


def test_invalid_code_class(tmp_path):
    invalid_data = {"class_name": "InvalidCode"}
    filepath = tmp_path / "invalid.json"
    with open(filepath, "w") as f:
        json.dump(invalid_data, f)

    with pytest.raises(AttributeError):
        load_code(filepath)


test_stab_code = StabilizerCode(stabilizers=np.array([["ZZZZ"], ["XXXX"]]), name="test")


def test_load_stabilizer_code(tmp_path):
    filepath = tmp_path / "test_stab_code.json"
    test_stab_code.save_code(filepath)

    instance = load_code(filepath)
    assert instance.physical_qubit_count == test_stab_code.physical_qubit_count
    assert instance.logical_qubit_count == test_stab_code.logical_qubit_count
    assert instance.code_distance == test_stab_code.code_distance
    assert instance.name == test_stab_code.name


hamming_7_4 = np.array(
    [[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1]],
    dtype=np.uint8,
)
test_css_code = CSSCode(hamming_7_4, hamming_7_4, name="test")


def test_load_css_code(tmp_path):
    filepath = tmp_path / "test_css_code.json"
    test_css_code.save_code(filepath)

    instance = load_code(filepath)
    assert instance.physical_qubit_count == test_css_code.physical_qubit_count
    assert instance.logical_qubit_count == test_css_code.logical_qubit_count
    assert instance.code_distance == test_css_code.code_distance
    assert instance.x_code_distance == test_css_code.x_code_distance
    assert instance.z_code_distance == test_css_code.z_code_distance
    assert instance.name == test_css_code.name
    assert np.all(
        instance.x_stabilizer_matrix.toarray()
        == test_css_code.x_stabilizer_matrix.toarray()
    )
    assert np.all(
        instance.z_stabilizer_matrix.toarray()
        == test_css_code.z_stabilizer_matrix.toarray()
    )
    assert np.all(
        instance.x_logical_operator_basis.toarray()
        == test_css_code.x_logical_operator_basis.toarray()
    )
    assert np.all(
        instance.z_logical_operator_basis.toarray()
        == test_css_code.z_logical_operator_basis.toarray()
    )


test_hgp_code = HypergraphProductCode(hamming_7_4, hamming_7_4, name="test")
# test_hgp_code.compute_exact_code_distance() <-- uncomment this to fail the test


def test_load_hgp_code(tmp_path):
    filepath = tmp_path / "test_hgp_code.json"
    test_hgp_code.save_code(filepath)

    instance = load_code(filepath)
    assert instance.physical_qubit_count == test_hgp_code.physical_qubit_count
    assert instance.logical_qubit_count == test_hgp_code.logical_qubit_count
    assert instance.code_distance == test_hgp_code.code_distance
    assert instance.x_code_distance == test_hgp_code.x_code_distance
    assert instance.z_code_distance == test_hgp_code.z_code_distance
    assert instance.name == test_hgp_code.name
    assert np.all(
        instance.seed_matrix_1.toarray() == test_hgp_code.seed_matrix_1.toarray()
    )
    assert np.all(
        instance.seed_matrix_2.toarray() == test_hgp_code.seed_matrix_2.toarray()
    )
    assert np.all(
        instance.x_logical_operator_basis.toarray()
        == test_hgp_code.x_logical_operator_basis.toarray()
    )
    assert np.all(
        instance.z_logical_operator_basis.toarray()
        == test_hgp_code.z_logical_operator_basis.toarray()
    )



