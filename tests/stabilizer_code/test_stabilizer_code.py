import pytest
import logging
import scipy
import numpy as np

from qec.stabilizer_code.stabilizer_code import StabilizerCode
from qec.quantum_codes import CodeTablesDE

# Define a binary parity check matrix for testing
binary_pcm = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]])

# Define corresponding Pauli stabilizer strings
pauli_strs = np.array([["ZZZZ"], ["XXXX"]])


def test_initialisation_with_binary_pcm():
    """
    Test the initialization of StabilizerCode using a binary parity check matrix.

    This test verifies that the StabilizerCode correctly interprets a binary
    parity check matrix, sets the appropriate attributes, and initializes
    the number of physical and logical qubits correctly.
    """
    # Initialize StabilizerCode with a binary parity check matrix
    temp_code = StabilizerCode(stabilizers=binary_pcm)

    # Check if the name attribute is correctly set
    assert temp_code.name == "stabilizer code", "StabilizerCode name mismatch."

    # Print the Pauli stabilizers for debugging purposes
    print(temp_code.pauli_stabilizers)

    # Verify that the Pauli stabilizers match the expected Pauli strings
    assert (
        temp_code.pauli_stabilizers == pauli_strs
    ).all(), "Pauli stabilizers mismatch."

    # Verify that the parity check matrix matches the input
    assert (
        temp_code.stabilizer_matrix.toarray() == binary_pcm
    ).all(), "Parity check matrix mismatch."

    # Verify the number of physical qubits (n) and logical qubits (k)
    assert (
        temp_code.physical_qubit_count == 4
    ), f"Expected n=4, but got n={temp_code.physical_qubit_count}"
    assert (
        temp_code.logical_qubit_count == 2
    ), f"Expected k=2, but got k={temp_code.logical_qubit_count}"

    # Uncomment the following line if you want to test the distance (d)
    # assert temp_code.code_distance == 2, f"Expected d=2, but got d={temp_code.code_distance}"


def test_initialisation_with_pauli_strings():
    """
    Test the initialization of StabilizerCode using Pauli stabilizer strings.

    This test ensures that the StabilizerCode correctly parses Pauli strings,
    constructs the corresponding parity check matrix, and initializes the
    number of physical and logical qubits appropriately.
    """
    # Initialize StabilizerCode with Pauli stabilizer strings
    temp_code = StabilizerCode(stabilizers=pauli_strs)

    # Check if the name attribute is correctly set
    assert temp_code.name == "stabilizer code", "StabilizerCode name mismatch."

    # Verify that the Pauli stabilizers match the input
    assert (
        temp_code.pauli_stabilizers == pauli_strs
    ).all(), "Pauli stabilizers mismatch."

    # Verify that the parity check matrix matches the expected binary PCM
    assert (
        temp_code.stabilizer_matrix.toarray() == binary_pcm
    ).all(), "Parity check matrix mismatch."

    # Verify the number of physical qubits (n) and logical qubits (k)
    assert (
        temp_code.physical_qubit_count == 4
    ), f"Expected n=4, but got n={temp_code.physical_qubit_count}"
    assert (
        temp_code.logical_qubit_count == 2
    ), f"Expected k=2, but got k={temp_code.logical_qubit_count}"

    # Uncomment the following line if you want to test the distance (d)
    # assert temp_code.code_distance == 2, f"Expected d=2, but got d={temp_code.code_distance}"


def test_initialisation_invalid_type():
    """
    Negative test for initializing StabilizerCode with an invalid input type.

    This test checks whether the StabilizerCode correctly raises a TypeError
    when initialized with an unsupported data type (e.g., a string).
    """
    # Attempt to initialize StabilizerCode with an invalid type and expect a TypeError
    with pytest.raises(
        TypeError,
        match="Please provide either a parity check matrix or a list of Pauli stabilizers.",
    ):
        temp_code = StabilizerCode(stabilizers="not a numpy array")


def test_wrong_pcm_shape():
    """
    Negative test for initializing StabilizerCode with a parity check matrix of incorrect shape.

    This test verifies that the StabilizerCode raises a ValueError when the
    provided parity check matrix has an odd number of columns, which is invalid.
    """
    # Define a parity check matrix with an odd number of columns
    wrong_pcm = np.array([[0, 1, 1], [1, 1, 0]])

    # Attempt to initialize StabilizerCode with the wrong PCM shape and expect a ValueError
    with pytest.raises(
        ValueError, match="The parity check matrix must have an even number of columns."
    ):
        temp_code = StabilizerCode(stabilizers=wrong_pcm)


def test_non_commuting_stabilizers():
    """
    Negative test for initializing StabilizerCode with non-commuting stabilizers.

    This test ensures that the StabilizerCode raises a ValueError when the
    provided stabilizer generators do not commute, which violates the stabilizer
    formalism requirements.
    """
    # Define non-commuting Pauli stabilizer strings
    non_commuting_stabilizers = np.array([["XXXX"], ["ZIII"]])

    # Attempt to initialize StabilizerCode with non-commuting Pauli strings and expect a ValueError
    with pytest.raises(ValueError, match="The stabilizers do not commute."):
        temp_code = StabilizerCode(stabilizers=non_commuting_stabilizers)

    # Define a binary parity check matrix that corresponds to non-commuting stabilizers
    non_commuting_pcm = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]])

    # Attempt to initialize StabilizerCode with the non-commuting PCM and expect a ValueError
    with pytest.raises(ValueError, match="The stabilizers do not commute."):
        temp_code = StabilizerCode(stabilizers=non_commuting_pcm)


def test_invalid_logical_operator_basis():
    """
    Test the validation of the logical operator basis in StabilizerCode.

    This test checks whether the StabilizerCode correctly identifies a valid
    and an invalid logical operator basis.
    """
    # Define valid Pauli stabilizer strings
    stabs = np.array([["ZZZZ"], ["XXXX"]])

    # Initialize StabilizerCode with valid stabilizers
    qcode = StabilizerCode(stabilizers=stabs)

    # Check the shape of logical operators (expected to be (4, 8))
    assert qcode.logical_operator_basis.shape == (
        4,
        8,
    ), f"Expected logicals shape (4,8), got {qcode.logical_operator_basis.shape}"

    # Verify that the current logical basis is valid
    assert qcode.check_valid_logical_basis(), "Logical operator basis should be valid."

    # Assign an invalid logical operator basis
    qcode.logical_operator_basis = np.array(
        [[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0]]
    )

    # Verify that the updated logical basis is invalid
    assert (
        not qcode.check_valid_logical_basis()
    ), "Logical operator basis should be invalid."


def test_compute_exact_code_distance():
    """
    Test the computation of the exact code distance in StabilizerCode.

    This test verifies that the StabilizerCode correctly computes the distance
    of a [[4, 2, 2]] quantum detection code.
    """
    # Define Pauli stabilizer strings for a \([[4, 2, 2]]\) code
    stabs = np.array([["ZZZZ"], ["XXXX"]])

    # Initialize StabilizerCode with the stabilizers
    qcode = StabilizerCode(stabilizers=stabs)

    # erase precomputed distance
    qcode.code_distance = None

    # Compute the exact code distance
    qcode.compute_exact_code_distance()

    # Verify that the computed distance matches the expected value
    assert (
        qcode.code_distance == 2
    ), f"Expected distance d=2, but got d={qcode.code_distance}"


def test_compute_exact_code_distance():
    qcode = CodeTablesDE(physical_qubit_count=10, logical_qubit_count=1)
    # erase precomputed distance
    qcode.code_distance = None

    # Compute the exact code distance
    qcode.compute_exact_code_distance()

    assert qcode.code_distance == 4


def test_estimate_min_distance():
    qcode = CodeTablesDE(physical_qubit_count=20, logical_qubit_count=1)
    # erase precomputed distance
    target_d = qcode.code_distance
    qcode.code_distance = None

    print(qcode.logical_basis_weights())

    qcode.estimate_min_distance(timeout_seconds=0.25, reduce_logical_basis=True)

    print(qcode.logical_basis_weights())
    # print(qcode.logicals.toarray())

    assert qcode.check_valid_logical_basis()

    print(qcode)

    assert qcode.code_distance == target_d


def test_check_valid_logical_basis_logging(caplog):
    # Test case where logical operators do not commute with stabilizers
    stabilizers = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    logical_operators = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    code = StabilizerCode(stabilizers=stabilizers)
    code.logical_operator_basis = logical_operators

    with caplog.at_level(logging.ERROR):
        assert not code.check_valid_logical_basis()
        assert "Logical operators do not commute with stabilizers." in caplog.text

    # Test case where logical operators do not anti-commute with one another
    stabilizers = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    logical_operators = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
    code = StabilizerCode(stabilizers=stabilizers)
    code.logical_operator_basis = logical_operators

    with caplog.at_level(logging.ERROR):
        assert not code.check_valid_logical_basis()
        assert (
            "The logical operators do not anti-commute with one another." in caplog.text
        )
