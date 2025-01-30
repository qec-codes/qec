import pytest
import logging
import numpy as np
import scipy

from qec.code_constructions import CSSCode
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse

# Define a binary parity check matrix for testing
# Hamming (7, 4) code
hamming_7_4 = np.array(
    [[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1]], dtype=int
)
# Repetition (5,1) code
repetition_5_1 = np.array(
    [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=int
)


def test_initialisation_with_xz_stabilizers():
    """
    Test the initialisation of a CSSCode object with x and z stabilizers.

    This verifies that the CSSCode correctly interprets an x and z stabilizer matrix,
    sets the appropiate attributes, and initializes the number of physical and
    logical qubits correctly.
    """
    # Initialize CSSCode with stabilizers
    x_stabilizer_matrix = hamming_7_4
    z_stabilizer_matrix = hamming_7_4
    temp_code = CSSCode(
        x_stabilizer_matrix=x_stabilizer_matrix, z_stabilizer_matrix=z_stabilizer_matrix
    )

    # Chek if the name attribute is correctly set
    assert temp_code.name == "CSS code", "CSSCode name mismatch."

    # Verify the number of physical qubits (N) and logical qubits (K)
    assert (
        temp_code.physical_qubit_count == 7
    ), f"Expected N=7, but got N={temp_code.physical_qubit_count}"
    assert (
        temp_code.logical_qubit_count == 1
    ), f"Expected K=1, but got K={temp_code.logical_qubit_count}"

    # Uncomment the following line if you want to test the distance (d)
    # assert temp_code.code_distance == 3, f"Expected d=3, but got d={temp_code.code_distance}"


def test_initialisation_with_xz_stabilizers_invalid_type():
    """
    Negative test for initializing CSSCode with an invalid input type.

    This test checks whether the CSSCode correctly raises a TypeError
    when initialized with an unsupported data type (e.g., a string).
    """
    # Attempt to initialize CSSCode with an invalid type and expect a TypeError
    with pytest.raises(
        TypeError,
        match="Please provide x and z stabilizer matrices as either a numpy array or a scipy sparse matrix.",
    ):
        CSSCode(
            x_stabilizer_matrix="not a numpy array",
            z_stabilizer_matrix="not a numpy array",
        )  # String input should raise TypeError


# Test h_x and h_z inputs are of the same size? i.e. have the same block length (number of columns)
# Note that the row dimension can differ, as h_x and h_z can have different number of stabilizers (rows)
def test_initialisation_with_xz_stabilizers_invalid_shape():
    """
    Negative test for initializing CSSCode with invalid stabilizer shapes.

    This test checks whether the CSSCode correctly raises a ValueError
    when initialized with x and z stabilizers of different column dimensions.
    """
    # Define a pair of invalid stabilizers with different column dimensions
    wrong_x = hamming_7_4
    wrong_z = repetition_5_1

    # Attemp to initialize CSSCode with invalid stabilizer shapes and expect a ValueError
    with pytest.raises(
        ValueError,
        match=f"Input matrices x_stabilizer_matrix and z_stabilizer_matrix must have the same number of columns.\
                              Current column count, x_stabilizer_matrix: {wrong_x.shape[1]}; z_stabilizer_matrix: {wrong_z.shape[1]}",
    ):
        CSSCode(
            x_stabilizer_matrix=wrong_x, z_stabilizer_matrix=wrong_z
        )  # Different column dimensions should raise ValueError


def test_initialization_with_non_commuting_xz_stabilizers():
    """
    Negative test for initializing CSSCode with non-commuting stabilizer matrices.

    This test ensures that the CSSCode raises a ValueError when the
    provided stabilizer matrices that do not commute, which violates the stabilizer
    formalism requirements (which for css codes is the requirement that hx@hz.T = 0).
    """
    # Define non-commuting x and z stabilizer matrices
    non_commuting_x = np.array([[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0]])
    non_commuting_z = np.array([[1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0]])

    # Attempt to initialize StabilizerCode with non-commuting Pauli strings and expect a ValueError
    with pytest.raises(
        ValueError,
        match="Input matrices hx and hz do not commute. I.e. they do not satisfy\
                              the requirement that hx@hz.T = 0.",
    ):
        CSSCode(
            x_stabilizer_matrix=non_commuting_x, z_stabilizer_matrix=non_commuting_z
        )  # Non-commuting stabilizers should raise ValueError


def test_invalid_logical_xz_basis():
    """
    Test the validation of the logical operator basis in CSSCode.

    This test checks whether the CSSCode correctly identifies a valid
    and an invalid logical x / z operator basis.
    """
    # Define valid Stabilizers
    x_stabilizer_matrix = hamming_7_4
    z_stabilizer_matrix = hamming_7_4

    # Initialize CSSCode with valid stabilizers
    qcode = CSSCode(
        x_stabilizer_matrix=x_stabilizer_matrix, z_stabilizer_matrix=z_stabilizer_matrix
    )

    # Check the dimensions of the logical operator basis (i.e. the logical qubit count)
    assert qcode.logical_qubit_count == (
        1
    ), f"Expected logical qubit count to be 1, got {qcode.logical_qubit_count}."

    # Verify that the current logical basis is valid
    assert qcode.check_valid_logical_basis(), "Logical operator basis should be valid."

    # Assign an invalid logical operator basis
    qcode.x_logical_operator_basis = convert_to_binary_scipy_sparse(
        np.array([[1, 0, 0, 0, 0, 0, 0]])
    )
    qcode.z_logical_operator_basis = convert_to_binary_scipy_sparse(
        np.array([[1, 0, 0, 0, 0, 0, 0]])
    )

    # Verify that the updated logical basis is invalid
    assert (
        not qcode.check_valid_logical_basis()
    ), "Logical operator basis should be invalid."


def test_check_valid_logical_xz_basis_logging1(caplog):
    # Test case where logical operators do not commute with stabilizers (using Steane code as CSSCode)
    x_stabilizer_matrix = hamming_7_4
    z_stabilizer_matrix = hamming_7_4

    print(x_stabilizer_matrix)

    code = CSSCode(
        x_stabilizer_matrix=x_stabilizer_matrix, z_stabilizer_matrix=z_stabilizer_matrix
    )

    x_logical_operators = convert_to_binary_scipy_sparse(
        np.array([[1, 1, 1, 1, 0, 0, 0]])
    )
    z_logical_operators = convert_to_binary_scipy_sparse(
        np.array([[1, 1, 0, 1, 0, 0, 0]])
    )
    code.x_logical_operator_basis = x_logical_operators
    code.z_logical_operator_basis = z_logical_operators

    with caplog.at_level(logging.ERROR):
        assert not code.check_valid_logical_basis()
        print(caplog.text)
        assert "Z logical operators do not commute with X stabilizers." in caplog.text


def test_check_valid_logical_xz_basis_logging2(caplog):
    # Test case where logical operators do not commute with stabilizers (using Steane code as CSSCode)
    x_stabilizer_matrix = hamming_7_4
    z_stabilizer_matrix = hamming_7_4

    print(x_stabilizer_matrix)

    code = CSSCode(
        x_stabilizer_matrix=x_stabilizer_matrix, z_stabilizer_matrix=z_stabilizer_matrix
    )

    x_logical_operators = convert_to_binary_scipy_sparse(
        np.array([[1, 1, 0, 1, 0, 0, 0]])
    )
    z_logical_operators = convert_to_binary_scipy_sparse(
        np.array([[1, 1, 1, 1, 0, 0, 0]])
    )
    code.x_logical_operator_basis = x_logical_operators
    code.z_logical_operator_basis = z_logical_operators

    with caplog.at_level(logging.ERROR):
        assert not code.check_valid_logical_basis()
        print(caplog.text)
        assert "X logical operators do not commute with Z stabilizers." in caplog.text


def test_check_valid_logical_xz_basis_logging3(caplog):
    # Test case where logical operators do not anti-commute with one another
    x_stabilizer_matrix = hamming_7_4
    z_stabilizer_matrix = hamming_7_4

    x_logical_operators = convert_to_binary_scipy_sparse(
        np.array([[1, 1, 1, 1, 0, 0, 0]])
    )
    z_logical_operators = convert_to_binary_scipy_sparse(
        np.array([[1, 1, 0, 0, 1, 1, 0]])
    )

    code = CSSCode(
        x_stabilizer_matrix=x_stabilizer_matrix, z_stabilizer_matrix=z_stabilizer_matrix
    )

    code.x_logical_operator_basis = x_logical_operators
    code.z_logical_operator_basis = z_logical_operators

    with caplog.at_level(logging.ERROR):
        assert not code.check_valid_logical_basis()
        assert "Logical operators do not pairwise anticommute." in caplog.text
        assert "Logical operators do not pairwise anticommute." in caplog.text


def test_fix_logical_operators_invalid_input():
    """
    Negative test for the fix_logical_operators method in CSSCode.

    This test checks whether the CSSCode correctly raises a TypeError
    when calling fix_logical_operators with an unsupported data type (e.g., an int).
    """
    # Define valid Stabilizers
    x_stabilizer_matrix = hamming_7_4
    z_stabilizer_matrix = hamming_7_4

    # Initialize CSSCode with valid stabilizers
    qcode = CSSCode(
        x_stabilizer_matrix=x_stabilizer_matrix, z_stabilizer_matrix=z_stabilizer_matrix
    )

    # Attempt to call fix_logical_operators with an invalid data type and expect a TypeError
    with pytest.raises(TypeError, match="fix_logical parameter must be a string"):
        qcode.fix_logical_operators(fix_logical=1)


def test_fix_logical_operators_invalid_parameters():
    """
    Negative test for the fix_logical_operators method in CSSCode.

    This test checks whether the CSSCode correctly raises a ValueError
    when calling fix_logical_operators with an unsupported parameter (e.g., "Y")
    """
    # Define valid Stabilizers
    x_stabilizer_matrix = hamming_7_4
    z_stabilizer_matrix = hamming_7_4

    # Initialize CSSCode with valid stabilizers
    qcode = CSSCode(
        x_stabilizer_matrix=x_stabilizer_matrix, z_stabilizer_matrix=z_stabilizer_matrix
    )

    # Attempt to call fix_logical_operators with an invalid parameter and expect a ValueError
    with pytest.raises(ValueError, match="Invalid fix_logical parameter"):
        qcode.fix_logical_operators(fix_logical="Y")


# TODO: Complete test
def test_fix_logical_operators():
    """
    Test the fix_logical_operators method in CSSCode.

    This test checks whether the fix_logical_operators method correctly
    updates the logical operator basis to a valid basis when the current
    basis is invalid.
    """
    pass


def test_steane_code_distance():
    # Define the Hamming code parity check matrix for Steane code
    # This is the [7,4,3] Hamming code
    hx = np.array([[1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1]])

    # In Steane code, hz is the same as hx
    hz = hx

    # Create the Steane code
    steane = CSSCode(hx, hz, name="Steane")

    # Compute exact distance with sufficient timeout
    dx, dz, fraction = steane.compute_exact_code_distance()

    # Verify the code parameters
    assert steane.physical_qubit_count == 7, "Should have 7 physical qubits"
    assert steane.logical_qubit_count == 1, "Should encode 1 logical qubit"

    # Test the computed distances
    assert dx == 3, f"X distance should be 3, got {dx}"
    assert dz == 3, f"Z distance should be 3, got {dz}"
    assert steane.code_distance == 3, "Code distance should be 3"

    # Check that we completed the search
    assert fraction == 1.0, "Should have completed the full search"

    # Test timeout behavior
    dx_quick, dz_quick, fraction_quick = steane.compute_exact_code_distance(
        timeout=0.001
    )
    assert fraction_quick < 1.0, "Quick search should not complete"


def test_steane_code_estimate_distance():
    # Define the Hamming code parity check matrix for Steane code
    # This is the [7,4,3] Hamming code
    hx = np.array([[1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1]])

    # In Steane code, hz is the same as hx
    hz = hx

    # Create the Steane code
    steane = CSSCode(hx, hz, name="Steane")

    # Estimate minimum distance with a timeout
    estimated_distance = steane.estimate_min_distance(
        timeout_seconds=0.25, reduce_logical_basis=True
    )

    assert steane.check_valid_logical_basis(), "Logical basis should be valid"

    # Verify the code parameters
    assert steane.physical_qubit_count == 7, "Should have 7 physical qubits"
    assert steane.logical_qubit_count == 1, "Should encode 1 logical qubit"

    # Test the estimated distance
    assert (
        estimated_distance >= 3
    ), f"Estimated distance should be at least 3, got {estimated_distance}"


def test_stabilizer_matrix():

    # Define the Hamming code parity check matrix for Steane code
    # This is the [7,4,3] Hamming code
    hx = np.array([[1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1]])

    # In Steane code, hz is the same as hx
    hz = hx

    # Create the Steane code
    steane = CSSCode(hx, hz, name="Steane")

    # Check the stabilizer matrices
    assert np.array_equal(steane.x_stabilizer_matrix.toarray(), hx), "X stabilizer matrix mismatch"
    assert np.array_equal(steane.z_stabilizer_matrix.toarray(), hz), "Z stabilizer matrix mismatch"

    stabilizer_matrix = scipy.sparse.block_diag([steane.x_stabilizer_matrix, steane.z_stabilizer_matrix])

    assert np.array_equal(steane.stabilizer_matrix.toarray(), stabilizer_matrix.toarray()), "Stabilizer matrix mismatch"

def test_logical_operator_basis():
    
    # Define the Hamming code parity check matrix for Steane code
    # This is the [7,4,3] Hamming code
    hx = np.array([[1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1]])

    # In Steane code, hz is the same as hx
    hz = hx

    # Create the Steane code
    steane = CSSCode(hx, hz, name="Steane")

    logical_operator_basis = scipy.sparse.block_diag([steane.x_logical_operator_basis, steane.z_logical_operator_basis])

    assert np.array_equal(steane.logical_operator_basis.toarray(), logical_operator_basis.toarray()), "Logical operator basis mismatch"