import pytest
import warnings
import numpy as np

from qec.stabilizer_code.stabilizer_code import StabiliserCode

# Define a binary parity check matrix for testing
binary_pcm = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]])

# Define corresponding Pauli stabilizer strings
pauli_strs = np.array([["ZZZZ"], ["XXXX"]])


def test_initialisation_with_binary_pcm():
    """
    Test the initialization of StabiliserCode using a binary parity check matrix.

    This test verifies that the StabiliserCode correctly interprets a binary
    parity check matrix, sets the appropriate attributes, and initializes
    the number of physical and logical qubits correctly.
    """
    # Initialize StabiliserCode with a binary parity check matrix
    temp_code = StabiliserCode(stabilisers=binary_pcm)

    # Check if the name attribute is correctly set
    assert temp_code.name == "stabiliser code", "StabiliserCode name mismatch."

    # Print the Pauli stabilizers for debugging purposes
    print(temp_code.pauli_stabilisers)

    # Verify that the Pauli stabilizers match the expected Pauli strings
    assert (
        temp_code.pauli_stabilisers == pauli_strs
    ).all(), "Pauli stabilizers mismatch."

    # Verify that the parity check matrix matches the input
    assert (temp_code.h.toarray() == binary_pcm).all(), "Parity check matrix mismatch."

    # Verify the number of physical qubits (n) and logical qubits (k)
    assert temp_code.n == 4, f"Expected n=4, but got n={temp_code.n}"
    assert temp_code.k == 2, f"Expected k=2, but got k={temp_code.k}"

    # Uncomment the following line if you want to test the distance (d)
    # assert temp_code.d == 2, f"Expected d=2, but got d={temp_code.d}"


def test_initialisation_with_pauli_strings():
    """
    Test the initialization of StabiliserCode using Pauli stabilizer strings.

    This test ensures that the StabiliserCode correctly parses Pauli strings,
    constructs the corresponding parity check matrix, and initializes the
    number of physical and logical qubits appropriately.
    """
    # Initialize StabiliserCode with Pauli stabilizer strings
    temp_code = StabiliserCode(stabilisers=pauli_strs)

    # Check if the name attribute is correctly set
    assert temp_code.name == "stabiliser code", "StabiliserCode name mismatch."

    # Verify that the Pauli stabilizers match the input
    assert (
        temp_code.pauli_stabilisers == pauli_strs
    ).all(), "Pauli stabilizers mismatch."

    # Verify that the parity check matrix matches the expected binary PCM
    assert (temp_code.h.toarray() == binary_pcm).all(), "Parity check matrix mismatch."

    # Verify the number of physical qubits (n) and logical qubits (k)
    assert temp_code.n == 4, f"Expected n=4, but got n={temp_code.n}"
    assert temp_code.k == 2, f"Expected k=2, but got k={temp_code.k}"

    # Uncomment the following line if you want to test the distance (d)
    # assert temp_code.d == 2, f"Expected d=2, but got d={temp_code.d}"


def test_initialisation_invalid_type():
    """
    Negative test for initializing StabiliserCode with an invalid input type.

    This test checks whether the StabiliserCode correctly raises a TypeError
    when initialized with an unsupported data type (e.g., a string).
    """
    # Attempt to initialize StabiliserCode with an invalid type and expect a TypeError
    with pytest.raises(
        TypeError,
        match="Please provide either a parity check matrix or a list of Pauli stabilisers.",
    ):
        temp_code = StabiliserCode(stabilisers="not a numpy array")


def test_wrong_pcm_shape():
    """
    Negative test for initializing StabiliserCode with a parity check matrix of incorrect shape.

    This test verifies that the StabiliserCode raises a ValueError when the
    provided parity check matrix has an odd number of columns, which is invalid.
    """
    # Define a parity check matrix with an odd number of columns
    wrong_pcm = np.array([[0, 1, 1], [1, 1, 0]])

    # Attempt to initialize StabiliserCode with the wrong PCM shape and expect a ValueError
    with pytest.raises(
        ValueError, match="The parity check matrix must have an even number of columns."
    ):
        temp_code = StabiliserCode(stabilisers=wrong_pcm)


def test_non_commuting_stabilisers():
    """
    Negative test for initializing StabiliserCode with non-commuting stabilizers.

    This test ensures that the StabiliserCode raises a ValueError when the
    provided stabilizer generators do not commute, which violates the stabilizer
    formalism requirements.
    """
    # Define non-commuting Pauli stabilizer strings
    non_commuting_stabilisers = np.array([["XXXX"], ["ZIII"]])

    # Attempt to initialize StabiliserCode with non-commuting Pauli strings and expect a ValueError
    with pytest.raises(ValueError, match="The stabilisers do not commute."):
        temp_code = StabiliserCode(stabilisers=non_commuting_stabilisers)

    # Define a binary parity check matrix that corresponds to non-commuting stabilizers
    non_commuting_pcm = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]])

    # Attempt to initialize StabiliserCode with the non-commuting PCM and expect a ValueError
    with pytest.raises(ValueError, match="The stabilisers do not commute."):
        temp_code = StabiliserCode(stabilisers=non_commuting_pcm)


def test_invalid_logical_operator_basis():
    """
    Test the validation of the logical operator basis in StabiliserCode.

    This test checks whether the StabiliserCode correctly identifies a valid
    and an invalid logical operator basis.
    """
    # Define valid Pauli stabilizer strings
    stabs = np.array([["ZZZZ"], ["XXXX"]])

    # Initialize StabiliserCode with valid stabilizers
    qcode = StabiliserCode(stabilisers=stabs)

    # Check the shape of logical operators (expected to be (4, 8))
    assert qcode.logicals.shape == (
        4,
        8,
    ), f"Expected logicals shape (4,8), got {qcode.logicals.shape}"

    # Verify that the current logical basis is valid
    assert qcode.check_valid_logical_basis(), "Logical operator basis should be valid."

    # Assign an invalid logical operator basis
    qcode.logicals = np.array([[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0]])

    # Verify that the updated logical basis is invalid
    assert (
        not qcode.check_valid_logical_basis()
    ), "Logical operator basis should be invalid."


def test_compute_exact_code_distance():
    """
    Test the computation of the exact code distance in StabiliserCode.

    This test verifies that the StabiliserCode correctly computes the distance
    of a simple \([[4, 2, 2]]\) quantum code.
    """
    # Define Pauli stabilizer strings for a \([[4, 2, 2]]\) code
    stabs = np.array([["ZZZZ"], ["XXXX"]])

    # Initialize StabiliserCode with the stabilizers
    qcode = StabiliserCode(stabilisers=stabs)

    # Compute the exact code distance
    qcode.compute_exact_code_distance()

    # Verify that the computed distance matches the expected value
    assert qcode.d == 2, f"Expected distance d=2, but got d={qcode.d}"


def test_10_3_3_code():
    """
    Test function for the [[10, 3, 3]] quantum stabilizer code.

    This test initializes the stabilizer matrix for the [[10, 3, 3]] code,
    computes the code distance, retrieves the code parameters, and asserts
    that they match the expected values.
    """
    # Define the stabilizer matrix for the [[10, 3, 3]] code
    stabiliser_matrix = np.array(
        [
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    # Initialize StabiliserCode with the stabilizer matrix
    qcode = StabiliserCode(stabilisers=stabiliser_matrix)

    # Compute the exact code distance
    qcode.compute_exact_code_distance()

    # Retrieve the code parameters: n (length), k (number of logical qubits), d (distance)
    n, k, d = qcode.get_code_parameters()

    # Assert that the code parameters match the expected values
    assert n == 10, f"Expected n=10, but got n={n}"
    assert k == 3, f"Expected k=3, but got k={k}"
    assert d == 3, f"Expected d=3, but got d={d}"


def test_10_2_4_code():
    """
    Test function for the [[10, 2, 4]] quantum stabilizer code.

    This test initializes the stabilizer matrix for the [[10, 2, 4]] code,
    computes the code distance, retrieves the code parameters, and asserts
    that they match the expected values.
    """
    # Define the stabilizer matrix for the [[10, 2, 4]] code
    stabiliser_matrix = np.array(
        [
            # X components              # Z components
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        ],
        dtype=int,
    )

    # Initialize StabiliserCode with the stabilizer matrix
    qcode = StabiliserCode(stabilisers=stabiliser_matrix)

    # Compute the exact code distance
    qcode.compute_exact_code_distance()

    # Retrieve the code parameters: n (length), k (number of logical qubits), d (distance)
    n, k, d = qcode.get_code_parameters()

    # Assert that the code parameters match the expected values
    assert n == 10, f"Expected n=10, but got n={n}"
    assert k == 2, f"Expected k=2, but got k={k}"
    assert d == 4, f"Expected d=4, but got d={d}"


def test_14_2_5_code():
    """
    Test function for the [[14, 2, 5]] quantum stabilizer code.

    This function initializes the stabilizer matrix for the [[14, 2, 5]] code,
    computes the code distance, retrieves the code parameters, and asserts
    that they match the expected values.
    """
    # Define the stabilizer matrix as a NumPy array
    # Each row consists of 14 X components followed by 14 Z components
    stabiliser_matrix = np.array(
        [
            # X components (14 bits)                 Z components (14 bits)
            [
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
            ],
            [
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
            ],
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
            ],
            [
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
            ],
        ],
        dtype=int,
    )

    # Initialize StabiliserCode with the stabilizer matrix
    qcode = StabiliserCode(stabilisers=stabiliser_matrix)

    # Compute the exact code distance
    qcode.compute_exact_code_distance()

    # Retrieve the code parameters: n (length), k (number of logical qubits), d (distance)
    n, k, d = qcode.get_code_parameters()

    # Assert that the code parameters match the expected values
    assert n == 14, f"Expected n=14, but got n={n}"
    assert k == 2, f"Expected k=2, but got k={k}"
    assert d == 5, f"Expected d=5, but got d={d}"


def test_16_1_6_code():
    """
    Test function for the [[16, 1, 6]] quantum stabilizer code.

    This test initializes the stabilizer matrix for the [[16, 1, 6]] code,
    computes the code distance, retrieves the code parameters, and asserts
    that they match the expected values.

    Raises:
        AssertionError: If any of the code parameters (n, k, d) do not match
                        the expected values for the [[16, 1, 6]] code.
    """
    # Define the stabilizer matrix as a NumPy array
    # Each row consists of 16 X components followed by 16 Z components
    stabiliser_matrix = np.array(
        [
            # X components (16 bits)                   Z components (16 bits)
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                1,
                1,
            ],
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
            ],
        ],
        dtype=int,
    )

    # Initialize the StabiliserCode with the stabilizer matrix
    qcode = StabiliserCode(stabilisers=stabiliser_matrix)

    # Compute the exact code distance
    d, fraction_considered = qcode.compute_exact_code_distance()

    # Retrieve the code parameters: n (number of physical qubits), k (number of logical qubits), d (distance)
    n, k, d = qcode.get_code_parameters()

    # Assert that the code parameters match the expected values
    assert n == 16, f"Expected n=16, but got n={n}"
    assert k == 1, f"Expected k=1, but got k={k}"
    assert d == 6, f"Expected d=6, but got d={d}"

    print(qcode)
    print(qcode.pauli_stabilisers)
    print(fraction_considered)

