import pytest
import numpy as np
import scipy
from qec.utils.code_utils import (
    GF4_to_binary,
    pauli_str_to_binary_pcm,
    binary_pcm_to_pauli_str,
)


@pytest.mark.parametrize(
    "gf4_matrix, expected_dense",
    [
        # 1) 2x2 GF(4) matrix with all zeros => all zeros in binary (2*2=4 columns)
        (
            np.array([[0, 0], [0, 0]], dtype=int),
            np.zeros((2, 4), dtype=int),
        ),
        # 2) 1x1 with value=1 => (1,0)
        (
            np.array([[1]], dtype=int),
            np.array([[1, 0]], dtype=int),
        ),
        # 3) 1x1 with value=2 => (0,1)
        (
            np.array([[2]], dtype=int),
            np.array([[1, 1]], dtype=int),
        ),
        # 4) 1x2 with [3,0].
        #    Value 3 => the code places a '1' in column (j + width), j=0 => col=2 => => row => [0, 0, 1, 0].
        (
            np.array([[3, 0]], dtype=int),
            np.array([[0, 0, 1, 0]], dtype=int),
        ),
        # 5) 2x2 with [[1,2],[3,2]]
        (
            np.array([[1, 2], [3, 2]], dtype=int),
            np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=int),
        ),
    ],
)
def test_GF4_to_binary_dense(gf4_matrix, expected_dense):
    """
    Test GF4_to_binary by comparing the resulting dense matrix
    to an expected dense NumPy array.
    """
    csr_result = GF4_to_binary(gf4_matrix)
    dense_result = csr_result.toarray()
    np.testing.assert_array_equal(
        dense_result,
        expected_dense,
        err_msg=f"\nGF4 matrix:\n{gf4_matrix}\nExpected:\n{expected_dense}\nGot:\n{dense_result}",
    )


def test_GF4_to_binary_random():
    """Test GF4_to_binary with a random GF(4) matrix; just check shape."""
    rng = np.random.default_rng(42)
    random_matrix = rng.integers(0, 4, size=(5, 4), dtype=int)
    random_matrix = scipy.sparse.csr_matrix(random_matrix)
    csr_result = GF4_to_binary(random_matrix)
    dense_result = csr_result.toarray()

    assert dense_result.shape == (
        5,
        8,
    ), f"Expected shape (5,8), got {dense_result.shape}"


def test_GF4_to_binary_empty():
    """Test GF4_to_binary with an empty matrix."""
    empty_matrix = np.array([], dtype=int).reshape(0, 0)
    csr_result = GF4_to_binary(empty_matrix)
    dense_result = csr_result.toarray()

    assert dense_result.shape == (0, 0)
    assert dense_result.size == 0


def test_GF4_to_binary_valid():
    """Test GF4_to_binary with valid GF4 elements."""
    input_matrix = np.array([[0, 1, 2], [3, 0, 1]], dtype=int)

    output_csr = GF4_to_binary(input_matrix)

    # Check shape
    assert output_csr.shape == (2, 6)

    # Check data consistency (example of how you might check a known outcome)
    # Adapt these to match the exact structure you expect
    assert np.all(output_csr.data == 1)


def test_GF4_to_binary_invalid():
    """Test GF4_to_binary raises ValueError for invalid inputs."""
    invalid_matrix = np.array(
        [
            [0, 1, 4],  # 4 is not in GF4
            [3, 5, 2],  # 5 is not in GF4
        ],
        dtype=int,
    )

    with pytest.raises(ValueError) as exc_info:
        _ = GF4_to_binary(invalid_matrix)
    assert "Input matrix must contain only elements from GF4" in str(exc_info.value)


@pytest.mark.parametrize(
    "pauli_array, expected_dense",
    [
        # 1) Single row: "III" => all I => (0,0) for each => row => [0,0, 0,0, 0,0]
        (
            np.array([["III"]], dtype=str),
            np.zeros((1, 6), dtype=int),
        ),
        # 2) Single row: "X" => (1,0)
        (
            np.array([["X"]], dtype=str),
            np.array([[1, 0]], dtype=int),
        ),
        # 3) Single row: "Z" => (0,1)
        (
            np.array([["Z"]], dtype=str),
            np.array([[0, 1]], dtype=int),
        ),
        # 4) Single row: "Y" => (1,1)
        (
            np.array([["Y"]], dtype=str),
            np.array([[1, 1]], dtype=int),
        ),
        # 5) Two rows: ["XI"], ["IZ"]
        #    "XI" => 'X'=(1,0), 'I'=(0,0) => row => [1,0, 0,0]
        #    "IZ" => 'I'=(0,0), 'Z'=(0,1) => row => [0,0, 0,1]
        (
            np.array([["XI"], ["IZ"]], dtype=str),
            np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=int),
        ),
    ],
)
def test_pauli_str_to_binary_pcm_dense(pauli_array, expected_dense):
    """
    Test pauli_str_to_binary_pcm by comparing the resulting dense matrix
    to an expected dense NumPy array.
    """
    csr_result = pauli_str_to_binary_pcm(pauli_array)
    dense_result = csr_result.toarray()
    np.testing.assert_array_equal(
        dense_result,
        expected_dense,
        err_msg=f"\nPauli array:\n{pauli_array}\nExpected:\n{expected_dense}\nGot:\n{dense_result}",
    )


def test_pauli_str_to_binary_pcm_random():
    """Test pauli_str_to_binary_pcm with random Pauli strings; just check shape."""
    possible_paulis = ["I", "X", "Y", "Z"]
    rng = np.random.default_rng(1234)
    num_strings = 5
    length = 4

    random_paulis = [
        "".join(rng.choice(possible_paulis, size=length)) for _ in range(num_strings)
    ]
    pauli_array = np.array(random_paulis, dtype=str).reshape(-1, 1)

    csr_result = pauli_str_to_binary_pcm(pauli_array)
    dense_result = csr_result.toarray()

    assert dense_result.shape == (
        num_strings,
        2 * length,
    ), f"Expected shape {(num_strings, 2 * length)}, got {dense_result.shape}"


def test_pauli_str_to_binary_pcm_empty():
    """Test pauli_str_to_binary_pcm with an empty array."""
    pauli_array = np.array([], dtype=str)
    csr_result = pauli_str_to_binary_pcm(pauli_array)
    dense_result = csr_result.toarray()

    assert dense_result.shape == (0, 0), f"Expected (), got {dense_result.shape}"
    assert dense_result.size == 0


def test_binary_pcm_to_pauli_str_basic():
    """Test binary_pcm_to_pauli_str with a simple small matrix."""
    # Suppose we have a 2x4 binary matrix, each row is (X-bits, Z-bits) of length 2
    # Row 0 => x_bits=[1,0], z_bits=[0,1] => "XZ"
    # Row 1 => x_bits=[1,1], z_bits=[1,0] => "YX"
    binary_pcm = np.array(
        [
            [1, 0, 0, 1],  # => XZ
            [1, 1, 1, 0],  # => YX
        ],
        dtype=int,
    )

    pauli_strs = binary_pcm_to_pauli_str(binary_pcm)
    expected = np.array([["XZ"], ["YX"]], dtype=str)

    np.testing.assert_array_equal(
        pauli_strs,
        expected,
        err_msg=f"\nBinary PCM:\n{binary_pcm}\nExpected:\n{expected}\nGot:\n{pauli_strs}",
    )


def test_binary_pcm_to_pauli_str_inverse():
    """
    Test that pauli_str_to_binary_pcm and binary_pcm_to_pauli_str are inverses:
    pauli_str -> binary -> pauli_str.
    """
    original = np.array([["IXYZ"], ["ZZII"], ["YYYY"]], dtype=str)
    csr_bin = pauli_str_to_binary_pcm(original)
    bin_dense = csr_bin.toarray()

    recovered = binary_pcm_to_pauli_str(bin_dense)
    np.testing.assert_array_equal(
        recovered, original, err_msg=f"\nOriginal:\n{original}\nRecovered:\n{recovered}"
    )


def test_binary_pcm_to_pauli_str_all_identities():
    """Test binary_pcm_to_pauli_str with rows of all zeros => only 'I'."""
    # For length=4 qubits, we need 8 bits (4 X-bits, 4 Z-bits).
    # All zeros => "IIII".
    binary_pcm = np.zeros((3, 8), dtype=int)  # 3 rows => "IIII" for each
    pauli_strs = binary_pcm_to_pauli_str(binary_pcm)
    expected = np.array([["IIII"], ["IIII"], ["IIII"]], dtype=str)

    np.testing.assert_array_equal(
        pauli_strs,
        expected,
        err_msg=f"\nBinary PCM:\n{binary_pcm}\nExpected:\n{expected}\nGot:\n{pauli_strs}",
    )


def test_binary_pcm_to_pauli_str_empty():
    """Test binary_pcm_to_pauli_str with an empty matrix."""
    empty_pcm = np.array([], dtype=int).reshape(0, 0)
    pauli_strs = binary_pcm_to_pauli_str(empty_pcm)

    assert pauli_strs.shape == (0, 1), f"Expected shape (0,1), got {pauli_strs.shape}"
    assert pauli_strs.size == 0
