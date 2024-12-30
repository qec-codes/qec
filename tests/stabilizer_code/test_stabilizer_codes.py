import pytest
import warnings
import numpy as np

from qec.stabilizer_code.stabilizer_code import StabiliserCode

binary_pcm = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]])

pauli_strs = np.array([["ZZZZ"], ["XXXX"]])


def test_initialisation_with_binary_pcm():
    # Initialisation from binary parity check matrix

    temp_code = StabiliserCode(stabilisers=binary_pcm)

    assert temp_code.name == "stabiliser code"
    print(temp_code.pauli_stabilisers)
    assert (temp_code.pauli_stabilisers == pauli_strs).all()
    assert (temp_code.h.toarray() == binary_pcm).all()
    assert temp_code.n == 4
    assert temp_code.k == 2
    assert temp_code.d == 2


def test_initialisation_with_pauli_strings():
    # Initialisation from pauli strings

    temp_code = StabiliserCode(stabilisers=pauli_strs)

    assert temp_code.name == "stabiliser code"
    assert (temp_code.pauli_stabilisers == pauli_strs).all()
    assert (temp_code.h.toarray() == binary_pcm).all()
    assert temp_code.n == 4
    assert temp_code.k == 2
    assert temp_code.d == 2


def test_initialisation_invalid_type():
    # Negative test for invalid initialisation inputs

    with pytest.raises(
        TypeError,
        match="Please provide either a parity check matrix or a list of Pauli stabilisers.",
    ):
        temp_code = StabiliserCode(stabilisers="not a numpy array")


def test_wrong_pcm_shape():
    # Negative test for odd number of pcm columns

    wrong_pcm = np.array([[0, 1, 1], [1, 1, 0]])

    with pytest.raises(
        ValueError, match="The parity check matrix must have an even number of columns."
    ):
        temp_code = StabiliserCode(stabilisers=wrong_pcm)


def test_non_commuting_stabilisers():
    # Negative test for non-commuting stabilisers

    non_commuting_stabilisers = np.array((["XXXX"], ["ZIII"]))

    with pytest.raises(ValueError, match="The stabilisers do not commute."):
        temp_code = StabiliserCode(stabilisers=non_commuting_stabilisers)
