import pytest
from qec.code_constructions import ToricCode


def test_toric_code_initialization():
    # Test initialization with both dx and dz specified
    code = ToricCode(dx=5, dz=7)
    assert code.x_code_distance == 5
    assert code.z_code_distance == 7
    assert code.code_distance == 5
    assert code.name == "(5x7)-Toric Code"
    assert code.logical_qubit_count == 2

    # Test initialization with only dx specified
    code = ToricCode(dx=5)
    assert code.x_code_distance == 5
    assert code.z_code_distance == 5
    assert code.code_distance == 5
    assert code.name == "(5x5)-Toric Code"
    assert code.logical_qubit_count == 2

    # Test initialization with only dz specified
    code = ToricCode(dz=7)
    assert code.x_code_distance == 7
    assert code.z_code_distance == 7
    assert code.code_distance == 7
    assert code.name == "(7x7)-Toric Code"
    assert code.logical_qubit_count == 2

    # Test initialization with neither dx nor dz specified (should raise ValueError)
    with pytest.raises(ValueError, match="Please specify dx or dz"):
        ToricCode()


if __name__ == "__main__":
    pytest.main()
