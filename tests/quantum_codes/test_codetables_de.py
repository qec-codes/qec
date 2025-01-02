import numpy as np
import scipy
from qec.quantum_codes import CodeTablesDE


def test_k1_codes():
    """
    Test the CodeTablesDE class for quantum codes with 1 logical qubit.

    This test iterates over a range of physical qubit counts from 3 to 10,
    creates a CodeTablesDE instance for each count with 1 logical qubit,
    and verifies that the code parameters match the expected values.

    The expected values for the code parameters are stored in the k1_d list,
    where the index corresponds to the physical qubit count and the value
    at that index is the expected distance.

    Assertions:
        - The code parameters returned by qcode.get_code_parameters() should
          match the tuple (n, 1, k1_d[n]), where n is the physical qubit count.

    Prints:
        - The CodeTablesDE instance for each physical qubit count.
    """
    k1_d = [0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4]
    for n in range(3, 11):
        qcode = CodeTablesDE(physical_qubit_count=n, logical_qubit_count=1)
        print(qcode)
        assert qcode.get_code_parameters() == (n, 1, k1_d[n])


def test_14_2_5_code():
    """
    Test the CodeTablesDE class for a quantum code with 14 physical qubits,
    2 logical qubits, and a distance of 5.

    This test initializes a CodeTablesDE object with the specified parameters
    and checks if the code parameters are correctly set to (14, 2, 5).

    Assertions:
        - The code parameters returned by get_code_parameters() should be (14, 2, 5).
    """
    qcode = CodeTablesDE(physical_qubit_count=14, logical_qubit_count=2)
    print(qcode)
    assert qcode.get_code_parameters() == (14, 2, 5)


def test_16_1_6_code():
    """
    Test the CodeTablesDE class for a quantum code with 16 physical qubits, 1 logical qubit, and distance 6.

    This test initializes a CodeTablesDE object with 16 physical qubits and 1 logical qubit,
    prints the object, and asserts that the code parameters are correctly set to (16, 1, 6).

    Raises:
        AssertionError: If the code parameters do not match the expected values (16, 1, 6).
    """
    qcode = CodeTablesDE(physical_qubit_count=16, logical_qubit_count=1)
    print(qcode)
    assert qcode.get_code_parameters() == (16, 1, 6)
