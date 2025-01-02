import numpy as np
import scipy
from qec.quantum_codes import CodeTablesDE


def test_k1_codes():
    k1_d = [0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4]
    for n in range(3, 11):
        qcode = CodeTablesDE(physical_qubit_count=n, logical_qubit_count=1)
        print(qcode)
        assert qcode.get_code_parameters() == (n, 1, k1_d[n])


def test_14_2_5_code():
    qcode = CodeTablesDE(physical_qubit_count=14, logical_qubit_count=2)
    print(qcode)
    assert qcode.get_code_parameters() == (14, 2, 5)


def test_16_1_6_code():
    qcode = CodeTablesDE(physical_qubit_count=16, logical_qubit_count=1)
    print(qcode)
    assert qcode.get_code_parameters() == (16, 1, 6)
