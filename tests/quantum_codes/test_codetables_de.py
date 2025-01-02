import numpy as np
import scipy
from qec.quantum_codes import CodeTablesDE


def test_k1_codes():
    k1_d = [0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4]
    for n in range(3, 11):
        qcode = CodeTablesDE(n=n, k=1)
        print(qcode)
        assert qcode.get_code_parameters() == (n, 1, k1_d[n])


def test_14_2_5_code():
    qcode = CodeTablesDE(n=14, k=2)
    print(qcode)
    assert qcode.get_code_parameters() == (14, 2, 5)


def test_16_1_6_code():
    qcode = CodeTablesDE(n=16, k=1)
    print(qcode)
    assert qcode.get_code_parameters() == (16, 1, 6)
