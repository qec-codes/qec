from qec.quantum_codes import FiveQubitCode


def test_params():
    qcode = FiveQubitCode()
    assert qcode.name == "5-qubit code"
    assert qcode.n == 5
    assert qcode.k == 1
