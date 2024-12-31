from qec.quantum_codes import FiveQubitCode


def test_params():
    qcode = FiveQubitCode()
    assert qcode.name == "5-qubit code"
    assert qcode.n == 5
    assert qcode.k == 1

    assert qcode.logicals.shape == (2, 5)


def test_print():
    qcode = FiveQubitCode()
    print(qcode)
    print(qcode.pauli_stabilisers)
    print(qcode.h.toarray())

    print("logical operators")
    print(qcode.logicals.toarray())
    
