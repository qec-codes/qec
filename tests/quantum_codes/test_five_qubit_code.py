from qec.quantum_codes import FiveQubitCode


def test_params():
    qcode = FiveQubitCode()
    assert qcode.name == "5-Qubit Code"
    assert qcode.n == 5
    assert qcode.k == 1
    assert qcode.logicals.shape == (2, 10)
    assert qcode.check_valid_logical_basis()
    assert qcode.check_stabilizers_commute()


def test_print():
    qcode = FiveQubitCode()
    qcode.compute_exact_code_distance()
    print(qcode)
    print(qcode.pauli_stabilisers)
    print(qcode.h.toarray())

    print("logical operators")
    print(qcode.logicals.toarray())


def test_distance():
    qcode = FiveQubitCode()
    qcode.compute_exact_code_distance()
    assert qcode.d == 3
