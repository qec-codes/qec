from qec.quantum_codes import FiveQubitCode


def test_params():
    qcode = FiveQubitCode()
    assert qcode.name == "5-Qubit Code"
    assert qcode.physical_qubit_count == 5
    assert qcode.logical_qubit_count == 1
    assert qcode.logical_operator_basis.shape == (2, 10)
    assert qcode.check_valid_logical_basis()
    assert qcode.check_stabilizers_commute()


def test_print():
    qcode = FiveQubitCode()
    print(qcode)
    qcode.compute_exact_code_distance()
    print(qcode)
    print(qcode.pauli_stabilizers)
    print(qcode.stabilizer_matrix.toarray())

    print("logical operators")
    print(qcode.logical_operator_basis.toarray())


def test_distance():
    qcode = FiveQubitCode()
    qcode.compute_exact_code_distance()
    assert qcode.code_distance == 3
