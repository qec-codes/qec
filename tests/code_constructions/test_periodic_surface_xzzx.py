from qec.code_constructions import PeriodicSurfaceXZZX


def test_5_1_3():
    qcode = PeriodicSurfaceXZZX(lx=2, lz=2)
    qcode.compute_exact_code_distance()
    assert qcode.physical_qubit_count == 5
    assert qcode.logical_qubit_count == 1
    assert qcode.code_distance == 3
    print(qcode)


def test_25_1_3():
    qcode = PeriodicSurfaceXZZX(lx=4, lz=4)
    assert qcode.physical_qubit_count == 25
    assert qcode.logical_qubit_count == 1
    print(qcode)
