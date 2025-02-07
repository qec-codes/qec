from qec.code_constructions import PeriodicSurfaceXZZX


def test_surface_code_initialization():
    qcode = PeriodicSurfaceXZZX(lx=2, lz=2)
    qcode.estimate_min_distance()
    assert qcode.physical_qubit_count == 5
    assert qcode.logical_qubit_count == 1
    assert qcode.code_distance == 3
