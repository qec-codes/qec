from qec.code_constructions import RotatedSurfaceXZZX


def test_rotated_surface_16_2_4():
    # Test initialization with specific lx and lz
    lx, lz = 4, 4
    qcode = RotatedSurfaceXZZX(lx, lz)
    assert qcode.physical_qubit_count == 16
    assert qcode.logical_qubit_count == 2


def test_rotated_surface_9_1_3():
    # Test initialization with specific lx and lz
    lx, lz = 3, 3
    qcode = RotatedSurfaceXZZX(lx, lz)
    assert qcode.physical_qubit_count == 9
    assert qcode.logical_qubit_count == 1
