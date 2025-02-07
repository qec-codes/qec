import pytest
from qec.code_constructions import PeriodicSurfaceXZZX


def test_surface_code_initialization():
    # Test initialization with both dx and dz specified
    qcode = PeriodicSurfaceXZZX(dx=5, dz=5)
    # qcode.compute_exact_code_distance(1)
    print(qcode)
    # assert qcode.code_distance == 2
