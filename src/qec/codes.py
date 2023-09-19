from typing import Union
import numpy as np
import scipy
from qec.css import CssCode
from qec.hgp import HyperGraphProductCode
from ldpc2.codes import rep_code, hamming_code, ring_code


class FourTwoTwoCode(CssCode):
    def __init__(self):
        hx = np.array([[1, 1, 1, 1]])
        hz = np.array([[1, 1, 1, 1]])

        CssCode.__init__(self, hx, hz, name="422")


class SteaneCode(CssCode):
    def __init__(self, hamming_code_size=None):
        if hamming_code_size is None:
            hamming_code_size = 3

        h = hamming_code(hamming_code_size)

        CssCode.__init__(self, h, h, name=f"Steane")


class SurfaceCode(HyperGraphProductCode):
    def __init__(self, d: int):
        if type(d) is not int:
            raise TypeError("Surface code distance must be an integer.")

        if d < 2:
            raise ValueError("Surface code distance must be at least 2.")

        code = rep_code(d)
        HyperGraphProductCode.__init__(self, code, code, name="Surface")


class ToricCode(HyperGraphProductCode):
    def __init__(self, d: int):
        if type(d) is not int:
            raise TypeError("Toric code distance must be an integer.")

        if d < 2:
            raise ValueError("Toric code distance must be at least 2.")

        code = ring_code(d)
        HyperGraphProductCode.__init__(self, code, code, name="Toric")
