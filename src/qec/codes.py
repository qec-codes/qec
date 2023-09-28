from typing import Union
import numpy as np
from qec.css import CssCode
from qec.hgp import HyperGraphProductCode
from ldpc2.codes import rep_code, hamming_code, ring_code
from qec.lifted_hgp import LiftedHypergraphProduct


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
    def __init__(self, lx: int, lz: int = None):
        if type(lx) is not int:
            raise TypeError("Surface code lattice size `lx` must be an integer.")

        if lz is None:
            lz = lx

        if type(lz) is not int:
            raise TypeError("Surface code lattice size `lz` must be an integer.")

        if lx < 2:
            raise ValueError("Surface code lattice size `lx` must be at least 2.")

        if lz < 2:
            raise ValueError("Surface code distance must be at least 2.")

        HyperGraphProductCode.__init__(self, rep_code(lx), rep_code(lz), name=f"Surface ({lx}x{lz})")


class ToricCode(HyperGraphProductCode):
    def __init__(self, d: int):
        if type(d) is not int:
            raise TypeError("Toric code distance must be an integer.")

        if d < 2:
            raise ValueError("Toric code distance must be at least 2.")

        code = ring_code(d)
        HyperGraphProductCode.__init__(self, code, code, name=f"Toric ({d}x{d})")

class TwistedToricCode(LiftedHypergraphProduct):

    def __init__(self,nx,nz):

        self.nx=nx
        self.nz=nz
        self.N=int(2*self.nx*self.nz)
        
        self.proto_1=np.array([[{0,1}]])
        self.proto_2=np.array([[{0,nz}]])

        LiftedHypergraphProduct.__init__(self.N//2,self.proto_2,self.proto_1, name = f"Twisted Toric ({nx},{nz})")