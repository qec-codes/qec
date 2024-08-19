from typing import Union
import numpy as np
from qec.css import CssCode
from qec.hgp import HyperGraphProductCode
from ldpc.codes import rep_code, hamming_code, ring_code
from qec.lifted_hgp import LiftedHypergraphProduct
import ldpc.protograph


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

        HyperGraphProductCode.__init__(
            self, rep_code(lx), rep_code(lz), name=f"Surface ({lx}x{lz})"
        )

class RotatedSurfaceCode(CssCode):

    def __init__(self, d: int):

        if d % 2 != 1:
            raise ValueError("Rotated surface code distance must be odd.")

        hz = np.zeros((d**2//2,d**2)).astype(np.uint8)
        for k in np.arange(1,d**2+1):
            stabs = [i for i in np.array(self.__data2st_Z(k, d))-(d+1)//2 if i>0 and i <= d**2//2]
            # #print(f"Qubit {k}, connects to Z ancillas: {stabs}")
            for i in stabs:
                hz[i-1,k-1]=1

        # #print()

        hx = np.zeros((d**2//2,d**2)).astype(np.uint8)
        for k in np.arange(1,d**2+1):
            stabs = [i for i in np.array(self.__data2st_X(k, d))-(d+1)//2 if i>0 and i <= d**2//2]
            # #print(f"Qubit {k}, connects to X ancillas: {stabs}")
            for i in stabs:
                hx[i-1,k-1]=1

        super().__init__(hx,hz, name=f"Rotated Surface ({d}x{d})")
        self.D = d

    def __data2st_Z(self,k, d):
        """
        Function to relate a data qubit to the 2 Z graph vertices that enclose it.
        :int k: Index of data qubit 1..d
        :int d: distance of the surface code
        :return: The 2 syndrome graph vertices that enclose the data qubit.
        """

        row = (k - 1) // d
        column = (k - 1) % d
        if not column % 2:
            return int((k+row+1)/2), int((k+row+d)/2+1)
        else:
            if row % 2:
                return int((k + (row + 1) + 1) / 2), int((k + (row + 1) + d) / 2)
            else:
                return int((k + (row - 1) + 1) / 2), int((k + (row - 1) + d) / 2 + 2)


    def __data2st_X(self,k, d):
        """
        Function to relate a data qubit to the 2 X graph vertices that enclose it.
        :int k: Index of data qubit 1..d
        :int d: distance of the surface code
        :return: The 2 syndrome graph vertices that enclose the data qubit.
        """
        row = (k - 1) // d
        column = (k - 1) % d
        if not row % 2:
            if not column % 2:
                return int(column * (d + 1) / 2 + (row + 2) / 2), \
                    int(column * (d + 1) / 2 + (row + 2) / 2 + (d + 1) / 2)
            else:
                return int((column + 1) * (d + 1) / 2 + (row + 2) / 2), \
                    int((column + 1) * (d + 1) / 2 + (row + 2) / 2 - (d + 1) / 2)
        else:
            if not column % 2:
                return int((column + 1) * (d + 1) / 2 + (row + 1) / 2), \
                    int((column + 1) * (d + 1) / 2 + (row + 1) / 2 - (d - 1) / 2)
            else:
                return int(column * (d + 1) / 2 + (row + 1) / 2), \
                    int(column * (d + 1) / 2 + (row + 1) / 2 + (d + 3) / 2)
    

class ToricCode(HyperGraphProductCode):
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

        HyperGraphProductCode.__init__(
            self, ring_code(lx), ring_code(lz), name=f"Toric ({lx}x{lz})"
        )


class TwistedToricCode(LiftedHypergraphProduct):
    def __init__(self, nx, nz):
        self.nx = nx
        self.nz = nz
        self.N = int(2 * self.nx * self.nz)

        self.proto_1 = ldpc.protograph.array([[(0, 1)]])
        self.proto_2 = ldpc.protograph.array([[(0, nz)]])

        LiftedHypergraphProduct.__init__(
            self, self.N // 2, self.proto_2, self.proto_1, name=f"Twisted Toric ({nx},{nz})"
        )


class HQ12Code(LiftedHypergraphProduct):
    def __init__(self):

        self.proto_1 = ldpc.protograph.array(
            [
                [(0), (0), (0), (0), (0)],
                [(0), (-2), (-4), (-7), (-11)],
                [(0), (-3), (-10), (-14), (-15)],
            ]
        )

        LiftedHypergraphProduct.__init__(
            self, 16, self.proto_1, self.proto_1, name="HQ12"
        )


class HQ16Code(LiftedHypergraphProduct):
    def __init__(self):

        lift = 21

        self.proto_1 = ldpc.protograph.array(
            [
                [(0), (0), (0), (0), (0)],
                [(0), (-4), (-5), (-7), (-17)],
                [(0), (-14), (-18), (-12), (-11)],
            ]
        )

        LiftedHypergraphProduct.__init__(
            self, lift, self.proto_1, self.proto_1, name=f"HQ16"
        )


class HQ20Code(LiftedHypergraphProduct):
    def __init__(self):

        lift = 30

        self.proto_1 = ldpc.protograph.array(
            [
                [(0), (0), (0), (0), (0)],
                [(0), (-2), (-14), (-24), (-25)],
                [(0), (-16), (-11), (-14), (-13)],
            ]
        )

        LiftedHypergraphProduct.__init__(
            self, lift, self.proto_1, self.proto_1, name=f"HQ20"
        )


class HQ24Code(LiftedHypergraphProduct):
    def __init__(self):

        lift = 42

        self.proto_1 = ldpc.protograph.array(
            [
                [(0), (0), (0), (0), (0)],
                [(0), (-6), (-7), (-9), (-30)],
                [(0), (-40), (-15), (-31), (-35)],
            ]
        )

        LiftedHypergraphProduct.__init__(
            self, lift, self.proto_1, self.proto_1, name=f"HQ24"
        )


class HGP6Code(HyperGraphProductCode):
    def __init__(self):
        h = np.array(
            [
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            ],
            dtype=np.uint8,
        )

        HyperGraphProductCode.__init__(self, h, h, name="HGP6")


class HGP8Code(HyperGraphProductCode):
    def __init__(self):
        h = np.array(
            [
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )

        HyperGraphProductCode.__init__(self, h, h, name="HGP8")


class HGP10Code(HyperGraphProductCode):
    def __init__(self):
        # fmt: off
        h=np.array(
            [
                [0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
                [0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0],
                [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0],
                [1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
                [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0],
                [0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0],
                [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                [0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0],
                [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0]
            ], dtype=np.uint8)
        # fmt: on

        HyperGraphProductCode.__init__(self, h, h, name="HGP10")
