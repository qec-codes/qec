import ldpc.codes.rep_code

import ldpc.codes
from qec.code_constructions import HypergraphProductCode

class SurfaceCode(HypergraphProductCode):
    """
    Represents a Surface Code, which is a type of quantum error correction code.

    The Surface Code is constructed using two repetition codes, one for the X stabilizers
    and one for the Z stabilizers. The distances dx and dz specify the code distances
    for the X and Z stabilizers, respectively, corresponding to the height and width of the lattice.

    The Surface Code is a specific instance of a Hypergraph Product Code. The hypergraph product
    construction allows for the creation of quantum error correction codes with desirable properties
    such as high distance and low-weight stabilizers. In this construction, two classical codes are
    used to generate the quantum code. For the Surface Code, these classical codes are simple repetition
    codes.

    Parameters
    ----------
    dx : int, optional
        The code distance for the X stabilizers (width of the lattice). If not specified, it will be set to the value of dz.
    dz : int, optional
        The code distance for the Z stabilizers (height of the lattice). If not specified, it will be set to the value of dx.

    Raises
    ------
    ValueError
        If both dx and dz are not specified.

    Attributes
    ----------
    x_code_distance : int
        The code distance for X errors.
    z_code_distance : int
        The code distance for Z errors.
    code_distance : int
        The minimum of x_code_distance and z_code_distance.

    Notes
    -----
    The Surface Code is constructed using the hypergraph product of two repetition codes. The repetition
    code is a simple classical code where each bit is repeated multiple times to provide redundancy. By
    taking the hypergraph product of two such codes, we obtain a quantum code with stabilizers that are
    low-weight and a code distance that is determined by the distances of the original repetition codes.
    """

    def __init__(self, dx: int = None, dz: int = None):

        if dx is None and dz is None:
            raise ValueError("Please specify dx or dz")
        if dx is None:
            dx = dz
        if dz is None:
            dz = dx

        h1 = ldpc.codes.rep_code(dz)
        h2 = ldpc.codes.rep_code(dx)

        super().__init__(h1, h2, name=f"({dx}x{dz})-Surface Code")

        self.x_code_distance = dx
        self.z_code_distance = dz

        self.code_distance = min(self.x_code_distance, self.z_code_distance)