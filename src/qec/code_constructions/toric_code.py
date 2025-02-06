
import ldpc.codes
from qec.code_constructions import HypergraphProductCode


class ToricCode(HypergraphProductCode):
    """
    Represents a Toric Code, which is a type of quantum error correction code.

    The Toric Code is constructed using two ring codes, one for the X stabilizers
    and one for the Z stabilizers. The distances dx and dz specify the code distances
    for the X and Z stabilizers, respectively, corresponding to the height and width of the lattice.

    The Toric Code is defined on a torus and is the generalization of the Surface Code with periodic
    boundary conditions. The Toric Code has a logical qubit count of 2.

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
        The code distance the X errors.
    z_code_distance : int
        The code distance for Z errors.
    code_distance : int
        The minimum of x_code_distance and z_code_distance.
    logical_qubit_count : int
        The number of logical qubits in the code, which is 2 for the Toric Code.
    """

    def __init__(self, dx: int = None, dz: int = None):
        if dx is None and dz is None:
            raise ValueError("Please specify dx or dz")
        if dx is None:
            dx = dz
        if dz is None:
            dz = dx

        h1 = ldpc.codes.ring_code(dz)
        h2 = ldpc.codes.ring_code(dx)

        super().__init__(h1, h2, name=f"({dx}x{dz})-Toric Code")

        self.x_code_distance = dx
        self.z_code_distance = dz

        self.code_distance = min(self.x_code_distance, self.z_code_distance)
