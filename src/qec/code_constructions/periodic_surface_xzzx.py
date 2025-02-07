import numpy as np
import scipy.sparse
from qec.code_constructions import StabilizerCode

class PeriodicSurfaceXZZX(StabilizerCode):
    """
    Represents a Rotated Toric XZZX Code, which is a type of quantum error correction code.

    This code is defined on a rotated surface code lattice of size dx x dz where all stabilizers measure XZZX.
    Z-Paulis are measured in the horizontal direction and X-stabilizers in the vertical direction.
    The boundary conditions are periodic.

    If the dimensions {dx, dz} are co-prime, this code encodes two logical qubits, otherwise it encodes 1.

    The code's parity check matrix can be derived as a composition of two full-rank shift matrices.

    This code is described in https://arxiv.org/abs/2009.07851

    Parameters
    ----------
    dx : int
        The size of the lattice in the horizontal direction.
    dz : int
        The size of the lattice in the vertical direction.

    Attributes
    ----------
    hx : scipy.sparse.csr_matrix
        The parity check matrix for X stabilizers.
    hz : scipy.sparse.csr_matrix
        The parity check matrix for Z stabilizers.
    """

    def __init__(self, dx: int, dz: int):
        def Pfr(n: int, shift: int) -> scipy.sparse.csr_matrix:
            """
            Generate a full-rank shift matrix.

            Parameters
            ----------
            n : int
                The size of the matrix.
            shift : int
                The shift to apply to the permutation.

            Returns
            -------
            scipy.sparse.csr_matrix
                The full-rank shift matrix.
            """
            # Create the base matrix with an identity matrix and a zero column
            base = scipy.sparse.hstack([scipy.sparse.identity(n-1), scipy.sparse.csc_matrix((n-1, 1))]).astype(np.uint8).tocsc()
            # Create the permutation array
            perm = np.arange(n)
            perm = (perm - shift) % n
            # Apply the permutation to the base matrix
            return base[:, perm]

        # Calculate the total number of qubits
        N = dx * dz + (dx - 1) * (dz - 1)
        # Generate the parity check matrices for Z and X stabilizers
        hz = Pfr(N, 0) + Pfr(N, 1)
        hx = Pfr(N, dz) + Pfr(N, (1 - dz))

        stabilizer_matrix = scipy.sparse.hstack([hx, hz])

        # Initialize the stab_code with the parity check matrices
        super().__init__(stabilizer_matrix, name=f"Rotated Surface XZZX ({dx}x{dz}) Code")