import numpy as np
import scipy.sparse
from qec.code_constructions import StabilizerCode


class PeriodicSurfaceXZZX(StabilizerCode):
    """
    Represents a Periodic Surface XZZX Code, which is a type of quantum error correction code.

    This code is defined on a standard surface code lattice with periodic boundary conditions.
    The stabilizers measure XZZX, and the qubits are labeled 1..N, row by row from left to right.

    The code's parity check matrix is defined as follows:

    H = [Hx | Hz]

    where Hz is a repetition code over N qubits, where N = lx * lz + (lx - 1) * (lz - 1),
    and Hz is such that the vertically measured stabilizers have periodic boundary conditions.

    Parameters
    ----------
    lx : int
        The size of the lattice in the horizontal direction.
    lz : int
        The size of the lattice in the vertical direction.
    noise_bias : str, optional
        The type of noise bias, default is "Z".

    Attributes
    ----------
    hx : scipy.sparse.csr_matrix
        The parity check matrix for X stabilizers.
    hz : scipy.sparse.csr_matrix
        The parity check matrix for Z stabilizers.
    stabilizer_matrix : scipy.sparse.csr_matrix
        The combined parity check matrix [Hx | Hz].
    name : str
        The name of the code.
    """

    def __init__(self, lx: int, lz: int, noise_bias: str = "Z"):
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
            base = (
                scipy.sparse.hstack(
                    [scipy.sparse.identity(n - 1), scipy.sparse.csc_matrix((n - 1, 1))]
                )
                .astype(np.uint8)
                .tocsc()
            )
            # Create the permutation array
            perm = np.arange(n)
            perm = (perm - shift) % n
            # Apply the permutation to the base matrix
            return base[:, perm]

        # Calculate the total number of qubits
        N = lx * lz + (lx - 1) * (lz - 1)
        # Generate the parity check matrices for Z and X stabilizers
        hz = Pfr(N, 0) + Pfr(N, 1)
        hx = Pfr(N, lz) + Pfr(N, (1 - lz))

        # Combine the parity check matrices into a single stabilizer matrix
        stabilizer_matrix = scipy.sparse.hstack([hx, hz])

        # Initialize the StabilizerCode with the combined parity check matrix
        super().__init__(
            stabilizer_matrix, name=f"Periodic Surface XZZX ({lx}x{lz}) Code"
        )
