import numpy as np
import scipy.sparse
from qec.code_constructions import StabilizerCode


class PeriodicSurfaceXZZX(StabilizerCode):
    """
    Represents a Periodic Surface XZZX Code, a type of quantum error correction code.

    This code is defined on a standard surface code lattice with periodic boundary conditions.
    The stabilizers measure XZZX, and the qubits are labeled sequentially from 1 to N, row by row from left to right.

    The code's parity check matrix is defined as follows:

    H = [Hx | Hz]

    where Hz is a repetition code over N qubits, with N = lx * lz + (lx - 1) * (lz - 1).
    Hz is constructed such that the vertically measured stabilizers have periodic boundary conditions.
    Hx and Hz are swapped if noise_bias is set to "X".

    Parameters
    ----------
    lx : int
        The size of the lattice in the horizontal direction.
    lz : int
        The size of the lattice in the vertical direction.
    noise_bias : str, optional
        The type of noise bias, default is "Z". This determines which stabilizer Pauli type is defined by the repetition code that spans all qubits.

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
        # Calculate the total number of qubits
        N = lx * lz + (lx - 1) * (lz - 1)

        # Generate the parity check matrices for Z and X stabilizers based on noise bias
        if noise_bias == "X":
            hz = self._full_row_rank_shift_matrix(
                N, 0
            ) + self._full_row_rank_shift_matrix(N, 1)
            hx = self._full_row_rank_shift_matrix(
                N, lz
            ) + self._full_row_rank_shift_matrix(N, (1 - lz))
        else:  # Default to "Z" noise bias
            hz = self._full_row_rank_shift_matrix(
                N, lx
            ) + self._full_row_rank_shift_matrix(N, (1 - lx))
            hx = self._full_row_rank_shift_matrix(
                N, 0
            ) + self._full_row_rank_shift_matrix(N, 1)

        # Combine the parity check matrices into a single stabilizer matrix
        stabilizer_matrix = scipy.sparse.hstack([hx, hz])

        # Initialize the StabilizerCode with the combined parity check matrix
        super().__init__(
            stabilizer_matrix, name=f"Periodic Surface XZZX ({lx}x{lz}) Code"
        )

    def _full_row_rank_shift_matrix(
        self, n: int, shift: int
    ) -> scipy.sparse.csr_matrix:
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
