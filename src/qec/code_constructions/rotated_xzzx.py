import scipy.sparse
from qec.code_constructions import StabilizerCode


class RotatedSurfaceXZZX(StabilizerCode):
    """
    Represents a Rotated Surface XZZX Code, a type of quantum error correction code.

    This code is defined on a rotated surface code lattice of size lx x lz where all stabilizers measure XZZX.
    The boundary conditions are periodic.

    Parameters
    ----------
    lx : int
        The size of the lattice in the horizontal direction.
    lz : int
        The size of the lattice in the vertical direction.

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

    Notes
    -----
    If lx and lz are both even, then the code encodes 2 logical qubits. Otherwise, it encodes 1 logical qubit.
    The code's parity check matrix can be derived as a composition of two full-rank shift matrices.
    This code is described in https://arxiv.org/abs/2009.07851.
    """

    def __init__(self, lx: int, lz: int):
        # Calculate the total number of qubits
        n = lx * lz
        # Calculate the number of stabilizers
        m = (lx - 1) * (lz - 1) + lz // 2 + (lz - 1) // 2 + lx // 2 + (lx - 1) // 2

        # Initialize sparse matrices for hx and hz using lil_matrix for efficient construction
        hx = scipy.sparse.lil_matrix((m, n), dtype=int)
        hz = scipy.sparse.lil_matrix((m, n), dtype=int)

        # Fill the hx and hz matrices for the main grid
        for j in range(lx - 1):
            for k in range(lz - 1):
                temp = j * (lz - 1)
                # Set the X and Z stabilizers for the main grid
                hx[temp + k, j * lz + k] = 1
                hx[temp + k, j * lz + k + lz + 1] = 1

                hz[temp + k, j * lz + k + 1] = 1
                hz[temp + k, j * lz + k + lz] = 1

        # Add the extra stabilizers to the top of the lattice
        temp = (lx - 1) * (lz - 1)
        count = 0
        for j in range(0, lz - 1, 2):
            # Set the Z Pauli components
            hz[temp + count, j] = 1
            hz[temp + count, (n - 1) - (lz - 1) + 1 + j] = 1

            # Set the X Pauli components
            hx[temp + count, j + 1] = 1
            hx[temp + count, (n - 1) - (lz - 1) + j] = 1

            count += 1

        # Add the extra stabilizers to the bottom of the lattice
        temp = (lx - 1) * (lz - 1) + lz // 2
        count = 0
        for j in range(1, lz - 1, 2):
            # Set the X Pauli components
            hx[temp + count, j + 1] = 1
            hx[temp + count, (n - 1) - (lz - 1) + j] = 1

            # Set the Z Pauli components
            hz[temp + count, j] = 1
            hz[temp + count, (n - 1) - (lz - 1) + 1 + j] = 1

            count += 1

        # Add the extra stabilizers to the right of the lattice
        temp = (lx - 1) * (lz - 1) + lz // 2 + (lz - 1) // 2
        count = 0
        for j in range(0, lx - 1, 2):
            # Set the Z Pauli components
            hz[temp + count, (j + 1) * lz + lz - 1] = 1
            hz[temp + count, j * lz] = 1

            # Set the X Pauli components
            hx[temp + count, (j + 1) * lz] = 1
            hx[temp + count, j * lz + lz - 1] = 1

            count += 1

        # Add the extra stabilizers to the left of the lattice
        temp = (lx - 1) * (lz - 1) + lz // 2 + (lz - 1) // 2 + lx // 2
        count = 0
        for j in range(1, lx - 1, 2):
            # Set the Z Pauli components
            hz[temp + count, (j + 1) * lz + lz - 1] = 1
            hz[temp + count, j * lz] = 1

            # Set the X Pauli components
            hx[temp + count, (j + 1) * lz] = 1
            hx[temp + count, j * lz + lz - 1] = 1

            count += 1

        # Convert hx and hz to CSR format for efficient arithmetic operations and storage
        hx = hx.tocsc()
        hz = hz.tocsc()

        # Combine the parity check matrices into a single stabilizer matrix
        stabilizer_matrix = scipy.sparse.hstack([hx, hz]).tocsr()

        # Initialize the StabilizerCode with the combined parity check matrix
        super().__init__(stabilizer_matrix, name=f"Rotated XZZX ({lx}x{lz}) Code")
