import scipy.sparse
from qec.code_constructions import CSSCode
import numpy as np


class RotatedSurfaceCode(CSSCode):
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
        hx = scipy.sparse.lil_matrix((m//2, n), dtype=np.uint8)
        hz = scipy.sparse.lil_matrix((m//2, n), dtype=np.uint8)

        # Fill the hx and hz matrices for the main grid
        
        hx_i = 0
        hz_i = 0
        for j in range(lx - 1):
            for k in range(lz - 1):
                if (j+k) % 2 == 0:
                    # Set the X and Z stabilizers for the main grid
                    hx[hx_i, j * lz + k] = 1
                    hx[hx_i, j * lz + k + 1] = 1
                    hx[hx_i, j * lz + k + lz + 1] = 1
                    hx[hx_i, j * lz + k + lz] = 1
                    hx_i += 1
                else:
                    hz[hz_i, j * lz + k] = 1
                    hz[hz_i, j * lz + k + 1] = 1
                    hz[hz_i, j * lz + k + lz + 1] = 1
                    hz[hz_i, j * lz + k + lz] = 1
                    hz_i += 1

        # Add the extra stabilizers to the top of the lattice
        for j in range(0, lz - 1, 2):

            # Set the Z Pauli components
            hz[hz_i, j] = 1
            hz[hz_i, j + 1] = 1
            hz_i += 1
        
        # Add the extra stabilizers to the bottom of the lattice
      
        for j in range(1, lz - 1, 2):
            # Set the Z Pauli components
            hz[hz_i, (n - 1) - (lz - 1) + j] = 1
            hz[hz_i, (n - 1) - (lz - 1) + j + 1] = 1
            hz_i += 1
   

  
        # Add the extra stabilizers to the right of the lattice
        for j in range(0, lx - 1, 2):

                # Set the X Pauli components
                hx[hx_i, (j + 1) * lz + lz - 1] = 1
                hx[hx_i, j * lz + lz - 1] = 1
                hx_i += 1

        # Add the extra stabilizers to the left of the lattice
        for j in range(1, lx - 1, 2):

            # Set the X Pauli components
            hx[hx_i, (j + 1) * lz] = 1
            hx[hx_i, j * lz] = 1
            hx_i += 1

        # Convert hx and hz to CSR format for efficient arithmetic operations and storage
        hx = hx.tocsc()
        hz = hz.tocsc()




        print(hx.toarray())
        print(hz.toarray()) 

        # Initialize the CSSCode with the constructed hx and hz matrices
        super().__init__(hx,hz, name=f"Rotated Surface ({lx}x{lz}) Code")
        self.compute_logical_basis()

if __name__ == "__main__":

    code = RotatedSurfaceCode(7, 7)
    # code.estimate_min_distance(timeout_seconds=1.0)
    print(code)

    # from ldpc.mod2 import row_span

    # rs = row_span(np.identity(9, dtype=int))

    # for r in rs:
    #     # print(r.shape)
    #     r=r.toarray()[0]

    #     if not np.any(code.x_stabilizer_matrix@r % 2):
    #         if not np.any(code.z_stabilizer_matrix@r % 2):
    #             print("Logical", r)
        
         



