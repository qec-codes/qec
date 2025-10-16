import scipy.sparse
from qec.code_constructions import CSSCode
import numpy as np
import logging

# Suppress Matplotlib debug logs (especially font_manager)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


class RotatedSurfaceCode(CSSCode):
    """
    Represents a Rotated Surface XZZX Code, a type of quantum error correction code.

    This code is defined on a rotated surface code lattice of size L x L where all stabilizers measure XZZX.
    The boundary conditions are periodic.

    Parameters
    ----------
    L : int
        The size of the lattice in the horizontal direction.
    L : int
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
    If L and L are both even, then the code encodes 2 logical qubits. Otherwise, it encodes 1 logical qubit.
    The code's parity check matrix can be derived as a composition of two full-rank shift matrices.
    This code is described in https://arxiv.org/abs/2009.07851.
    """

    def __init__(self, L:int):
        # Calculate the total number of qubits
        n = L * L
        # Calculate the number of stabilizers
        m = (L - 1) * (L - 1) + L // 2 + (L - 1) // 2 + L // 2 + (L - 1) // 2

        # Initialize sparse matrices for hx and hz using lil_matrix for efficient construction
        hx = scipy.sparse.lil_matrix((m//2, n), dtype=np.uint8)
        hz = scipy.sparse.lil_matrix((m//2, n), dtype=np.uint8)

        # Fill the hx and hz matrices for the main grid
        
        hx_i = 0
        hz_i = 0
        for j in range(L - 1):
            for k in range(L - 1):
                if (j+k) % 2 == 0:
                    # Set the X and Z stabilizers for the main grid
                    hx[hx_i, j * L + k] = 1
                    hx[hx_i, j * L + k + 1] = 1
                    hx[hx_i, j * L + k + L + 1] = 1
                    hx[hx_i, j * L + k + L] = 1
                    hx_i += 1
                else:
                    hz[hz_i, j * L + k] = 1
                    hz[hz_i, j * L + k + 1] = 1
                    hz[hz_i, j * L + k + L + 1] = 1
                    hz[hz_i, j * L + k + L] = 1
                    hz_i += 1

        # Add the extra stabilizers to the top of the lattice
        for j in range(0, L - 1, 2):

            # Set the Z Pauli components
            hz[hz_i, j] = 1
            hz[hz_i, j + 1] = 1
            hz_i += 1
        
        # Add the extra stabilizers to the bottom of the lattice
      
        for j in range(1, L - 1, 2):
            # Set the Z Pauli components
            hz[hz_i, (n - 1) - (L - 1) + j] = 1
            hz[hz_i, (n - 1) - (L - 1) + j + 1] = 1
            hz_i += 1
  
        # Add the extra stabilizers to the left of the lattice
        for j in range(1, L - 1, 2):

            # Set the Z Pauli components
            hx[hx_i, (j + 1) * L] = 1
            hx[hx_i, j * L] = 1
            hx_i += 1

        # Add the extra stabilizers to the right of the lattice
        for j in range(0, L - 1, 2):

            # Set the X Pauli components
            hx[hx_i, (j + 1) * L + L - 1] = 1
            hx[hx_i, j * L + L - 1] = 1
            hx_i += 1


        # hx=hx.toarray()
        # hz = hz.toarray()

        # print(hx)
        # print(hz)

        # print(hx@hz.T % 2)  # Should be all zeros


        # Convert hx and hz to CSR format for efficient arithmetic operations and storage
        hx = hx.tocsc()
        hz = hz.tocsc()

    


        # # Initialize the CSSCode with the constructed hx and hz matrices
        super().__init__(hx,hz, name=f"Rotated Surface ({L}x{L}) Code")
        self.x_code_distance = self.z_code_distance = self.code_distance = L

    def get_node_coordinates(self):

        L = self.code_distance

        self.qubit_coordinates = []
        self.x_check_coordinates = []
        self.z_check_coordinates = []

        for i in range(L):
            for j in range(L):
                # self.qubit_coordinates.append( (i,j) )
    
                self.qubit_coordinates.append( (j,(L-1)-(i)) )
       

        for i in range(L - 1):
            for j in range(L - 1):
                # self.check_coordinates.append( (i+0.5,j+0.5) )
                if (i+j) % 2 == 0:
                    self.x_check_coordinates.append( (j+0.5,(L-1)-(i+0.5)) )
                else:
                    self.z_check_coordinates.append( (j+0.5,(L-1)-(i+0.5)) )
                    
        # Add the extra stabilizers to the top of the lattice
        for j in range(0, L - 1, 2):
            
            # self.check_coordinates.append( (L-1+0.5,j+0.5) )
            self.z_check_coordinates.append( (j+0.5,(L-1+0.5)) )

        # Add the extra stabilizers to the bottom of the lattice
        for j in range(1, L - 1, 2):
            # self.check_coordinates.append( (-0.5,j+0.5) )
            self.z_check_coordinates.append( (j+0.5,(-0.5)) )


        # Add the extra stabilizers to the left of the lattice
        for j in range(1, L - 1, 2):
            # Set the Z Pauli components
            # self.check_coordinates.append( (j+0.5,L-1+0.5) )
            self.x_check_coordinates.append( (-0.5,(L-1)-(j+0.5)) )


        # Add the extra stabilizers to the right of the lattice
        for j in range(0, L - 1, 2):

            # self.check_coordinates.append( (j+0.5,-0.5) )
            self.x_check_coordinates.append( (L-0.5,(L-1)-(j+0.5)) )

    def get_x_edge_coordinates(self):
        self.x_edge_coordinates = []
        hx = self.x_stabilizer_matrix.toarray()
        for i in range(hx.shape[0]):
            for j in range(hx.shape[1]):
                if hx[i,j]:
                    self.x_edge_coordinates.append( (self.qubit_coordinates[j], self.x_check_coordinates[i]) )

    def get_z_edge_coordinates(self):
        self.z_edge_coordinates = []
        hz = self.z_stabilizer_matrix.toarray()
        for i in range(hz.shape[0]):
            for j in range(hz.shape[1]):
                if hz[i,j]:
                    self.z_edge_coordinates.append( (self.qubit_coordinates[j], self.z_check_coordinates[i]) )

if __name__ == "__main__":

    code = RotatedSurfaceCode(11)
    # code.estimate_min_distance(timeout_seconds=1.0)
    print(code)


