import numpy as np 
import scipy.sparse
import ldpc.mod2

from qec.codes.code_utils import pauli_str_to_binary_pcm, binary_pcm_to_pauli_str 

class StabiliserCode():

    def __init__(
            self,
            stabilisers: np.ndarray[int | str],
            name: str = None
    ):
        self.name = name if name else "stabiliser code"
        self.pauli_stabilisers = None
        self.h = None
        self.n = None
        self.k = None
        self.d = None

        self.x_logicals = None
        self.z_logicals = None

        if isinstance(stabilisers, np.ndarray):

            if isinstance(stabilisers.dtype.type(), str):
                self.pauli_stabilisers = stabilisers 
                self.h = pauli_str_to_binary_pcm(stabilisers)

            elif isinstance(stabilisers.dtype.type(), np.integer):
                if stabilisers.shape[1] % 2 == 0:
                    self.pauli_stabilisers = binary_pcm_to_pauli_str(stabilisers)
                    self.h = scipy.sparse.csr_matrix(stabilisers)
                else:
                    raise ValueError("The parity check matrix must have an even number of columns.")

        else:
            raise TypeError("Please provide either a parity check matrix (np.ndarray[int]) or an array of Pauli strings \
(np.ndarray[str]).")

        self.n = self.h.shape[1] // 2

        # Check that stabilisers commute:
        temp_product = self.h[:, :self.n] @ self.h[:, self.n:].T
        if np.any((temp_product + temp_product.T).data % 2):
            raise ValueError("The stabilisers do not commute.")

        self.k = self.n - ldpc.mod2.rank(self.h, method = "dense")
        
        #TO-DO
        if self.h.shape[1] <= 15:
            self.d = ldpc.mod2.compute_exact_code_distance(pcm = self.h)
        else:
            self.d, _, _ = ldpc.mod2.estimate_code_distance(pcm = self.h, timeout_seconds = 0.5)
            raise UserWarning("The code distance is estimated, not exact. This is due to the parity-check matrix of the stabilizer code being too large.")

    def compute_logical_basis(self) -> np.ndarray:

        kernel_h = ldpc.mod2.kernel(self.h)

        swapped_kernel = scipy.sparse.hstack([kernel_h[:, :self.n], kernel_h[:, self.n:]]) 

        logical_stack = scipy.sparse.vstack([self.h, swapped_kernel])

        p_rows = ldpc.mod2.pivot_rows(logical_stack)

        return logical_stack[p_rows[self.n - self.k :]]

    def save_code(self, save_dense : bool = False):
        pass

    def load_code(self):
        pass

    def __repr__(self):
        return f"Name: {self.name}, Class: Stabiliser Code"
                 
    def __str__(self):
        return f"Name: {self.name}, \\ \
                 Class: Stabiliser Code \\ \
                 Parameters: [[{self.n, self.k, self.d}]]"
