import numpy as np
import scipy.sparse
import ldpc.mod2

from qec.utils.code_utils import pauli_str_to_binary_pcm, binary_pcm_to_pauli_str
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse


class StabiliserCode(object):
    def __init__(self, stabilisers: np.typing.ArrayLike, name: str = None):
        self.name = name if name else "stabiliser code"
        self.h = None
        self.n = None
        self.k = None
        self.d = None

        self.logicals = None

        if isinstance(stabilisers, list):
            stabilisers = np.array(stabilisers)

        if not isinstance(stabilisers, (np.ndarray, scipy.sparse.spmatrix)):
            raise TypeError(
                "Please provide either a parity check matrix or a list of Pauli stabilisers."
            )

        if stabilisers.dtype.kind in {"U", "S"}:
            self.h = pauli_str_to_binary_pcm(stabilisers)

        else:
            if stabilisers.shape[1] % 2 == 0:
                self.h = convert_to_binary_scipy_sparse(stabilisers)
            else:
                raise ValueError(
                    "The parity check matrix must have an even number of columns."
                )

        self.n = self.h.shape[1] // 2

        # Check that stabilisers commute:
        temp_product = self.h[:, : self.n] @ self.h[:, self.n :].T
        if np.any((temp_product + temp_product.T).data % 2):
            raise ValueError("The stabilisers do not commute.")

        # Compute the dimension of the code
        self.k = self.n - ldpc.mod2.rank(self.h, method="dense")

        # Compute a basis for the logical operators of the code

        self.logicals = self.compute_logical_basis()

    @property
    def pauli_stabilisers(self):
        return binary_pcm_to_pauli_str(self.h)

    @pauli_stabilisers.setter
    def pauli_stabilisers(self, pauli_stabilisers: np.ndarray):
        self.h = pauli_str_to_binary_pcm(pauli_stabilisers)

    def compute_logical_basis(self) -> np.ndarray:
        kernel_h = ldpc.mod2.kernel(self.h)

        swapped_kernel = scipy.sparse.hstack(
            [kernel_h[:, : self.n], kernel_h[:, self.n :]]
        )

        logical_stack = scipy.sparse.vstack([self.h, swapped_kernel])

        p_rows = ldpc.mod2.pivot_rows(logical_stack)

        return logical_stack[p_rows[self.n - self.k :]]

    def compute_code_distance(self) -> int:
        return NotImplemented

    def save_code(self, save_dense: bool = False):
        pass

    def load_code(self):
        pass

    def __repr__(self):
        return f"Name: {self.name}, Class: Stabiliser Code"

    def __str__(self):
        return f"<Name: {self.name}, Class: Stabiliser Code, Parameters: [[{self.n}, {self.k}, {self.d}]]>"
