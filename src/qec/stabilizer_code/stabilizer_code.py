import numpy as np
import scipy.sparse
import ldpc.mod2

from qec.utils.code_utils import pauli_str_to_binary_pcm, binary_pcm_to_pauli_str
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse
from qec.utils.symplectic_pauli_utils import symplectic_product, check_binary_pauli_matrices_commute

class StabiliserCode(object):
    """
    A quantum stabiliser code, which defines and manipulates stabiliser generators,
    computes logical operators, and stores parameters such as the number of physical qubits
    and the number of logical qubits.

    Parameters
    ----------
    stabilisers : np.typing.ArrayLike or scipy.sparse.spmatrix or list
        Either a binary parity check matrix (with an even number of columns),
        or a list of Pauli strings that specify the stabilisers of the code.
    name : str, optional
        A name for the code. Defaults to "stabiliser code".

    Attributes
    ----------
    name : str
        The name of the code.
    h : scipy.sparse.spmatrix
        The binary parity check matrix representation of the stabilisers.
    n : int
        The number of physical qubits in the code.
    k : int
        The number of logical qubits in the code.
    d : int
        (Not computed by default) The distance of the code, if known or computed.
    logicals : scipy.sparse.spmatrix or None
        A basis for the logical operators of the code.

    """

    def __init__(self, stabilisers: np.typing.ArrayLike, name: str = None):
        """
        Construct a StabiliserCode instance from either a parity check matrix or a list of
        Pauli stabilisers.

        Parameters
        ----------
        stabilisers : np.typing.ArrayLike or scipy.sparse.spmatrix or list
            Either a binary parity check matrix (with an even number of columns),
            or a list of Pauli strings that specify the stabilisers of the code.
        name : str, optional
            A name for the code. If None, it defaults to "stabiliser code".

        Raises
        ------
        TypeError
            If `stabilisers` is not an array-like, sparse matrix, or list of Pauli strings.
        ValueError
            If the parity check matrix does not have an even number of columns,
            or the stabilisers do not mutually commute.
        """
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

        # Check that stabilisers commute
        if not self.check_stabilizers_commute():
            raise ValueError("The stabilisers do not commute.")

        # Compute the number of logical qubits
        self.k = self.n - ldpc.mod2.rank(self.h, method="dense")

        # Compute a basis for the logical operators of the code
        self.logicals = self.compute_logical_basis()

    @property
    def pauli_stabilisers(self):
        """
        Get or set the stabilisers in Pauli string format.

        Returns
        -------
        np.ndarray
            An array of Pauli strings representing the stabilisers.
        """
        return binary_pcm_to_pauli_str(self.h)

    @pauli_stabilisers.setter
    def pauli_stabilisers(self, pauli_stabilisers: np.ndarray):
        """
        Set the stabilisers using Pauli strings.

        Parameters
        ----------
        pauli_stabilisers : np.ndarray
            An array of Pauli strings representing the stabilisers.

        Raises
        ------
        AssertionError
            If the newly set stabilisers do not commute.
        """
        self.h = pauli_str_to_binary_pcm(pauli_stabilisers)
        assert self.check_stabilizers_commute(), "The stabilisers do not commute."

    def check_stabilizers_commute(self) -> bool:
        """
        Check whether the current set of stabilisers mutually commute.

        Returns
        -------
        bool
            True if all stabilisers commute, otherwise False.
        """
        return check_binary_pauli_matrices_commute(self.h, self.h)

    def compute_logical_basis(self) -> np.ndarray:
        """
        Compute a basis for the logical operators of the code by extending the parity check
        matrix. The resulting basis operators are stored in `self.logicals`.

        Returns
        -------
        np.ndarray
            A basis for the logical operators in binary representation.

        Notes
        -----
        This method uses the kernel of the parity check matrix to find operators that
        commute with all stabilisers, and then identifies a subset that spans the space
        of logical operators.
        """
        kernel_h = ldpc.mod2.kernel(self.h)

        swapped_kernel = scipy.sparse.hstack(
            [kernel_h[:, self.n :], kernel_h[:, : self.n]]
        )

        logical_stack = scipy.sparse.vstack([self.h, swapped_kernel])
        p_rows = ldpc.mod2.pivot_rows(logical_stack)

        return logical_stack[p_rows[self.n - self.k :]]

    def check_valid_logical_basis(self) -> bool:
        """
        Validate that the stored logical operators form a proper logical basis for the code.

        Checks that they commute with the stabilisers, pairwise anti-commute (in the symplectic
        sense), and have full rank.

        Returns
        -------
        bool
            True if the logical operators form a valid basis, otherwise False.
        """
        try:
            assert check_binary_pauli_matrices_commute(self.h, self.logicals), \
                "Logical operators do not commute with stabilisers."

            logical_product = symplectic_product(self.logicals, self.logicals)
            logical_product.eliminate_zeros()
            assert not logical_product.nnz == 0, \
                "The logical operators do not anti-commute with one another."

            assert ldpc.mod2.rank(self.logicals, method="dense") == 2 * self.k, \
                "The logical operators do not form a basis for the code."

            assert self.logicals.shape[0] == 2 * self.k, \
                "The logical operators are not linearly independent."

        except AssertionError as e:
            print(e)
            return False

        return True

    def compute_code_distance(self) -> int:
        """
        Compute the distance of the code.

        Returns
        -------
        int
            The distance of the code.

        Notes
        -----
        This is not currently implemented and will raise `NotImplementedError` when called.
        """
        # Placeholder for future implementation
        return NotImplemented

    def save_code(self, save_dense: bool = False):
        """
        Save the stabiliser code to disk.

        Parameters
        ----------
        save_dense : bool, optional
            If True, saves the parity check matrix as a dense format.
            Otherwise, saves the parity check matrix as a sparse format.
        """
        pass

    def load_code(self):
        """
        Load the stabiliser code from a saved file.
        """
        pass

    def __repr__(self):
        """
        Return an unambiguous string representation of the StabiliserCode instance.

        Returns
        -------
        str
            An unambiguous representation for debugging and development.
        """
        return f"Name: {self.name}, Class: Stabiliser Code"

    def __str__(self):
        """
        Return a string describing the stabiliser code, including its parameters.

        Returns
        -------
        str
            A human-readable string with the name, n, k, and d parameters of the code.
        """
        return f"< Stabiliser Code, Name: {self.name}, Parameters: [[{self.n}, {self.k}, {self.d}]] >"
