import numpy as np
import scipy.sparse
import ldpc.mod2
import time
from typing import Tuple, Optional

from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse
from qec.utils.binary_pauli_utils import (
    symplectic_product,
    check_binary_pauli_matrices_commute,
    pauli_str_to_binary_pcm,
    binary_pcm_to_pauli_str,
    binary_pauli_hamming_weight,
)


class StabilizerCode(object):
    """
    A quantum stabilizer code, which defines and manipulates stabilizer generators,
    computes logical operators, and stores parameters such as the number of physical qubits
    and the number of logical qubits.

    Parameters
    ----------
    stabilizers : np.typing.ArrayLike or scipy.sparse.spmatrix or list
        Either a binary parity check matrix (with an even number of columns),
        or a list of Pauli strings that specify the stabilizers of the code.
    name : str, optional
        A name for the code. Defaults to "stabilizer code".

    Attributes
    ----------
    name : str
        The name of the code.
    h : scipy.sparse.spmatrix
        The binary parity check matrix representation of the stabilizers.
    n : int
        The number of physical qubits in the code.
    k : int
        The number of logical qubits in the code.
    d : int
        (Not computed by default) The distance of the code, if known or computed.
    logicals : scipy.sparse.spmatrix or None
        A basis for the logical operators of the code.

    """

    def __init__(self, stabilizers: np.typing.ArrayLike, name: str = None):
        """
        Construct a StabilizerCode instance from either a parity check matrix or a list of
        Pauli stabilizers.

        Parameters
        ----------
        stabilizers : np.typing.ArrayLike or scipy.sparse.spmatrix or list
            Either a binary parity check matrix (with an even number of columns),
            or a list of Pauli strings that specify the stabilizers of the code.
        name : str, optional
            A name for the code. If None, it defaults to "stabilizer code".

        Raises
        ------
        TypeError
            If `stabilizers` is not an array-like, sparse matrix, or list of Pauli strings.
        ValueError
            If the parity check matrix does not have an even number of columns,
            or the stabilizers do not mutually commute.
        """
        self.name = name if name else "stabilizer code"

        self.h = None
        self.n = None
        self.k = None
        self.d = None
        self.logicals = None

        if isinstance(stabilizers, list):
            stabilizers = np.array(stabilizers)

        if not isinstance(stabilizers, (np.ndarray, scipy.sparse.spmatrix)):
            raise TypeError(
                "Please provide either a parity check matrix or a list of Pauli stabilizers."
            )

        if stabilizers.dtype.kind in {"U", "S"}:
            self.h = pauli_str_to_binary_pcm(stabilizers)
        else:
            if stabilizers.shape[1] % 2 == 0:
                self.h = convert_to_binary_scipy_sparse(stabilizers)
            else:
                raise ValueError(
                    "The parity check matrix must have an even number of columns."
                )

        self.n = self.h.shape[1] // 2

        # Check that stabilizers commute
        if not self.check_stabilizers_commute():
            raise ValueError("The stabilizers do not commute.")

        # Compute the number of logical qubits
        self.k = self.n - ldpc.mod2.rank(self.h, method="dense")

        # Compute a basis for the logical operators of the code
        self.logicals = self.compute_logical_basis()

    @property
    def pauli_stabilizers(self):
        """
        Get or set the stabilizers in Pauli string format.

        Returns
        -------
        np.ndarray
            An array of Pauli strings representing the stabilizers.
        """
        return binary_pcm_to_pauli_str(self.h)

    @pauli_stabilizers.setter
    def pauli_stabilizers(self, pauli_stabilizers: np.ndarray):
        """
        Set the stabilizers using Pauli strings.

        Parameters
        ----------
        pauli_stabilizers : np.ndarray
            An array of Pauli strings representing the stabilizers.

        Raises
        ------
        AssertionError
            If the newly set stabilizers do not commute.
        """
        self.h = pauli_str_to_binary_pcm(pauli_stabilizers)
        assert self.check_stabilizers_commute(), "The stabilizers do not commute."

    def check_stabilizers_commute(self) -> bool:
        """
        Check whether the current set of stabilizers mutually commute.

        Returns
        -------
        bool
            True if all stabilizers commute, otherwise False.
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
        commute with all stabilizers, and then identifies a subset that spans the space
        of logical operators.
        """
        kernel_h = ldpc.mod2.kernel(self.h)

        # Sort the rows of the kernel by weight
        row_weights = np.diff(kernel_h.indptr)
        sorted_rows = np.argsort(row_weights)
        kernel_h = kernel_h[sorted_rows, :]

        swapped_kernel = scipy.sparse.hstack(
            [kernel_h[:, self.n :], kernel_h[:, : self.n]]
        )

        logical_stack = scipy.sparse.vstack([self.h, swapped_kernel])
        p_rows = ldpc.mod2.pivot_rows(logical_stack)

        self.logicals = logical_stack[p_rows[self.n - self.k :]]
        basis_minimum_hamming_weight = np.min(
            binary_pauli_hamming_weight(self.logicals)
        )

        # update distance based on the minimum hamming weight of the logical operators in this basis
        if self.d is None:
            self.d = basis_minimum_hamming_weight
        elif basis_minimum_hamming_weight < self.d:
            self.d = basis_minimum_hamming_weight
        else:
            pass

        return logical_stack[p_rows[self.n - self.k :]]

    def check_valid_logical_basis(self) -> bool:
        """
        Validate that the stored logical operators form a proper logical basis for the code.

        Checks that they commute with the stabilizers, pairwise anti-commute (in the symplectic
        sense), and have full rank.

        Returns
        -------
        bool
            True if the logical operators form a valid basis, otherwise False.
        """
        try:
            assert check_binary_pauli_matrices_commute(
                self.h, self.logicals
            ), "Logical operators do not commute with stabilizers."

            logical_product = symplectic_product(self.logicals, self.logicals)
            logical_product.eliminate_zeros()
            assert (
                not logical_product.nnz == 0
            ), "The logical operators do not anti-commute with one another."

            assert (
                ldpc.mod2.rank(self.logicals, method="dense") == 2 * self.k
            ), "The logical operators do not form a basis for the code."

            assert (
                self.logicals.shape[0] == 2 * self.k
            ), "The logical operators are not linearly independent."

        except AssertionError as e:
            print(e)
            return False

        return True

    def compute_exact_code_distance(
        self, timeout: float = 0.5
    ) -> Tuple[Optional[int], float]:
        """
        Compute the distance of the code by searching through linear combinations of
        logical operators and stabilizers, returning a tuple of the minimal Hamming weight
        found and the fraction of logical operators considered before timing out.

        Parameters
        ----------
        timeout : float, optional
            The time limit (in seconds) for the exhaustive search. Default is 0.5 seconds. To obtain the exact distance, set to `np.inf`.

        Returns
        -------
        Tuple[Optional[int], float]
            A tuple containing:
            - The best-known distance of the code as an integer (or `None` if no distance was found).
            - The fraction of logical combinations considered before the search ended.

        Notes
        -----
        - We compute the row span of both the stabilizers and the logical operators.
        - For every logical operator in the logical span, we add (mod 2) each stabilizer
        in the stabilizer span to form candidate logical operators.
        - We compute the Hamming weight of each candidate operator (i.e. how many qubits
        are acted upon by the operator).
        - We track the minimal Hamming weight encountered. If `timeout` is exceeded,
        we immediately return the best distance found so far.

        Examples
        --------
        >>> code = StabilizerCode(["XZZX", "ZZXX"])
        >>> dist, fraction = code.compute_exact_code_distance(timeout=1.0)
        >>> print(dist, fraction)
        """
        start_time = time.time()

        stabilizer_span = ldpc.mod2.row_span(self.h)[1:]
        logical_span = ldpc.mod2.row_span(self.logicals)[1:]

        if self.d is None:
            distance = np.inf
        else:
            distance = self.d

        logicals_considered = 0
        total_logical_operators = stabilizer_span.shape[0] * logical_span.shape[0]

        for logical in logical_span:
            if time.time() - start_time > timeout:
                break
            for stabilizer in stabilizer_span:
                if time.time() - start_time > timeout:
                    break
                candidate_logical = logical + stabilizer
                candidate_logical.data %= 2

                hamming_weight = binary_pauli_hamming_weight(candidate_logical)[0]
                if hamming_weight < distance:
                    distance = hamming_weight
                logicals_considered += 1

        self.d = distance
        fraction_considered = logicals_considered / total_logical_operators

        return (
            (int(distance), fraction_considered)
            if distance != np.inf
            else (None, fraction_considered)
        )

    def get_code_parameters(self) -> tuple:
        """
        Return the parameters of the code as a tuple: (n, k, d).

        Returns
        -------
        tuple
            A tuple of integers representing the number of physical qubits, logical qubits,
            and the distance of the code.
        """
        return self.n, self.k, self.d

    def save_code(self, save_dense: bool = False):
        """
        Save the stabilizer code to disk.

        Parameters
        ----------
        save_dense : bool, optional
            If True, saves the parity check matrix as a dense format.
            Otherwise, saves the parity check matrix as a sparse format.
        """
        pass

    def load_code(self):
        """
        Load the stabilizer code from a saved file.
        """
        pass

    def __repr__(self):
        """
        Return an unambiguous string representation of the StabilizerCode instance.

        Returns
        -------
        str
            An unambiguous representation for debugging and development.
        """
        return f"Name: {self.name}, Class: Stabilizer Code"

    def __str__(self):
        """
        Return a string describing the stabilizer code, including its parameters.

        Returns
        -------
        str
            A human-readable string with the name, n, k, and d parameters of the code.
        """
        return f"< Stabilizer Code, Name: {self.name}, Parameters: [[{self.n}, {self.k}, {self.d}]] >"
