import numpy as np
import scipy.sparse
from tqdm import tqdm
from ldpc import BpOsdDecoder
import ldpc.mod2
import time
from typing import Tuple, Optional, Union, Sequence
import itertools

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

    def __init__(self, stabilizers: Union[np.ndarray, scipy.sparse.spmatrix, list], name: str = None):
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

        if isinstance(stabilizers, np.ndarray) and stabilizers.dtype.kind in {"U", "S"}:
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

    def compute_logical_basis(self) -> scipy.sparse.spmatrix:
        """
        Compute a basis for the logical operators of the code by extending the parity check
        matrix. The resulting basis operators are stored in `self.logicals`.

        Returns
        -------
        scipy.sparse.spmatrix
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
            [kernel_h[:, self.n:], kernel_h[:, :self.n]]
        )

        logical_stack = scipy.sparse.vstack([self.h, swapped_kernel])
        p_rows = ldpc.mod2.pivot_rows(logical_stack)

        self.logicals = logical_stack[p_rows[self.h.shape[0]:]]
        basis_minimum_hamming_weight = np.min(
            binary_pauli_hamming_weight(self.logicals)
        )

        # Update distance based on the minimum hamming weight of the logical operators in this basis
        if self.d is None:
            self.d = basis_minimum_hamming_weight
        elif basis_minimum_hamming_weight < self.d:
            self.d = basis_minimum_hamming_weight
        else:
            pass

        return logical_stack[p_rows[self.h.shape[0]:]]

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

    def reduce_logical_operator_basis(
        self, candidate_logicals: Union[Sequence, np.ndarray, scipy.sparse.spmatrix] = []
    ):
        """
        Reduce the logical operator basis to include lower-weight logicals.

        Parameters
        ----------
        candidate_logicals : Union[Sequence, np.ndarray, scipy.sparse.spmatrix], optional
            A list or array of candidate logical operators to be considered for reducing the basis.
            Defaults to an empty list.
        """
        if len(candidate_logicals) != 0:
            # Convert candidates to a sparse matrix if they aren't already
            if not isinstance(candidate_logicals, scipy.sparse.spmatrix):
                candidate_logicals = scipy.sparse.csr_matrix(
                    np.array(candidate_logicals)
                )

            # Stack the candidate logicals with the existing logicals
            temp1 = scipy.sparse.vstack([candidate_logicals, self.logicals]).tocsr()

            # Compute the Hamming weight over GF4 (number of qubits with non-identity operators)
            # Split into X and Z parts
            X_part = temp1[:, :self.n]
            Z_part = temp1[:, self.n:]

            # Perform element-wise maximum to simulate GF4 logical OR
            logical_or = X_part.maximum(Z_part)

            # Count the number of qubits with non-zero operators (Hamming weight over GF4)
            row_weights = logical_or.getnnz(axis=1)

            # Sort the rows by Hamming weight (ascending)
            sorted_rows = np.argsort(row_weights)
            temp1 = temp1[sorted_rows, :]

            # Combine with the stabilizer matrix
            temp = scipy.sparse.vstack([self.h, temp1]).tocsr()

            # Perform row reduction to find a new logical basis
            p_rows = ldpc.mod2.pivot_rows(temp)
            self.logicals = temp[p_rows[self.h.shape[0]: self.h.shape[0] + self.k]]

    def estimate_min_distance(
        self,
        timeout_seconds: float = 0.25,
        p: float = 0.25,
        max_iter: int = 10,
        error_rate: float = 0.1,
        bp_method: str = "ms",
        schedule: str = "parallel",
        ms_scaling_factor: float = 1.0,
        osd_method: str = "osd_0",
        osd_order: int = 0,
        reduce_logical_basis: bool = False,
    ) -> int:
        """
        Estimate the minimum distance of the stabilizer code using a BP+OSD decoder-based search.

        Parameters
        ----------
        timeout_seconds : float, optional
            The time limit (in seconds) for searching random linear combinations.
        p : float, optional
            Probability used to randomly include or exclude each logical operator
            when generating trial logical operators.
        max_iter : int, optional
            Maximum number of BP decoder iterations.
        error_rate : float, optional
            Crossover probability for the BP+OSD decoder.
        bp_method : str, optional
            Belief Propagation method (e.g., "ms" for min-sum).
        schedule : str, optional
            Update schedule for BP (e.g., "parallel").
        ms_scaling_factor : float, optional
            Scaling factor for min-sum updates.
        osd_method : str, optional
            Order-statistic decoding method (e.g., "osd_0").
        osd_order : int, optional
            OSD order.
        reduce_logical_basis : bool, optional
            If True, attempts to reduce the logical operator basis to include lower-weight operators.

        Returns
        -------
        int
            The best-known estimate of the code distance found within the time limit.
        """
        if self.logicals is None:
            self.logicals = self.compute_logical_basis()

        # Build a stacked matrix of stabilizers and logicals
        # Stabilizers: rows 0..(h.shape[0]-1)
        # Logicals: rows h.shape[0]..(h.shape[0] + logicals.shape[0] - 1)
        stack = scipy.sparse.vstack([self.h, self.logicals]).tocsr()

        # Initial distance estimate from the current logicals
        if self.d is None:
            min_distance = np.min(binary_pauli_hamming_weight(self.logicals))
        else:
            min_distance = self.d

        # Set up BP+OSD decoder
        bp_osd = BpOsdDecoder(
            stack,
            error_rate=error_rate,
            max_iter=max_iter,
            bp_method=bp_method,
            schedule=schedule,
            ms_scaling_factor=ms_scaling_factor,
            osd_method=osd_method,
            osd_order=osd_order,
        )

        # List to store candidate logical operators for basis reduction
        candidate_logicals = []

        # 1) First, try each logical operator individually
        for i in range(self.logicals.shape[0]):
            dummy_syndrome = np.zeros(stack.shape[0], dtype=np.uint8)
            dummy_syndrome[self.h.shape[0] + i] = 1  # pick exactly one logical operator
            candidate = bp_osd.decode(dummy_syndrome)
            # Calculate symplectic weight: number of qubits where either X or Z is present
            w = np.count_nonzero(candidate[: self.n] | candidate[self.n :])
            if w < min_distance:
                min_distance = w
                if reduce_logical_basis:
                    candidate_logicals.append(candidate)

        # 2) Randomly search for better representatives of logical operators
        start_time = time.time()
        with tqdm(total=timeout_seconds, desc="Estimating distance") as pbar:
            while time.time() - start_time < timeout_seconds:
                elapsed = time.time() - start_time
                # Update progress bar based on elapsed time
                pbar.update(elapsed - pbar.n)

                # Randomly pick a combination of logical rows
                # (with probability p, set the corresponding row in the syndrome to 1)
                random_syndrome = np.zeros(stack.shape[0], dtype=np.uint8)
                while True:
                    random_mask = np.random.choice([0, 1], size=self.logicals.shape[0], p=[1 - p, p])
                    if np.any(random_mask):
                        break
                for idx, bit in enumerate(random_mask):
                    if bit == 1:
                        random_syndrome[self.h.shape[0] + idx] = 1

                candidate = bp_osd.decode(random_syndrome)
                w = np.count_nonzero(candidate[: self.n] | candidate[self.n :])
                if w < min_distance:
                    min_distance = w
                    if reduce_logical_basis:
                        candidate_logicals.append(candidate)

                pbar.set_description(
                    f"Estimating distance: min-weight found <= {min_distance}, time: {elapsed:.1f}/{timeout_seconds:.1f}s"
                )

        # 3) If requested, reduce the logical operator basis to include lower-weight operators
        if reduce_logical_basis and len(candidate_logicals) > 0:
            self.reduce_logical_operator_basis(candidate_logicals)

        # Update and return the estimated distance
        self.d = min_distance
        return min_distance
