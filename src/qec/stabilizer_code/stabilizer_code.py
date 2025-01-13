from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse
from qec.utils.binary_pauli_utils import (
    symplectic_product,
    check_binary_pauli_matrices_commute,
    pauli_str_to_binary_pcm,
    binary_pcm_to_pauli_str,
    binary_pauli_hamming_weight,
)

import numpy as np
import scipy.sparse
from tqdm import tqdm
from ldpc import BpOsdDecoder
import ldpc.mod2
import time
from typing import Tuple, Optional, Union, Sequence
import logging

logging.basicConfig(level=logging.DEBUG)


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
    stabilizer_matrix : scipy.sparse.spmatrix
        The binary parity check matrix representation of the stabilizers.
    phyical_qubit_count : int
        The number of physical qubits in the code.
    logical_qubit_count : int
        The number of logical qubits in the code.
    code_distance : int
        (Not computed by default) The distance of the code, if known or computed.
    logicals : scipy.sparse.spmatrix or None
        A basis for the logical operators of the code.
    """

    def __init__(
        self,
        stabilizers: Union[np.ndarray, scipy.sparse.spmatrix, list],
        name: str = None,
    ):
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

        self.stabilizer_matrix = None
        self.physical_qubit_count = None
        self.logical_qubit_count = None
        self.code_distance = None
        self.logical_operator_basis = None

        if isinstance(stabilizers, list):
            stabilizers = np.array(stabilizers)

        if not isinstance(stabilizers, (np.ndarray, scipy.sparse.spmatrix)):
            raise TypeError(
                "Please provide either a parity check matrix or a list of Pauli stabilizers."
            )

        if isinstance(stabilizers, np.ndarray) and stabilizers.dtype.kind in {"U", "S"}:
            self.stabilizer_matrix = pauli_str_to_binary_pcm(stabilizers)
        else:
            if stabilizers.shape[1] % 2 == 0:
                self.stabilizer_matrix = convert_to_binary_scipy_sparse(stabilizers)
            else:
                raise ValueError(
                    "The parity check matrix must have an even number of columns."
                )

        self.physical_qubit_count = self.stabilizer_matrix.shape[1] // 2

        # Check that stabilizers commute
        if not self.check_stabilizers_commute():
            raise ValueError("The stabilizers do not commute.")

        # Compute the number of logical qubits
        self.logical_qubit_count = self.physical_qubit_count - ldpc.mod2.rank(
            self.stabilizer_matrix, method="dense"
        )

        # Compute a basis for the logical operators of the code
        self.logical_operator_basis = self.compute_logical_basis()

    @property
    def pauli_stabilizers(self):
        """
        Get or set the stabilizers in Pauli string format.

        Returns
        -------
        np.ndarray
            An array of Pauli strings representing the stabilizers.
        """
        return binary_pcm_to_pauli_str(self.stabilizer_matrix)

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
        self.stabilizer_matrix = pauli_str_to_binary_pcm(pauli_stabilizers)
        if not self.check_stabilizers_commute():
            raise ValueError("The stabilizers do not commute.")

    def check_stabilizers_commute(self) -> bool:
        """
        Check whether the current set of stabilizers mutually commute.

        Returns
        -------
        bool
            True if all stabilizers commute, otherwise False.
        """
        return check_binary_pauli_matrices_commute(
            self.stabilizer_matrix, self.stabilizer_matrix
        )

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
        kernel_h = ldpc.mod2.kernel(self.stabilizer_matrix)

        # Sort the rows of the kernel by weight
        row_weights = np.diff(kernel_h.indptr)
        sorted_rows = np.argsort(row_weights)
        kernel_h = kernel_h[sorted_rows, :]

        swapped_kernel = scipy.sparse.hstack(
            [
                kernel_h[:, self.physical_qubit_count :],
                kernel_h[:, : self.physical_qubit_count],
            ]
        )

        logical_stack = scipy.sparse.vstack([self.stabilizer_matrix, swapped_kernel])
        p_rows = ldpc.mod2.pivot_rows(logical_stack)

        self.logical_operator_basis = logical_stack[
            p_rows[self.stabilizer_matrix.shape[0] :]
        ]

        if self.logical_operator_basis.nnz == 0:
            self.code_distance = np.inf
            return self.logical_operator_basis

        basis_minimum_hamming_weight = np.min(
            binary_pauli_hamming_weight(self.logical_operator_basis).flatten()
        )

        # Update distance based on the minimum hamming weight of the logical operators in this basis
        if self.code_distance is None:
            self.code_distance = basis_minimum_hamming_weight
        elif basis_minimum_hamming_weight < self.code_distance:
            self.code_distance = basis_minimum_hamming_weight
        else:
            pass

        return logical_stack[p_rows[self.stabilizer_matrix.shape[0] :]]

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
                self.stabilizer_matrix, self.logical_operator_basis
            ), "Logical operators do not commute with stabilizers."

            logical_product = symplectic_product(
                self.logical_operator_basis, self.logical_operator_basis
            )
            logical_product.eliminate_zeros()
            assert (
                logical_product.nnz != 0
            ), "The logical operators do not anti-commute with one another."

            assert (
                ldpc.mod2.rank(self.logical_operator_basis, method="dense")
                == 2 * self.logical_qubit_count
            ), "The logical operators do not form a basis for the code."

            assert (
                self.logical_operator_basis.shape[0] == 2 * self.logical_qubit_count
            ), "The logical operators are not linearly independent."

        except AssertionError as e:
            logging.error(e)
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

        stabilizer_span = ldpc.mod2.row_span(self.stabilizer_matrix)[1:]
        logical_span = ldpc.mod2.row_span(self.logical_operator_basis)[1:]

        if self.code_distance is None:
            distance = np.inf
        else:
            distance = self.code_distance

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

        self.code_distance = distance
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
        return self.physical_qubit_count, self.logical_qubit_count, self.code_distance

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
        return f"< Stabilizer Code, Name: {self.name}, Parameters: [[{self.physical_qubit_count}, {self.logical_qubit_count}, {self.code_distance}]] >"

    def reduce_logical_operator_basis(
        self,
        candidate_logicals: Union[Sequence, np.ndarray, scipy.sparse.spmatrix] = [],
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
                    scipy.sparse.csr_matrix(candidate_logicals)
                )

            # Stack the candidate logicals with the existing logicals
            temp1 = scipy.sparse.vstack(
                [candidate_logicals, self.logical_operator_basis]
            ).tocsr()

            # Compute the Hamming weight over GF4 (number of qubits with non-identity operators)
            # Split into X and Z parts
            row_weights = binary_pauli_hamming_weight(temp1).flatten()

            # Sort the rows by Hamming weight (ascending)
            sorted_rows = np.argsort(row_weights)
            temp1 = temp1[sorted_rows, :]

            # Add the stabilizer matrix to the top of the stack
            temp1 = scipy.sparse.vstack([self.stabilizer_matrix, temp1])

            # Calculate the rank of the stabilizer matrix (todo: find way of removing this step)
            stabilizer_rank = ldpc.mod2.rank(self.stabilizer_matrix)

            # Perform row reduction to find a new logical basis
            p_rows = ldpc.mod2.pivot_rows(temp1)
            self.logical_operator_basis = temp1[p_rows[stabilizer_rank:]]

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
        if self.logical_operator_basis is None:
            self.logical_operator_basis = self.compute_logical_basis()

        def decoder_setup():
            # # Remove redundnant rows from stabilizer matrix
            p_rows = ldpc.mod2.pivot_rows(self.stabilizer_matrix)
            full_rank_stabilizer_matrix = self.stabilizer_matrix[p_rows]
            # full_rank_stabilizer_matrix = self.stabilizer_matrix

            # Build a stacked matrix of stabilizers and logicals
            stack = scipy.sparse.vstack(
                [full_rank_stabilizer_matrix, self.logical_operator_basis]
            ).tocsr()

            # Initial distance estimate from the current logicals

            min_distance = np.min(
                binary_pauli_hamming_weight(self.logical_operator_basis)
            )

            max_distance = np.max(self.logical_basis_weights())

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

            return (
                bp_osd,
                stack,
                full_rank_stabilizer_matrix,
                min_distance,
                max_distance,
            )

        # setup the decoder
        bp_osd, stack, full_rank_stabilizer_matrix, min_distance, max_distance = (
            decoder_setup()
        )

        # List to store candidate logical operators for basis reduction
        candidate_logicals = []

        # 2) Randomly search for better representatives of logical operators
        start_time = time.time()
        with tqdm(total=timeout_seconds, desc="Estimating distance") as pbar:
            weight_one_syndromes_searched = 0
            while time.time() - start_time < timeout_seconds:
                elapsed = time.time() - start_time
                # Update progress bar based on elapsed time
                pbar.update(elapsed - pbar.n)

                # Initialize an empty dummy syndrome
                dummy_syndrome = np.zeros(stack.shape[0], dtype=np.uint8)

                if weight_one_syndromes_searched < self.logical_operator_basis.shape[0]:
                    dummy_syndrome[
                        full_rank_stabilizer_matrix.shape[0]
                        + weight_one_syndromes_searched
                    ] = 1  # pick exactly one logical operator
                    weight_one_syndromes_searched += 1

                else:
                    # Randomly pick a combination of logical rows
                    # (with probability p, set the corresponding row in the syndrome to 1)
                    while True:
                        random_mask = np.random.choice(
                            [0, 1],
                            size=self.logical_operator_basis.shape[0],
                            p=[1 - p, p],
                        )
                        if np.any(random_mask):
                            break
                    for idx, bit in enumerate(random_mask):
                        if bit == 1:
                            dummy_syndrome[self.stabilizer_matrix.shape[0] + idx] = 1

                candidate = bp_osd.decode(dummy_syndrome)

                w = np.count_nonzero(
                    candidate[: self.physical_qubit_count]
                    | candidate[self.physical_qubit_count :]
                )

                if w < min_distance:
                    min_distance = w
                if w < max_distance:
                    if reduce_logical_basis:
                        lc = np.hstack(
                            [
                                candidate[self.physical_qubit_count :],
                                candidate[: self.physical_qubit_count],
                            ]
                        )
                        candidate_logicals.append(lc)

                # 3) If requested, reduce the logical operator basis to include lower-weight operators
                if (
                    len(candidate_logicals) >= self.logical_qubit_count
                    and reduce_logical_basis
                ):
                    self.reduce_logical_operator_basis(candidate_logicals)
                    (
                        bp_osd,
                        stack,
                        full_rank_stabilizer_matrix,
                        min_distance,
                        max_distance,
                    ) = decoder_setup()
                    candidate_logicals = []
                    weight_one_syndromes_searched = 0

                pbar.set_description(
                    f"Estimating distance: min-weight found <= {min_distance}, basis weights: {self.logical_basis_weights()}"
                )

        if reduce_logical_basis and len(candidate_logicals) > 0:
            self.reduce_logical_operator_basis(candidate_logicals)
            candidate_logicals = []
            weight_one_syndromes_searched = 0
            max_distance = np.max(self.logical_basis_weights())

        # Update and return the estimated distance
        self.code_distance = min_distance

    def logical_basis_weights(self):
        """
        Return the Hamming weights of the logical operators in the current basis.

        Returns
        -------
        np.ndarray
            An array of integers representing the Hamming weights of the logical operators.
        """
        return binary_pauli_hamming_weight(self.logical_operator_basis).flatten()
