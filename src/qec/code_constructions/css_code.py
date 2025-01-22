from qec.code_constructions import StabilizerCode
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse

# Added / ammended from old code
from typing import Union, Tuple
import numpy as np
import ldpc.mod2
import scipy
from ldpc import BpOsdDecoder
from tqdm import tqdm
import time
import logging
from typing import Optional, Sequence

logging.basicConfig(level=logging.DEBUG)


class CSSCode(StabilizerCode):
    """
    A class for generating and manipulating Calderbank-Shor-Steane (CSS) quantum error-correcting codes.

    Prameters
    ---------
    x_stabilizer_matrix (hx): Union[np.ndarray, scipy.sparse.spmatrix]
        The X-check matrix.
    z_stabilizer_matrix (hz): Union[np.ndarray, scipy.sparse.spmatrix]
        The Z-check matrix.
    name: str, optional
        A name for this CSS code. Defaults to "CSS code".

    Attributes
    ----------
    x_stabilizer_matrix (hx): Union[np.ndarray, scipy.sparse.spmatrix]
        The X-check matrix.
    z_stabilizer_matrix (hz): Union[np.ndarray, scipy.sparse.spmatrix]
        The Z-check matrix.
    name (str):
        A name for this CSS code.
    physical_qubit_count (N): int
        The number of physical qubits in the code.
    logical_qubit_count (K): int
        The number of logical qubits in the code. Dimension of the code.
    code_distance (d): int
        (Not computed by default) Minimum distance of the code.
    x_logical_operator_basis (lx): (Union[np.ndarray, scipy.sparse.spmatrix]
        Logical X operator basis.
    z_logical_operator_basis (lz): (Union[np.ndarray, scipy.sparse.spmatrix]
        Logical Z operator basis.
    """

    def __init__(
        self,
        x_stabilizer_matrix: Union[np.ndarray, scipy.sparse.spmatrix],
        z_stabilizer_matrix: Union[np.ndarray, scipy.sparse.spmatrix],
        name: str = None,
    ):
        """
        Initialise a new instance of the CSSCode class.

        Parameters
        ----------
        x_stabilizer_matrix (hx): Union[np.ndarray, scipy.sparse.spmatrix]
            The X-check matrix.
        z_stabilizer_matrix (hz): Union[np.ndarray, scipy.sparse.spmatrix]
            The Z-check matrix.
        name: str, optional
            A name for this CSS code. Defaults to "CSS code".
        """

        # Assign a default name if none is provided
        if name is None:
            self.name = "CSS code"
        else:
            self.name = name

        self.x_logical_operator_basis = None
        self.z_logical_operator_basis = None

        self.x_code_distance = None
        self.z_code_distance = None

        # Check if the input matrices are NumPy arrays or SciPy sparse matrices
        if not isinstance(x_stabilizer_matrix, (np.ndarray, scipy.sparse.spmatrix)):
            raise TypeError(
                "Please provide x and z stabilizer matrices as either a numpy array or a scipy sparse matrix."
            )

        # Convert matrices to sparse representation and set them as class attributes (replaced the old code "convert_to_sparse")
        self.x_stabilizer_matrix = convert_to_binary_scipy_sparse(x_stabilizer_matrix)
        self.z_stabilizer_matrix = convert_to_binary_scipy_sparse(z_stabilizer_matrix)

        # Calculate the number of physical qubits from the matrix dimension
        self.physical_qubit_count = self.x_stabilizer_matrix.shape[1]

        # Validate the number of qubits for both matrices
        try:
            assert self.physical_qubit_count == self.z_stabilizer_matrix.shape[1]
        except AssertionError:
            raise ValueError(
                f"Input matrices x_stabilizer_matrix and z_stabilizer_matrix must have the same number of columns.\
                              Current column count, x_stabilizer_matrix: {x_stabilizer_matrix.shape[1]}; z_stabilizer_matrix: {z_stabilizer_matrix.shape[1]}"
            )

        # Validate if the input matrices commute
        try:
            assert not np.any(
                (self.x_stabilizer_matrix @ self.z_stabilizer_matrix.T).data % 2
            )
        except AssertionError:
            raise ValueError(
                "Input matrices hx and hz do not commute. I.e. they do not satisfy\
                              the requirement that hx@hz.T = 0."
            )

        # Compute a basis of the logical operators
        self.compute_logical_basis()

    def compute_logical_basis(self):
        """
        Compute the logical operator basis for the given CSS code.

        Returns
        -------
        Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]
            Logical X and Z operator bases (lx, lz).

        Notes
        -----
        This method uses the kernel of the X and Z stabilizer matrices to find operators that commute with all the stabilizers,
        and then identifies the subsets of which are not themselves linear combinations of the stabilizers.
        """

        # Compute the kernel of hx and hz matrices

        # Z logicals

        # Compute the kernel of hx
        ker_hx = ldpc.mod2.kernel(self.x_stabilizer_matrix)  # kernel of X-stabilisers
        # Sort the rows of ker_hx by weight
        row_weights = ker_hx.getnnz(axis=1)
        sorted_rows = np.argsort(row_weights)
        ker_hx = ker_hx[sorted_rows, :]
        # Z logicals are elements of ker_hx (that commute with all the X-stabilisers) that are not linear combinations of Z-stabilisers
        logical_stack = scipy.sparse.vstack([self.z_stabilizer_matrix, ker_hx]).tocsr()
        self.rank_hz = ldpc.mod2.rank(self.z_stabilizer_matrix)
        # The first self.rank_hz pivot_rows of logical_stack are the Z-stabilisers. The remaining pivot_rows are the Z logicals
        pivots = ldpc.mod2.pivot_rows(logical_stack)
        self.z_logical_operator_basis = logical_stack[pivots[self.rank_hz :], :]

        # X logicals

        # Compute the kernel of hz
        ker_hz = ldpc.mod2.kernel(self.z_stabilizer_matrix)
        # Sort the rows of ker_hz by weight
        row_weights = ker_hz.getnnz(axis=1)
        sorted_rows = np.argsort(row_weights)
        ker_hz = ker_hz[sorted_rows, :]
        # X logicals are elements of ker_hz (that commute with all the Z-stabilisers) that are not linear combinations of X-stabilisers
        logical_stack = scipy.sparse.vstack([self.x_stabilizer_matrix, ker_hz]).tocsr()
        self.rank_hx = ldpc.mod2.rank(self.x_stabilizer_matrix)
        # The first self.rank_hx pivot_rows of logical_stack are the X-stabilisers. The remaining pivot_rows are the X logicals
        pivots = ldpc.mod2.pivot_rows(logical_stack)
        self.x_logical_operator_basis = logical_stack[pivots[self.rank_hx :], :]

        # set the dimension of the code (i.e. the number of logical qubits)
        self.logical_qubit_count = self.x_logical_operator_basis.shape[0]

        # find the minimum weight logical operators
        self.x_code_distance = self.physical_qubit_count
        self.z_code_distance = self.physical_qubit_count

        for i in range(self.logical_qubit_count):
            if self.x_logical_operator_basis[i].nnz < self.x_code_distance:
                self.x_code_distance = self.x_logical_operator_basis[i].nnz
            if self.z_logical_operator_basis[i].nnz < self.z_code_distance:
                self.z_code_distance = self.z_logical_operator_basis[i].nnz
        self.code_distance = np.min([self.x_code_distance, self.z_code_distance])

        # FIXME: How does this differ from rank_hx and rank_hz descibed above (ldpc.mod2.rank())?
        # compute the hx and hz rank
        self.rank_hx = self.physical_qubit_count - ker_hx.shape[0]
        self.rank_hz = self.physical_qubit_count - ker_hz.shape[0]

        return (self.x_logical_operator_basis, self.z_logical_operator_basis)

    # TODO: Add a function to save the logical operator basis to a file

    def check_valid_logical_basis(self) -> bool:
        """
        Validate that the stored logical operators form a proper logical basis for the code.

        Checks that they commute with the stabilizers, pairwise anti-commute, and have full rank.

        Returns
        -------
        bool
            True if the logical operators form a valid basis, otherwise False.
        """

        # If logical bases are not computed yet, compute them
        if (
            self.x_logical_operator_basis is None
            or self.z_logical_operator_basis is None
        ):
            self.x_logical_operator_basis, self.z_logical_operator_basis = (
                self.compute_logical_basis(
                    self.x_stabilizer_matrix, self.z_stabilizer_matrix
                )
            )
            self.logical_qubit_count = self.x_logical_operator_basis.shape[0]

        try:
            # Test dimension
            assert (
                self.logical_qubit_count
                == self.z_logical_operator_basis.shape[0]
                == self.x_logical_operator_basis.shape[0]
            ), "Logical operator basis dimensions do not match."

            # Check logical basis linearly independent (i.e. full rank)
            assert (
                ldpc.mod2.rank(self.x_logical_operator_basis)
                == self.logical_qubit_count
            ), "X logical operator basis is not full rank, and hence not linearly independent."
            assert (
                ldpc.mod2.rank(self.z_logical_operator_basis)
                == self.logical_qubit_count
            ), "Z logical operator basis is not full rank, and hence not linearly independent."

            # Perform various tests to validate the logical bases

            # Check that the logical operators commute with the stabilizers
            try:
                assert not np.any(
                    (self.x_logical_operator_basis @ self.z_stabilizer_matrix.T).data
                    % 2
                ), "X logical operators do not commute with Z stabilizers."
            except AssertionError as e:
                logging.error(e)
                return False

            try:
                assert not np.any(
                    (self.z_logical_operator_basis @ self.x_stabilizer_matrix.T).data
                    % 2
                ), "Z logical operators do not commute with X stabilizers."
            except AssertionError as e:
                logging.error(e)
                return False

            # Check that the logical operators anticommute with each other (by checking that the rank of the product is full rank)
            test = self.x_logical_operator_basis @ self.z_logical_operator_basis.T
            test.data = test.data % 2
            assert (
                ldpc.mod2.rank(test) == self.logical_qubit_count
            ), "Logical operators do not pairwise anticommute."

            test = self.z_logical_operator_basis @ self.x_logical_operator_basis.T
            test.data = test.data % 2
            assert (
                ldpc.mod2.rank(test) == self.logical_qubit_count
            ), "Logical operators do not pairwise anticommute."

            # TODO: Check that the logical operators are not themselves stabilizers?

        except AssertionError as e:
            logging.error(e)
            return False

        return True

    def compute_exact_code_distance(
        self, timeout: float = 0.5
    ) -> Tuple[Optional[int], Optional[int], float]:
        """
        Compute the exact distance of the CSS code by searching through linear combinations
        of logical operators and stabilisers, ensuring balanced progress between X and Z searches.

        Parameters
        ----------
        timeout : float, optional
            The time limit (in seconds) for the exhaustive search. Default is 0.5 seconds.
            To obtain the exact distance, set to `np.inf`.

        Returns
        -------
        Tuple[Optional[int], Optional[int], float]
            A tuple containing:
            - The best-known X distance of the code (or None if no X distance was found)
            - The best-known Z distance of the code (or None if no Z distance was found)
            - The fraction of total combinations considered before timeout

        Notes
        -----
        - Searches X and Z combinations in an interleaved manner to ensure balanced progress
        - For each type (X/Z):
            - We compute the row span of both stabilisers and logical operators
            - For every logical operator in the logical span, we add (mod 2) each stabiliser
            - We compute the Hamming weight of each candidate operator
            - We track the minimal Hamming weight encountered
        """
        start_time = time.time()

        # Get stabiliser spans
        x_stabiliser_span = ldpc.mod2.row_span(self.x_stabilizer_matrix)[1:]
        z_stabiliser_span = ldpc.mod2.row_span(self.z_stabilizer_matrix)[1:]

        # Get logical spans
        x_logical_span = ldpc.mod2.row_span(self.x_logical_operator_basis)[1:]
        z_logical_span = ldpc.mod2.row_span(self.z_logical_operator_basis)[1:]

        # Initialize distances
        if self.x_code_distance is None:
            x_code_distance = np.inf
        else:
            x_code_distance = self.x_code_distance

        if self.z_code_distance is None:
            z_code_distance = np.inf
        else:
            z_code_distance = self.z_code_distance

        # Prepare iterators for both X and Z combinations
        x_combinations = (
            (x_l, x_s) for x_l in x_logical_span for x_s in x_stabiliser_span
        )
        z_combinations = (
            (z_l, z_s) for z_l in z_logical_span for z_s in z_stabiliser_span
        )

        total_x_combinations = x_stabiliser_span.shape[0] * x_logical_span.shape[0]
        total_z_combinations = z_stabiliser_span.shape[0] * z_logical_span.shape[0]
        total_combinations = total_x_combinations + total_z_combinations
        combinations_considered = 0

        # Create iterables that we can exhaust
        x_iter = iter(x_combinations)
        z_iter = iter(z_combinations)
        x_exhausted = False
        z_exhausted = False

        while not (x_exhausted and z_exhausted):
            if time.time() - start_time > timeout:
                break

            # Try X combination if not exhausted
            if not x_exhausted:
                try:
                    x_logical, x_stabiliser = next(x_iter)
                    candidate_x = x_logical + x_stabiliser
                    candidate_x.data %= 2
                    x_weight = candidate_x.getnnz()
                    if x_weight < x_code_distance:
                        x_code_distance = x_weight
                    combinations_considered += 1
                except StopIteration:
                    x_exhausted = True

            # Try Z combination if not exhausted
            if not z_exhausted:
                try:
                    z_logical, z_stabiliser = next(z_iter)
                    candidate_z = z_logical + z_stabiliser
                    candidate_z.data %= 2
                    z_weight = candidate_z.getnnz()
                    if z_weight < z_code_distance:
                        z_code_distance = z_weight
                    combinations_considered += 1
                except StopIteration:
                    z_exhausted = True

        # Update code distances
        self.x_code_distance = x_code_distance if x_code_distance != np.inf else None
        self.z_code_distance = z_code_distance if z_code_distance != np.inf else None
        self.code_distance = (
            min(x_code_distance, z_code_distance)
            if x_code_distance != np.inf and z_code_distance != np.inf
            else None
        )

        # Calculate fraction of combinations considered
        fraction_considered = combinations_considered / total_combinations

        return (
            int(x_code_distance) if x_code_distance != np.inf else None,
            int(z_code_distance) if z_code_distance != np.inf else None,
            fraction_considered,
        )

    def estimate_min_distance(
        self,
        timeout_seconds: float = 0.25,
        p: float = 0.25,
        reduce_logical_basis: bool = False,
        decoder: Optional[BpOsdDecoder] = None,
    ) -> int:
        """
        Estimate the minimum distance of the CSS code using a BP+OSD decoder-based search.

        Parameters
        ----------
        timeout_seconds : float, optional
            Time limit in seconds for the search. Default: 0.25
        p : float, optional
            Probability for including each logical operator in trial combinations. Default: 0.25
        reduce_logical_basis : bool, optional
            Whether to attempt reducing the logical operator basis. Default: False
        decoder : Optional[BpOsdDecoder], optional
            Pre-configured BP+OSD decoder. If None, initializes with default settings.

        Returns
        -------
        int
            Best estimate of code distance found within the time limit.
        """
        start_time = time.time()

        # Ensure logical operator bases are computed
        if (
            self.x_logical_operator_basis is None
            or self.z_logical_operator_basis is None
        ):
            self.compute_logical_basis()

        # Setup decoders and parameters for both X and Z
        bp_osd_z, x_stack, full_rank_x, x_min_distance, x_max_distance = (
            self._setup_distance_estimation_decoder(
                self.x_stabilizer_matrix, self.x_logical_operator_basis, decoder
            )
        )
        bp_osd_x, z_stack, full_rank_z, z_min_distance, z_max_distance = (
            self._setup_distance_estimation_decoder(
                self.z_stabilizer_matrix, self.z_logical_operator_basis, decoder
            )
        )

        candidate_logicals_x = []
        candidate_logicals_z = []

        x_weight_one_searched = 0
        z_weight_one_searched = 0

        with tqdm(total=timeout_seconds, desc="Estimating distance") as pbar:
            while time.time() - start_time < timeout_seconds:
                elapsed = time.time() - start_time
                pbar.update(elapsed - pbar.n)

                if np.random.rand() < 0.5:
                    # X Logical operators
                    if x_weight_one_searched < self.z_logical_operator_basis.shape[0]:
                        dummy_syndrome_x = np.zeros(z_stack.shape[0], dtype=np.uint8)
                        dummy_syndrome_x[
                            full_rank_z.shape[0] + x_weight_one_searched
                        ] = 1
                        x_weight_one_searched += 1
                    else:
                        dummy_syndrome_x = self._generate_random_logical_combination_for_distance_estimation(
                            z_stack, p, self.z_stabilizer_matrix.shape[0]
                        )

                    candidate_x = bp_osd_x.decode(dummy_syndrome_x)
                    x_weight = np.count_nonzero(candidate_x)
                    if x_weight < x_min_distance:
                        x_min_distance = x_weight

                    if x_weight < x_max_distance and reduce_logical_basis:
                        candidate_logicals_x.append(candidate_x)

                        # Reduce X logical operator basis independently
                        if len(candidate_logicals_x) >= 5:
                            self._reduce_logical_operator_basis(
                                candidate_logicals_x, []
                            )
                            (
                                bp_osd_x,
                                z_stack,
                                full_rank_z,
                                z_min_distance,
                                z_max_distance,
                            ) = self._setup_distance_estimation_decoder(
                                self.z_stabilizer_matrix,
                                self.z_logical_operator_basis,
                                decoder,
                            )
                            candidate_logicals_x = []
                            x_weight_one_searched = 0

                else:
                    # Z Logical operators
                    if z_weight_one_searched < self.x_logical_operator_basis.shape[0]:
                        dummy_syndrome_z = np.zeros(x_stack.shape[0], dtype=np.uint8)
                        dummy_syndrome_z[
                            full_rank_x.shape[0] + z_weight_one_searched
                        ] = 1
                        z_weight_one_searched += 1
                    else:
                        dummy_syndrome_z = self._generate_random_logical_combination_for_distance_estimation(
                            x_stack, p, self.x_stabilizer_matrix.shape[0]
                        )

                    candidate_z = bp_osd_z.decode(dummy_syndrome_z)
                    z_weight = np.count_nonzero(candidate_z)
                    if z_weight < z_min_distance:
                        z_min_distance = z_weight

                    if z_weight < z_max_distance and reduce_logical_basis:
                        candidate_logicals_z.append(candidate_z)

                        # Reduce Z logical operator basis independently
                        if len(candidate_logicals_z) >= 5:
                            self._reduce_logical_operator_basis(
                                [], candidate_logicals_z
                            )
                            (
                                bp_osd_z,
                                x_stack,
                                full_rank_x,
                                x_min_distance,
                                x_max_distance,
                            ) = self._setup_distance_estimation_decoder(
                                self.x_stabilizer_matrix,
                                self.x_logical_operator_basis,
                                decoder,
                            )
                            candidate_logicals_z = []
                            z_weight_one_searched = 0

                x_weights, z_weights = self.logical_basis_weights()
                pbar.set_description(
                    f"Estimating distance: dx <= {x_min_distance}, dz <= {z_min_distance}, x-weights: {np.mean(x_weights):.2f}, z-weights: {np.mean(z_weights):.2f}"
                )

        self._reduce_logical_operator_basis(candidate_logicals_x, candidate_logicals_z)

        # Update distances
        self.x_code_distance = x_min_distance
        self.z_code_distance = z_min_distance
        self.code_distance = min(x_min_distance, z_min_distance)

        return self.code_distance

    def _setup_distance_estimation_decoder(
        self, stabilizer_matrix, logical_operator_basis, decoder=None
    ) -> Tuple[BpOsdDecoder, scipy.sparse.spmatrix, scipy.sparse.spmatrix, int, int]:
        """
        Helper function to set up the BP+OSD decoder for distance estimation.

        Parameters
        ----------
        stabilizer_matrix : scipy.sparse.spmatrix
            Stabilizer matrix of the code.
        logical_operator_basis : scipy.sparse.spmatrix
            Logical operator basis of the code.
        decoder : Optional[BpOsdDecoder], optional
            Pre-configured decoder. If None, initializes with default settings.

        Returns
        -------
        Tuple[BpOsdDecoder, scipy.sparse.spmatrix, scipy.sparse.spmatrix, int, int]
            Decoder, stacked matrix, stabilizer matrix, minimum distance, and maximum distance.
        """
        # Remove redundant rows from stabilizer matrix
        p_rows = ldpc.mod2.pivot_rows(stabilizer_matrix)
        full_rank_stabilizer_matrix = stabilizer_matrix[p_rows]

        # Build a stacked matrix of stabilizers and logicals
        stack = scipy.sparse.vstack(
            [full_rank_stabilizer_matrix, logical_operator_basis]
        ).tocsr()

        # Initial distance estimate from current logicals
        min_distance = np.min(logical_operator_basis.getnnz(axis=1))
        max_distance = np.max(logical_operator_basis.getnnz(axis=1))

        # Set up BP+OSD decoder if not provided
        if decoder is None:
            decoder = BpOsdDecoder(
                stack,
                error_rate=0.1,
                max_iter=10,
                bp_method="ms",
                schedule="parallel",
                ms_scaling_factor=1.0,
                osd_method="osd_0",
                osd_order=0,
            )

        return decoder, stack, full_rank_stabilizer_matrix, min_distance, max_distance

    def _generate_random_logical_combination_for_distance_estimation(
        self, stack: scipy.sparse.spmatrix, p: float, stabilizer_count: int
    ) -> np.ndarray:
        """
        Generate a random logical combination for the BP+OSD decoder.

        Parameters
        ----------
        stack : scipy.sparse.spmatrix
            The stacked stabilizer and logical operator matrix.
        p : float
            Probability for including each logical operator in the combination.
        stabilizer_count : int
            Number of stabilizer rows in the stacked matrix.

        Returns
        -------
        np.ndarray
            Randomly generated syndrome vector.
        """
        random_mask = np.random.choice([0, 1], size=stack.shape[0], p=[1 - p, p])
        random_mask[:stabilizer_count] = (
            0  # Ensure no stabilizer-only rows are selected
        )

        while not np.any(random_mask):
            random_mask = np.random.choice([0, 1], size=stack.shape[0], p=[1 - p, p])
            random_mask[:stabilizer_count] = 0

        dummy_syndrome = np.zeros(stack.shape[0], dtype=np.uint8)
        dummy_syndrome[np.nonzero(random_mask)[0]] = 1

        return dummy_syndrome

    def _reduce_logical_operator_basis(
        self,
        candidate_logicals_x: Union[Sequence, np.ndarray, scipy.sparse.spmatrix] = [],
        candidate_logicals_z: Union[Sequence, np.ndarray, scipy.sparse.spmatrix] = [],
    ):
        """
        Reduce the logical operator bases (for X and Z) to include lower-weight logicals.

        Parameters
        ----------
        candidate_logicals_x : Union[Sequence, np.ndarray, scipy.sparse.spmatrix], optional
            A list or array of candidate X logical operators to consider for reducing the X basis.
            Defaults to an empty list.
        candidate_logicals_z : Union[Sequence, np.ndarray, scipy.sparse.spmatrix], optional
            A list or array of candidate Z logical operators to consider for reducing the Z basis.
            Defaults to an empty list.
        """
        # Reduce X logical operator basis
        if candidate_logicals_x:
            # Convert candidates to a sparse matrix if they aren't already
            if not isinstance(candidate_logicals_x, scipy.sparse.spmatrix):
                candidate_logicals_x = scipy.sparse.csr_matrix(candidate_logicals_x)

            # Stack the candidate X logicals with the existing X logicals
            temp_x = scipy.sparse.vstack(
                [candidate_logicals_x, self.x_logical_operator_basis]
            ).tocsr()

            # Calculate Hamming weights for sorting
            x_row_weights = temp_x.getnnz(axis=1)
            sorted_x_rows = np.argsort(x_row_weights)
            temp_x = temp_x[sorted_x_rows, :]

            # Add the X stabilizer matrix to the top of the stack
            temp_x = scipy.sparse.vstack([self.x_stabilizer_matrix, temp_x]).tocsr()

            # Determine rank of the X stabilizer matrix
            rank_hx = ldpc.mod2.rank(self.x_stabilizer_matrix)

            # Perform row reduction to find a new X logical basis
            pivots_x = ldpc.mod2.pivot_rows(temp_x)
            self.x_logical_operator_basis = temp_x[pivots_x[rank_hx:], :]

        # Reduce Z logical operator basis
        if candidate_logicals_z:
            # Convert candidates to a sparse matrix if they aren't already
            if not isinstance(candidate_logicals_z, scipy.sparse.spmatrix):
                candidate_logicals_z = scipy.sparse.csr_matrix(candidate_logicals_z)

            # Stack the candidate Z logicals with the existing Z logicals
            temp_z = scipy.sparse.vstack(
                [candidate_logicals_z, self.z_logical_operator_basis]
            ).tocsr()

            # Calculate Hamming weights for sorting
            z_row_weights = temp_z.getnnz(axis=1)
            sorted_z_rows = np.argsort(z_row_weights)
            temp_z = temp_z[sorted_z_rows, :]

            # Add the Z stabilizer matrix to the top of the stack
            temp_z = scipy.sparse.vstack([self.z_stabilizer_matrix, temp_z]).tocsr()

            # Determine rank of the Z stabilizer matrix
            rank_hz = ldpc.mod2.rank(self.z_stabilizer_matrix)

            # Perform row reduction to find a new Z logical basis
            pivots_z = ldpc.mod2.pivot_rows(temp_z)
            self.z_logical_operator_basis = temp_z[pivots_z[rank_hz:], :]

    def fix_logical_operators(self, fix_logical: str = "X"):
        """
        Create a canonical basis of logical operators where X-logical and Z-logical operators pairwise anticommute.

        Parameters
        ----------
        fix_logical : str, optional
            Specify which logical operator basis to fix. "X" adjusts Z-logicals based on X-logicals, and "Z" adjusts
            X-logicals based on Z-logicals. Default is "X".

        Raises
        ------
        TypeError
            If `fix_logical` is not a string.
        ValueError
            If `fix_logical` is not "X" or "Z".

        Returns
        -------
        bool
            True if the logical operator basis is valid after fixing; False otherwise.

        Notes
        -----
        This method ensures that the symplectic product of the logical bases results in the identity matrix.
        If any issues occur during the adjustment, the method logs an error.
        """
        if not isinstance(fix_logical, str):
            raise TypeError("fix_logical parameter must be a string")

        if fix_logical.lower() == "x":
            temp = self.z_logical_operator_basis @ self.x_logical_operator_basis.T
            temp.data = temp.data % 2
            temp = scipy.sparse.csr_matrix(ldpc.mod2.inverse(temp), dtype=np.uint8)
            self.z_logical_operator_basis = temp @ self.z_logical_operator_basis
            self.z_logical_operator_basis.data = self.z_logical_operator_basis.data % 2

        elif fix_logical.lower() == "z":
            temp = self.x_logical_operator_basis @ self.z_logical_operator_basis.T
            temp.data = temp.data % 2
            temp = scipy.sparse.csr_matrix(ldpc.mod2.inverse(temp), dtype=np.uint8)
            self.x_logical_operator_basis = temp @ self.x_logical_operator_basis
            self.x_logical_operator_basis.data = self.x_logical_operator_basis.data % 2
        else:
            raise ValueError("Invalid fix_logical parameter")

        try:
            assert self.check_valid_logical_basis()
        except AssertionError:
            logging.error("Logical basis is not valid after fixing logical operators.")
            return False

        try:
            lx_lz = self.x_logical_operator_basis @ self.z_logical_operator_basis.T
            lx_lz.data = lx_lz.data % 2
            assert (
                lx_lz != scipy.sparse.eye(self.logical_qubit_count, format="csr")
            ).nnz == 0
        except AssertionError:
            logging.error("Logical basis is not valid after fixing logical operators.")
            return False

        return True

    def logical_basis_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        x_weights = []
        z_weights = []
        for i in range(self.logical_qubit_count):
            x_weights.append(self.x_logical_operator_basis[i].nnz)
            z_weights.append(self.z_logical_operator_basis[i].nnz)

        return (np.array(x_weights), np.array(z_weights))

    def __str__(self):
        """
        Return a string representation of the CSSCode object.

        Returns:
            str: String representation of the CSS code.
        """
        return f"{self.name} Code: [[N={self.physical_qubit_count}, K={self.logical_qubit_count}, dx<={self.x_code_distance}, dz<={self.z_code_distance}]]"
