from qec.stabilizer_code import StabilizerCode
from qec.utils.sparse_binary_utils import convert_to_binary_scipy_sparse

# Added / ammended from old code
from typing import Union, Tuple, List, Sequence
import numpy as np
import ldpc.mod2
import scipy
import qec.utils
from ldpc import BpOsdDecoder
import itertools
from tqdm import tqdm
import time
import os
import pathlib
import logging 

logging.basicConfig(level=logging.DEBUG)


ArrayLike = Union[Sequence, np.ndarray] # For functions that accept inputs that can either be NumPy arrays or other sequence-like objects


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

        # Check if the input matrices are NumPy arrays or SciPy sparse matrices
        if not isinstance(x_stabilizer_matrix, (np.ndarray, scipy.sparse.spmatrix)):
            raise TypeError("Please provide x and z stabilizer matrices as either a numpy array or a scipy sparse matrix.")

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
            assert not np.any((self.x_stabilizer_matrix @ self.z_stabilizer_matrix.T).data % 2)
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
        row_weights = np.diff(ker_hx.indptr) # Better performance to use: row_weights = ker_hx.getnnz(axis=1)?
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
        row_weights = np.diff(ker_hz.indptr) # Better performance to use: row_weights = ker_hz.getnnz(axis=1)
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
        self.dx = self.physical_qubit_count
        self.dz = self.physical_qubit_count

        for i in range(self.logical_qubit_count):
            if self.x_logical_operator_basis[i].nnz < self.dx:
                self.dx = self.x_logical_operator_basis[i].nnz
            if self.z_logical_operator_basis[i].nnz < self.dz:
                self.dz = self.z_logical_operator_basis[i].nnz
        self.code_distance = np.min([self.dx, self.dz])

        #FIXME: How does this differ from rank_hx and rank_hz descibed above (ldpc.mod2.rank())?
        # compute the hx and hz rank
        self.rank_hx = self.physical_qubit_count - ker_hx.shape[0]
        self.rank_hz = self.physical_qubit_count - ker_hz.shape[0]

        return (self.x_logical_operator_basis, self.z_logical_operator_basis)
    
    # TODO: Add a function to save the logical operator basis to a file

    def check_valid_logical_xz_basis(self) -> bool:
        """
        Validate that the stored logical operators form a proper logical basis for the code.

        Checks that they commute with the stabilizers, pairwise anti-commute, and have full rank.

        Returns
        -------
        bool
            True if the logical operators form a valid basis, otherwise False.
        """

        # If logical bases are not computed yet, compute them
        if self.x_logical_operator_basis is None or self.z_logical_operator_basis is None:
            self.x_logical_operator_basis, self.z_logical_operator_basis = self.compute_logical_basis(self.x_stabilizer_matrix, self.z_stabilizer_matrix)
            self.logical_qubit_count = self.x_logical_operator_basis.shape[0]
        
        try:
            # Test dimension
            assert (
                self.logical_qubit_count == self.z_logical_operator_basis.shape[0] == self.x_logical_operator_basis.shape[0]
            ), "Logical operator basis dimensions do not match."

            # Check logical basis linearly independent (i.e. full rank)
            assert (
                ldpc.mod2.rank(self.x_logical_operator_basis) == self.logical_qubit_count
            ), "X logical operator basis is not full rank, and hence not linearly independent."
            assert (
                ldpc.mod2.rank(self.z_logical_operator_basis) == self.logical_qubit_count
            ), "Z logical operator basis is not full rank, and hence not linearly independent."

            # Perform various tests to validate the logical bases

            # Check that the logical operators commute with the stabilizers
            try:
                assert (
                    not np.any((self.x_logical_operator_basis @ self.z_stabilizer_matrix.T).data % 2)
                ), "X logical operators do not commute with Z stabilizers."
            except AssertionError as e:
                logging.error(e)
                return False

            try:
                assert (
                    not np.any((self.z_logical_operator_basis @ self.x_stabilizer_matrix.T).data % 2)
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
        
    # TODO: Add "compute_exact_code_distance" function to compute the exact code distance of the code

    # FIXME: Update to follow StabilizerCodes function format?
    def estimate_min_distance(
        self, reduce_logical_basis: bool = False, timeout_seconds: float = 0.25, p: float = 0.25
    ) -> int:
        """
        Estimate the minimum distance of the stabilizer code using a BP+OSD decoder-based search.

        Parameters
        ----------
        reduce_logical_basis : bool, optional
            Whether to reduce the logical operator basis during the search. Default is `False`.
        timeout_seconds : float, optional
            The time limit (in seconds) for the exhaustive search. Default is 0.5 seconds. To obtain the exact distance, set to `np.inf`.
        p : float, optional
            Probability used to randomly include or exclude each logical operator
            when generating trial logical operators. Default is 0.25.

        Returns
        -------
        int
            The best-known estimate of the code distance found within the time limit.
        """
        
        start_time = time.time()

        if self.x_logical_operator_basis is None or self.z_logical_operator_basis is None:
            # Compute a basis of the logical operators
            self.x_logical_operator_basis, self.z_logical_operator_basis = self.compute_logical_basis()
            # Calculate the dimension of the code
            self.logical_qubit_count = self.x_logical_operator_basis.shape[0]

        self.dx = self.physical_qubit_count
        self.dz = self.physical_qubit_count
        max_lx = 0
        max_lz = 0

        # self.x_logical_operator_basis = self.x_logical_operator_basis.tocsr()
        # self.z_logical_operator_basis = self.z_logical_operator_basis.tocsr()

        # self.rank_hx = ldpc.mod2.rank(self.x_stabilizer_matrix)
        # self.rank_hz = ldpc.mod2.rank(self.z_stabilizer_matrix)

        for i in range(self.logical_qubit_count):
            if self.x_logical_operator_basis[i].nnz > max_lx:
                max_lx = self.x_logical_operator_basis[i].nnz
            if self.x_logical_operator_basis[i].nnz < self.dx:
                self.dx = self.x_logical_operator_basis[i].nnz

            if self.z_logical_operator_basis[i].nnz > max_lz:
                max_lz = self.z_logical_operator_basis[i].nnz
            if self.z_logical_operator_basis[i].nnz < self.dz:
                self.dz = self.z_logical_operator_basis[i].nnz

        candidate_logicals_x = []
        candidate_logicals_z = []

        x_stack = scipy.sparse.vstack([self.x_stabilizer_matrix, self.x_logical_operator_basis])
        z_stack = scipy.sparse.vstack([self.z_stabilizer_matrix, self.z_logical_operator_basis])

        bp_osdx = BpOsdDecoder(
            x_stack,
            error_rate=0.1,
            max_iter=10,
            bp_method="ms",
            ms_scaling_factor=1.0,
            schedule="parallel",
            osd_method="osd_0",
            osd_order=0,
        )

        bp_osdz = BpOsdDecoder(
            z_stack,
            error_rate=0.1,
            max_iter=10,
            bp_method="ms",
            schedule="parallel",
            ms_scaling_factor=1.0,
            osd_method="osd_0",
            osd_order=0,
        )

        for i in range(self.logical_qubit_count):

            dummy_syndrome_x = np.zeros(x_stack.shape[0], dtype=np.uint8)
            dummy_syndrome_z = np.zeros(z_stack.shape[0], dtype=np.uint8)
            dummy_syndrome_x[self.x_stabilizer_matrix.shape[0] + i] = 1
            dummy_syndrome_z[self.z_stabilizer_matrix.shape[0] + i] = 1

            decoded_logical_x = bp_osdz.decode(dummy_syndrome_z)
            logical_size = np.count_nonzero(decoded_logical_x)
            if (logical_size < max_lx) and reduce_logical_basis:
                candidate_logicals_x.append(decoded_logical_x)
            if logical_size < self.dx:
                self.dx = logical_size

            decoded_logical_z = bp_osdx.decode(dummy_syndrome_x)
            logical_size = np.count_nonzero(decoded_logical_z)
            if (logical_size < max_lz) and reduce_logical_basis:
                candidate_logicals_z.append(decoded_logical_z)
            if logical_size < self.dz:
                self.dz = logical_size

        self.code_distance = np.min([self.dx, self.dz])

        if reduce_logical_basis:
            self.reduce_logical_operator_basis(
                candidate_logicals_x, candidate_logicals_z
            )

        logical_x_stack = scipy.sparse.vstack([self.x_stabilizer_matrix, self.x_logical_operator_basis])
        logical_z_stack = scipy.sparse.vstack([self.z_stabilizer_matrix, self.z_logical_operator_basis])

        candidate_logicals_x = []
        candidate_logicals_z = []

        with tqdm(total=timeout_seconds) as pbar:
            while time.time() < timeout_seconds + start_time:
                # Your loop content here
                
                # Calculate elapsed time and update progress bar
                elapsed_time = time.time() - start_time
                # Update progress bar with formatted elapsed time (1 decimal point)
                # pbar.set_postfix_str(f"Time: {elapsed_time:.1f}s", refresh=True)
                
                # Update the progress bar by 1 (or any other logic for progress update)
                pbar.update(elapsed_time - pbar.n)

                pbar.set_description(f"dx<{self.dx}, dz<{self.dz}, Time: {elapsed_time:.1f}s/{timeout_seconds:.1f}s")


                # p = 0.25
            
                logical_op_indices_x = np.random.choice([0, 1], size=logical_x_stack.shape[0], p=[1-p, p])
                # to ensure it actually is a logical operator
                logical_op_indices_x[self.x_stabilizer_matrix.shape[0] + np.random.randint(self.logical_qubit_count)] = 1
                logical_op_indices_x = np.nonzero(logical_op_indices_x)[0]

                logical_op_x = np.zeros(logical_x_stack.shape[1], dtype=np.uint8)

                for i in logical_op_indices_x:
                    logical_op_x += (logical_x_stack.getrow(i).toarray().flatten().astype(np.uint8)) 
                
                logical_op_x = logical_op_x % 2

                x_stack = scipy.sparse.vstack([self.x_stabilizer_matrix, logical_op_x]).astype(np.uint8)

                # exit(22)

                bp_osdx = BpOsdDecoder(
                    x_stack,
                    error_rate=0.1,
                    max_iter=10,
                    bp_method="ms",
                    ms_scaling_factor=0.9,
                    schedule="parallel",
                    osd_method="osd_0",
                    osd_order=0,
                )

                dummy_syndrome_x = np.zeros(x_stack.shape[0], dtype=np.uint8)
                dummy_syndrome_x[-1] = 1
                decoded_logical_z = bp_osdx.decode(dummy_syndrome_x)
                logical_size = np.count_nonzero(decoded_logical_z)
                if (logical_size < max_lz) and reduce_logical_basis:
                    candidate_logicals_z.append(decoded_logical_z)
                if logical_size < self.dz:
                    self.dz = logical_size

                logical_op_indices_z = np.random.choice([0, 1], size=logical_z_stack.shape[0], p=[1-p, p])
                # to ensure it actually is a logical operator
                logical_op_indices_z[self.z_stabilizer_matrix.shape[0] + np.random.randint(self.logical_qubit_count)] = 1
                logical_op_indices_z = np.nonzero(logical_op_indices_z)[0]

                logical_op_z = np.zeros(logical_z_stack.shape[1], dtype=np.uint8)

                for i in logical_op_indices_z:
                    logical_op_z += (logical_z_stack.getrow(i).toarray().flatten().astype(np.uint8))

                logical_op_z = logical_op_z % 2

                z_stack = scipy.sparse.vstack([self.z_stabilizer_matrix, logical_op_z]).astype(np.uint8)

                # exit(22)

                bp_osdz = BpOsdDecoder(
                    z_stack,
                    error_rate=0.1,
                    max_iter=10,
                    bp_method="ms",
                    ms_scaling_factor=0.9,
                    schedule="parallel",
                    osd_method="osd_0",
                    osd_order=0,
                )

                dummy_syndrome_z = np.zeros(z_stack.shape[0], dtype=np.uint8)
                dummy_syndrome_z[-1] = 1
                decoded_logical_x = bp_osdz.decode(dummy_syndrome_z)
                logical_size = np.count_nonzero(decoded_logical_x)
                if (logical_size < max_lx) and reduce_logical_basis:
                    candidate_logicals_x.append(decoded_logical_x)
                if logical_size < self.dx:
                    self.dx = logical_size


                if len(candidate_logicals_x) > self.logical_qubit_count:
                    if reduce_logical_basis:
                        self.reduce_logical_operator_basis(
                            candidate_logicals_x, candidate_logicals_z
                        )
                
                self.code_distance = np.min([self.dx, self.dz])



        if reduce_logical_basis:
            self.reduce_logical_operator_basis(
                candidate_logicals_x, candidate_logicals_z
            )
        self.code_distance = np.min([self.dx, self.dz])

        return self.code_distance

    def reduce_logical_operator_basis(
        self, candidate_logicals_x: ArrayLike = [], candidate_logicals_z: ArrayLike = []
    ):
        """
        Reduce the logical operator basis to include lower-weight logicals.

        Parameters
        ----------
        candidate_logicals_x : ArrayLike, optional
            An array of candidate logical x operators to be considered for reducing the basis.
            Defaults to an empty list.
        candidate_logicals_z : ArrayLike, optional
            An array of candidate logical z operators to be considered for reducing the basis.
            Defaults to an empty list.

        """

        if len(candidate_logicals_x) != 0:
            candidate_logicals_x = scipy.sparse.csr_matrix(
                np.array(candidate_logicals_x)
            )

            temp1 = scipy.sparse.vstack([candidate_logicals_x, self.x_logical_operator_basis]).tocsr()

            row_weights = np.diff(temp1.indptr)
            sorted_rows = np.argsort(row_weights)
            temp1 = temp1[sorted_rows, :]

            temp = scipy.sparse.vstack([self.x_stabilizer_matrix, temp1]).tocsr()

            self.x_logical_operator_basis = temp[
                ldpc.mod2.pivot_rows(temp)[self.rank_hx : self.rank_hx + self.logical_qubit_count]
            ]

        if len(candidate_logicals_z) != 0:
            candidate_logicals_z = scipy.sparse.csr_matrix(
                np.array(candidate_logicals_z)
            )

            temp1 = scipy.sparse.vstack([candidate_logicals_z, self.z_logical_operator_basis]).tocsr()

            row_weights = np.diff(temp1.indptr)
            sorted_rows = np.argsort(row_weights)
            temp1 = temp1[sorted_rows, :]

            temp = scipy.sparse.vstack([self.z_stabilizer_matrix, temp1]).tocsr()
            self.z_logical_operator_basis = temp[
                ldpc.mod2.pivot_rows(temp)[self.rank_hz : self.rank_hz + self.logical_qubit_count]
            ]

    def fix_logical_operators(self, fix_logical: str = "X"):

        if not isinstance(fix_logical,str):
            raise TypeError("fix_logical parameter must be a string")

        if fix_logical.lower() == "x":
            temp = self.z_logical_operator_basis@self.x_logical_operator_basis.T
            temp.data = temp.data % 2
            temp = ldpc.mod2.inverse(temp)
            self.z_logical_operator_basis = temp@self.z_logical_operator_basis
            self.z_logical_operator_basis.data = self.z_logical_operator_basis.data % 2
  

        elif fix_logical.lower() == "z":
            temp = self.x_logical_operator_basis@self.z_logical_operator_basis.T
            temp.data = temp.data % 2
            temp = ldpc.mod2.inverse(temp)
            self.x_logical_operator_basis = temp@self.x_logical_operator_basis
            self.x_logical_operator_basis.data = self.x_logical_operator_basis.data % 2
        else:
            raise ValueError("Invalid fix_logical parameter")
        
        


    @property
    def logical_operator_weights(self) -> Tuple[np.ndarray, np.ndarray]:
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
        return f"{self.name} Code: [[N={self.physical_qubit_count}, K={self.logical_qubit_count}, dx<={self.dx}, dz<={self.dz}]]"
      
