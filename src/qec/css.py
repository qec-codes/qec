from typing import Union, Tuple
import numpy as np
import ldpc.mod2
import scipy
import qec.util
from qec.stab_code import StabCode
from ldpc import BpOsdDecoder
import itertools
from tqdm import tqdm
import time


class CssCode(StabCode):
    """
    CSSCode class for generating and manipulating Calderbank-Shor-Steane (CSS) quantum error-correcting codes.

    Attributes:
        hx (Union[np.ndarray, scipy.sparse.spmatrix]): The X-check matrix.
        hz (Union[np.ndarray, scipy.sparse.spmatrix]): The Z-check matrix.
        name (str): A name for this CSS code.
        N (int): Number of qubits in the code.
        K (int): Dimension of the code.
        d (int): Minimum distance of the code.
        lx (Union[np.ndarray, scipy.sparse.spmatrix]): Logical X operator basis.
        lz (Union[np.ndarray, scipy.sparse.spmatrix]): Logical Z operator basis.
    """

    def __init__(
        self,
        hx: Union[np.ndarray, scipy.sparse.spmatrix],
        hz: Union[np.ndarray, scipy.sparse.spmatrix],
        name: str = None,
    ):
        """
        Initialise a new instance of the CssCode class.

        Args:
            hx (Union[np.ndarray, scipy.sparse.spmatrix]): The X-check matrix.
            hz (Union[np.ndarray, scipy.sparse.spmatrix]): The Z-check matrix.
            name (str, optional): A name for this CSS code. Defaults to "CSS".
        """

        # Assign a default name if none is provided
        if name is None:
            self.name = "CSS"
        else:
            self.name = name

        self.lx = None
        self.lz = None

        # Convert matrices to sparse representation and set them as class attributes
        self.hx = qec.util.convert_to_sparse(hx)
        self.hz = qec.util.convert_to_sparse(hz)

        # Calculate the number of qubits from the matrix dimension
        self.N = self.hx.shape[1]

        # Validate the number of qubits for both matrices
        try:
            assert self.N == self.hz.shape[1]
        except AssertionError:
            raise ValueError(
                f"Input matrices hx and hz must have the same number of columns.\
                              Current column count, hx: {hx.shape[1]}; hz: {hz.shape[1]}"
            )

        # Validate if the input matrices commute
        try:
            assert not np.any((self.hx @ self.hz.T).data % 2)
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

        Returns:
            Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]: Logical X and Z operator bases (lx, lz).
        """

        # Compute the kernel of hx and hz matrices

        # Z logicals

        # Compute the kernel of hx
        ker_hx = ldpc.mod2.kernel(self.hx)  # kernel of X-stabilisers
        # Sort the rows of ker_hx by weight
        row_weights = np.diff(ker_hx.indptr)
        sorted_rows = np.argsort(row_weights)
        ker_hx = ker_hx[sorted_rows, :]
        # Z logicals are elements of ker_hx (that commute with all the X-stabilisers) that are not linear combinations of Z-stabilisers
        logical_stack = scipy.sparse.vstack([self.hz, ker_hx]).tocsr()
        rank_hz = ldpc.mod2.rank(self.hz)
        # The first rank_hz pivot_rows of logical_stack are the Z-stabilisers. The remaining pivot_rows are the Z logicals
        pivots = ldpc.mod2.pivot_rows(logical_stack)
        self.lz = logical_stack[pivots[rank_hz:], :]

        # X logicals

        # Compute the kernel of hz
        ker_hz = ldpc.mod2.kernel(self.hz)
        # Sort the rows of ker_hz by weight
        row_weights = np.diff(ker_hz.indptr)
        sorted_rows = np.argsort(row_weights)
        ker_hz = ker_hz[sorted_rows, :]
        # X logicals are elements of ker_hz (that commute with all the Z-stabilisers) that are not linear combinations of X-stabilisers
        logical_stack = scipy.sparse.vstack([self.hx, ker_hz]).tocsr()
        rank_hx = ldpc.mod2.rank(self.hx)
        # The first rank_hx pivot_rows of logical_stack are the X-stabilisers. The remaining pivot_rows are the X logicals
        pivots = ldpc.mod2.pivot_rows(logical_stack)
        self.lx = logical_stack[pivots[rank_hx:], :]

        # set the dimension of the code
        self.K = self.lx.shape[0]

        # find the minimum weight logical operators
        self.dx = self.N
        self.dz = self.N

        for i in range(self.K):
            if self.lx[i].nnz < self.dx:
                self.dx = self.lx[i].nnz
            if self.lz[i].nnz < self.dz:
                self.dz = self.lz[i].nnz
        self.d = np.min([self.dx, self.dz])

        return (self.lx, self.lz)

    def test_logical_basis(self) -> bool:
        """
        Validate the computed logical operator bases.
        """

        # If logical bases are not computed yet, compute them
        if self.lx is None or self.lz is None:
            self.lx, self.lz = self.compute_logical_basis(self.hx, self.hz)
            self.K = self.lx.shape[0]

        # Test dimension
        assert self.K == self.lz.shape[0] == self.lx.shape[0]

        # Check logical basis linearly independent
        assert ldpc.mod2.rank(self.lx) == self.K
        assert ldpc.mod2.rank(self.lz) == self.K

        # Perform various tests to validate the logical bases

        # Check that the logical operators commute with the stabilisers
        assert not np.any((self.lx @ self.hz.T).data % 2)
        assert not np.any((self.lz @ self.hx.T).data % 2)

        # Check that the logical operators anticommute with each other
        test = self.lx @ self.lz.T
        test.data = test.data % 2
        assert ldpc.mod2.rank(test) == self.K

        test = self.lz @ self.lx.T
        test.data = test.data % 2
        assert ldpc.mod2.rank(test) == self.K

        return True

    def estimate_min_distance(
        self, reduce_logicals: bool = False, timeout_seconds: float = 0.25
    ) -> int:
        if self.lx is None or self.lz is None:
            # Compute a basis of the logical operators
            self.lx, self.lz = self.compute_logical_basis()
            # Calculate the dimension of the code
            self.K = self.lx.shape[0]

        self.dx = self.N
        self.dz = self.N
        max_lx = 0
        max_lz = 0

        self.lx = self.lx.tocsr()
        self.lz = self.lz.tocsr()

        rx = ldpc.mod2.rank(self.hx)
        rz = ldpc.mod2.rank(self.hz)

        for i in range(self.K):
            if self.lx[i].nnz > max_lx:
                max_lx = self.lx[i].nnz
            if self.lx[i].nnz < self.dx:
                self.dx = self.lx[i].nnz

            if self.lz[i].nnz > max_lz:
                max_lz = self.lz[i].nnz
            if self.lz[i].nnz < self.dz:
                self.dz = self.lz[i].nnz

        candidate_logicals_x = []
        candidate_logicals_z = []

        x_stack = scipy.sparse.vstack([self.hx, self.lx])
        z_stack = scipy.sparse.vstack([self.hz, self.lz])

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

        bp_osdz = BpOsdDecoder(
            z_stack,
            error_rate=0.1,
            max_iter=10,
            bp_method="ms",
            schedule="parallel",
            ms_scaling_factor=0.9,
            osd_method="osd_0",
            osd_order=0,
        )


        for i in range(self.K):

            dummy_syndrome_x = np.zeros(x_stack.shape[0], dtype=np.uint8)
            dummy_syndrome_z = np.zeros(z_stack.shape[0], dtype=np.uint8)
            dummy_syndrome_x[self.hx.shape[0] + i] = 1
            dummy_syndrome_z[self.hz.shape[0] + i] = 1

            decoded_logical_x = bp_osdz.decode(dummy_syndrome_z)
            logical_size = np.count_nonzero(decoded_logical_x)
            if (logical_size < max_lx) and reduce_logicals:
                candidate_logicals_x.append(decoded_logical_x)
            if logical_size < self.dx:
                self.dx = logical_size

            decoded_logical_z = bp_osdx.decode(dummy_syndrome_x)
            logical_size = np.count_nonzero(decoded_logical_z)
            if (logical_size < max_lz) and reduce_logicals:
                candidate_logicals_z.append(decoded_logical_z)
            if logical_size < self.dz:
                self.dz = logical_size

        if reduce_logicals:
            if len(candidate_logicals_x) != 0:
                candidate_logicals_x = scipy.sparse.csr_matrix(
                    np.array(candidate_logicals_x)
                )

                temp1 = scipy.sparse.vstack([candidate_logicals_x, self.lx]).tocsr()

                row_weights = np.diff(temp1.indptr)
                sorted_rows = np.argsort(row_weights)
                temp1 = temp1[sorted_rows, :]

                temp = scipy.sparse.vstack(
                    [self.hx, temp1]
                ).tocsr()

                self.lx = temp[ldpc.mod2.pivot_rows(temp)[rx : rx + self.K]]

            if len(candidate_logicals_z) != 0:
                candidate_logicals_z = scipy.sparse.csr_matrix(
                    np.array(candidate_logicals_z)
                )

                temp1 = scipy.sparse.vstack([candidate_logicals_z, self.lz]).tocsr()

                row_weights = np.diff(temp1.indptr)
                sorted_rows = np.argsort(row_weights)
                temp1 = temp1[sorted_rows, :]

                temp = scipy.sparse.vstack(
                    [self.hz, temp1]
                ).tocsr()
                self.lz = temp[ldpc.mod2.pivot_rows(temp)[rz : rz + self.K]]

        self.d = np.min([self.dx, self.dz])

        return self.d
    
    @property
    def logical_operator_weights(self)->Tuple[np.ndarray,np.ndarray]:
        x_weights = []
        z_weights = []
        for i in range(self.K):
            x_weights.append(self.lx[i].nnz)
            z_weights.append(self.lz[i].nnz)

        return (np.array(x_weights),np.array(z_weights))

    def __str__(self):
        """
        Return a string representation of the CssCode object.

        Returns:
            str: String representation of the CSS code.
        """
        return f"{self.name} Code: [[N={self.N}, K={self.K}, dx<={self.dx}, dz<={self.dz}]]"
