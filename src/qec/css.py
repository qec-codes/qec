from typing import Union
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
        self.lx, self.lz = self.compute_logical_basis()

        # Calculate the dimension of the code
        self.K = self.lx.shape[0]

        # print(self.K, self.lz.shape[0])

        # Ensure that lx and lz have the same dimension
        assert self.K == self.lz.shape[0]

        self.d = np.nan

    def compute_logical_basis(self):
        """
        Compute the logical operator basis for the given CSS code.

        Returns:
            Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]: Logical X and Z operator bases (lx, lz).
        """

        # Compute the kernel of hx and hz matrices
        
        # Z logicals
                
        ker_hx = ldpc.mod2.kernel(self.hx) #kernel of X-stabilisers
        # Z logicals are elements of ker_hx (commute with all X-stabilisers) that are not linear combinations of Z-stabilisers
        logical_stack = scipy.sparse.vstack([self.hz, ker_hx]).tocsr()
        rank_hz = ldpc.mod2.rank(self.hz)
        assert rank_hz == len(ldpc.mod2.pivot_rows(self.hz)) == (self.hz.shape[1] - ker_hx.shape[0])
        # The first rank_hz pivot_rows of logical_stack are the Z-stabilisers. The remaining pivot_rows are the Z logicals
        pivots = ldpc.mod2.pivot_rows(logical_stack)
        print(pivots[rank_hz:])
        lz = logical_stack[pivots[rank_hz:], :]

        # X logicals
        ker_hz = ldpc.mod2.kernel(self.hz)
        logical_stack = scipy.sparse.vstack([self.hx, ker_hz]).tocsr()
        rank_hx = ldpc.mod2.rank(self.hx)
        assert rank_hx == len(ldpc.mod2.pivot_rows(self.hx)) == (self.hx.shape[1] - ker_hz.shape[0])
        # The first rank_hx pivot_rows of logical_stack are the X-stabilisers. The remaining pivot_rows are the X logicals
        pivots = ldpc.mod2.pivot_rows(logical_stack)
        print(pivots[rank_hx:])
        lx = logical_stack[pivots[rank_hx:], :]

        assert ldpc.mod2.rank(lx) == lx.shape[0] == self.N - rank_hx - rank_hz
        assert ldpc.mod2.rank(lz) == lz.shape[0] == self.N - rank_hz - rank_hx

        # assert lx.shape[0] == self.N - rank_hx - rank_hz

        return (lx, lz)

    def test_logical_basis(self)->bool:
        """
        Validate the computed logical operator bases.
        """

        # If logical bases are not computed yet, compute them
        if self.lx is None or self.lz is None:
            self.lx, self.lz = self.compute_logical_basis(self.hx, self.hz)
            self.K = self.lx.shape[0]

        # Test dimension
        assert self.K == self.lz.shape[0] == self.lx.shape[0]

        # Perform various tests to validate the logical bases
        assert not np.any((self.lx @ self.hz.T).data % 2)
        test = self.lx @ self.lz.T
        test.data = test.data % 2
        assert ldpc.mod2.rank(test) == self.K

        assert not np.any((self.lz @ self.hx.T).data % 2)
        test = self.lz @ self.lx.T
        test.data = test.data % 2
        assert ldpc.mod2.rank(test) == self.K

        return True
    
    # def temp(self):
    #     assert self.K == self.lz.shape[0] == self.lx.shape[0]
    #     print(ldpc.mod2.rank(self.lx))
    
    def estimate_min_distance(self, timeout_seconds: float = 0.25) -> int:

        # if self.lx is None or self.lz is None:
        #     # Compute a basis of the logical operators
        #     self.lx, self.lz = self.compute_logical_basis()
        #     # Calculate the dimension of the code
        #     self.K = self.lx.shape[0]

        # min_x = self.N
        # min_z = self.N
        # max_lx = 0
        # max_lz = 0

        # self.lx = self.lx.tocsr()
        # self.lz = self.lz.tocsr()

        # for i in range(self.K):
        #     if self.lx[i].nnz > max_lx:
        #         max_lx = self.lx[i].nnz
        #     if self.lx[i].nnz < min_x:
        #         min_x = self.lx[i].nnz

        #     if self.lz[i].nnz > max_lz:
        #         max_lz = self.lz[i].nnz
        #     if self.lz[i].nnz < min_z:
        #         min_z = self.lz[i].nnz

        rankHx = ldpc.mod2.rank(self.hx)
        print("Rank hx: ", rankHx)

        self.test_logical_basis()

        print(ldpc.mod2.rank(self.lx))
        print(self.lx.shape)
        print(self.lz.shape)

        assert ldpc.mod2.rank(self.lx)  == self.K

        temp = scipy.sparse.vstack([self.hx, self.lx])

        rank=ldpc.mod2.rank(temp)

        print(rank-rankHx)

        print(ldpc.mod2.pivot_rows(temp)[rankHx:])


        return 1

        candidate_logicals_x = []
        candidate_logicals_z = []

        # for i in range(self.K):
        #     # if self.lx[i].nnz < min_x:
        #     #     min_x = self.lx[i].nnz

        #     # if self.lz[i].nnz < min_z:
        #     #     min_z = self.lz[i].nnz

        #     self.lx=self.lx.tocsr()
        #     self.lz = self.lz.tocsr()

        #     x_stack = scipy.sparse.vstack([self.hx, self.lx[i]])
        #     z_stack = scipy.sparse.vstack([self.hz, self.lz[i]])

        #     bp_osdx = BpOsdDecoder(
        #         x_stack,
        #         error_rate = 0.1,
        #         max_iter = 10,
        #         bp_method = "ms",
        #         ms_scaling_factor = 0.9,
        #         schedule = "parallel",
        #         osd_method = "osd0",
        #         osd_order = 0)
            
        #     bp_osdz = BpOsdDecoder(
        #         z_stack,
        #         error_rate = 0.1,
        #         max_iter = 10,
        #         bp_method = "ms",
        #         schedule = "parallel",
        #         ms_scaling_factor = 0.9,
        #         osd_method = "osd0",
        #         osd_order = 0)

        #     dummy_syndrome_x = np.zeros( self.hx.shape[0] + 1, dtype=np.uint8)
        #     dummy_syndrome_z = np.zeros( self.hz.shape[0] + 1, dtype=np.uint8)
        #     dummy_syndrome_x[-1] = 1
        #     dummy_syndrome_z[-1] = 1

        #     decoded_logical_x = bp_osdz.decode(dummy_syndrome_z)
        #     logical_size = np.count_nonzero(decoded_logical_x)
        #     if logical_size < max_lx:
        #         candidate_logicals_x.append(decoded_logical_x)
        #     if logical_size < min_x:
        #         min_x = logical_size

        #     decoded_logical_z = bp_osdx.decode(dummy_syndrome_x)
        #     logical_size = np.count_nonzero(decoded_logical_z)
        #     if logical_size < max_lz:
        #         candidate_logicals_z.append(decoded_logical_z)
        #     if logical_size < min_z:
        #         min_z = logical_size

        # candidate_logicals_z = scipy.sparse.csr_matrix(np.array(candidate_logicals_z))
        # candidate_logicals_x = scipy.sparse.csr_matrix(np.array(candidate_logicals_x))

        rx = ldpc.mod2.rank(self.hx)
        rz = ldpc.mod2.rank(self.hz)

        # temp = scipy.sparse.vstack([self.hx,candidate_logicals_x,self.lx]).tocsr()
        temp = scipy.sparse.vstack([self.hx,self.lx])

        # self.lx = temp[ldpc.mod2.pivot_rows(temp)[rx:rx+self.K]]
        print("rank x: ", rx)
        print(len(ldpc.mod2.pivot_rows(temp)))
        print(self.K)
        print(len(ldpc.mod2.pivot_rows(temp)[rx:]))
        print()

        # temp = scipy.sparse.vstack([self.hz,candidate_logicals_z,self.lz]).tocsr()
        temp = scipy.sparse.vstack([self.hz,self.lz]).tocsr()

        self.lz = temp[ldpc.mod2.pivot_rows(temp)[rz:rz+self.K]] 

        assert self.lx.shape[0] == self.K

        self.lx.eliminate_zeros()
        self.lz.eliminate_zeros()



        # combs = list(itertools.combinations(range(self.K), 2))
        # no_combs = len(combs)

        # # Get the start time
        # start_time = time.time()

        # # Your while loop
        # while True:
        #     # Your loop logic here
        #     # print("hello")
            
        #     # Check if the specified time has elapsed
        #     elapsed_time = time.time() - start_time
        #     if elapsed_time >= timeout_seconds:
        #         break  # Exit the loop if the timeout has been reached

        #     dummy_syndrome_x = np.zeros( self.hx.shape[0] + self.K, dtype=np.uint8)
        #     dummy_syndrome_z = np.zeros( self.hz.shape[0] + self.K, dtype=np.uint8)
      
        #     i, j = combs[np.random.randint(no_combs)]

        #     dummy_syndrome_x[self.hx.shape[0] + i] = 1
        #     dummy_syndrome_x[self.hx.shape[0] + j] = 1

        #     dummy_syndrome_z[self.hz.shape[0] + i] = 1
        #     dummy_syndrome_z[self.hz.shape[0] + j] = 1


        #     # print(dummy_syndrome_x)
        #     # print(dummy_syndrome_z)

        #     logical_x = bp_osdz.decode(dummy_syndrome_z)
        #     logical_z = bp_osdx.decode(dummy_syndrome_x)
            
        #     dx = np.count_nonzero(logical_x)
        #     dz = np.count_nonzero(logical_z)


        #     if dz < min_z:
        #         min_z = dz
        #     if dx < min_x:
        #         min_x = dx


        print(min_x, min_z)
        return np.min([min_x, min_z])






