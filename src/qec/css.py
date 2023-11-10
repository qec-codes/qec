from typing import Union
import numpy as np
import ldpc.mod2
import scipy
import qec.util
from qec.stab_code import StabCode


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
        
        kernel_hx = ldpc.mod2.kernel(self.hx)
        rank_hx = self.hx.shape[1] - kernel_hx.shape[0]
        
        kernel_hz = ldpc.mod2.kernel(self.hz)
        rank_hz = self.hz.shape[1] - kernel_hz.shape[0]

        # Compute the logical Z operator basis
        logical_stack = scipy.sparse.vstack([self.hz, kernel_hx]).tocsr()
        ## find the first set of linearly independent rows
        p_rows = ldpc.mod2.pivot_rows(logical_stack)
        ## The linearly independents rows \in kernel_hz are logical operators
        lz = logical_stack[p_rows[rank_hz:]]

        # Compute the logical X operator basis
        logical_stack = scipy.sparse.vstack([self.hx, kernel_hz]).tocsr()
        p_rows = ldpc.mod2.pivot_rows(logical_stack)
        lx = logical_stack[p_rows[rank_hx:]]

        return (lx, lz)

    def test_logical_basis(self)->bool:
        """
        Validate the computed logical operator bases.
        """

        # If logical bases are not computed yet, compute them
        if self.lx is None or self.lz is None:
            self.lx, self.lz = self.compute_logical_basis(self.hx, self.hz)

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
