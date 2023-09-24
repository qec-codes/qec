from typing import Union
import numpy as np
from udlr import gf2sparse
import scipy
import qec.util


class StabCode(object):
    def __init__(
        self,
        pauli_stabs: np.ndarray = None,
        h: Union[np.ndarray, scipy.sparse.spmatrix] = None,
        name: str = None,
    ):
        self.name = name if name else "Stabiliser"
        self.h = None
        self.d = np.nan

        if pauli_stabs is not None:
            if not isinstance(pauli_stabs, np.ndarray):
                raise TypeError("Input must be a numpy array of Pauli elements.")

            self.N = pauli_stabs.shape[1]

            if not pauli_stabs.dtype.type == np.str_:
                if not pauli_stabs.dtype in [np.uint8, np.int8, np.int32, np.int64]:
                    raise TypeError("Input dtype must be uint8, int8, int, or str.")

            if pauli_stabs.dtype in [int, np.int8, np.uint8]:
                if not np.all(np.isin(pauli_stabs, [0, 1, 2, 3])):
                    raise ValueError(
                        "Please use the following mapping for Pauli operators of integers: 0 -> I, 1 -> X, 2 -> Y, 3 -> Z."
                    )

            if pauli_stabs.dtype.type == np.str_:
                if not np.all(np.isin(pauli_stabs, ["I", "X", "Y", "Z"])):
                    raise ValueError(
                        "Please use the following mapping for the string representation of Pauli operators: I -> 'I', X -> 'X', Y -> 'Y', Z -> 'Z'."
                    )

            row_idx = []
            col_idx = []
            data = []
            for i, row in enumerate(pauli_stabs):
                for j, elem in enumerate(row):
                    if elem == 1 or elem == "X":
                        row_idx.append(i)
                        col_idx.append(j)
                        data.append(1)

                    elif elem == 2 or elem == "Y":
                        row_idx.append(i)
                        col_idx.append(j)
                        data.append(1)
                        row_idx.append(i)
                        col_idx.append(j + self.N)
                        data.append(1)

                    elif elem == 3 or elem == "Z":
                        row_idx.append(i)
                        col_idx.append(j + self.N)
                        data.append(1)

            self.h = scipy.sparse.csr_matrix(
                (data, (row_idx, col_idx)), shape=(pauli_stabs.shape[0], 2 * self.N)
            )

        elif h is not None:
            if not h.shape[1] % 2 == 0:
                raise ValueError("Input matrix h must have an even number of columns.")
            self.h = qec.util.convert_to_sparse(h)
            self.N = h.shape[1] // 2

        else:
            raise ValueError(
                "Please provide either a stabiliser matrix or a check matrix."
            )

        self.h_left = self.h[:, : self.N]
        self.h_right = self.h[:, self.N :]

        # check that the stabilisers commute

        if np.any(
            ((self.h_left @ self.h_right.T) + (self.h_right @ self.h_left.T)).data % 2
        ):
            raise ValueError("Stabilisers do not commute.")

        self.logical_basis = self.compute_logical_basis()

        self.K = self.logical_basis.shape[0] // 2

        self.logical_basis_left = self.logical_basis[:, : self.N]
        self.logical_basis_right = self.logical_basis[:, self.N :]

    def compute_logical_basis(self):
        kernel_h = gf2sparse.kernel(self.h)

        rank = kernel_h.shape[1] - kernel_h.shape[0]

        kernel_h_left = kernel_h[:, : self.N]
        kernel_h_right = kernel_h[:, self.N :]

        swapped_kernel = scipy.sparse.hstack([kernel_h_right, kernel_h_left])

        # Compute the logical operator basis
        logical_stack = scipy.sparse.hstack([self.h.T, swapped_kernel.T])
        plu = gf2sparse.PluDecomposition(logical_stack)
        kernel_rows = plu.pivots[rank:] - rank
        l_basis = kernel_h[kernel_rows]

        return l_basis

    def test_logical_basis(self):
        """
        Validate the computed logical operator bases.
        """

        assert not np.any(
            (
                self.h_right @ self.logical_basis_left.T
                + self.h_left @ self.logical_basis_right.T
            ).data
            % 2
        )

        test = (
            self.logical_basis_right @ self.logical_basis_left.T
            + self.logical_basis_left @ self.logical_basis_right.T
        )
        test.data = test.data % 2

        test_plu = gf2sparse.PluDecomposition(test)
        assert test_plu.rank == self.logical_basis.shape[0]

    def __str__(self):
        """
        Return a string representation of the CssCode object.

        Returns:
            str: String representation of the CSS code.
        """
        return f"{self.name} Code: [[N={self.N}, K={self.K}, dmin={self.d}]]"
