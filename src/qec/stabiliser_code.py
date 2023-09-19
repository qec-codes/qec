from typing import Union
import numpy as np
from ldpc2 import gf2sparse
import scipy
import qec.util


class StabiliserCode(object):
    def __init__(
        self,
        pauli_stabs: np.ndarray = None,
        h: Union[np.ndarray, scipy.sparse.spmatrix] = None,
        name: str = None,
    ):
        name if name else "Stabiliser"
        self.h = None

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
            self.N = h.shape[1]

        else:
            raise ValueError(
                "Please provide either a stabiliser matrix or a check matrix."
            )


