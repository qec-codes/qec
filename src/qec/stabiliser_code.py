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

        if pauli_stabs is not None:
            if not isinstance(pauli_stabs, np.ndarray):
                TypeError("Input must be a numpy array of Pauli elements.")
            if pauli_stabs.dtyle not in [np.uint8, np.int8, int, str]:
                TypeError("Input dtype must be uint8, int8, int, or str.")
            if pauli_stabs.dtype in [int, np.int8, np.uint8]:
                if not np.all(np.isin(pauli_stabs, [0, 1, 2, 3])):
                    ValueError(
                        "Please use the following mapping for Pauli operators of integers: 0 -> I, 1 -> X, 2 -> Y, 3 -> Z."
                    )
            if pauli_stabs.dtype == str:
                if not np.all(np.isin(pauli_stabs, ["I", "X", "Y", "Z"])):
                    ValueError(
                        "Please use the following mapping for the string representation of Pauli operators: I -> 'I', X -> 'X', Y -> 'Y', Z -> 'Z'."
                    )

        NotImplemented
