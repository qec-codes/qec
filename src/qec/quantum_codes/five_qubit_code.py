import numpy as np
from qec.stabilizer_code.stabilizer_code import StabiliserCode


class FiveQubitCode(StabiliserCode):
    """
    Five-Qubit Quantum Error-Correcting Code.

    The `FiveQubitCode` class implements the [[5, 1, 3]] quantum stabilizer code,
    which is the smallest possible quantum error-correcting code capable of
    correcting an arbitrary single-qubit error. This code encodes one logical
    qubit into five physical qubits and has a distance of three, allowing it
    to detect up to two errors and correct one.

    Parameters
    ----------
    None

    Attributes
    ----------
    d : int
        The distance of the quantum code. For the five-qubit code, this is set to 3.

    Inherits
    --------
    StabiliserCode
        The base class providing functionalities for stabilizer-based quantum
        error-correcting codes, including initialization, distance computation,
        and parameter retrieval.

    Examples
    --------
    >>> five_qubit = FiveQubitCode()
    >>> five_qubit.n
    5
    >>> five_qubit.k
    1
    >>> five_qubit.d
    3
    """

    def __init__(self):
        """
        Initialize the Five-Qubit Code with predefined Pauli stabilizers.

        The constructor sets up the stabilizer generators for the [[5, 1, 3]]
        quantum code using their corresponding Pauli strings. It then calls the
        superclass initializer to establish the stabilizer matrix and other
        essential parameters.

        Parameters
        ----------
        None

        Raises
        ------
        ValueError
            If the provided stabilizer generators do not satisfy the necessary
            commutation relations required for a valid stabilizer code.
        """
        # Define the Pauli stabilizer generators for the five-qubit code
        pauli_stabilisers = [["XZZXI"], ["IXZZX"], ["XIXZZ"], ["ZXIXZ"]]

        # Initialize the StabiliserCode with the defined stabilizers and a custom name
        super().__init__(stabilisers=pauli_stabilisers, name="5-Qubit Code")

        # Set the distance attribute specific to the five-qubit code
        self.d = 3
