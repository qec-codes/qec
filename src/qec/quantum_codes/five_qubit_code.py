from qec.stabilizer_code.stabilizer_code import StabilizerCode


class FiveQubitCode(StabilizerCode):
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
    code_distance : int
        The distance of the quantum code. For the five-qubit code, this is set to 3.

    Inherits
    --------
    StabilizerCode
        The base class providing functionalities for stabilizer-based quantum
        error-correcting codes, including initialization, distance computation,
        and parameter retrieval.

    Examples
    --------
    >>> five_qubit = FiveQubitCode()
    >>> five_qubit.phyiscal_qubit_count
    5
    >>> five_qubit.logical_qubit_count
    1
    >>> five_qubit.code_distance
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
        pauli_stabilizers = [["XZZXI"], ["IXZZX"], ["XIXZZ"], ["ZXIXZ"]]

        # Initialize the StabilizerCode with the defined stabilizers and a custom name
        super().__init__(stabilizers=pauli_stabilizers, name="5-Qubit Code")

        # Set the distance attribute specific to the five-qubit code
        self.code_distance = 3
