from qec.stabilizer_code import StabilizerCode
from qec.utils.codetables_de_utils import get_codetables_de_matrix, pcm_to_csr_matrix


class CodeTablesDE(StabilizerCode):
    """
    A code object built from data obtained from Markus Grassl's codetables.de website (with `q=4`).

    This class inherits from `StabilizerCode` and initialises the code
    by querying the codetables.de website for the specified parameters `(n, k)`,
    constructing the stabilizer (PCM) matrix, and passing it up to the
    `StabilizerCode` parent class.

    Parameters
    ----------
    physical_qubit_count : int
        Length of the code (number of physical qubits).
    logical_qubit_count : int
        Dimension of the code (number of logical qubits).

    Attributes
    ----------
    name : str
        Name assigned to this code instance. Defaults to "CodeTablesDE".
    url : str
        The URL from which this code's data was retrieved.
    code_distance : int
        The code's minimum distance. This is updated if the reported upper bound
        from the codetables.de website is smaller than the base class default.

    See Also
    --------
    StabilizerCode : Parent class providing stabilizer code functionality.
    get_codetables_de_matrix : Function that queries codetables.de to retrieve code data.
    pcm_to_csr_matrix : Function that converts a PCM-like list of column indices into a CSR matrix.

    Notes
    -----
    - The data is retrieved from:
      https://codetables.de
      maintained by Markus Grassl.
    """

    def __init__(self, physical_qubit_count: int, logical_qubit_count: int):
        """
        Initialise a code from Markus Grassl's codetables.de website with `q=4`, `n`, and `k`.

        This method queries the codetables.de database for a stabilizer matrix
        describing a code with parameters (q=4, n, k). The matrix is then
        converted to a CSR (Compressed Sparse Row) format and passed to
        the parent `StabilizerCode` class.

        Parameters
        ----------
        n : int
            Length of the code (number of physical qubits).
        k : int
            Dimension of the code (number of logical qubits).

        Notes
        -----
        - `d_upper` from the query result is used to potentially update `self.code_distance`
          if it is smaller than the default distance assigned by `StabilizerCode`.
        - Since this code is defined over GF(4), `q` is hardcoded as 4.
        - Data is retrieved from Markus Grassl's website (https://codetables.de).

        Raises
        ------
        ValueError
            If no valid matrix data can be retrieved from codetables.de, or
            if the site indicates that such a code does not exist.
        """
        # Retrieve code data from codetables.de
        ct_dict = get_codetables_de_matrix(
            q=4, n=physical_qubit_count, k=logical_qubit_count
        )

        # Construct the stabilizer matrix in CSR format
        # The matrix is 2*n columns wide, as is typical for GF(4) stabilizers.
        stabilizer_matrix = pcm_to_csr_matrix(
            ct_dict["pcm"], num_cols=2 * int(ct_dict["n"])
        )

        # Initialise the parent class with this stabilizer matrix
        super().__init__(stabilizers=stabilizer_matrix)

        # Name of this code and the URL from which we retrieved it
        self.name = "CodeTablesDE"
        self.url = ct_dict["url"]

        # Update distance if the reported upper bound is smaller than the default
        if int(ct_dict["d_upper"]) < self.code_distance:
            self.code_distance = int(ct_dict["d_upper"])
