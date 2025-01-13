import logging
# Suppress debug and info messages from urllib3 and requests libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

from scipy.sparse import csr_matrix

import requests
from bs4 import BeautifulSoup
import json

def get_codetables_de_matrix(q, n, k, output_json_path=None, write_to_file=False):
    """
    Retrieve quantum code data from Markus Grassl's codetables.de website.

    This function queries the URL:
    ``https://codetables.de/QECC/QECC.php?q={q}&n={n}&k={k}``,
    attempting to fetch data for a quantum code with the specified parameters
    over GF(q). The HTML response is parsed to extract:

    - The lower bound (``d_lower``) and upper bound (``d_upper``) on the code distance.
    - The stabilizer matrix (as lines within a ``<pre>`` block).

    The stabilizer matrix is then converted into a list of rows, each containing
    the column indices of any '1' entries (the ``pcm``). The result is returned
    as a dictionary, and optionally written to a JSON file.

    Parameters
    ----------
    q : int
        The field size (e.g. 2, 4, etc.).
    n : int
        The length of the code (number of physical qubits).
    k : int
        The dimension of the code (number of logical qubits).
    output_json_path : str or None, optional
        File path to which the resulting dictionary will be written if
        ``write_to_file`` is set to True. If None and ``write_to_file`` is True,
        raises a ValueError.
    write_to_file : bool, optional
        Whether to write the resulting dictionary to a JSON file.

    Returns
    -------
    dict
        A dictionary with the fields:
        ``{"n", "k", "d_upper", "d_lower", "url", "pcm"}``.

        - ``pcm`` is a list of lists, where each inner list contains the column
          indices of '1's for that row of the stabilizer matrix.
        - ``url`` is the codetables.de URL used for the query.
        - ``d_upper`` and ``d_lower`` are the distance bounds, if found.

    Raises
    ------
    ValueError
        If the server response is not 200 OK, or if no valid stabilizer matrix
        lines could be found in the HTML (i.e., no code data for those parameters).
        Also raised if ``write_to_file`` is True and ``output_json_path`` is None.

    Notes
    -----
    - Data is sourced from `codetables.de <https://codetables.de>`__,
      maintained by Markus Grassl.
    - The function does not return an actual matrix but rather a convenient
      representation of it (the ``pcm``). Use ``pcm_to_csr_matrix`` or another
      helper to convert it into a numerical/sparse form.
    """
    url = f"https://codetables.de/QECC/QECC.php?q={q}&n={n}&k={k}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(
            f"Failed to retrieve data (status code: {resp.status_code}). URL was: {url}"
        )

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1) Extract lower and upper distance bounds from <table> elements
    lower_bound = None
    upper_bound = None
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                heading = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)
                if "lower bound" in heading:
                    lower_bound = value
                elif "upper bound" in heading:
                    upper_bound = value

    # 2) Extract the stabilizer matrix lines from <pre> tags
    matrix_lines = []
    for tag in soup.find_all("pre"):
        text = tag.get_text()
        if "stabilizer matrix" in text.lower():
            lines = text.splitlines()
            capture = False
            for line in lines:
                if "stabilizer matrix" in line.lower():
                    capture = True
                    continue
                if capture:
                    # Stop at 'last modified:' or if the line is empty
                    if "last modified:" in line.lower():
                        break
                    if line.strip() != "":
                        matrix_lines.append(line.strip())

    if not matrix_lines:
        raise ValueError(f"No valid stabilizer matrix found at {url}")

    # 3) Convert lines -> list of column-index lists
    pcm_list = []
    for line in matrix_lines:
        line = line.strip().strip("[]").replace("|", " ")
        elements = line.split()
        row_cols = [i for i, val in enumerate(elements) if val == "1"]
        pcm_list.append(row_cols)

    if not pcm_list:
        raise ValueError(f"No valid rows containing '1' found at {url}")

    # 4) Build final dictionary
    result_dict = {
        "n": n,
        "k": k,
        "d_upper": upper_bound,
        "d_lower": lower_bound,
        "url": url,
        "pcm": pcm_list,
    }

    # 5) Optionally write to JSON file
    if write_to_file:
        if output_json_path is None:
            raise ValueError("output_json_path must be provided if write_to_file=True.")
        with open(output_json_path, "w") as out_file:
            json.dump(result_dict, out_file, indent=2)

    return result_dict


def pcm_to_csr_matrix(pcm, num_cols=None):
    """
    Convert a "pcm" to a SciPy CSR matrix.

    Each inner list of ``pcm`` is interpreted as the column indices in which
    row `i` has a value of 1. The resulting CSR matrix will thus have as many
    rows as ``len(pcm)``. The number of columns can either be:

    - Inferred automatically (``num_cols=None``) by taking 1 + max(column index).
    - Specified by the user. If a column index is >= num_cols, a ValueError is raised.

    Parameters
    ----------
    pcm : list of lists of int
        Each element ``pcm[i]`` is a list of column indices where row i has '1'.
    num_cols : int or None, optional
        The desired number of columns (width of the matrix).
        If None, the width is auto-detected from the maximum column index.

    Returns
    -------
    csr_matrix
        A sparse matrix of shape ``(len(pcm), num_cols)``.

    Raises
    ------
    ValueError
        If any column index exceeds the specified ``num_cols``.
        Also raised if no rows or invalid columns exist.

    See Also
    --------
    get_codetables_de_matrix : Returns a dictionary with ``pcm`` field from codetables.de.

    Notes
    -----
    Data is typically retrieved from `codetables.de <https://codetables.de>`__
    and fed into this function to produce a numerical/sparse representation.
    """
    if not pcm:
        # No rows at all => shape (0, num_cols) or (0, 0) if num_cols is None
        if num_cols is None:
            return csr_matrix((0, 0), dtype=int)
        else:
            return csr_matrix((0, num_cols), dtype=int)

    row_indices = []
    col_indices = []
    data = []

    max_col_found = -1

    # Collect row/col for each '1' entry
    for row_idx, col_list in enumerate(pcm):
        for c in col_list:
            row_indices.append(row_idx)
            col_indices.append(c)
            data.append(1)
            if c > max_col_found:
                max_col_found = c

    num_rows = len(pcm)

    # Auto-detect columns if not specified
    if num_cols is None:
        num_cols = max_col_found + 1
    else:
        # If the user specified num_cols, ensure the data fits
        if max_col_found >= num_cols:
            raise ValueError(
                f"Column index {max_col_found} is out of range for a matrix of width {num_cols}."
            )

    return csr_matrix(
        (data, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=int
    )


def load_codetables_de_matrix_from_json(json_data):
    """
    Construct a CSR matrix from a codetables.de JSON/dict output.

    This function takes either a dictionary (as returned by
    ``get_codetables_de_matrix``) or a JSON string that decodes to the same
    structure, and converts the ``pcm`` field into a SciPy CSR matrix.

    Parameters
    ----------
    json_data : dict or str
        Must contain at least the following keys:
        ``{"n", "k", "d_upper", "d_lower", "url", "pcm"}``.
        - ``pcm`` is a list of lists of column indices.

    Returns
    -------
    csr_matrix
        The stabilizer matrix in CSR format.
    dict
        The original dictionary that was passed in (or parsed from JSON).

    Raises
    ------
    ValueError
        If ``json_data`` is not a dict, if it cannot be parsed into one,
        or if required keys are missing.

    Notes
    -----
    - Data is assumed to come from Markus Grassl's `codetables.de <https://codetables.de>`__.
    - This utility is helpful when the data is stored or transmitted in JSON form
      but needs to be loaded back into a matrix representation for further processing.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    if not isinstance(json_data, dict):
        raise ValueError(
            "json_data must be a dict or a JSON string that decodes to a dict."
        )

    required_keys = {"n", "k", "d_upper", "d_lower", "url", "pcm"}
    if not required_keys.issubset(json_data.keys()):
        raise ValueError(f"JSON data missing required keys: {required_keys}")

    pcm = json_data["pcm"]
    sparse_matrix = pcm_to_csr_matrix(pcm)
    return sparse_matrix, json_data
