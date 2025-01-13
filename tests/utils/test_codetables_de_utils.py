import pytest
import json
from unittest.mock import patch, MagicMock

from qec.utils.codetables_de_utils import (
    get_codetables_de_matrix,
    pcm_to_csr_matrix,
    load_codetables_de_matrix_from_json,
)

# ---------------------------------------------------------------------
# Mocks and Sample Data
# ---------------------------------------------------------------------
MOCK_HTML_SUCCESS = """
<html>
<body>
    <table>
        <tr><td>lower bound:</td><td>3</td></tr>
        <tr><td>upper bound:</td><td>5</td></tr>
    </table>
    <pre>
    Construction of a [[10,4,5]] code
    stabilizer matrix:
      [1 1 0|0 0 1]
      [0 1 1|1 0 0]
last modified: 2024-01-01
    </pre>
</body>
</html>
"""

MOCK_HTML_NO_MATRIX = """
<html>
<body>
    <table>
        <tr><td>lower bound:</td><td>3</td></tr>
        <tr><td>upper bound:</td><td>5</td></tr>
    </table>
    <p>No matrix here!</p>
</body>
</html>
"""

MOCK_HTML_BAD_STATUS = """This content won't matter because status != 200"""


# ---------------------------------------------------------------------
# Tests for get_codetables_de_matrix
# ---------------------------------------------------------------------
@patch("qec.utils.codetables_de_utils.requests.get")
def test_get_codetables_de_matrix_success(mock_get, tmp_path):
    """Test the successful retrieval and parsing of a code from HTML."""
    mock_resp = MagicMock(status_code=200)
    mock_resp.text = MOCK_HTML_SUCCESS
    mock_get.return_value = mock_resp

    # No file write
    data_dict = get_codetables_de_matrix(q=4, n=10, k=4)
    assert data_dict["n"] == 10
    assert data_dict["k"] == 4
    assert data_dict["d_lower"] == "3"
    assert data_dict["d_upper"] == "5"
    assert len(data_dict["pcm"]) == 2  # We found 2 rows in the matrix
    assert data_dict["pcm"][0] == [0, 1, 5]  # Indices of '1'
    assert data_dict["pcm"][1] == [1, 2, 3]

    # Test file writing scenario
    output_file = tmp_path / "matrix.json"
    written_data = get_codetables_de_matrix(
        q=4, n=10, k=4, output_json_path=str(output_file), write_to_file=True
    )
    assert output_file.exists(), "JSON file should have been written."
    with open(output_file, "r") as f:
        loaded = json.load(f)
    assert loaded == written_data


@patch("qec.utils.codetables_de_utils.requests.get")
def test_get_codetables_de_matrix_no_matrix(mock_get):
    """Test the scenario where no stabilizer matrix lines are found."""
    mock_resp = MagicMock(status_code=200)
    mock_resp.text = MOCK_HTML_NO_MATRIX
    mock_get.return_value = mock_resp

    with pytest.raises(ValueError, match="No valid stabilizer matrix found"):
        get_codetables_de_matrix(q=4, n=10, k=4)


@patch("qec.utils.codetables_de_utils.requests.get")
def test_get_codetables_de_matrix_bad_status(mock_get):
    """Test when the response status code is not 200."""
    mock_resp = MagicMock(status_code=404)
    mock_resp.text = MOCK_HTML_BAD_STATUS
    mock_get.return_value = mock_resp

    with pytest.raises(ValueError, match="Failed to retrieve data"):
        get_codetables_de_matrix(q=4, n=10, k=4)


def test_get_codetables_de_matrix_write_no_path():
    """Test attempting to write output with no output_json_path given."""
    with pytest.raises(ValueError, match="output_json_path must be provided"):
        get_codetables_de_matrix(q=4, n=10, k=4, write_to_file=True)


# ---------------------------------------------------------------------
# Tests for pcm_to_csr_matrix
# ---------------------------------------------------------------------
def test_pcm_to_csr_matrix_auto_width():
    pcm = [[0, 2], [1]]
    mat = pcm_to_csr_matrix(pcm)
    assert mat.shape == (2, 3)  # auto-detected from max col index=2 => width=3
    assert mat[0, 0] == 1
    assert mat[0, 1] == 0
    assert mat[0, 2] == 1
    assert mat[1, 1] == 1


def test_pcm_to_csr_matrix_fixed_width():
    pcm = [[0, 2], [1]]
    mat = pcm_to_csr_matrix(pcm, num_cols=4)
    assert mat.shape == (2, 4)
    assert mat[0, 2] == 1


def test_pcm_to_csr_matrix_empty():
    mat = pcm_to_csr_matrix([])
    assert mat.shape == (0, 0)


def test_pcm_to_csr_matrix_out_of_range():
    # If the user sets num_cols=2 but data has col index=2 => out of range
    pcm = [[0, 2]]
    with pytest.raises(ValueError, match="Column index 2 is out of range"):
        pcm_to_csr_matrix(pcm, num_cols=2)


# ---------------------------------------------------------------------
# Tests for load_codetables_de_matrix_from_json
# ---------------------------------------------------------------------
def test_load_codetables_de_matrix_from_json_good():
    # A dict with all required fields
    data = {
        "n": 10,
        "k": 4,
        "d_lower": "3",
        "d_upper": "5",
        "url": "https://codetables.de",
        "pcm": [[0, 2], [1, 3]],
    }
    mat, returned = load_codetables_de_matrix_from_json(data)
    assert returned is data
    assert mat.shape == (2, 4)  # auto-detected width=4 (max col=3 => width=4)
    assert mat[0, 0] == 1
    assert mat[1, 3] == 1


def test_load_codetables_de_matrix_from_json_string():
    data_str = json.dumps(
        {
            "n": 10,
            "k": 4,
            "d_lower": "3",
            "d_upper": "5",
            "url": "https://codetables.de",
            "pcm": [[0, 2], [1, 3]],
        }
    )
    mat, returned = load_codetables_de_matrix_from_json(data_str)
    assert mat.shape == (2, 4)
    assert returned["n"] == 10


def test_load_codetables_de_matrix_from_json_missing_keys():
    data_missing = {"n": 10, "k": 4, "pcm": [[0, 2], [1, 3]]}
    with pytest.raises(ValueError, match="missing required keys"):
        load_codetables_de_matrix_from_json(data_missing)


def test_load_codetables_de_matrix_from_json_bad_type():
    with pytest.raises(ValueError, match="must be a dict or a JSON string"):
        load_codetables_de_matrix_from_json(12345)
