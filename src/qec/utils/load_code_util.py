import json
from typing import Union, Dict, Any
import inspect
from pathlib import Path
from qec.utils.sparse_binary_utils import dict_to_binary_csr_matrix
import qec.code_constructions


def load_code(filepath: Union[str, Path]):
    """
    Loads a quantum error correction code from a JSON file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to JSON file containing code data

    Returns
    -------
    Union[StabilizerCode, CSSCode, HypergraphProductCode]
        The quantum error correction instance.

    Raises
    ------
        FileNotFoundError
            If filepath does not exist.
        ValueError
            If code class in JSON is not recognized.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    with open(filepath, "r") as f:
        code_data = json.load(f)

    if "class_name" not in code_data.keys():
        raise ValueError("Missing 'class_name' in input file.")

    class_name = eval(
        "qec.code_constructions."+ code_data["class_name"]
    )

    valid_params = inspect.signature(class_name.__init__).parameters.keys()
    filtered_params = {}

    for key, value in code_data.items():
        if key in valid_params:
            if isinstance(value, dict) and all(
                k in value for k in ["indices", "indptr", "shape"]
            ):
                filtered_params[key] = dict_to_binary_csr_matrix(
                    value
                )  # Convert only if it's a sparse matrix
            else:
                filtered_params[key] = value  # Keep as-is if not a matrix
        else:
            pass

    return class_name(**filtered_params)
