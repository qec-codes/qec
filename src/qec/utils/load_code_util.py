import json
from typing import Union
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
    object
        An instance of the quantum error correction code class.

    Raises
    ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the input file is missing the 'class_name' key.
        AttributeError
            If the specified class does not exist in qec.code_constructions.
    """

    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Error: No file found at the specified path: {file_path}"
        )

    with open(file_path, "r") as file:
        code_data = json.load(file)

    if "class_name" not in code_data:
        raise ValueError(
            "Error: The input JSON file must contain a 'class_name' key specifying the class to instantiate."
        )

    try:
        class_reference = eval(
            f"qec.code_constructions.{code_data['class_name']}", {"qec": qec}
        )
    except AttributeError:
        raise AttributeError(
            f"Error: The specified class '{code_data['class_name']}' does not exist in qec.code_constructions."
        )

    constructor_parameters = inspect.signature(
        class_reference.__init__
    ).parameters.keys()
    filtered_input_parameters = {}

    for key, value in code_data.items():
        if key in constructor_parameters:
            if isinstance(value, dict) and all(
                k in value for k in ["indices", "indptr", "shape"]
            ):
                filtered_input_parameters[key] = dict_to_binary_csr_matrix(
                    value
                )  # Convert sparse matrix
            else:
                filtered_input_parameters[key] = value  # Keep as-is if not a matrix

    # Instantiate the class
    code_instance = class_reference(**filtered_input_parameters)

    # Add extra attributes from JSON that are valid class attributes but not constructor parameters
    class_attributes = dir(code_instance)
    for key, value in code_data.items():
        if key not in constructor_parameters and key in class_attributes:
            if isinstance(value, dict) and all(
                k in value for k in ["indices", "indptr", "shape"]
            ):
                value = dict_to_binary_csr_matrix(value)  # Convert sparse matrix
            else:
                pass

            if (not isinstance(value, str) or value != "?") and value is not None:
                setattr(code_instance, key, value)

    return code_instance


def load_code_from_id(code_id: int):
    """
    Load a quantum error correction code from a JSON file based on its ID from the package data.

    The code files are packaged as data in the directory:
        qec/code_instances/saved_codes
    and are named as f"{code_id}.json".

    Parameters
    ----------
    code_id : int
        The identifier of the saved code.

    Returns
    -------
    object
        An instance of the quantum error correction code class loaded from the JSON data.

    Raises
    ------
    FileNotFoundError
        If the JSON file corresponding to code_id is not found in the package data.
    """
    import importlib.resources as pkg_resources

    filename = f"{code_id}.json"
    package = "qec.code_instances.saved_codes"
    if not pkg_resources.is_resource(package, filename):
        raise FileNotFoundError(
            f"File '{filename}' does not exist in the package data at '{package}'."
        )
    with pkg_resources.path(package, filename) as resource_path:
        if not resource_path.exists():
            raise FileNotFoundError(f"File '{resource_path}' does not exist on disk.")
        return load_code(resource_path)
