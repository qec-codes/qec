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

    if code_data["class_name"] == "StabilizerCode":
        stabilizer_matrix = dict_to_binary_csr_matrix(code_data["stabilizer_matrix"])
        instance = qec.code_constructions.StabilizerCode(
            stabilizer_matrix, name=code_data["name"]
        )
        instance.code_distance = (
            code_data["parameters"]["code_distance"]
            if code_data["parameters"]["code_distance"] != "?"
            else None
        )

    elif code_data["class_name"] == "CSSCode":
        x_stabilizer_matrix = dict_to_binary_csr_matrix(code_data["x_stabilizer_matrix"])
        z_stabilizer_matrix = dict_to_binary_csr_matrix(code_data["z_stabilizer_matrix"])
        instance = qec.code_constructions.CSSCode(
            x_stabilizer_matrix, z_stabilizer_matrix, code_data["name"]
        )
        instance.x_code_distance = (
            code_data["parameters"]["x_code_distance"]
            if code_data["parameters"]["x_code_distance"] != "?"
            else None
        )
        instance.z_code_distance = (
            code_data["parameters"]["z_code_distance"]
            if code_data["parameters"]["z_code_distance"] != "?"
            else None
        )

    elif code_data["class_name"] == "HypergraphProductCode":
        seed_matrix_1 = dict_to_binary_csr_matrix(code_data["seed_matrix_1"])
        seed_matrix_2 = dict_to_binary_csr_matrix(code_data["seed_matrix_2"])
        instance = qec.code_constructions.HypergraphProductCode(
            seed_matrix_1, seed_matrix_2, code_data["name"]
        )
        instance.code_distance = (
            code_data["parameters"]["code_distance"]
            if code_data["parameters"]["code_distance"] != "?"
            else None
        )
        instance.x_code_distance = (
            code_data["parameters"]["x_code_distance"]
            if code_data["parameters"]["x_code_distance"] != "?"
            else None
        )
        instance.z_code_distance = (
            code_data["parameters"]["z_code_distance"]
            if code_data["parameters"]["z_code_distance"] != "?"
            else None
        )

    else:
        raise ValueError(
            f"The code class: {code_data['class_name']} is not recognised."
        )

    return instance


def load_code_from_dict(code_data: Dict[str, Any]):
    """
    Load a quantum code class from a dictionary and instantiate it dynamically.

    Parameters
    ----------
    code_data : dict
        Dictionary containing the saved quantum code, including the class name
        and parameters required for initialization.

    Returns
    -------
    object
        An instance of the dynamically loaded class with the parameters from the dictionary.

    Raises
    ------
    ValueError
        If the class name is not found in the dictionary.
    """

    if "class_name" not in code_data.keys():
        raise ValueError("Missing 'class_name' in input dictionary.")

    class_name = eval(
        "qec.code_constructions." + code_data["class_name"]
    )  # Dynamically retrieve class reference
    valid_params = inspect.signature(class_name.__init__).parameters.keys()
    filtered_params = {}

    # Convert any detected sparse matrices and filter valid parameters
    for key, value in code_data.items():
        if key in valid_params:
            if isinstance(value, dict) and all(
                k in value for k in ["data", "indices", "indptr", "shape"]
            ):
                filtered_params[key] = dict_to_binary_csr_matrix(
                    value
                )  # Convert only if it's a sparse matrix
            else:
                filtered_params[key] = value  # Keep as-is if not a matrix
        else:
            pass

    return class_name(**filtered_params)  # Instantiate the class
