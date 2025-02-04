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
        raise FileNotFoundError(f"Error: No file found at the specified path: {file_path}")

    with open(file_path, "r") as file:
        code_data = json.load(file)

    if "class_name" not in code_data:
        raise ValueError("Error: The input JSON file must contain a 'class_name' key specifying the class to instantiate.")

    try:
        class_reference = eval("qec.code_constructions." + code_data["class_name"])
    except AttributeError:
        raise AttributeError(f"Error: The specified class '{code_data['class_name']}' does not exist in qec.code_constructions.")

    constructor_parameters = inspect.signature(class_reference.__init__).parameters.keys()
    filtered_input_parameters = {}

    for key, value in code_data.items():
        if key in constructor_parameters:
            if isinstance(value, dict) and all(k in value for k in ["indices", "indptr", "shape"]):
                filtered_input_parameters[key] = dict_to_binary_csr_matrix(value)  # Convert sparse matrix
            else:
                filtered_input_parameters[key] = value  # Keep as-is if not a matrix

    # Instantiate the class
    code_instance = class_reference(**filtered_input_parameters)

    # Add extra attributes from JSON that are valid class attributes but not constructor parameters
    class_attributes = dir(code_instance)
    print(class_attributes)
    for key, value in code_data.items():
        if key not in constructor_parameters and key in class_attributes:
            if isinstance(value, dict) and all(k in value for k in ["indices", "indptr", "shape"]):
                value = dict_to_binary_csr_matrix(value)  # Convert sparse matrix
            else:
                pass
         
            setattr(code_instance, key, value)
    
    return code_instance
