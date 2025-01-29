import json 
from typing import Union
from pathlib import Path
from qec.utils.sparse_binary_utils import dict_to_csr_matrix

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
    from qec.code_constructions import StabilizerCode, CSSCode, HypergraphProductCode
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    with open(filepath, 'r') as f:
        code_data = json.load(f)

    if code_data['class_name'] == 'StabilizerCode':
        stabilizer_matrix = dict_to_csr_matrix(code_data['stabilizer_matrix'])
        instance = StabilizerCode(stabilizer_matrix, name = code_data['name'])
        instance.code_distance = code_data['parameters']['d'] if code_data['parameters']['d'] != '?' else None

    elif code_data['class_name'] == 'CSSCode':
        x_stabilizer_matrix = dict_to_csr_matrix(code_data['x_stabilizer_matrix'])
        z_stabilizer_matrix = dict_to_csr_matrix(code_data['z_stabilizer_matrix'])
        instance = CSSCode(x_stabilizer_matrix, z_stabilizer_matrix, code_data['name'])
        instance.x_code_distance = code_data['parameters']['dx'] if code_data['parameters']['dx'] != '?' else None
        instance.z_code_distance = code_data['parameters']['dz'] if code_data['parameters']['dz'] != '?' else None

    elif code_data['class_name'] == 'HypergraphProductCode':
        seed_matrix_1 = dict_to_csr_matrix(code_data['seed_matrix_1'])
        seed_matrix_2 = dict_to_csr_matrix(code_data['seed_matrix_2'])
        instance = HypergraphProductCode(seed_matrix_1, seed_matrix_2, code_data['name'])
        instance.x_code_distance = code_data['parameters']['dx'] if code_data['parameters']['dx'] != '?' else None
        instance.z_code_distance = code_data['parameters']['dz'] if code_data['parameters']['dz'] != '?' else None

    else:
        raise ValueError(f"The code class: {code_data['class_name']} is not recognised.")

    return instance
