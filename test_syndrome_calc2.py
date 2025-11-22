"""
Test the ID to index mapping for syndrome calculation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from qec.utils.draw_css_code_d3 import prepare_css_code_visualization_data


def test_mapping():
    """Test that the mapping between IDs and indices is correct."""
    distance = 7
    code = RotatedSurfaceCode(distance)
    
    # Get matrices
    Hx = code.x_stabilizer_matrix.toarray()
    Hz = code.z_stabilizer_matrix.toarray()
    
    # Prepare visualization data
    data = prepare_css_code_visualization_data(
        code,
        qubit_radius=16,
        check_radius=16,
        spacing=80
    )
    
    # The PROBLEM: We're creating the mapping based on enumeration of data['nodes']
    # But data['nodes'] includes both qubits AND checks!
    # So the indices are 0, 1, 2, ... for ALL nodes, not just qubits
    
    print("CURRENT (WRONG) MAPPING:")
    print("=" * 60)
    qubit_id_to_index_wrong = {node['id']: i for i, node in enumerate(data['nodes']) if node['type'] == 'qubit'}
    x_check_id_to_index_wrong = {node['id']: i for i, node in enumerate(data['nodes']) if node['type'] == 'x_check'}
    z_check_id_to_index_wrong = {node['id']: i for i, node in enumerate(data['nodes']) if node['type'] == 'z_check'}
    
    print(f"Qubit mapping (first 5): {dict(list(qubit_id_to_index_wrong.items())[:5])}")
    print(f"X-check mapping (first 5): {dict(list(x_check_id_to_index_wrong.items())[:5])}")
    print(f"Z-check mapping (first 5): {dict(list(z_check_id_to_index_wrong.items())[:5])}")
    print()
    
    print("CORRECT MAPPING:")
    print("=" * 60)
    # We need to map to MATRIX indices, not node list indices
    # The matrix columns/rows are indexed 0, 1, 2, ... for qubits/checks separately
    
    qubit_nodes = [node for node in data['nodes'] if node['type'] == 'qubit']
    x_check_nodes = [node for node in data['nodes'] if node['type'] == 'x_check']
    z_check_nodes = [node for node in data['nodes'] if node['type'] == 'z_check']
    
    # Map IDs to matrix indices (0-indexed within each type)
    qubit_id_to_index_correct = {node['id']: i for i, node in enumerate(qubit_nodes)}
    x_check_id_to_index_correct = {node['id']: i for i, node in enumerate(x_check_nodes)}
    z_check_id_to_index_correct = {node['id']: i for i, node in enumerate(z_check_nodes)}
    
    print(f"Qubit mapping (first 5): {dict(list(qubit_id_to_index_correct.items())[:5])}")
    print(f"X-check mapping (first 5): {dict(list(x_check_id_to_index_correct.items())[:5])}")
    print(f"Z-check mapping (first 5): {dict(list(z_check_id_to_index_correct.items())[:5])}")
    print()
    
    # But wait, we need to check if the node IDs correspond to the matrix order
    # Let's check if q0 -> index 0, q1 -> index 1, etc.
    print("Checking if node IDs are ordered:")
    print(f"First 10 qubit nodes: {[node['id'] for node in qubit_nodes[:10]]}")
    print(f"First 10 X-check nodes: {[node['id'] for node in x_check_nodes[:10]]}")
    print(f"First 10 Z-check nodes: {[node['id'] for node in z_check_nodes[:10]]}")
    print()
    
    # Test syndrome with correct mapping
    print("TEST: Inject X error on q0 and check Z syndrome")
    x_error = np.zeros(49, dtype=int)
    x_error[qubit_id_to_index_correct['q0']] = 1
    
    z_syndrome = (Hz @ x_error) % 2
    triggered_z_checks = np.where(z_syndrome == 1)[0].tolist()
    print(f"X error on q0 (index {qubit_id_to_index_correct['q0']})")
    print(f"Triggered Z-check indices: {triggered_z_checks}")
    
    # Find which Z-check nodes correspond to these indices
    z_check_ids = [z_check_nodes[idx]['id'] for idx in triggered_z_checks]
    print(f"Triggered Z-check node IDs: {z_check_ids}")


if __name__ == "__main__":
    test_mapping()
