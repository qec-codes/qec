"""
Test syndrome calculation for the surface code.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from qec.utils.draw_css_code_d3 import prepare_css_code_visualization_data


def test_syndrome_calculation():
    """Test that syndrome calculation works correctly."""
    distance = 7
    code = RotatedSurfaceCode(distance)
    
    print(f"Distance {distance} rotated surface code:")
    print(f"  Number of qubits: {code.x_stabilizer_matrix.shape[1]}")
    print(f"  Number of X-checks: {code.x_stabilizer_matrix.shape[0]}")
    print(f"  Number of Z-checks: {code.z_stabilizer_matrix.shape[0]}")
    print()
    
    # Get matrices
    Hx = code.x_stabilizer_matrix.toarray()
    Hz = code.z_stabilizer_matrix.toarray()
    
    print(f"Hx shape: {Hx.shape}")
    print(f"Hz shape: {Hz.shape}")
    print()
    
    # Prepare visualization data
    data = prepare_css_code_visualization_data(
        code,
        qubit_radius=16,
        check_radius=16,
        spacing=80
    )
    
    # Create mappings
    qubit_id_to_index = {node['id']: i for i, node in enumerate(data['nodes']) if node['type'] == 'qubit'}
    x_check_id_to_index = {node['id']: i for i, node in enumerate(data['nodes']) if node['type'] == 'x_check'}
    z_check_id_to_index = {node['id']: i for i, node in enumerate(data['nodes']) if node['type'] == 'z_check'}
    
    print(f"Number of qubit nodes: {len(qubit_id_to_index)}")
    print(f"Number of X-check nodes: {len(x_check_id_to_index)}")
    print(f"Number of Z-check nodes: {len(z_check_id_to_index)}")
    print()
    
    # Print first few mappings
    print("First 5 qubit IDs and their indices:")
    for i, (qid, idx) in enumerate(qubit_id_to_index.items()):
        if i < 5:
            print(f"  {qid} -> {idx}")
    print()
    
    print("First 5 X-check IDs and their indices:")
    for i, (cid, idx) in enumerate(x_check_id_to_index.items()):
        if i < 5:
            print(f"  {cid} -> {idx}")
    print()
    
    print("First 5 Z-check IDs and their indices:")
    for i, (cid, idx) in enumerate(z_check_id_to_index.items()):
        if i < 5:
            print(f"  {cid} -> {idx}")
    print()
    
    # Test: inject X error on qubit 0
    print("TEST 1: Inject X error on qubit index 0")
    num_qubits = code.x_stabilizer_matrix.shape[1]
    x_error = np.zeros(num_qubits, dtype=int)
    x_error[0] = 1
    
    # Calculate Z syndrome (Hz detects X errors)
    z_syndrome = (Hz @ x_error) % 2
    
    print(f"X error vector: {x_error}")
    print(f"Z syndrome: {z_syndrome}")
    print(f"Triggered Z-checks (indices): {np.where(z_syndrome == 1)[0].tolist()}")
    print()
    
    # Show which Z-checks are connected to qubit 0
    print("Hz matrix column 0 (which Z-checks see qubit 0):")
    print(f"  {Hz[:, 0]}")
    print(f"  Non-zero rows (Z-checks connected to qubit 0): {np.where(Hz[:, 0] == 1)[0].tolist()}")
    print()
    
    # Test: inject Z error on qubit 0
    print("TEST 2: Inject Z error on qubit index 0")
    z_error = np.zeros(num_qubits, dtype=int)
    z_error[0] = 1
    
    # Calculate X syndrome (Hx detects Z errors)
    x_syndrome = (Hx @ z_error) % 2
    
    print(f"Z error vector: {z_error}")
    print(f"X syndrome: {x_syndrome}")
    print(f"Triggered X-checks (indices): {np.where(x_syndrome == 1)[0].tolist()}")
    print()
    
    # Show which X-checks are connected to qubit 0
    print("Hx matrix column 0 (which X-checks see qubit 0):")
    print(f"  {Hx[:, 0]}")
    print(f"  Non-zero rows (X-checks connected to qubit 0): {np.where(Hx[:, 0] == 1)[0].tolist()}")
    print()
    
    # Check the edges to see connectivity
    print("Checking edges connected to first qubit:")
    first_qubit_id = list(qubit_id_to_index.keys())[0]
    print(f"  First qubit ID: {first_qubit_id}")
    
    connected_edges = [e for e in data['edges'] if e['source'] == first_qubit_id or e['target'] == first_qubit_id]
    print(f"  Number of connected edges: {len(connected_edges)}")
    for edge in connected_edges[:10]:
        print(f"    {edge['source']} <-> {edge['target']} (type: {edge['type']})")


if __name__ == "__main__":
    test_syndrome_calculation()
