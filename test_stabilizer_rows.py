"""
Test that all rows of Hx and Hz are correctly identified as stabilizers.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from qec.code_constructions.surface_code import SurfaceCode

def test_stabilizer_rows(code, code_name):
    """Test that all rows of Hx and Hz are identified as stabilizers."""
    
    Hx = code.x_stabilizer_matrix.toarray() if hasattr(code.x_stabilizer_matrix, 'toarray') else code.x_stabilizer_matrix
    Hz = code.z_stabilizer_matrix.toarray() if hasattr(code.z_stabilizer_matrix, 'toarray') else code.z_stabilizer_matrix
    
    print(f"\n{'='*60}")
    print(f"Testing: {code_name}")
    print(f"{'='*60}")
    print(f"Hx shape: {Hx.shape}")
    print(f"Hz shape: {Hz.shape}")
    print(f"Number of qubits: {code.physical_qubit_count}")
    
    # Test all rows of Hx
    print(f"\nTesting {Hx.shape[0]} rows of Hx as e_x errors (X-stabilizers)...")
    hx_failures = []
    for i, row in enumerate(Hx):
        e_x = row.astype(int)  # Row of Hx is an X-stabilizer error pattern
        e_z = np.zeros(code.physical_qubit_count, dtype=int)
        
        # Check syndrome
        syndrome_x = (Hx @ e_z) % 2
        syndrome_z = (Hz @ e_x) % 2
        
        if np.any(syndrome_x) or np.any(syndrome_z):
            print(f"  Row {i}: ❌ NON-ZERO SYNDROME!")
            print(f"    syndrome_x: {syndrome_x}")
            print(f"    syndrome_z: {syndrome_z}")
            hx_failures.append(i)
            continue
        
        # Check if e_x is in row space of Hx
        # We need to solve Hx^T @ sol = e_x
        Hx_T = Hx.T
        
        # Use Gaussian elimination to solve
        # Augmented matrix [Hx^T | e_x]
        augmented = np.hstack([Hx_T, e_x.reshape(-1, 1)])
        m, n = Hx_T.shape  # m = num_qubits, n = num_x_checks
        
        # Row reduce
        aug_copy = augmented.copy()
        pivot_row = 0
        for col in range(n):
            # Find pivot
            found_pivot = False
            for row in range(pivot_row, m):
                if aug_copy[row, col] == 1:
                    # Swap rows
                    if row != pivot_row:
                        aug_copy[[pivot_row, row]] = aug_copy[[row, pivot_row]]
                    found_pivot = True
                    break
            
            if not found_pivot:
                continue
            
            # Eliminate
            for row in range(m):
                if row != pivot_row and aug_copy[row, col] == 1:
                    aug_copy[row] = (aug_copy[row] + aug_copy[pivot_row]) % 2
            
            pivot_row += 1
        
        # Check for inconsistency: row with [0 0 ... 0 | 1]
        has_solution = True
        for row in range(m):
            if np.all(aug_copy[row, :n] == 0) and aug_copy[row, n] == 1:
                has_solution = False
                break
        
        if not has_solution:
            print(f"  Row {i}: ❌ NOT IN ROW SPACE OF Hx")
            hx_failures.append(i)
        else:
            print(f"  Row {i}: ✓")
    
    # Test all rows of Hz
    print(f"\nTesting {Hz.shape[0]} rows of Hz as e_z errors (Z-stabilizers)...")
    hz_failures = []
    for i, row in enumerate(Hz):
        e_z = row.astype(int)  # Row of Hz is a Z-stabilizer error pattern
        e_x = np.zeros(code.physical_qubit_count, dtype=int)
        
        # Check syndrome
        syndrome_x = (Hx @ e_z) % 2
        syndrome_z = (Hz @ e_x) % 2
        
        if np.any(syndrome_x) or np.any(syndrome_z):
            print(f"  Row {i}: ❌ NON-ZERO SYNDROME!")
            print(f"    syndrome_x: {syndrome_x}")
            print(f"    syndrome_z: {syndrome_z}")
            hz_failures.append(i)
            continue
        
        # Check if e_z is in row space of Hz
        # We need to solve Hz^T @ sol = e_z
        Hz_T = Hz.T
        
        # Use Gaussian elimination to solve
        # Augmented matrix [Hz^T | e_z]
        augmented = np.hstack([Hz_T, e_z.reshape(-1, 1)])
        m, n = Hz_T.shape  # m = num_qubits, n = num_z_checks
        
        # Row reduce
        aug_copy = augmented.copy()
        pivot_row = 0
        for col in range(n):
            # Find pivot
            found_pivot = False
            for row in range(pivot_row, m):
                if aug_copy[row, col] == 1:
                    # Swap rows
                    if row != pivot_row:
                        aug_copy[[pivot_row, row]] = aug_copy[[row, pivot_row]]
                    found_pivot = True
                    break
            
            if not found_pivot:
                continue
            
            # Eliminate
            for row in range(m):
                if row != pivot_row and aug_copy[row, col] == 1:
                    aug_copy[row] = (aug_copy[row] + aug_copy[pivot_row]) % 2
            
            pivot_row += 1
        
        # Check for inconsistency: row with [0 0 ... 0 | 1]
        has_solution = True
        for row in range(m):
            if np.all(aug_copy[row, :n] == 0) and aug_copy[row, n] == 1:
                has_solution = False
                break
        
        if not has_solution:
            print(f"  Row {i}: ❌ NOT IN ROW SPACE OF Hz")
            hz_failures.append(i)
        else:
            print(f"  Row {i}: ✓")
    
    # Summary
    print(f"\n{'-'*60}")
    print(f"Summary for {code_name}:")
    print(f"  Hx rows: {Hx.shape[0] - len(hx_failures)}/{Hx.shape[0]} passed")
    print(f"  Hz rows: {Hz.shape[0] - len(hz_failures)}/{Hz.shape[0]} passed")
    
    if hx_failures:
        print(f"  ❌ Failed Hx rows: {hx_failures}")
    if hz_failures:
        print(f"  ❌ Failed Hz rows: {hz_failures}")
    
    if not hx_failures and not hz_failures:
        print(f"  ✅ ALL TESTS PASSED!")
    
    return len(hx_failures) == 0 and len(hz_failures) == 0


if __name__ == "__main__":
    # Test rotated surface codes
    print("\n" + "="*60)
    print("TESTING ROTATED SURFACE CODES")
    print("="*60)
    
    all_passed = True
    for d in [3, 5, 7]:
        code = RotatedSurfaceCode(d)
        passed = test_stabilizer_rows(code, f"Rotated Surface Code d={d}")
        all_passed = all_passed and passed
    
    # Test standard surface codes
    print("\n" + "="*60)
    print("TESTING STANDARD SURFACE CODES")
    print("="*60)
    
    for d in [3, 5, 7]:
        code = SurfaceCode(d)
        passed = test_stabilizer_rows(code, f"Surface Code d={d}")
        all_passed = all_passed and passed
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    if all_passed:
        print("✅ ALL CODES PASSED ALL TESTS!")
    else:
        print("❌ SOME TESTS FAILED!")
