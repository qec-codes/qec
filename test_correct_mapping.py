"""
Complete test: Rows of Hx as e_x, Rows of Hz as e_z
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode

code = RotatedSurfaceCode(3)
Hx = code.x_stabilizer_matrix.toarray()
Hz = code.z_stabilizer_matrix.toarray()

print(f"Hx shape: {Hx.shape}")
print(f"Hz shape: {Hz.shape}")
print()

# Test 1: Each row of Hx as e_x (X-stabilizers)
print("="*60)
print("Test 1: Rows of Hx as e_x (X-stabilizers)")
print("="*60)
all_pass = True
for i in range(Hx.shape[0]):
    e_x = Hx[i]
    e_z = np.zeros_like(Hx[i])
    
    syndrome_x = (Hx @ e_z) % 2
    syndrome_z = (Hz @ e_x) % 2
    
    syndrome_zero = not np.any(syndrome_x) and not np.any(syndrome_z)
    status = "✓" if syndrome_zero else "✗"
    print(f"Row {i}: {status}")
    if not syndrome_zero:
        all_pass = False

print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print()

# Test 2: Each row of Hz as e_z (Z-stabilizers)
print("="*60)
print("Test 2: Rows of Hz as e_z (Z-stabilizers)")
print("="*60)
all_pass = True
for i in range(Hz.shape[0]):
    e_x = np.zeros_like(Hz[i])
    e_z = Hz[i]
    
    syndrome_x = (Hx @ e_z) % 2
    syndrome_z = (Hz @ e_x) % 2
    
    syndrome_zero = not np.any(syndrome_x) and not np.any(syndrome_z)
    status = "✓" if syndrome_zero else "✗"
    print(f"Row {i}: {status}")
    if not syndrome_zero:
        all_pass = False

print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
