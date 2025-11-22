"""
Test Z stabilizers specifically to see what's happening.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from ldpc.mod2 import PluDecomposition

# Create a small code
code = RotatedSurfaceCode(3)
Hx = code.x_stabilizer_matrix.toarray()
Hz = code.z_stabilizer_matrix.toarray()

print("Code: Rotated Surface Code d=3")
print(f"Hx shape: {Hx.shape}")
print(f"Hz shape: {Hz.shape}")
print()

# Test row 0 of Hz as a Z-stabilizer
print("="*60)
print("Testing row 0 of Hz as a Z-stabilizer")
print("="*60)

e_z = Hz[0]  # Row of Hz is a Z-error pattern
e_x = np.zeros_like(Hz[0])

print(f"e_z: {e_z}")
print(f"e_x: {e_x}")
print()

# Check syndrome
syndrome_x = (Hx @ e_z) % 2
syndrome_z = (Hz @ e_x) % 2

print(f"syndrome_x (Hx @ e_z): {syndrome_x}")
print(f"syndrome_z (Hz @ e_x): {syndrome_z}")
print(f"Syndrome is zero: {not np.any(syndrome_x) and not np.any(syndrome_z)}")
print()

# Now check if e_z is in row space of Hz using PLU
Hz_T = Hz.T
print(f"Hz shape: {Hz.shape}")
print(f"Hz^T shape: {Hz_T.shape}")

plu_hz = PluDecomposition(Hz_T)
print(f"PLU rank: {plu_hz.rank}")
print(f"PLU L shape: {plu_hz.L.shape}")
print(f"PLU U shape: {plu_hz.U.shape}")
print()

# Try to solve Hz^T @ sol = e_z
print("Solving Hz^T @ sol = e_z...")

# Forward substitution: L @ y = P @ e_z
P_array = np.array(plu_hz.P, dtype=int)
P_e_z = e_z[P_array]
print(f"P @ e_z: {P_e_z}")

# Since we're solving for e_z which is a ROW of Hz,
# and Hz^T has Hz's rows as its columns,
# e_z should definitely be in the column space of Hz^T (= row space of Hz)

# Let's manually verify: Hz^T has shape (num_qubits, num_z_checks)
# Column i of Hz^T is row i of Hz
# So e_z = row 0 of Hz should equal column 0 of Hz^T
print(f"\nRow 0 of Hz: {Hz[0]}")
print(f"Column 0 of Hz^T: {Hz_T[:, 0]}")
print(f"Are they equal? {np.array_equal(Hz[0], Hz_T[:, 0])}")
print()

# So Hz^T @ [1, 0, 0, 0] should give us e_z
test_sol = np.zeros(Hz.shape[0], dtype=int)
test_sol[0] = 1
result = (Hz_T @ test_sol) % 2
print(f"Hz^T @ [1,0,0,0]: {result}")
print(f"e_z: {e_z}")
print(f"Match? {np.array_equal(result, e_z)}")
print()

# Now let's see what the PLU solver does
print("Checking PLU matrices...")
print(f"L matrix:\n{plu_hz.L.toarray()}")
print(f"\nU matrix:\n{plu_hz.U.toarray()}")
print(f"\nP permutation: {plu_hz.P}")
