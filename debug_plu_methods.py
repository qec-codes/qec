"""
Test correct PLU usage for checking row space membership
"""

import sys
from pathlib import Path
import numpy as np
from ldpc.mod2 import PluDecomposition

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode

code = RotatedSurfaceCode(5)
Hz = code.z_stabilizer_matrix.toarray()

print("Goal: Check if e_z (row 6 of Hz) is in the row space of Hz")
print(f"Hz shape: {Hz.shape}")
print()

e_z = Hz[6]
print(f"e_z = row 6 of Hz: {e_z}")
print()

# Method 1: PLU of Hz^T, solve Hz^T @ sol = e_z
print("="*60)
print("Method 1: PLU(Hz^T), solve Hz^T @ sol = e_z")
print("="*60)
Hz_T = Hz.T
plu_ht = PluDecomposition(Hz_T)
print(f"Hz^T shape: {Hz_T.shape}, rank: {plu_ht.rank}")

# Try to solve using the standard approach
P_dense = plu_ht.P.toarray()
P_e_z = (P_dense @ e_z) % 2
print(f"P @ e_z: {P_e_z}")
print(f"Non-zero after rank? {np.any(P_e_z[plu_ht.rank:])}")
print()

# Method 2: PLU of Hz, solve e_z @ Hz^T = sol (or Hz @ sol^T = e_z^T)
print("="*60)
print("Method 2: PLU(Hz), check if e_z in row space directly")
print("="*60)
plu_h = PluDecomposition(Hz)
print(f"Hz shape: {Hz.shape}, rank: {plu_h.rank}")

# For row space: e_z should be a linear combination of rows
# After permutation, rows of P @ Hz form the basis
# Check: can (P^T @ e_z) be expressed using first 'rank' rows?

P_h_dense = plu_h.P.toarray()
PT_e_z = (P_h_dense.T @ e_z) % 2  # P^T @ e_z
print(f"P^T @ e_z: {PT_e_z[:20]}...")
print(f"Non-zero after rank? {np.any(PT_e_z[plu_h.rank:])}")
print()

# Method 3: Use row-reduced echelon form thinking
print("="*60)
print("Method 3: Manual row space check using Gaussian elimination")
print("="*60)

# To check if e_z is in row space of Hz:
# Form augmented matrix [Hz^T | e_z] and row reduce
# If consistent, e_z is in row space

augmented = np.hstack([Hz.T, e_z.reshape(-1, 1)])
print(f"Augmented [Hz^T | e_z] shape: {augmented.shape}")

# Row reduce
aug_copy = augmented.copy()
m, n = Hz.T.shape
pivot_row = 0

for col in range(n):
    # Find pivot
    found = False
    for row in range(pivot_row, m):
        if aug_copy[row, col] == 1:
            if row != pivot_row:
                aug_copy[[pivot_row, row]] = aug_copy[[row, pivot_row]]
            found = True
            break
    
    if not found:
        continue
    
    # Eliminate
    for row in range(m):
        if row != pivot_row and aug_copy[row, col] == 1:
            aug_copy[row] = (aug_copy[row] + aug_copy[pivot_row]) % 2
    
    pivot_row += 1

# Check for inconsistency
has_solution = True
for row in range(m):
    if np.all(aug_copy[row, :n] == 0) and aug_copy[row, n] == 1:
        has_solution = False
        print(f"INCONSISTENT at row {row}")
        break

print(f"Has solution (in row space)? {has_solution}")
