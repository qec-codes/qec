"""
Check row 6 of Hz for Rotated Surface Code d=5
"""

import sys
from pathlib import Path
import numpy as np
from ldpc.mod2 import rank, PluDecomposition

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode

code = RotatedSurfaceCode(5)
Hx = code.x_stabilizer_matrix.toarray()
Hz = code.z_stabilizer_matrix.toarray()

print(f"Code: Rotated Surface Code d=5")
print(f"Hx shape: {Hx.shape}")
print(f"Hz shape: {Hz.shape}")
print(f"Number of qubits: {code.physical_qubit_count}")
print(f"Hz rank: {rank(Hz)}")
print()

print(f"Row 6 of Hz: {Hz[6]}")
print(f"Non-zero positions in row 6: {np.where(Hz[6] == 1)[0].tolist()}")
print()

# Now check the PLU
Hz_T = Hz.T
print(f"Hz^T shape: {Hz_T.shape}")
plu_hz = PluDecomposition(Hz_T)
print(f"PLU rank: {plu_hz.rank}")
print()

# Check if row 6 of Hz (which is column 6 of Hz^T) is in the column space
e_z = Hz[6]
print(f"Testing if row 6 of Hz is in column space of Hz^T...")
print(f"e_z = {e_z}")

# Manual check: Hz^T @ e_6 should equal e_z
e_6 = np.zeros(Hz.shape[0], dtype=int)
e_6[6] = 1
result = (Hz_T @ e_6) % 2
print(f"Hz^T @ e_6 = {result}")
print(f"Matches e_z? {np.array_equal(result, e_z)}")
print()

# Now let's manually check the PLU solver logic
print("Checking PLU decomposition...")
print(f"L shape: {plu_hz.L.shape}")
print(f"U shape: {plu_hz.U.shape}")
print(f"P shape: {plu_hz.P.shape}")
print()

# Apply permutation
P_dense = plu_hz.P.toarray()
P_e_z = (P_dense @ e_z) % 2
print(f"After permutation P @ e_z:")
print(f"First 13 elements: {P_e_z[:13]}")
print(f"Remaining elements: {P_e_z[13:]}")
print(f"Any non-zero after rank? {np.any(P_e_z[plu_hz.rank:])}")

if np.any(P_e_z[plu_hz.rank:]):
    print(f"\nNon-zero positions after rank {plu_hz.rank}:")
    nz_idx = np.where(P_e_z[plu_hz.rank:] == 1)[0]
    print(f"  Positions (relative to rank): {nz_idx.tolist()}")
    print(f"  Absolute positions: {(nz_idx + plu_hz.rank).tolist()}")
