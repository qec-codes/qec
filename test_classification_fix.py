"""
Test that S^T PLU decomposition correctly identifies stabilizers vs logicals
"""

import numpy as np
from src.qec.code_constructions.surface_code import SurfaceCode
from ldpc.mod2 import PluDecomposition

def plu_solve_gf2(plu, y):
    """Solve S^T @ x = y using PLU decomposition (GF2)"""
    rank = plu.rank
    num_rows = plu.P.shape[0]
    num_cols = plu.U.shape[1]
    
    # Apply permutation
    Py = (plu.P @ y) % 2
    
    # Forward substitution: Lb = Py
    b = np.zeros(rank, dtype=int)
    for i in range(rank):
        sum_val = 0
        for j in range(i):
            sum_val ^= (plu.L[i, j] * b[j])
        b[i] = Py[i] ^ sum_val
    
    # Check if y in image
    for i in range(rank, num_rows):
        sum_val = 0
        for j in range(rank):
            sum_val ^= (plu.L[i, j] * b[j])
        if (Py[i] ^ sum_val) != 0:
            return None  # Not in image
    
    # Backward substitution: Ux = b
    x = np.zeros(num_cols, dtype=int)
    pivot_cols = plu.pivots
    for i in range(rank - 1, -1, -1):
        sum_val = 0
        for j in range(pivot_cols[i] + 1, num_cols):
            sum_val ^= (plu.U[i, j] * x[j])
        x[pivot_cols[i]] = b[i] ^ sum_val
    
    return x


code = SurfaceCode(3)
S = code.stabilizer_matrix.toarray()
S_T = S.T

print("Surface Code d=3")
print(f"  S shape: {S.shape}")  # (12, 26)
print(f"  S^T shape: {S_T.shape}")  # (26, 12)
print()

# Compute PLU on S^T
plu = PluDecomposition(S_T)
print(f"PLU of S^T:")
print(f"  Rank: {plu.rank}")
print(f"  Can solve for errors in row space of S (column space of S^T)")
print()

# Test 1: First stabilizer (first row of S)
print("=== Test 1: First Stabilizer (S[0,:]) ===")
error_1 = S[0, :]
print(f"Error pattern: {error_1[:10]}... (first 10 elements)")

# Check syndrome
syndrome_1 = (S @ error_1) % 2
print(f"Syndrome: {syndrome_1[:6]}... (first 6 elements)")
print(f"Syndrome is zero: {np.all(syndrome_1 == 0)}")

if np.all(syndrome_1 == 0):
    # Solve S^T @ coeffs = error
    coeffs = plu_solve_gf2(plu, error_1)
    
    if coeffs is not None:
        print(f"✓ PLU found solution")
        print(f"  Coefficients: {coeffs}")
        print(f"  Stabilizers used: {np.where(coeffs == 1)[0].tolist()}")
        
        # Reconstruct: sum rows of S where coeffs[i] = 1
        reconstructed = np.zeros(26, dtype=int)
        for i in range(len(coeffs)):
            if coeffs[i] == 1:
                reconstructed = (reconstructed + S[i, :]) % 2
        
        matches = np.all(reconstructed == error_1)
        print(f"  Reconstruction matches: {matches}")
        
        if matches:
            print(f"  ✅ CLASSIFICATION: STABILIZER")
        else:
            print(f"  ❌ CLASSIFICATION: LOGICAL (reconstruction failed)")
    else:
        print(f"✗ PLU returned None - error not in row space")
        print(f"  ❌ CLASSIFICATION: LOGICAL")
else:
    print(f"  Syndrome non-zero - not a stabilizer or logical")
print()

# Test 2: XOR of two stabilizers
print("=== Test 2: XOR of S[0,:] and S[1,:] ===")
error_2 = (S[0, :] + S[1, :]) % 2
print(f"Error pattern: {error_2[:10]}... (first 10 elements)")

syndrome_2 = (S @ error_2) % 2
print(f"Syndrome: {syndrome_2[:6]}... (first 6 elements)")
print(f"Syndrome is zero: {np.all(syndrome_2 == 0)}")

if np.all(syndrome_2 == 0):
    coeffs = plu_solve_gf2(plu, error_2)
    
    if coeffs is not None:
        print(f"✓ PLU found solution")
        print(f"  Coefficients: {coeffs}")
        print(f"  Stabilizers used: {np.where(coeffs == 1)[0].tolist()}")
        
        reconstructed = np.zeros(26, dtype=int)
        for i in range(len(coeffs)):
            if coeffs[i] == 1:
                reconstructed = (reconstructed + S[i, :]) % 2
        
        matches = np.all(reconstructed == error_2)
        print(f"  Reconstruction matches: {matches}")
        
        if matches:
            print(f"  ✅ CLASSIFICATION: STABILIZER")
        else:
            print(f"  ❌ CLASSIFICATION: LOGICAL (reconstruction failed)")
    else:
        print(f"✗ PLU returned None")
        print(f"  ❌ CLASSIFICATION: LOGICAL")
else:
    print(f"  Syndrome non-zero")
print()

# Test 3: Logical operator (in kernel of S, not in row space of S)
print("=== Test 3: Logical Operator ===")
# For surface code, logical X is X on all qubits in a horizontal line
# Let's use the code's logical basis
x_logicals = code.x_logical_operator_basis
if len(x_logicals) > 0:
    logical_op = x_logicals[0].toarray()[0]  # First logical X
    # Convert to error vector format [X_errors; Z_errors]
    error_3 = np.concatenate([logical_op, np.zeros(code.physical_qubit_count, dtype=int)])
    
    print(f"Logical X operator: {logical_op}")
    print(f"Error pattern: {error_3[:10]}... (first 10 elements)")
    
    syndrome_3 = (S @ error_3) % 2
    print(f"Syndrome: {syndrome_3[:6]}... (first 6 elements)")
    print(f"Syndrome is zero: {np.all(syndrome_3 == 0)}")
    
    if np.all(syndrome_3 == 0):
        coeffs = plu_solve_gf2(plu, error_3)
        
        if coeffs is not None:
            print(f"✓ PLU found solution")
            print(f"  Coefficients: {coeffs}")
            
            reconstructed = np.zeros(26, dtype=int)
            for i in range(len(coeffs)):
                if coeffs[i] == 1:
                    reconstructed = (reconstructed + S[i, :]) % 2
            
            matches = np.all(reconstructed == error_3)
            print(f"  Reconstruction matches: {matches}")
            
            if matches:
                print(f"  ❌ UNEXPECTED: Logical classified as STABILIZER")
            else:
                print(f"  ✅ CLASSIFICATION: LOGICAL (reconstruction failed)")
        else:
            print(f"✓ PLU returned None - error not in row space")
            print(f"  ✅ CLASSIFICATION: LOGICAL")
    else:
        print(f"  Syndrome non-zero - this logical has detectable errors")
else:
    print("No logical operators found in code basis")
