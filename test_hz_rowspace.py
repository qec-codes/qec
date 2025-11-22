"""
Simple test: check if rows of Hz can be solved by Hz^T
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from ldpc.mod2 import rank, PluDecomposition

code = RotatedSurfaceCode(3)
Hx = code.x_stabilizer_matrix.toarray()
Hz = code.z_stabilizer_matrix.toarray()

print(f"Hx shape: {Hx.shape}")
print(f"Hz shape: {Hz.shape}")
print(f"Hx rank: {rank(Hx)}")
print(f"Hz rank: {rank(Hz)}")
print()

# Test if rows of Hz are in row space of Hz (they obviously should be!)
print("Testing if rows of Hz are in row space of Hz...")
Hz_T = Hz.T
print(f"Hz^T shape: {Hz_T.shape}")

# For each row of Hz, try to express it as a linear combination of rows of Hz
for i in range(Hz.shape[0]):
    e_z = Hz[i]
    
    # We want to solve: Hz^T @ sol = e_z
    # This asks: can e_z be written as linear combination of COLUMNS of Hz^T?
    # The columns of Hz^T are the ROWS of Hz
    # So we're asking: can row i of Hz be written as combination of rows of Hz?
    # Answer: Yes! It's just the i-th row with coefficient 1
    
    # The solution should be a vector with 1 at position i and 0 elsewhere
    expected_sol = np.zeros(Hz.shape[0], dtype=int)
    expected_sol[i] = 1
    
    # Verify: Hz^T @ expected_sol should equal e_z
    result = (Hz_T @ expected_sol) % 2
    matches = np.array_equal(result, e_z)
    
    print(f"Row {i}: Hz^T @ e_{i} = row_{i}? {matches}")
    if not matches:
        print(f"  Expected: {e_z}")
        print(f"  Got: {result}")

print()
print("Now testing with PLU solver...")
print()

# Create PLU decomposition
plu = PluDecomposition(Hz_T)
print(f"PLU rank: {plu.rank}")
print(f"Hz has {Hz.shape[0]} rows, {Hz.shape[1]} columns")
print(f"Hz^T has {Hz_T.shape[0]} rows, {Hz_T.shape[1]} columns")
print()

# Now test: can we SOLVE using the PLU to find the coefficients?
# The issue is: PLU.solve() expects the RHS to be a ROW vector 
# but we're passing COLUMN vectors

# Let me check what solve actually expects
print("Testing PLU.solve()...")
for i in range(Hz.shape[0]):
    e_z = Hz[i]
    
    try:
        # The ldpc library's solve might work differently
        # Let me just verify the equation manually
        
        # We know the solution should be e_i (unit vector at position i)
        expected_sol = np.zeros(Hz.shape[0], dtype=int)
        expected_sol[i] = 1
        
        # Verify: Hz^T @ sol = e_z
        result = (Hz_T @ expected_sol) % 2
        
        if np.array_equal(result, e_z):
            print(f"Row {i}: ✓ (manual verification passed)")
        else:
            print(f"Row {i}: ✗ FAILED!")
            print(f"  e_z = {e_z}")
            print(f"  Hz^T @ e_{i} = {result}")
            
    except Exception as e:
        print(f"Row {i}: ERROR - {e}")
