"""
Verify the correct stabilizer logic for CSS codes.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode

# Create a small code
code = RotatedSurfaceCode(3)
Hx = code.x_stabilizer_matrix.toarray()
Hz = code.z_stabilizer_matrix.toarray()

print("Hx (X-stabilizer matrix) shape:", Hx.shape)
print("Hz (Z-stabilizer matrix) shape:", Hz.shape)
print()

# Row 0 of Hx represents an X-stabilizer: X on qubits where Hx[0,j]=1
print("Row 0 of Hx:", Hx[0])
print("This represents X-errors on qubits:", np.where(Hx[0] == 1)[0])
print()

# If we apply this as an X-ERROR pattern:
e_x = Hx[0]  # X-errors
e_z = np.zeros_like(Hx[0])  # No Z-errors

# Compute syndromes:
syndrome_x = (Hx @ e_z) % 2  # X-checks applied to Z-errors
syndrome_z = (Hz @ e_x) % 2  # Z-checks applied to X-errors

print("When applying row 0 of Hx as X-errors:")
print("  syndrome_x (Hx @ e_z):", syndrome_x)
print("  syndrome_z (Hz @ e_x):", syndrome_z)
print()

# Row 0 of Hz represents a Z-stabilizer: Z on qubits where Hz[0,j]=1
print("Row 0 of Hz:", Hz[0])
print("This represents Z-errors on qubits:", np.where(Hz[0] == 1)[0])
print()

# If we apply this as a Z-ERROR pattern:
e_z = Hz[0]  # Z-errors
e_x = np.zeros_like(Hz[0])  # No X-errors

# Compute syndromes:
syndrome_x = (Hx @ e_z) % 2  # X-checks applied to Z-errors
syndrome_z = (Hz @ e_x) % 2  # Z-checks applied to X-errors

print("When applying row 0 of Hz as Z-errors:")
print("  syndrome_x (Hx @ e_z):", syndrome_x)
print("  syndrome_z (Hz @ e_x):", syndrome_z)
print()

print("="*60)
print("CONCLUSION:")
print("="*60)
print("For a CSS code:")
print("  - Rows of Hx are X-STABILIZERS → apply as X-errors")
print("  - Rows of Hz are Z-STABILIZERS → apply as Z-errors")
print()
print("For error classification:")
print("  - e_x should be in row space of Hx (rows are X-stabilizers)")
print("  - e_z should be in row space of Hz (rows are Z-stabilizers)")
print()
print("PLU decomposition needed:")
print("  - PLU(Hx^T) to solve: Hx^T @ sol = e_x")
print("  - PLU(Hz^T) to solve: Hz^T @ sol = e_z")
