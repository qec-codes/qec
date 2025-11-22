"""
Verify the correct syndrome equation and test both interpretations
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode

code = RotatedSurfaceCode(3)
Hx = code.x_stabilizer_matrix.toarray()
Hz = code.z_stabilizer_matrix.toarray()

print("Testing BOTH interpretations:")
print(f"Hx shape: {Hx.shape}")
print(f"Hz shape: {Hz.shape}")
print()

# INTERPRETATION 1: What I currently have
print("="*60)
print("INTERPRETATION 1: Rows of Hx are X-stabilizers (X errors)")
print("="*60)
e_x = Hx[0]  # X errors
e_z = np.zeros_like(Hx[0])  # No Z errors

syndrome_x = (Hx @ e_z) % 2
syndrome_z = (Hz @ e_x) % 2

print(f"Testing row 0 of Hx as e_x:")
print(f"  syndrome_x (Hx @ e_z): {syndrome_x}")
print(f"  syndrome_z (Hz @ e_x): {syndrome_z}")
print(f"  Total syndrome zero? {not np.any(syndrome_x) and not np.any(syndrome_z)}")
print()

# INTERPRETATION 2: What you're suggesting
print("="*60)
print("INTERPRETATION 2: Rows of Hx should be tested as e_z")
print("="*60)
e_z = Hx[0]  # Z errors (???)
e_x = np.zeros_like(Hx[0])  # No X errors

syndrome_x = (Hx @ e_z) % 2
syndrome_z = (Hz @ e_x) % 2

print(f"Testing row 0 of Hx as e_z:")
print(f"  syndrome_x (Hx @ e_z): {syndrome_x}")
print(f"  syndrome_z (Hz @ e_x): {syndrome_z}")
print(f"  Total syndrome zero? {not np.any(syndrome_x) and not np.any(syndrome_z)}")
print()

# Check which one gives zero syndrome
print("="*60)
print("CONCLUSION:")
print("="*60)
print("Interpretation 1 (Hx rows as e_x): syndrome is", "ZERO ✓" if not np.any(syndrome_x) and not np.any(syndrome_z) else "NON-ZERO ✗")

e_z = Hx[0]
e_x = np.zeros_like(Hx[0])
syndrome_x_v2 = (Hx @ e_z) % 2
syndrome_z_v2 = (Hz @ e_x) % 2
print("Interpretation 2 (Hx rows as e_z): syndrome is", "ZERO ✓" if not np.any(syndrome_x_v2) and not np.any(syndrome_z_v2) else "NON-ZERO ✗")
