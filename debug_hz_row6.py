"""
Check what row 6 of Hz looks like for Surface Code d=5
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from qec.code_constructions.surface_code import SurfaceCode

code = SurfaceCode(5)
Hz = code.z_stabilizer_matrix.toarray()

print(f"Hz shape: {Hz.shape}")
print(f"Row 6 of Hz: {Hz[6]}")
print(f"Non-zero positions: {np.where(Hz[6] == 1)[0].tolist()}")
print(f"Number of qubits: {code.physical_qubit_count}")
