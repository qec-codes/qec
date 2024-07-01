from ldpc.mod2 import rank, kernel
from ldpc.code_util import compute_code_parameters
import numpy as np

matrix = np.array([
    [1, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1]
])

H = kernel(matrix).toarray()

print(repr(H))

print(compute_code_parameters(H))




