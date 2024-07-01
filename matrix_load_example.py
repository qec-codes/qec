import numpy as np
import scipy

## Surface code parameters, set the parameters of the surface code you want to load.
N=9
K=1
D=3

## Load the matrix from the dense txt file
folder="rotated_surface_code_txt/"
code_label=f"rotated_surface_code_{N}_{K}_{D}"

hx = np.loadtxt(folder+code_label+"_hx.txt").astype(int) # X stabilisers, MxN matrix where each row is a stabiliser
hz = np.loadtxt(folder+code_label+"_hz.txt").astype(int) # Z stabilisers, MxN matrix where each row is a stabiliser
lx = np.loadtxt(folder+code_label+"_lx.txt").astype(int) # X logical operators, KxN matrix where each row is a logical operator
lz = np.loadtxt(folder+code_label+"_lz.txt").astype(int) # Z logical operators, KxN matrix where each row is a logical operator


## Load the matrix from the sparse npz file

folder="rotated_surface_code_npz/"
code_label=f"rotated_surface_code_{N}_{K}_{D}"

hx = scipy.sparse.load_npz(folder+code_label+"_hx.npz") # X stabilisers, MxN matrix where each row is a stabiliser
hz = scipy.sparse.load_npz(folder+code_label+"_hz.npz") # Z stabilisers, MxN matrix where each row is a stabiliser
lx = scipy.sparse.load_npz(folder+code_label+"_lx.npz") # X logical operators, KxN matrix where each row is a logical operator
lz = scipy.sparse.load_npz(folder+code_label+"_lz.npz") # Z logical operators, KxN matrix where each row is a logical operator

