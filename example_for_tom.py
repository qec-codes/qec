from ldpc.codes import hamming_code, random_binary_code
from qec.css import CssCode
from qec.hgp import HyperGraphProductCode

H = hamming_code(3)

steane_code = CssCode(H, H, name="Steane")

print(steane_code)

hgp_hamming = HyperGraphProductCode(H, H, name="HGP Hamming")

print(hgp_hamming)


## now let's generate a random HGP code
random_binary_matrix = random_binary_code(12,20,10, seed=31415)
print(random_binary_matrix.toarray())

hgp_random = HyperGraphProductCode(random_binary_matrix, random_binary_matrix, name="HGP Random")

print(hgp_random)

## the method below is inherited from CssCode class
hgp_random.estimate_min_distance(reduce_logical_basis=True)

print(hgp_random)

