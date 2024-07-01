import ldpc.codes.hamming_code
import numpy as np
import ldpc.codes
import ldpc.mod2

H = ldpc.codes.hamming_code(3).toarray()

H = np.array([[0,1,0],[0,0,1],[1,1,0],[1,0,0],[1,0,1],[0,1,1],[1,1,1]])
print(H)


# a = H@H.T%2

# print(repr(a))

# print(ldpc.mod2.inverse(a)@a%2) 



# inv = H.T@ldpc.mod2.left_inverse(H)

# print(H@inv%2)

plu = ldpc.mod2.PluDecomposition(H,True,False)


print(plu.U.toarray())

inverse = plu.U.T@plu.L
inverse.data = inverse.data%2

print(inverse.toarray())

print(inverse@H%2)