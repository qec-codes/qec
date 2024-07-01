import galois

GF = galois.GF(4)

a = GF([3, 0])

print(GF([3,0])*GF([1,0]))

pmat = GF([[0,0,1,1],[3,3,3,3]])

P,L,U = pmat.plu_decompose()

print(U)

z = GF.Zeros((4,4))
print(z)