import numpy as np
from qec.codes import TwistedToricCode
import scipy.sparse

code = TwistedToricCode(3,2)
print(code)
print("Ax=", code.proto_1)
print("Az=", code.proto_2)
print("Lift parameter:", code.lift_parameter)
# temp = code.lx@code.lz.T
# print(temp.toarray()%2)
lx = code.lx.toarray().astype(int)
lz = code.lz.toarray().astype(int)
print("Lx=",lx)
print("Lz=",lz)
# print(lz)


# scipy.sparse.save_npz("hx_surface_5.npz", code.hx)
# scipy.sparse.save_npz("lx_surface_5.npz", code.lx)


# pcm = code.hx.toarray().astype(int)

# out = ""
# for i in range(pcm.shape[0]):
#     for j in range(pcm.shape[1]):
#         if pcm[i, j] != 0:
#             out += f"pcm.insert_entry({i},{j});"

# print(out)
