import numpy as np
from qec.codes import SurfaceCode
import scipy.sparse

code = SurfaceCode(5)

scipy.sparse.save_npz("hx_surface_5.npz", code.hx)
scipy.sparse.save_npz("lx_surface_5.npz", code.lx)


pcm = code.hx.toarray().astype(int)

out = ""
for i in range(pcm.shape[0]):
    for j in range(pcm.shape[1]):
        if pcm[i, j] != 0:
            out += f"pcm.insert_entry({i},{j});"

print(out)
