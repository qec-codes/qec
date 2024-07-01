import pytest
import numpy as np
import scipy.sparse

import ldpc.codes
import ldpc.mod2
import qec.hgp
import qec.codes


def test_surface_code():
    qcode = qec.codes.SurfaceCode(2, 10)
    qcode.estimate_min_distance()
    qcode.test_logical_basis()

    assert qcode.dx == 10
    assert qcode.dz == 2

    print(qcode)


def test_toric_code():
    qcode = qec.codes.ToricCode(4)
    qcode.estimate_min_distance()
    qcode.test_logical_basis()

    print(qcode)

def test_rotated_surface_code():

    for d in range(3,20,2):
        qcode = qec.codes.RotatedSurfaceCode(d)
        # qcode.estimate_min_distance()
        qcode.test_logical_basis()

        print(qcode)
        path="rotated_surface_code_npz/"
        code_label=f"rotated_surface_code_{qcode.N}_{qcode.K}_{qcode.d}"
        qcode.save_npz(path=path,code_label=code_label)
        hx = scipy.sparse.load_npz(path+code_label+"_hx.npz")
        hz = scipy.sparse.load_npz(path+code_label+"_hz.npz")
        lx = scipy.sparse.load_npz(path+code_label+"_lx.npz")
        lz = scipy.sparse.load_npz(path+code_label+"_lz.npz")
        qcode = qec.css.CssCode(hx=hx,hz=hz)
        print(qcode)


        path="rotated_surface_code_txt/"
        code_label=f"rotated_surface_code_{qcode.N}_{qcode.K}_{qcode.d}"
        qcode.save_txt(path=path,code_label=code_label)
        hx = np.loadtxt(path+code_label+"_hx.txt").astype(int)
        hz = np.loadtxt(path+code_label+"_hz.txt").astype(int)
        lx = np.loadtxt(path+code_label+"_lx.txt").astype(int)
        lz = np.loadtxt(path+code_label+"_lz.txt").astype(int)
        qcode = qec.css.CssCode(hx=hx,hz=hz)
        print(qcode)

    # print(qcode.hx.toarray())


if __name__ == "__main__":
    test_surface_code()
    test_toric_code()
    test_rotated_surface_code()
