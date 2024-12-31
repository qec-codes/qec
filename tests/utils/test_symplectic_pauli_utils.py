import scipy
import numpy as np
from qec.utils.symplectic_pauli_utils import symplectic_product
from qec.utils.code_utils import pauli_str_to_binary_pcm, binary_pcm_to_pauli_str
import scipy.sparse


def test_commuting_paulis():
    mat = [["XXXX"], ["ZZZZ"]]
    mat = pauli_str_to_binary_pcm(mat)
    sp = symplectic_product(mat, mat)
    sp.eliminate_zeros()

    assert sp.nnz == 0

    mat = [["XXII"], ["IIZZ"]]
    mat = pauli_str_to_binary_pcm(mat)
    sp = symplectic_product(mat, mat)
    sp.eliminate_zeros()

    assert sp.nnz == 0

    mat = [["XIII"], ["IIIZ"]]
    mat = pauli_str_to_binary_pcm(mat)
    sp = symplectic_product(mat, mat)
    sp.eliminate_zeros()

    assert sp.nnz == 0

    mat = [["XIZI"], ["ZIXI"]]
    mat = pauli_str_to_binary_pcm(mat)
    sp = symplectic_product(mat, mat)
    sp.eliminate_zeros()

    assert sp.nnz == 0


def test_non_commuting_paulis():
    mat = [["X"], ["Z"]]
    mat = pauli_str_to_binary_pcm(mat)
    sp = symplectic_product(mat, mat)
    assert sp.nnz != 0

    mat = [["XZZ"], ["ZXX"]]
    mat = pauli_str_to_binary_pcm(mat)
    sp = symplectic_product(mat, mat)
    sp.eliminate_zeros()
    assert sp.nnz != 0

    mat = [["ZZZ"], ["XXX"]]
    mat = pauli_str_to_binary_pcm(mat)

    sp = symplectic_product(mat, mat)
    sp.eliminate_zeros()
    assert sp.nnz != 0
