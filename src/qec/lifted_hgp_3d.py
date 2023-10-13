from os import error
import numpy as np
import qec.protograph as pt
from qec.css import CssCode
import scipy.sparse
from qec.css_old import css_code


def kron3(A, B, C):
    return np.kron(np.kron(A, B), C)


class LiftedHGP3D(css_code):
    ''' Class for constructing a 3D lifted hypgergraph product code from 3 
    Protographs
    '''

    def __init__(self, seed_protograph_a: pt.array,
                 seed_protograph_b: pt.array,
                 seed_protograph_c: pt.array, lift_parameter: int):
        '''Initializes a CSS code with two parity check matrices and one type 
        of meta checks

        Arguments:
            seed_protograph_a (Protograph)
            seed_protograph_b (Protograph): optional input, if not given set 
                                            equal to A
            seed_protograph_c (Protograph): optional input, if not given set 
                                            equal to A

        '''
        self.lift_parameter = lift_parameter

        self.initialize_boundary_operators(
            seed_protograph_a, seed_protograph_b, seed_protograph_c)

        print(type(self.hz))
        print(self.hz.toarray())

        super().__init__(self.hx.toarray(), self.hz.toarray())

    def initialize_boundary_operators(self, proto_A, proto_B, proto_C):

        A_m, A_n = proto_A.shape
        B_m, B_n = proto_B.shape
        C_m, C_n = proto_C.shape

        I_A_m = pt.identity(A_m)
        I_A_n = pt.identity(A_n)

        I_B_m = pt.identity(B_m)
        I_B_n = pt.identity(B_n)

        I_C_m = pt.identity(C_m)
        I_C_n = pt.identity(C_n)

        # delta_0

        # r1 = kron3(proto_A, I_B_n, I_C_n)
        # print(r1)
        # r1.to_binary(13)

        r1 = kron3(proto_A, I_B_n, I_C_n).to_binary(self.lift_parameter)
        r2 = kron3(I_A_n, proto_B, I_C_n).to_binary(self.lift_parameter)
        r3 = kron3(I_A_n, I_B_n, proto_C).to_binary(self.lift_parameter)
        self.delta_0 = scipy.sparse.vstack([r1, r2, r3])

        # print(r3)

        # delta_1
        r11 = kron3(I_A_m, proto_B, I_C_n).to_binary(self.lift_parameter)
        r12 = kron3(proto_A, I_B_m, I_C_n).to_binary(self.lift_parameter)
        r21 = kron3(I_A_m, I_B_n, proto_C).to_binary(self.lift_parameter)

        r22 = scipy.sparse.csr_matrix(np.array([]),dtype=np.uint8).reshape=(r21.shape[0], r12.shape[1])
        r23 = kron3(proto_A, I_B_n, I_C_m).to_binary(self.lift_parameter)

        r13 = scipy.sparse.csr_matrix(np.array([]), dtype=np.uint8).reshape=(r11.shape[0], r23.shape[1])

        r32 = kron3(I_A_n, I_B_m, proto_C).to_binary(self.lift_parameter)
        r33 = kron3(I_A_n, proto_B, I_C_m).to_binary(self.lift_parameter)
        r31 = np.zeros((r32.shape[0], r21.shape[1]))

        row1 = scipy.sparse.hstack([r11, r12, r13])
        row2 = scipy.sparse.hstack([r21, r22, r23])
        row3 = scipy.sparse.hstack([r31, r32, r33])

        self.delta_1 = scipy.sparse.vstack([row1, row2, row3])

        # delta_2
        c1 = kron3(I_A_m, I_B_m, proto_C).to_binary(self.lift_parameter)
        c2 = kron3(I_A_m, proto_B, I_C_m).to_binary(self.lift_parameter)
        c3 = kron3(proto_A, I_B_m, I_C_m).to_binary(self.lift_parameter)
        self.delta_2 = scipy.sparse.hstack([c1, c2, c3])

        # parity checks matrices
        self.hz = self.delta_0.T.astype(np.uint8)
        self.hx = self.delta_1.astype(np.uint8)

        # meta-check matrices
        self.mx = self.delta_2.astype(np.uint8)
        self.mz = self.delta_1.T.astype(np.uint8)

    

        # hgp code parameters
        self.N = self.hx.shape[1]