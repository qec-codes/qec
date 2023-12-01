import numpy as np
import ldpc.protograph as pt
from qec.css import CssCode


class LiftedHypergraphProduct(CssCode):
    def __init__(self, lift_parameter, a, b=None, name: str = None):
        """
        Generates the lifted hypergraph product of the protographs a and b
        """
        self.a = a

        self.a_m, self.a_n = self.a.shape

        if b is None:
            self.b = pt.copy(self.a)
        else:
            self.b = b

        self.b_m, self.b_n = self.b.shape

        self.hx1_proto = np.kron(self.a, pt.identity(self.b_n))
        self.hx2_proto = np.kron(pt.identity(self.a_m), self.b.T)
        self.hx_proto = pt.hstack([self.hx1_proto, self.hx2_proto])

        self.hz1_proto = np.kron(pt.identity(self.a_n), self.b)
        self.hz2_proto = np.kron(self.a.T, pt.identity(self.b_m))
        self.hz_proto = pt.hstack([self.hz1_proto, self.hz2_proto])

        self.lift_parameter = lift_parameter

        hx = self.hx_proto.to_binary(lift_parameter)
        hz = self.hz_proto.to_binary(lift_parameter)

        CssCode.__init__(self, hx, hz, name=name)

    @property
    def protograph(self):
        px = pt.vstack([pt.zeros(self.hz_proto.shape), self.hx_proto])
        pz = pt.vstack([self.hz_proto, pt.zeros(self.hx_proto.shape)])
        return pt.hstack([px, pz])

    @property
    def hx1(self):
        return self.hx1_proto.to_binary(self.lift_parameter)

    @property
    def hx2(self):
        return self.hx2_proto.to_binary(self.lift_parameter)

    @property
    def hz1(self):
        return self.hz1_proto.to_binary(self.lift_parameter)

    @property
    def hz2(self):
        return self.hz2_proto.to_binary(self.lift_parameter)
