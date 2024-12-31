import numpy as np
from qec.stabilizer_code.stabilizer_code import StabiliserCode


class FiveQubitCode(StabiliserCode):
    def __init__(self):
        pauli_stabilisers = [["XZZXI"], ["IXZZX"], ["XIXZZ"], ["ZXIXZ"]]
        super().__init__(pauli_stabilisers, name="5-Qubit Code")
