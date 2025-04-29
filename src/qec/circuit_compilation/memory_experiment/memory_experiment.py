import numpy as np
from qec.code_constructions import HypergraphProductCode
from qec.circuit_compilation.noise_model import NoiseModel
import stim

class MemoryExperiment:
    """
    Class for compiling memory experiments.

    Parameters
    ----------
    code : HypergraphProductCode
        The code to be used in the experiment.
    noise_model : NoiseModel, optional
        The noise model to be used in the experiment. If None, the experiment is noise-free.

    Attributes
    ----------
    code : HypergraphProductCode
        The input code to be used in the experiment.
    noise_model : NoiseModel
        The noise model to be used in the experiment.
    num_x_checks : int
        The number of X stabilizers in the code.
    num_z_checks : int
        The number of Z stabilizers in the code.
    num_data : int
        The number of data qubits in the code.
    data_qubits : list
        The list of data qubit ids of the code.
    x_stabilizer_qubits : list
        The list of X stabilizer qubit ids of the code.
    z_stabilizer_qubits : list
        The list of Z stabilizer qubit ids of the code
    total_qubits : int
        The total number of qubits in the code (data + stabilizer).
    """

    def __init__(self,
                 code: HypergraphProductCode, # currently we only support HGP codes
                 noise_model: NoiseModel = None, # <-- optional noise model
                 ):

        self.code = code
        self.noise_model = noise_model

        if not isinstance(code, HypergraphProductCode):
            raise ValueError('Currently only Hypergraph-product codes are supported.')
                
        self.num_x_checks, self.num_data = self.code.x_stabilizer_matrix.shape
        self.num_z_checks, _ = self.code.z_stabilizer_matrix.shape
        
        self.total_qubits = self.num_data + self.num_x_checks + self.num_z_checks
        
        self.data_qubits = [*range(self.num_data)]
        self.x_stabilizer_qubits = [*range(self.num_data, self.num_data + self.num_x_checks)]
        self.z_stabilizer_qubits = [*range(self.num_data + self.num_x_checks, self.num_data + self.num_x_checks + self.num_z_checks)]

        
    def circuit(self,
                basis : str = 'Z',
                rounds : int = 1,
                noise : bool = False,
                ) -> stim.Circuit:
        
        """
        Compiles a memory experiment circuit. 

        Parameters
        ----------
        basis : str
            The basis of the memory experiment. Can be 'Z' or 'X'.
        rounds : int
            The number of stabilizer rounds the experiment consists of.
        noise : bool
            If True, the circuit will include the noise, described by the noise 
            model passed to the MemoryExperiment object.

        Returns
        -------
        stim.Circuit
            The compiled circuit for the memory experiment.
        
        """

        cycle = stim.Circuit()
        cycle.append('H', self.x_stabilizer_qubits)
        cycle.append('TICK')
        
        cnots = self.code._stabilizer_schedule()
        
        previous_color = cnots[0][1]

        for qubits, color in cnots:
            cycle.append('TICK') if color != previous_color else None
            cycle.append('CNOT', qubits)
            previous_color = color

        cycle.append('TICK')
        cycle.append('H', self.x_stabilizer_qubits)
        cycle.append('TICK')
        cycle.append('MR', self.z_stabilizer_qubits + self.x_stabilizer_qubits)

        head = stim.Circuit()
        head.append('R' + basis, self.data_qubits)
        head.append('RZ', self.z_stabilizer_qubits + self.x_stabilizer_qubits)
        head.append('TICK')
        head += cycle

        if basis == 'Z':
            for i in range(self.num_z_checks):
                head.append('DETECTOR', [stim.target_rec(-i - 1 - self.num_x_checks)])

        elif basis == 'X':
            for i in range(self.num_x_checks):
                head.append('DETECTOR', [stim.target_rec(-i - 1)])

        body = cycle.copy()

        if basis == 'Z':
            for i in range(self.num_z_checks):
                body.append('DETECTOR', [stim.target_rec(-i - self.num_x_checks - 1),
                                        stim.target_rec(-i - ( self.num_z_checks + 2 * self.num_x_checks) - 1)])

        elif basis == 'X':
            for i in range(self.num_x_checks):
                body.append('DETECTOR', [stim.target_rec(-i - 1),
                                    stim.target_rec(-i - (self.num_z_checks + self.num_x_checks) - 1)])

        tail = stim.Circuit()
        tail.append('M' + basis, self.data_qubits)

        if basis == 'Z':
            for i, stabilizer in enumerate(self.code.z_stabilizer_matrix.toarray()):
                last_measurement_of_stabilizer = [stim.target_rec(i - self.num_data - self.num_x_checks - self.num_z_checks)]
                stabilized_data = [stim.target_rec(-1 * (self.num_data - j)) for j in np.where(stabilizer == 1)[0]]
                tail.append('DETECTOR', stabilized_data + last_measurement_of_stabilizer)

            for i, observable in enumerate(self.code.z_logical_operator_basis.toarray()):
                observed_data = [stim.target_rec(-1 * (self.num_data - j)) for j in np.where(observable == 1)[0]]
                tail.append('OBSERVABLE_INCLUDE', observed_data, i)
        
        elif basis == 'X':
            for i, stabilizer in enumerate(self.code.x_stabilizer_matrix.toarray()):
                last_measurement_of_stabilizer = [stim.target_rec(i - self.num_data - self.num_x_checks)]
                stabilized_data = [stim.target_rec(-1 * (self.num_data - j)) for j in np.where(stabilizer == 1)[0]]
                tail.append('DETECTOR', stabilized_data + last_measurement_of_stabilizer)

            for i, observable in enumerate(self.code.x_logical_operator_basis.toarray()):
                observed_data = [stim.target_rec(-1 * (self.num_data - j)) for j in np.where(observable == 1)[0]]
                tail.append('OBSERVABLE_INCLUDE', observed_data, i)

        out_circuit = head + body * (rounds - 1) + tail 

        if noise and self.noise_model is not None:
            # Apply the noise model to the circuit
            out_circuit = self.noise_model.noisy_circuit(out_circuit)

        return out_circuit
