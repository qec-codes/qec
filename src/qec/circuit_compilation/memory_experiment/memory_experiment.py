################################################################################
# Missing features:
# - 'cardinal' circuit schedule
# - noise model
# - our own circuit class (instead of stim.Circuit)
################################################################################


import rustworkx as rx
import numpy as np
from qec.code_constructions import HypergraphProductCode
import stim

class MemoryExperiment:

    def __init__(self,
                 code: HypergraphProductCode, # currently we only support HGP codes
                 # noise_model: NoiseModel
                 ):

        self.code = code
        # self.noise_model = noise_model


        if not isinstance(code, HypergraphProductCode):
            raise ValueError('Currently only Hypergraph-product codes are supported.')
    
    def _coloration_circuit(self, basis : str, rounds : int, noise : bool):

        num_x_checks, num_data = self.code.x_stabilizer_matrix.shape
        num_z_checks, _ = self.code.z_stabilizer_matrix.shape
        
        data_qubits = [*range(num_data)]
        x_stabilizer_qubits = [*range(num_data, num_data + num_x_checks)]
        z_stabilizer_qubits = [*range(num_data + num_x_checks, num_data + num_x_checks + num_z_checks)]

        #------------------------------------------------------------------------------
        # coloration schedule:
        #------------------------------------------------------------------------------
        
        z_tanner_graph = rx.PyGraph(multigraph = False)
        x_tanner_graph = rx.PyGraph(multigraph = False)

        x_nodes = [x_tanner_graph.add_node(i) for i in x_stabilizer_qubits]
        z_nodes = [z_tanner_graph.add_node(i) for i in z_stabilizer_qubits]
        x_data_nodes = [x_tanner_graph.add_node(i) for i in data_qubits]
        z_data_nodes = [z_tanner_graph.add_node(i) for i in data_qubits]

        for j in range(num_data):
            for i in range(num_x_checks):
                if self.code.x_stabilizer_matrix[i, j] == 1:
                    x_tanner_graph.add_edge(x_nodes[i], x_data_nodes[j], 1)
            for k in range(num_z_checks):
                if self.code.z_stabilizer_matrix[k, j] == 1:
                    z_tanner_graph.add_edge(z_nodes[k], z_data_nodes[j], 1)

        x_colored = rx.graph_bipartite_edge_color(x_tanner_graph)
        x_ordered = sorted(x_colored.items(), key = lambda item: item[1])
        z_colored = rx.graph_bipartite_edge_color(z_tanner_graph)
        z_ordered = sorted(z_colored.items(), key = lambda item: item[1])

        #------------------------------------------------------------------------------
        # cycle to repeat:
        #------------------------------------------------------------------------------

        cycle = stim.Circuit()
        cycle.append('H', x_stabilizer_qubits)

        previous_color = 0
        for cnot in [(z_tanner_graph.get_edge_endpoints_by_index(edge[0]), edge[1]) for edge in z_ordered]:
            control = z_tanner_graph.get_node_data(cnot[0][1])
            target = z_tanner_graph.get_node_data(cnot[0][0])
            cycle.append('CNOT', [control, target])
            cycle.append('TICK') if cnot[1] != previous_color else None
            previous_color = cnot[1]

        previous_color = 0 
        for cnot in [(x_tanner_graph.get_edge_endpoints_by_index(edge[0]), edge[1]) for edge in x_ordered]:
            control = x_tanner_graph.get_node_data(cnot[0][0])
            target = x_tanner_graph.get_node_data(cnot[0][1])
            cycle.append('CNOT', [control, target])
            cycle.append('TICK') if cnot[1] != previous_color else None
            previous_color = cnot[1]


        cycle.append('TICK')
        cycle.append('H', x_stabilizer_qubits)
        cycle.append('TICK')
        cycle.append('MR' + basis, x_stabilizer_qubits + z_stabilizer_qubits)

        
        #------------------------------------------------------------------------------
        # head of circuit:
        #------------------------------------------------------------------------------
        
        head = stim.Circuit()

        head.append('R' + basis, data_qubits)
        head.append('RZ', x_stabilizer_qubits + z_stabilizer_qubits)
        head.append('TICK')

        head += cycle
        
        if basis == 'Z':
            for i in range(num_z_checks):
                head.append('DETECTOR', [stim.target_rec(-i - 1)])

        elif basis == 'X':
            for i in range(num_x_checks):
                head.append('DETECTOR', [stim.target_rec(-i - 1 - num_z_checks)])

        
        #------------------------------------------------------------------------------
        # body of circuit:
        #------------------------------------------------------------------------------
        
        body = cycle.copy()

        if basis == 'Z':
            for i in range(num_z_checks):
                body.append('DETECTOR', [stim.target_rec(-i - 1),
                                         stim.target_rec(-i - (num_z_checks + num_x_checks) - 1)])

        elif basis == 'X':
            for i in range(num_x_checks):
                body.append('DETECTOR', [stim.target_rec(-i - 1 - num_z_checks),
                                         stim.target_rec(-i - (2 * num_z_checks + num_x_checks) - 1)])

        
        #------------------------------------------------------------------------------
        # tail of circuit:
        #------------------------------------------------------------------------------

        tail = stim.Circuit()
        tail.append('M' + basis, data_qubits)

        if basis == 'Z':
            for i, stabilizer in enumerate(self.code.z_stabilizer_matrix.toarray()):
                last_measurement_of_stabilizer = [stim.target_rec(- i - num_data - 1)]
                stabilized_data = [stim.target_rec(-1 * (num_data - j)) for j in np.where(stabilizer == 1)[0]]
                tail.append('DETECTOR', stabilized_data + last_measurement_of_stabilizer)

            for i, observable in enumerate(self.code.z_logical_operator_basis.toarray()):
                observed_data = [stim.target_rec(-1 * (num_data - j)) for j in np.where(observable == 1)[0]]
                tail.append('OBSERVABLE_INCLUDE', observed_data, i)

        elif basis == 'X':
            for i, stabilizer in enumerate(self.code.x_stabilizer_matrix.toarray()):
                last_measurement_of_stabilizer = [stim.target_rec(- i - num_data - num_z_checks - 1)]
                stabilized_data = [stim.target_rec(-1 * (num_data - j)) for j in np.where(stabilizer == 1)[0]]
                tail.append('DETECTOR', stabilized_data + last_measurement_of_stabilizer)

            for i, observable in enumerate(self.code.x_logical_operator_basis.toarray()):
                observed_data = [stim.target_rec(-1 * (num_data - j)) for j in np.where(observable == 1)[0]]
                tail.append('OBSERVABLE_INCLUDE', observed_data, i)
            
        return head + body * (rounds - 1) + tail



            

        




