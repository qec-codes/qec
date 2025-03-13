import rustworkx as rx
import numpy as np
from qec.code_constructions import HypergraphProductCode
from qec.circuit_compilation import NoiseModel

class MemoryExperiment:

    def __init__(self,
                 code: HypergraphProductCode, # currently we only support HGP codes
                 noise_model: NoiseModel):

        self.code = code
        self.noise_model = noise_model


        if not isinstance(code, HypergraphProductCode):
            raise ValueError('Currently only Hypergraph-product codes are supported.')
    
    def _coloration_schedule(self, stabilizer_matrix):

        num_checks, num_data = stabilizer_matrix.shape
        
        tanner_graph = rx.PyGraph(multigraph = False)
        
        checks = [tanner_graph.add_node({'check_id' : i}) for i in range(num_checks)]
        data  = [tanner_graph.add_node({'data_id' : i + num_checks}) for i in range(num_data)]
        
        for i in range(num_checks):
            for j in range(num_data):
                if stabilizer_matrix[i, j] == 1:
                    tanner_graph.add_edge(checks[i], data[j], 1)
        
        colored_edges = rx.graph_bipartite_edge_color(tanner_graph)
        ordered_edges = sorted(colored_edges.items(), key = lambda item: item[1])

        return [tanner_graph.get_edge_endpoints_by_index(edge[0]) for edge in ordered_edges]

    def circuit(self, base : str, rounds : int, noise : bool):

        




