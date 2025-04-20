import numpy as np
import scipy
from typing import Union, Tuple
import ldpc.mod2
import time
import rustworkx as rx

from qec.code_constructions import CSSCode
from qec.utils.sparse_binary_utils import (
    convert_to_binary_scipy_sparse,
    binary_csr_matrix_to_dict,
)


class HypergraphProductCode(CSSCode):
    """
    Implements a Hypergraph Product (HGP) code - derived from two classical linear binary codes.
    
    Parameters
    ----------
    seed_matrix_1 : 
        A classical linear binary code used as a "seed" in the HGP construction method.
    seed_matrix_2 :
        A classical linear binary code used as a "seed" in the HGP construction method.
    name : str, default = None
        The name of the code. If None, the name is set to: "Hypergraph product code" 
    
    Attributes
    ----------
    seed_matrix_1 : scipy.sparse.spmatrix
        The input seed_matrix_1 stored as a scipy sparse matrix.
    seed_matrix_2 : scipy.sparse.spmatrix
        The input seed_matrix_2 stored as a scipy sparse matrix.
    _n1 : int
        Number of columns in seed_matrix_1
    _n2 : int
        Number of columns in seed_matrix_2
    _m1 : int
        Number of rows in seed_matrix_1 (the number of columns of it's transpose)
    _m2 : int
        Number of rows in seed_matrix_2 (the number of columns of it's transpose)
    
    Notes
    -----

    The X and Z stabilizer matrices are given by [1]_: 
    
    .. math::

        \begin{align}
            H_{X} &= \begin{pmatrix}
                        H_{1}\otimes I_{n_{2}} & \,\,I_{r_{1}}\otimes H_{2}^{T}
                     \end{pmatrix}\tag*{(1)}\\
            H_{Z} &= \begin{pmatrix}
                        I_{n_{1}}\otimes H_{2} & \,\,H_{1}^{T}\otimes I_{r_{2}}
                     \end{pmatrix}~, \tag*{(2)}
        \end{align}

    where  :math:`H_1` and  :math:`H_2` correspond to the parity check matrix of the first and second "seed" codes.
    

    .. [1] J.-P. Tillich and G. Zemor, “Quantum LDPC Codes With Positive Rate and Minimum Distance Proportional to the Square Root of the Blocklength”, IEEE Transactions on Information Theory 60, 1193 (2014)
    """

    def __init__(
        self,
        seed_matrix_1: Union[np.ndarray, scipy.sparse.spmatrix],
        seed_matrix_2: Union[np.ndarray, scipy.sparse.spmatrix],
        name: str = None,
    ):
        self.name = name if name else "Hypergraph product code"

        if not all(
            isinstance(seed_m, (np.ndarray, scipy.sparse.spmatrix))
            for seed_m in (seed_matrix_1, seed_matrix_2)
        ):
            raise TypeError(
                "The seed matrices must be either numpy arrays or scipy sparse matrices."
            )

        self.seed_matrix_1 = convert_to_binary_scipy_sparse(seed_matrix_1)
        self.seed_matrix_2 = convert_to_binary_scipy_sparse(seed_matrix_2)

        # maybe move the below to a private _construct_stabilizer_matrices function?
        # --------------------------------------------------------------------------
        self._n1 = seed_matrix_1.shape[1]
        self._n2 = seed_matrix_2.shape[1]

        self._m1 = seed_matrix_1.shape[0]
        self._m2 = seed_matrix_2.shape[0]

        x_left = scipy.sparse.kron(seed_matrix_1, scipy.sparse.eye(self._n2))
        x_right = scipy.sparse.kron(scipy.sparse.eye(self._m1), seed_matrix_2.T)
        self.x_stabilizer_matrix = scipy.sparse.hstack([x_left, x_right])

        z_left = scipy.sparse.kron(scipy.sparse.eye(self._n1), seed_matrix_2)
        z_right = scipy.sparse.kron(seed_matrix_1.T, scipy.sparse.eye(self._m2))
        self.z_stabilizer_matrix = scipy.sparse.hstack([z_left, z_right])
        # --------------------------------------------------------------------------

        super().__init__(self.x_stabilizer_matrix, self.z_stabilizer_matrix, self.name)

        self.code_distance = None

    def compute_exact_code_distance(self) -> int:
        """
        Computes the exact code distance of the HGP code.

        Returns
        -------
        int
            The distance of the code.

        Notes
        -----
        The distance of a HGP code is given as:

        .. math::

            \min(d_1, d_2, d_1^T, d_2^T)

        corresponding to the distance of the seed codes and the distance of their transposes.
        """

        rank_seed_m1 = ldpc.mod2.rank(self.seed_matrix_1)
        rank_seed_m2 = ldpc.mod2.rank(self.seed_matrix_2)

        if self.seed_matrix_1.shape[1] != rank_seed_m1:
            self.d1 = ldpc.mod2.compute_exact_code_distance(self.seed_matrix_1)
        else:
            self.d1 = np.inf

        if self.seed_matrix_2.shape[1] != rank_seed_m2:
            self.d2 = ldpc.mod2.compute_exact_code_distance(self.seed_matrix_2)
        else:
            self.d2 = np.inf

        # note: rank(A) = rank(A^T):
        if self.seed_matrix_1.shape[0] != rank_seed_m1:
            self.d1T = ldpc.mod2.compute_exact_code_distance(self.seed_matrix_1.T)
        else:
            self.d1T = np.inf

        if self.seed_matrix_2.shape[0] != rank_seed_m2:
            self.d2T = ldpc.mod2.compute_exact_code_distance(self.seed_matrix_2.T)
        else:
            self.d2T = np.inf

        self.x_code_distance = min(self.d1T, self.d2)
        self.z_code_distance = min(self.d1, self.d2T)
        self.code_distance = min(self.x_code_distance, self.z_code_distance)

        return self.code_distance

    def estimate_min_distance(self, timeout_seconds: float = 0.025) -> int:
        """
        Estimate the minimum X and Z distance of the HGP code.        
        
        Parameters
        ----------
        timeout_seconds : float, optional
            Time limit in seconds for the full search. Default: 0.25

        Returns
        -------
        int
            Best estimate of the (overall) code distance found within time limit.

        """

        rank_seed_m1 = ldpc.mod2.rank(self.seed_matrix_1)
        rank_seed_m2 = ldpc.mod2.rank(self.seed_matrix_2)

        d1_timeout_seconds = timeout_seconds / 4
        if self.seed_matrix_1.shape[1] != rank_seed_m1:
            d1_start_time = time.time()
            d1_min_estimate, _, _ = ldpc.mod2.estimate_code_distance(
                self.seed_matrix_1, d1_timeout_seconds, 0
            )
            d1_run_time = time.time() - d1_start_time
        else:
            d1_min_estimate = np.inf
            d1_run_time = 0

        d1T_timeout_seconds = (
            (d1_timeout_seconds * 4 - d1_run_time) / 3
            if d1_run_time <= d1_timeout_seconds
            else timeout_seconds / 4
        )
        if self.seed_matrix_1.shape[0] != rank_seed_m1:
            d1T_start_time = time.time()
            d1T_min_estimate, _, _ = ldpc.mod2.estimate_code_distance(
                self.seed_matrix_1.T, d1T_timeout_seconds, 0
            )
            d1T_run_time = time.time() - d1T_start_time
        else:
            d1T_min_estimate = np.inf
            d1T_run_time = 0

        d2_timeout_seconds = (
            (d1T_timeout_seconds * 3 - d1T_run_time) / 2
            if d1T_run_time <= d1T_timeout_seconds
            else timeout_seconds / 4
        )
        if self.seed_matrix_2.shape[1] != rank_seed_m2:
            d2_start_time = time.time()
            d2_min_estimate, _, _ = ldpc.mod2.estimate_code_distance(
                self.seed_matrix_2, d2_timeout_seconds, 0
            )
            d2_run_time = time.time() - d2_start_time
        else:
            d2_min_estimate = np.inf
            d2_run_time = 0

        d2T_timeout_seconds = (
            (d2_timeout_seconds * 2 - d2_run_time)
            if d2_run_time <= d2_timeout_seconds
            else timeout_seconds / 4
        )
        if self.seed_matrix_2.shape[0] != rank_seed_m2:
            d2T_min_estimate, _, _ = ldpc.mod2.estimate_code_distance(
                self.seed_matrix_2.T, d2T_timeout_seconds, 0
            )
        else:
            d2T_min_estimate = np.inf

        self.x_code_distance = min(d1T_min_estimate, d2_min_estimate)
        self.z_code_distance = min(d1_min_estimate, d2T_min_estimate)
        self.code_distance = min(self.x_code_distance, self.z_code_distance)

        return self.code_distance

    def compute_logical_basis(
        self,
    ) -> Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]:
        """
        Compute the logical operator basis for the given HGP code.

        Returns
        -------
        Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]
            Logical X and Z operator bases (lx, lz).
        """

        ker_h1 = ldpc.mod2.kernel(self.seed_matrix_1)
        ker_h2 = ldpc.mod2.kernel(self.seed_matrix_2)
        ker_h1T = ldpc.mod2.kernel(self.seed_matrix_1.T)
        ker_h2T = ldpc.mod2.kernel(self.seed_matrix_2.T)

        row_comp_h1 = ldpc.mod2.row_complement_basis(self.seed_matrix_1)
        row_comp_h2 = ldpc.mod2.row_complement_basis(self.seed_matrix_2)
        row_comp_h1T = ldpc.mod2.row_complement_basis(self.seed_matrix_1.T)
        row_comp_h2T = ldpc.mod2.row_complement_basis(self.seed_matrix_2.T)

        temp = scipy.sparse.kron(ker_h1, row_comp_h2)
        lz1 = scipy.sparse.hstack(
            [
                temp,
                scipy.sparse.csr_matrix(
                    (temp.shape[0], self._m1 * self._m2), dtype=np.uint8
                ),
            ]
        )

        temp = scipy.sparse.kron(row_comp_h1T, ker_h2T)
        lz2 = scipy.sparse.hstack(
            [
                scipy.sparse.csr_matrix(
                    (temp.shape[0], self._n1 * self._n2), dtype=np.uint8
                ),
                temp,
            ]
        )

        self.z_logical_operator_basis = scipy.sparse.csr_matrix(
            scipy.sparse.vstack([lz1, lz2], dtype=np.uint8)
        )

        temp = scipy.sparse.kron(row_comp_h1, ker_h2)
        lx1 = scipy.sparse.hstack(
            [
                temp,
                scipy.sparse.csr_matrix(
                    (temp.shape[0], self._m1 * self._m2), dtype=np.uint8
                ),
            ]
        )

        temp = scipy.sparse.kron(ker_h1T, row_comp_h2T)
        lx2 = scipy.sparse.hstack(
            [
                scipy.sparse.csr_matrix(
                    (temp.shape[0], self._n1 * self._n2), dtype=np.uint8
                ),
                temp,
            ]
        )

        self.x_logical_operator_basis = scipy.sparse.csr_matrix(
            scipy.sparse.vstack([lx1, lx2], dtype=np.uint8)
        )

        # Follows the way it is done in CSSCode -> move it into __init__?
        # ----------------------------------------------------------------
        self.logical_qubit_count = self.x_logical_operator_basis.shape[0]
        # ----------------------------------------------------------------

        return (self.x_logical_operator_basis, self.z_logical_operator_basis)

    def __str__(self):
        """
        String representation of the HGP code. Includes the name and [[n, k, d]] properties of the code.

        Returns
        -------
        str
            String representation of the HGP code.
        """

        return f"{self.name} Hypergraphproduct Code: [[N={self.physical_qubit_count}, K={self.logical_qubit_count}, dx={self.x_code_distance}, dz={self.z_code_distance}]]"

    def _class_specific_save(self):
        class_specific_data = {
            "code_distance": self.code_distance
            if self.code_distance is not None
            else "?",
            "x_code_distance": self.x_code_distance
            if self.x_code_distance is not None
            else "?",
            "z_code_distance": self.z_code_distance
            if self.z_code_distance is not None
            else "?",
            "seed_matrix_1": binary_csr_matrix_to_dict(self.seed_matrix_1),
            "seed_matrix_2": binary_csr_matrix_to_dict(self.seed_matrix_2),
            "x_logical_operator_basis": binary_csr_matrix_to_dict(
                self.x_logical_operator_basis
            ) if self._x_logical_operator_basis is not None else "?",
            "z_logical_operator_basis": binary_csr_matrix_to_dict(
                self.z_logical_operator_basis
            ) if self._z_logical_operator_basis is not None else "?",
        }

        return class_specific_data
    
    # TODO: debug the schedule below, results in non-deterministic detectors

    # def _stabilizer_schedule(self) -> list[Tuple[list[int, int], str]]:
    #     """
    #     Returns the "cardinal" [2]_ stabilizer schedule for circuit compilation.

    #     Returns
    #     -------
    #     list
    #         A list of tuples, where each tuple contains a pair of qubit indices
    #         and a color id. CNOTs with the same color id can be applied in parallel.

    #     Notes
    #     -----
    #     To obtain the balanced graph needed for the optimal circuit depth, we use the
    #     "balanced sign" idea from: https://arxiv.org/abs/2504.02673
    #     However, instead of their heuristics for the sign assignment, we use the coloring
    #     of the seed code's Tanner graph amd assign signs (cardinal directions) to the
    #     edges dependening on whether the integer representing the color is even or odd.

    #     Furthermore, as the N - S stabilizers can be applied in any order without introducing
    #     any additional CNOTs, we do not try to balance the second seed code's tanner graph, simply 
    #     color it. 

    #     .. [2] Tremblay, M. A., Delfosse, N., & Beverland, M. E. (2022). Constant-overhead quantum error correction with thin planar connectivity. Physical Review Letters, 129(5), 050504.
    #     """

    #     seed_1_tanner = rx.PyGraph(multigraph = False)
    #     seed_1_data_nodes = [seed_1_tanner.add_node(i) for i in range(self._n1)]
    #     seed_1_check_nodes = [seed_1_tanner.add_node(i) for i in range(self._m1)]
    #     for j in range(self._n1):
    #         for i in range(self._m1):
    #             if self.seed_matrix_1[i, j] == 1:
    #                 seed_1_tanner.add_edge(seed_1_check_nodes[i], seed_1_data_nodes[j], 1)

    #     seed_1_colored = rx.graph_bipartite_edge_color(seed_1_tanner)
    #     seed_1_ordered = sorted(seed_1_colored.items(), key=lambda x: x[1])

    #     seed_2_tanner = rx.PyGraph(multigraph = False)
    #     seed_2_data_nodes = [seed_2_tanner.add_node(i) for i in range(self._n2)]
    #     seed_2_check_nodes = [seed_2_tanner.add_node(i) for i in range(self._m2)]
    #     for j in range(self._n2):
    #         for i in range(self._m2):
    #             if self.seed_matrix_2[i, j] == 1:
    #                 seed_2_tanner.add_edge(seed_2_check_nodes[i], seed_2_data_nodes[j], 1)

    #     east_tanner = rx.PyGraph(multigraph = False)
    #     west_tanner = rx.PyGraph(multigraph = False)
    #     north_south_tanner = rx.PyGraph(multigraph = False)

    #     num_sector_I = self._n1 * self._n2      # data
    #     num_sector_II = self._m1 * self._n2     # X stabilizers
    #     num_sector_III = self._n1 * self._m2    # Z stabilizers
    #     num_sector_IV = self._m1 * self._m2     # data

    #     num_data = num_sector_I + num_sector_IV
    #     num_stabilizers = num_sector_II + num_sector_III

    #     for tanner in [east_tanner, west_tanner, north_south_tanner]:
    #         for i in range(num_data):
    #             tanner.add_node(i)

    #         for i in range(num_stabilizers):
    #             tanner.add_node(i + num_data)

    #     for edge in seed_1_ordered:
    #         check, data = seed_1_tanner.get_edge_endpoints_by_index(edge[0])
    #         is_even = edge[1] % 2 == 0
    #         target_graph = east_tanner if is_even else west_tanner
    #         alternative_graph = west_tanner if is_even else east_tanner

    #         for i in range(self._n2):
    #             target_graph.add_edge(data + i * self._n1,
    #                                     check + num_data - self._n1 + i * self._m1, 1)
            
    #         for i in range(self._m2):
    #             alternative_graph.add_edge(data + i * self._n1 + num_data + num_sector_II,
    #                                         check + i * self._m1 - self._n1 + num_sector_I, 1)
                
    #     for check, data in seed_2_tanner.edge_list():
    #         for i in range(self._n1):
    #             north_south_tanner.add_edge(data * self._n1 + i,
    #                                         check * self._n1 + i + num_sector_IV + num_sector_II, 1)
    #         for i in range(self._m1):
    #             north_south_tanner.add_edge(data * self._m1 + i + num_data,
    #                                         check * self._m1 + i - num_sector_II + num_sector_I, 1)
                

    #     colored_east = rx.graph_bipartite_edge_color(east_tanner)
    #     colored_west = rx.graph_bipartite_edge_color(west_tanner)
    #     colored_north_south = rx.graph_bipartite_edge_color(north_south_tanner)

    #     ordered_east = sorted(colored_east.items(), key=lambda x: x[1])
    #     ordered_west = sorted(colored_west.items(), key=lambda x: x[1])
    #     ordered_north_south = sorted(colored_north_south.items(), key=lambda x: x[1])

    #     final_list = []

    #     for edge_id, color in ordered_east:
    #         qubit_1, qubit_2 = east_tanner.get_edge_endpoints_by_index(edge_id)
    #         final_list.append(([qubit_2, qubit_1], f"E{color}"))

    #     for edge_id, color in ordered_north_south:
    #         qubit_1, qubit_2 = north_south_tanner.get_edge_endpoints_by_index(edge_id)
    #         final_list.append(([qubit_1, qubit_2], f"NS{color}"))

    #     for edge_id, color in ordered_west:
    #         qubit_1, qubit_2 = west_tanner.get_edge_endpoints_by_index(edge_id)
    #         final_list.append(([qubit_2, qubit_1], f"W{color}"))

    #     return final_list