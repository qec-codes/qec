import dataclasses
import numpy as np
import stim
from typing import Dict, List, Tuple, Union, Optional, Set

# ------------------------------------------------------------------------------
# FIXME: Adjust operator types (are there more?)

CLIFFORD_1Q = "C1"
CLIFFORD_2Q = "C2"
ANNOTATION = "info"
MPP = "MPP"
MEASURE_RESET_1Q = "MR1"
JUST_MEASURE_1Q = "M1"
JUST_RESET_1Q = "R1"
NOISE = "!?"

OP_TYPES = {
    "I": CLIFFORD_1Q,
    "X": CLIFFORD_1Q,
    "Y": CLIFFORD_1Q,
    "Z": CLIFFORD_1Q,
    "C_XYZ": CLIFFORD_1Q,
    "C_ZYX": CLIFFORD_1Q,
    "H": CLIFFORD_1Q,
    "H_XY": CLIFFORD_1Q,
    "H_XZ": CLIFFORD_1Q,
    "H_YZ": CLIFFORD_1Q,
    "S": CLIFFORD_1Q,
    "SQRT_X": CLIFFORD_1Q,
    "SQRT_X_DAG": CLIFFORD_1Q,
    "SQRT_Y": CLIFFORD_1Q,
    "SQRT_Y_DAG": CLIFFORD_1Q,
    "SQRT_Z": CLIFFORD_1Q,
    "SQRT_Z_DAG": CLIFFORD_1Q,
    "S_DAG": CLIFFORD_1Q,
    "CNOT": CLIFFORD_2Q,
    "CX": CLIFFORD_2Q,
    "CY": CLIFFORD_2Q,
    "CZ": CLIFFORD_2Q,
    "ISWAP": CLIFFORD_2Q,
    "ISWAP_DAG": CLIFFORD_2Q,
    "SQRT_XX": CLIFFORD_2Q,
    "SQRT_XX_DAG": CLIFFORD_2Q,
    "SQRT_YY": CLIFFORD_2Q,
    "SQRT_YY_DAG": CLIFFORD_2Q,
    "SQRT_ZZ": CLIFFORD_2Q,
    "SQRT_ZZ_DAG": CLIFFORD_2Q,
    "SWAP": CLIFFORD_2Q,
    "XCX": CLIFFORD_2Q,
    "XCY": CLIFFORD_2Q,
    "XCZ": CLIFFORD_2Q,
    "YCX": CLIFFORD_2Q,
    "YCY": CLIFFORD_2Q,
    "YCZ": CLIFFORD_2Q,
    "ZCX": CLIFFORD_2Q,
    "ZCY": CLIFFORD_2Q,
    "ZCZ": CLIFFORD_2Q,
    "MPP": MPP,
    "MR": MEASURE_RESET_1Q,
    "MRX": MEASURE_RESET_1Q,
    "MRY": MEASURE_RESET_1Q,
    "MRZ": MEASURE_RESET_1Q,
    "M": JUST_MEASURE_1Q,
    "MX": JUST_MEASURE_1Q,
    "MY": JUST_MEASURE_1Q,
    "MZ": JUST_MEASURE_1Q,
    "R": JUST_RESET_1Q,
    "RX": JUST_RESET_1Q,
    "RY": JUST_RESET_1Q,
    "RZ": JUST_RESET_1Q,
    "DETECTOR": ANNOTATION,
    "OBSERVABLE_INCLUDE": ANNOTATION,
    "QUBIT_COORDS": ANNOTATION,
    "SHIFT_COORDS": ANNOTATION,
    "TICK": ANNOTATION,
    "E": ANNOTATION,
    "DEPOLARIZE1": NOISE,
    "DEPOLARIZE2": NOISE,
    "PAULI_CHANNEL_1": NOISE,
    "PAULI_CHANNEL_2": NOISE,
    "X_ERROR": NOISE,
    "Y_ERROR": NOISE,
    "Z_ERROR": NOISE,
    # Not supported.
    # 'CORRELATED_ERROR': NOISE,
    # 'E': NOISE,
    # 'ELSE_CORRELATED_ERROR',
}
OP_MEASURE_BASES = {
    "M": "Z",
    "MX": "X",
    "MY": "Y",
    "MZ": "Z",
    "MPP": "",
}
COLLAPSING_OPS = {
    op
    for op, t in OP_TYPES.items()
    if t == JUST_RESET_1Q or t == JUST_MEASURE_1Q or t == MPP or t == MEASURE_RESET_1Q
}

# ------------------------------------------------------------------------------
# class NoisyGate:
#     """
#     Represents a noisy gate moment in a quantum circuit.
#     """
#     def __init__(self, op: str, targets: List[int], noise: dict[str, float]):
#         self.op = op
#         self.targets = targets
#         self.noise = noise

#     def noisy_gate(self, )


# @dataclasses.dataclass(frozen=True)
class NoiseModel:
    """
    Noise model class for stim circuits. --> Soon to be generalised.

    This class can be used to create a custom noise model, load a noise model from a file,
    load a noise model from a backend or apply a general noise model of the following type:
    - depolarizing noise (uniform and non-uniform)
    - amplitude damping noise
    - phase damping noise
    - thermal relaxation noise
    - adversarial noise
    - phenomenological noise
    - pauli noise

    Args:

    Attributes:
        system_qubits: All qubits used by the circuit. These are the qubits eligible for idling noise.
        immune_qubits: Qubits to not apply noise to (even if they are operated on).
    """

    def __init__(
        self,
        idle_depolarization: float,
        # before_round_data_depolarization: float = 0.0, # noise before the start of each round
        # additional_depolarization_waiting_for_m_or_r: float = 0,
        # idle_per_duration: float = 0.0,
        # additional_idle: float, ???
        measure: Dict[str, float],
        # before_measure_flip_probability: float = 0.0,
        reset: Dict[str, float],
        # after_reset_flip_probability: float = 0.0,
        gates: Dict[str, float],
        any_clifford_1: Optional[float] = None,
        any_clifford_2: Optional[float] = None,
        # after_clifford_depolarization: float = 0.0, # noise after any clifford o
        # use_correlated_parity_measurement_errors: bool = False ???
    ):
        """
        Initializes a general noise model that supports various types of noise.

        ====
        Example:
        {('X_ERROR', 0.1), ('Y_ERROR', 0.05), ('Z_ERROR', 0.02), ('DEPOLARIZE1', 0.05), ('DEPOLARIZE1', 0.2)}
        """

        self.idle_depolarization = idle_depolarization
        self.measure = measure
        self.reset = reset
        self.gates = gates
        self.any_clifford_1 = any_clifford_1
        self.any_clifford_2 = any_clifford_2

        self.supported_types = {
            "X_ERROR",
            "Y_ERROR",
            "Z_ERROR",
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "CORRELATED_ERROR",
            "E",
            "ELSE_CORRELATED_ERROR",
            "HERALDED_ERASE",
            "HERALDED_PAULI_CHANNEL_1",
            "II_ERROR",
            "I_ERROR",
            "PAULI_CHANNEL_1",
            "PAULI_CHANNEL_2",
        }
        self._validate_spec()

    def _validate_spec(self):
        """
        Validates the noise specification.

        Raises
        ------
        ValueError
            If an unsupported noise type is included.
        """
        # TODO: Fix for new intitialisation items

        # for qubit, noise_list in self.noise_spec.items():
        #     for noise_type, prob in noise_list:
        #         #print(noise_type)
        #         if noise_type not in self.supported_types:
        #             raise ValueError(f"Unsupported noise type: {noise_type}")
        #         if prob < 0 or prob > 1:
        #             raise ValueError(f"Probability for {noise_type} must be between 0 and 1.")
        pass

    # TODO: Check if this works / is necessary - can maybe be added to utils
    @staticmethod
    def _get_noise_type(noise_type: str) -> str:
        """
        Converts noise type to a format compatible with Stim. 
        This can be used to convert noise types from other libraries or formats to the format used in Stim.
        --> e.g. noise from Qiskit or other frameworks.

        Parameters
        ----------
        noise_type : str
            The noise type to convert.

        Returns
        -------
        str
            The converted noise type.
        """
        # if noise_type == 'X_ERROR':
        #     return 'X'
        # elif noise_type == 'Y_ERROR':
        #     return 'Y'
        # elif noise_type == 'Z_ERROR':
        #     return 'Z'
        # else:
        #     return noise_type

    # -------------------------------------------------------------------------
    # TODO: Complete the following methods - Could be more to onsider
    # -------------------------------------------------------------------------

    @staticmethod
    def uniform_depolarizing_noise(p: float) -> "NoiseModel":
        """
        Creates a uniform depolarizing noise model.

        Parameters
        ----------
        p : float
            The probability of depolarizing noise.

        Returns
        -------
        NoiseModel
            A NoiseModel instance with uniform depolarizing noise applied after all Clifford gates,
            to idle qubits, and to resets and measurements as X_ERROR.
        """
        return NoiseModel(
            idle_depolarization=p,  # idle noise is DEPOLARIZE1(p)
            measure={"X": p, "Y": p, "Z": p},  # inject X_ERROR(p) before measurements
            reset={"X": p, "Y": p, "Z": p},  # inject X_ERROR(p) after resets
            gates={},  # Leave per-gate overrides empty
            any_clifford_1=p,  # Use DEPOLARIZE1(p) after all 1Q Clifford gates
            any_clifford_2=p,  # Use DEPOLARIZE2(p) after all 2Q Clifford gates
        )

    @staticmethod
    def non_uniform_depolarizing_noise(p: float) -> "NoiseModel":
        pass

    @staticmethod
    def phenomenological_noise(p: float) -> "NoiseModel":
        """
        Creates a phenamenological noise model by setting the before_round_data_depolarization=p1 (float)
        and before_measure_flip_probability=p2(float). This will insert a DEPOLARIZE1(p1) operation at the
        start of each round targeting every data qubit, and an X_ERROR(p2) just before each measurement operation.
        """
        pass

    @staticmethod
    def adversarial_noise(p: float) -> "NoiseModel":
        pass

    @staticmethod
    def thermal_relaxation_noise(p: float) -> "NoiseModel":
        pass

    @staticmethod
    def pauli_noise(p: float) -> "NoiseModel":
        pass

    @staticmethod
    def amplitude_damping_noise(p: float) -> "NoiseModel":
        pass

    # -------------------------------------------------------------------------
    # Noisy gate functions
    # -------------------------------------------------------------------------

    # -> idle_depolarization: float,
    # before_round_data_depolarization: float = 0.0, # noise before the start of each round
    # additional_depolarization_waiting_for_m_or_r: float = 0,
    # idle_per_duration: float = 0.0,
    # additional_idle: float, ???
    # -> measure: Dict[str, float],
    # before_measure_flip_probability: float = 0.0,
    # -> reset: Dict[str, float],
    # after_reset_flip_probability: float = 0.0,
    # -> gates: Dict[str, float],
    # -> any_clifford_1: float,
    # -> any_clifford_2: float,
    # after_clifford_depolarization: float = 0.0, # noise after any clifford o
    # use_correlated_parity_measurement_errors: bool = False ???

    # -------------------------------------------------------------------------
    # Nosy circuit
    # -------------------------------------------------------------------------

    def noisy_circuit(
        self,
        circuit: stim.Circuit,
        *,  # TODO: Check - want below args to be from noise model initialisation
        system_qubits: Optional[Set[int]] = None,
        immune_qubits: set[int] | None = None,
    ) -> stim.Circuit:
        """
        Applies the defined noise to the given Stim circuit.

        Parameters
        ----------
        circuit : stim.Circuit
            The Stim circuit to which the noise model will be applied.

        system_qubits : set[int], optional
            The set of qubits to which the noise model will be applied. If None, all qubits in the circuit will be used.
        immune_qubits : set[int], optional
            The set of qubits to which the noise model will not be applied. If None, no qubits will be immune.

        Returns
        -------
        stim.Circuit
            A new Stim circuit with noise applied.
        """
        if system_qubits is None:
            system_qubits = set(range(circuit.num_qubits))
        if immune_qubits is None:
            immune_qubits = set()

        
        noisy = stim.Circuit()
        tick_count = 0
        qubits_touched = {q: 0 for q in system_qubits}  # Track qubit activity within the current TICK

        for inst in circuit:
            # FIXME: qubits_touched: Dict[int, int] = {q: 0 for q in system_qubits}
            op = inst.name
            targets = [int(t.value) for t in inst.targets_copy()]
            # noisy.append(inst)

            # Track qubit activity (for possible idle noise)
            if OP_TYPES.get(op) in {
                CLIFFORD_1Q,
                CLIFFORD_2Q,
                JUST_MEASURE_1Q,
                JUST_RESET_1Q,
                MEASURE_RESET_1Q,
            }:
                for q in targets:
                    qubits_touched[q] += 1
                    # if q not in immune_qubits:
                    #     qs_last_touched[q] += 1 # if qubits acted on by an operation, increment their count

            # Append gate noise
            if OP_TYPES.get(op) == CLIFFORD_1Q:
                noisy.append(inst)
                for q in targets:
                    if q not in immune_qubits:
                        if op in self.gates:
                            p = self.gates[op]
                        else:                
                            p = self.any_clifford_1
                        if p > 0:
                            noisy.append_operation("DEPOLARIZE1", [q], p)

            elif OP_TYPES.get(op) == CLIFFORD_2Q:
                noisy.append(inst)
                if all(q not in immune_qubits for q in targets):
                    if op in self.gates:
                        p = self.gates[op]
                    else:
                        p = self.any_clifford_2
                    if p > 0:
                        noisy.append_operation("DEPOLARIZE2", targets, p)

            # Append measurement/reset noise
            elif OP_TYPES.get(op) == JUST_MEASURE_1Q:
                for q in targets:
                    basis = OP_MEASURE_BASES.get(op, "Z") # Defaults to "Z" if not found
                    p = self.measure.get(basis, 0.0)  # If value associated to measurement key doesnt exist - set to 0
                    if p > 0 and q not in immune_qubits:
                        if basis == "Z":
                            noisy.append_operation("X_ERROR", [q], p)
                        elif basis == "X":
                            noisy.append_operation("Z_ERROR", [q], p)
                        elif basis == "Y":
                            # Optional: could model Y flips with either X or Z, or a depolarizing channel
                            noisy.append_operation("Y_ERROR", [q], p)
                noisy.append(inst)
            
            elif OP_TYPES.get(op) == MEASURE_RESET_1Q:
                # Add measuremnt noise (before the operation)
                for q in targets:
                    basis = op[-1] if len(op) > 2 else "Z"  # e.g. MRX -> "X"
                    p_measure = self.measure.get(basis, 0.0) # If value associated to measurement key doesnt exist - set to 0
                    if p_measure > 0 and q not in immune_qubits:
                        if basis == "Z":
                            noisy.append_operation("X_ERROR", [q], p_measure)
                        elif basis == "X":
                            noisy.append_operation("Z_ERROR", [q], p_measure)
                        elif basis == "Y":
                            # Optional: could model Y flips with either X or Z, or a depolarizing channel
                            noisy.append_operation("Y_ERROR", [q], p_measure)
                
                # Append the actual MEASURE_RESET_1Q operation (measurement + reset)
                noisy.append(inst)

                # Append reset noise (after the operation)
                for q in targets:
                    p_reset = self.reset.get(basis, 0.0) # If value associated to reset key doesnt exist - set to 0
                    if p_reset > 0 and q not in immune_qubits:
                        if basis == "Z":
                            noisy.append_operation("X_ERROR", [q], p_reset)
                        elif basis == "X":
                            noisy.append_operation("Z_ERROR", [q], p_reset)
                        elif basis == "Y":
                            # Optional: could model Y flips with either X or Z, or a depolarizing channel
                            noisy.append_operation("Y_ERROR", [q], p_reset)

            elif OP_TYPES.get(op) == JUST_RESET_1Q:
                noisy.append(inst)
                for q in targets:
                    basis = op[-1] if len(op) > 1 else "Z"  # e.g. RX -> "X"
                    p = self.reset.get(basis, 0.0) # If value associated to reset key doesnt exist - set to 0
                    if p > 0 and q not in immune_qubits:
                        if basis == "Z":
                            noisy.append_operation("X_ERROR", [q], p)
                        elif basis == "X":
                            noisy.append_operation("Z_ERROR", [q], p)
                        elif basis == "Y":
                            # Optional: could model Y flips with either X or Z, or a depolarizing channel
                            noisy.append_operation("Y_ERROR", [q], p)

            # Detect idle qubits within each TICK and apply idle noise
            # TODO: Initialise qubits_touched to 0 at the start of each TICK?

            # All other operations (like DETECTOR, OBSERVABLE_INCLUDE, SHIFT_COORDS, etc.)
            elif op != "TICK":
                noisy.append(inst)

            elif op == "TICK":
                # Apply the idle noise BEFORE appending TICK
                if self.idle_depolarization > 0:
                    for q in system_qubits:
                        if q not in immune_qubits and qubits_touched[q] == 0:
                            noisy.append_operation("DEPOLARIZE1", [q], self.idle_depolarization)

                # Append the TICK operation
                noisy.append(inst)
                tick_count += 1

                # Reset the activity tracker for the next TICK interval
                qubits_touched = {q: 0 for q in system_qubits}

            # TODO: Idle noise added after each TICK operation
            # if op == "TICK":
            #     tick_count += 1
            #     noisy.append(inst)
            #     for q in system_qubits:
            #         if q not in immune_qubits and qubits_touched[q] == 0:
            #             if self.idle > 0:
            #                 noisy.append_operation("DEPOLARIZE1", [q], self.idle)

            #     # Reset for next tick
            #     qubits_touched = {q: 0 for q in system_qubits}

            # ----------------------------------------------------------
            # OLD CODE BELOW
            # ----------------------------------------------------------
            # if op == "TICK":
            #     tick_count += 1
            #     noisy.append(inst)
            #     for q in system_qubits:
            #         if q in immune_qubits:
            #             continue
            #         if qubits_touched[q] == 0:
            #             # If the qubit was not touched by any operation, apply idle noise
            #             if self.idle > 0:
            #                 noisy.append_operation("DEPOLARIZE1", [q], self.idle)
            #     # for q in system_qubits:
            #     #     if q in immune_qubits:
            #     #         continue
            #     #     if qs_last_touched[q] < tick_count:
            #     #         if self.idle > 0:
            #     #             noisy.append_operation("DEPOLARIZE1", [q], self.idle)
            #     #         qs_last_touched[q] = tick_count

        return noisy

