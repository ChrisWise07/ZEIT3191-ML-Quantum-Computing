import imp
from typing import Dict, Tuple
from qiskit.circuit import QuantumCircuit, Parameter, ClassicalRegister, Clbit
import math
import numpy as np


def return_initial_quantum_circuit(
    num_qubits: int, num_classical_bits: int
) -> QuantumCircuit:
    """
    Returns an inital quantum circuit (i.e. circuit with no gates yet)
    with the given number of qubits and classical bits.

    Args:
        num_qubits: The number of qubits in the circuit.
        num_classical_bits: The number of classical bits in the circuit.

    Returns:
        A quantum circuit with the given number of qubits and classical
        bits.
    """
    return QuantumCircuit(num_qubits, num_classical_bits)


def unitary_defined_entangled_cnot(
    rotations: Dict[str, float] = {
        "single_unitary_theta": 0,
        "single_unitary_phi": 0,
        "single_unitary_lam": 0,
        "single_unitary_gamma": 0,
        "controlled_unitary_theta": 0,
        "controlled_unitary_phi": 0,
        "controlled_unitary_lam": 0,
        "controlled_unitary_gamma": 0,
    },
    rotation_errors: Dict[str, float] = {
        "theta_error": 0,
        "phi_error": 0,
        "lam_error": 0,
        "gamma_error": 0,
    },
    circuit_depth: int = 1,
) -> QuantumCircuit:
    """
    Returns a two qubit circuit with a unitary gate followed by a
    controled unitary gate, with the given rotations and rotation
    errors, applied d times, where d is the circuit depth.

    Args:
        rotations:
            A dictionary of the form {
                'single_unitary_theta': float,
                'single_unitary_phi': float,
                'single_unitary_lam': float,
                'single_unitary_gamma': float,
                'controlled_unitary_theta': float,
                'controlled_unitary_phi': float,
                'controlled_unitary_lam': float,
                'controlled_unitary_gamma': float,
            }
        rotation_errors:
            A dictionary of the form {
                'theta_erorr': float,
                'phi_error': float,
                'lam_error': float,
            }
        circuit_depth:
            The number of repetitions of the unitary gate.

    Returns:
        A two qubit quantum circuit with the defined unitary gates
        applied d times

    """
    (
        single_unitary_theta,
        single_unitary_phi,
        single_unitary_lam,
        single_unitary_gamma,
    ) = (
        rotations["single_unitary_theta"],
        rotations["single_unitary_phi"],
        rotations["single_unitary_lam"],
        rotations["single_unitary_gamma"],
    )

    (
        controlled_unitary_theta,
        controlled_unitary_phi,
        controlled_unitary_lam,
        controlled_unitary_gamma,
    ) = (
        rotations["controlled_unitary_theta"],
        rotations["controlled_unitary_phi"],
        rotations["controlled_unitary_lam"],
        rotations["controlled_unitary_gamma"],
    )

    theta_error, phi_error, lam_error, gamma_error = (
        rotation_errors["theta_error"],
        rotation_errors["phi_error"],
        rotation_errors["lam_error"],
        rotation_errors["gamma_error"],
    )

    circuit = return_initial_quantum_circuit(
        num_qubits=2, num_classical_bits=2
    )

    for _ in range(circuit_depth):
        circuit.u(
            single_unitary_theta - theta_error,
            single_unitary_phi - phi_error,
            single_unitary_lam - lam_error,
            0,
        )
        circuit.cu(
            controlled_unitary_theta - theta_error,
            controlled_unitary_phi - phi_error,
            controlled_unitary_lam - lam_error,
            controlled_unitary_gamma - gamma_error,
            0,
            1,
        )

    circuit.measure([0, 1], [0, 1])

    return circuit


def single_qubit_with_unitary_operation_applied_d_times(
    circuit_depth: int = 1,
    measurmment_depth: int = 1,
    preparation_depth: int = 1,
    initlisation_array: np.ndarray = np.array(
        [
            1,
            0,
        ]
    ),
) -> QuantumCircuit:
    """
    Returns a single qubit circuit with a parameterised unitary gate
    applied d times, where d is the circuit depth.

    Args:
        circuit_depth:
            The number of repetitions of the unitary gate.
        measurmment_depth:
            The number of repetitions of the measurement.
        prepartition_depth:
            The number of repetitions of the prepartition.

    Returns:
        A single qubit quantum circuit with defined unitary gate
        applied d times.

    """

    theta, phi, lam = Parameter("theta"), Parameter("phi"), Parameter("lam")

    circuit = return_initial_quantum_circuit(
        num_qubits=preparation_depth,
        num_classical_bits=measurmment_depth * preparation_depth,
    )

    [
        circuit.initialize(
            initlisation_array,
            qubit_index,
        )
        for qubit_index in range(preparation_depth)
    ]

    for _ in range(circuit_depth):
        circuit.u(theta, phi, lam, 0)

    for prepartition_index in range(preparation_depth):
        circuit.measure
        circuit.measure(
            prepartition_index,
            range(
                (prepartition_index * measurmment_depth),
                (prepartition_index * measurmment_depth) + measurmment_depth,
            ),
        )

    return circuit
