from qiskit import QuantumCircuit
from random import random
from qiskit.circuit import Gate


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


def simulated_entangled_noisy_cnot(
    theta: float, phi: float, lam: float
) -> QuantumCircuit:
    """
    Returns a circuit with a noisy CNOT gate.

    Args:
        noise_probability:
            The probability of performing a bit flip after each gate is
            applied

    Returns:
        A quantum circuit with a noisy CNOT gate.
    """
    circuit = return_initial_quantum_circuit(
        num_qubits=2, num_classical_bits=2
    )
    circuit.h(0)
    circuit.u(theta, phi, lam, 0).inverse()
    circuit.cx(0, 1)
    circuit.u(theta, phi, lam, 0)
    circuit.u(theta, phi, lam, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def apply_noise_to_circuit(
    circuit: QuantumCircuit,
    qubit_ids: list,
    gate: Gate,
    noise_probability: float,
) -> QuantumCircuit:
    """
    Returns a circuit with specified gate applied to qubits with probability
    equal to the noise probability.

    Args:
        circuit: The circuit to apply the noise to.
        qubit_ids: The qubits to apply the noise to.
        noise_probability:
            The probability of performing a "bit flip" after gate is
            applied

    Returns:
        A quantum circuit with a probability of extra gate applied to
        simulate noise.
    """
    if random() < noise_probability:
        circuit.append(gate, qubit_ids)


def entangled_cnot() -> QuantumCircuit:
    """
    Returns a circuit with CNOT gate

    Returns:
        A quantum circuit with a noisy CNOT gate.
    """
    circuit = return_initial_quantum_circuit(
        num_qubits=2, num_classical_bits=2
    )
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def clean_entangled_cnot(
    theta: float, phi: float, lam: float
) -> QuantumCircuit:
    """
    Returns a circuit with CNOT gate

    Returns:
        A quantum circuit with a clean CNOT gate.
    """
    circuit = return_initial_quantum_circuit(
        num_qubits=2, num_classical_bits=2
    )
    circuit.h(0)
    circuit.u(theta, phi, lam, 0).inverse()
    circuit.cx(0, 1)
    circuit.u(theta, phi, lam, 0).inverse()
    circuit.u(theta, phi, lam, 1).inverse()
    circuit.measure([0, 1], [0, 1])
    return circuit
