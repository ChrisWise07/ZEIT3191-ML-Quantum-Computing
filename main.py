from collections import Counter
from typing import Dict
from qiskit import IBMQ, execute, Aer
from qiskit.circuit import QuantumCircuit
from matplotlib.pyplot import bar, savefig
from qiskit.tools.monitor import job_monitor
from quantum_circuits_creator import (
    simulated_entangled_noisy_cnot,
    entangled_cnot,
)
from utils.ibmq_utils import (
    find_ibmq_provider_with_enough_qubits_and_shortest_queue,
)
from utils.general_utils import file_handler
from json import dumps, load


def plot_ket_distribution(ket_distribution: dict) -> None:
    """
    Plots the given ket distribution.

    Args:
        ket_distribution:
            A dictionary mapping ket states to their frequency.
    """
    bar(
        list(ket_distribution.keys()),
        list(ket_distribution.values()),
        color="blue",
    )
    savefig("noise_probability_test.png")


def execute_circuit_record_result(
    circuit: QuantumCircuit, backend=Aer.get_backend("qasm_simulator")
) -> dict:
    """
    Executes the given circuit and returns the result.

    Args:
        circuit: The circuit to execute.
        backend: The backend to use.

    Returns:
        The result of the circuit.
    """
    return (
        execute(circuit, backend=backend, shots=1)
        .result()
        .get_counts(circuit)
        .most_frequent()
    )


def simulate_entangled_cnot() -> Dict:
    """
    Simulates the entangled CNOT gate and returns the resulting ket
    distribution.

    Returns:
        The ket distribution of the entangled CNOT gate.
    """
    return Counter(
        [
            execute_circuit_record_result(
                simulated_entangled_noisy_cnot({"h": 0.0, "cx": 0.0595})
            )
            for _ in range(2000)
        ]
    )


def record_results_from_circuit_on_live_qc_(circuit: QuantumCircuit):
    """
    Records the results of the given circuit on a live IBMQ quantum
    computer.

    Args:
        circuit: The circuit to execute.
    """
    job = execute(
        circuit,
        backend=find_ibmq_provider_with_enough_qubits_and_shortest_queue(),
        shots=2000,
    )
    job_monitor(job)

    file_handler(
        path="entangled_cnot_results.txt",
        mode="w",
        func=lambda f: f.write(
            dumps(job.result().get_counts(circuit), indent=4)
        ),
    )


def calculate_chi_squared_statistic_between_ket_distributions(
    ket_distribution_1: dict, ket_distribution_2: dict
) -> float:
    """
    Calculates the chi squared statistic between the given ket
    distributions.

    Args:
        ket_distribution_1:
            A dictionary mapping ket states to their frequency.
        ket_distribution_2:
            A dictionary mapping ket states to their frequency.

    Returns:
        The chi squared statistic between the given ket distributions.
    """
    chi_squared_statistic = 0
    for ket_state in ket_distribution_1:
        chi_squared_statistic += (
            ket_distribution_1.get(ket_state)
            - ket_distribution_2.get(ket_state)
        ) ** 2 / ket_distribution_1.get(ket_state)
    return chi_squared_statistic


def main():
    """
    Main function.
    """
    simulated_ket_distibution = simulate_entangled_cnot()
    actual_ket_distribution = file_handler(
        path="entangled_cnot_results.txt",
        mode="r",
        func=lambda f: load(f),
    )
    chi_squared_statistic = (
        calculate_chi_squared_statistic_between_ket_distributions(
            simulated_ket_distibution, actual_ket_distribution
        )
    )
    print(chi_squared_statistic)

    plot_ket_distribution(simulated_ket_distibution)


if __name__ == "__main__":
    main()
