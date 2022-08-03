from typing import Callable, Dict, List, Tuple
from qiskit import execute, IBMQ, transpile
from qiskit.circuit import QuantumCircuit
from matplotlib.pyplot import bar, savefig
from qiskit.tools.monitor import job_monitor
from math import pi
from quantum_circuits_creator import (
    simulated_entangled_noisy_cnot,
    entangled_cnot,
    clean_entangled_cnot,
    unitary_defined_entangled_cnot,
)

from utils.general_utils import file_handler
from json import dumps, load
from utils.ibmq_utils import return_live_and_fake_backend_with_shortest_queue
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend


LIVE_QC_KET_DISTRIBUTIONS = file_handler(
    path="entangled_cnot_results.txt",
    mode="r",
    func=lambda f: load(f),
)

LIVE_QC_CLEAN_KET_DISTRIBUTIONS = file_handler(
    path="entangled_cnot_results_clean.txt",
    mode="r",
    func=lambda f: load(f),
)

EPSILON = 0.000001

SIMULATED_NOISE_OBJECTS = {
    "noise_model": None,
    "coupling_map": None,
    "basis_gates": None,
}

SIMULATOR = []  # type: List[AerSimulator]


def plot_ket_distribution(ket_distribution: dict) -> None:
    """
    Plots the given ket distribution.

    Args:
        ket_distribution:
            A dictionary mapping ket states to their frequency.
    """
    print(ket_distribution)
    bar(
        list(ket_distribution.keys()),
        list(ket_distribution.values()),
        color="blue",
    )
    savefig("noise_probability_test_custom_ugate.png")


def return_results_of_noisy_circuit_execution(circuit: QuantumCircuit) -> dict:
    """
    Executes the given circuit and returns the result.

    Args:
        circuit: The circuit to execute.
        backend: The backend to use.

    Returns:
        The result of the circuit.
    """

    # Execute noisy simulation and get counts
    return (
        SIMULATOR[0]
        .run(transpile(circuit, SIMULATOR[0]), shots=10000)
        .result()
        .get_counts(0)
    )


def return_results_of_clean_circuit_execution(circuit: QuantumCircuit) -> dict:
    """
    Executes the given circuit and returns the result.

    Args:
        circuit: The circuit to execute.
        backend: The backend to use.

    Returns:
        The result of the circuit.
    """
    return (
        execute(
            circuit,
            backend=SIMULATOR_BACKEND,
            shots=10000,
        )
        .result()
        .get_counts(circuit)
    )


def simulate_circuit(
    theta: float,
    phi: float,
    lam: float,
    execution_function: Callable,
    circuit_creator: Callable,
) -> Dict:
    """
    Simulates the entangled CNOT gate and returns the resulting ket
    distribution.

    Args:


    Returns:
        The ket distribution of the entangled CNOT gate.
    """
    return execution_function(circuit_creator(theta, phi, lam))


def record_results_from_circuit_on_live_qc(
    circuit: QuantumCircuit, backend: IBMQBackend = None, filename: str = None
):
    """
    Records the results of the given circuit on a live IBMQ quantum
    computer.

    Args:
        circuit: The circuit to execute.
        backend_name: The name of the backend to use.
    """
    # if backend_name is None:
    #     backend_name = (
    #         find_ibmq_provider_with_enough_qubits_and_shortest_queue()
    #     )

    job = execute(
        circuit,
        backend=backend,
        shots=10000,
    )
    job_monitor(job)

    file_handler(
        path=filename,
        mode="w",
        func=lambda f: f.write(
            dumps(job.result().get_counts(circuit), indent=4)
        ),
    )


def calculate_chi_squared_statistic_between_ket_distributions(
    observed_ket_distribution: dict, expected_ket_distribution: dict
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
    for ket_state in expected_ket_distribution:
        num_expected_ket_state = expected_ket_distribution.get(ket_state)
        num_observed_ket_state = observed_ket_distribution.get(ket_state)

        if num_observed_ket_state is None:
            num_observed_ket_state = 0

        chi_squared_statistic += (
            num_observed_ket_state - num_expected_ket_state
        ) ** 2 / (num_expected_ket_state + EPSILON)

    return chi_squared_statistic


def calculate_total_variation_distance_between_ket_distributions(
    observed_ket_distribution: dict, expected_ket_distribution: dict
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
    weights = {"00": 1.0, "01": 1.0, "10": 1.0, "11": 1.0}

    tvd = 0
    for ket_state in expected_ket_distribution:
        num_observed_ket_state = observed_ket_distribution.get(ket_state)
        ket_weight = weights.get(ket_state)

        if num_observed_ket_state is None:
            num_observed_ket_state = 0
        tvd += ket_weight * abs(
            expected_ket_distribution.get(ket_state) - num_observed_ket_state
        )

    return 0.5 * tvd


def optimise_hgate_wrapper_function(
    theta: float, phi: float, lam: float
) -> float:

    return -calculate_total_variation_distance_between_ket_distributions(
        simulate_circuit(
            theta,
            phi,
            lam,
            return_results_of_clean_circuit_execution,
            unitary_defined_entangled_cnot,
        ),
        LIVE_QC_CLEAN_KET_DISTRIBUTIONS,
    )


def optimise_hgate_with_noise_wrapper_function(
    theta: float, phi: float, lam: float
) -> float:

    return -calculate_total_variation_distance_between_ket_distributions(
        simulate_circuit(
            theta,
            phi,
            lam,
            return_results_of_noisy_circuit_execution,
            unitary_defined_entangled_cnot,
        ),
        LIVE_QC_CLEAN_KET_DISTRIBUTIONS,
    )


def print_circuit_to_file(circuit: QuantumCircuit):
    """
    Prints the given circuit to a file.

    Args:
        circuit: The circuit to print.
    """
    file_handler(
        path="entangled_cnot_circuit.txt",
        mode="w",
        func=lambda f: f.write(circuit.draw(output="text")),
    )


def optimise_euler_angles(
    wrapper_function: Callable,
) -> Tuple[float, float, float]:
    """
    Optimises the noise parameters.
    """
    from bayes_opt import BayesianOptimization

    optimiser = BayesianOptimization(
        f=wrapper_function,
        pbounds={
            "theta": (0, pi),
            "phi": (0, 2 * pi - EPSILON),
            "lam": (0, 2 * pi - EPSILON),
        },
        random_state=1,
        verbose=1,
    )

    optimiser.maximize(
        init_points=300,
        n_iter=300,
    )

    return optimiser.max


def main():
    """
    Main function.
    """
    backend, fake_backend = return_live_and_fake_backend_with_shortest_queue(
        num_required_qubits=2
    )
    print(backend.name())
    print(fake_backend.name())

    SIMULATOR.append(fake_backend)

    results = optimise_euler_angles(optimise_hgate_with_noise_wrapper_function)

    print(f"Final results: {results}")

    record_results_from_circuit_on_live_qc(
        unitary_defined_entangled_cnot(**results["params"]),
        backend,
        filename="entangled_cnot_results_learnt_gate_with_noise.txt",
    )


if __name__ == "__main__":
    main()
