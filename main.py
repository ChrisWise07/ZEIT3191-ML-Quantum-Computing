from collections import Counter
from typing import Dict
from qiskit import execute, Aer
from qiskit.circuit import QuantumCircuit
from matplotlib.pyplot import bar, savefig
from qiskit.tools.monitor import job_monitor
from math import pi
from quantum_circuits_creator import (
    simulated_entangled_noisy_cnot,
    entangled_cnot,
    clean_entangled_cnot,
)
from utils.ibmq_utils import (
    find_ibmq_provider_with_enough_qubits_and_shortest_queue,
    return_objects_for_noisy_simulation,
)
from utils.general_utils import file_handler
from json import dumps, load

LIVE_QC_KET_DISTRIBUTIONS = file_handler(
    path="entangled_cnot_results_clean.txt",
    mode="r",
    func=lambda f: load(f),
)

EPSILON = 0.000001

NOISE_MODEL, COUPLING_MAP, BASIS_GATES = return_objects_for_noisy_simulation()

SIMULATOR_BACKEND = Aer.get_backend("qasm_simulator")


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


def execute_circuit_record_result(circuit: QuantumCircuit) -> dict:
    """
    Executes the given circuit and returns the result.

    Args:
        circuit: The circuit to execute.
        backend: The backend to use.

    Returns:
        The result of the circuit.
    """
    # redfine noise model here
    # noise_model = NoiseModel()
    # noise_model.add_all_qubit_quantum_error(iswap_error, 'iswap')
    # noise_model.add_basis_gates(['unitary'])
    # print(noise_model.basis_gates)

    return (
        execute(
            circuit,
            backend=SIMULATOR_BACKEND,
            coupling_map=COUPLING_MAP,
            basis_gates=BASIS_GATES,
            noise_model=NOISE_MODEL,
            shots=10000,
        )
        .result()
        .get_counts(circuit)
    )


def simulate_entangled_cnot(theta: float, phi: float, lam: float) -> Dict:
    """
    Simulates the entangled CNOT gate and returns the resulting ket
    distribution.

    Args:


    Returns:
        The ket distribution of the entangled CNOT gate.
    """
    return execute_circuit_record_result(
        simulated_entangled_noisy_cnot(theta, phi, lam)
    )


def record_results_from_circuit_on_live_qc(circuit: QuantumCircuit):
    """
    Records the results of the given circuit on a live IBMQ quantum
    computer.

    Args:
        circuit: The circuit to execute.
    """
    job = execute(
        circuit,
        backend=find_ibmq_provider_with_enough_qubits_and_shortest_queue(),
        shots=10000,
    )
    job_monitor(job)

    file_handler(
        path="entangled_cnot_results_clean.txt",
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
    tvd = 0
    for ket_state in expected_ket_distribution:
        num_observed_ket_state = observed_ket_distribution.get(ket_state)

        if num_observed_ket_state is None:
            num_observed_ket_state = 0

        tvd += abs(
            expected_ket_distribution.get(ket_state) - num_observed_ket_state
        )

    return 0.5 * tvd


def quantum_noise_optimisation_wrapper_function(
    theta: float, phi: float, lam: float
) -> float:

    return -calculate_total_variation_distance_between_ket_distributions(
        simulate_entangled_cnot(theta, phi, lam), LIVE_QC_KET_DISTRIBUTIONS
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


def optimise_noise_parameters():
    """
    Optimises the noise parameters.
    """
    from bayes_opt import BayesianOptimization

    optimiser = BayesianOptimization(
        f=quantum_noise_optimisation_wrapper_function,
        pbounds={
            "theta": (0, pi),
            "phi": (0, 2 * pi - EPSILON),
            "lam": (0, 2 * pi - EPSILON),
        },
        random_state=1,
    )

    optimiser.maximize(
        init_points=250,
        n_iter=500,
    )

    print(optimiser.max)


def optimise_euler_angles_to_minimise_noise():
    """
    Optimises the noise parameters.
    """
    from bayes_opt import BayesianOptimization

    optimiser = BayesianOptimization(
        f=quantum_noise_optimisation_wrapper_function,
        pbounds={
            "theta": (0, pi),
            "phi": (0, 2 * pi - EPSILON),
            "lam": (0, 2 * pi - EPSILON),
        },
        random_state=1,
    )

    optimiser.maximize(
        init_points=250,
        n_iter=500,
    )

    print(optimiser.max)


def main():
    """
    Main function.
    """
    # optimise_noise_parameters()
    optimise_euler_angles_to_minimise_noise()


if __name__ == "__main__":
    main()
