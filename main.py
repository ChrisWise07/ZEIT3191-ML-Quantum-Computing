from curses import raw
from typing import Callable, Dict, List, Tuple, Union
from qiskit import execute, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.tools.monitor import job_monitor
from math import pi
from quantum_circuits_creator import (
    unitary_defined_entangled_cnot,
    single_qubit_with_unitary_operation_applied_d_times,
)
from utils.general_utils import file_handler
from json import dumps, load
from utils.ibmq_utils import return_live_and_fake_backend_with_shortest_queue
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.providers.aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from textwrap import wrap

# LIVE_QC_KET_DISTRIBUTIONS = file_handler(
#     path="entangled_cnot_results.txt",
#     mode="r",
#     func=lambda f: load(f),
# )

# LIVE_QC_CLEAN_KET_DISTRIBUTIONS = file_handler(
#     path="entangled_cnot_results_clean.txt",
#     mode="r",
#     func=lambda f: load(f),
# )

EPSILON = 0.0001

SIMULATOR = []  # type: List[AerSimulator]

TOTAL_NUM_SHOTS = 1024 * 2

NUMBER_OF_CYCLES = 12

CYLCE_LENGTH = 8

CIRCUIT_DEPTH = CYLCE_LENGTH * NUMBER_OF_CYCLES

CLEAN_RESULTS_CIRCUIT_DEPTH_CYCLE = np.array(
    [
        0,
        1.5,
        1.5,
        2.5,
        2.0,
        1.5,
        1.5,
        0.5,
    ]
    * NUMBER_OF_CYCLES
)

NUM_INIT_POINTS = 25


def plot_ket_distribution(ket_distribution: dict) -> None:
    """
    Plots the given ket distribution.

    Args:
        ket_distribution:
            A dictionary mapping ket states to their frequency.
    """
    plt.bar(
        list(ket_distribution.keys()),
        list(ket_distribution.values()),
        color="blue",
    )
    plt.savefig("noise_probability_test_custom_ugate.png")


def execute_and_return_counts_of_values(
    circuit: QuantumCircuit,
) -> Dict[str, int]:
    """
    Executes the given circuit and returns the counts of the values

    Args:
        circuit: The circuit to execute.

    Returns:
        The counts of the values.
    """

    return (
        SIMULATOR[0]
        .run(circuit, shots=TOTAL_NUM_SHOTS)
        .result()
        .get_counts(circuit)
        .items()
    )


def process_raw_measurement_data_by_finding_average_of_measurements(
    raw_measurement_data: List[Tuple[str, int]], measurement_depth: int = None
) -> Dict[str, int]:
    processed_measurment_data = {}
    for key, value in raw_measurement_data:
        processed_bit = "".join(
            [
                str(round(bit_group.count("1") / measurement_depth))
                for bit_group in wrap(key, measurement_depth)
            ]
        )

        if processed_measurment_data.get(processed_bit):
            processed_measurment_data[processed_bit] += value
        else:
            processed_measurment_data[processed_bit] = value

    return processed_measurment_data


def return_average_value_of_circuit(
    circuit: QuantumCircuit, measurement_depth: int
) -> float:
    """
    Returns the average final value for the given circuit meaured over
    the given number of shots.

    Args:
        circuit: The circuit to measure.
        measurement_depth: The number of times to measure a qubit

    Returns:
        The average final value for the given circuit measured over the
        given number of shots.
    """
    total = 0

    for (
        binary_value,
        count,
    ) in process_raw_measurement_data_by_finding_average_of_measurements(
        raw_measurement_data=execute_and_return_counts_of_values(circuit),
        measurement_depth=measurement_depth,
    ).items():

        total += int(binary_value, 2) * count

    return total / TOTAL_NUM_SHOTS


def record_results_from_circuit_on_live_qc(
    circuit: QuantumCircuit, backend: IBMQBackend = None, filename: str = None
):
    """
    Records the results of the given circuit on a live IBMQ quantum
    computer.

    Args:
        circuit: The circuit to execute.
        backend_name: The name of the backend to use.
        filename: The name of the file to save the results to.
    """
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

    tvd = 0
    for ket_state in expected_ket_distribution:
        num_observed_ket_state = observed_ket_distribution.get(ket_state)

        if num_observed_ket_state is None:
            num_observed_ket_state = 0
        tvd += abs(
            expected_ket_distribution.get(ket_state) - num_observed_ket_state
        )

    return 0.5 * tvd


@njit(cache=True)
def calculate_total_variation_distance_between_distributions(
    distribution_1: np.ndarray, distribution_2: np.ndarray
) -> float:
    """
    Calculates the total variation distance between the given distributions.

    Args:
        distribution_1:
            A list of values.
        distribution_2:
            A list of values.

    Returns:
        The total variation distance between the given distributions.
    """
    return 0.5 * np.sum(np.abs(distribution_1 - distribution_2))


def return_array_of_averages_over_circuit_depths(
    circuit_creator: Callable[[int, int], QuantumCircuit],
    unitary_rotations: Dict[str, float],
    measurement_depth: int,
    max_circuit_depth: int,
) -> List[float]:
    """
    Calculates the average of the results of the given circuit on the
    given backend.

    Args:
        circuit_creator: The circuit creator to use.
        unitary_rotations:
            A dictionary mapping unitary rotations to their parameters
            of the form (parameter, value).
        measurement_depth: The depth of the measurement circuit.
        max_circuit_depth: The maximum depth of the circuit.

    Returns:

    """
    return [
        return_average_value_of_circuit(circuit, measurement_depth)
        for circuit in [
            transpile(
                circuit_creator(depth, measurement_depth),
            ).assign_parameters(unitary_rotations.values(), inplace=False)
            for depth in range(2, max_circuit_depth, 2)
        ]
    ]


def optimise_gates_with_noise_wrapper_function(
    theta_error: float, phi_error: float, lam_error: float, gamma_error: float
) -> float:

    return -calculate_total_variation_distance_between_distributions(
        return_array_of_averages_over_circuit_depths(
            circuit_creator=unitary_defined_entangled_cnot,
            rotation_error_dict={
                "theta_error": theta_error,
                "phi_error": phi_error,
                "lam_error": lam_error,
                "gamma_error": gamma_error,
            },
        ),
        CLEAN_RESULTS_CIRCUIT_DEPTH_CYCLE,
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
            "theta_error": (0, pi - EPSILON),
            "phi_error": (0, 2 * pi - EPSILON),
            "lam_error": (0, 2 * pi - EPSILON),
            "gamma_error": (0, 2 * pi - EPSILON),
        },
        random_state=1,
        verbose=1,
    )

    optimiser.maximize(
        init_points=NUM_INIT_POINTS,
        n_iter=NUM_INIT_POINTS * 5,
    )

    return optimiser.max


def plot_line_graph_results(
    x_axis_values: List[float],
    y_axis_values: List[float],
    x_axis_label: str,
    y_axis_label: str,
    title: str,
):
    """
    Plots the given x and y axis values.

    Args:
        x_axis_values: The x axis values to plot.
        y_axis_values: The y axis values to plot.
        x_axis_label: The label for the x axis.
        y_axis_label: The label for the y axis.
    """
    plt.plot(
        x_axis_values,
        y_axis_values,
        color="red",
        marker="o",
    )
    plt.title(title, fontsize=14)
    plt.xlabel(x_axis_label, fontsize=14)
    plt.ylabel(y_axis_label, fontsize=14)
    plt.grid(True)
    return plt


def return_live_and_equivalent_fake_backend(
    noisy_simulation: bool = False,
) -> Tuple[Union[IBMQBackend, None], AerSimulator]:
    """
    Returns the live and equivalent fake backend.

    Args:
        noisy_simulation: Whether to use noisy simulation.

    Returns:
        The live and equivalent fake backend.
    """
    if noisy_simulation:
        return return_live_and_fake_backend_with_shortest_queue(
            num_required_qubits=2
        )

    return None, AerSimulator()


def find_coherent_rotation_errors():
    noisey_sim = True

    if noisey_sim:
        (
            backend,
            fake_backend,
        ) = return_live_and_fake_backend_with_shortest_queue(
            num_required_qubits=2
        )
        SIMULATOR.append(fake_backend)
    else:
        SIMULATOR.append(AerSimulator())

    results = optimise_euler_angles(optimise_gates_with_noise_wrapper_function)

    print(f"Final results: {results}")

    record_results_from_circuit_on_live_qc(
        unitary_defined_entangled_cnot(results["params"], 1),
        backend,
        filename="results/entangled_cnot_results_learnt_gate_with_noise_circuit_depth.txt",
    )


def find_measurement_error():
    circuit_depth = 1
    measurement_depth = 10
    shots = 10

    live_backend, fake_backend = return_live_and_equivalent_fake_backend(
        noisy_simulation=True
    )
    SIMULATOR.append(fake_backend)

    circuit = single_qubit_with_unitary_operation_applied_d_times(
        circuit_depth=circuit_depth, measurmment_depth=measurement_depth
    )

    if circuit_depth:
        circuit.assign_parameters(
            [pi, 0, pi],
            inplace=True,
        )

    total = 0

    for key, value in (
        SIMULATOR[0]
        .run(transpile(circuit, SIMULATOR[0]), shots=shots)
        .result()
        .get_counts(circuit)
        .items()
    ):
        print(f"{key}: {value}")
        total += (key.count("0") / measurement_depth) * value
        print(f"Total: {total}")

    print(f"Total: {total}")
    print(f"Average Error: {total / shots}")


def find_state_preparation_error():
    circuit_depth = 0
    measurement_depth = 21
    preparation_depth = 2
    shots = 1000

    live_backend, fake_backend = return_live_and_equivalent_fake_backend(
        noisy_simulation=True
    )
    SIMULATOR.append(fake_backend)

    circuit = single_qubit_with_unitary_operation_applied_d_times(
        circuit_depth=circuit_depth,
        measurmment_depth=measurement_depth,
        preparation_depth=preparation_depth,
    )

    if circuit_depth:
        circuit.assign_parameters(
            [pi, 0, pi],
            inplace=True,
        )

    # print(circuit)

    results = (
        SIMULATOR[0]
        .run(
            transpile(circuit, SIMULATOR[0], optimization_level=3), shots=shots
        )
        .result()
        .get_counts(circuit)
        .items()
    )

    # print(results)

    print(
        process_raw_measurement_data_by_finding_average_of_measurements(
            raw_measurement_data=results, measurement_depth=measurement_depth
        )
    )


def graph_average_value_over_circuit_depth():
    number_of_measurements = 21
    max_circuit_depth = 100
    error = 0.8

    live_backend, fake_backend = return_live_and_equivalent_fake_backend(
        noisy_simulation=True
    )
    SIMULATOR.append(fake_backend)

    list_of_averages = return_array_of_averages_over_circuit_depths(
        circuit_creator=single_qubit_with_unitary_operation_applied_d_times,
        unitary_rotations={
            "lambda": pi + error,
            "phi": 0,
            "theta": pi + error,
        },
        measurement_depth=number_of_measurements,
        max_circuit_depth=max_circuit_depth + 1,
    )

    plot_line_graph_results(
        x_axis_values=range(2, max_circuit_depth + 1, 2),
        y_axis_values=list_of_averages,
        x_axis_label="Circuit depth",
        y_axis_label="Average value",
        title="Average value over circuit depth",
    ).savefig(
        "results/average_value_over_circuit_depth_x_gate_every_second_more_measurements_with_error.png"
    )


def main():
    """
    Main function.
    """
    find_state_preparation_error()


if __name__ == "__main__":
    main()
