from typing import Callable, Dict, List, Tuple, Union
from qiskit import execute, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.tools.monitor import job_monitor
from math import pi
from quantum_circuits_creator import (
    unitary_defined_entangled_cnot,
    single_qubit_with_unitary_operation_applied_d_times,
)
from utils.general_utils import (
    file_handler,
    sin_with_complex,
    cos_with_real,
    return_init_np_array_for_single_qubit,
)
from utils.ibmq_utils import return_live_and_fake_backend_with_shortest_queue
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.providers.aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from textwrap import wrap
from equations_for_prob_measuring_state import (
    probability_of_measuring_one_given_ground_state,
    probability_of_measuring_zero_given_ground_state,
    probability_of_measuring_one_given_excited_state,
    probability_of_measuring_zero_given_excited_state,
    whole_equation_for_probability_of_measuring_one,
    equation_for_kraus_probabilities,
    whole_equation_for_probability_of_measuring_one_no_complex,
    equation_for_kraus_probabilities_no_complex,
)
import openpyxl


EPSILON = 0.0001

SIMULATOR = []  # type: List[AerSimulator]

TOTAL_NUM_SHOTS = 1000 * 20

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

NUM_INIT_POINTS = 100

PROBABILITY_DISTRIBUTION = np.array([0.95, 0.05, 1.0, 0.25, 0.75, 1.0, 2.0])


def return_large_scale_prob_distro(
    theta_range: List[int], phi_range: List[int], workbook_name: str
) -> np.ndarray:
    workbook = openpyxl.load_workbook(workbook_name)
    sheet = workbook.active

    return np.array(
        [
            round(
                sheet.cell(
                    row=theta_index,
                    column=phi_index,
                ).value,
                4,
            )
            for theta_index in range(*theta_range)
            for phi_index in range(*phi_range)
        ]
        + [0] * (len(range(*theta_range)) * len(range(*phi_range)))
    )


LARGE_PROBABILITY_DISTRIBUTION = return_large_scale_prob_distro(
    theta_range=[3, 103],
    phi_range=[5, 6],
    workbook_name="results/probability_data_2.xlsx",
)


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
    )


def execute_and_return_counts_of_values_while_monitoring(
    circuit: Union[QuantumCircuit, List[QuantumCircuit]],
) -> Union[Dict[str, int], List[Dict[str, int]]]:
    """
    Executes the given circuit and returns the counts of the values.
    During execution the job is monitored. This function is designed for
    execution on a live qc

    Args:
        circuit: The circuit to execute.

    Returns:
        The counts of the values.
    """
    job = execute(circuit, backend=SIMULATOR[0], shots=TOTAL_NUM_SHOTS)

    job_monitor(job)

    return job.result().get_counts()


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


@njit(cache=True)
def calculate_average_variation_distance_between_distributions(
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
    return np.sum(np.abs(distribution_1 - distribution_2)) / len(
        distribution_1
    )


@njit(cache=True)
def calculate_largest_variation_distance_between_distributions(
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
    return np.max(np.abs(distribution_1 - distribution_2))


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
    noisy_simulation: bool = False, num_required_qubits: int = 1
) -> Tuple[Union[IBMQBackend, None], AerSimulator]:
    """
    Returns the live and equivalent fake backend.

    Args:
        noisy_simulation: Whether to use noisy simulation.
        num_required_qubits: the number of qubits required

    Returns:
        The live and equivalent fake backend.
    """
    if noisy_simulation:
        return return_live_and_fake_backend_with_shortest_queue(
            num_required_qubits=num_required_qubits
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
    measurement_depth = 19
    preparation_depth = 1
    shots = 10000

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


def generic_optimiser_function(
    wrapper_function: Callable, pbounds: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """
    Optimises a function using Bayesian Optimization Python Library.

    Args:
        wrapper_function: The function to optimise.
        pbounds: The bounds for the optimisation.

    Returns:
        The optimised parameters.
    """
    from bayes_opt import BayesianOptimization

    optimiser = BayesianOptimization(
        f=wrapper_function,
        pbounds=pbounds,
        random_state=1,
        verbose=1,
    )

    optimiser.maximize(
        init_points=NUM_INIT_POINTS,
        n_iter=NUM_INIT_POINTS * 5,
    )

    return optimiser.max


def error_equations_wrapper_function(
    epsilon_theta: float,
    mu_theta: float,
    nu_theta: float,
    tau_theta: float,
) -> float:
    """
    Wrapper function for the optimisation function.

    Args:
        epsilon: The epsilon value.
        mu: The mu value.
        nu: The nu value.
        tau: The tau value.

    Returns:
        Approximate solutions for error functions.
    """
    epsilon = sin_with_complex(epsilon_theta, 0)
    mu = sin_with_complex(mu_theta, 0)
    nu = cos_with_real(nu_theta)
    tau = cos_with_real(tau_theta)

    zero_prob_given_ground_state = (
        probability_of_measuring_zero_given_ground_state(
            mu=mu, epsilon=epsilon
        )
    )

    one_prob_given_ground_state = (
        probability_of_measuring_one_given_ground_state(nu=nu, epsilon=epsilon)
    )

    total_ground_state_probabilities = (
        zero_prob_given_ground_state + one_prob_given_ground_state
    )

    zero_prob_given_excited_state = (
        probability_of_measuring_zero_given_excited_state(mu=mu, tau=tau)
    )

    one_prob_given_excited_state = (
        probability_of_measuring_one_given_excited_state(nu=nu, tau=tau)
    )

    total_excited_state_probabilities = (
        zero_prob_given_excited_state + one_prob_given_excited_state
    )

    distribution_1 = np.array(
        [
            zero_prob_given_ground_state,
            one_prob_given_ground_state,
            total_ground_state_probabilities,
            zero_prob_given_excited_state,
            one_prob_given_excited_state,
            total_excited_state_probabilities,
            total_excited_state_probabilities
            + total_ground_state_probabilities,
        ]
    )

    return -(
        calculate_largest_variation_distance_between_distributions(
            distribution_1=distribution_1,
            distribution_2=PROBABILITY_DISTRIBUTION,
        )
        # + 2
        # * calculate_total_variation_distance_between_distributions(
        #     distribution_1=distribution_1, distribution_2=distribution_2
        # )
        # + calculate_average_variation_distance_between_distributions(
        #     distribution_1=distribution_1, distribution_2=distribution_2
        # )
    )


def big_error_equation_wrapper_function(
    eplison: float,
    nu: float,
    mu: float,
    tau: float,
    kxtheta: float,
    kytheta: float,
    kztheta: float,
    kitheta: float,
    kxphi: float,
    kyphi: float,
    kzphi: float,
    kiphi: float,
) -> float:
    """
    Wrapper function for the optimisation function.

    Args:
        epsilon: The epsilon value.
        mu: The mu value.
        nu: The nu value.
        tau: The tau value.

    Returns:
        Approximate solutions for error functions.
    """

    return -np.square(
        LARGE_PROBABILITY_DISTRIBUTION
        - np.array(
            [
                whole_equation_for_probability_of_measuring_one(
                    theta=theta_index * np.pi / 6,
                    phi=phi_index * np.pi / 6,
                    eplison=eplison,
                    nu=nu,
                    mu=mu,
                    tau=tau,
                    kxtheta=kxtheta,
                    kytheta=kytheta,
                    kztheta=kztheta,
                    kxphi=kxphi,
                    kyphi=kyphi,
                    kzphi=kzphi,
                )
                for theta_index in range(12)
                for phi_index in range(12)
            ]
            + [
                equation_for_kraus_probabilities(
                    theta=theta_index * np.pi / 6,
                    phi=phi_index * np.pi / 6,
                    eplison=eplison,
                    mu=mu,
                    kxtheta=kxtheta,
                    kytheta=kytheta,
                    kztheta=kztheta,
                    kitheta=kitheta,
                    kxphi=kxphi,
                    kyphi=kyphi,
                    kzphi=kzphi,
                    kiphi=kiphi,
                )
                for theta_index in range(12)
                for phi_index in range(12)
            ]
        )
    ).mean()


def big_error_equation_no_complex_wrapper_function(
    eplison: float,
    x: float,
    y: float,
    z: float,
) -> float:
    """
    Wrapper function for the optimisation function.

    Args:
        epsilon: The epsilon value.
        mu: The mu value.
        nu: The nu value.
        tau: The tau value.

    Returns:
        Approximate solutions for error functions.
    """
    num_theta = 100
    theta_interval = np.pi / (num_theta / 2) * 2

    return -np.mean(
        np.square(
            LARGE_PROBABILITY_DISTRIBUTION
            - np.array(
                [
                    whole_equation_for_probability_of_measuring_one_no_complex(
                        theta=theta_index * theta_interval,
                        eplison=eplison,
                        x=x,
                        y=y,
                        z=z,
                    )
                    for theta_index in range(num_theta)
                ]
                + [
                    equation_for_kraus_probabilities_no_complex(
                        theta=theta_index * theta_interval,
                        eplison=eplison,
                        x=x,
                        y=y,
                        z=z,
                    )
                    for theta_index in range(num_theta)
                ]
            )
        )
    )


def find_approximate_solutions_to_error_equations() -> Dict[str, float]:
    """
    Finds approximate solutions to the error equations.

    Returns:
        The approximate solutions to the error equations.
    """

    return generic_optimiser_function(
        wrapper_function=error_equations_wrapper_function,
        pbounds={
            "epsilon_theta": (0, pi),
            "mu_theta": (0, pi),
            "nu_theta": (0, pi),
            "tau_theta": (0, pi),
        },
    )


def find_approximate_solutions_to_big_error_equation() -> Dict[str, float]:
    """
    Finds approximate solutions to the error equations.

    Returns:
        The approximate solutions to the error equations.
    """

    return generic_optimiser_function(
        wrapper_function=big_error_equation_wrapper_function,
        pbounds={
            "eplison": (-pi / 2 - EPSILON, pi / 2 - EPSILON),
            "nu": (-pi / 2 - EPSILON, pi / 2 - EPSILON),
            "mu": (-pi / 2 - EPSILON, pi / 2 - EPSILON),
            "tau": (-pi / 2 - EPSILON, pi / 2 - EPSILON),
            "kxtheta": (0, 1.0 - EPSILON),
            "kytheta": (0, 1.0 - EPSILON),
            "kztheta": (0, 1.0 - EPSILON),
            "kitheta": (0, 1.0 - EPSILON),
            "kxphi": (0, 1.0 - EPSILON),
            "kyphi": (0, 1.0 - EPSILON),
            "kzphi": (0, 1.0 - EPSILON),
            "kiphi": (0, 1.0 - EPSILON),
        },
    )


def return_data_from_live_execution_over_range_of_circuits(
    number_of_theta_angles: int, number_of_phi_angles: int
) -> List[Dict[str, int]]:
    return execute_and_return_counts_of_values_while_monitoring(
        [
            single_qubit_with_unitary_operation_applied_d_times(
                circuit_depth=0,
                measurmment_depth=1,
                preparation_depth=1,
                initlisation_array=return_init_np_array_for_single_qubit(
                    theta=theta_index * np.pi / (number_of_theta_angles / 2),
                    phi=phi_index * np.pi / (number_of_phi_angles / 2),
                ),
            )
            for theta_index in range(number_of_theta_angles)
            for phi_index in range(number_of_phi_angles)
        ]
    )


def save_data_to_excel_sheet(
    data: List[Dict[str, int]],
    row_range: Tuple[int, int],
    column_range: Tuple[int, int],
    workbook: openpyxl.Workbook,
) -> None:
    """
    Saves data from data to cells in excel spreadsheet
    """

    sheet = workbook.active

    [
        sheet.cell(
            row=theta_index,
            column=phi_index,
            value=data.pop(0).get("0", 0) / TOTAL_NUM_SHOTS,
        )
        for theta_index in range(*row_range)
        for phi_index in range(*column_range)
    ]

    workbook.save("results/probability_data.xlsx")


def find_probability_data_for_various_qubit_initialisations() -> None:
    """
    Over a range of initialisation values find probability of measuring
    0 and record in excel spread sheet
    """
    starting_row, starting_column = 3, 5
    live_backend, fake_backend = return_live_and_equivalent_fake_backend(
        noisy_simulation=True, num_required_qubits=1
    )
    SIMULATOR.append(live_backend)

    total_num_theta_angles = 100
    total_num_phi_angles = 1

    workbook_name = "results/probability_data_2.xlsx"
    workbook = openpyxl.load_workbook(workbook_name)

    save_data_to_excel_sheet(
        data=return_data_from_live_execution_over_range_of_circuits(
            number_of_theta_angles=total_num_theta_angles,
            number_of_phi_angles=total_num_phi_angles,
        ),
        row_range=(starting_row, starting_row + total_num_theta_angles),
        column_range=(
            starting_column,
            starting_column + total_num_phi_angles,
        ),
        workbook=workbook,
    )

    workbook.save(workbook_name)


def draw_graphs_for_various_qubit_initialisations_probability_data() -> None:
    """
    Draw graphs for various qubit initialisations
    """
    starting_row, starting_column = 20, 3
    workbook = openpyxl.load_workbook("results/probability_data.xlsx")
    sheet = workbook.active
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    theta_axis_data = [
        value
        for theta_index in range(12)
        for value in [theta_index * np.pi / 6] * 12
    ]

    phi_axis_data = [phi_index * np.pi / 6 for phi_index in range(12)] * 12

    probability_axis_data = [
        sheet.cell(
            row=starting_row + theta_index,
            column=starting_column + phi_index,
        ).value
        for theta_index in range(12)
        for phi_index in range(12)
    ]

    ax.scatter(
        theta_axis_data,
        phi_axis_data,
        probability_axis_data,
    )

    ax.set_xlabel("theta")
    ax.set_ylabel("phi")
    ax.set_zlabel("probability of measuring 0")

    plt.savefig("results/probability_of_measuring_zero_data.png")


def draw_and_save_circuit_diagram(circuit: QuantumCircuit, path: str) -> None:
    """
    Simple function to mpl draw a circuit and save the png

    Args:
        circuit:
            The circuit to draw
        path:
            The path to save the image at
    """
    circuit.draw(output="mpl").savefig(path)


def main():
    """
    Main function.
    """

    print(
        generic_optimiser_function(
            wrapper_function=big_error_equation_no_complex_wrapper_function,
            pbounds={
                "eplison": (-pi / 2, pi / 2),
                "x": (0, 1.0),
                "y": (0, 1.0),
                "z": (0, 1.0),
            },
        )
    )


if __name__ == "__main__":
    main()
