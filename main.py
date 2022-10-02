import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from qiskit.circuit import QuantumCircuit
from qiskit.tools.monitor import job_monitor
from math import pi
from quantum_circuits_creator import (
    single_qubit_with_unitary_operation_applied_d_times,
)
from utils.general_utils import (
    file_handler,
    return_init_np_array_for_single_qubit,
    calculate_mse_between_two_distributions,
)
from utils.ibmq_utils import return_live_and_fake_backend_with_shortest_queue
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
import matplotlib.pyplot as plt
import numpy as np
from equations_for_prob_measuring_state import (
    trig_probability_equation_for_measuring_zero_no_complex,
    trig_kraus_probability_bounding_equation,
    static_probability_equation_for_measuring_zero_no_complex,
    static_kraus_probability_bounding_equation,
    partial_solved_trig_equation_for_kraus_probabilities_no_complex,
    partial_solved_trig_probability_equation_for_measuring_zero_no_complex,
    state_dependent_small_theta_no_complex_kraus_bounding_equation,
    state_depedent_small_theta_no_complex_prob_equation,
)
import openpyxl
from numba import njit


EPSILON = 0.0001

SIMULATOR = []  # type: List[AerSimulator]

TOTAL_NUM_SHOTS = 20000

NUM_OF_THETA_VALUES = 100
THETA_VALUES = np.linspace(0, np.pi, NUM_OF_THETA_VALUES)


def return_large_scale_prob_distro(
    theta_range: List[int], phi_range: List[int], workbook_name: str
) -> np.ndarray:
    workbook = openpyxl.load_workbook(workbook_name)
    sheet = workbook.active

    return np.array(
        [
            sheet.cell(
                row=theta_index,
                column=phi_index,
            ).value
            for theta_index in range(*theta_range)
            for phi_index in range(*phi_range)
        ]
    )


EXPERIMENT_PROBABILITY_DISTRIBUTION = return_large_scale_prob_distro(
    theta_range=[3, 3 + NUM_OF_THETA_VALUES],
    phi_range=[5, 6],
    workbook_name="results/probability_data_3.xlsx",
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
    from qiskit import execute

    job = execute(circuit, backend=SIMULATOR[0], shots=TOTAL_NUM_SHOTS)

    job_monitor(job)

    return job.result().get_counts()


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


def generic_bayes_optimiser_function(
    wrapper_function: Callable,
    pbounds: Dict[str, Tuple[float, float]],
    init_points: int,
    n_iter: int,
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
        init_points=init_points,
        n_iter=n_iter,
    )

    return optimiser.max


def general_pso_optimisation_handler(
    num_dimensions: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    objective_func: Callable,
    objective_func_kwargs: Dict[str, Any],
    options: Dict[str, float] = {"c1": 0.5, "c2": 0.3, "w": 0.9},
    num_particles: int = 100,
    iterations: int = 10000,
    initial_position: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    General handler for running a PSO optimisation

    Args:
        num_dimensions:
            The number of dimensions to optimise over
        bounds:
            The bounds to optimise over
        objective_func:
            The objective function to optimise
    """
    import pyswarms as ps

    optimizer = ps.single.GlobalBestPSO(
        n_particles=num_particles,
        dimensions=num_dimensions,
        options=options,
        bounds=bounds,
        init_pos=initial_position,
    )

    cost, pos = optimizer.optimize(
        objective_func,
        iters=iterations,
        verbose=verbose,
        **objective_func_kwargs,
    )

    return cost, pos


@njit(cache=True)
def calculate_mse_between_experimental_and_simulated_data(
    simulated_data: np.ndarray,
    experimental_data: np.ndarray = EXPERIMENT_PROBABILITY_DISTRIBUTION,
) -> float:
    """
    Calculates the mean squared error between the experimental and simulated
    data.

    Args:
        experimental_data: The experimental data.
        simulated_data: The simulated data.

    Returns:
        The mean squared error between the experimental and simulated data.
    """
    return calculate_mse_between_two_distributions(
        dist_1=simulated_data, dist_2=experimental_data
    )


def generate_prob_data_over_theta(
    prob_measuring_zero_equation_func: Callable,
    kraus_prob_bounding_equation_func: Callable,
    args_for_equation_funcs: Tuple[float],
) -> np.ndarray:
    """
    Generates the probability data over theta.

    Args:
        prob_measuring_zero_equation_func: The function to calculate the
            probability of measuring zero.
        kraus_prob_bounding_equation: The function to calculate the
            probability of measuring zero.
        theta_interval: The interval between theta values.
    Returns:
        The probability data over theta.
    """

    return np.array(
        [
            prob_measuring_zero_equation_func(theta, *args_for_equation_funcs)
            + kraus_prob_bounding_equation_func(
                theta, *args_for_equation_funcs
            )
            for theta in THETA_VALUES
        ]
    )


def pso_wrapper_for_mse_prob_distro_difference_for_parameter_estimation(
    list_of_particle_params: np.ndarray,
    prob_measuring_zero_equation_func: Callable,
    kraus_prob_bounding_equation_func: Callable,
) -> np.ndarray:
    """
    Wrapper function using the PSO algorithm to find the minimum mse
    between measured probability distribution and calculated
    probabilities from static probability equation.

    Args:
        particle_params: The parameters for the particle.

    Returns:
        The mse between the measured probability distribution and the
        calculated probability distribution.
    """
    return np.array(
        [
            calculate_mse_between_experimental_and_simulated_data(
                simulated_data=generate_prob_data_over_theta(
                    prob_measuring_zero_equation_func=prob_measuring_zero_equation_func,
                    kraus_prob_bounding_equation_func=kraus_prob_bounding_equation_func,
                    args_for_equation_funcs=particle_params,
                )
            )
            for particle_params in list_of_particle_params
        ]
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
                    theta=theta,
                    phi=phi,
                ),
            )
            for theta in np.linspace(0, np.pi, number_of_theta_angles)
            for phi in np.linspace(0, 2 * np.pi, number_of_phi_angles)
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
    theta_values = np.linspace(0, np.pi, row_range[1] - row_range[0])

    [
        sheet.cell(
            row=theta_index,
            column=phi_index,
            value=data.pop(0).get("0", 0) / TOTAL_NUM_SHOTS,
        )
        and sheet.cell(
            row=theta_index,
            column=phi_index - 1,
            value=theta_values[theta_index - row_range[0]],
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
    print(f"Using {live_backend}")
    SIMULATOR.append(live_backend)

    total_num_theta_angles = 100
    total_num_phi_angles = 1

    workbook_name = "results/probability_data_3.xlsx"
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


def compare_bayes_pso_optimisation_for_various_equations() -> None:
    pso_num_particles = 50
    pso_num_iterations = 2500
    pso_6_dimension_bounds = (
        np.array([-pi / 2, -pi / 2, 0, 0, 0, 0]),
        np.array([pi / 2, pi / 2, 1, 1, 1, 1]),
    )
    pso_4_dimension_bounds = (
        np.array([-pi / 2, 0, 0, 0]),
        np.array([pi / 2, 1, 1, 1]),
    )

    print("\n\nPSO optimisation of static probability equation")
    general_pso_optimisation_handler(
        num_dimensions=6,
        bounds=pso_6_dimension_bounds,
        objective_func=pso_wrapper_for_mse_prob_distro_difference_for_parameter_estimation,
        objective_func_kwargs={
            "prob_measuring_zero_equation_func": (
                static_probability_equation_for_measuring_zero_no_complex
            ),
            "kraus_prob_bounding_equation_func": (
                static_kraus_probability_bounding_equation
            ),
        },
        num_particles=pso_num_particles,
        iterations=pso_num_iterations,
    )

    print("\n\nPSO optimisation of trig probability equation")
    general_pso_optimisation_handler(
        num_dimensions=6,
        bounds=pso_6_dimension_bounds,
        objective_func=pso_wrapper_for_mse_prob_distro_difference_for_parameter_estimation,
        objective_func_kwargs={
            "prob_measuring_zero_equation_func": (
                trig_probability_equation_for_measuring_zero_no_complex
            ),
            "kraus_prob_bounding_equation_func": (
                trig_kraus_probability_bounding_equation
            ),
        },
        num_particles=pso_num_particles,
        iterations=pso_num_iterations,
    )

    print(
        "\n\nPSO optimisation of trig probability equation with 4 dimensions"
    )
    general_pso_optimisation_handler(
        num_dimensions=4,
        bounds=pso_4_dimension_bounds,
        objective_func=pso_wrapper_for_mse_prob_distro_difference_for_parameter_estimation,
        objective_func_kwargs={
            "prob_measuring_zero_equation_func": (
                partial_solved_trig_equation_for_kraus_probabilities_no_complex
            ),
            "kraus_prob_bounding_equation_func": (
                partial_solved_trig_probability_equation_for_measuring_zero_no_complex
            ),
        },
        num_particles=pso_num_particles,
        iterations=pso_num_iterations,
    )

    print(
        "\n\nPSO optimisation of state dependent small theta probability equation"
    )
    general_pso_optimisation_handler(
        num_dimensions=4,
        bounds=pso_4_dimension_bounds,
        objective_func=pso_wrapper_for_mse_prob_distro_difference_for_parameter_estimation,
        objective_func_kwargs={
            "prob_measuring_zero_equation_func": (
                state_depedent_small_theta_no_complex_prob_equation
            ),
            "kraus_prob_bounding_equation_func": (
                state_dependent_small_theta_no_complex_kraus_bounding_equation
            ),
        },
        num_particles=pso_num_particles,
        iterations=pso_num_iterations,
    )

    # bayes_init_points = 25
    # bayes_num_iterations = 250
    # bayes_6_dimension_bounds = {
    #     "epsilon": (-pi / 2, pi / 2),
    #     "mu": (-pi / 2, pi / 2),
    #     "x": (0, 1),
    #     "y": (0, 1),
    #     "z": (0, 1),
    #     "l": (0, 1),
    # }
    # bayes_4_dimension_bounds = {
    #     "epsilon": (-pi / 2, pi / 2),
    #     "x": (0, 1),
    #     "y": (0, 1),
    #     "z": (0, 1),
    # }

    # print("\n\nBayesian optimisation of static probability equation")
    # print(
    #     generic_bayes_optimiser_function(
    #         wrapper_function=lambda **kwargs: -1
    #         * calculate_mse_between_experimental_and_static_equation_generated_data(
    #             **kwargs
    #         ),
    #         pbounds=bayes_6_dimension_bounds,
    #         init_points=bayes_init_points,
    #         n_iter=bayes_num_iterations,
    #     )
    # )

    # print("\n\nBayesian optimisation of trig probability equation")
    # print(
    #     generic_bayes_optimiser_function(
    #         wrapper_function=lambda **kwargs: -1
    #         * calculate_mse_between_experimental_and_trig_equation_generated_data(
    #             **kwargs
    #         ),
    #         pbounds=bayes_6_dimension_bounds,
    #         init_points=bayes_init_points,
    #         n_iter=bayes_num_iterations,
    #     )
    # )

    # print(
    #     "\n\nBayesian optimisation of trig probability equation with 4 dimensions"
    # )
    # print(
    #     generic_bayes_optimiser_function(
    #         wrapper_function=lambda **kwargs: -1
    #         * calculate_mse_between_experimental_and_partial_solved_trig_equation_generated_data(
    #             **kwargs
    #         ),
    #         pbounds=bayes_4_dimension_bounds,
    #         init_points=bayes_init_points,
    #         n_iter=bayes_num_iterations,
    #     )
    # )


def generate_init_array(
    linspace: np.ndarray,
    num_pso_particles: int,
    list_generating_func: Callable[[int], Tuple[float]],
) -> List[np.ndarray]:

    return [
        np.array(
            [
                list_generating_func(init_value)
                for _ in range(num_pso_particles)
            ]
        )
        for init_value in linspace
    ]


def produce_init_maps() -> None:
    import json

    pso_num_particles = 25
    pso_num_iterations = 1000
    pso_4_dimension_bounds = (
        np.array([-pi / 2, 0, 0, 0]),
        np.array([pi / 2, 1, 1, 1]),
    )

    num_data_points = 10
    num_repeat = 5

    epsilon_init_arrays = generate_init_array(
        np.linspace(-pi / 2, pi / 2, num_data_points),
        pso_num_particles,
        list_generating_func=lambda init_value: [
            init_value,
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
        ],
    )

    x_init_arrays = generate_init_array(
        np.linspace(0, 1, num_data_points),
        pso_num_particles,
        list_generating_func=lambda init_value: [
            random.uniform(-pi / 2, pi / 2),
            init_value,
            random.uniform(0, 1),
            random.uniform(0, 1),
        ],
    )

    y_init_arrays = generate_init_array(
        np.linspace(0, 1, num_data_points),
        pso_num_particles,
        list_generating_func=lambda init_value: [
            random.uniform(-pi / 2, pi / 2),
            random.uniform(0, 1),
            init_value,
            random.uniform(0, 1),
        ],
    )

    z_init_arrays = generate_init_array(
        np.linspace(0, 1, num_data_points),
        pso_num_particles,
        list_generating_func=lambda init_value: [
            random.uniform(-pi / 2, pi / 2),
            random.uniform(0, 1),
            random.uniform(0, 1),
            init_value,
        ],
    )

    init_map_with_init_arrays = [
        ("epsilon", {}, epsilon_init_arrays, 0),
        ("x", {}, x_init_arrays, 1),
        ("y", {}, y_init_arrays, 2),
        ("z", {}, z_init_arrays, 3),
    ]

    for name, init_map, init_arrays, value_pos in init_map_with_init_arrays:
        for init_array in init_arrays:
            cost_list, pos_list = [], []
            for _ in range(num_repeat):
                cost, pos = general_pso_optimisation_handler(
                    num_dimensions=4,
                    bounds=pso_4_dimension_bounds,
                    objective_func=pso_wrapper_for_mse_prob_distro_difference_for_parameter_estimation,
                    objective_func_kwargs={
                        "prob_measuring_zero_equation_func": (
                            partial_solved_trig_equation_for_kraus_probabilities_no_complex
                        ),
                        "kraus_prob_bounding_equation_func": (
                            partial_solved_trig_probability_equation_for_measuring_zero_no_complex
                        ),
                    },
                    num_particles=pso_num_particles,
                    iterations=pso_num_iterations,
                    initial_position=init_array,
                    verbose=False,
                )
                cost_list.append(cost)
                pos_list.append(pos)

            init_map[str(init_array[0][value_pos])] = (
                np.mean(cost_list),
                np.mean(pos_list, axis=0).tolist(),
            )

        file_handler(
            path=f"{name}_init_map.json",
            mode="w",
            func=lambda f: json.dump(init_map, f),
        )


def main():
    """
    Main function.
    """
    produce_init_maps()


if __name__ == "__main__":
    main()
