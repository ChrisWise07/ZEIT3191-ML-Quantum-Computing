from qiskit import IBMQ
from typing import Dict, List, Tuple, Union
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError


def find_ibmq_provider_with_enough_qubits_and_shortest_queue(
    num_required_qubits: int = 5,
) -> IBMQBackend:
    """
    Find IBMQ provider with required number of qubits and the shortest
    queue.

    Returns:
        Name of IBMQ provider with the shortest queue.
    """
    shortest_queue_length = float("inf")
    backend_with_shortest_queue = None

    IBMQ.load_account()

    provider = IBMQ.get_provider("ibm-q")

    for backend in provider.backends():
        try:
            num_qubits = len(backend.properties().qubits)
        except AttributeError:
            # provider is simulator
            continue

        if num_qubits and num_qubits >= num_required_qubits:
            queue_length = backend.status().pending_jobs
            if queue_length < shortest_queue_length:
                shortest_queue_length = queue_length
                backend_with_shortest_queue = backend

    return backend_with_shortest_queue


def return_live_and_fake_backend_with_shortest_queue(
    num_required_qubits: int = 5,
) -> Tuple[IBMQBackend, AerSimulator]:
    """
    Find IBMQ provider with required number of qubits and the shortest
    queue.

    Returns:
        Name of IBMQ provider with the shortest queue.
    """
    backend_num_qubit_map = {}

    IBMQ.load_account()

    provider = IBMQ.get_provider("ibm-q")

    for backend in provider.backends():
        try:
            num_qubits = len(backend.properties().qubits)
            if num_qubits > num_required_qubits:
                backend_num_qubit_map.update(
                    {backend: backend.status().pending_jobs}
                )
        except AttributeError:
            # provider is simulator
            continue

    for backend in {
        k: v
        for k, v in sorted(
            backend_num_qubit_map.items(), key=lambda item: item[1]
        )
    }.keys():
        try:
            fake_backend = AerSimulator.from_backend(
                FakeProvider().get_backend(
                    f"fake_{backend.name().split('_')[-1]}"
                )
            )
            return backend, fake_backend
        except QiskitBackendNotFoundError:
            # no fake backend found
            continue


def return_objects_for_noisy_simulation(
    backend,
) -> Dict[str, Union[NoiseModel, List[str], List[Tuple[int, int]]]]:
    """
    Returns objects for noisy simulation.

    Args:
        backend_name: Name of the backend.

    Returns:
        A dictionary with the following keys:
            noise_model: The noise model to use.
            coupling_map: The coupling map to use.
            basis_gates: The basis gates to use.
    """
    noise_model = NoiseModel.from_backend(backend)
    noise_model.add_basis_gates(["u", "cx"])

    return {
        "noise_model": noise_model,
        "coupling_map": backend.configuration().coupling_map,
        "basis_gates": noise_model.basis_gates,
    }
