from qiskit import IBMQ
from typing import Any, Tuple
from qiskit.providers.aer.noise import NoiseModel


def find_ibmq_provider_with_enough_qubits_and_shortest_queue(
    num_required_qubits: int = 5,
) -> Any:
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


def return_objects_for_noisy_simulation() -> Tuple[NoiseModel, dict, dict]:
    """
    Returns objects for noisy simulation.

    Returns:
        noise_model: The noise model to use.
        coupling_map: The coupling map to use.
        basis_gates: The basis gates to use.

    """
    backend = IBMQ.load_account().get_backend("ibm_nairobi")
    noise_model = NoiseModel.from_backend(backend)

    return (
        noise_model,
        backend.configuration().coupling_map,
        noise_model.basis_gates,
    )
