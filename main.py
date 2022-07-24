from collections import Counter
from qiskit import execute, Aer
from qiskit.circuit import QuantumCircuit
from matplotlib.pyplot import bar, savefig
from quantum_circuits_creator import noisy_cnot


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


def main():
    plot_ket_distribution(
        Counter(
            [
                execute_circuit_record_result(
                    noisy_cnot({"h": 0.1, "cx": 0.1})
                )
                for _ in range(1000)
            ]
        )
    )


if __name__ == "__main__":
    main()
