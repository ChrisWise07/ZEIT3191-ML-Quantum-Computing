from qiskit import execute, Aer
import matplotlib.pyplot as plt

from quantum_circuits_creator import noisy_cnot

if __name__ == "__main__":
    counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    backend = Aer.get_backend("qasm_simulator")

    for i in range(500):
        circuit = noisy_cnot({"h": 0.1, "cx": 0.1})
        counts[
            execute(circuit, backend=backend, shots=1)
            .result()
            .get_counts(circuit)
            .most_frequent()
        ] += 1

    plt.bar(list(counts.keys()), counts.values(), color="g")
    plt.savefig("noise_probability_test.png")
