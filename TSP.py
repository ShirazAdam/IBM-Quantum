import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from itertools import permutations
from scipy.optimize import minimize, OptimizeResult
from typing import List, Tuple, Dict

print("\n" + "~" * 30)
print("Travelling Salesman Problem - Quantum Approach")
print("~" * 30 + "\n")
# Each row/column represents a city, values are distances between cities
# Diagonal should be 0 (distance from city to itself)

# 3 CITIES (9 qubits)
distance_matrix: ndarray = np.array([[0, 2, 9], [1, 0, 6], [15, 7, 0]])

# 4 CITIES (16 qubits)
# distance_matrix: ndarray = np.array([
#     [0, 2, 9, 10],
#     [1, 0, 6, 4],
#     [15, 7, 0, 8],
#     [6, 3, 12, 0]
# ])

# 5 CITIES (25 qubits)
# distance_matrix: ndarray = np.array([
#     [0, 10, 15, 20, 25],
#     [10, 0, 35, 25, 30],
#     [15, 35, 0, 30, 20],
#     [20, 25, 30, 0, 15],
#     [25, 30, 20, 15, 0]
# ])

n_cities: int = len(distance_matrix)
print(f"\nSolving TSP for {n_cities} cities")
print(f"Distance matrix:\n{distance_matrix}\n")

def tsp_to_ising(
    distance_matrix: ndarray, penalty: float = 30
) -> tuple[SparsePauliOp, int]:
    n: int = len(distance_matrix)
    num_qubits: int = n * n

    pauli_list: list[tuple[str, float]] = []

    # Objective is to minimise the distance
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != k:
                    next_j: int = (j + 1) % n
                    coeff: float = distance_matrix[i][k] / 4.0

                    qubit1: int = i * n + j
                    qubit2: int = k * n + next_j

                    if qubit1 < num_qubits and qubit2 < num_qubits and qubit1 != qubit2:
                        pauli_str: list[str] = ["I"] * num_qubits
                        pauli_str[qubit1] = "Z"
                        pauli_str[qubit2] = "Z"
                        pauli_list.append(("".join(reversed(pauli_str)), coeff))

    # Constraints with penalties
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                qubit1: int = i * n + j1
                qubit2: int = i * n + j2
                pauli_str: list[str] = ["I"] * num_qubits
                pauli_str[qubit1] = "Z"
                pauli_str[qubit2] = "Z"
                pauli_list.append(("".join(reversed(pauli_str)), penalty))

    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                qubit1: int = i1 * n + j
                qubit2: int = i2 * n + j
                pauli_str: list[str] = ["I"] * num_qubits
                pauli_str[qubit1] = "Z"
                pauli_str[qubit2] = "Z"
                pauli_list.append(("".join(reversed(pauli_str)), penalty))

    if not pauli_list:
        pauli_list.append(("I" * num_qubits, 0.0))

    hamiltonian: SparsePauliOp = SparsePauliOp.from_list(pauli_list)
    return hamiltonian, num_qubits

# Create Hamiltonian
hamiltonian: SparsePauliOp
num_qubits: int
hamiltonian, num_qubits = tsp_to_ising(distance_matrix, penalty=40)
print(f"Number of qubits: {num_qubits}")
print(f"Hamiltonian terms: {len(hamiltonian)}")

# Create a simple variational ansatz
def create_ansatz(num_qubits: int, params: ndarray) -> QuantumCircuit:
    qc: QuantumCircuit = QuantumCircuit(num_qubits)

    # Initial layer of RY rotations for quantum circuit
    param_idx: int = 0
    for i in range(num_qubits):
        qc.ry(params[param_idx], i)
        param_idx += 1

    # Entangling layer for quantum circuit
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Second layer of RY rotations for quantum circuit
    for i in range(num_qubits):
        qc.ry(params[param_idx], i)
        param_idx += 1

    return qc

num_params: int = num_qubits * 2
print(f"Circuit parameters: {num_params}")

# Energy evaluation function
def evaluate_energy(params: ndarray) -> float:
    # Create circuit
    qc: QuantumCircuit = create_ansatz(num_qubits, params)

    # Get statevector
    statevector: Statevector = Statevector(qc)

    # Calculate expectation value
    expectation: float = statevector.expectation_value(hamiltonian).real

    return expectation

print("\nRunning VQE optimisation...\n")

np.random.seed(42)
initial_params: ndarray = np.random.uniform(0, 2 * np.pi, num_params)

initial_energy: float = evaluate_energy(initial_params)
print(f"Initial energy: {initial_energy:.4f}")

# Run optimisation
result: OptimizeResult = minimize(
    evaluate_energy,
    initial_params,
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\nOptimisation complete!")
print(f"Final energy: {result.fun:.4f}")
print(f"Function evaluations: {result.nfev}")

print("\nSampling optimal circuit...\n")

optimal_qc: QuantumCircuit = create_ansatz(num_qubits, result.x)
optimal_qc.measure_all()

simulator: AerSimulator = AerSimulator()
transpiled_qc: QuantumCircuit = transpile(optimal_qc, simulator)
job = simulator.run(transpiled_qc, shots=2000)
counts: Dict[str, int] = job.result().get_counts()

# Sort by counts
sorted_counts: list[tuple[str, int]] = sorted(
    counts.items(), key=lambda x: x[1], reverse=True
)

print(f"\nTop 10 measurement results:")

def decode_bitstring(bitstring: str, n_cities: int) -> list[int]:
    tour: list[int] = []
    for j in range(n_cities):
        found: bool = False
        for i in range(n_cities):
            idx: int = i * n_cities + j
            if (
                idx < len(bitstring) and bitstring[-(idx + 1)] == "1"
            ):  # Reverse indexing
                tour.append(i)
                found = True
                break
        if not found:
            tour.append(-1)
    return tour

def calculate_tour_distance(tour: list[int], distance_matrix: ndarray) -> float:
    if -1 in tour or len(set(tour)) != len(tour) or len(tour) != len(distance_matrix):
        return float("inf")
    total: float = 0
    for i in range(len(tour)):
        total += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
    return total

best_valid_tour: list[int] | None = None
best_valid_distance: float = float("inf")

for idx, (bitstring, count) in enumerate(sorted_counts[:10]):
    tour: list[int] = decode_bitstring(bitstring, n_cities)
    distance: float = calculate_tour_distance(tour, distance_matrix)
    prob: float = count / 2000

    validity: str = "✓ Valid" if distance != float("inf") else "✗ Invalid"
    print(
        f"{idx + 1:2d}. {bitstring} | Tour: {tour} | Distance: {distance:>6.1f} | Shots: {count:4d} ({prob:.1%}) | {validity}"
    )

    if distance < best_valid_distance:
        best_valid_tour = tour
        best_valid_distance = distance

print("\n" + "=" * 60)
if best_valid_tour:
    print(f"Best quantum solution: {best_valid_tour}")
    print(f"Distance: {best_valid_distance:.1f}")
else:
    print("No valid tour found")
    print("Try increasing penalty or running with different random seed")

# Classical solution
print("\nClassical optimal solution (brute force):\n")

def solve_tsp_classical(distance_matrix: ndarray) -> tuple[list[int], float]:
    n: int = len(distance_matrix)
    cities: list[int] = list(range(n))
    min_distance: float = float("inf")
    best_tour: list[int] | None = None

    for perm in permutations(cities[1:]):
        tour: list[int] = [cities[0]] + list(perm)
        distance: float = sum(
            distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n)
        )
        if distance < min_distance:
            min_distance = distance
            best_tour = tour

    return best_tour, min_distance

classical_tour: list[int]
classical_distance: float
classical_tour, classical_distance = solve_tsp_classical(distance_matrix)
print(f"Optimal tour: {classical_tour}")
print(f"Optimal distance: {classical_distance}")

# Comparison
print("\nSummary:\n")
if best_valid_tour:
    gap: float = ((best_valid_distance - classical_distance) / classical_distance) * 100
    print(f"Quantum solution: {best_valid_tour} (distance: {best_valid_distance:.1f})")
    print(f"Classical solution: {classical_tour} (distance: {classical_distance})")
    print(f"Optimality gap: {gap:.1f}%")
    if gap == 0:
        print("Quantum algorithm found the optimal solution!")
    elif gap < 20:
        print("Quantum algorithm found a good solution!")
    else:
        print("Solution quality could be improved")
else:
    print(f"Classical solution: {classical_tour} (distance: {classical_distance})")
    print("Quantum algorithm did not find a valid solution")

print("\n" + "~" * 30)
print("Tips to improve results:")
print("Increase optimiser iterations (maxiter)")
print("Adjust penalty parameter (30-50)")
print("Add more circuit layers (deeper ansatz)")
print("Try different optimisation methods (SLSQP, L-BFGS-B)")
print("Run multiple times with different random seeds")
print("~" * 30)
