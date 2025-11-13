from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import OptimizeResult, minimize

print("\nKnapsack Problem - Quantum Approach\n")
print("\nProblem: Select items to maximize value while staying")
print("within weight capacity.\n")

# Define the knapsack problem
# Each item has a value and weight

# OPTION 1: Simple 4-item problem
items: List[Dict[str, int]] = [
    {"value": 10, "weight": 5},
    {"value": 40, "weight": 4},
    {"value": 30, "weight": 6},
    {"value": 50, "weight": 3},
]
capacity: int = 10

# OPTION 2: 6-item problem (uncomment to use)
# items: List[Dict[str, int]] = [
#     {"value": 15, "weight": 12},
#     {"value": 10, "weight": 10},
#     {"value": 8, "weight": 8},
#     {"value": 9, "weight": 9},
#     {"value": 20, "weight": 5},
#     {"value": 5, "weight": 6}
# ]
# capacity: int = 26

# OPTION 3: Random problem
# n_items: int = 5
# np.random.seed(42)
# items: List[Dict[str, int]] = [
#     {"value": np.random.randint(5, 50), "weight": np.random.randint(1, 10)}
#     for _ in range(n_items)
# ]
# capacity: int = sum(item["weight"] for item in items) // 2

n_items: int = len(items)
values: ndarray = np.array([item["value"] for item in items])
weights: ndarray = np.array([item["weight"] for item in items])
total_weight: int = sum(weights)
total_value: int = sum(values)

print(f"Number of items: {n_items}")
print(f"Knapsack capacity: {capacity}")
print(f"Total weight of all items: {total_weight}")
print(f"Total value of all items: {total_value}\n")

print("Items:")
for i, item in enumerate(items):
    print(f"  Item {i}: Value = {item['value']}, Weight = {item['weight']}")


def visualize_solution(
    items: List[Dict[str, int]],
    selection: List[int],
    capacity: int,
    filename: str = "knapsack_solution.png",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    n: int = len(items)
    x: ndarray = np.arange(n)
    width: float = 0.35

    colors: List[str] = [
        "lightgreen" if selection[i] == 1 else "lightgray" for i in range(n)
    ]

    values_list: List[int] = [item["value"] for item in items]
    weights_list: List[int] = [item["weight"] for item in items]

    ax1.bar(x - width / 2, values_list, width, label="Value", alpha=0.8)
    ax1.bar(x + width / 2, weights_list, width, label="Weight", alpha=0.8)

    for i, color in enumerate(colors):
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=color)

    ax1.set_xlabel("Item Index", fontsize=12)
    ax1.set_ylabel("Value / Weight", fontsize=12)
    ax1.set_title("Items Overview (Selected items highlighted)", fontsize=14)
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    selected_indices: List[int] = [i for i, s in enumerate(selection) if s == 1]
    selected_values: List[int] = [values_list[i] for i in selected_indices]
    selected_weights: List[int] = [weights_list[i] for i in selected_indices]

    total_selected_value: int = sum(selected_values)
    total_selected_weight: int = sum(selected_weights)

    ax2.barh(
        ["Total Value", "Total Weight", "Capacity"],
        [total_selected_value, total_selected_weight, capacity],
        color=["green", "blue", "red"],
        alpha=0.7,
    )

    ax2.set_xlabel("Amount", fontsize=12)
    ax2.set_title(f"Solution Summary\nSelected items: {selected_indices}", fontsize=14)
    ax2.grid(axis="x", alpha=0.3)

    for i, v in enumerate([total_selected_value, total_selected_weight, capacity]):
        ax2.text(v + 1, i, str(v), va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Visualisation saved as '{filename}'")
    plt.close()


def knapsack_to_ising(
    values: ndarray, weights: ndarray, capacity: int, penalty: float = 50
) -> Tuple[SparsePauliOp, int]:
    n: int = len(values)
    num_qubits: int = n

    pauli_list: List[Tuple[str, float]] = []

    for i in range(n):
        pauli_str: List[str] = ["I"] * num_qubits
        pauli_str[i] = "Z"
        coeff: float = values[i] / 2.0
        pauli_list.append(("".join(reversed(pauli_str)), coeff))
        pauli_list.append(("I" * num_qubits, -values[i] / 2.0))

    for i in range(n):
        pauli_str: List[str] = ["I"] * num_qubits
        pauli_str[i] = "Z"
        slack: float = sum(weights) / 2.0 - capacity
        coeff: float = -penalty * weights[i] * slack
        pauli_list.append(("".join(reversed(pauli_str)), coeff))

    for i in range(n):
        for j in range(i, n):
            pauli_str: List[str] = ["I"] * num_qubits
            pauli_str[i] = "Z"
            pauli_str[j] = "Z"

            if i == j:
                coeff: float = penalty * (weights[i] ** 2) / 4.0
            else:
                coeff: float = penalty * weights[i] * weights[j] / 2.0

            pauli_list.append(("".join(reversed(pauli_str)), coeff))

    constant: float = penalty * (sum(weights) / 2.0 - capacity) ** 2
    pauli_list.append(("I" * num_qubits, constant))

    hamiltonian: SparsePauliOp = SparsePauliOp.from_list(pauli_list)
    return hamiltonian, num_qubits


# Create Hamiltonian
print("\nCreating Hamiltonian...\n")

hamiltonian: SparsePauliOp
num_qubits: int
hamiltonian, num_qubits = knapsack_to_ising(values, weights, capacity, penalty=100)
print(f"Number of qubits: {num_qubits}")
print(f"Hamiltonian terms: {len(hamiltonian)}")


def create_ansatz(num_qubits: int, params: ndarray, depth: int = 2) -> QuantumCircuit:
    qc: QuantumCircuit = QuantumCircuit(num_qubits)

    param_idx: int = 0

    for layer in range(depth):
        for i in range(num_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1

        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        for i in range(num_qubits):
            qc.rz(params[param_idx], i)
            param_idx += 1

    for i in range(num_qubits):
        qc.ry(params[param_idx], i)
        param_idx += 1

    return qc


depth: int = 2
num_params: int = num_qubits * (2 * depth + 1)
print(f"Circuit depth: {depth}")
print(f"Circuit parameters: {num_params}")


def evaluate_energy(params: ndarray) -> float:
    qc: QuantumCircuit = create_ansatz(num_qubits, params, depth)
    statevector: Statevector = Statevector(qc)
    expectation: float = statevector.expectation_value(hamiltonian).real
    return expectation


print("\nRunning VQE optimization...\n")

np.random.seed(42)
initial_params: ndarray = np.random.uniform(0, 2 * np.pi, num_params)
initial_energy: float = evaluate_energy(initial_params)
print(f"Initial energy: {initial_energy:.4f}")

result: OptimizeResult = minimize(
    evaluate_energy,
    initial_params,
    method="COBYLA",
    options={"maxiter": 150, "disp": True},
)

print("\nOptimisation complete!\n")
print(f"Final energy: {result.fun:.4f}")
print(f"Function evaluations: {result.nfev}")

print("\nSampling optimal circuit...\n")

optimal_qc: QuantumCircuit = create_ansatz(num_qubits, result.x, depth)
optimal_qc.measure_all()

simulator: AerSimulator = AerSimulator()
transpiled_qc: QuantumCircuit = transpile(optimal_qc, simulator)
job = simulator.run(transpiled_qc, shots=2000)
counts: Dict[str, int] = job.result().get_counts()

sorted_counts: List[Tuple[str, int]] = sorted(
    counts.items(), key=lambda x: x[1], reverse=True
)

print("\nTop 10 measurement results:\n")


def decode_selection(bitstring: str) -> List[int]:
    return [int(b) for b in bitstring[::-1]]


def evaluate_solution(
    selection: List[int], values: ndarray, weights: ndarray, capacity: int
) -> Tuple[int, int, bool]:
    total_value: int = sum(values[i] * selection[i] for i in range(len(selection)))
    total_weight: int = sum(weights[i] * selection[i] for i in range(len(selection)))
    is_valid: bool = total_weight <= capacity
    return total_value, total_weight, is_valid


best_valid_selection: List[int] | None = None
best_valid_value: int = 0

for idx, (bitstring, count) in enumerate(sorted_counts[:10]):
    selection: List[int] = decode_selection(bitstring)
    total_value: int
    total_weight: int
    is_valid: bool
    total_value, total_weight, is_valid = evaluate_solution(
        selection, values, weights, capacity
    )
    prob: float = count / 2000

    validity: str = "Valid" if is_valid else "Overweight"
    selected_items: List[int] = [i for i, s in enumerate(selection) if s == 1]

    print(
        f"{idx + 1:2d}. {bitstring} | Items: {selected_items} | "
        f"Value: {total_value:3d} | Weight: {total_weight:3d}/{capacity} | "
        f"Shots: {count:4d} ({prob:.1%}) | {validity}"
    )

    if is_valid and total_value > best_valid_value:
        best_valid_value = total_value
        best_valid_selection = selection

if best_valid_selection:
    selected_items: List[int] = [
        i for i, s in enumerate(best_valid_selection) if s == 1
    ]
    total_weight: int = sum(
        weights[i] * best_valid_selection[i] for i in range(len(best_valid_selection))
    )

    print("Best quantum solution found:")
    print(f"Selected items: {selected_items}")
    print(f"Total value: {best_valid_value}")
    print(f"Total weight: {total_weight}/{capacity}")
    print("Items details:")
    for i in selected_items:
        print(f" Item {i}: Value = {values[i]}, Weight = {weights[i]}")

    visualize_solution(items, best_valid_selection, capacity)
else:
    print("No valid solution found")
    print("Try increasing penalty or more optimiser iterations")

print("\nClassical optimal solution (Dynamic Programming):\n")


def solve_knapsack_classical(
    values: ndarray, weights: ndarray, capacity: int
) -> Tuple[List[int], int]:
    n: int = len(values)

    dp: ndarray = np.zeros((n + 1, capacity + 1), dtype=int)

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]

            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    max_value: int = dp[n][capacity]
    selection: List[int] = [0] * n
    w: int = capacity

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selection[i - 1] = 1
            w -= weights[i - 1]

    return selection, max_value


classical_selection: List[int]
classical_value: int
classical_selection, classical_value = solve_knapsack_classical(
    values, weights, capacity
)

selected_items_classical: List[int] = [
    i for i, s in enumerate(classical_selection) if s == 1
]
classical_weight: int = sum(
    weights[i] * classical_selection[i] for i in range(len(classical_selection))
)

print(f"Optimal selection: {selected_items_classical}")
print(f"Maximum value: {classical_value}")
print(f"Total weight: {classical_weight}/{capacity}")
print("Items details:")
for i in selected_items_classical:
    print(f" Item {i}: Value = {values[i]}, Weight = {weights[i]}")

print("\nSummary:\n")

if best_valid_selection:
    gap: float = ((classical_value - best_valid_value) / classical_value) * 100
    print(f"Quantum solution value: {best_valid_value}")
    print(f"Classical optimal value: {classical_value}")
    print(f"Gap: {gap:.1f}%")

    if best_valid_value == classical_value:
        print("Quantum algorithm found the optimal solution!")
    elif gap < 20:
        print("Quantum algorithm found a good approximation!")
    else:
        print("Solution can be improved")
else:
    print(f"Classical optimal value: {classical_value}")
    print("Quantum algorithm did not find a valid solution")

print("\n" + "~" * 30)
print("Adjust penalty parameter (50-200)")
print("Increase circuit depth (depth=3)")
print("Increase optimiser iterations (maxiter=200-300)")
print("Try different random seeds")
print("For 6+ items, expect longer runtime")
print("~" * 30)
