from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import OptimizeResult, minimize

print("~" * 30)
print("Max-Cut Problem - Quantum Approach")
print("~" * 30)
print("\nProblem: Partition graph nodes into two sets to maximize")
print("the number of edges between the sets.\n")

# Define the graph as an adjacency matrix
# Example: 4-node graph
# You can also define it as a list of edges: [(0,1), (1,2), etc.]

# OPTION 1: Simple 4-node graph
adjacency_matrix: ndarray = np.array(
    [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
)

# OPTION 2: Larger 6-node graph (uncomment to use)
# adjacency_matrix: ndarray = np.array([
#     [0, 1, 1, 0, 0, 1],
#     [1, 0, 1, 1, 0, 0],
#     [1, 1, 0, 1, 1, 0],
#     [0, 1, 1, 0, 1, 1],
#     [0, 0, 1, 1, 0, 1],
#     [1, 0, 0, 1, 1, 0]
# ])

# OPTION 3: Random graph
# n_nodes: int = 5
# density: float = 0.5  # Probability of edge existing
# adjacency_matrix: ndarray = (np.random.random((n_nodes, n_nodes)) < density).astype(int)
# adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) // 2  # Make symmetric
# np.fill_diagonal(adjacency_matrix, 0)  # No self-loops

n_nodes: int = len(adjacency_matrix)
edges: List[Tuple[int, int]] = []
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        if adjacency_matrix[i][j] == 1:
            edges.append((i, j))

print(f"Number of nodes: {n_nodes}")
print(f"Number of edges: {len(edges)}")
print(f"Edges: {edges}")
print(f"\nAdjacency matrix:\n{adjacency_matrix}\n")


def visualise_graph(adjacency_matrix: ndarray, partition: List[int] = None) -> None:
    G: nx.Graph = nx.Graph()
    n: int = len(adjacency_matrix)

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)

    if partition:
        colors: List[str] = [
            "lightblue" if partition[i] == 0 else "lightcoral" for i in range(n)
        ]
    else:
        colors: List[str] = ["lightgray"] * n

    plt.figure(figsize=(8, 6))
    pos: Dict = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        node_color=colors,
        with_labels=True,
        node_size=800,
        font_size=16,
        font_weight="bold",
        edge_color="gray",
        width=2,
    )

    if partition:
        cut_edges: List[Tuple[int, int]] = []
        for i, j in G.edges():
            if partition[i] != partition[j]:
                cut_edges.append((i, j))
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color="red", width=3)
        plt.title(f"Max-Cut Solution (Cut size: {len(cut_edges)})")
    else:
        plt.title("Graph Visualisation")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("maxcut_graph.png", dpi=150, bbox_inches="tight")
    print("Graph saved as 'maxcut_graph.png'")
    plt.close()


visualise_graph(adjacency_matrix)


def maxcut_to_ising(adjacency_matrix: ndarray) -> Tuple[SparsePauliOp, int]:
    n: int = len(adjacency_matrix)
    num_qubits: int = n

    pauli_list: List[Tuple[str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] == 1:
                # Add Z_i * Z_j term (negative to maximize cut)
                pauli_str: List[str] = ["I"] * num_qubits
                pauli_str[i] = "Z"
                pauli_str[j] = "Z"
                pauli_list.append(("".join(reversed(pauli_str)), -0.5))

                # Add constant term (will be same for all states, can skip)
                # pauli_list.append(('I' * num_qubits, 0.5))

    if not pauli_list:
        pauli_list.append(("I" * num_qubits, 0.0))

    hamiltonian: SparsePauliOp = SparsePauliOp.from_list(pauli_list)
    return hamiltonian, num_qubits


hamiltonian: SparsePauliOp
num_qubits: int
hamiltonian, num_qubits = maxcut_to_ising(adjacency_matrix)
print(f"Number of qubits: {num_qubits}")
print(f"Hamiltonian terms: {len(hamiltonian)}")


def create_ansatz(num_qubits: int, params: ndarray, depth: int = 2) -> QuantumCircuit:
    qc: QuantumCircuit = QuantumCircuit(num_qubits)

    param_idx: int = 0

    for layer in range(depth):
        for i in range(num_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1

        for i in range(num_qubits):
            qc.cx(i, (i + 1) % num_qubits)

    for i in range(num_qubits):
        qc.ry(params[param_idx], i)
        param_idx += 1

    return qc


depth: int = 2
num_params: int = num_qubits * (depth + 1)
print(f"Circuit depth: {depth}")
print(f"Circuit parameters: {num_params}")


def evaluate_energy(params: ndarray) -> float:
    qc: QuantumCircuit = create_ansatz(num_qubits, params, depth)
    statevector: Statevector = Statevector(qc)
    expectation: float = statevector.expectation_value(hamiltonian).real
    return expectation


print("\nRunning VQE optimisation...\n")

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

print("\nOptimisation complete!")
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

print("\nTop 10 measurement results:")


def decode_partition(bitstring: str) -> List[int]:
    return [int(b) for b in bitstring[::-1]]


def calculate_cut_size(partition: List[int], adjacency_matrix: ndarray) -> int:
    n: int = len(partition)
    cut_size: int = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] == 1 and partition[i] != partition[j]:
                cut_size += 1
    return cut_size


best_partition: List[int] | None = None
best_cut_size: int = 0

for idx, (bitstring, count) in enumerate(sorted_counts[:10]):
    partition: List[int] = decode_partition(bitstring)
    cut_size: int = calculate_cut_size(partition, adjacency_matrix)
    prob: float = count / 2000

    print(
        f"{idx + 1:2d}. {bitstring} | Partition: {partition} | "
        f"Cut size: {cut_size:2d} | Shots: {count:4d} ({prob:.1%})"
    )

    if cut_size > best_cut_size:
        best_cut_size = cut_size
        best_partition = partition

print("\n" + "~" * 30)
print("Best quantum solution found:")
print(f"Partition: {best_partition}")
print(f"Cut size: {best_cut_size} edges")

set0: List[int] = [i for i, p in enumerate(best_partition) if p == 0]
set1: List[int] = [i for i, p in enumerate(best_partition) if p == 1]
print(f"Set 0: {set0}")
print(f"Set 1: {set1}")

visualise_graph(adjacency_matrix, best_partition)

print("\n" + "~" * 30)
print("Classical optimal solution (brute force):")
print("~" * 30)


def solve_maxcut_classical(adjacency_matrix: ndarray) -> Tuple[List[int], int]:
    n: int = len(adjacency_matrix)
    max_cut: int = 0
    best_partition: List[int] | None = None

    # Try all 2^n possible partitions
    for mask in range(2**n):
        partition: List[int] = [(mask >> i) & 1 for i in range(n)]
        cut_size: int = calculate_cut_size(partition, adjacency_matrix)

        if cut_size > max_cut:
            max_cut = cut_size
            best_partition = partition

    return best_partition, max_cut


classical_partition: List[int]
classical_cut_size: int
classical_partition, classical_cut_size = solve_maxcut_classical(adjacency_matrix)

print(f"Optimal partition: {classical_partition}")
print(f"Maximum cut size: {classical_cut_size} edges")

set0_classical: List[int] = [i for i, p in enumerate(classical_partition) if p == 0]
set1_classical: List[int] = [i for i, p in enumerate(classical_partition) if p == 1]
print(f"  Set 0: {set0_classical}")
print(f"  Set 1: {set1_classical}")

# Comparison
print("\n" + "~" * 30)
print("Summary:")
print("~" * 30)
gap: float = (
    ((classical_cut_size - best_cut_size) / classical_cut_size) * 100
    if classical_cut_size > 0
    else 0
)
print(f"Quantum solution cut size: {best_cut_size}")
print(f"Classical optimal cut size: {classical_cut_size}")
print(f"Gap: {gap:.1f}%")

if best_cut_size == classical_cut_size:
    print("Quantum algorithm found the optimal solution!")
elif gap < 20:
    print("Quantum algorithm found a good approximation!")
else:
    print("Solution could be improved")

print("\n" + "~" * 30)
print("Increase circuit depth (depth=3 or 4)")
print("Increase optimiser iterations (maxiter=200-300)")
print("Try different random seeds")
print("For larger graphs (6+ nodes), expect longer runtime")
print("~" * 30)
