# IBM-Quantum
Applications written in Python using QisKit for IBM Quantum technologies.

# Quantum Solutions to NP-Complete Problems with Qiskit

This repository contains quantum computing implementations of classic NP-complete and NP-hard problems using IBM's Qiskit framework and Python. These implementations use the Variational Quantum Eigensolver (VQE) algorithm to find approximate solutions.

## 📋 Problems Implemented

### 1. Travelling Salesman Problem (TSP)
**File:** `TSP.py`

**Problem Description:**
Find the shortest route that visits all cities exactly once and returns to the starting city.

**Complexity:**
- **Problem Type:** NP-hard
- **Qubits Required:** n² (where n = number of cities)
- **Example:** 3 cities = 9 qubits, 4 cities = 16 qubits

**Key Features:**
- Converts TSP to Ising Hamiltonian formulation
- Uses constraint penalties to enforce valid tours
- Compares against classical brute-force solution
- Full type annotations for all variables and functions

**Performance:**
- 3 cities: ~1-2 minutes
- 4 cities: ~5-10 minutes, requires 4-8 GB RAM
- 5+ cities: Not recommended for simulation

**Classical Comparison:** Brute force (O(n!))

---

### 2. Max-Cut Problem
**File:** `Max-Cut.py`

**Problem Description:**
Partition graph nodes into two sets to maximise the number of edges between the sets.

**Complexity:**
- **Problem Type:** NP-complete
- **Qubits Required:** n (where n = number of nodes)
- **Example:** 4 nodes = 4 qubits, 6 nodes = 6 qubits

**Key Features:**
- Graph visualisation with colour-coded partitions
- Highlights cut edges in red
- Supports custom graphs via adjacency matrix
- Includes random graph generation option
- Saves visualisation as `maxcut_graph.png`

**Performance:**
- 4 nodes: ~30 seconds
- 6 nodes: ~1-2 minutes
- 8+ nodes: Expect longer runtimes

**Classical Comparison:** Brute force (O(2ⁿ))

**Applications:**
- Network partitioning
- Circuit design
- Clustering problems
- VLSI design

---

### 3. Knapsack Problem
**File:** `Knapsack.py`

**Problem Description:**
Select items to maximise total value whilst staying within a weight capacity constraint.

**Complexity:**
- **Problem Type:** NP-complete (0/1 Knapsack)
- **Qubits Required:** n (where n = number of items)
- **Example:** 4 items = 4 qubits, 6 items = 6 qubits

**Key Features:**
- Visualises item values, weights, and selected items
- Constraint handling via penalty terms
- Compares against optimal dynamic programming solution
- Shows detailed breakdown of selected items
- Saves visualisation as `knapsack_solution.png`

**Performance:**
- 4 items: ~30 seconds
- 6 items: ~1-2 minutes
- 8+ items: Expect longer runtimes

**Classical Comparison:** Dynamic Programming (O(n·W) - pseudo-polynomial)

**Applications:**
- Resource allocation
- Portfolio optimisation
- Cargo loading
- Budget planning

---

## 🛠️ Installation

### Required Packages
```bash
pip install qiskit qiskit-algorithms qiskit-aer scipy numpy matplotlib networkx
```

### Package Versions
- Python 3.9+
- Qiskit 1.0+
- qiskit-algorithms
- qiskit-aer
- scipy
- numpy
- matplotlib
- networkx (for Max-Cut visualisation)

---

## 🚀 Usage

### Basic Usage
Each script can be run directly:
```powershell
python TSP.py
python Max-Cut.py
python Knapsack.py
```

### Customising Problems

#### TSP - Change Number of Cities
```python
# 3 cities (default)
distance_matrix: ndarray = np.array([
    [0, 2, 9],
    [1, 0, 6],
    [15, 7, 0]
])

# 4 cities
distance_matrix: ndarray = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])
```

#### Max-Cut - Define Custom Graph
```python
# Via adjacency matrix
adjacency_matrix: ndarray = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])

# Or generate random graph
n_nodes: int = 5
density: float = 0.5
adjacency_matrix: ndarray = (np.random.random((n_nodes, n_nodes)) < density).astype(int)
adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) // 2
np.fill_diagonal(adjacency_matrix, 0)
```

#### Knapsack - Define Items
```python
items: List[Dict[str, int]] = [
    {"value": 10, "weight": 5},
    {"value": 40, "weight": 4},
    {"value": 30, "weight": 6},
    {"value": 50, "weight": 3}
]
capacity: int = 10
```

### Tuning Parameters

All implementations support these tuning parameters:

**Circuit Depth:**
```python
depth: int = 2  # Increase for better solutions (but slower)
```

**Optimiser Iterations:**
```python
options={'maxiter': 150}  # Increase for better convergence
```

**Penalty Values (for constrained problems):**
```python
# TSP
hamiltonian, num_qubits = tsp_to_ising(distance_matrix, penalty=40)

# Knapsack
hamiltonian, num_qubits = knapsack_to_ising(values, weights, capacity, penalty=100)
```

---

## 📊 Algorithm Details

### VQE (Variational Quantum Eigensolver)

All implementations use VQE, a hybrid quantum-classical algorithm:

1. **Parameterised Quantum Circuit (Ansatz):**
   - Creates a quantum circuit with adjustable parameters
   - Uses RY rotations and CNOT gates for entanglement

2. **Energy Evaluation:**
   - Measures expectation value of the problem Hamiltonian
   - Uses statevector simulation for exact calculation

3. **Classical Optimisation:**
   - Uses COBYLA optimiser to minimise energy
   - Iteratively adjusts circuit parameters

4. **Solution Extraction:**
   - Samples the optimal circuit multiple times
   - Decodes bitstrings into problem solutions
   - Validates constraints and selects best solution

### Problem Formulation

Each problem is converted to an **Ising Hamiltonian**:

**TSP:**
```
H = Σ distances(i,j) * Z_i * Z_j + penalty * Σ constraint_violations
```

**Max-Cut:**
```
H = -0.5 * Σ (Z_i * Z_j) for all edges (i,j)
```

**Knapsack:**
```
H = -Σ values * x_i + penalty * (Σ weights * x_i - capacity)²
```

---

## 📈 Output Examples

### TSP Output
```
=== VQE Results ===
Optimal energy: -8.2500
Optimiser evaluations: 145

Top 10 measurement results:
 1. 100010001 | Tour: [0, 2, 1] | Distance:  17.0 | Shots:  847 (42.4%) | ✓ Valid
 2. 001100010 | Tour: [1, 0, 2] | Distance:  23.0 | Shots:  412 (20.6%) | ✓ Valid

✓ Best quantum solution: [0, 2, 1]
  Distance: 17.0

Classical optimal solution:
✓ Optimal tour: [0, 2, 1]
  Optimal distance: 17
```

### Max-Cut Output
```
=== VQE Results ===
Final energy: -2.5000

Top 10 measurement results:
 1. 1001 | Partition: [1, 0, 0, 1] | Cut size:  4 | Shots: 1023 (51.2%)

✓ Best quantum solution found:
  Partition: [1, 0, 0, 1]
  Cut size: 4 edges
  Set 0: [1, 2]
  Set 1: [0, 3]

Graph saved as 'maxcut_graph.png'
```

### Knapsack Output
```
=== VQE Results ===
Final energy: -45.2300

Top 10 measurement results:
 1. 1011 | Items: [0, 1, 3] | Value: 100 | Weight:  12/10 | Shots:  892 (44.6%) | ✗ Overweight
 2. 0110 | Items: [1, 2] | Value:  70 | Weight:  10/10 | Shots:  654 (32.7%) | ✓ Valid

✓ Best quantum solution found:
  Selected items: [1, 2]
  Total value: 70
  Total weight: 10/10

Visualisation saved as 'knapsack_solution.png'
```

---

## ⚡ Performance Tips

### Getting Better Results

1. **Increase optimiser iterations:**
   ```python
   options={'maxiter': 200}  # Default is 100-150
   ```

2. **Adjust penalty parameters:**
   - Too low: Constraint violations
   - Too high: Poor optimisation
   - Recommended range: 30-100

3. **Increase circuit depth:**
   ```python
   depth: int = 3  # More expressive, but slower
   ```

4. **Run multiple times:**
   - VQE can get stuck in local minima
   - Try different random seeds

5. **Start small:**
   - Test with minimal problem sizes first
   - TSP: 3 cities, Max-Cut: 4 nodes, Knapsack: 4 items

### Memory Requirements

| Problem Size | Qubits | RAM Required | Time Estimate |
|--------------|--------|--------------|---------------|
| Small (≤5)   | ≤9     | 1-2 GB       | 1-5 min       |
| Medium (6-7) | ≤16    | 4-8 GB       | 5-15 min      |
| Large (8+)   | ≤25+   | 16+ GB       | 30+ min       |

---

## 🔬 Theoretical Background

### Why Quantum Computing for NP Problems?

Classical computers struggle with NP-complete problems because:
- Solution space grows exponentially
- No known polynomial-time algorithms
- Brute force becomes impractical quickly

Quantum computers offer potential advantages through:
- **Superposition:** Explore multiple solutions simultaneously
- **Entanglement:** Create correlations between qubits
- **Quantum tunnelling:** Escape local minima in optimisation

### Current Limitations

⚠️ **Important Notes:**
- These are **noisy intermediate-scale quantum (NISQ)** algorithms
- Run on classical simulators (not actual quantum hardware)
- No proven exponential speedup for NP-complete problems
- Best for small problem instances and proof-of-concept

### Quantum Advantage?

Currently, quantum algorithms for NP-complete problems:
- ✅ Demonstrate quantum computing concepts
- ✅ May find good approximate solutions
- ✅ Useful for small-scale optimisation
- ❌ Don't outperform classical algorithms yet
- ❌ Limited by simulator capacity and noise

---

## 🎯 Other NP Problems You Can Implement

Using similar techniques, you can solve:

### Graph Problems
- **Vertex Cover** - Minimum vertices covering all edges
- **Graph Colouring** - Colour nodes with minimum colours
- **Clique Problem** - Find largest complete subgraph
- **Hamiltonian Cycle** - Visit all vertices exactly once

### Optimisation Problems
- **Bin Packing** - Pack items into minimum bins
- **Job Scheduling** - Minimise makespan
- **Set Cover** - Cover all elements with minimum sets
- **Vehicle Routing** - Optimise delivery routes

### Logic Problems
- **Boolean Satisfiability (SAT)** - Satisfy Boolean formulae
- **3-SAT** - SAT with 3 variables per clause
- **Subset Sum** - Find subset with target sum

---

## 📚 References

### Qiskit Documentation
- [Qiskit Documentation](https://docs.quantum.ibm.com/)
- [Qiskit Algorithms](https://qiskit.org/ecosystem/algorithms/)
- [VQE Tutorial](https://qiskit.org/documentation/tutorials/algorithms/04_vqe_advanced.html)

### Academic Papers
- Farhi et al. (2014) - "A Quantum Approximate Optimization Algorithm"
- Peruzzo et al. (2014) - "A variational eigenvalue solver on a photonic quantum processor"
- Lucas (2014) - "Ising formulations of many NP problems"

### Learning Resources
- [Qiskit Textbook](https://qiskit.org/textbook/)
- [IBM Quantum Learning](https://learning.quantum.ibm.com/)
- [Quantum Computing for Computer Scientists](https://www.cambridge.org/core/books/quantum-computing-for-computer-scientists/8AEA723BEE5CC9F5C03FDD4BA850C711)

---

## 🤝 Contributing

Contributions are welcome! Potential improvements:
- Additional NP-complete problems
- Better ansatz designs
- Noise mitigation techniques
- Real quantum hardware integration
- Performance optimisations
- Better visualisations

---

## 📄 Licence

MIT Licence - feel free to use and modify for your projects.

---

## ⚠️ Disclaimer

These implementations are for educational and research purposes. They demonstrate quantum computing concepts but are not production-ready solutions. For real-world optimisation problems, classical algorithms currently perform better.

---

## 💡 Tips for Learning

1. **Start with Max-Cut** - Simplest problem, easiest to understand
2. **Then try Knapsack** - Introduces constraint handling
3. **Finally tackle TSP** - Most complex, combines multiple concepts
4. **Experiment with parameters** - See how they affect solutions
5. **Compare with classical** - Understand current limitations
6. **Read the code comments** - Detailed explanations throughout

---

## 📞 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check Qiskit documentation
- Join the [Qiskit Slack community](https://qiskit.slack.com/)

Happy quantum computing! 🚀⚛️
