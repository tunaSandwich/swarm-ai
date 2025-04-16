# Connectivity Metrics for Drone Swarm Simulation

This document describes metrics used to evaluate the network connectivity of the simulated drone swarm. Good connectivity ensures that information *could* potentially flow through the mesh, which is the underlying goal the positioning algorithms aim to achieve.

## Why Measure Connectivity?
In this simulation, drones position themselves to maintain potential communication links. We need quantitative ways to measure how well they achieve this collective goal. These metrics can be used:
*   As part of the reward function (for RL) or fitness function (for SI).
*   To evaluate the performance of the chosen positioning algorithm over time.

## Key Metrics

1.  **Number of Active Links / Average Node Degree:**
    *   **Definition:** The total count of active communication links in the network at a given time step, or the average number of neighbors per drone. A link exists if two drones are within `COMM_RANGE`.
    *   **Calculation:**
        *   Total Links: Iterate through all unique pairs of drones (i, j), count pairs where `distance(i, j) < COMM_RANGE`.
        *   Average Degree: Sum the number of neighbors for each drone and divide by the number of active drones.
    *   **Pros:** Very simple to calculate and understand. Directly reflects local connectivity.
    *   **Cons:** A high average degree doesn't guarantee the entire network is connected (it could be split into multiple well-connected clusters).
    *   **Usage:** Good for basic reward signals or quick health checks.

2.  **Largest Connected Component (LCC) Size:**
    *   **Definition:** The number of drones belonging to the largest single group where any drone can reach any other drone within that group via one or more hops (links).
    *   **Calculation:** Represent the network as a graph (nodes = drones, edges = active links). Use algorithms like Breadth-First Search (BFS) or Depth-First Search (DFS) starting from each node to find all connected components and identify the largest one.
    *   **Pros:** Directly indicates if the network is fragmented (`LCC size < number of active drones`) or fully connected (`LCC size == number of active drones`).
    *   **Cons:** Doesn't measure *how robustly* connected the network is (e.g., a single bridge drone failure could split a fully connected network). Binary outcome (connected/not connected).
    *   **Usage:** Excellent indicator of overall network integrity. Aiming for `LCC size == N` is a clear objective.

3.  **Algebraic Connectivity (Fiedler Value):**
    *   **Definition:** The second-smallest eigenvalue (λ₂) of the graph Laplacian matrix (L = D - A, where D is the degree matrix and A is the adjacency matrix).
    *   **Calculation:**
        1.  Construct the adjacency matrix `A` (A_ij = 1 if link exists, 0 otherwise).
        2.  Construct the degree matrix `D` (diagonal matrix with D_ii = degree of node i).
        3.  Calculate the Laplacian matrix `L = D - A`.
        4.  Compute the eigenvalues of `L`. The second smallest eigenvalue is the algebraic connectivity.
    *   **Pros:** A powerful measure. λ₂ > 0 if and only if the graph is connected. The magnitude of λ₂ relates to how well-connected the graph is (higher values suggest more robustness to node/link removal). Can be used as a continuous measure of connectivity quality.
    *   **Cons:** Computationally more expensive than the other metrics, especially for larger networks. Requires eigenvalue calculation. Less intuitive to understand directly.
    *   **Usage:** Ideal as a robust connectivity measure for analysis or potentially as a component in the AI's objective function if computational cost is acceptable for the simulation scale (15-30 drones should be fine).

## Implementation Notes
*   The **NetworkX** library in Python is highly recommended for graph representation and calculating these metrics. It provides functions for:
    *   Creating graphs from node/edge lists.
    *   Calculating node degrees (`G.degree()`).
    *   Finding connected components (`nx.connected_components(G)`).
    *   Calculating the Laplacian matrix (`nx.laplacian_matrix(G)`) and its eigenvalues (using `scipy.linalg.eigvalsh` or `numpy.linalg.eigvalsh`).

## Recommendation for Project
*   **Primary:** Track **Average Node Degree** (simple) and **LCC Size** (shows fragmentation). Aim to maximize LCC Size to equal the number of active drones.
*   **Secondary/Advanced:** Calculate **Algebraic Connectivity (Fiedler Value)** for a more nuanced analysis of connectivity quality, especially if using it within the AI's objective function is desired.
