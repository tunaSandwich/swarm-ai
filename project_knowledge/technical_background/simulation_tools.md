# Simulation Tools & Libraries

This document lists recommended software libraries and tools, primarily within the Python ecosystem, that are suitable for developing the "AI-Driven Connectivity Maintenance for Drone Swarm Simulation" project.

## 1. Core Language

*   **Python (Version 3.x):** The primary development language due to its extensive libraries for scientific computing, data analysis, AI/ML, and visualization, as well as its ease of prototyping.

## 2. Simulation Core & Numerics

*   **NumPy:** Fundamental package for numerical operations. Essential for:
    *   Handling drone positions and velocities (arrays).
    *   Calculating distances efficiently.
    *   Vectorized operations for performance.
*   **Standard Python Libraries:** `math` (for basic functions), `random` (for stochastic elements in AI or initialization), `collections` (for data structures if needed).

## 3. Visualization

*   **Matplotlib:** A widely used plotting library. Suitable for:
    *   Generating static plots of drone positions and links at specific time steps.
    *   Creating animated visualizations by updating plots over time (`matplotlib.animation`).
    *   Plotting metrics (connectivity, energy) over time.
*   **Pygame:** A library designed for creating games, but excellent for simple, real-time 2D visualizations and basic interactivity. Suitable for:
    *   Drawing drone shapes and links directly onto a display surface.
    *   Handling user input (e.g., pausing simulation, clicking on drones).
    *   Potentially smoother real-time animation than Matplotlib for many agents.
*   *(Consider)* **Seaborn:** Built on Matplotlib, provides higher-level plotting functions and potentially more aesthetically pleasing statistical visualizations (e.g., distribution of node degrees).

**Recommendation:** Choose *either* Matplotlib (especially `animation`) or Pygame for the primary visualization based on preference and desired interactivity level. Matplotlib is often easier for generating final plots for reports.

## 4. Network/Graph Analysis

*   **NetworkX:** The standard Python library for creating, manipulating, and studying complex networks (graphs). Essential for:
    *   Representing the drone swarm as a graph (nodes=drones, edges=links).
    *   Calculating node degrees (`G.degree()`).
    *   Finding connected components (`nx.connected_components(G)`).
    *   Calculating the graph Laplacian (`nx.laplacian_matrix(G)`).
    *   Potentially calculating algebraic connectivity (requires SciPy/NumPy for eigenvalues).
    *   Drawing basic graph layouts (though direct position plotting might be clearer).

## 5. AI/Machine Learning

*   **NumPy/Standard Python:** For implementing simpler algorithms like basic Q-Learning (using dictionaries or NumPy arrays for the Q-table) or PSO from scratch.
*   *(Optional - If using DQN)* **TensorFlow** or **PyTorch:** Major deep learning frameworks. Necessary if implementing a Deep Q-Network (DQN) where a neural network approximates the Q-function. *Adds significant complexity; only use if comfortable and if simpler Q-learning proves insufficient.*

## 6. Development Environment

*   **IDE:** Any standard Python IDE (e.g., VS Code, PyCharm, Spyder).
*   **Version Control:** Git (highly recommended for tracking changes and collaboration, even if working alone). Platforms like GitHub or GitLab for backup and potential sharing.
*   **Package Management:** `pip` (standard Python package installer) and `virtualenv` or `conda` (for managing project dependencies and creating isolated environments).

**Focus:** Prioritize using NumPy, NetworkX, and either Matplotlib or Pygame for the core simulation and visualization. Implement the chosen AI algorithm using standard Python/NumPy first, only adding heavier libraries like TensorFlow/PyTorch if demonstrably necessary and within your comfort zone.
