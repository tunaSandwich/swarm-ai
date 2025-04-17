# Simulation Tools & Libraries (Corridor Formation)

This document lists recommended software libraries and tools, primarily within the Python ecosystem, suitable for developing the "AI-Driven Drone Corridor Formation Simulation" project. This aligns with the tools already identified in the existing codebase review.

## 1. Core Language

*   **Python (Version 3.x):** The primary development language.

## 2. Simulation Core & Numerics

*   **NumPy:** Fundamental package for numerical operations. **Crucial** for:
    *   Handling drone positions, velocities, vectors.
    *   Efficiently calculating distances.
    *   **Implementing geometric calculations related to the Start-End centerline (projections, perpendicular distances) needed for corridor adherence metrics and AI logic.**
*   **Standard Python Libraries:** `math`, `random`, `collections`.

## 3. Visualization

*   **Matplotlib:** Widely used plotting library. Suitable for:
    *   Rendering simulation states (drone positions, links) potentially overlayed with the Start/End points and ideal centerline (as implemented in `environment.py`).
    *   Generating final plots summarizing corridor-specific metrics over time.
    *   Can be used for animation (`matplotlib.animation`).
*   *(Alternative)* **Pygame:** Also suitable for 2D visualization, potentially offering smoother real-time interaction if desired.

## 4. Network/Graph Analysis

*   **NetworkX:** The standard Python library for graph analysis. Essential for:
    *   Representing the drone communication network based on `COMM_RANGE`.
    *   **Analyzing connectivity specifically *within* the formed corridor (e.g., creating subgraphs of drones within the corridor).**
    *   Calculating path-related metrics (shortest path hops, checking path existence) on the corridor subgraph.
    *   Calculating local density/degree within the corridor.

## 5. AI/Machine Learning

*   **NumPy/Standard Python:** For implementing the chosen AI algorithm (RL Q-Learning/DQN or SI Flocking/PSO adaptations) from scratch or with basic structures.
*   *(Optional - If using DQN)* **TensorFlow** or **PyTorch:** Major deep learning frameworks needed for neural network Q-function approximation. Consider the added complexity.

## 6. Data Handling & Analysis

*   **Pandas:** Useful for organizing and analyzing the time-series metrics collected during simulation runs (as already implemented in `main.py`).

## 7. Development Environment

*   **IDE:** VS Code, PyCharm, Spyder, etc.
*   **Version Control:** Git/GitHub/GitLab.
*   **Package Management:** `pip` with `venv` or `conda`.

**Focus:** The existing toolset (NumPy, NetworkX, Matplotlib, Pandas) is well-suited. Emphasis during development will be on using NumPy effectively for geometric corridor calculations and NetworkX for subgraph analysis relevant to the corridor's structure and connectivity.
