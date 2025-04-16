# Getting Started Steps: AI Drone Connectivity Simulation

This document outlines a step-by-step approach to build the simulation project. Follow these steps sequentially, leveraging AI assistance with the prompts provided in `Step_By_Step_Prompts.md`.

## 1. Environment Setup
*   Create a dedicated project folder.
*   Set up a Python virtual environment (e.g., `venv`, `conda`).
*   Install essential libraries (`numpy`, `matplotlib` or `pygame`, `networkx`).
*   *(Use AI for specific installation commands if needed)*

## 2. Build the Simulation Core (No AI Yet)
*   Create the main Python script (e.g., `main.py`).
*   Define the `Drone` class (ID, position, energy) based on `Simulation_Model.md`.
*   Create a simulation environment class/functions to manage drones and boundaries.
*   Initialize a list of `Drone` objects at random positions.
*   *(Use AI for class structure generation)*

## 3. Implement Basic Movement & Boundaries
*   Create the main simulation loop (step-by-step updates).
*   Implement **random** movement for drones initially.
*   Implement boundary checking/handling as defined in `Simulation_Model.md`.
*   *(Use AI for movement and boundary code snippets)*

## 4. Visualize Positions & Links
*   Add basic visualization (Pygame or Matplotlib) to the loop.
*   Draw drone positions (circles/points).
*   Implement communication link logic (distance < `COMM_RANGE`).
*   Draw lines for active links between drones.
*   *(Use AI for visualization code and link logic)*

## 5. Calculate Basic Connectivity
*   Integrate the NetworkX library.
*   Build a NetworkX graph from active links in each step.
*   Calculate and display a simple connectivity metric (e.g., Average Node Degree, LCC Size) based on `Connectivity_Metrics.md`.
*   *(Use AI for NetworkX integration and metric calculation)*

## 6. Implement ONE Positioning Algorithm
*   Choose **one** algorithm (RL Q-learning or SI PSO) based on `Positioning_Algorithms.md`.
*   Implement the core logic (state/action/reward OR fitness/update rules). Start simply.
*   *(Use AI for algorithm-specific function drafting)*

## 7. Integrate AI into Movement
*   Replace the random movement logic from Step 3.
*   Drones now move based on the chosen AI algorithm's output (action or velocity).
*   Observe the simulation for emergent organizing behavior.
*   *(Use AI for guidance on connecting AI output to drone movement)*

## 8. Layer in Energy Model
*   Add energy depletion (idle, movement costs) from `Simulation_Model.md`.
*   Optionally modify the AI reward/fitness to consider energy.
*   Implement low-energy state changes or node removal if desired.
*   *(Use AI for energy model implementation)*

## 9. Iterate & Refine
*   Tune AI parameters (learning rate, reward weights, PSO coefficients).
*   Run simulations, observe, debug.
*   Improve visualizations; collect metric data over time for plotting.
*   *(Use AI for debugging assistance and parameter tuning advice)*

## 10. Analyze & Report
*   Generate final plots of metrics over time.
*   Structure and write the final report based on `Deliverables.md`.
*   *(Use AI for report outlining and result interpretation help)*
