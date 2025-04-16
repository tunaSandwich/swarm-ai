# Step-by-Step AI Prompts: Drone Connectivity Simulation

Use these prompts with your AI assistant to help execute each stage defined in `Getting_Started_Steps.md`.

## 1. Environment Setup

*   "What are the `pip install` commands for `numpy`, `matplotlib`, `pygame`, and `networkx`?"
*   "Generate a basic `.gitignore` file for a Python project."
*   "Show the commands to create and activate a Python `venv` virtual environment on [Your Operating System: Windows/macOS/Linux]."

## 2. Build the Simulation Core (No AI Yet)

*   "Generate a basic Python class structure for `Drone` and `SimulationEnvironment` based on attributes discussed in `Simulation_Model.md`."
*   "Show how to initialize a list of `N` Drone objects with random starting positions within an `AREA_WIDTH` x `AREA_HEIGHT` space."

## 3. Implement Basic Movement & Boundaries

*   "Provide a Python function `move_drone_randomly(drone)` that updates its position by a small random step."
*   "Show how to implement the 'Stop' boundary handling method from `Simulation_Model.md` within the drone's update function."
*   "Create the main simulation loop structure in Python that iterates a fixed number of steps and calls an update function for each drone."

## 4. Visualize Positions & Links

*   "Using Pygame, show how to draw a list of drones as circles on the screen and update their positions in the main loop."
*   "Provide a Python function `get_active_links(drones, comm_range)` that returns a list of drone pairs [(drone_i, drone_j), ...] that are within `COMM_RANGE`."
*   "Show how to draw lines between linked drones in Pygame, using the output from `get_active_links`."
*   *(Alternative)* "Show how to create a basic Matplotlib scatter plot visualizing drone positions, updating it within the simulation loop using `matplotlib.animation`."

## 5. Calculate Basic Connectivity

*   "Given a list of drones and a list of active links (pairs of drone IDs), show how to create a NetworkX graph."
*   "Using NetworkX, calculate and print the average node degree for the current drone graph."
*   "Show the NetworkX code to find the number of connected components and the size of the largest connected component (LCC)."

## 6. Implement ONE Positioning Algorithm

*   *(If RL)* "Draft the state representation array for a drone using Q-learning, including relative positions to 3 nearest neighbors, based on `Positioning_Algorithms.md`."
*   *(If RL)* "Provide a Python function `calculate_rl_reward(drone, links)` incorporating link count and movement cost."
*   *(If RL)* "Show the core Q-learning update rule: `Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))` in Python syntax."
*   *(If SI)* "Draft a Python fitness function `calculate_pso_fitness(drone, neighbors)` prioritizing link count within `COMM_RANGE`."
*   *(If SI)* "Show the Python implementation of the PSO velocity and position update equations from `Positioning_Algorithms.md`."

## 7. Integrate AI into Movement

*   *(If RL)* "Explain how to choose an action using an epsilon-greedy policy based on the drone's current state and Q-table."
*   *(If RL)* "How do I translate the chosen discrete action (N/S/E/W/Stay) into a position update for the drone?"
*   *(If SI)* "Modify the drone's update function to use the calculated PSO velocity to update its position, respecting `MAX_SPEED`."

## 8. Layer in Energy Model

*   "Add attributes for `energy`, `E_IDLE_COST`, `E_MOVE_COST` to the `Drone` class."
*   "Implement the energy depletion logic (idle and movement costs) within the drone's update step, referencing `Simulation_Model.md`."
*   "Modify the [RL reward / PSO fitness] function to include a penalty for low energy or high movement cost."

## 9. Iterate & Refine

*   "My drones are just clustering together and not spreading out. Here is my [RL reward / PSO fitness] function: [Paste Code]. Based on `Positioning_Algorithms.md`, how can I add a spacing penalty?"
*   "What are common parameters to tune for [Q-learning: learning rate alpha, discount factor gamma, exploration epsilon / PSO: inertia w, coefficients c1, c2] and what effect do they typically have?"
*   "Suggest how to log connectivity metrics (like LCC size) over time during the simulation run into a list or file."

## 10. Analyze & Report

*   "Show how to use Matplotlib to plot the LCC size logged over time."
*   "Help me outline the 'Methodology' section of my report, covering the simulation model details from `Simulation_Model.md` and the implemented [Q-learning / PSO] algorithm."
*   "Based on this plot showing [Describe metric trend, e.g., LCC size increasing then plateauing], what are some potential interpretations or discussion points for my report?"
