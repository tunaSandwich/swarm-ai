# Positioning Algorithms for Connectivity Maintenance

This document details the candidate AI/Machine Learning approaches for enabling drones in the simulation to autonomously position themselves to maximize or maintain network connectivity within the swarm, adhering to the defined project scope.

## Core Challenge
The fundamental task is decentralized control: each drone must decide where to move based primarily on local information to achieve a global objective (good network connectivity) while considering constraints like energy.

## 1. Reinforcement Learning (RL) Approach

**Concept:** Drones learn optimal movement policies through trial-and-error by receiving rewards or penalties based on the outcomes of their actions within the simulated environment. For this project's scope, focusing on **Independent Learners (ILs)** is recommended, where each drone runs its own simple RL agent.

**Key Components:**

*   **Agent:** Each individual drone acts as an RL agent.
*   **Environment:** The simulation space, including other drones and their states.
*   **State Representation (`s`):** The information available to a drone to make decisions. Keep this simple and local:
    *   Own current position (x, y).
    *   Own current energy level (e.g., normalized 0-1).
    *   *Option A (Simpler):* Number of currently active links (local degree).
    *   *Option B (More Info):* Relative positions (distance, bearing) to the N nearest neighbors (e.g., N=3 or 5).
    *   *Consider:* A flag indicating if energy is critically low.
*   **Action Space (`a`):** The possible moves a drone can make. A discrete action space is recommended:
    *   `Move North`
    *   `Move South`
    *   `Move East`
    *   `Move West`
    *   `Stay Still`
    *   (Movement distance per step is fixed by the simulation model).
*   **Reward Function (`R(s, a, s')`):** *This is critical for guiding learning.* The reward signals what behavior is desirable. A combination of factors is usually needed:
    *   **Connectivity Reward:** Positive reward for each active communication link maintained at state `s'`. (e.g., `+1` per link). This is the primary driver.
    *   **Spacing Reward/Penalty (Optional but Recommended):**
        *   Penalty for being too close to any neighbor (e.g., `< 0.5 * COMM_RANGE`).
        *   Penalty for being too far from the nearest neighbor (encourages cohesion).
        *   Alternatively, reward for being within an "ideal" distance band (e.g., `0.7 * COMM_RANGE` to `0.9 * COMM_RANGE`).
    *   **Movement Cost:** Negative reward proportional to the energy cost of the action taken (e.g., `-0.1` if moved, `0` if stayed still). Ties decisions to energy constraint.
    *   **Low Energy Penalty (Optional):** Small negative reward if energy level is below a threshold, discouraging risky moves when low.
    *   **Goal:** Tune these components so the agent maximizes connectivity while managing energy. Start simple (e.g., link reward + movement cost) and add complexity if needed.

**Suggested Algorithm:**

*   **Q-Learning (Tabular):** Suitable if the discretized state space is small enough. Each drone maintains a table `Q(s, a)` estimating the value of taking action `a` in state `s`. Updates use the Bellman equation based on received rewards. Simpler to implement.
*   **Deep Q-Network (DQN):** If the state space becomes too large (e.g., using continuous neighbor distances), a neural network can approximate the Q-function. Requires more careful implementation and tuning. *Recommendation: Start with Q-Learning unless necessary.*

## 2. Swarm Intelligence (SI) Approach

**Concept:** Mimics collective behaviors found in nature (like bird flocking or ant colonies). Drones follow simple rules based on local interactions, leading to emergent global behavior (good connectivity). Particle Swarm Optimization (PSO) is a relevant candidate.

**Adaptation for Connectivity (PSO-like):**

*   **Particles:** Each drone acts as a particle in the search space (the 2D/3D environment).
*   **Position (`x_i`):** The drone's current coordinates.
*   **Velocity (`v_i`):** The drone's current direction and speed of movement.
*   **Fitness Function (`f(x_i)`):** A function evaluating how "good" a drone's current position `x_i` is, primarily based on connectivity. This replaces the reward function in RL. Needs careful design:
    *   *Goal:* Maximize connectivity.
    *   *Example Components:*
        *   Maximize number of links within `COMM_RANGE`.
        *   Maximize sum of "quality" scores for each neighbor based on distance (higher score for ideal distance band, lower score for too close/too far).
        *   Minimize distance to the centroid of neighbors (promotes cohesion).
        *   Penalize low energy states (e.g., by reducing the fitness score).
*   **Personal Best (`pbest_i`):** The best position found *so far* by drone `i` (according to the fitness function).
*   **Neighborhood Best (`lbest_i`):** The best position found *so far* by any drone within drone `i`'s local neighborhood (e.g., drones within communication range). Using `lbest` promotes decentralized behavior. (Using `gbest` - global best - is also possible but less swarm-like).
*   **Update Rules:** Drones adjust their velocity and position at each time step based on:
    *   Their current velocity (inertia).
    *   Their `pbest` position (cognitive component - return to own best spot).
    *   Their `lbest` position (social component - move towards neighbor's best spot).
    ```
    v_i(t+1) = w * v_i(t) + c1 * rand() * (pbest_i - x_i(t)) + c2 * rand() * (lbest_i - x_i(t))
    x_i(t+1) = x_i(t) + v_i(t+1)
    ```
    (Where `w`, `c1`, `c2` are tuning parameters: inertia weight, cognitive coefficient, social coefficient).

## 3. Comparison for This Project

| Feature          | Reinforcement Learning (ILs)            | Swarm Intelligence (PSO-like)           |
| :--------------- | :-------------------------------------- | :-------------------------------------- |
| **Learning**     | Learns policy via trial-and-error     | Optimization based on fitness function |
| **Adaptability** | Can potentially learn more complex, adaptive strategies | Behavior directly driven by fitness func. |
| **Tuning**       | Reward function design is crucial & can be tricky | Fitness function & PSO parameters (`w, c1, c2`) need tuning |
| **Complexity**   | Q-learning simple; DQN more complex   | Conceptually straightforward; implementation details matter |
| **Exploration**  | Explicit exploration mechanisms (e.g., epsilon-greedy) | Implicit via random factors & inertia |
| **Optima**       | Can get stuck in local optima; depends on exploration | Can get stuck in local optima; depends on parameters/diversity |

**Recommendation:** Both approaches are viable for this proof-of-concept.
*   Start with the one you are more comfortable implementing.
*   RL (Q-learning) might highlight adaptability more directly.
*   SI (PSO) might be simpler to get basic cohesive movement working if fitness is well-defined.

Choose **one** primary algorithm to implement thoroughly first, rather than attempting both partially.
