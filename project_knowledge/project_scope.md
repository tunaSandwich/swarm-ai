# Project Scope: AI-Driven Connectivity Maintenance Simulation

This document defines the specific boundaries for the "AI-Driven Connectivity Maintenance for a Drone Swarm Simulation" project. Its purpose is to ensure the project remains focused and achievable within the context of creating a compelling proof-of-concept.

## IN SCOPE

The following elements are considered **within** the scope of this project:

1.  **Simulation Environment:**
    *   Development of a simulation environment, likely in Python.
    *   Representation of a 2D or simple 3D bounded space.

2.  **Drone Swarm:**
    *   Simulation of a **small** number of drones (target range: 15-30 agents).
    *   Modeling basic drone movement capabilities (e.g., discrete steps, simple velocity updates).
    *   Each drone represented by its position and internal state (e.g., energy level).

3.  **AI/ML for Positioning:**
    *   Implementation and testing of **one or two** specific AI/ML algorithms focused on autonomous positioning. Candidates include:
        *   Reinforcement Learning (RL), likely using independent learners (e.g., Q-learning, potentially simple DQN per agent). Focus on defining appropriate state, action, and reward structures.
        *   Swarm Intelligence (SI), such as an adaptation of Particle Swarm Optimization (PSO). Focus on defining the fitness function and update rules.
    *   The primary goal of the AI is to maximize/maintain network connectivity within the swarm.

4.  **Network & Communication Model:**
    *   Modeling network links based on a **simple proximity threshold:** A communication link exists between two drones if their distance is less than a predefined `COMM_RANGE`.
    *   Focus on **link existence/topology**, not data transmission simulation.
    *   Calculating and tracking network connectivity metrics (e.g., number of links, average node degree, potentially size of the largest connected component, or exploring algebraic connectivity if time permits).

5.  **Energy Model:**
    *   Implementation of a **simple energy consumption model:**
        *   Drones start with a maximum energy level.
        *   Energy depletes over time (idle cost).
        *   Energy depletes faster when moving (movement cost).
    *   Potentially using energy level as input for AI decisions or triggering node removal/state change at low energy thresholds.

6.  **Visualization & Reporting:**
    *   Visualizing the simulation state: drone positions and active communication links.
    *   Generating plots showing key metrics (connectivity, energy) over simulation time.
    *   Writing a concise final report (~3-5 pages) summarizing the project, methodology, results, and discussion.

## OUT OF SCOPE

The following elements are explicitly **outside** the scope of this project:

1.  **Data Traffic Simulation:**
    *   Simulating the transmission of actual data packets (no bandwidth, latency, throughput, packet loss, or QoS modeling).
    *   Ignoring specific data types (internet traffic, military, financial).

2.  **Network Protocol Implementation:**
    *   Implementing specific MAC layer protocols (e.g., CSMA/CA) or Network layer routing protocols (e.g., AODV, OLSR, DSR). Routing is implicitly handled by the AI positioning for direct communication within range.

3.  **Heterogeneous Networks & Backhaul:**
    *   Simulating larger "gateway" or "backhaul" drones.
    *   Modeling connection to the wider internet or ground stations.
    *   Inter-tier communication protocols.

4.  **Advanced Physics & Environment:**
    *   Detailed or realistic physics simulation (e.g., aerodynamics, wind effects, complex collision physics beyond simple overlap avoidance if needed).
    *   Modeling complex or variable outdoor terrain.

5.  **Hardware & SWaP:**
    *   Detailed modeling of Size, Weight, and Power (SWaP) constraints beyond the simplified communication range and energy model.
    *   Consideration of specific hardware components (sensors, radios, processors).
    *   Any aspect of physical hardware implementation or deployment.

6.  **Security:**
    *   Implementing or analyzing network security features (encryption, authentication, intrusion detection, jamming resistance).

7.  **Scale:**
    *   Simulation of large-scale swarms (hundreds or thousands of drones). The focus is on demonstrating the core AI logic works for a small group.
