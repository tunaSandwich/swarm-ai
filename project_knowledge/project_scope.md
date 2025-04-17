# Project Scope: AI-Driven Drone Corridor Formation Simulation

This document defines the specific boundaries for the "AI-Driven Drone Corridor Formation Simulation" project. Its purpose is to ensure the project remains focused on demonstrating the autonomous formation of a robust communication corridor between a Start and End point.

## IN SCOPE

The following elements are considered **within** the scope of this project:

1.  **Simulation Environment & Core:**
    *   Leveraging the existing Python-based simulation framework (`Drone`, `Environment` classes).
    *   Simulation within a 2D bounded area (`SIMULATION_AREA_SIZE`).
    *   Managing a swarm of `NUM_DRONES` (target range: 15-30+).
    *   Utilizing NumPy for position/vector calculations.

2.  **Path Definition:**
    *   Defining fixed **Start Point** and **End Point** coordinates within the simulation area.
    *   Defining target parameters for the communication corridor, such as:
        *   `CORRIDOR_WIDTH`: The desired lateral width of the drone band relative to the direct Start-End line.
        *   `TARGET_DENSITY` or `TARGET_SPACING`: A goal for how closely packed drones should be within the corridor.

3.  **AI/ML for Corridor Formation:**
    *   Implementation and testing of **one or two** specific AI/ML algorithms (e.g., RL or SI) adapted for **collective corridor formation**.
    *   The AI's primary goal is to guide drones to positions that:
        *   Lie within the defined `CORRIDOR_WIDTH`.
        *   Contribute to achieving the `TARGET_DENSITY`.
        *   Maintain connectivity primarily with neighbors *along the corridor's direction* (towards Start/End) and *sideways within the corridor*.
        *   Implicitly promote progress from Start towards End.

4.  **Network & Communication Model:**
    *   Using NetworkX to model connectivity based on `COMM_RANGE`, as already implemented.
    *   Focusing analysis on connectivity *within* the formed corridor structure.
    *   Calculating metrics relevant to corridor quality (see Metrics section).

5.  **Energy Model:**
    *   Incorporating the existing energy model (`INITIAL_ENERGY`, `idle_energy_cost`, `move_energy_cost`).
    *   Potentially using energy level as input for AI decisions or triggering node state changes ('inactive').

6.  **Metrics & Analysis:**
    *   Calculating and tracking metrics specifically related to corridor formation and quality, such as:
        *   Average drone distance from the ideal Start-End centerline.
        *   Average local density within the corridor.
        *   Path continuity metrics (e.g., existence of *a* path, perhaps number of hops on the shortest path within the corridor).
        *   Number of active drones / average energy (as already implemented).
    *   Using Pandas DataFrames to store metrics over time (as already implemented).

7.  **Visualization & Reporting:**
    *   Using Matplotlib to visualize drone positions, links, and the emergent corridor structure (extending existing `render` method).
    *   Generating plots summarizing corridor-relevant metrics over time (using Matplotlib/Pandas).
    *   Writing a concise final report (~3-5 pages) explaining the corridor formation approach and results.

## OUT OF SCOPE

The following elements are explicitly **outside** the scope of this project:

1.  **Specific Data Simulation:**
    *   Simulating actual internet data packets, bandwidth, latency, throughput, QoS. The focus is on forming the *structure* for potential relay.

2.  **Advanced Network Protocols:**
    *   Implementing specific MAC or complex Network layer routing protocols beyond the connectivity graph. Routing is implicit via proximity within the corridor.

3.  **General Mesh Optimization:**
    *   Maximizing overall mesh connectivity across the *entire* simulation area (the goal is corridor-specific).

4.  **Advanced Physics & Environment:**
    *   Complex physics (aerodynamics, wind).
    *   Variable terrain or obstacle avoidance (assuming an open 2D plane).

5.  **Hardware Implementation:**
    *   SWaP constraints beyond the existing energy model and `COMM_RANGE`.
    *   Any physical hardware deployment considerations.

6.  **Advanced Security Aspects:**
    *   Encryption, authentication, intrusion detection, etc.

7.  **Explicit Chain Management:**
    *   Designing AI to manage distinct, numbered parallel chains. The goal is an emergent dense corridor, not explicit chain assignment.
