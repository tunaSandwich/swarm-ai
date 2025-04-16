# Simulation Model Definition

This document outlines the core rules, components, and parameters governing the simulation environment for the "AI-Driven Connectivity Maintenance" project. Simplicity and focus on demonstrating the AI positioning logic are key priorities.

## 1. Environment

*   **Type:** 2D Bounded Area.
*   **Dimensions:** A square area of size `AREA_WIDTH` x `AREA_HEIGHT` (e.g., 100x100 arbitrary units). Define these as configurable parameters.
*   **Boundaries:** Drones are constrained to stay within these boundaries.
    *   *Boundary Handling:* Choose one method (configurable parameter):
        *   **Stop:** Drones stop moving if their next step would cross the boundary.
        *   **Bounce:** Drones reflect off the boundary (velocity component perpendicular to the boundary is reversed).
        *   *(Avoid Wrap-around/Torus for this project as it complicates connectivity visualization)*.

## 2. Drone Model

*   **Representation:** Each drone is an individual agent/object.
*   **State Variables:** Each drone `i` maintains at least the following state:
    *   `id`: Unique identifier.
    *   `position` (`pos_x`, `pos_y`): Current coordinates within the environment.
    *   `velocity` (`vel_x`, `vel_y`): Current movement vector (used for position updates if not using direct discrete actions). Can be capped by `MAX_SPEED`.
    *   `energy` (`E`): Current energy level (float, e.g., 0.0 to `E_MAX`).
    *   `status`: Current status (e.g., 'active', 'low_energy', 'inactive').

## 3. Movement Model

*   **Time:** Discrete time steps (`dt`).
*   **Position Update:** Based on the chosen AI action/velocity:
    *   *If using discrete actions (N/S/E/W/Stay):* `position(t+dt) = position(t) + action_vector * MOVE_STEP_SIZE`.
    *   *If using velocity (e.g., from PSO):* `position(t+dt) = position(t) + velocity(t) * dt`. Ensure velocity does not exceed `MAX_SPEED`.
*   **Parameters:**
    *   `MOVE_STEP_SIZE`: Distance covered in one move action (if using discrete actions).
    *   `MAX_SPEED`: Maximum allowed speed (if using velocity).

## 4. Communication Model

*   **Type:** Simple Disk Model (Threshold-based).
*   **Rule:** A bidirectional communication link exists between drone `i` and drone `j` if and only if the Euclidean distance between them is less than `COMM_RANGE`.
    *   `distance(i, j) = sqrt((pos_x_i - pos_x_j)^2 + (pos_y_i - pos_y_j)^2)`
    *   `link_exists(i, j) = (distance(i, j) < COMM_RANGE)`
*   **Assumptions:**
    *   Instantaneous link establishment/breakage based on distance.
    *   Perfect communication within range (no noise, interference, or packet loss).
    *   Symmetric links (if i can hear j, j can hear i).
*   **Parameter:**
    *   `COMM_RANGE`: The maximum distance for communication (configurable).

## 5. Energy Model

*   **Initial Energy:** All drones start with `E = E_MAX`.
*   **Energy Depletion:** At each time step `dt`:
    *   **Idle Cost:** Energy decreases by `E_IDLE_COST * dt` just for being active.
    *   **Movement Cost:** Energy decreases by `E_MOVE_COST * distance_moved` (or `E_MOVE_COST` per move action if using discrete steps).
*   **Low Energy Threshold:** Define `E_LOW_THRESHOLD`.
    *   *Consequence:* When `E < E_LOW_THRESHOLD`, the drone might enter a 'low_energy' state. This could:
        *   Trigger different behavior in the AI (e.g., prioritize staying still).
        *   Eventually lead to 'inactive' state (node removal) if energy reaches zero.
*   **Parameters:**
    *   `E_MAX`: Starting energy level.
    *   `E_IDLE_COST`: Energy consumed per time unit while idle.
    *   `E_MOVE_COST`: Energy consumed per unit distance moved or per move action.
    *   `E_LOW_THRESHOLD`: Threshold for low energy state/behavior change.

## 6. Time Model

*   **Type:** Discrete Time Steps.
*   **Step Duration (`dt`):** A fixed time interval for each simulation update cycle (e.g., `dt = 0.1` seconds, though the absolute value might not matter as much as relative costs).
*   **Simulation Loop:** In each step:
    1.  All drones perceive their state / environment.
    2.  All drones compute their next action (using AI logic).
    3.  Update energy levels based on previous action/idle state.
    4.  Update positions based on computed actions/velocities.
    5.  Update communication links based on new positions.
    6.  Calculate metrics.
    7.  Visualize state.
    8.  Repeat.
