# Simulation Model Definition (Corridor Formation)

This document outlines the core rules, components, and parameters governing the simulation environment, specifically tailored for the "AI-Driven Drone Corridor Formation" project.

## 1. Environment

*   **Type:** 2D Bounded Area.
*   **Dimensions:** `AREA_WIDTH` x `AREA_HEIGHT` (e.g., 200x100 units). Note: Area might need to be more rectangular to accommodate the path.
*   **Boundaries:** Drones constrained within boundaries (Stop or Bounce method).

## 2. Path & Corridor Definition

*   **Start Point (`START_POINT`):** Fixed coordinates (e.g., `[10, AREA_HEIGHT/2]`).
*   **End Point (`END_POINT`):** Fixed coordinates (e.g., `[AREA_WIDTH - 10, AREA_HEIGHT/2]`).
*   **Ideal Centerline:** The straight line segment connecting `START_POINT` and `END_POINT`. *(Needed for calculations)*.
*   **Corridor Parameters:**
    *   `CORRIDOR_WIDTH`: The target maximum perpendicular distance from the centerline within which drones should operate (e.g., 10 units).
    *   `TARGET_DENSITY` OR `TARGET_SPACING`: A parameter defining the desired packing of drones within the corridor. Could be:
        *   Target number of neighbors within a local radius `R_density`.
        *   Target average distance between adjacent drones along the path.
*   **Drone Awareness (Assumption):** Assume drones have access to `START_POINT`, `END_POINT` coordinates, or can calculate the vector pointing along the ideal path centerline. *(How this info is obtained is abstracted away for the simulation)*.

## 3. Drone Model

*   **Representation:** Each drone is an individual agent/object.
*   **State Variables:**
    *   `id`: Unique identifier.
    *   `position` (`pos_x`, `pos_y`).
    *   `velocity` (`vel_x`, `vel_y`) - capped by `MAX_SPEED`.
    *   `energy` (`E`) - range `0.0` to `E_MAX`.
    *   `status`: 'active', 'low_energy', 'inactive'.
    *   *(Potentially add derived state calculated each step: `distance_from_centerline`, `local_density`)*.

## 4. Movement Model

*   **Time:** Discrete time steps (`dt`).
*   **Position Update:** Based on AI output (discrete action or velocity adjustment). `position(t+dt) = position(t) + move_vector`.
*   **Parameters:**
    *   `MOVE_STEP_SIZE` (if discrete actions).
    *   `MAX_SPEED` (if velocity control).

## 5. Communication Model

*   **Type:** Simple Disk Model (Threshold-based).
*   **Rule:** Link exists between drone `i` and `j` if `distance(i, j) < COMM_RANGE`.
*   **Assumptions:** Instantaneous, perfect, symmetric links within range.
*   **Parameter:** `COMM_RANGE`. *(Note: `COMM_RANGE` implicitly influences achievable `TARGET_SPACING`)*.

## 6. Energy Model

*   **Initial Energy:** `E_MAX`.
*   **Energy Depletion:**
    *   Idle Cost: `E_IDLE_COST * dt`.
    *   Movement Cost: `E_MOVE_COST * distance_moved`.
*   **Low Energy Threshold:** `E_LOW_THRESHOLD` triggering state change or AI behavior modification.
*   **Parameters:** `E_MAX`, `E_IDLE_COST`, `E_MOVE_COST`, `E_LOW_THRESHOLD`.

## 7. Time Model

*   **Type:** Discrete Time Steps (`dt`).
*   **Simulation Loop:** Perceive -> AI Compute Action -> Update Energy -> Update Position -> Update Links -> Calculate Metrics -> Visualize -> Repeat.
