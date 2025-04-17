# Connectivity & Corridor Quality Metrics

This document describes metrics used to evaluate the success of the drone swarm in forming the desired **communication corridor** and maintaining its structure. These metrics are crucial for analyzing simulation results and potentially guiding the AI learning process.

## Primary Metrics for Corridor Quality

These metrics directly assess how well the swarm achieves the specific geometric and structural goals of the corridor.

1.  **Corridor Adherence (Drone Distance from Centerline):**
    *   **Definition:** Measures how well drones stay within the intended corridor boundaries.
    *   **Calculation:** For each active drone, calculate its perpendicular distance to the ideal Start-End centerline. Compute statistics:
        *   Average distance from centerline.
        *   Maximum distance from centerline.
        *   Percentage of drones *within* `CORRIDOR_WIDTH / 2` of the centerline.
    *   **Goal:** Minimize average/max distance; maximize percentage within width.

2.  **Local Density within Corridor:**
    *   **Definition:** Measures if the drone packing inside the corridor matches the target.
    *   **Calculation:** For each drone *within* the corridor boundaries:
        *   Count the number of neighbors within a defined local radius (`R_density`).
        *   Calculate the average density across all drones within the corridor.
    *   **Goal:** Average local density should approach the value implied by `TARGET_DENSITY` or `TARGET_SPACING`. Monitor distribution to check for excessive clumping or gaps.

3.  **Path Continuity & Efficiency:**
    *   **Definition:** Assesses if a functional communication path exists from Start to End *through the corridor*.
    *   **Calculation:**
        *   **Corridor Subgraph:** Create a NetworkX graph containing only drones currently *within* the `CORRIDOR_WIDTH` and the links between them.
        *   **Path Existence:** Identify a "source group" (drones within corridor near Start Point) and a "target group" (drones within corridor near End Point). Check if *any* path exists in the corridor subgraph between any node in the source group and any node in the target group (`nx.has_path` or check connected components).
        *   **Shortest Path Hops:** If a path exists, find the length (number of hops) of the shortest path within the corridor subgraph between the source and target groups (`nx.shortest_path_length`).
    *   **Goal:** Maintain path existence consistently. Minimize the number of hops (implies efficient spacing).

## Secondary / General Metrics (Still Useful)

These provide broader context about the swarm's state.

4.  **Number of Active Drones:**
    *   **Definition:** Total count of drones not in an 'inactive' state.
    *   **Calculation:** Simple count.
    *   **Goal:** Monitor impact of energy depletion.

5.  **Average Energy Level:**
    *   **Definition:** Mean energy across all active drones.
    *   **Calculation:** Sum energy / number of active drones.
    *   **Goal:** Monitor overall energy consumption patterns.

## Metrics to De-emphasize (for this goal)

*   **Overall Average Node Degree (across all drones):** Less relevant as connections outside the corridor are not prioritized.
*   **LCC Size / Algebraic Connectivity (of the *entire* graph):** The overall graph might be disconnected if drones correctly form *only* the corridor. Connectivity *within the corridor subgraph* (metric #3) is what matters most.

## Implementation Notes
*   Calculations involving the Start-End centerline require basic vector math (projections, perpendicular distances). NumPy is ideal for this.
*   NetworkX remains essential for graph creation (especially the corridor subgraph) and path analysis functions.
*   Clearly define the "near Start" and "near End" regions for path continuity checks.

## Recommendation for Project
*   **Primary Focus:** Track **Corridor Adherence (Avg/Max Distance, % In Width)**, **Local Density (Avg within Corridor)**, and **Path Continuity (Existence, Shortest Hops)** over time.
*   **Secondary:** Monitor **Active Drones** and **Average Energy**.
