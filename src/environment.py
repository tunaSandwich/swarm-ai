import numpy as np
import networkx as nx
from .drone import Drone  # Assuming drone.py is in the same directory
import matplotlib.pyplot as plt

# Define constants for special nodes
START_NODE_ID = 'start_node'
END_NODE_ID = 'end_node'

class Environment:
    """Manages the simulation space, drones, and connectivity."""
    def __init__(self, size, num_drones, comm_range, initial_energy, move_energy_cost, idle_energy_cost, start_point, end_point, corridor_width):
        self.size = np.array(size)  # e.g., [width, height] or [width, height, depth]
        self.num_drones = num_drones
        self.comm_range = comm_range
        self.initial_energy = initial_energy
        self.move_energy_cost = move_energy_cost
        self.idle_energy_cost = idle_energy_cost
        self.drones = {}  # Dictionary to store drone objects {drone_id: Drone_instance}
        self.time = 0.0
        self.connectivity_graph = nx.Graph() # To represent drone links

        # Corridor parameters
        self.start_point = np.array(start_point, dtype=float)
        self.end_point = np.array(end_point, dtype=float)
        self.corridor_width = float(corridor_width)
        self.corridor_axis = self.end_point - self.start_point
        self.corridor_length_sq = np.dot(self.corridor_axis, self.corridor_axis)
        # Ensure corridor axis is not zero vector to avoid division by zero
        if self.corridor_length_sq < 1e-9:
             raise ValueError("Start and End points are too close.")
        self.corridor_direction = self.corridor_axis / np.sqrt(self.corridor_length_sq)

        # Add start/end points as nodes to the graph
        self.connectivity_graph.add_node(START_NODE_ID, pos=self.start_point, type='anchor')
        self.connectivity_graph.add_node(END_NODE_ID, pos=self.end_point, type='anchor')

        # --- Visualization Setup ---
        self.fig, self.ax = plt.subplots()
        plt.ion() # Turn on interactive mode
        # --- End Visualization Setup ---

        self._initialize_drones()
        self._update_connectivity()

    def _initialize_drones(self):
        """Creates and places drones randomly within the environment bounds."""
        print(f"Initializing {self.num_drones} drones near Start Point: {self.start_point}")
        # Define a small area around the start point for initial placement
        initial_spread = 5.0 # Drones will start within +/- 2.5 units of the start point

        for i in range(self.num_drones):
            # Random initial position within bounds
            # initial_pos = np.random.rand(len(self.size)) * self.size
            # New: Place near start point with a small random offset
            offset = (np.random.rand(len(self.size)) - 0.5) * initial_spread
            initial_pos = self.start_point + offset
            
            # Ensure drones start within the overall simulation bounds
            initial_pos = np.clip(initial_pos, 0, self.size)

            drone = Drone(drone_id=i,
                          initial_position=initial_pos,
                          initial_energy=self.initial_energy,
                          comm_range=self.comm_range,
                          move_energy_cost=self.move_energy_cost,
                          idle_energy_cost=self.idle_energy_cost)
            self.drones[i] = drone
            # Add drone node, connecting it to start/end if initially in range
            self.connectivity_graph.add_node(i, pos=drone.position, type='drone')
            # Don't add drone-to-anchor edges here, _update_connectivity handles it

        print(f"Environment initialized with {self.num_drones} drones.")
        self._update_connectivity() # Initial connectivity calculation

    def _update_connectivity(self):
        """Updates the connectivity graph based on current drone positions and anchors."""
        # Keep track of nodes that should exist (active drones + anchors)
        active_drone_ids = {id for id, drone in self.drones.items() if drone.state == "active"}
        current_node_ids = set(active_drone_ids)
        current_node_ids.add(START_NODE_ID) # Ensure anchors are always considered
        current_node_ids.add(END_NODE_ID)

        # Remove nodes from graph if drone is no longer active
        # Anchors (start/end) should never be removed unless logic changes
        nodes_to_remove = [node for node in self.connectivity_graph.nodes() if node not in current_node_ids]
        self.connectivity_graph.remove_nodes_from(nodes_to_remove)

        # Add nodes if any drone became active again or if anchors were somehow removed
        for drone_id in active_drone_ids:
            if drone_id not in self.connectivity_graph:
                self.connectivity_graph.add_node(drone_id, pos=self.drones[drone_id].position, type='drone')
        if START_NODE_ID not in self.connectivity_graph:
             self.connectivity_graph.add_node(START_NODE_ID, pos=self.start_point, type='anchor')
        if END_NODE_ID not in self.connectivity_graph:
             self.connectivity_graph.add_node(END_NODE_ID, pos=self.end_point, type='anchor')

        # Update positions and edges for active drones and anchors
        self.connectivity_graph.clear_edges()
        # active_nodes = list(self.connectivity_graph.nodes()) # Now includes anchors

        # Update drone positions in the graph
        for drone_id in active_drone_ids:
            drone = self.drones[drone_id]
            self.connectivity_graph.nodes[drone_id]['pos'] = drone.position

        # Calculate drone-to-drone connectivity
        active_drone_list = list(active_drone_ids)
        for i in range(len(active_drone_list)):
            d1_id = active_drone_list[i]
            d1 = self.drones[d1_id]
            for j in range(i + 1, len(active_drone_list)):
                d2_id = active_drone_list[j]
                d2 = self.drones[d2_id]
                distance = np.linalg.norm(d1.position - d2.position)
                if distance <= self.comm_range:
                    self.connectivity_graph.add_edge(d1_id, d2_id, weight=distance)

        # Calculate drone-to-anchor connectivity
        for drone_id in active_drone_ids:
            drone = self.drones[drone_id]
            dist_to_start = np.linalg.norm(drone.position - self.start_point)
            dist_to_end = np.linalg.norm(drone.position - self.end_point)

            if dist_to_start <= self.comm_range:
                self.connectivity_graph.add_edge(drone_id, START_NODE_ID, weight=dist_to_start)
            if dist_to_end <= self.comm_range:
                self.connectivity_graph.add_edge(drone_id, END_NODE_ID, weight=dist_to_end)

        # print(f"Time {self.time:.2f}: Connectivity updated. Edges: {self.connectivity_graph.number_of_edges()}")

    def step(self, dt):
        """Advances the simulation by one time step."""
        self.time += dt
        # print(f"\n--- Stepping Simulation to Time: {self.time:.2f} ---") # Reduce print frequency

        # 1. Get actions from AI/Control logic (placeholder)
        actions = self._get_actions()

        # 2. Apply actions (move drones) - Pass environment size
        for drone_id, action in actions.items():
            if drone_id in self.drones and self.drones[drone_id].state == "active":
                velocity = action # Assuming action is directly a velocity vector for now
                self.drones[drone_id].move(velocity, dt, self.size)

        # 3. Update drone states (energy depletion, state changes)
        for drone in self.drones.values():
            drone.update_state(dt)

        # 4. Update connectivity based on new positions and states
        self._update_connectivity()

        # 5. Calculate metrics
        metrics = self._calculate_metrics()

        # 6. Check termination conditions (placeholder)
        done = self._check_termination()

        return metrics, done

    def _get_actions(self):
        """Calculates drone actions (velocity) using Boids/Flocking rules for corridor formation."""
        actions = {}

        # --- Boids Parameters ---
        MAX_SPEED = 5.0           # Max speed a drone can travel per second
        SEPARATION_DISTANCE = 10.0  # Minimum desired distance between drones
        # Ensure SEPARATION_DISTANCE < self.comm_range
        if SEPARATION_DISTANCE >= self.comm_range:
             print(f"Warning: SEPARATION_DISTANCE ({SEPARATION_DISTANCE}) should be less than COMM_RANGE ({self.comm_range})")
             # Adjust if necessary, ensure it's significantly smaller than comm_range
             SEPARATION_DISTANCE = self.comm_range * 0.5 
             print(f"Adjusted SEPARATION_DISTANCE to {SEPARATION_DISTANCE}")


        # Weights for steering behaviors (PLACEHOLDERS - need tuning)
        WEIGHT_SEPARATION = 1.5  # Slightly reduced
        WEIGHT_COHESION = 0.2    # Significantly reduced
        WEIGHT_CORRIDOR = 1.2    # Kept same
        WEIGHT_ALIGNMENT = 1.5   # Increased
        # --- End Boids Parameters ---

        active_drones = {id: d for id, d in self.drones.items() if d.state == 'active'}
        active_drone_ids = list(active_drones.keys())
        num_active = len(active_drone_ids)

        if num_active == 0:
            return actions # No active drones

        # Pre-calculate positions for efficiency
        positions = {id: d.position for id, d in active_drones.items()}

        for drone_id in active_drone_ids:
            current_drone = active_drones[drone_id]
            pos = positions[drone_id]

            separation_vector = np.zeros_like(pos)
            cohesion_vector = np.zeros_like(pos)
            corridor_vector = np.zeros_like(pos)
            alignment_vector = np.zeros_like(pos) # NEW: Initialize alignment vector
            
            neighbor_positions_for_cohesion = []
            num_separation_neighbors = 0

            # --- Calculate Steering Vectors ---
            for other_id in active_drone_ids:
                if drone_id == other_id:
                    continue

                other_pos = positions[other_id]
                vec_to_other = other_pos - pos
                dist_sq = np.dot(vec_to_other, vec_to_other)
                dist = np.sqrt(dist_sq) if dist_sq > 1e-9 else 0 # Avoid sqrt(0)

                # 1. Separation
                if dist > 1e-9 and dist < SEPARATION_DISTANCE:
                    # Force is stronger when closer, pointing away from the neighbor
                    separation_vector -= (vec_to_other / dist) * (1.0 - dist / SEPARATION_DISTANCE)
                    num_separation_neighbors += 1

                # 2. Cohesion (Neighbors within comm_range but outside separation distance)
                elif dist > 1e-9 and dist < self.comm_range:
                     neighbor_positions_for_cohesion.append(other_pos)

            # Average Separation Vector (Optional: can prevent extreme forces in dense clusters)
            # if num_separation_neighbors > 0:
            #     separation_vector /= num_separation_neighbors


            # 2. Cohesion Calculation (if neighbors exist)
            if neighbor_positions_for_cohesion:
                center_of_mass = np.mean(neighbor_positions_for_cohesion, axis=0)
                vec_to_center = center_of_mass - pos
                # Steer towards the center of mass - scale by distance? Or keep simple?
                # For now, use the raw vector towards the center. Normalizing might be needed during tuning.
                cohesion_vector = vec_to_center
                # Optional: Cap cohesion force magnitude if needed


            # 3. Corridor Following
            # Vector from the start point of the corridor line to the drone
            drone_vec_from_start = pos - self.start_point
            # Project this vector onto the corridor axis vector to find the projection length
            # Note: self.corridor_axis is end_point - start_point
            # Note: self.corridor_length_sq is dot(self.corridor_axis, self.corridor_axis)
            projection_scalar_ratio = np.dot(drone_vec_from_start, self.corridor_axis) / self.corridor_length_sq
            
            # Find the closest point ON THE INFINITE LINE defined by start/end
            closest_center_pt_on_line = self.start_point + projection_scalar_ratio * self.corridor_axis
            
            # Vector from the closest point on the line to the drone
            perp_vec_to_drone = pos - closest_center_pt_on_line
            dist_to_centerline = np.linalg.norm(perp_vec_to_drone)

            # Apply force only if outside the desired corridor width
            if dist_to_centerline > self.corridor_width / 2.0:
                # Vector pointing back towards the centerline (opposite of perp_vec_to_drone)
                # Normalize to get direction only
                direction_to_center = -perp_vec_to_drone / dist_to_centerline
                
                # Scale force maybe? For now, use constant magnitude push towards line.
                # magnitude = (dist_to_centerline - self.corridor_width / 2.0) # Linear scaling factor
                magnitude = 1.0 # Constant force magnitude
                corridor_vector = direction_to_center * magnitude

            # 4. Alignment / Forward Thrust (NEW)
            # Simplest form: always try to move along the corridor direction
            alignment_vector = self.corridor_direction


            # --- Combine Steering Vectors ---
            # Weight the vectors - Tuning these weights is crucial!
            desired_velocity = (separation_vector * WEIGHT_SEPARATION +
                                cohesion_vector * WEIGHT_COHESION +
                                corridor_vector * WEIGHT_CORRIDOR +
                                alignment_vector * WEIGHT_ALIGNMENT) # Add alignment


            # --- Normalize and Cap Speed ---
            speed = np.linalg.norm(desired_velocity)
            if speed > MAX_SPEED:
                desired_velocity = (desired_velocity / speed) * MAX_SPEED
            elif speed < 1e-6: # Handle potential zero vector if no forces act on drone
                # Default behavior: maybe a tiny random nudge or push towards end?
                # Or just stay put if truly balanced. Let's default to zero for now.
                desired_velocity = np.zeros_like(pos)
                # Alternative: Add a small default forward thrust along corridor direction
                # desired_velocity = self.corridor_direction * MAX_SPEED * 0.1

            actions[drone_id] = desired_velocity

        # Assign zero velocity to inactive drones explicitly
        for drone_id, drone in self.drones.items():
            if drone.state != 'active':
                 # Ensure inactive drones have a zero velocity action
                 actions[drone_id] = np.zeros_like(drone.position) 

        return actions

    def _calculate_metrics(self):
        """Calculates and returns relevant simulation metrics."""
        num_edges = self.connectivity_graph.number_of_edges()
        # Get all nodes currently in the graph (includes drones and anchors)
        all_graph_nodes = list(self.connectivity_graph.nodes())
        # Get only the active drone nodes *currently* in the graph
        active_drone_nodes_in_graph = [nid for nid, data in self.connectivity_graph.nodes(data=True) if data.get('type') == 'drone']
        num_active_drones = len(active_drone_nodes_in_graph)

        # Calculate average energy and corridor metrics for *active drones*
        total_energy = 0
        total_dist_from_center = 0
        num_in_corridor = 0

        # Iterate only over active drone nodes
        if num_active_drones > 0:
            for drone_id in active_drone_nodes_in_graph:
                # Ensure drone_id is still valid (should be, but safe check)
                if drone_id in self.drones:
                    drone = self.drones[drone_id]
                    total_energy += drone.energy

                    # Calculate distance from centerline
                    drone_vec = drone.position - self.start_point
                    # Use projection formula, avoid division by zero if corridor length is somehow zero
                    projection_scalar = 0.0
                    if self.corridor_length_sq > 1e-9:
                         projection_scalar = np.dot(drone_vec, self.corridor_axis) / self.corridor_length_sq
                    # We care about distance even if slightly outside the projected segment [0,1] for containment check
                    # projection_scalar = np.clip(projection_scalar, 0, 1)
                    closest_center_pt = self.start_point + projection_scalar * self.corridor_axis
                    dist_to_center = np.linalg.norm(drone.position - closest_center_pt)

                    total_dist_from_center += dist_to_center
                    if dist_to_center <= self.corridor_width / 2.0:
                        num_in_corridor += 1
                else:
                    # This case should ideally not happen if graph/drone dict are synced
                    print(f"Warning: Drone ID {drone_id} found in graph but not in self.drones during metric calculation.")

            avg_energy = total_energy / num_active_drones
            avg_dist_from_center = total_dist_from_center / num_active_drones
            percent_in_corridor = (num_in_corridor / num_active_drones) * 100.0
        else:
            avg_energy = 0
            avg_dist_from_center = 0
            percent_in_corridor = 0

        # Calculate overall graph metrics (including anchors)
        # avg_degree = (2 * num_edges / len(all_graph_nodes)) if len(all_graph_nodes) > 0 else 0
        # is_connected = False
        # if len(all_graph_nodes) > 0:
        #     is_connected = nx.is_connected(self.connectivity_graph)
            # Consider if 'is_connected' should refer to the whole graph or just drones

        # --- Connectivity Metrics (focused on Drones) ---
        drone_subgraph = self.connectivity_graph.subgraph(active_drone_nodes_in_graph)

        if drone_subgraph.number_of_nodes() > 0:
            # Is the drone subgraph connected?
            is_drone_swarm_connected = nx.is_connected(drone_subgraph)
            # Largest component size among *drones* only
            largest_component_nodes = max(nx.connected_components(drone_subgraph), key=len)
            largest_component_size = len(largest_component_nodes)
            num_components = nx.number_connected_components(drone_subgraph)
            avg_drone_degree = (2 * drone_subgraph.number_of_edges() / drone_subgraph.number_of_nodes())
        else:
            is_drone_swarm_connected = False # No drones -> not connected
            largest_component_size = 0
            num_components = 0
            avg_drone_degree = 0

        metrics = {
            "time": self.time,
            "num_active_drones": num_active_drones,
            # "num_edges_total": num_edges, # Optionally report total edges including anchors
            "num_edges_drones": drone_subgraph.number_of_edges(), # Edges between drones only
            "average_degree_drones": avg_drone_degree, # Avg degree within drone subgraph
            "is_swarm_connected": is_drone_swarm_connected, # Connectivity of drone swarm
            "largest_component_size": largest_component_size, # Size relative to active drones
            "num_components": num_components, # Number of separate drone groups
            "average_energy": avg_energy,
            "avg_dist_from_center": avg_dist_from_center,
            "percent_in_corridor": percent_in_corridor
        }
        # print(f"Metrics: {metrics}") # Reduce print frequency
        return metrics

    def _check_termination(self):
        """Checks if the simulation should terminate."""
        # Example: Terminate if time limit reached or all drones inactive
        if self.time >= 10.0: # Arbitrary time limit for now
            # print("Termination condition met: Time limit reached.")
            return True
        # Check if there are any active drones left in the dictionary
        if not any(d.state == "active" for d in self.drones.values()):
            print("Termination condition met: All drones inactive.")
            return True
        return False

    def render(self):
        """Visualizes the current state of the simulation using Matplotlib."""
        self.ax.clear()

        # Set plot limits slightly larger than environment size
        margin = 5
        self.ax.set_xlim(-margin, self.size[0] + margin)
        self.ax.set_ylim(-margin, self.size[1] + margin)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel("X coordinate")
        self.ax.set_ylabel("Y coordinate")
        self.ax.set_title(f"Drone Swarm Simulation - Time: {self.time:.2f}s")

        # --- Draw Corridor ---
        # Draw centerline
        self.ax.plot([self.start_point[0], self.end_point[0]],
                     [self.start_point[1], self.end_point[1]], 'k--', alpha=0.5, label='Centerline')
        # Draw boundaries (perpendicular lines)
        perp_vec = np.array([-self.corridor_direction[1], self.corridor_direction[0]]) * (self.corridor_width / 2.0)
        s1 = self.start_point + perp_vec
        e1 = self.end_point + perp_vec
        s2 = self.start_point - perp_vec
        e2 = self.end_point - perp_vec
        self.ax.plot([s1[0], e1[0]], [s1[1], e1[1]], 'k:', alpha=0.3)
        self.ax.plot([s2[0], e2[0]], [s2[1], e2[1]], 'k:', alpha=0.3)
        # Draw Start/End points distinctly
        self.ax.plot(self.start_point[0], self.start_point[1], 'go', markersize=10, label='Start Point')
        self.ax.plot(self.end_point[0], self.end_point[1], 'ro', markersize=10, label='End Point')
        # --- End Corridor ---

        # --- Draw Drones and Connectivity ---
        node_positions = nx.get_node_attributes(self.connectivity_graph, 'pos')
        node_types = nx.get_node_attributes(self.connectivity_graph, 'type')
        drone_nodes = [n for n, d in self.drones.items() if d.state == 'active']
        inactive_drone_nodes = [n for n, d in self.drones.items() if d.state != 'active']
        anchor_nodes = [START_NODE_ID, END_NODE_ID]

        # Draw active drones
        active_pos = {n: node_positions[n] for n in drone_nodes if n in node_positions}
        if active_pos:
            nx.draw_networkx_nodes(self.connectivity_graph, pos=active_pos,
                                   nodelist=active_pos.keys(), node_color='blue', node_size=50, ax=self.ax, label='Active Drone')

        # Draw inactive drones
        inactive_pos = {n: self.drones[n].position for n in inactive_drone_nodes} # Get current pos even if not in graph
        if inactive_pos:
             pos_array = np.array(list(inactive_pos.values()))
             self.ax.scatter(pos_array[:, 0], pos_array[:, 1], color='gray', s=30, alpha=0.5, label='Inactive Drone')

        # Draw anchor points (using positions stored in graph)
        anchor_pos = {n: node_positions[n] for n in anchor_nodes if n in node_positions}
        # Colors defined earlier work better than node_color here
        # nx.draw_networkx_nodes(self.connectivity_graph, pos=anchor_pos,
        #                        nodelist=anchor_nodes, node_color=['green', 'red'], node_size=100, ax=self.ax, node_shape='s')

        # Draw edges (connectivity)
        nx.draw_networkx_edges(self.connectivity_graph, pos=node_positions,
                               edge_color='gray', alpha=0.5, ax=self.ax)

        # --- Add Legend (Combine handles manually) ---
        handles, labels = self.ax.get_legend_handles_labels()
        # Add custom handles if needed (e.g., if scatter wasn't labeled)
        from matplotlib.lines import Line2D
        custom_handles = [Line2D([0], [0], marker='o', color='w', label='Active Drone', markersize=7, markerfacecolor='blue'),
                          Line2D([0], [0], marker='o', color='w', label='Inactive Drone', markersize=7, markerfacecolor='gray', alpha=0.5)]
                         # Line2D([0], [0], marker='s', color='w', label='Start/End', markersize=10, markerfacecolor='green')] # Less reliable color mapping

        # Filter handles based on what was actually plotted
        unique_labels = {}
        final_handles = []
        for handle, label in zip(handles + custom_handles, labels + [h.get_label() for h in custom_handles]):
             if label not in unique_labels:
                 unique_labels[label] = handle
                 final_handles.append(handle)

        self.ax.legend(handles=final_handles, loc='upper right', fontsize='small')
        # --- End Legend ---

        plt.draw()
        plt.pause(0.001) # Small pause to allow plot to update

    def close_visualization(self):
        """Closes the visualization window."""
        plt.ioff() # Turn off interactive mode
        plt.show() # Keep the final plot window open until manually closed 
