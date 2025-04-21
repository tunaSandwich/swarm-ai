import numpy as np
import networkx as nx
from .drone import Drone  # Assuming drone.py is in the same directory
import matplotlib.pyplot as plt

# Define constants for special nodes
START_NODE_ID = 'start_node'
END_NODE_ID = 'end_node'

class Environment:
    """Manages the simulation space, drones, and connectivity."""
    def __init__(self, size, num_drones, comm_range, initial_energy, move_energy_cost, idle_energy_cost, start_point, end_point, corridor_width,
                 # New parameters for corridor boids
                 anchor_range=20.0, min_neighbors=2,
                 weight_separation=1.5, weight_alignment=1.0, weight_cohesion=1.0,
                 weight_corridor=1.0, weight_anchor=1.5, weight_connectivity=2.0, max_speed=5.0,
                 separation_distance=10.0, weight_goal=1.0,
                 weight_start_anchor=0.6):
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
        self.corridor_length = np.linalg.norm(self.corridor_axis)
        self.corridor_length_sq = self.corridor_length * self.corridor_length
        # Ensure corridor axis is not zero vector to avoid division by zero
        if self.corridor_length_sq < 1e-9:
             raise ValueError("Start and End points are too close.")
        self.corridor_direction = self.corridor_axis / self.corridor_length

        # Boids Parameters
        self.max_speed = max_speed
        self.separation_distance = separation_distance
        # Ensure SEPARATION_DISTANCE < self.comm_range
        if self.separation_distance >= self.comm_range:
             print(f"Warning: SEPARATION_DISTANCE ({self.separation_distance}) should be less than COMM_RANGE ({self.comm_range})")
             self.separation_distance = self.comm_range * 0.8 # Adjust to be clearly smaller
             print(f"Adjusted SEPARATION_DISTANCE to {self.separation_distance}")

        # Weights for steering behaviors
        self.weight_separation = weight_separation
        self.weight_alignment = weight_alignment
        self.weight_cohesion = weight_cohesion
        self.weight_corridor = weight_corridor
        self.weight_anchor = weight_anchor
        self.weight_start_anchor = weight_start_anchor
        self.weight_connectivity = weight_connectivity
        self.weight_goal = weight_goal

        # New Rule Parameters
        self.anchor_range = anchor_range # Distance from start/end to apply anchor force
        self.min_neighbors = min_neighbors # Min neighbors before connectivity rescue kicks in

        # State flag for established corridor
        self.corridor_established = False

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
        """Each drone is initialized near the start_point with a small random offset, ensuring they stay within the environment boundaries."""
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

        # Update corridor established flag based on END node connections
        end_connections = metrics.get('end_node_drone_neighbors', 0)
        self.corridor_established = end_connections >= 3

        if self.corridor_established:
            # Optional: Print only once when established
            if not hasattr(self, '_corridor_print_done') or not self._corridor_print_done:
                 print(f"Time {self.time:.2f}: Corridor established! Drones holding position.")
                 self._corridor_print_done = True
        else:
             self._corridor_print_done = False # Reset if connection lost

        # 6. Check termination conditions (placeholder)
        done = self._check_termination()

        return metrics, done

    def _get_actions(self):
        """Calculates drone actions (velocity) using enhanced Boids rules for corridor formation."""
        actions = {}

        # Check if corridor is established - if so, all drones hold position
        if self.corridor_established:
            for drone_id in self.drones:
                 # Check if drone_id exists and get its position dimension
                 if drone_id in self.drones:
                     actions[drone_id] = np.zeros_like(self.drones[drone_id].position)
                 # Else: If somehow a drone_id isn't in self.drones, skip (shouldn't happen)
            return actions

        active_drones = {id: d for id, d in self.drones.items() if d.state == 'active'}
        active_drone_ids = list(active_drones.keys())
        num_active = len(active_drone_ids)

        if num_active == 0:
            return actions # No active drones

        # Pre-calculate positions and velocities for efficiency
        positions = {id: d.position for id, d in active_drones.items()}
        # Assuming drones have a 'velocity' attribute. If not, we need to add it or estimate it.
        # Let's add a placeholder velocity if it doesn't exist
        velocities = {id: getattr(d, 'velocity', np.zeros_like(d.position)) for id, d in active_drones.items()}


        for drone_id in active_drone_ids:
            current_drone = active_drones[drone_id]
            pos = positions[drone_id]
            vel = velocities[drone_id] # Current velocity for alignment calculation

            # Initialize steering vectors
            separation_vector = np.zeros_like(pos)
            alignment_vector = np.zeros_like(pos) # Will store average neighbor velocity
            cohesion_vector = np.zeros_like(pos) # Will store vector to center of mass
            corridor_vector = np.zeros_like(pos) # Steers towards corridor centerline
            anchor_vector = np.zeros_like(pos)   # Steers towards Start/End base
            connectivity_vector = np.zeros_like(pos) # Steers towards nearest neighbor if isolated
            goal_vector = np.zeros_like(pos) # NEW: Steers towards the end point along the corridor

            neighbor_positions = []
            neighbor_velocities = []
            neighbor_distances = {} # Store distances for finding nearest neighbor {other_id: dist}
            num_neighbors = 0
            nearest_neighbor_id = -1
            min_dist_sq = float('inf')

            # --- Find Neighbors and Calculate Separation/Cohesion/Alignment ---
            for other_id in active_drone_ids:
                if drone_id == other_id:
                    continue

                other_pos = positions[other_id]
                other_vel = velocities[other_id]
                vec_to_other = other_pos - pos
                dist_sq = np.dot(vec_to_other, vec_to_other)

                # Check if within communication range (for cohesion, alignment, connectivity)
                if dist_sq < self.comm_range**2:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-9 else 1e-9 # Avoid division by zero
                    num_neighbors += 1
                    neighbor_positions.append(other_pos)
                    neighbor_velocities.append(other_vel)
                    neighbor_distances[other_id] = dist
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        nearest_neighbor_id = other_id


                    # 1. Separation (within separation distance)
                    if dist < self.separation_distance:
                        # Force is stronger when closer, pointing away from the neighbor
                        separation_vector -= (vec_to_other / dist) * (1.0 - dist / self.separation_distance)


            # --- Finalize Cohesion and Alignment (if neighbors exist) ---
            if num_neighbors > 0:
                # 2. Cohesion: Steer towards the center of mass of neighbors
                center_of_mass = np.mean(neighbor_positions, axis=0)
                vec_to_center = center_of_mass - pos
                # Limit the desired velocity from cohesion to max_speed
                desired_cohesion_vel = (vec_to_center / np.linalg.norm(vec_to_center)) * self.max_speed if np.linalg.norm(vec_to_center) > 1e-9 else np.zeros_like(pos)
                cohesion_vector = self._limit_force(desired_cohesion_vel - vel) # Calculate steering force and limit it

                # 3. Alignment: Steer towards the average heading of neighbors
                average_velocity = np.mean(neighbor_velocities, axis=0)
                # Limit the desired velocity from alignment to max_speed
                desired_alignment_vel = (average_velocity / np.linalg.norm(average_velocity)) * self.max_speed if np.linalg.norm(average_velocity) > 1e-9 else np.zeros_like(pos)
                alignment_vector = self._limit_force(desired_alignment_vel - vel) # Calculate steering force and limit it

            # --- Calculate Corridor Following ---
            # (Existing logic seems reasonable, keep it, but calculate steering force)
            drone_vec_from_start = pos - self.start_point
            projection_scalar_ratio = np.dot(drone_vec_from_start, self.corridor_axis) / self.corridor_length_sq
            closest_center_pt_on_line = self.start_point + projection_scalar_ratio * self.corridor_axis
            perp_vec_to_drone = pos - closest_center_pt_on_line
            dist_to_centerline = np.linalg.norm(perp_vec_to_drone)

            # Apply force only if outside the desired corridor width
            if dist_to_centerline > self.corridor_width / 2.0:
                direction_to_center = -perp_vec_to_drone / dist_to_centerline if dist_to_centerline > 1e-9 else np.zeros_like(pos)
                # Desired velocity is purely to correct position, scaled by max_speed?
                desired_corridor_vel = direction_to_center * self.max_speed
                # The force should ideally only act perpendicularly to the current velocity?
                # For now, use standard steering force calculation.
                corridor_vector = self._limit_force(desired_corridor_vel - vel) # Calculate steering force and limit it


            # --- Calculate Base Anchoring ---
            # Project drone's position onto the corridor axis
            projected_dist_from_start = np.dot(drone_vec_from_start, self.corridor_direction)

            if 0 <= projected_dist_from_start < self.anchor_range:
                # Near Start: pull towards Start base
                vec_to_start = self.start_point - pos
                desired_anchor_vel = (vec_to_start / np.linalg.norm(vec_to_start)) * self.max_speed if np.linalg.norm(vec_to_start) > 1e-9 else np.zeros_like(pos)
                anchor_vector = self._limit_force(desired_anchor_vel - vel)
                # Apply START anchor weight here
                anchor_vector *= self.weight_start_anchor
            elif self.corridor_length - self.anchor_range < projected_dist_from_start <= self.corridor_length + 1e-9: # Allow slight overshoot
                 # Near End: pull towards End base
                vec_to_end = self.end_point - pos
                desired_anchor_vel = (vec_to_end / np.linalg.norm(vec_to_end)) * self.max_speed if np.linalg.norm(vec_to_end) > 1e-9 else np.zeros_like(pos)
                anchor_vector = self._limit_force(desired_anchor_vel - vel)
                # Apply END anchor weight here (using the general self.weight_anchor)
                anchor_vector *= self.weight_anchor
            else:
                 # If not near start or end, anchor_vector remains zero (or we apply zero weight)
                 # Multiplying by zero weight below handles this implicitly if vector wasn't zero
                 # To be explicit:
                 anchor_vector = np.zeros_like(pos)


            # --- Calculate Connectivity Rescue ---
            if num_neighbors < self.min_neighbors and nearest_neighbor_id != -1:
                 # Not enough neighbors, pull towards the nearest one found
                 nearest_neighbor_pos = positions[nearest_neighbor_id]
                 vec_to_nearest = nearest_neighbor_pos - pos
                 desired_rescue_vel = (vec_to_nearest / np.linalg.norm(vec_to_nearest)) * self.max_speed if np.linalg.norm(vec_to_nearest) > 1e-9 else np.zeros_like(pos)
                 connectivity_vector = self._limit_force(desired_rescue_vel - vel)

            # 7. Goal Steering (NEW)
            # Always apply a force gently pushing towards the End Point along the corridor axis
            desired_goal_vel = self.corridor_direction * self.max_speed
            goal_vector = self._limit_force(desired_goal_vel - vel)


            # --- Combine Steering Forces with Weights --- Apply weights to the limited forces
            # Note: Separation is calculated differently (as direct force), maybe limit it too?
            # Let's limit separation force similarly for consistency.
            limited_separation_vector = self._limit_force(separation_vector) # Limit the raw separation force

            total_force = (limited_separation_vector * self.weight_separation +
                           alignment_vector * self.weight_alignment +
                           cohesion_vector * self.weight_cohesion +
                           corridor_vector * self.weight_corridor +
                           anchor_vector + # Anchor vector is now pre-weighted
                           connectivity_vector * self.weight_connectivity +
                           goal_vector * self.weight_goal)

            # --- Apply Force to Calculate Acceleration and New Velocity ---
            # Acceleration = Force / Mass (Assume mass = 1 for simplicity)
            acceleration = total_force
            new_velocity = vel + acceleration * 1.0 # dt is applied in move(), assume dt=1 for velocity calc step?
                                                      # Or maybe the force calculation inherently includes dt? Let's stick to vel + accel for now.

            # --- Cap Final Speed ---
            speed = np.linalg.norm(new_velocity)
            if speed > self.max_speed:
                new_velocity = (new_velocity / speed) * self.max_speed
            elif speed < 1e-9:
                 # If no force caused movement, maybe maintain velocity? Or decay?
                 # For now, let it be zero if calculation results in zero.
                 new_velocity = np.zeros_like(pos)


            # Store the calculated *final* velocity as the action
            actions[drone_id] = new_velocity

            # Update drone's internal velocity state for next step's calculations
            current_drone.velocity = new_velocity # Assign the calculated velocity back


        # Assign zero velocity to inactive drones explicitly
        for drone_id, drone in self.drones.items():
            if drone.state != 'active':
                 # Ensure inactive drones have a zero velocity action
                 actions[drone_id] = np.zeros_like(drone.position)

        return actions

    # Helper function to compute steering force (limit magnitude)
    # Renamed from _steer_towards to _limit_force
    def _limit_force(self, force, max_force=0.4):
        # Limits the magnitude of a steering force vector
        # Reduced max_force default from 1.0 to 0.4 to dampen oscillations
        # max_force can be tuned, perhaps relate it to max_speed or acceleration limit
        norm = np.linalg.norm(force)
        if norm > max_force:
            force = (force / norm) * max_force
        return force

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

        # --- Check Start-to-End Connectivity ---
        start_to_end_connected = False
        if START_NODE_ID in self.connectivity_graph and END_NODE_ID in self.connectivity_graph:
            start_to_end_connected = nx.has_path(self.connectivity_graph, START_NODE_ID, END_NODE_ID)

        # --- Count End Node Drone Neighbors ---
        end_node_drone_neighbors = 0
        if END_NODE_ID in self.connectivity_graph:
            for neighbor_id in self.connectivity_graph.neighbors(END_NODE_ID):
                # Check if the neighbor is a drone (not another anchor or non-existent node)
                if neighbor_id in active_drone_nodes_in_graph: # Check against drones currently in graph
                    end_node_drone_neighbors += 1

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
            "percent_in_corridor": percent_in_corridor,
            "start_to_end_connected": start_to_end_connected, # Add the new metric
            "end_node_drone_neighbors": end_node_drone_neighbors # Add count of drones connected to End node
        }
        # print(f"Metrics: {metrics}") # Reduce print frequency
        return metrics

    def _check_termination(self):
        """Checks if the simulation should terminate."""
        # Example: Terminate if time limit reached or all drones inactive
        if self.time >= 100.0: # Arbitrary time limit for now
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
