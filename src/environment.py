import numpy as np
import networkx as nx
from .drone import Drone  # Assuming drone.py is in the same directory
import matplotlib.pyplot as plt

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
            self.connectivity_graph.add_node(i, pos=drone.position) # Add node to graph

        print(f"Environment initialized with {self.num_drones} drones.")

    def _update_connectivity(self):
        """Updates the connectivity graph based on current drone positions."""
        # Keep track of nodes that should exist (active drones)
        active_drone_ids = {id for id, drone in self.drones.items() if drone.state == "active"}

        # Remove nodes from graph if drone is no longer active
        nodes_to_remove = [node for node in self.connectivity_graph.nodes() if node not in active_drone_ids]
        self.connectivity_graph.remove_nodes_from(nodes_to_remove)

        # Add nodes if any drone became active again (less likely but possible)
        for drone_id in active_drone_ids:
            if drone_id not in self.connectivity_graph:
                self.connectivity_graph.add_node(drone_id, pos=self.drones[drone_id].position)

        # Update positions and edges for active drones
        self.connectivity_graph.clear_edges()
        active_nodes = list(self.connectivity_graph.nodes())
        for i in range(len(active_nodes)):
            d1_id = active_nodes[i]
            d1 = self.drones[d1_id]
            # Update position attribute for visualization
            self.connectivity_graph.nodes[d1_id]['pos'] = d1.position

            for j in range(i + 1, len(active_nodes)):
                d2_id = active_nodes[j]
                d2 = self.drones[d2_id]
                distance = np.linalg.norm(d1.position - d2.position)
                if distance <= self.comm_range:
                    self.connectivity_graph.add_edge(d1_id, d2_id, weight=distance)

        # Update node positions in the graph for visualization
        # for drone_id, drone in self.drones.items():
        #     if drone_id in self.connectivity_graph:
        #          self.connectivity_graph.nodes[drone_id]['pos'] = drone.position

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
        """Placeholder for getting actions from the AI/control system."""
        # For now, let's make them move randomly slightly
        actions = {}
        target_speed = 5.0 # Define a target speed for movement
        separation_distance = 5.0 # Minimum desired distance between drones
        separation_strength = 1.5 # How strongly drones push away from each other

        # Get positions of all active drones for efficient neighbor checking
        active_drones_pos = {id: d.position for id, d in self.drones.items() if d.state == 'active'}

        for drone_id, drone in self.drones.items():
            if drone.state == "active":
                # --- Corridor Following Logic ---
                drone_vec = drone.position - self.start_point
                
                # Project drone position onto the corridor axis
                projection_scalar = np.dot(drone_vec, self.corridor_axis) / self.corridor_length_sq
                projection_scalar = np.clip(projection_scalar, 0, 1) # Clamp between start and end projection
                
                # Find the closest point on the centerline to the drone
                closest_center_pt = self.start_point + projection_scalar * self.corridor_axis
                
                # Vector from centerline to drone (perpendicular component)
                perp_vec = drone.position - closest_center_pt
                dist_to_center = np.linalg.norm(perp_vec)

                # --- Calculate Desired Velocity --- 
                # 1. Force towards End Point (parallel to corridor axis)
                velocity_parallel = self.corridor_direction * target_speed

                # 2. Force towards Centerline (if outside half-width)
                velocity_perpendicular = np.zeros_like(self.corridor_direction)
                if dist_to_center > self.corridor_width / 2.0:
                    # Direction towards center is -perp_vec
                    # Scale force by how far outside it is
                    correction_strength = 1.0 # Adjust strength as needed
                    velocity_perpendicular = (-perp_vec / dist_to_center) * target_speed * correction_strength 
                    # print(f"Drone {drone_id} correcting position. Dist: {dist_to_center:.2f}") # Debug
                
                # --- Separation Force --- 
                velocity_separation = np.zeros_like(self.corridor_direction)
                num_neighbors_too_close = 0
                for neighbor_id, neighbor_pos in active_drones_pos.items():
                    if drone_id == neighbor_id:
                        continue # Don't compare drone to itself
                    
                    vec_to_neighbor = neighbor_pos - drone.position
                    dist_sq = np.dot(vec_to_neighbor, vec_to_neighbor) # Use squared distance for efficiency

                    if dist_sq < separation_distance**2 and dist_sq > 1e-6: # If neighbor is too close
                        dist = np.sqrt(dist_sq)
                        # Calculate force direction (away from neighbor)
                        repulsion_direction = -vec_to_neighbor / dist
                        # Scale force inversely with distance (stronger when closer)
                        repulsion_force = repulsion_direction * (1.0 - dist / separation_distance) 
                        velocity_separation += repulsion_force
                        num_neighbors_too_close += 1
                
                # Average the separation force if multiple neighbors are close
                # if num_neighbors_too_close > 0:
                #     velocity_separation /= num_neighbors_too_close # Optional: Average vs Sum

                # Scale separation force
                velocity_separation *= target_speed * separation_strength

                # 3. Combine forces (simple addition for now)
                # Maybe add a small random component for exploration/jitter?
                # random_jitter = (np.random.rand(len(self.size)) - 0.5) * 0.5
                # Combine corridor guidance and separation
                desired_velocity = velocity_parallel + velocity_perpendicular + velocity_separation # + random_jitter
                
                # Normalize final velocity to target speed (optional, prevents excessive speed)
                current_speed = np.linalg.norm(desired_velocity)
                if current_speed > target_speed:
                   desired_velocity = (desired_velocity / current_speed) * target_speed
                elif current_speed < 1e-6: # Avoid division by zero if velocity is near zero
                     # If no other forces, add slight push towards end
                     desired_velocity = self.corridor_direction * target_speed * 0.1 

                actions[drone_id] = desired_velocity
                # print(f"Drone {drone_id}: Vel Par={velocity_parallel}, Vel Perp={velocity_perpendicular}, Final={desired_velocity}") # Debug
            else:
                actions[drone_id] = np.zeros_like(self.start_point) # Inactive drones don't move

        return actions

    def _calculate_metrics(self):
        """Calculates and returns relevant simulation metrics."""
        num_edges = self.connectivity_graph.number_of_edges()
        active_nodes = list(self.connectivity_graph.nodes())
        num_active_nodes = len(active_nodes)

        # Calculate average energy of active drones
        total_energy = 0
        total_dist_from_center = 0
        num_in_corridor = 0

        if num_active_nodes > 0:
            for id in active_nodes:
                drone = self.drones[id]
                total_energy += drone.energy

                # Calculate distance from centerline
                drone_vec = drone.position - self.start_point
                projection_scalar = np.dot(drone_vec, self.corridor_axis) / self.corridor_length_sq
                # We care about distance even if slightly outside the projected segment [0,1]
                # projection_scalar = np.clip(projection_scalar, 0, 1)
                closest_center_pt = self.start_point + projection_scalar * self.corridor_axis
                dist_to_center = np.linalg.norm(drone.position - closest_center_pt)
                
                total_dist_from_center += dist_to_center
                if dist_to_center <= self.corridor_width / 2.0:
                    num_in_corridor += 1

            avg_energy = total_energy / num_active_nodes
            avg_dist_from_center = total_dist_from_center / num_active_nodes
            percent_in_corridor = (num_in_corridor / num_active_nodes) * 100.0
        else:
            avg_energy = 0
            avg_dist_from_center = 0
            percent_in_corridor = 0

        avg_degree = (2 * num_edges / num_active_nodes) if num_active_nodes > 0 else 0
        is_connected = False
        largest_cc_size = 0
        if num_active_nodes > 0:
            is_connected = nx.is_connected(self.connectivity_graph)
            if not self.connectivity_graph.nodes:
                 largest_cc_size = 0
            else:
                 # Use try-except block as nx.connected_components might raise error on empty graph, though unlikely here
                 try:
                     largest_cc = max(nx.connected_components(self.connectivity_graph), key=len, default=set())
                     largest_cc_size = len(largest_cc)
                 except ValueError:
                     largest_cc_size = 0 # Handle potential empty graph case

        metrics = {
            "time": self.time,
            "num_active_drones": num_active_nodes,
            "num_edges": num_edges,
            "average_degree": avg_degree,
            "is_connected": is_connected,
            "largest_component_size": largest_cc_size,
            "average_energy": avg_energy, # Add average energy to metrics
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

        # --- Draw Corridor Boundaries --- 
        # Calculate perpendicular vector to the corridor axis
        perp_direction = np.array([-self.corridor_direction[1], self.corridor_direction[0]])
        half_width_vec = perp_direction * (self.corridor_width / 2.0)
        
        # Points for the two boundary lines
        line1_start = self.start_point + half_width_vec
        line1_end = self.end_point + half_width_vec
        line2_start = self.start_point - half_width_vec
        line2_end = self.end_point - half_width_vec

        # Draw the lines
        self.ax.plot([line1_start[0], line1_end[0]], [line1_start[1], line1_end[1]], 'g--', alpha=0.5, label='Corridor Boundary')
        self.ax.plot([line2_start[0], line2_end[0]], [line2_start[1], line2_end[1]], 'g--', alpha=0.5)
        # --- End Corridor Drawing ---

        # --- Draw Start/End Points --- 
        self.ax.plot(self.start_point[0], self.start_point[1], 'go', markersize=10, label='Start')
        self.ax.plot(self.end_point[0], self.end_point[1], 'ro', markersize=10, label='End')
        # --- End Start/End Drawing ---

        # Get positions for drawing
        pos = nx.get_node_attributes(self.connectivity_graph, 'pos')
        if not pos: # Exit if no nodes to draw
            # Need to handle the plot closing gracefully if it's empty early on
            # self.ax.set_title(f"Drone Swarm Simulation - Time: {self.time:.2f}s - No Active Drones")
            # plt.draw()
            # plt.pause(0.01)
            return

        # Determine node colors based on state (optional)
        node_colors = []
        active_nodes_in_graph = list(self.connectivity_graph.nodes())
        for node_id in active_nodes_in_graph:
             # Check if node_id exists in self.drones before accessing state
            if node_id in self.drones:
                drone_state = self.drones[node_id].state
                if drone_state == "active":
                    node_colors.append('blue')
                elif drone_state == "low_power": # Example state
                    node_colors.append('orange')
                else: # inactive
                    node_colors.append('grey')
            else:
                 # Should not happen if _update_connectivity is correct, but handle defensively
                 node_colors.append('red') # Indicate an issue


        # Draw the network
        nx.draw_networkx_edges(self.connectivity_graph, pos, ax=self.ax, alpha=0.4, edge_color='gray')
        # Ensure we only try to draw nodes that actually exist in the graph
        nx.draw_networkx_nodes(self.connectivity_graph, pos, ax=self.ax, nodelist=active_nodes_in_graph, node_size=50, node_color=node_colors)
        # nx.draw_networkx_labels(self.connectivity_graph, pos, ax=self.ax, font_size=8) # Optional: labels

        # Set plot limits and aspect ratio
        self.ax.set_xlim(0, self.size[0])
        self.ax.set_ylim(0, self.size[1])
        if len(self.size) == 3: # Basic handling for 3D Z-limit if needed
             self.ax.set_zlim(0, self.size[2])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title(f"Drone Swarm Simulation - Time: {self.time:.2f}s - Active: {len(active_nodes_in_graph)}")
        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")
        self.ax.legend(loc='upper right') # Add legend for start/end/boundary

        # Redraw the canvas
        plt.draw()
        plt.pause(0.01) # Small pause to allow plot to update

    def close_visualization(self):
        """Closes the visualization window."""
        plt.ioff() # Turn off interactive mode
        plt.show() # Keep the final plot window open until manually closed 
