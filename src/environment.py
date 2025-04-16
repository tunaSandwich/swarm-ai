import numpy as np
import networkx as nx
from .drone import Drone  # Assuming drone.py is in the same directory
import matplotlib.pyplot as plt

class Environment:
    """Manages the simulation space, drones, and connectivity."""
    def __init__(self, size, num_drones, comm_range, initial_energy):
        self.size = np.array(size)  # e.g., [width, height] or [width, height, depth]
        self.num_drones = num_drones
        self.comm_range = comm_range
        self.initial_energy = initial_energy
        self.drones = {}  # Dictionary to store drone objects {drone_id: Drone_instance}
        self.time = 0.0
        self.connectivity_graph = nx.Graph() # To represent drone links

        # --- Visualization Setup ---
        self.fig, self.ax = plt.subplots()
        plt.ion() # Turn on interactive mode
        # --- End Visualization Setup ---

        self._initialize_drones()
        self._update_connectivity()

    def _initialize_drones(self):
        """Creates and places drones randomly within the environment bounds."""
        for i in range(self.num_drones):
            # Random initial position within bounds
            initial_pos = np.random.rand(len(self.size)) * self.size
            drone = Drone(drone_id=i,
                          initial_position=initial_pos,
                          initial_energy=self.initial_energy,
                          comm_range=self.comm_range)
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

        # 2. Apply actions (move drones)
        for drone_id, action in actions.items():
            if drone_id in self.drones and self.drones[drone_id].state == "active":
                velocity = action # Assuming action is directly a velocity vector for now
                self.drones[drone_id].move(velocity, dt)
                # TODO: Apply energy consumption
                # self.drones[drone_id].energy -= calculate_energy_cost(velocity, dt)

        # 3. Update drone states (e.g., check energy levels)
        for drone in self.drones.values():
            drone.update_state()

        # 4. Update connectivity based on new positions
        self._update_connectivity()

        # 5. Calculate metrics (placeholder)
        metrics = self._calculate_metrics()

        # 6. Check termination conditions (placeholder)
        done = self._check_termination()

        return metrics, done

    def _get_actions(self):
        """Placeholder for getting actions from the AI/control system."""
        # For now, let's make them move randomly slightly
        actions = {}
        for drone_id, drone in self.drones.items():
            if drone.state == "active":
                # Random velocity vector (scaled down)
                # Make movement less erratic for visualization
                random_velocity = (np.random.rand(len(self.size)) - 0.5) * 0.1 # Reduced magnitude
                actions[drone_id] = random_velocity
        return actions

    def _calculate_metrics(self):
        """Calculates and returns relevant simulation metrics."""
        num_edges = self.connectivity_graph.number_of_edges()
        active_nodes = list(self.connectivity_graph.nodes())
        num_active_nodes = len(active_nodes)

        avg_degree = (2 * num_edges / num_active_nodes) if num_active_nodes > 0 else 0
        is_connected = False
        largest_cc_size = 0
        if num_active_nodes > 0:
            # Ensure graph is not empty before checking connectivity
            is_connected = nx.is_connected(self.connectivity_graph)
            if self.connectivity_graph:
                 largest_cc = max(nx.connected_components(self.connectivity_graph), key=len, default=set())
                 largest_cc_size = len(largest_cc)
            else:
                largest_cc_size = 0


        metrics = {
            "time": self.time,
            "num_active_drones": num_active_nodes,
            "num_edges": num_edges,
            "average_degree": avg_degree,
            "is_connected": is_connected,
            "largest_component_size": largest_cc_size,
            # TODO: Add average energy metric
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

        # Get positions for drawing
        pos = nx.get_node_attributes(self.connectivity_graph, 'pos')
        if not pos: # Exit if no nodes to draw
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
        nx.draw_networkx_nodes(self.connectivity_graph, pos, ax=self.ax, node_size=50, node_color=node_colors)
        # nx.draw_networkx_labels(self.connectivity_graph, pos, ax=self.ax, font_size=8) # Optional: labels

        # Set plot limits and aspect ratio
        self.ax.set_xlim(0, self.size[0])
        self.ax.set_ylim(0, self.size[1])
        if len(self.size) == 3: # Basic handling for 3D Z-limit if needed
             self.ax.set_zlim(0, self.size[2])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title(f"Drone Swarm Simulation - Time: {self.time:.2f}s")
        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")

        # Redraw the canvas
        plt.draw()
        plt.pause(0.01) # Small pause to allow plot to update

    def close_visualization(self):
        """Closes the visualization window."""
        plt.ioff() # Turn off interactive mode
        plt.show() # Keep the final plot window open until manually closed 
