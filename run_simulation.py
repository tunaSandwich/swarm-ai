import time
from src.environment import Environment, START_NODE_ID, END_NODE_ID # Assuming environment.py is in src
import numpy as np

def main():
    # --- Simulation Parameters ---
    # Environment dimensions (e.g., 2D)
    env_size = [100, 100]
    num_drones = 10
    comm_range = 25.0
    initial_energy = 1000.0
    move_energy_cost = 0.1 # Energy cost per unit distance per unit time
    idle_energy_cost = 0.01 # Energy cost per unit time when idle

    # Corridor definition
    start_point = [10, 50]
    end_point = [90, 50]
    corridor_width = 20.0 # Total width of the corridor

    # Boids parameters (using defaults from Environment class where possible, or sensible values)
    anchor_range = 20.0
    min_neighbors = 2
    weight_separation = 1.8 # Slightly higher to prevent clumping
    weight_alignment = 1.0
    weight_cohesion = 1.0
    weight_corridor = 2.5 # Higher weight to stay in corridor
    weight_anchor = 1.5
    weight_connectivity = 2.0
    max_speed = 3.0 # Drone's maximum speed
    separation_distance = 8.0 # Desired minimum distance between drones
    weight_goal = 1.2 # Slightly stronger pull towards goal
    weight_start_anchor = 0.8 # Specific weight for start anchor

    # Time step for simulation
    dt = 0.1
    max_simulation_time = 200 # Maximum time to run the simulation in seconds

    print("Initializing Drone Swarm Environment...")
    env = Environment(
        size=env_size,
        num_drones=num_drones,
        comm_range=comm_range,
        initial_energy=initial_energy,
        move_energy_cost=move_energy_cost,
        idle_energy_cost=idle_energy_cost,
        start_point=start_point,
        end_point=end_point,
        corridor_width=corridor_width,
        anchor_range=anchor_range,
        min_neighbors=min_neighbors,
        weight_separation=weight_separation,
        weight_alignment=weight_alignment,
        weight_cohesion=weight_cohesion,
        weight_corridor=weight_corridor,
        weight_anchor=weight_anchor,
        weight_connectivity=weight_connectivity,
        max_speed=max_speed,
        separation_distance=separation_distance,
        weight_goal=weight_goal,
        weight_start_anchor=weight_start_anchor
    )
    print("Environment Initialized.")

    print("Starting Simulation Loop...")
    done = False
    try:
        while not done and env.time < max_simulation_time:
            metrics, done_step = env.step(dt)
            env.render() # Visualize the current state

            # Print some metrics periodically
            if int(env.time * 10) % 20 == 0: # Print every 2 seconds
                print(f"Time: {metrics['time']:.2f}s, Active Drones: {metrics['num_active_drones']}, "
                      f"Swarm Connected: {metrics['is_swarm_connected']}, Start-End Path: {metrics['start_to_end_connected']}, "
                      f"End Neighbors: {metrics['end_node_drone_neighbors']}, Corridor Established: {env.corridor_established}")

            if done_step:
                print(f"Simulation termination condition met at time {env.time:.2f}s.")
                done = True

            # Optional: Control simulation speed
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("Simulation ended.")
        print(f"Final Metrics: {env._calculate_metrics()}")
        env.close_visualization()
        print("Visualization closed. Press Enter to exit.")
        input() # Keep plot open until user presses Enter

if __name__ == "__main__":
    main() 
