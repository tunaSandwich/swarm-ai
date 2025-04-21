import time
from .environment import Environment
import matplotlib.pyplot as plt # Import for plot handling
import pandas as pd # Import pandas for easier data handling

# --- Simulation Parameters ---
SIMULATION_AREA_SIZE = [100, 100] # Width, Height (e.g., meters)
NUM_DRONES = 20
COMM_RANGE = 25.0 # Communication range in meters
INITIAL_ENERGY = 100.0
# Energy Costs
MOVE_ENERGY_COST_PER_UNIT = 0.5 # Energy cost per unit distance moved
IDLE_ENERGY_COST_PER_SECOND = 0.1 # Energy cost per second while idle/active

# --- Corridor Parameters ---
START_POINT = [10.0, 50.0]
END_POINT = [90.0, 50.0]
CORRIDOR_WIDTH = 30.0 # Increased width

TIME_STEP = 0.1 # Simulation time step in seconds
MAX_SIM_TIME = 50.0 # Increased simulation time to see energy effects
RENDER_EVERY_STEP = True # Control rendering frequency

# --- Boid Parameters (passed to Environment) ---
# These can be tuned
MAX_SPEED = 5.0
SEPARATION_DISTANCE = 15.0 # Increased separation distance
ANCHOR_RANGE = 25.0 # Keep anchor range for now
MIN_NEIGHBORS = 2 # Min neighbors before rescue force applies

# Weights for steering behaviors - Adjusted based on simulation results
MAX_FORCE_LIMIT = 0.5 # New parameter to limit individual steering forces (passed to environment? No, used in _limit_force default)
# Tuned further based on results (clustering at end)
WEIGHT_SEPARATION = 1.9 # Keep separation strong
WEIGHT_ALIGNMENT = 0.6 # Slightly increase alignment
WEIGHT_COHESION = 0.1 # Reduce cohesion even further
WEIGHT_CORRIDOR = 1.6 # Increase corridor following again to keep drones inside
WEIGHT_ANCHOR = 0.2 # Reduce anchor force further
WEIGHT_START_ANCHOR = 0.6 # Separate, stronger anchor force for START node
WEIGHT_CONNECTIVITY = 2.0 # Keep connectivity rescue
WEIGHT_GOAL = 0.8 # Reduce goal steering slightly

# --- Main Simulation Logic ---
def run_simulation():
    """Initializes and runs the drone swarm simulation."""
    print("Starting Drone Swarm Simulation...")

    # 1. Initialize the environment, passing all parameters
    env = Environment(size=SIMULATION_AREA_SIZE,
                      num_drones=NUM_DRONES,
                      comm_range=COMM_RANGE,
                      initial_energy=INITIAL_ENERGY,
                      move_energy_cost=MOVE_ENERGY_COST_PER_UNIT,
                      idle_energy_cost=(IDLE_ENERGY_COST_PER_SECOND * TIME_STEP), # Pass cost per step
                      start_point=START_POINT,
                      end_point=END_POINT,
                      corridor_width=CORRIDOR_WIDTH,
                      # Pass Boid parameters
                      anchor_range=ANCHOR_RANGE,
                      min_neighbors=MIN_NEIGHBORS,
                      weight_separation=WEIGHT_SEPARATION,
                      weight_alignment=WEIGHT_ALIGNMENT,
                      weight_cohesion=WEIGHT_COHESION,
                      weight_corridor=WEIGHT_CORRIDOR,
                      weight_anchor=WEIGHT_ANCHOR,
                      weight_connectivity=WEIGHT_CONNECTIVITY,
                      max_speed=MAX_SPEED,
                      separation_distance=SEPARATION_DISTANCE,
                      weight_goal=WEIGHT_GOAL,
                      weight_start_anchor=WEIGHT_START_ANCHOR) # Pass start anchor weight

    # Initial state rendering
    if RENDER_EVERY_STEP:
        env.render()
        plt.pause(1) # Pause briefly to see initial state

    # List to store metrics from each step
    all_metrics = []

    # 2. Simulation Loop
    start_real_time = time.time()
    simulation_running = True
    step_count = 0
    while simulation_running:
        # Advance the simulation by one time step
        metrics, done = env.step(dt=TIME_STEP)
        all_metrics.append(metrics)

        # Render the current state
        if RENDER_EVERY_STEP and step_count % 5 == 0: # Render every 5 steps to speed up slightly
             env.render()

        # Check termination conditions
        if done or env.time >= MAX_SIM_TIME:
            simulation_running = False

        step_count += 1
        # Optional: Add a small delay to slow down simulation for observation
        # time.sleep(0.05)

    end_real_time = time.time()
    print(f"\n--- Simulation Finished ---")
    print(f"Ran {step_count} steps.")
    print(f"Total Simulation Time: {env.time:.2f} seconds")
    print(f"Total Real Time Elapsed: {end_real_time - start_real_time:.2f} seconds")

    # Close the dynamic visualization properly BEFORE creating new plots
    if RENDER_EVERY_STEP:
        print("Closing dynamic visualization window...")
        # Keep env.close_visualization() which calls plt.show() at the end
        # plt.close(env.fig) # Don't close the figure window yet

    # 3. Post-Simulation Analysis / Visualization
    print(f"Collected {len(all_metrics)} metric snapshots.")
    if not all_metrics:
        print("No metrics collected.")
        return

    # Convert metrics list to DataFrame for easier plotting
    metrics_df = pd.DataFrame(all_metrics)
    print("Final Metrics:", metrics_df.iloc[-1].to_dict() if not metrics_df.empty else "N/A")

    # Create plots for key metrics - Increase grid size for new plots
    fig_metrics, axs = plt.subplots(5, 1, figsize=(10, 18), sharex=True) # Increased rows from 3 to 5

    # Plot Average Energy
    axs[0].plot(metrics_df['time'], metrics_df['average_energy'], label='Average Energy')
    axs[0].set_ylabel('Energy')
    axs[0].set_title('Average Drone Energy Over Time')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Connectivity Metric (e.g., Largest Component Size)
    axs[1].plot(metrics_df['time'], metrics_df['largest_component_size'], label='Largest Component Size')
    axs[1].set_ylabel('Number of Drones')
    axs[1].set_title('Size of Largest Connected Component Over Time')
    axs[1].grid(True)
    axs[1].legend()

    # Plot Number of Active Drones
    axs[2].plot(metrics_df['time'], metrics_df['num_active_drones'], label='Active Drones')
    axs[2].set_xlabel('Simulation Time (s)')
    axs[2].set_ylabel('Number of Drones')
    axs[2].set_title('Number of Active Drones Over Time')
    axs[2].grid(True)
    axs[2].legend()

    # Plot Average Distance from Centerline (Placeholder data for now)
    if 'avg_dist_from_center' in metrics_df.columns:
        axs[3].plot(metrics_df['time'], metrics_df['avg_dist_from_center'], label='Avg. Dist. from Center')
        axs[3].set_ylabel('Distance (m)')
        axs[3].set_title('Average Drone Distance from Corridor Centerline')
        axs[3].axhline(y=CORRIDOR_WIDTH / 2, color='r', linestyle='--', label='Corridor Edge')
        axs[3].grid(True)
        axs[3].legend()
    else:
        axs[3].set_title('Average Drone Distance from Corridor Centerline (Metric N/A)')
        axs[3].grid(True)

    # Plot Percentage of Drones in Corridor (Placeholder data for now)
    if 'percent_in_corridor' in metrics_df.columns:
        axs[4].plot(metrics_df['time'], metrics_df['percent_in_corridor'], label='% In Corridor')
        axs[4].set_xlabel('Simulation Time (s)') # Add xlabel to the last plot
        axs[4].set_ylabel('Percentage (%)')
        axs[4].set_ylim(0, 105) # Set y-limit 0-100%
        axs[4].set_title('Percentage of Active Drones within Corridor Width')
        axs[4].grid(True)
        axs[4].legend()
    else:
        axs[4].set_title('Percentage of Active Drones within Corridor Width (Metric N/A)')
        axs[4].set_xlabel('Simulation Time (s)') # Add xlabel to the last plot
        axs[4].grid(True)

    plt.tight_layout()
    # Instead of showing here, let env.close_visualization handle the final plt.show()
    # plt.show()

    # Ensure the main simulation plot stays open until closed by user
    if RENDER_EVERY_STEP:
        print("Displaying final simulation state and metrics plots...")
        env.close_visualization() # This now calls plt.show() for all figures
    else:
         # If not rendering steps, show metrics plot now
         plt.show()

# --- Entry Point ---
if __name__ == "__main__":
    run_simulation() 
