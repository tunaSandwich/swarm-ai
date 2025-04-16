import time
from .environment import Environment
import matplotlib.pyplot as plt # Import for plot handling

# --- Simulation Parameters ---
SIMULATION_AREA_SIZE = [100, 100] # Width, Height (e.g., meters)
NUM_DRONES = 20
COMM_RANGE = 25.0 # Communication range in meters
INITIAL_ENERGY = 100.0
TIME_STEP = 0.1 # Simulation time step in seconds
MAX_SIM_TIME = 10.0 # Maximum simulation time
RENDER_EVERY_STEP = True # Control rendering frequency

# --- Main Simulation Logic ---
def run_simulation():
    """Initializes and runs the drone swarm simulation."""
    print("Starting Drone Swarm Simulation...")

    # 1. Initialize the environment
    env = Environment(size=SIMULATION_AREA_SIZE,
                      num_drones=NUM_DRONES,
                      comm_range=COMM_RANGE,
                      initial_energy=INITIAL_ENERGY)

    # Initial state rendering
    if RENDER_EVERY_STEP:
        env.render()

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
        if RENDER_EVERY_STEP:
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

    # 3. Post-Simulation Analysis / Visualization
    print(f"Collected {len(all_metrics)} metric snapshots.")
    # Example: Print the last metrics
    if all_metrics:
        print("Final Metrics:", all_metrics[-1])
    # TODO: Add plotting of metrics over time

    # Close the visualization properly
    if RENDER_EVERY_STEP:
        print("Closing visualization window...")
        env.close_visualization()

# --- Entry Point ---
if __name__ == "__main__":
    run_simulation() 
