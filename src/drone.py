import numpy as np

class Drone:
    """Represents a single drone in the simulation."""
    def __init__(self, drone_id, initial_position, initial_energy, comm_range, move_energy_cost, idle_energy_cost):
        self.id = drone_id
        self.position = np.array(initial_position, dtype=float)  # [x, y] or [x, y, z]
        self.energy = float(initial_energy)
        self.comm_range = float(comm_range)
        self.move_energy_cost = float(move_energy_cost) # Energy cost per unit distance moved
        self.idle_energy_cost = float(idle_energy_cost) # Energy cost per simulation step (time delta)
        self.neighbors = []  # List of drone IDs within communication range
        self.state = "active" # Potential states: active, inactive, low_power

        # Optional: Remove or comment out print for cleaner output later
        # print(f"Drone {self.id} initialized at {self.position} with {self.energy} energy.")

    def move(self, velocity_vector, dt, environment_size):
        """Updates the drone's position based on velocity, time step, and boundaries. Consumes energy for movement."""
        if self.state != "active" or self.energy <= 0:
            return # Cannot move if inactive or out of energy

        displacement = np.array(velocity_vector) * dt
        distance_moved = np.linalg.norm(displacement)

        # Calculate energy cost for movement
        energy_cost = distance_moved * self.move_energy_cost
        self.energy -= energy_cost

        # If enough energy, update position
        if self.energy > 0:
            new_position = self.position + displacement
            # Boundary checks (bounce off)
            for i in range(len(environment_size)):
                if new_position[i] < 0:
                    new_position[i] = 0 # Stop at boundary
                    # Optionally reflect velocity: velocity_vector[i] *= -1
                elif new_position[i] > environment_size[i]:
                    new_position[i] = environment_size[i] # Stop at boundary
                    # Optionally reflect velocity: velocity_vector[i] *= -1

            self.position = new_position
            # print(f"Drone {self.id} moved to {self.position}, Energy left: {self.energy:.2f}") # Optional debug print
        else:
            # Not enough energy to complete the move fully (or partially)
            # Option 1: Don't move at all if cost > remaining energy
            # Option 2: Move partially? For simplicity, let's prevent move if it drains energy below zero.
            self.energy += energy_cost # Revert energy deduction if move is cancelled
            self.energy = max(0, self.energy) # Clamp energy at 0 just in case
            # print(f"Drone {self.id} insufficient energy to move. Energy: {self.energy:.2f}") # Optional debug print


    def update_state(self, dt):
        """Updates the drone's internal state and consumes idle energy."""
        if self.state != "active":
            return

        # Consume idle energy (cost per second * time delta)
        self.energy -= self.idle_energy_cost * dt
        self.energy = max(0, self.energy) # Ensure energy doesn't go below zero

        # Update state based on energy
        if self.energy <= 0:
            self.state = "inactive"
            # print(f"Drone {self.id} became inactive due to low energy.") # Optional debug print
        # else: # Optional: Add other states like 'low_power'
            # if self.energy < SOME_LOW_POWER_THRESHOLD:
            #     self.state = "low_power"

    def __repr__(self):
        return f"Drone(ID={self.id}, Pos={self.position}, Energy={self.energy:.2f}, State={self.state})" 
