import numpy as np

class Drone:
    """Represents a single drone in the simulation."""
    def __init__(self, drone_id, initial_position, initial_energy, comm_range):
        self.id = drone_id
        self.position = np.array(initial_position, dtype=float)  # [x, y] or [x, y, z]
        self.energy = float(initial_energy)
        self.comm_range = float(comm_range)
        self.neighbors = []  # List of drone IDs within communication range
        self.state = "active" # Potential states: active, inactive, low_power

        print(f"Drone {self.id} initialized at {self.position} with {self.energy} energy.")

    def move(self, velocity_vector, dt):
        """Updates the drone's position based on velocity and time step."""
        # Simple linear movement for now
        displacement = np.array(velocity_vector) * dt
        self.position += displacement
        # TODO: Add energy consumption for movement
        # TODO: Add boundary checks (stay within environment limits)
        print(f"Drone {self.id} moved to {self.position}")

    def update_state(self):
        """Updates the drone's internal state (e.g., based on energy)."""
        # Placeholder for state logic (e.g., low power mode)
        if self.energy <= 0:
            self.state = "inactive"
        pass

    def __repr__(self):
        return f"Drone(ID={self.id}, Pos={self.position}, Energy={self.energy:.2f}, State={self.state})" 
