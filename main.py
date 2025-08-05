import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class BoatParameters:
    """Physical parameters of the boat"""
    mass: float = 1000.0  # kg
    moment_inertia: float = 500.0  # kg*m^2 (about vertical axis)
    length: float = 10.0  # m (boat length for reference)
    C_D: float = 0.5  # Drag coefficient (placeholder)
    thrust: float = 2000.0  # N (constant thrust, placeholder)
    C_Y_beta: float = .1 # Sideforce coefficient (placeholder)


    # Water density
    rho_water: float = 1025.0  # kg/m^3

    # Reference area for drag calculation
    ref_area: float = 5.0  # m^2

    # skeg area
    S_skeg: float = 1.0 #m^2


class BoatState:
    """State vector for the boat"""

    def __init__(self):
        # Position in earth frame
        self.x = 0.0  # m (east)
        self.y = 0.0  # m (north)
        self.psi = 0.0  # rad (heading angle from north, positive clockwise)

        # Velocity in body frame
        self.u = 0.0  # m/s (forward velocity)
        self.v = 0.0  # m/s (sideways velocity)
        self.r = 0.0  # rad/s (yaw rate)

    @property
    def beta(self) -> float:
        """Sideslip angle in radians"""
        if abs(self.u) < 1e-6:
            return 0.0
        return np.arctan2(self.v, self.u)

    @property
    def V(self) -> float:
        """Total velocity magnitude"""
        return np.sqrt(self.u ** 2 + self.v ** 2)


class BoatSimulator:
    """3-DOF boat dynamics simulator"""

    def __init__(self, params: BoatParameters):
        self.params = params
        self.state = BoatState()
        self.time = 0.0

        # History for plotting
        self.history = {
            'time': [],
            'x': [],
            'y': [],
            'psi': [],
            'u': [],
            'v': [],
            'r': [],
            'beta': [],
            'V': []
        }

    def forces_and_moments(self) -> Tuple[float, float, float]:
        """
        Calculate forces and moments based on current velocity and sideslip angle
        Returns: (F_x, F_y, M_z) in body frame
        """
        u, v = self.state.u, self.state.v
        V = self.state.V
        beta = self.state.beta

        # Thrust force (along body x-axis)
        F_thrust = self.params.thrust

        # Drag force (opposite to velocity direction in body frame)
        if V > 1e-6:
            q = 0.5 * self.params.rho_water * V ** 2  # Dynamic pressure
            F_drag_total = self.params.C_D * self.params.ref_area * q

            # Drag components in body frame
            F_drag_x = -F_drag_total * (u / V)
            F_drag_y = -F_drag_total * (v / V)

            # Components due to rudder use in body frame
            F_skeg_y = self.params.C_Y_beta * self.params.S_skeg/self.params.ref_area * self.params.ref_area * q
        else:
            F_drag_x = 0.0
            F_drag_y = 0.0
            F_skeg_y = 0.0
            q = 0

        # Total forces in body frame
        F_x = F_thrust + F_drag_x
        F_y = F_drag_y + F_skeg_y

        # Moment about vertical axis (simplified model)
        # This is a placeholder - in reality would depend on hull shape, rudder, etc.
        # For now, assume some yaw damping and a moment proportional to sideslip
        C_N_beta = 0.1  # Yaw moment coefficient due to sideslip
        C_N_r = -0.05  # Yaw damping coefficient

        M_z = (C_N_beta * beta + C_N_r * self.state.r) * q * self.params.ref_area * self.params.length

        return F_x, F_y, M_z

    def derivatives(self) -> Tuple[float, ...]:
        """
        Calculate state derivatives
        Returns: (x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot)
        """
        u, v, r, psi = self.state.u, self.state.v, self.state.r, self.state.psi

        # Get forces and moments
        F_x, F_y, M_z = self.forces_and_moments()

        # Position derivatives (earth frame)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        x_dot = u * cos_psi - v * sin_psi
        y_dot = u * sin_psi + v * cos_psi
        psi_dot = r

        # Velocity derivatives (body frame)
        # Including centrifugal terms
        u_dot = F_x / self.params.mass + v * r
        v_dot = F_y / self.params.mass - u * r
        r_dot = M_z / self.params.moment_inertia

        return x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot

    def step(self, dt: float):
        """Advance simulation by one time step using forward Euler"""
        # Get derivatives
        derivs = self.derivatives()

        # Forward Euler integration
        self.state.x += derivs[0] * dt
        self.state.y += derivs[1] * dt
        self.state.psi += derivs[2] * dt
        self.state.u += derivs[3] * dt
        self.state.v += derivs[4] * dt
        self.state.r += derivs[5] * dt

        # Normalize heading angle
        self.state.psi = np.arctan2(np.sin(self.state.psi), np.cos(self.state.psi))

        # Update time
        self.time += dt

        # Store history
        self.history['time'].append(self.time)
        self.history['x'].append(self.state.x)
        self.history['y'].append(self.state.y)
        self.history['psi'].append(np.degrees(self.state.psi))
        self.history['u'].append(self.state.u)
        self.history['v'].append(self.state.v)
        self.history['r'].append(np.degrees(self.state.r))
        self.history['beta'].append(np.degrees(self.state.beta))
        self.history['V'].append(self.state.V)

    def simulate(self, duration: float, dt: float = 0.1):
        """Run simulation for specified duration"""
        steps = int(duration / dt)

        for _ in range(steps):
            self.step(dt)

    def plot_results(self):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Trajectory
        axes[0, 0].plot(self.history['x'], self.history['y'], 'b-', linewidth=2)
        axes[0, 0].plot(self.history['x'][0], self.history['y'][0], 'go', markersize=8, label='Start')
        axes[0, 0].plot(self.history['x'][-1], self.history['y'][-1], 'ro', markersize=8, label='End')
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Y Position (m)')
        axes[0, 0].set_title('Boat Trajectory')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        axes[0, 0].axis('equal')

        # Heading angle
        axes[0, 1].plot(self.history['time'], self.history['psi'], 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Heading Angle (deg)')
        axes[0, 1].set_title('Heading vs Time')
        axes[0, 1].grid(True)

        # Velocities
        axes[0, 2].plot(self.history['time'], self.history['u'], 'b-', linewidth=2, label='u (forward)')
        axes[0, 2].plot(self.history['time'], self.history['v'], 'r-', linewidth=2, label='v (sideways)')
        axes[0, 2].plot(self.history['time'], self.history['V'], 'k--', linewidth=2, label='V (total)')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Velocity (m/s)')
        axes[0, 2].set_title('Velocities vs Time')
        axes[0, 2].grid(True)
        axes[0, 2].legend()

        # Sideslip angle
        axes[1, 0].plot(self.history['time'], self.history['beta'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Sideslip Angle β (deg)')
        axes[1, 0].set_title('Sideslip Angle vs Time')
        axes[1, 0].grid(True)

        # Yaw rate
        axes[1, 1].plot(self.history['time'], self.history['r'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Yaw Rate (deg/s)')
        axes[1, 1].set_title('Yaw Rate vs Time')
        axes[1, 1].grid(True)

        # Phase portrait (u vs v)
        axes[1, 2].plot(self.history['u'], self.history['v'], 'c-', linewidth=2)
        axes[1, 2].plot(self.history['u'][0], self.history['v'][0], 'go', markersize=8, label='Start')
        axes[1, 2].plot(self.history['u'][-1], self.history['v'][-1], 'ro', markersize=8, label='End')
        axes[1, 2].set_xlabel('Forward Velocity u (m/s)')
        axes[1, 2].set_ylabel('Sideways Velocity v (m/s)')
        axes[1, 2].set_title('Velocity Phase Portrait')
        axes[1, 2].grid(True)
        axes[1, 2].legend()

        plt.tight_layout()
        plt.show()


# Example usage and test cases
if __name__ == "__main__":
    # Create boat with default parameters
    params = BoatParameters()
    sim = BoatSimulator(params)

    # Set initial conditions
    sim.state.u = 2.0  # Initial forward velocity
    sim.state.v = 0.5  # Initial sideways velocity
    sim.state.psi = np.radians(30)  # Initial heading (30 degrees from north)

    print("Starting 3-DOF Boat Simulation")
    print(f"Initial conditions:")
    print(f"  Position: ({sim.state.x:.1f}, {sim.state.y:.1f}) m")
    print(f"  Heading: {np.degrees(sim.state.psi):.1f} deg")
    print(f"  Velocity: u={sim.state.u:.2f}, v={sim.state.v:.2f} m/s")
    print(f"  Sideslip angle: {np.degrees(sim.state.beta):.2f} deg")

    # Run simulation
    sim.simulate(duration=60.0, dt=0.1)

    print(f"\nFinal conditions:")
    print(f"  Position: ({sim.state.x:.1f}, {sim.state.y:.1f}) m")
    print(f"  Heading: {np.degrees(sim.state.psi):.1f} deg")
    print(f"  Velocity: u={sim.state.u:.2f}, v={sim.state.v:.2f} m/s")
    print(f"  Sideslip angle: {np.degrees(sim.state.beta):.2f} deg")

    # Plot results
    sim.plot_results()