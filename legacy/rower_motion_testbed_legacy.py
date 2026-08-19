import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class RowingSimulatorParams:
    """Mock parameters class to match your simulator structure"""

    def __init__(self):
        # Stroke timing parameters
        self.active_phase_duration = 0.8  # seconds

        # Rower parameters
        self.n_rowers = 8
        self.n_body_parts = 3
        self.rower_spacing = 2.0  # meters

        # Force parameters
        self.F_max_x = 500.0  # N
        self.F_max_z = 100.0  # N
        self.k_yaw = 50.0
        self.k_roll = 100.0


class MockRowingSimulator:
    def __init__(self):
        self.params = RowingSimulatorParams()
        self.stroke_period = 2.0  # seconds (30 strokes per minute)
        self.recovery_duration = self.stroke_period - self.params.active_phase_duration
        self.yaw_control_offset = 0.0

    def get_body_part_kinematics(self, t: float, rower_idx: int, part_idx: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Your body part kinematics function
        """
        # for now lets space rowers 2 meters apart
        rower_spacing = 2  # m distance between rowers

        # Different motion for different body parts
        amplitude = 0.1 * (part_idx + 1) / self.params.n_body_parts
        amplitude = .5  # todo: reintorduce some amount of amiplitude into the rower motion
        vert_amplitude = 0

        # Simple oscillatory motion as placeholder
        omega = 2 * np.pi / self.stroke_period

        # Determine phase within stroke cycle
        t_mod = t % self.stroke_period

        if t_mod <= self.params.active_phase_duration:
            # Active phase
            t_mod = (t % self.stroke_period) % self.params.active_phase_duration
            omega = np.pi / self.params.active_phase_duration
            offset = 0  # angular ofset for the begining fo this strok
        else:
            # recovery phase
            t_mod = ((t % self.stroke_period)-self.params.active_phase_duration) % self.recovery_duration
            omega = np.pi / self.recovery_duration  # ToDO: smooth the transition from drive to recovery by making omega and offset a function of t%stroke period or by avergin the 2 until smooth
            offset = np.pi

        # matched with stroke nd recovery
        if (rower_idx == 0) and (part_idx == 0):
            print(f"phase_ratio_body: {(omega * t_mod + offset) / np.pi}")

        x_rel = amplitude * np.cos(omega * t_mod + offset) #+ (self.params.n_rowers / 2 - rower_idx) * self.params.rower_spacing
        v_rel = -amplitude * omega * np.sin(omega * t_mod + offset)
        a_rel = -amplitude * omega ** 2 * np.cos(omega * t_mod + offset)


        clip_width = 4/12 * part_idx+1
        clip_offset = -clip_width/2

        x_rel = np.cos(omega * t_mod + offset)  # + (self.params.n_rowers / 2 - rower_idx) * self.params.rower_spacing
        x_rel = amplitude * np.tanh(clip_width*x_rel + clip_offset)
        # Velocity: d/dt[amplitude * tanh(clip_width * cos(omega*t + offset) + clip_offset)]
        # Using chain rule: amplitude * sech²(clip_width * cos_term + clip_offset) * clip_width * (-sin(omega*t + offset)) * omega
        cos_term = np.cos(omega * t_mod + offset)
        sin_term = np.sin(omega * t_mod + offset)
        sech_squared = 1.0 / (np.cosh(clip_width * cos_term + clip_offset) ** 2)  # sech²(x) = 1/cosh²(x)
        v_rel = -amplitude * clip_width * omega * sin_term * sech_squared

        # Acceleration: d/dt[v_rel]
        # This is more complex - need to differentiate the product
        # d/dt[-amplitude * clip_width * omega * sin(omega*t + offset) * sech²(clip_width * cos(omega*t + offset) + clip_offset)]

        # First term: derivative of sin term
        term1 = -amplitude * clip_width * omega ** 2 * cos_term * sech_squared

        # Second term: derivative of sech² term (using chain rule)
        # d/dt[sech²(u)] = -2*sech²(u)*tanh(u)*du/dt where u = clip_width * cos_term + clip_offset
        tanh_term = np.tanh(clip_width * cos_term + clip_offset)
        dsech_dt = -2 * sech_squared * tanh_term * clip_width * (-sin_term) * omega
        term2 = -amplitude * clip_width * omega * sin_term * dsech_dt

        a_rel = term1 + term2

        # x_rel = np.cos(omega * t_mod + offset)
        # v_rel = -amplitude * omega * np.sin(omega * t_mod + offset)
        # a_rel = -amplitude * omega ** 2 * np.cos(omega * t_mod + offset)

        # x_rel = amplitude * omega * t_mod ** 2 #+ (self.params.n_rowers / 2 - rower_idx) * self.params.rower_spacing
        # v_rel = 2 * amplitude * omega * t_mod * omega
        # a_rel = 2 * amplitude * omega * omega

        # TODO: match this up with drive and recovery timing

        position = np.array([x_rel, 0, vert_amplitude * 0.05 * np.sin(2 * omega * t)])
        velocity = np.array([v_rel, 0, vert_amplitude * 0.1 * omega * np.cos(2 * omega * t)])
        acceleration = np.array([a_rel, 0, vert_amplitude * -0.2 * omega ** 2 * np.sin(2 * omega * t)])

        return position, velocity, acceleration

    def get_oarlock_forces(self, t: float, psi: float, phi: float,
                           rower_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Your oarlock forces function
        """
        # Determine phase within stroke cycle
        t_mod = t % self.stroke_period

        if t_mod <= self.params.active_phase_duration:
            # Active phase
            phase_ratio = t_mod / self.params.active_phase_duration
            force_multiplier = np.sin(np.pi * phase_ratio)

            print(f"phase_ratio_oar: {phase_ratio}")

            # Base forces
            f_x = self.params.F_max_x * force_multiplier
            f_z = self.params.F_max_z * force_multiplier

            # Yaw control (only during active phase)
            # Evaluate yaw angle at beginning of stroke for control decision
            if t_mod < 0.01:  # Beginning of stroke
                self.yaw_control_offset = -self.params.k_yaw * psi
            else:
                self.yaw_control_offset = getattr(self, 'yaw_control_offset', 0)

            f_x_controlled = f_x + self.yaw_control_offset

        else:
            # Recovery phase
            f_x_controlled = 0.0
            f_z = 0.0

        # Roll control (active throughout stroke cycle)
        roll_control = -self.params.k_roll * phi
        f_z_controlled = f_z + roll_control

        # Forces in hull coordinate system
        left_force = np.array([f_x_controlled, 0, f_z_controlled])
        right_force = np.array([f_x_controlled, 0, -f_z_controlled])  # Opposite z-component

        return left_force, right_force


def plot_rowing_dynamics():
    """Plot body kinematics and oarlock forces over time"""

    # Create simulator instance
    sim = MockRowingSimulator()

    # Time array for 3 complete stroke cycles
    n_cycles = 3
    t_end = n_cycles * sim.stroke_period
    dt = 0.01
    time = np.arange(0, t_end, dt)

    # Arrays to store results
    positions = []
    velocities = []
    accelerations = []
    left_forces = []
    right_forces = []
    stroke_phases = []

    # Parameters for function calls
    rower_idx = 0
    part_idx = 12
    psi = 0.0  # yaw angle
    phi = 0.0  # roll angle

    # Suppress print statements for cleaner output
    import sys
    from contextlib import redirect_stdout
    import io

    f = io.StringIO()
    with redirect_stdout(f):
        # Calculate kinematics and forces for each time step
        for t in time:
            # Get body kinematics
            pos, vel, acc = sim.get_body_part_kinematics(t, rower_idx, part_idx)
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)

            # Get oarlock forces
            left_f, right_f = sim.get_oarlock_forces(t, psi, phi, rower_idx)
            left_forces.append(left_f)
            right_forces.append(right_f)

            # Calculate stroke phase for visualization
            t_mod = t % sim.stroke_period
            if t_mod <= sim.params.active_phase_duration:
                phase = t_mod / sim.params.active_phase_duration  # 0 to 1 during active
            else:
                phase = 1 + (t_mod - sim.params.active_phase_duration) / sim.recovery_duration  # 1 to 2 during recovery
            stroke_phases.append(phase)

    # Convert to numpy arrays
    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    left_forces = np.array(left_forces)
    right_forces = np.array(right_forces)
    stroke_phases = np.array(stroke_phases)

    # Create the plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Rowing Kinematics and Forces Comparison', fontsize=16, fontweight='bold')

    # Add stroke phase background
    for ax_row in axes:
        for ax in ax_row:
            # Highlight active phases (phase 0-1) in light blue
            for i in range(n_cycles):
                t_start = i * sim.stroke_period
                t_active_end = t_start + sim.params.active_phase_duration
                ax.axvspan(t_start, t_active_end, alpha=0.2, color='lightblue', label='Active Phase' if i == 0 else "")
                ax.axvspan(t_active_end, (i + 1) * sim.stroke_period, alpha=0.2, color='lightcoral',
                           label='Recovery Phase' if i == 0 else "")

    # Plot 1: Position and Forces X-component
    ax1 = axes[0, 0]
    ax1.plot(time, positions[:, 0], 'b-', linewidth=2, label='Body X Position (m)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time, left_forces[:, 0], 'r--', linewidth=2, label='Left Oar X Force (N)')
    ax1_twin.plot(time, right_forces[:, 0], 'g--', linewidth=2, label='Right Oar X Force (N)')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)', color='b')
    ax1_twin.set_ylabel('Force (N)', color='r')
    ax1.set_title('X-Direction: Body Position vs Oar Forces')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Velocity and Force derivatives
    ax2 = axes[0, 1]
    ax2.plot(time, velocities[:, 0], 'b-', linewidth=2, label='Body X Velocity (m/s)')
    # Calculate force derivatives
    force_derivative = np.gradient(left_forces[:, 0], dt)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time, force_derivative, 'r--', linewidth=2, label='Force X Derivative (N/s)')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)', color='b')
    ax2_twin.set_ylabel('Force Derivative (N/s)', color='r')
    ax2.set_title('X-Direction: Body Velocity vs Force Rate')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Acceleration and Forces
    ax3 = axes[1, 0]
    ax3.plot(time, accelerations[:, 0], 'b-', linewidth=2, label='Body X Acceleration (m/s²)')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time, left_forces[:, 0], 'r--', linewidth=2, label='Left Oar X Force (N)')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)', color='b')
    ax3_twin.set_ylabel('Force (N)', color='r')
    ax3.set_title('X-Direction: Body Acceleration vs Oar Forces')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Phase diagram
    ax4 = axes[1, 1]
    ax4.plot(time, stroke_phases, 'k-', linewidth=2, label='Stroke Phase')
    ax4.axhline(y=1, color='orange', linestyle=':', alpha=0.7, label='Active/Recovery Boundary')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Stroke Phase (0-2)')
    ax4.set_title('Stroke Phase Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 2.1)

    # Plot 5: Z-direction comparison
    ax5 = axes[2, 0]
    ax5.plot(time, positions[:, 2], 'b-', linewidth=2, label='Body Z Position (m)')
    ax5_twin = ax5.twinx()
    ax5_twin.plot(time, left_forces[:, 2], 'r--', linewidth=2, label='Left Oar Z Force (N)')
    ax5_twin.plot(time, right_forces[:, 2], 'g--', linewidth=2, label='Right Oar Z Force (N)')

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Z Position (m)', color='b')
    ax5_twin.set_ylabel('Z Force (N)', color='r')
    ax5.set_title('Z-Direction: Body Position vs Oar Forces')
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Combined phase plot
    ax6 = axes[2, 1]
    # Normalize for comparison
    pos_norm = (positions[:, 0] - np.mean(positions[:, 0])) / np.std(positions[:, 0])
    force_norm = left_forces[:, 0] / np.max(np.abs(left_forces[:, 0]))

    ax6.plot(time, pos_norm, 'b-', linewidth=2, label='Normalized Body Position')
    ax6.plot(time, force_norm, 'r--', linewidth=2, label='Normalized Oar Force')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Normalized Values')
    ax6.set_title('Phase Relationship (Normalized)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print some analysis
    print("\n=== ANALYSIS ===")
    print(f"Stroke Period: {sim.stroke_period:.2f} s")
    print(f"Active Phase Duration: {sim.params.active_phase_duration:.2f} s")
    print(f"Recovery Phase Duration: {sim.recovery_duration:.2f} s")
    print(f"Body Position Range: {np.min(positions[:, 0]):.3f} to {np.max(positions[:, 0]):.3f} m")
    print(f"Max Oar Force: {np.max(left_forces[:, 0]):.1f} N")

    # Check phase alignment
    active_mask = stroke_phases < 1
    force_during_active = np.mean(np.abs(left_forces[active_mask, 0]))
    force_during_recovery = np.mean(np.abs(left_forces[~active_mask, 0]))
    print(f"Average force during active phase: {force_during_active:.1f} N")
    print(f"Average force during recovery phase: {force_during_recovery:.1f} N")


if __name__ == "__main__":
    plot_rowing_dynamics()