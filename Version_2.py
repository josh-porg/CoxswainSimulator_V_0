import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Callable


@dataclass
class RowingParameters:
    """Parameters for the rowing simulation"""
    # Boat properties
    # hull_mass: float = 49.6  # kg
    hull_mass: float = 100 # kg
    hull_inertia: np.ndarray = None  # 3x3 inertia tensor in hull frame

    # Rower properties
    n_rowers: int = 4
    rower_mass: float = 85.0  # kg per rower
    n_body_parts: int = 12  # body parts per rower
    rower_spacing = 2 # m between rowers

    # Rowing kinematics
    cadence: float = 30.0  # strokes per minute
    active_phase_duration: float = 0.712  # seconds

    # Force parameters
    F_max_x: float = 1200.0  # N, maximum longitudinal oarlock force
    F_max_z: float = 200.0  # N, maximum vertical oarlock force

    # Control gains
    k_roll: float = 1000.0  # Roll control gain
    k_yaw: float = 500.0  # Yaw control gain

    # Environment
    gravity: float = 9.81  # m/s^2
    h0: float = 0.0  # undisturbed water surface level

    def __post_init__(self):
        if self.hull_inertia is None:
            # Default inertia tensor (typical values for a rowing shell)
            self.hull_inertia = np.diag([10.0, 500.0, 500.0])  # kg⋅m²


class RowingShell:
    """6DOF rowing shell dynamics simulation"""

    def __init__(self, params: RowingParameters):
        self.params = params
        self.total_mass = params.hull_mass + params.n_rowers * params.rower_mass

        # Time parameters
        self.stroke_period = 60.0 / params.cadence  # seconds per stroke
        self.recovery_duration = self.stroke_period - params.active_phase_duration

        # State variables: [G_h (3), Euler angles (3), velocity (3), angular velocity (3)]
        self.state_size = 12

    def euler_to_rotation_matrix(self, psi: float, theta: float, phi: float) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix
        Args:
            psi: yaw angle
            theta: pitch angle
            phi: roll angle
        Returns:
            3x3 rotation matrix
        """
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        R = np.array([
            [cos_theta * cos_psi,
             sin_phi * sin_theta * cos_psi - cos_phi * sin_psi,
             cos_phi * sin_theta * cos_psi + sin_phi * sin_psi],
            [cos_theta * sin_psi,
             sin_phi * sin_theta * sin_psi + cos_phi * cos_psi,
             cos_phi * sin_theta * sin_psi - sin_phi * cos_psi],
            [-sin_theta, sin_phi * cos_theta, cos_phi * cos_theta]
        ]).T
        return R

    def skew_symmetric_matrix(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def get_oarlock_forces(self, t: float, psi: float, phi: float,
                           rower_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate oarlock forces for given rower
        Args:
            t: current time
            psi: yaw angle
            phi: roll angle
            rower_idx: rower index
        Returns:
            tuple of (left_oarlock_force, right_oarlock_force) as 3D vectors
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

    def get_body_part_kinematics(self, t: float, rower_idx: int, part_idx: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Placeholder for body part kinematics
        Returns position, velocity, acceleration of body part in hull frame
        """
        # Placeholder implementation - replace with actual kinematics
        # Based on experimental measurements or biomechanical models

        # for now lets space rowers 2 meters apart
        rower_spacing = 2 #m distance between rowers

        # Different motion for different body parts
        # amplitude = 0.5 * (part_idx + 1) / self.params.n_body_parts
        amplitude = .4  # so .3 to .5 seems about right for a realistic rowing stroke
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
            t_mod = ((t % self.stroke_period) - self.params.active_phase_duration) % self.recovery_duration
            omega = np.pi / self.recovery_duration  # ToDO: smooth the transition from drive to recovery by making omega and offset a function of t%stroke period or by avergin the 2 until smooth
            offset = np.pi

        # matched with stroke nd recovery
        if (rower_idx == 0) and (part_idx == 0):
            print(f"phase_ratio_body: {(omega * t_mod + offset) / np.pi}")
        x_rel = amplitude * np.cos(omega * t_mod + offset) + (self.params.n_rowers / 2 - rower_idx) * self.params.rower_spacing
        v_rel = -amplitude * omega * np.sin(omega * t_mod + offset)
        a_rel = -amplitude * omega ** 2 * np.cos(omega * t_mod + offset)

        clip_width = 2 / 12 * part_idx + 1 # 0 is dont clip
        clip_offset = -clip_width / 2
        # clip_offset = 0

        x_rel = np.cos(omega * t_mod + offset)  # + (self.params.n_rowers / 2 - rower_idx) * self.params.rower_spacing
        x_rel = amplitude * np.tanh(clip_width * x_rel + clip_offset)
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

        # x_rel = amplitude * omega * t_mod**2 +  + (
        #             self.params.n_rowers / 2 - rower_idx) * self.params.rower_spacing
        # v_rel = 2*amplitude * omega * t_mod * omega
        # a_rel = 2*amplitude * omega * omega

        #TODO: match this up with drive and recovery timing

        # not matched to stroke and recovery
        # x_rel = amplitude * np.sin(omega * t) + (self.params.n_rowers/2 - rower_idx) * self.params.rower_spacing
        # v_rel = amplitude * omega * np.cos(omega * t)
        # a_rel = -amplitude * omega ** 2 * np.sin(omega * t)

        position = np.array([x_rel, 0, vert_amplitude * 0.05 * np.sin(2 * omega * t)])
        velocity = np.array([v_rel, 0, vert_amplitude * 0.1 * omega * np.cos(2 * omega * t)])
        acceleration = np.array([a_rel, 0, vert_amplitude * -0.2 * omega ** 2 * np.sin(2 * omega * t)])

        return position, velocity, acceleration

    def get_hydrodynamic_forces(self, state: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Placeholder for hydrodynamic forces and moments
        Args:
            state: current state vector [position, angles, velocity, angular_velocity]
            t: current time
        Returns:
            tuple of (force_vector, moment_vector) in absolute frame
        """
        # Extract state variables
        position = state[0:3]
        angles = state[3:6]
        velocity = state[6:9]
        angular_velocity = state[9:12]

        # Placeholder implementation - replace with actual hydrodynamic model
        # Simple drag model for demonstration

        speed = np.linalg.norm(velocity)
        if speed > 0:
            drag_coefficient = 50.0  # N⋅s²/m²
            drag_force = -drag_coefficient * speed * velocity
        else:
            drag_force = np.zeros(3)

        # Simple damping for angular motion
        angular_damping = -100.0 * angular_velocity

        # Add some wave resistance based on vertical motion
        wave_resistance = np.array([0, 0, -100.0 * velocity[2]])

        bouyancy = np.array([0,0, -1*state[2]])
        bouyancy = np.zeros_like(bouyancy)

        total_force = drag_force + wave_resistance + bouyancy
        total_moment = angular_damping

        return total_force, total_moment

    def compute_mass_matrix(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the time-dependent mass matrix M(t)
        """
        # Extract current orientation
        psi, theta, phi = state[3:6]
        omega = state[9:12]

        R = self.euler_to_rotation_matrix(psi, theta, phi)

        # Initialize matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))

        # Sum contributions from all body parts
        for j in range(self.params.n_rowers):
            for i in range(self.params.n_body_parts):
                # Get body part mass (simplified - equal distribution)
                m_ij = self.params.rower_mass / self.params.n_body_parts

                # Get body part position in hull frame
                x_ij, _, _ = self.get_body_part_kinematics(t, j, i)
                v_ij = R.T @ x_ij  # Transform to hull frame

                # print(f"x_ij: {x_ij}")

                # Add to mass matrix terms
                A += m_ij * self.skew_symmetric_matrix(v_ij)
                B += m_ij * self.skew_symmetric_matrix(v_ij) @ self.skew_symmetric_matrix(v_ij)

        # Inertia tensor in absolute frame
        I_abs = R @ self.params.hull_inertia @ R.T

        # Assemble mass matrix
        M = np.zeros((6, 6))
        M[0:3, 0:3] = self.total_mass * np.eye(3)
        M[0:3, 3:6] = A
        M[3:6, 0:3] = -A
        M[3:6, 3:6] = I_abs + B

        return M

    def compute_forces_and_moments(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute right-hand side forces and moments
        """
        # Extract state variables
        G_h = state[0:3]
        angles = state[3:6]
        velocity = state[6:9]
        omega = state[9:12]

        psi, theta, phi = angles
        R = self.euler_to_rotation_matrix(psi, theta, phi)

        # Initialize force and moment vectors
        f_total = np.zeros(3)
        M_total = np.zeros(3)

        # 1. Oarlock forces
        oarlock_positions = []  # Positions of oarlocks relative to hull center
        hand_positions = []  # Positions of hands relative to hull center

        for j in range(self.params.n_rowers):
            # Simplified oarlock and hand positions
            x_oarlock = (self.params.n_rowers/2 - j) * self.params.rower_spacing  # Longitudinal spacing
            oarlock_left = np.array([x_oarlock, -0.8, 0.3])
            oarlock_right = np.array([x_oarlock, 0.8, 0.3])
            hand_left = np.array([x_oarlock - 0.5, -0.3, 0.5])
            hand_right = np.array([x_oarlock - 0.5, 0.3, 0.5])

            # Get oarlock forces
            F_ol, F_or = self.get_oarlock_forces(t, psi, phi, j)

            # Transform to absolute frame
            F_ol_abs = R.T @ F_ol
            F_or_abs = R.T @ F_or

            # Add to total force
            f_total += F_ol_abs + F_or_abs # TODO reintroduce oarlock forces

            # Lever arm contribution (simplified)
            lever_length = 2.0  # meters
            r_h_to_oarlock_ratio = 0.3  # ratio of handle to oarlock distance

            # Moments from oarlock forces
            M_ol = np.cross(oarlock_left, F_ol_abs) - r_h_to_oarlock_ratio * np.cross(hand_left, F_ol_abs)
            M_or = np.cross(oarlock_right, F_or_abs) - r_h_to_oarlock_ratio * np.cross(hand_right, F_or_abs)

            # M_total += M_ol + M_or # TODO: oarlock moments cause blowup. fix it!

        # 2. Rower inertial forces
        f_inertial = np.zeros(3)
        M_inertial = np.zeros(3)


        for j in range(self.params.n_rowers):
            for i in range(self.params.n_body_parts):
                m_ij = self.params.rower_mass / self.params.n_body_parts

                x_ij, v_ij, a_ij = self.get_body_part_kinematics(t, j, i)

                # Transform to absolute frame
                x_ij_abs = R.T @ x_ij
                v_ij_abs = R.T @ v_ij
                a_ij_abs = R.T @ a_ij

                # Coriolis and centrifugal terms
                # coriolis = 2 * np.cross(omega, v_ij_abs) # i belive these should be in hull frame not abs
                # centrifugal = np.cross(omega, np.cross(omega, x_ij_abs))

                coriolis = 2 * np.cross(omega, R.T @ v_ij)
                centrifugal = np.cross(omega, np.cross(omega, R.T @ x_ij))

                # Add inertial forces
                f_inertial -= m_ij * (R.T @ a_ij + coriolis + centrifugal)
                # M_inertial -= m_ij * np.cross(R.T @ x_ij, R.T @ a_ij + coriolis + centrifugal) # TODO: reintoduce initrial moment

            # print(f"f_inertial: {f_inertial}")

        #TODO: reintroduce interial forces when done debugging
        f_total += f_inertial
        M_total += M_inertial

        # 3. Gravity
        gravity_force = np.array([0, 0, -self.total_mass * self.params.gravity])
        # f_total += gravity_force #TODO: reintroduce gravity forces

        # 4. Hydrodynamic forces
        F_hydro, M_hydro = self.get_hydrodynamic_forces(state, t)
        f_total += F_hydro
        # M_total += M_hydro

        # print(f"F_hydro:{F_hydro}")

        # Gyroscopic terms for angular momentum equation
        I_abs = R @ self.params.hull_inertia @ R.T
        gyro_term = -np.cross(omega, I_abs @ omega)
        # M_total += gyro_term

        # print(f"f_total:{f_total}")
        # print(f"M_total:{M_total}")
        # M_total = np.zeros_like(M_total) # for dubug remove roational dof TODO: reintroduce moments

        return np.concatenate([f_total, M_total])

    @staticmethod
    def euler_angle_kinematics(euler_angles, omega):
        """
        Convert angular velocity vector to Euler angle derivatives.

        Args:
            euler_angles: [phi, theta, psi] (roll, pitch, yaw) in radians
            omega: [p, q, r] angular velocity vector in body frame (rad/s)

        Returns:
            dangles_dt: [dphi_dt, dtheta_dt, dpsi_dt] Euler angle derivatives
        """
        phi, theta, psi = euler_angles
        p, q, r = omega

        # Avoid singularity at theta = ±π/2
        cos_theta = np.cos(theta)
        if abs(cos_theta) < 1e-6:
            # Near singularity - use alternative approach or limit
            print("Warning: Near gimbal lock condition (theta ≈ ±π/2)")
            cos_theta = np.sign(cos_theta) * 1e-6

        sin_theta = np.sin(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        tan_theta = np.tan(theta)

        # Kinematic transformation matrix (body angular rates to Euler derivatives)
        # This is the inverse of the standard rotation sequence transformation
        dphi_dt = p + (q * sin_phi + r * cos_phi) * tan_theta
        dtheta_dt = q * cos_phi - r * sin_phi
        dpsi_dt = (q * sin_phi + r * cos_phi) / cos_theta

        dangles_dt = np.array([dphi_dt, dtheta_dt, dpsi_dt])
        return dangles_dt

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        System dynamics for ODE solver
        Args:
            t: current time
            y: state vector [G_h, angles, velocity, angular_velocity]
        Returns:
            derivative of state vector
        """
        # Extract state
        position = y[0:3]
        angles = y[3:6]
        velocity = y[6:9]
        omega = y[9:12]

        # Position derivatives
        dposition_dt = velocity

        # kimematic equations for roatation
        dangles_dt = self.euler_angle_kinematics(angles, omega)

        # Velocity and angular velocity derivatives from equations of motion
        state_for_forces = np.concatenate([position, angles, velocity, omega])

        # Compute mass matrix and forces
        M = self.compute_mass_matrix(state_for_forces, t)
        f = self.compute_forces_and_moments(state_for_forces, t)

        # print(f"mass: {M}")
        # print(f"forces: {f}")


        # Solve M * acceleration = f
        try:
            acceleration = np.linalg.solve(M, f)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            acceleration = np.zeros(6)
            print(f"Warning: Singular mass matrix at t={t}")

        dvelocity_dt = acceleration[0:3]
        domega_dt = acceleration[3:6]

        # print(f"angualr accelration: {acceleration[3:6]}")

        return np.concatenate([dposition_dt, dangles_dt, dvelocity_dt, domega_dt])

    def simulate(self, t_span: Tuple[float, float], initial_state: np.ndarray,
                 max_step: float = 0.1) -> dict:
        """
        Run the simulation
        Args:
            t_span: (t_start, t_end) time span
            initial_state: initial state vector [G_h, angles, velocity, omega]
            max_step: maximum integration step size
        Returns:
            dictionary with simulation results
        """

        def dynamics_wrapper(t, y):
            return self.dynamics(t, y)

        # Solve ODE
        # solution = solve_ivp(
        #     dynamics_wrapper,
        #     t_span,
        #     initial_state,
        #     method='RK45',
        #     max_step=max_step,
        #     rtol=1e-3,
        #     atol=1e-6
        #     #rtol=1e-6,
        #     #atol=1e-8
        # )

        # solve manually
        state = initial_state
        n_steps = int(t_span[1]/max_step)
        y = np.zeros([len(state), n_steps])
        t = 0
        for i in range(0, n_steps):
            y[:,i] = state
            state += dynamics_wrapper(t,state) * max_step
            t += max_step

        # if not solution.success:
        #     print(f"Integration failed: {solution.message}")

        # results = {
        #     't': solution.t,
        #     'position': solution.y[0:3, :],
        #     'angles': solution.y[3:6, :],
        #     'velocity': solution.y[6:9, :],
        #     'angular_velocity': solution.y[9:12, :],
        #     'success': solution.success,
        #     'message': solution.message
        # }
        results = {
            't': [i*max_step for i in range(0,n_steps)],
            'position': y[0:3, :],
            'angles': y[3:6, :],
            'velocity': y[6:9, :],
            'angular_velocity': y[9:12, :],
            'success': True,
            'message': "this is unnecesary"
        }

        return results


def plot_results(results: dict):
    """Plot simulation results"""
    t = results['t']
    pos = results['position']
    angles = results['angles'] * 180 / np.pi  # Convert to degrees
    vel = results['velocity']
    omega = results['angular_velocity'] * 180 / np.pi  # Convert to degrees/s

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Rowing Shell 6DOF Simulation Results')

    # Position plots
    axes[0, 0].plot(t, pos[0], 'b-', label='X (surge)')
    axes[0, 0].plot(t, pos[1], 'r-', label='Y (sway)')
    axes[0, 0].plot(t, pos[2], 'g-', label='Z (heave)')
    axes[0, 0].set_ylabel('Position [m]')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Position')

    # Angle plots
    axes[0, 1].plot(t, angles[0], 'b-', label='Yaw (ψ)')
    axes[0, 1].plot(t, angles[1], 'r-', label='Pitch (θ)')
    axes[0, 1].plot(t, angles[2], 'g-', label='Roll (φ)')
    axes[0, 1].set_ylabel('Angles [deg]')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title('Orientation')

    # Velocity plots
    axes[1, 0].plot(t, vel[0], 'b-', label='u')
    axes[1, 0].plot(t, vel[1], 'r-', label='v')
    axes[1, 0].plot(t, vel[2], 'g-', label='w')
    axes[1, 0].set_ylabel('Velocity [m/s]')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Linear Velocity')

    # Angular velocity plots
    axes[1, 1].plot(t, omega[0], 'b-', label='p')
    axes[1, 1].plot(t, omega[1], 'r-', label='q')
    axes[1, 1].plot(t, omega[2], 'g-', label='r')
    axes[1, 1].set_ylabel('Angular Velocity [deg/s]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title('Angular Velocity')

    # Trajectory (top view)
    axes[2, 0].plot(pos[0], pos[1], 'b-', linewidth=2)
    axes[2, 0].set_xlabel('X [m]')
    axes[2, 0].set_ylabel('Y [m]')
    axes[2, 0].grid(True)
    axes[2, 0].set_title('Trajectory (Top View)')
    axes[2, 0].axis('equal')

    # Speed
    speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
    axes[2, 1].plot(t, speed, 'k-', linewidth=2)
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_ylabel('Speed [m/s]')
    axes[2, 1].grid(True)
    axes[2, 1].set_title('Boat Speed')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create simulation parameters
    params = RowingParameters(
        n_rowers=4,
        cadence=40.0,
        hull_mass=49.6,
        rower_mass=85.0
    )

    # Initialize simulation
    shell = RowingShell(params)

    # Initial conditions: [position, angles, velocity, angular_velocity]
    initial_state = np.array([
        0.0, 0.0, 0.0,  # Initial position (x, y, z)
        0.0, 0.0, 0.0,  # Initial angles (yaw, pitch, roll)
        5.0, 0.0, 0.0,  # Initial velocity (forward speed)
        0.0, 0.0, 0.0  # Initial angular velocity
    ])

    # Run simulation for 30 seconds
    results = shell.simulate((0, 30), initial_state)

    # Plot results
    if results['success']:
        plot_results(results)
        print("Simulation completed successfully!")
    else:
        print(f"Simulation failed: {results['message']}")