import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List




@dataclass
class BoatParameters:
    """Physical parameters of the boat"""
    mass: float = 100.0 + 8 * 68 + 68  # kg (shell + 8 rowers + coxswain)
    #moment_inertia: float = 500.0  # kg*m^2 (about vertical axis)
    length: float = 17.3  # m (boat length for reference)
    width: float = .75  # m (boat width for reference)
    draft: float = .3 # m(submerged depth of hull)
    moment_inertia = (1/12 * .75 * 17.3**3) * mass / (length * width) # moment of inertia is second moment of area times denisty - assumes unifrom density, and recangle
    F_max_x: float = 1200 #178*8 #2000.0  # N (maximum force a rower excerts on their oarlock) want to move to
    F_max_z: float = 200  # 178*8 #2000.0  # N (maximum force a rower excerts on their oarlock) want to move to
    ref_area: float = .5 # m^2 (submerged corssection area)
    n_oars = 8 # number of oars

    # Water density
    rho_water: float = 1025.0  # kg/m^3
    mu_water: float = 0.00982 # N*s/m (absolute viscosity)
    nu_water: float = mu_water / rho_water # m^2/s (kinematic viscosity)

    # Hull coefficients
    hull_C_D: float = 8* 12.5 * 2 / rho_water / ref_area # Hull drag coefficient (computed from paper -scaled to coefficecent and scaled to make good results)
    hull_C_Y_beta: float = -0.1  # Hull sideforce due to sideslip
    hull_C_N_beta: float = 0.05  # Hull yaw moment due to sideslip
    hull_area: float = 4.0  # m^2 (hull reference area)

    # parameters for drag computed by first approximation from A model for the dyamics of rowing boats formaggia et al
    gamma = length * 2 * np.pi * (width + draft) / 2 # m^2 wetted surface (approximated as a cylinder)
    gamma_x = length * draft # m^2 (projection of wetted surface area perpendicular to the x-axis)
    gamma_z = 2 * np.pi * (width + draft) / 2 # m^2 (projection of wetted surface area perpendicular to the x-axis)
    C_d_x: float = .01 # shape resistance coefficient (typically about .01)
    C_f_0: float = .075 # skin friction coefficient (typically .075)
    C_d_w: float = .02 # wave resistance coefficient (for typical skulls .02)

    # skeg geometry
    skeg_span: float = 7 * 17.3/941  # m (span)
    skeg_chord: float = 11 * 17.3/941  # m (mean geometric chord)
    skeg_sweep_angle: float = np.radians(5)  # 15° sweep

    # Skeg coefficients
    skeg_C_D: float = 0.1  # Skeg drag coefficient
    #skeg_C_Y_beta: float = -0.2  # Skeg sideforce due to sideslip
    skeg_x_ac = 126 * 17.3/941 # distance from bowball to location of skeg aerodynamic center
    skeg_C_Y_delta_f: float = 0.3  # Skeg sideforce due to rudder deflection
    skeg_C_N_beta: float = 0.1  # Skeg yaw moment due to sideslip
    skeg_C_N_delta_f: float = 0.15  # Skeg yaw moment due to rudder deflection
    skeg_area: float = .25  # m^2 (skeg reference area)

    # assume cg is at midpoint of boat
    x_cg = length/2

    # Control input
    delta_f: float = np.pi/16  # rad (rudder deflection angle) # TODO: reset delta f to zero when done debugging
    rate = 30 # strokes per minute (stroke rate)
    # rower based control
    k_yaw: float = 500
    k_roll: float = 1000

    # Damping coefficients
    C_N_r: float = -0.05  # Yaw damping coefficient

    def __post_init__(self):
        """Initialize computed parameters after dataclass creation"""
        # Compute skeg area from geometry
        self.skeg_area = self.skeg_span * self.skeg_chord

        # Compute skeg aspect ratio
        skeg_aspect_ratio = self.skeg_span ** 2 / self.skeg_area

        # Use Polhamus formula to compute C_Y_beta (equivalent to C_L_alpha)
        self.skeg_C_Y_beta = self.polhamus_lift_coefficient(skeg_aspect_ratio, self.skeg_sweep_angle)

        # dimentionless moment arms
        self.x_bar_cg = self.x_cg / self.length
        skeg_x_bar_ac = self.skeg_x_ac / self.length

        # compute C_N_beta
        self.skeg_C_N_beta = self.skeg_area / self.ref_area * self.skeg_C_Y_beta * (skeg_x_bar_ac - self.x_bar_cg)

        # compute C_D of skeg ( approximately for now assuming mix of laminar and turbluent and a resoanble velocity
        v_guess = 4 # guessed velocity in meters per second for drag computation
        reynolds_number = v_guess * self.skeg_chord / 1E-6 # reynolds number for room temp water
        C_D_laminar = 2.656 / np.sqrt(reynolds_number) # if flow is laminar
        C_D_transition = .062 / reynolds_number**(1/7)  # if flow is in trasiotion from laminar to turbulent
        C_D_turbulent = .148 / reynolds_number ** (1 / 5)  # if flow is in trasiotion from laminar to turbulent
        self.skeg_C_D: float = C_D_turbulent  # Skeg drag coefficient

        print(f"Skeg initialized with:")
        print(f"  Aspect Ratio: {skeg_aspect_ratio:.2f}")
        print(f"  Sweep Angle: {np.degrees(self.skeg_sweep_angle):.1f} deg")
        print(f"  C_Y_beta (Polhamus): {self.skeg_C_Y_beta:.3f} /rad")

    @property
    def total_C_D(self) -> float:
        """Total boat drag coefficient (hull + skeg)"""
        return self.hull_C_D + self.skeg_C_D * self.skeg_area / self.ref_area

    @property
    def total_C_Y_beta(self) -> float:
        """Total Boat sideforce coefficient due to sideslip (hull + skeg)"""
        return self.hull_C_Y_beta + self.skeg_area / self.ref_area * self.skeg_C_Y_beta

    @property
    def total_area(self) -> float:
        """Total reference area (hull + skeg)"""
        return self.hull_area + self.skeg_area

    @property
    def stroke_period(self) -> float:
        return 60.0 / self.rate  # seconds per stroke

    @property
    def active_phase_duration(self) -> float:
        return .00015625 * (self.rate - 24) ** 2 - .008125 * (self.rate - 24) + .8 # from A model for the dyamics of rowing boats formaggia et al

    @staticmethod
    def polhamus_lift_coefficient(aspect_ratio: float, sweep_angle_rad: float = 0.0) -> float:
        """
        Compute lift coefficient using Polhamus formula for delta wings/low aspect ratio wings

        Args:
            aspect_ratio: Wing aspect ratio (b^2/S)
            sweep_angle_rad: Leading edge sweep angle in radians (default 0 for unswept)

        Returns:
            CL_alpha: Lift coefficient per radian of angle of attack
        """
        # Polhamus formula for low aspect ratio wings
        # CL_alpha = (2 * pi * AR) / (2 + sqrt(4 + AR^2 * (1 + tan^2(sweep))))

        cos_sweep = np.cos(sweep_angle_rad)
        tan_sweep = np.tan(sweep_angle_rad)

        # Modified aspect ratio accounting for sweep
        AR_eff = aspect_ratio * cos_sweep ** 2

        # Polhamus formula
        denominator = 2 + np.sqrt(4 + AR_eff ** 2 * (1 + tan_sweep ** 2))
        CL_alpha = (2 * np.pi * AR_eff) / denominator

        return CL_alpha


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
            'rate': [],
            'beta': [],
            'V': []
        }

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
        t_mod = t % self.params.stroke_period

        if t_mod <= self.params.active_phase_duration:
            # Active phase
            phase_ratio = t_mod / self.params.active_phase_duration
            force_multiplier = np.sin(np.pi * phase_ratio)

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
            #for now we also need the uncontrolled f_x
            f_x = 0.0

        # Roll control (active throughout stroke cycle)
        roll_control = -self.params.k_roll * phi
        f_z_controlled = f_z + roll_control

        # Forces in hull coordinate system
        # left_force = np.array([f_x_controlled, 0, f_z_controlled])
        # right_force = np.array([f_x_controlled, 0, -f_z_controlled])  # Opposite z-component
        # currently uncontrolled TODO: reintroduce control when 6DOF
        left_force = np.array([f_x, 0, f_z])
        right_force = np.array([f_x, 0, -f_z])  # Opposite z-component

        return left_force, right_force

    def forces_and_moments(self) -> Tuple[float, float, float]:
        """
        Calculate forces and moments based on current velocity and sideslip angle
        Returns: (F_x, F_y, M_z) in body frame
        """
        u, v = self.state.u, self.state.v
        V = self.state.V
        beta = self.state.beta

        # Thrust force (along body x-axis)
        tau_a = .00015625 * (self.params.rate - 24) ** 2 - .008125 * (self.params.rate - 24) + .8 # from A model for the dyamics of rowing boats formaggia et al
        F_oarlock_x = self.params.F_max_x * np.sin(np.pi * self.time / tau_a) # from A model for the dyamics of rowing boats formaggia et al
        # F_thrust = self.params.thrust * np.abs(np.sin(np.pi * self.time/2))
        F_thrust = np.max([0.0,F_oarlock_x * self.params.n_oars])
        F_thrust = self.get_oarlock_forces(self.time, 0, 0)[0][0] # todo replace  with real roll angle when 6DOF

        F_thrust = F_thrust * self.params.n_oars

        # Drag force (opposite to velocity direction in body frame)
        if V > 1e-6:
            q = 0.5 * self.params.rho_water * V ** 2  # Dynamic pressure
            F_drag_total = self.params.total_C_D * self.params.ref_area * q

            # drag computed by first approximation from A model for the dyamics of rowing boats formaggia et al
            F_D_shape = q * self.params.gamma_x * self.params.C_d_x # shape drag
            Re = u * self.params.length / self.params.nu_water # Reynolds number TODO: this should be mean submerged length and mean velocity
            C_d_vis = self.params.C_f_0 / (np.log(Re) - 2)**2 # viscus drag coefficient
            F_D_vis = q * self.params.gamma * C_d_vis # viscus drag
            F_D_wave = q * self.params.gamma_z * self.params.C_d_w # wave drag

            F_drag_total = F_D_shape + F_D_vis + F_D_wave # total steady state drag from A model for the dyamics of rowing boats formaggia et al

            # Drag components in body frame
            F_drag_x = -F_drag_total * np.cos(beta)
            F_drag_y = -F_drag_total * np.sin(beta)

            #TODO: add induced drag

            # Side force in body frame
            F_side_beta = self.params.total_C_Y_beta * beta * self.params.ref_area * q
            F_side_delta_f = self.params.skeg_C_Y_delta_f * self.params.delta_f * np.cos(np.pi*self.time/10) * self.params.ref_area * q

            F_side_total = F_side_beta + F_side_delta_f

            # rotate from airflow coordinates to body
            F_side_x = - F_side_total * np.sin(beta)
            F_side_y = F_side_total * np.cos(beta)

        else:
            F_drag_x = 0.0
            F_drag_y = 0.0
            F_side_x = 0.0
            F_side_y = 0.0
            q = 0


        # generate forces due to wind
        # F_wind = 500
        # psi_wind = 1/2* np.pi
        # F_wind_x_inertial = F_wind * np.cos(-psi_wind)
        # F_wind_y_inertial = F_wind * np.sin(-psi_wind)
        # F_wind_x = F_wind_x_inertial * np.cos(self.state.psi) - F_wind_y_inertial * np.sin(self.state.psi)
        # F_wind_y = F_wind_x_inertial * np.sin(self.state.psi) + F_wind_y_inertial * np.cos(self.state.psi)

        # Total forces in body frame
        F_x = F_thrust + F_drag_x + F_side_x #+ F_wind_x
        F_y = F_drag_y + F_side_y #+ F_wind_y

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
        self.history['rate'].append(np.degrees(self.state.r))
        self.history['beta'].append(np.degrees(self.state.beta))
        self.history['V'].append(self.state.V)

    def simulate(self, duration: float, dt: float = 0.01):
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
        axes[1, 1].plot(self.history['time'], self.history['rate'], 'm-', linewidth=2)
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