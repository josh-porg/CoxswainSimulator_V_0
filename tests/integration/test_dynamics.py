"""Integration tests for the assembled 6-DOF dynamics.

These exercise the simulator end to end but stay fast by using short runs
and a coarse-but-stable step.  Physics-level invariants are checked here:
conservation, static equilibrium, frame consistency, and that every degree
of freedom is actually live.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.core.frames import PITCH, ROLL, YAW, hull_to_abs
from coxswain.core.rigid_body import assemble_mass_matrix
from coxswain.core.state import State
from coxswain.crew.oarlock import OarForceProfile
from coxswain.sim import RowingSimulator
from coxswain.sim.control import BalanceController, Coxswain, HeadingController

GRAVITY = 9.81


@pytest.fixture(scope="module")
def boat():
    return catalog.eight(rate=32.0)


@pytest.fixture(scope="module")
def sim(boat):
    return RowingSimulator(boat)


@pytest.fixture(scope="module")
def trimmed(sim):
    return State.from_vector(sim.initial_state(surge_speed=5.0))


# --------------------------------------------------------------------------
# mass matrix, in situ
# --------------------------------------------------------------------------
def test_mass_matrix_is_positive_definite_through_a_whole_stroke(sim, trimmed):
    """The regression guard for the original blow-up, with a real crew."""
    for t in np.linspace(0.0, sim.boat.timing.period, 25, endpoint=False):
        matrix = sim.mass_matrix(t, trimmed)
        smallest = np.linalg.eigvalsh(matrix).min()
        assert smallest > 0.0, f"indefinite mass matrix at t={t:.3f}"


def test_mass_matrix_is_symmetric(sim, trimmed):
    matrix = sim.mass_matrix(0.3, trimmed)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-9, atol=1e-6)


def test_crew_increases_rotational_inertia(sim, trimmed):
    """With the legacy sign the crew subtracted inertia and drove it negative."""
    with_crew = sim.mass_matrix(0.3, trimmed)[3:6, 3:6]
    bare = assemble_mass_matrix(sim.boat.total_mass, sim.boat.hull_inertia,
                                np.zeros(0), np.zeros((0, 3)))[3:6, 3:6]
    assert np.all(np.diag(with_crew) > np.diag(bare))


def test_translational_block_is_the_total_mass(sim, trimmed):
    matrix = sim.mass_matrix(0.0, trimmed)
    np.testing.assert_allclose(np.diag(matrix[0:3, 0:3]),
                               sim.boat.total_mass, rtol=1e-12)


# --------------------------------------------------------------------------
# static equilibrium
# --------------------------------------------------------------------------
def test_buoyancy_balances_weight_in_trim(sim, trimmed):
    forces = sim.breakdown(0.0, trimmed)
    net = forces.buoyancy_force[2] + forces.gravity_force[2]
    assert abs(net) < 0.01 * sim.boat.total_mass * GRAVITY


def test_pitch_moment_balances_in_trim(sim, trimmed):
    forces = sim.breakdown(0.0, trimmed)
    net = forces.buoyancy_moment[1] + forces.gravity_moment[1]
    assert abs(net) < 100.0


def test_gravity_force_uses_the_total_mass(sim, trimmed):
    forces = sim.breakdown(0.0, trimmed)
    assert forces.gravity_force[2] == pytest.approx(
        -sim.boat.total_mass * GRAVITY, rel=1e-12)


def test_a_boat_at_rest_in_trim_stays_put(boat):
    """No crew motion, no oars: the state must barely move."""
    still = catalog.eight(rate=32.0,
                          force_profile=OarForceProfile(max_x=0.0, max_z=0.0))
    coxswain = Coxswain(balance=BalanceController(),
                        heading=HeadingController(enabled=False))
    sim = RowingSimulator(still, coxswain=coxswain)
    heave, pitch = still.trim_attitude(0.0)
    y0 = np.concatenate([[0.0, 0.0, heave], [0.0, pitch, 0.0], np.zeros(6)])

    result = sim.run(duration=2.0, initial_state=y0, dt=0.004)
    assert result.is_finite
    assert np.abs(result.heave - heave).max() < 0.05
    assert np.degrees(np.abs(result.pitch - pitch)).max() < 2.0


# --------------------------------------------------------------------------
# frame consistency
# --------------------------------------------------------------------------
def test_hydrodynamic_forces_rotate_with_the_hull(sim):
    """The same physical state, yawed, must give a yawed force.

    This is the invariant the legacy code broke by adding an
    absolute-frame crew reaction into body-frame equations.
    """
    base = State.create(position=[0.0, 0.0, 0.015], attitude=[0.0, 0.0, 0.0],
                        velocity=[5.0, 0.0, 0.0])
    yaw = 0.7
    rotated = State.create(position=[0.0, 0.0, 0.015],
                           attitude=[0.0, 0.0, yaw],
                           velocity=hull_to_abs(np.array([0.0, 0.0, yaw]))
                           @ np.array([5.0, 0.0, 0.0]))

    plain = sim.breakdown(0.0, base)
    turned = sim.breakdown(0.0, rotated)
    rot = hull_to_abs(np.array([0.0, 0.0, yaw]))

    np.testing.assert_allclose(turned.resistance_force,
                               rot @ plain.resistance_force, atol=1e-6)
    np.testing.assert_allclose(turned.oar_force, rot @ plain.oar_force,
                               atol=1e-6)


@pytest.mark.slow
def test_yawing_the_boat_does_not_change_its_speed_history(boat):
    """Heading is a symmetry of the dynamics; only the track should rotate."""
    coxswain = Coxswain(heading=HeadingController(enabled=False))
    sim = RowingSimulator(boat, coxswain=coxswain)

    straight = sim.run(duration=3.0, surge_speed=5.0, dt=0.005)

    turned_start = sim.initial_state(surge_speed=0.0)
    yaw = 1.1
    turned_start[5] = yaw
    turned_start[6:9] = hull_to_abs(np.array([0.0, 0.0, yaw])) @ \
        np.array([5.0, 0.0, 0.0])
    turned = RowingSimulator(boat, coxswain=coxswain).run(
        duration=3.0, initial_state=turned_start, dt=0.005)

    np.testing.assert_allclose(turned.surge_speed, straight.surge_speed,
                               atol=2e-3)


# --------------------------------------------------------------------------
# all six degrees of freedom are live
# --------------------------------------------------------------------------
def test_every_degree_of_freedom_responds(sim):
    """The legacy 6-DOF model had all moments commented out, so attitude and
    angular velocity stayed identically zero for the whole run."""
    start = sim.initial_state(surge_speed=5.0)
    start[3] = np.radians(2.0)     # roll
    start[4] += np.radians(0.5)    # pitch
    start[5] = np.radians(3.0)     # yaw
    start[7] = 0.2                 # sway velocity

    result = sim.run(duration=2.0, initial_state=start, dt=0.004)
    assert result.is_finite

    for name, channel in (("surge", result.surge), ("sway", result.sway),
                          ("heave", result.heave), ("roll", result.roll),
                          ("pitch", result.pitch), ("yaw", result.yaw)):
        assert np.ptp(channel) > 1e-9, f"{name} never moved"


def test_angular_velocity_is_non_trivial(sim):
    start = sim.initial_state(surge_speed=5.0)
    start[3] = np.radians(2.0)
    result = sim.run(duration=2.0, initial_state=start, dt=0.004)
    assert np.abs(result.omega).max() > 1e-6


def test_the_boat_makes_progress(sim):
    result = sim.run(duration=4.0, surge_speed=5.0, dt=0.005)
    assert result.distance() > 15.0


# --------------------------------------------------------------------------
# stability of the closed loop
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_roll_is_bounded_over_many_strokes(sim):
    result = sim.run(duration=8.0, surge_speed=5.0, dt=0.005)
    assert np.degrees(np.abs(result.roll)).max() < 3.0


@pytest.mark.slow
def test_a_roll_disturbance_is_rejected(sim):
    start = sim.initial_state(surge_speed=5.0)
    start[3] = np.radians(4.0)
    result = sim.run(duration=6.0, initial_state=start, dt=0.005)
    late = result.roll[result.last_cycles(1)]
    assert np.degrees(np.abs(late)).max() < np.radians(4.0) * 57.3 * 0.5


def test_the_boat_capsizes_without_the_crew_balancing(boat):
    """Documents that the instability is real, not a modelling artefact.

    Hull hydrostatics alone give an eight about -1050 N m/rad of righting
    moment against +2580 N m/rad of crew-weight upset, so with the balance
    loop disabled it must go over.
    """
    coxswain = Coxswain(balance=BalanceController(enabled=False))
    sim = RowingSimulator(boat, coxswain=coxswain)
    start = sim.initial_state(surge_speed=5.0)
    start[3] = np.radians(1.0)

    result = sim.run(duration=4.0, initial_state=start, dt=0.004)
    assert np.degrees(np.abs(result.roll)).max() > 20.0


@pytest.mark.slow
def test_heading_is_held_against_the_sweep_yaw_couple(sim):
    """An alternating sweep rig applies a net yaw moment every drive."""
    result = sim.run(duration=10.0, surge_speed=5.0, dt=0.005)
    assert np.degrees(np.abs(result.yaw)).max() < 5.0


@pytest.mark.slow
def test_yaw_wanders_without_steering(boat):
    coxswain = Coxswain(heading=HeadingController(enabled=False))
    sim = RowingSimulator(boat, coxswain=coxswain)
    result = sim.run(duration=10.0, surge_speed=5.0, dt=0.005)
    assert np.degrees(np.abs(result.yaw)).max() > 1.0


@pytest.mark.slow
def test_the_rudder_can_turn_the_boat(boat):
    coxswain = Coxswain(rudder_override=lambda t, s: np.radians(15.0))
    sim = RowingSimulator(boat, coxswain=coxswain)
    result = sim.run(duration=6.0, surge_speed=5.0, dt=0.005)
    assert result.yaw[-1] < np.radians(-2.0), "positive rudder turns starboard"


@pytest.mark.slow
def test_a_heading_change_is_tracked(boat):
    coxswain = Coxswain(heading=HeadingController(target=np.radians(10.0)))
    sim = RowingSimulator(boat, coxswain=coxswain)
    result = sim.run(duration=14.0, surge_speed=5.0, dt=0.006)
    assert np.degrees(result.yaw[-1]) == pytest.approx(10.0, abs=4.0)


# --------------------------------------------------------------------------
# integrator agreement
# --------------------------------------------------------------------------
def test_rk4_and_the_adaptive_solver_agree(sim):
    start = sim.initial_state(surge_speed=5.0)
    fixed = sim.run(duration=2.0, initial_state=start, dt=0.002, method="rk4")
    adaptive = sim.run(duration=2.0, initial_state=start, dt=0.002,
                       method="adaptive")
    n = min(len(fixed.time), len(adaptive.time))
    np.testing.assert_allclose(fixed.surge_speed[:n], adaptive.surge_speed[:n],
                               atol=0.02)


def test_results_converge_as_the_step_shrinks(sim):
    start = sim.initial_state(surge_speed=5.0)
    coarse = sim.run(duration=2.0, initial_state=start, dt=0.008)
    fine = sim.run(duration=2.0, initial_state=start, dt=0.002)
    assert abs(coarse.surge_speed[-1] - fine.surge_speed[-1]) < 0.05


def test_run_rejects_a_malformed_initial_state(sim):
    with pytest.raises(ValueError, match="initial state must have shape"):
        sim.run(duration=1.0, initial_state=np.zeros(10))


def test_run_rejects_an_unknown_method(sim):
    with pytest.raises(ValueError, match="unknown method"):
        sim.run(duration=1.0, method="magic")


# --------------------------------------------------------------------------
# force bookkeeping
# --------------------------------------------------------------------------
def test_breakdown_totals_are_the_sum_of_the_parts(sim, trimmed):
    forces = sim.breakdown(0.4, trimmed)
    expected_force = (forces.crew_force + forces.oar_force
                      + forces.buoyancy_force + forces.gravity_force
                      + forces.resistance_force + forces.appendage_force)
    np.testing.assert_allclose(forces.total_force(), expected_force, atol=1e-9)
    np.testing.assert_allclose(forces.generalised()[:3], expected_force,
                               atol=1e-9)


def test_oar_force_is_zero_at_the_catch_and_finish(sim, trimmed):
    period = sim.boat.timing.period
    drive = sim.boat.timing.drive_duration
    for t in (0.0, drive, 0.9 * period):
        forces = sim.breakdown(t, trimmed)
        assert np.linalg.norm(forces.oar_force) < 1.0


def test_oar_force_drives_the_boat_forward_during_the_drive(sim, trimmed):
    forces = sim.breakdown(0.5 * sim.boat.timing.drive_duration, trimmed)
    assert forces.oar_force[0] > 0.0


def test_crew_reaction_opposes_the_hull_early_in_the_drive(sim, trimmed):
    """Off the catch the crew accelerates bow-ward, shoving the hull astern.

    Only *early* in the drive: with the measured Caplan & Gardner
    kinematics the crew centre of mass reaches peak speed about a third of
    the way through the drive and decelerates from there, so the reaction
    reverses sign well before the finish.  That reversal is the physical
    origin of the boat "running" on the recovery.
    """
    forces = sim.breakdown(0.1 * sim.boat.timing.drive_duration, trimmed)
    assert forces.crew_force[0] < 0.0


def test_crew_reaction_reverses_within_the_drive(sim, trimmed):
    """The sign change happens inside the drive, not at the finish."""
    drive = sim.boat.timing.drive_duration
    early = sim.breakdown(0.1 * drive, trimmed).crew_force[0]
    late = sim.breakdown(0.9 * drive, trimmed).crew_force[0]
    assert early < 0.0 < late


def test_crew_centre_of_mass_moves_bow_ward_over_the_drive(sim):
    """Legs extend, so the seat and the body travel towards the bow."""
    boat = sim.boat
    at_catch = boat.crew_centre_of_mass(0.0)[0]
    at_finish = boat.crew_centre_of_mass(boat.timing.drive_duration)[0]
    assert at_finish > at_catch


def test_crew_reaction_integrates_to_zero_over_a_stroke(sim, trimmed):
    """Periodic crew motion transfers no net momentum to the hull.

    Whatever the crew do inside a stroke, they end where they started, so
    the cycle-average of the reaction must vanish.  Only the oars can
    change the system's momentum.  A non-zero mean here would mean the
    kinematics are not closing, or that a transport term has the wrong
    sign.
    """
    period = sim.boat.timing.period
    times = np.linspace(0.0, period, 400, endpoint=False)
    reaction = np.array(
        [sim.breakdown(t, trimmed).crew_force[0] for t in times]
    )
    scale = np.abs(reaction).max()
    assert abs(reaction.mean()) < 0.02 * scale


def test_resistance_opposes_forward_motion(sim, trimmed):
    forces = sim.breakdown(0.3, trimmed)
    assert forces.resistance_force[0] < 0.0


def test_gyroscopic_term_does_no_work(sim):
    state = State.create(attitude=[0.05, -0.02, 0.3],
                         velocity=[5.0, 0.0, 0.0], omega=[0.1, 0.05, -0.2])
    forces = sim.breakdown(0.3, state)
    assert forces.gyroscopic.dot(state.omega) == pytest.approx(0.0, abs=1e-6)
