r"""The oar angle as a state: the torque balance, on its own.

Tested in isolation, before anything is wired into the simulator, because
the claim being made is a physical one and it can be checked without the
hull, the crew or the river.

The claim
---------
Substituting [CR06]'s ``F = C2 slip^2`` under a **prescribed** oar angle
does not drive the boat.  Measured on the catalogue eight at rate 28,
integrating ``F_n cos(phi)`` over the drive:

    ========  ==========================  ==========================
    v (m/s)   prescribed angle (N s)      dynamic angle (N s)
    ========  ==========================  ==========================
    2.80      +459.3                      +256.1
    4.85      **+0.2**                    +162.5
    6.00      **-171.2**                  +126.7
    ========  ==========================  ==========================

At racing speed the prescribed model delivers **nothing**, and above it
the blade brakes the boat, because ``v cos(phi)`` overwhelms
``l phi_dot`` through mid-drive, slip changes sign, and the oar is being
dragged through the water on a timetable regardless of whether there is
anything to push against.

Let the angle respond and the impulse is positive at every speed and
**falls as the boat speeds up** -- which is the restoring term that the
efficiency-only wiring of ``tests/test_blade_tier1.py`` lacked, and the
reason that wiring ran away.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew.oardynamics import OarDynamics


REFLECTED_CREW = 60.0        # kg at the handle; see the module docstring
DRIVE_TORQUE = 900.0         # N m about the pin


@pytest.fixture(scope="module")
def eight():
    from coxswain.boats import catalog

    return catalog.eight(rate=28.0)


@pytest.fixture(scope="module")
def oar(eight):
    lock = eight.rig.seats[0].oarlocks[0].oar
    inertia = REFLECTED_CREW * lock.inboard ** 2 + lock.inertia_about_lock
    return OarDynamics.from_boat(eight, inertia=inertia)


def _pull(_t):
    """A constant pull. Negative: it drives ``phi`` down, towards the finish."""
    return -DRIVE_TORQUE


# ---------------------------------------------------------------------------
# the balance itself
# ---------------------------------------------------------------------------
def test_inertia_is_required_and_must_be_positive(eight):
    """No default, deliberately.

    The oar's own inertia is twenty times too small and the rest is the
    rower's body. A default here would become a fitted parameter nobody
    remembered fitting, which is the class of thing this whole programme
    exists to remove.
    """
    import inspect

    parameters = inspect.signature(OarDynamics).parameters
    assert parameters["inertia"].default is inspect.Parameter.empty
    with pytest.raises(ValueError):
        OarDynamics.from_boat(eight, inertia=0.0)
    with pytest.raises(ValueError):
        OarDynamics.from_boat(eight, inertia=-5.0)


def test_the_drive_runs_from_bow_ward_to_stern_ward(oar):
    with pytest.raises(ValueError):
        OarDynamics(blade=oar.blade, inboard=oar.inboard,
                    outboard=oar.outboard, inertia=oar.inertia,
                    catch_angle=np.radians(-35.0),
                    finish_angle=np.radians(55.0))


def test_with_no_water_it_is_a_constant_acceleration(oar):
    """The integrator, checked against something with a closed form.

    With the blade force switched off the balance is ``I phi_ddot =
    tau``, so ``phi(t) = phi_0 - (tau/2I) t^2`` exactly. If this does not
    hold, nothing downstream of it means anything.
    """
    import dataclasses as dc

    class _NoWater:
        def normal_force(self, angle, rate, speed, depth=None, cover=None):
            return np.zeros_like(np.asarray(angle, dtype=float))

    dry = dc.replace(oar, blade=_NoWater())
    result = dry.drive(_pull, boat_speed=0.0, dt=0.001)

    alpha = DRIVE_TORQUE / dry.inertia
    expected = dry.catch_angle - 0.5 * alpha * result.time ** 2
    assert np.allclose(result.angle, expected, atol=1e-6)
    assert result.rate[-1] == pytest.approx(-alpha * result.duration,
                                            rel=1e-6)


def test_the_blade_torque_opposes_the_sweep(oar):
    """It is a resistance, not a second engine.

    ``F_n`` carries the sign of ``-slip`` and slip carries ``l phi_dot``,
    so a blade moving sternward under its own sweep is always pushed back.
    """
    angle = np.radians(10.0)
    for rate in (-3.0, -1.5, -0.5):
        torque = float(oar.blade_torque(angle, rate, boat_speed=0.0))
        assert torque > 0.0, rate      # opposes a negative phi_dot


def test_slip_matches_the_blade_models_own_definition(oar):
    """One definition, two places, and they must not drift apart."""
    angle = np.radians(np.linspace(-35.0, 55.0, 25))
    rate = np.full_like(angle, -2.0)
    mine = oar.slip(angle, rate, 4.85)
    theirs = oar.blade.slip_velocity(angle, rate, 4.85)
    assert np.allclose(mine, theirs)


# ---------------------------------------------------------------------------
# the physics that justifies the change
# ---------------------------------------------------------------------------
def test_the_drive_completes_and_sweeps_the_rigged_arc(oar):
    result = oar.drive(_pull, boat_speed=4.85, dt=0.002)
    assert result.finished
    assert np.degrees(result.swept) == pytest.approx(90.0, abs=1.0)
    assert 0.3 < result.duration < 2.0, result.duration


def test_propulsive_impulse_is_positive_at_racing_speed(oar):
    """The gate. A prescribed angle delivers +0.2 N s here."""
    result = oar.drive(_pull, boat_speed=4.85, dt=0.002)
    assert result.propulsive_impulse > 50.0, result.propulsive_impulse


def test_a_prescribed_angle_delivers_nothing_at_racing_speed(eight, oar):
    """The comparison the whole change rests on, measured not asserted.

    Same force model, same boat, same speed -- only the angle differs in
    where it comes from.
    """
    timing = eight.timing
    t = np.linspace(0.0, timing.drive_fraction * timing.period, 2000)
    angle = np.asarray(eight.oar_sweep(t, timing), dtype=float)
    rate = np.asarray(eight.oar_sweep.rate(t, timing), dtype=float)

    def impulse_at(speed):
        force = np.asarray(oar.blade.normal_force(angle, rate, speed),
                           dtype=float)
        return float(np.trapezoid(force * np.cos(angle), t))

    assert impulse_at(2.80) > 300.0            # fine where it was calibrated
    assert abs(impulse_at(4.85)) < 25.0        # nothing at racing speed
    assert impulse_at(6.00) < -50.0            # and braking above it

    # The dynamic angle, by contrast, drives the boat at all three.
    for speed in (2.80, 4.85, 6.00):
        assert oar.drive(_pull, boat_speed=speed,
                         dt=0.002).propulsive_impulse > 50.0, speed


def test_thrust_falls_as_the_boat_speeds_up(oar):
    """The restoring term, and the reason this is not the earlier failure.

    Wiring the blade in as an efficiency alone gave a gain that RISES
    with speed and nothing to oppose it, so the boat ran away downwards.
    The force model supplies the other half: a faster hull means less
    slip, less blade force, less thrust.
    """
    impulses = [oar.drive(_pull, boat_speed=v, dt=0.002).propulsive_impulse
                for v in (2.8, 4.0, 4.85, 6.0)]
    assert impulses == sorted(impulses, reverse=True), impulses
    assert impulses[0] > 1.5 * impulses[-1], impulses


def test_the_drive_duration_is_an_output(oar):
    """And it shortens as the boat speeds up, for the same pull.

    Currently the drive duration is a formula of stroke rate alone. Here
    nobody supplies it: the rower pulls, the blade resists, and how long
    the drive takes is whatever that produces. That is what makes it
    testable against [HF09]'s on-water measurements.
    """
    durations = [oar.drive(_pull, boat_speed=v, dt=0.002).duration
                 for v in (2.8, 4.0, 4.85, 6.0)]
    assert durations == sorted(durations, reverse=True), durations
    assert durations[0] > 1.2 * durations[-1], durations


def test_a_harder_pull_shortens_the_drive(oar):
    """The other input that sets it, and the obvious direction."""
    soft = oar.drive(lambda t: -600.0, 4.85, dt=0.002)
    hard = oar.drive(lambda t: -1400.0, 4.85, dt=0.002)
    assert hard.duration < soft.duration
    assert hard.propulsive_impulse > soft.propulsive_impulse


def test_boat_speed_may_be_a_function_of_time(oar):
    """So the same routine serves a fixed-speed study and a coupled run."""
    steady = oar.drive(_pull, boat_speed=4.85, dt=0.002)
    varying = oar.drive(_pull, boat_speed=lambda t: 4.85, dt=0.002)
    assert varying.duration == pytest.approx(steady.duration, rel=1e-9)
    assert varying.propulsive_impulse == pytest.approx(
        steady.propulsive_impulse, rel=1e-9)


def test_the_integrator_has_converged_at_the_step_it_uses(oar):
    """Halving the step must not move the answer."""
    coarse = oar.drive(_pull, 4.85, dt=0.002)
    fine = oar.drive(_pull, 4.85, dt=0.0005)
    assert fine.duration == pytest.approx(coarse.duration, rel=5e-3)
    assert fine.propulsive_impulse == pytest.approx(
        coarse.propulsive_impulse, rel=5e-3)


def test_geometry_comes_from_the_boat(eight, oar):
    lock = eight.rig.seats[0].oarlocks[0].oar
    assert oar.inboard == pytest.approx(lock.inboard)
    assert oar.outboard == pytest.approx(lock.outboard)
    assert oar.blade.outboard == pytest.approx(lock.outboard)
    assert oar.catch_angle == pytest.approx(eight.oar_sweep.catch_angle)
    assert oar.finish_angle == pytest.approx(eight.oar_sweep.finish_angle)


def test_a_pull_too_weak_to_finish_says_so(oar):
    """Rather than reporting a drive duration that is really a timeout.

    At a high boat speed and a feeble pull the blade can hold the oar
    short of the finish angle. ``finished`` is what distinguishes that
    from a drive that completed, and ``duration`` must not be read
    without it.
    """
    result = oar.drive(lambda t: -5.0, boat_speed=6.0, dt=0.002,
                       max_time=0.5)
    assert not result.finished
    assert result.duration == pytest.approx(0.5, abs=0.01)
    assert result.angle[-1] > oar.finish_angle


def test_at_the_catch_the_water_helps_the_rower(oar):
    """The blade is not purely a brake, and the sign is worth pinning.

    At the catch the sweep rate is zero, so the only relative motion is
    the boat carrying the blade through the water. The water resists
    *that*, which pushes the blade sternward -- the same direction the
    rower is going. So the blade torque at the catch is **negative**: it
    drives the oar towards the finish.

    This is the anchored part of the drive, where slip is positive, and
    it is why the blade-path figure shows the blade running backwards
    through the water only later in the stroke. An earlier version of
    this test assumed the blade must always oppose the rower and asserted
    the opposite sign.
    """
    torque = float(oar.blade_torque(oar.catch_angle, 0.0, boat_speed=4.85))
    slip = float(oar.slip(oar.catch_angle, 0.0, 4.85))
    assert slip > 0.0, slip           # anchored: the boat is passing the blade
    assert torque < 0.0, torque       # and that sweeps the oar aft

    # With the boat stopped there is no such help, and the blade can only
    # resist -- which is the case the companion test covers.
    assert float(oar.blade_torque(oar.catch_angle, 0.0, 0.0)) == \
        pytest.approx(0.0, abs=1e-9)
