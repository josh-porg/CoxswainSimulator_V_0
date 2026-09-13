r"""The oar angle as a state: the torque balance, on its own.

Tested in isolation, before anything is wired into the simulator, because
the claim being made is a physical one and it can be checked without the
hull, the crew or the river.

The claim
---------
Substituting [CR06]'s ``F = C2 slip^2`` under a **prescribed** oar angle
does not drive the boat.  Measured on the catalogue eight at rate 28,
integrating ``F_n cos(phi)`` over the drive:

    ========  ========================  ======================
    v (m/s)   prescribed angle (N s)    dynamic angle (N s)
    ========  ========================  ======================
    2.80      +459.3                    +270.1
    4.85      **+0.2**                  +184.9
    6.00      **-171.2**                +152.7
    ========  ========================  ======================

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
    """The default: inertia DERIVED from the crew, not chosen."""
    return OarDynamics.from_boat(eight)


@pytest.fixture(scope="module")
def flat_oar(eight):
    """A constant inertia, for the checks that need a closed form."""
    lock = eight.rig.seats[0].oarlocks[0].oar
    inertia = REFLECTED_CREW * lock.inboard ** 2 + lock.inertia_about_lock
    return OarDynamics.from_boat(eight, inertia=inertia)


def _pull(_t):
    """A constant pull. Negative: it drives ``phi`` down, towards the finish."""
    return -DRIVE_TORQUE


# ---------------------------------------------------------------------------
# the balance itself
# ---------------------------------------------------------------------------
def test_the_default_inertia_is_derived_and_not_a_number(eight):
    """It could have been a fitted constant. It is not.

    The first version of this made ``inertia`` required with no default,
    on the grounds that any default would become a parameter nobody
    remembered fitting. That was the right instinct and the wrong
    remedy: the generalised inertia for this coordinate is computable
    from de Leva masses and the joint chain, both already in the model.
    So the default is the derived profile, and it is a function of the
    oar angle rather than a number.
    """
    import inspect

    parameters = inspect.signature(OarDynamics).parameters
    assert parameters["inertia"].default is inspect.Parameter.empty

    built = OarDynamics.from_boat(eight)
    assert callable(built.inertia), "the default should be a profile"

    with pytest.raises(ValueError):
        OarDynamics.from_boat(eight, inertia=0.0)
    with pytest.raises(ValueError):
        OarDynamics.from_boat(eight, inertia=-5.0)


def test_the_drive_runs_from_bow_ward_to_stern_ward(flat_oar):
    with pytest.raises(ValueError):
        OarDynamics(blade=flat_oar.blade, inboard=flat_oar.inboard,
                    outboard=flat_oar.outboard, inertia=flat_oar.inertia,
                    catch_angle=np.radians(-35.0),
                    finish_angle=np.radians(55.0))


def test_with_no_water_it_is_a_constant_acceleration(flat_oar):
    """The integrator, checked against something with a closed form.

    With the blade force switched off the balance is ``I phi_ddot =
    tau``, so ``phi(t) = phi_0 - (tau/2I) t^2`` exactly. If this does not
    hold, nothing downstream of it means anything.
    """
    import dataclasses as dc

    class _NoWater:
        def normal_force(self, angle, rate, speed, depth=None, cover=None):
            return np.zeros_like(np.asarray(angle, dtype=float))

    dry = dc.replace(flat_oar, blade=_NoWater())
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

    # Measured with the blade force at the blade CENTRE (l = 2.30 m on this
    # eight; it was the tip, 2.56 m).  A shorter lever arm slows the blade,
    # so the prescribed sweep fails EARLIER: impulse +299 N s at 2.80 m/s,
    # crossing zero at 4.36 m/s (it was ~4.85), -69 N s at 4.85, -215 at 6.
    assert impulse_at(2.80) > 250.0            # fine where it was calibrated
    assert impulse_at(4.40) == pytest.approx(0.0, abs=25.0)   # nothing there
    assert impulse_at(4.85) < -25.0            # braking at racing speed
    assert impulse_at(6.00) < -150.0           # and harder above it

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
    """The blade acts at its centre: [CR06]'s ``l``, not the oar's tip."""
    lock = eight.rig.seats[0].oarlocks[0].oar
    assert oar.inboard == pytest.approx(lock.inboard)
    assert oar.outboard == pytest.approx(lock.blade_centre_outboard)
    assert oar.blade.outboard == pytest.approx(lock.blade_centre_outboard)
    assert oar.outboard < lock.outboard
    assert oar.catch_angle == pytest.approx(eight.oar_sweep.catch_angle)
    assert oar.finish_angle == pytest.approx(eight.oar_sweep.finish_angle)


def test_the_blade_centre_is_half_a_blade_in_from_the_tip():
    """Overall length is to the tip (Concept2's convention); [CR06] applies
    the blade force half a blade length in: 2.36 m on their sweep oar,
    1.805 m on their scull."""
    from coxswain.boats.rig import SCULLING_OAR, SWEEP_OAR

    assert SWEEP_OAR.blade_centre_outboard == pytest.approx(3.70 - 1.14 - 0.26)
    assert SCULLING_OAR.blade_centre_outboard == pytest.approx(
        2.88 - 0.88 - 0.215)
    # the shipped trainer's gearing still reads the overall length
    assert SWEEP_OAR.gearing == pytest.approx(1.14 / 3.70)


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


# ---------------------------------------------------------------------------
# the unfitted prediction
# ---------------------------------------------------------------------------
#: [HF09] Table 1: eight elite coxless pairs on the water, drive in ms.
HF09_DRIVE_MS = {20.6: 862, 24.2: 810, 27.7: 779, 31.5: 752}


def _water_drive_fractions():
    return {rate: ms / (60000.0 / rate) for rate, ms in HF09_DRIVE_MS.items()}


def test_the_predicted_drive_fraction_lands_on_the_water(eight, oar):
    """The result that justifies the whole change, and it is not fitted.

    Drive duration is currently a formula of stroke rate, refitted to
    ergometer kinematics, and it misses on-water pairs by 18-28%. Here it
    is not a formula at all: the rower pulls, the blade resists, and how
    long the drive takes is what the torque balance produces.

    Across a 4.4x range of handle power -- 183 to 808 W per rower, which
    brackets anything a crew could actually do -- the predicted drive
    fraction spans 0.319 to 0.406. Measured on the water across
    20.6-31.5 spm it spans 0.296 to 0.395. The ranges essentially
    coincide, and **the ergometer-fitted formula exceeds the measurement
    at every single rate, by 18 to 28%**. (Its range, 0.378-0.465, is
    shifted up rather than disjoint -- its lowest value is below the
    measured highest -- so the comparison that means anything is rate by
    rate, not range against range.)

    Note what is and is not claimed. This is not a point prediction: the
    handle power is an input and a different one moves the answer. What
    it is, is a prediction whose whole plausible range sits where the
    measurements are, from a model with nothing fitted to them -- de Leva
    masses, the rig's own geometry, [CR06]'s blade coefficient, and a
    torque balance.
    """
    from coxswain.crew.stroke import StrokeTiming

    period = eight.timing.period
    fractions = []
    for torque in (250.0, 350.0, 450.0, 600.0, 750.0, 900.0, 1100.0):
        result = oar.drive(lambda t, k=torque: -k, 4.85, dt=0.002)
        fractions.append(result.duration / period)

    predicted_low, predicted_high = min(fractions), max(fractions)
    water = _water_drive_fractions()
    water_low, water_high = min(water.values()), max(water.values())

    # The predicted range overlaps the measured one substantially.
    overlap = (min(predicted_high, water_high)
               - max(predicted_low, water_low))
    assert overlap > 0.5 * (water_high - water_low), (
        (predicted_low, predicted_high), (water_low, water_high))

    # And the formula it replaces exceeds the measurement at EVERY rate.
    # Rate by rate, not range against range: the two ranges overlap at
    # the top, so the looser comparison would be false.
    for rate, measured in water.items():
        predicted = StrokeTiming(rate).drive_fraction
        assert predicted > measured, (rate, predicted, measured)
        assert predicted / measured - 1.0 > 0.15, (rate, predicted, measured)


def test_the_inertia_is_derived_from_the_crew_not_chosen(eight):
    """``sum_i m_i |d x_i / d phi|^2``, from masses already in the model."""
    from coxswain.crew.oardynamics import InertiaProfile, reflected_inertia

    angle, inertia = reflected_inertia(eight)
    assert angle.size > 10
    assert np.all(inertia > 0.0)
    # Heavy early, light late: legs move a lot of mass per radian of oar,
    # arms very little. Ordered catch-to-finish, so decreasing.
    assert inertia[0] > 3.0 * inertia[-1], (inertia[0], inertia[-1])

    profile = InertiaProfile.of(eight)
    assert float(profile(profile.angle.max())) > 50.0
    assert float(profile(profile.angle.min())) < 40.0
    # Clamped outside the range where the reduction is valid.
    assert float(profile(np.radians(80.0))) == pytest.approx(
        float(profile(profile.angle.max())))


def test_a_varying_inertia_carries_its_own_term(eight, oar):
    """``I phi_ddot + (1/2)(dI/dphi) phi_dot^2 = tau``, not ``I phi_ddot = tau``.

    Dropping the second term is a first-order error here, because the
    inertia changes by a factor of several across the drive. Checked by
    asking the model for the acceleration at a point where the oar is
    moving, and comparing against the balance computed by hand.
    """
    angle, rate = np.radians(10.0), -2.0
    torque = -600.0
    moment, slope = oar.inertia_at(angle)
    assert slope != 0.0, "the profile should not be flat here"

    blade = float(oar.blade_torque(angle, rate, 4.85))
    expected = (torque + blade - 0.5 * slope * rate ** 2) / moment
    got = float(oar.acceleration(angle, rate, torque, 4.85))
    assert got == pytest.approx(expected, rel=1e-12)

    # And it differs materially from the naive form.
    naive = (torque + blade) / moment
    assert abs(got - naive) > 0.01 * abs(naive)


def test_a_constant_inertia_has_no_such_term(flat_oar):
    import dataclasses as dc

    flat = dc.replace(flat_oar, inertia=80.0)
    moment, slope = flat.inertia_at(np.radians(10.0))
    assert moment == pytest.approx(80.0)
    assert slope == 0.0
