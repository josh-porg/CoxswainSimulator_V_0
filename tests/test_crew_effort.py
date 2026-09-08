r"""The crew's skill, and what they have left to give.

Two things arrive together here because they are the same question from
opposite ends: how consistent a crew is stroke to stroke, and how long
they can hold what the coxswain asks for.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew.exertion import (ROWER_CRITICAL_POWER, WPrimeBalance,
                                    mean_handle_power, optimal_pace)
from coxswain.crew.variability import (ELITE, JUNIOR, SKILL_ANCHORS,
                                       for_skill, skill_label)


def test_the_measured_skill_points_are_not_interpolated_away():
    """Elite and junior are measured [K-VAR].  A slider set to either
    must produce the measured number exactly, not whatever a smooth
    curve happens to give at that point -- otherwise the calibration is
    quietly replaced by the interpolation."""
    assert for_skill(0.75).power_sigma == pytest.approx(ELITE.power_sigma)
    assert for_skill(0.75).timing_sigma == pytest.approx(ELITE.timing_sigma)
    assert for_skill(0.35).power_sigma == pytest.approx(JUNIOR.power_sigma)
    assert for_skill(0.35).timing_sigma == pytest.approx(JUNIOR.timing_sigma)


def test_skill_runs_the_right_way_and_ideal_is_perfect():
    """More skill is less scatter, monotonically, and the top of the
    slider is the uniformity every result in this project was computed
    with before variability existed."""
    sigmas = [for_skill(s).power_sigma for s in np.linspace(0.0, 1.0, 21)]
    assert all(b <= a + 1e-12 for a, b in zip(sigmas, sigmas[1:]))
    assert for_skill(1.0).power_sigma == 0.0
    assert for_skill(1.0).timing_sigma == 0.0
    assert skill_label(1.0) == "ideal"
    assert skill_label(0.0) == "novice"


def test_timing_degrades_faster_than_power():
    """Across the measured presets the timing-to-power ratio rises from
    0.35 to 0.59.  A less experienced crew is disproportionately worse
    at going together than at pulling evenly, and tying timing to power
    by one ratio would flatten that."""
    ratio_elite = ELITE.timing_sigma / ELITE.power_sigma
    ratio_junior = JUNIOR.timing_sigma / JUNIOR.power_sigma
    assert ratio_junior > ratio_elite


def test_race_pace_spends_the_reserve_exactly_at_the_line():
    """``P = CP + W'/T`` is the whole point of the two-parameter model:
    row it and you cross the line with nothing left and nothing wasted."""
    reserve = WPrimeBalance()
    duration = 360.0
    power = optimal_pace(duration)
    remaining, t, dt = reserve.capacity, 0.0, 0.5
    while remaining > 0.0 and t < 10 * duration:
        remaining = reserve.step(remaining, power, dt)
        t += dt
    assert t == pytest.approx(duration, rel=0.02)


def test_a_harder_call_cannot_be_held_as_long():
    """What a coxswain is actually trading."""
    reserve = WPrimeBalance()
    base = optimal_pace(360.0)

    def lasts(power):
        remaining, t, dt = reserve.capacity, 0.0, 0.5
        while remaining > 0.0 and t < 4000.0:
            remaining = reserve.step(remaining, power, dt)
            t += dt
        return t

    assert lasts(base * 1.30) < lasts(base * 1.10) < lasts(base)
    # Below critical power there is no reserve to spend at all.
    assert lasts(ROWER_CRITICAL_POWER * 0.95) >= 4000.0


def test_the_stepwise_reserve_agrees_with_the_batch_one():
    """The real-time seam must not be a second, differing model -- the
    same trap the simulator's own step()/run() split has to avoid."""
    reserve = WPrimeBalance()
    power, dt, n = 340.0, 0.25, 400
    batch = reserve.integrate(np.full(n, power), dt)[-1]
    remaining = reserve.capacity
    for _ in range(n):
        remaining = reserve.step(remaining, power, dt)
    assert remaining == pytest.approx(float(batch), rel=1e-9, abs=1e-6)


def test_the_catalog_rows_above_what_anyone_can_hold():
    """Worth pinning, because it is the reason race pace had to be
    computed rather than assumed: the default force scale is a power no
    crew sustains, and nothing noticed until the reserve was tracked."""
    from coxswain.boats import catalog

    boat = catalog.eight(rate=32, rower_mass=75, rower_stature=1.83,
                         coxswain_mass=55)
    watts = mean_handle_power(boat, samples=180)
    assert watts > ROWER_CRITICAL_POWER * 1.3, watts
    # And a crew asked for it empties in a couple of minutes.
    assert WPrimeBalance().endurance(watts) < 180.0


def test_balance_experience_grades_authority_and_learning():
    """More experience is more authority and faster learning, and the
    ideal end lands on the calibrated numbers rather than near them."""
    from coxswain.boats import catalog
    from coxswain.crew.balance import PhaseAuthority
    from coxswain.sim.control import balance_for_experience

    boat = catalog.eight(rate=30, rower_mass=75, rower_stature=1.83,
                         coxswain_mass=55)
    novice = balance_for_experience(boat, 0.0)
    ideal = balance_for_experience(boat, 1.0)

    assert novice.authority.drive < ideal.authority.drive
    assert novice.authority.recovery < ideal.authority.recovery
    assert novice.stiffness < ideal.stiffness
    assert novice.trim.learning_gain < ideal.trim.learning_gain

    # The top of the slider is the module's own calibrated rig: 150 N of
    # spare handle force and two degrees of lean.
    calibrated = PhaseAuthority.from_boat(boat)
    assert ideal.authority.drive == pytest.approx(calibrated.drive)
    assert ideal.authority.recovery == pytest.approx(calibrated.recovery)


def test_a_crew_can_barely_balance_on_the_recovery():
    """The asymmetry that makes a boat something to sit.

    The blades are the only thing to push against, so with them out of
    the water the authority collapses -- and what is left comes mostly
    from leaning the trunk, not from the hands.  A controller with a
    flat moment limit misses this entirely and holds an eight far
    steadier than any crew does.
    """
    from coxswain.boats import catalog
    from coxswain.sim.control import balance_for_experience

    boat = catalog.eight(rate=30, rower_mass=75, rower_stature=1.83,
                         coxswain_mass=55)
    authority = balance_for_experience(boat, 1.0).authority
    assert authority.recovery < authority.drive / 10.0


def test_particles_are_high_only():
    """The least important thing on the screen, and the first to go."""
    from coxswain.viz.menu import quality_settings

    assert quality_settings("high")[3] is True
    assert quality_settings("standard")[3] is False
    assert quality_settings("minimal")[3] is False


def _eight():
    from coxswain.boats import catalog

    return catalog.eight(rate=30, rower_mass=75, rower_stature=1.83,
                         coxswain_mass=55)


def test_an_unset_boat_loses_length_and_a_set_one_does_not():
    """Roll error has to cost seconds, and only when it is real.

    The blade tip is 3.41 m out, so a degree and a third of heel puts it
    in the water.  A blade already under when the catch arrives went in
    early, and the oar swept angle the drive never gets back -- the
    stroke starts short.  Feathering does not help: it makes the skim
    drag nearly free and makes no difference at all to the lost length,
    which is set by the heel alone.
    """
    import numpy as np

    from coxswain.crew.blade_contact import BladeContact

    boat = _eight()
    contact = BladeContact.from_boat(boat)
    rate = abs(float(boat.oar_sweep.rate(0.2 * boat.timing.period,
                                         boat.timing)))
    sweep = boat.oar_sweep.total_sweep

    # Sat: the crew get the whole stroke.
    assert contact.length_fraction(0.0, rate, sweep) == pytest.approx(1.0)
    assert contact.length_fraction(np.radians(1.0), rate,
                                   sweep) == pytest.approx(1.0)
    # Unset: they do not, and it gets worse quickly.
    two = contact.length_fraction(np.radians(2.0), rate, sweep)
    three = contact.length_fraction(np.radians(3.0), rate, sweep)
    assert three < two < 1.0
    assert two < 0.95


def test_blade_contact_reaches_the_drive_not_only_the_drag():
    """The lost length must actually scale the oar force.

    Both halves of this model existed and were tested, and neither ran:
    nothing in the project ever constructed a BladeContact, and the lost
    length had no consumer at all -- it was computed by nobody.  The
    skim drag was at least plumbed in behind a None check.  This asserts
    the drive itself now feels it.
    """
    import numpy as np

    from coxswain.crew.blade_contact import BladeContact
    from coxswain.sim.control import Coxswain
    from coxswain.sim.simulator import RowingSimulator

    boat = _eight()
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    heeled = RowingSimulator(boat, coxswain=cox).initial_state(surge_speed=4.2)
    heeled[3] = np.radians(3.0)

    plain = RowingSimulator(_eight(), coxswain=cox)
    touching = RowingSimulator(_eight(), coxswain=cox,
                               blade_contact=BladeContact.from_boat(boat))
    # Surge acceleration at the same heeled state, on the drive.
    when = 0.2 * boat.timing.period
    a_plain = plain.derivative(when, heeled)[6]
    a_touch = touching.derivative(when, heeled)[6]
    assert a_touch < a_plain, (a_plain, a_touch)


def test_a_feathered_skim_drags_and_unsettles_but_does_not_catch():
    """The distinction that matters on the recovery.

    A blade in the water FEATHERED skims: it presents its edge, costs
    drag, and pushes the boat back level -- but it does not grip, so it
    takes nothing off the drive.  Only a squared blade must catch.

    Three things have to hold for that to be true, and all three are
    separate places in the code:

    * the skim loads are applied on the RECOVERY only;
    * they use the feathered (edge-on, saturating) drag, which is more
      than an order of magnitude below face-on;
    * the oar delivers no force at all on the recovery, so a skim
      cannot shorten a drive that has not started.
    """
    import numpy as np

    from coxswain.crew.blade_contact import BladeContact
    from coxswain.crew.oarlock import oar_force

    boat = _eight()
    contact = BladeContact.from_boat(boat)
    assert contact.feathered is True

    # Nothing to shorten: the recovery carries no oar force.
    period = boat.timing.period
    lock = boat.rig.seats[0].oarlocks[0]
    for fraction in (0.55, 0.75, 0.95):
        assert not boat.timing.is_drive(fraction * period)
        force = oar_force(fraction * period, boat.timing, lock.side,
                          boat.force_profile, boat.oar_sweep)
        assert np.linalg.norm(np.asarray(force)[:3]) == 0.0

    # And feathering is what makes the skim survivable: edge-on drag is
    # far below face-on at the same heel.
    heel = np.radians(3.0)
    feathered_drag, _ = contact.loads(heel, 4.6)
    squared = BladeContact.from_boat(boat, feathered=False)
    squared_drag, _ = squared.loads(heel, 4.6)
    assert abs(squared_drag) > 10.0 * abs(feathered_drag)


def test_the_skim_pushes_the_boat_back_towards_level():
    """The other half of the trade: contact is a powerful stabiliser.

    A dragging blade is expensive, but it is also the strongest righting
    moment available to a crew that has lost the boat -- far more than
    they can produce with hands and lean.  That is why an unset boat
    oscillates rather than simply falling over.
    """
    import numpy as np

    from coxswain.crew.blade_contact import BladeContact

    contact = BladeContact.from_boat(_eight())
    _, moment = contact.loads(np.radians(3.0), 4.6)
    # Heeled one way, pushed the other.
    assert moment < 0.0
    _, other = contact.loads(np.radians(-3.0), 4.6)
    assert other > 0.0


def test_the_square_up_window_bounds_how_early_a_blade_can_catch():
    """A feathered blade cannot be made to catch, however deep it is.

    So the earliest a heeled blade can grip is the moment it squares,
    and the most length a crew can lose is the arc swept between
    squaring and the catch they meant to take.  Unbounded, the model let
    an arbitrarily deep blade catch arbitrarily early: at eight degrees
    of heel it removed the ENTIRE sweep, which no amount of heel can do.
    """
    import numpy as np

    from coxswain.crew.blade_contact import BladeContact

    boat = _eight()
    contact = BladeContact.from_boat(boat)
    rate = abs(float(boat.oar_sweep.rate(0.2 * boat.timing.period,
                                         boat.timing)))
    sweep = boat.oar_sweep.total_sweep

    # The window is a fraction of the recovery, not of the whole cycle.
    window = contact.square_up_window(boat.timing)
    assert 0.0 < window < float(boat.timing.recovery_duration)

    # Shallow: the depth still decides, and the bound changes nothing.
    shallow = np.radians(2.0)
    assert (contact.length_fraction(shallow, rate, sweep, timing=boat.timing)
            == pytest.approx(contact.length_fraction(shallow, rate, sweep)))

    # Deep: the bound takes over, and it does not go to zero.
    for degrees in (5.0, 8.0, 15.0):
        bounded = contact.length_fraction(np.radians(degrees), rate, sweep,
                                          timing=boat.timing)
        loose = contact.length_fraction(np.radians(degrees), rate, sweep)
        assert bounded > loose
        assert bounded > 0.3, (degrees, bounded)

    # However deep, the loss saturates at the window's worth of arc.
    floor = contact.length_fraction(np.radians(30.0), rate, sweep,
                                    timing=boat.timing)
    assert floor == pytest.approx(
        contact.length_fraction(np.radians(15.0), rate, sweep,
                                timing=boat.timing))


def test_wind_costs_nothing_in_a_calm_and_something_in_a_blow():
    """The differential is what makes wiring wind safe.

    The hull's resistance was fitted to measured totals from real boats
    in real air, so the still-air share is already inside it.  Adding the
    whole aerodynamic load on top charges it twice.  The excess is zero
    in a calm -- the calibration untouched -- and only the weather's own
    contribution in a blow.
    """
    import numpy as np

    from coxswain.hydro.wind import AeroModel

    boat = _eight()
    aero = AeroModel.calibrate(boat)
    rotation = np.eye(3)
    moving = np.array([5.0, 0.0, 0.0])

    calm_force, calm_moment = aero.excess_loads(np.zeros(3), moving, rotation)
    assert float(calm_force[0]) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(np.asarray(calm_moment), 0.0, atol=1e-9)

    # A headwind costs, a tailwind pays.
    head, _ = aero.excess_loads(np.array([-5.0, 0.0, 0.0]), moving, rotation)
    tail, _ = aero.excess_loads(np.array([5.0, 0.0, 0.0]), moving, rotation)
    assert float(head[0]) < 0.0 < float(tail[0])
