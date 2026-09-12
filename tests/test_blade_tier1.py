r"""Why the blade model cannot be wired in as an efficiency alone.

``Boat.blade_model`` has existed, tested and switched off, for a long
time.  When it is set the simulator stops using the oar's fixed
``blade_efficiency = 0.78`` and scales the transmitted force by the
instantaneous value from [CR06]'s slip model instead.  That looks like a
cheap first step towards a real blade -- it introduces the speed
dependence the force path is missing -- and it is not one.

**It has no viable operating point.**  Measured on the eight at rate 28,
scale 0.45, where the prescribed model settles at 3.92 m/s:

    ========  ==================  =========
    duration  settled speed       started
    ========  ==================  =========
    70 s      1.88 m/s            3.4 m/s
    70 s      2.17 m/s            6.5 m/s
    250 s     0.63 m/s            3.4 m/s
    250 s     0.64 m/s            6.5 m/s
    ========  ==================  =========

At 70 s it has not converged, which is why the figure recorded in
``SOURCES.md`` sec. 7 -- "5.10 to 4.28 m/s" -- is a transient and not an
equilibrium.  Run it out and the boat collapses to a crawl from either
direction.  Flattening the oar sweep does not rescue it: 0.63 m/s at
flatness 0.00, 0.66 at 0.30, 1.60 at 0.60, against 3.92 for the model it
replaces.

The mechanism
-------------
Efficiency is evaluated at the **instantaneous** hull surge, and the hull
surge swings 56% peak to peak.  The hull's minimum sits near 40% of the
drive -- the crew is accelerating sternward hardest there -- and the oar
force peaks at 40% of the drive as well, so **the force peaks almost
exactly where the hull is slowest**.  (Not at the catch; the trough is
later than that, which is what makes a synthetic cosine surge the wrong
tool for showing it.)  Measured over a settled cycle at a mean 3.904 m/s:

    ===========================================  ======
    force-weighted blade efficiency               value
    ===========================================  ======
    evaluated at the mean speed                   0.628
    evaluated at the instantaneous speed          0.494
    the lumped constant it replaces               0.780
    ===========================================  ======

At the instant of peak oar force the hull is doing 2.76 m/s against a mean
of 3.90, and the efficiency there is 0.40.

So thrust is cut to 63% of what the calibrated model delivers, the boat
slows, the catch dip deepens, the efficiency falls again.  That is
positive feedback, and it runs away because **the wiring supplies only the
destabilising half of the physics**: a real blade whose slip rises also
makes *less force*, and that restoring term lives in the force model, not
in the efficiency factor.  You cannot take one without the other.

The conclusion for the programme
--------------------------------
Tier 1 is not "the blade model, switched on".  It is the slip-quadratic
**force**, which needs the oar angle as a dynamic state -- the phase that
was scheduled after this one.  The two are not separable.  See
``docs/PHYSICS_PROGRAMME.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain import physics


RATE = 28.0
SCALE = 0.45


@pytest.fixture(scope="module")
def eight():
    from coxswain.boats import catalog

    return catalog.eight(rate=RATE)


def _drive_samples(boat, n=600):
    timing = boat.timing
    t = np.linspace(0.0, timing.drive_fraction * timing.period, n)
    return (t,
            np.asarray(boat.oar_sweep(t, timing), dtype=float),
            np.asarray(boat.oar_sweep.rate(t, timing), dtype=float),
            np.asarray(boat.force_profile.magnitude(t, timing), dtype=float))


def test_blade_efficiency_rises_with_boat_speed(eight):
    """The sign that makes it destabilising as a multiplicative gain.

    Thrust scaled by something that grows with speed is positive
    feedback. It is physically correct -- a faster boat does slip less --
    which is why the fix is not to remove it but to add the term that
    opposes it.
    """
    blade = physics.resolve("research").blade_model(eight)
    _t, angle, rate, weight = _drive_samples(eight)

    values = []
    for speed in (2.0, 3.0, 4.0, 5.0, 6.0):
        eff = np.asarray(blade.efficiency(angle, rate, speed), dtype=float)
        values.append(float((eff * weight).sum() / weight.sum()))
    assert values == sorted(values), values
    assert values[-1] > 2.0 * values[0], values


def test_the_force_peaks_where_the_blade_is_already_inefficient(eight):
    """The loading sits in the worse part of the drive.

    Even at a *constant* boat speed, with no surge dip at all, the oar
    force peaks at 40% of the drive where the efficiency is 0.562 --
    below the 0.628 force-weighted average over the whole drive, and well
    below the 0.78 constant it would replace. So switching the blade
    model on costs thrust before any feedback is considered.

    Note what is *not* claimed: the worst **slip** is at the finish, not
    here. Slip is largest where the sweep rate falls to zero and the boat
    carries the blade through the water on its own. Efficiency and slip
    magnitude do not peak in the same place, and an earlier version of
    this test asserted that they did.
    """
    blade = physics.resolve("research").blade_model(eight)
    _t, angle, rate, weight = _drive_samples(eight)
    speed = 3.904

    eff = np.asarray(blade.efficiency(angle, rate, speed), dtype=float)
    weighted = float((eff * weight).sum() / weight.sum())
    at_peak = float(eff[int(np.argmax(weight))])
    lumped = eight.rig.seats[0].oarlocks[0].oar.blade_efficiency

    assert at_peak < weighted < lumped, (at_peak, weighted, lumped)


def test_efficiency_floors_at_both_ends_of_the_drive(eight):
    """Where the runaway gets its teeth.

    The sweep rate passes through zero at the catch and at the finish, so
    the blade is momentarily not moving through the water under its own
    power at all and ``1 - |slip| / |blade speed|`` floors at zero. A
    slower boat spends proportionally more of its drive near those ends,
    which is the other half of why the feedback runs away rather than
    settling somewhere lower.
    """
    blade = physics.resolve("research").blade_model(eight)
    _t, angle, rate, _weight = _drive_samples(eight)
    eff = np.asarray(blade.efficiency(angle, rate, 3.904), dtype=float)

    assert eff[0] == pytest.approx(0.0, abs=1e-9)
    assert eff[-1] == pytest.approx(0.0, abs=1e-9)
    assert eff.max() > 0.7


def test_the_blade_model_is_not_on_by_default(eight):
    """Because of everything above. The default must stay tier 0."""
    assert eight.blade_model is None
    assert physics.resolve(physics.SHIPPED).blade_model(eight) is None


@pytest.mark.slow
def test_the_catch_dip_deepens_the_loss(eight):
    """Efficiency at the instantaneous surge, against at the mean.

    Needs the real trajectory: the hull's speed minimum sits near 40% of
    the drive -- the same place as peak oar force -- and a synthetic
    cosine troughing at the catch puts it in the wrong place. Measured
    over a settled cycle at a mean 3.904 m/s: 0.628 evaluated at the mean
    speed, 0.494 at the instantaneous one, against the 0.780 constant
    being replaced.
    """
    import numpy as _np
    from coxswain.boats import catalog
    from coxswain.sim.control import Coxswain
    from coxswain.sim.simulator import RowingSimulator

    boat = physics.resolve(physics.SHIPPED).apply(catalog.eight(rate=RATE))
    boat.power_scales = _np.full(boat.n_seats, SCALE)
    sim = RowingSimulator(
        boat, coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    result = sim.run(duration=70.0, dt=0.01, surge_speed=3.4)

    times = _np.asarray(result.time)
    surge = _np.asarray(result.velocity)[0]
    tail = times > 70.0 - 4 * boat.timing.period
    times, surge = times[tail], surge[tail]
    mean_speed = float(surge.mean())

    blade = physics.resolve("research").blade_model(boat)
    angle = _np.asarray(boat.oar_sweep(times, boat.timing), dtype=float)
    rate = _np.asarray(boat.oar_sweep.rate(times, boat.timing), dtype=float)
    weight = _np.asarray(boat.force_profile.magnitude(times, boat.timing),
                         dtype=float)

    def weighted(speed):
        eff = _np.asarray(blade.efficiency(angle, rate, speed), dtype=float)
        return float((eff * weight).sum() / weight.sum())

    at_mean, at_instant = weighted(mean_speed), weighted(surge)
    lumped = boat.rig.seats[0].oarlocks[0].oar.blade_efficiency

    assert at_instant < at_mean < lumped, (at_instant, at_mean, lumped)
    # The surge dip costs about a fifth of the efficiency on its own.
    assert at_instant < 0.9 * at_mean, (at_instant, at_mean)
    # And the hull really is near its slowest when the force peaks.
    at_peak_force = float(surge[int(_np.argmax(weight))])
    assert at_peak_force < 0.85 * mean_speed, (at_peak_force, mean_speed)


@pytest.mark.slow
def test_the_efficiency_only_wiring_does_not_hold_the_boat_up():
    """The measurement the docstring reports, at a quarter of the length.

    70 s is not long enough to converge -- that is the trap the earlier
    recorded figure fell into -- so this runs 250 s and checks the boat
    has collapsed well below the speed the prescribed model holds. When
    tier 1 gains the slip FORCE and the oar angle becomes a state, this
    should stop being true, and it is written so that it fails loudly
    when that happens rather than passing quietly.
    """
    from coxswain.boats import catalog
    from coxswain.validation import scorecard

    prescribed = physics.resolve(physics.SHIPPED).apply(
        catalog.eight(rate=RATE))
    baseline = scorecard.settle(prescribed, SCALE, 3.4).speed
    assert baseline > 3.5, baseline

    bladed = physics.resolve("research").apply(catalog.eight(rate=RATE))
    collapsed = scorecard.settle(bladed, SCALE, 3.4, duration=250.0).speed
    assert collapsed < 0.5 * baseline, (baseline, collapsed)
