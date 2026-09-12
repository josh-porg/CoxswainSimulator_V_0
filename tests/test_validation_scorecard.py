"""The validation battery: that it is honest, and that it can fail.

A scorecard is only worth having if it can report bad news, so most of
what is checked here is the reporting rather than the physics: that a
target which cannot be run says so instead of being quietly dropped, that
a target which does not apply to a profile is marked ``n/a`` rather than
passed, and that the one measurement the whole programme turns on comes
out as a FAIL against the current model -- which it must, because the
defect is real and the baseline is supposed to record it.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain import physics, validation
from coxswain.validation import scorecard, targets


# ---------------------------------------------------------------------------
# the targets themselves
# ---------------------------------------------------------------------------
def test_every_target_carries_its_provenance():
    for target in validation.TARGETS:
        assert target.source, target.key
        assert target.note, target.key
        assert target.what, target.key


def test_keys_are_unique():
    keys = [t.key for t in validation.TARGETS]
    assert len(keys) == len(set(keys))


def test_a_runnable_target_has_a_band_and_a_pending_one_does_not():
    """A band is a claim. A pending target has no claim to make yet."""
    for target in validation.ready():
        assert target.band is not None, target.key
        assert target.band[0] < target.band[1], target.key
    for target in validation.pending():
        assert target.band is None, target.key


def test_ready_and_pending_partition_the_battery():
    assert set(validation.ready()) | set(validation.pending()) \
        == set(validation.TARGETS)
    assert not set(validation.ready()) & set(validation.pending())


def test_pending_targets_say_why():
    """The shape of what is missing is what a reader needs to judge the rest."""
    for target in validation.pending():
        assert ("PENDING" in target.note or "NOT MODELLED" in target.note), \
            target.key


def test_by_key_names_what_there_is():
    with pytest.raises(KeyError) as caught:
        targets.by_key("no_such_target")
    assert "blade_efficiency_zero_crossing" in str(caught.value)


# ---------------------------------------------------------------------------
# the harness reports honestly
# ---------------------------------------------------------------------------
def _fake_runs(etas_over_v, speeds, swing=0.45, watts=400.0):
    """Settled records with a chosen efficiency-against-speed relation."""
    out = []
    for ratio, speed in zip(etas_over_v, speeds):
        eta = ratio * speed
        out.append(scorecard.Settled(
            scale=0.5, speed=speed, surge_swing=swing,
            crew_power=watts, drag_power=eta * watts))
    return out


def test_a_line_through_the_origin_is_detected():
    """The defect's signature: eta/v constant, zero crossing at the origin."""
    runs = _fake_runs([0.086] * 4, [2.8, 3.9, 5.0, 5.9])
    slope, crossing, spread = scorecard._fit_efficiency(runs)
    assert slope == pytest.approx(0.086, abs=1e-6)
    assert crossing < 1e-6
    assert spread < 1e-6


def test_a_flat_relation_never_crosses_zero():
    """The degenerate case the fit has to survive.

    A perfectly speed-independent efficiency has zero slope, so the
    crossing speed is undefined; reporting it as infinity is right, and
    dividing by the slope would not be.
    """
    runs = _fake_runs([0.4 / v for v in (2.8, 3.9, 5.0, 5.9)],
                      [2.8, 3.9, 5.0, 5.9])
    _slope, crossing, _spread = scorecard._fit_efficiency(runs)
    assert crossing > 1e3


def test_a_blade_with_real_slip_physics_would_not_be_detected():
    """The guard has to be able to pass, or it is not a guard.

    A relation with a genuine intercept -- efficiency high and only
    weakly speed-dependent, which is what a slip model should give --
    must score above the band's floor and show real spread in eta/v.
    """
    speeds = np.array([2.8, 3.9, 5.0, 5.9])
    etas = 0.55 + 0.03 * speeds          # intercept 0.55, gentle slope
    runs = _fake_runs(etas / speeds, speeds)
    _slope, crossing, spread = scorecard._fit_efficiency(runs)
    low, _high = targets.by_key("blade_efficiency_zero_crossing").band
    assert crossing > low
    assert spread > targets.by_key("blade_efficiency_linearity").band[0]


def test_targets_above_the_profiles_tier_are_not_applicable_not_passed():
    """A vacuous target must never be reported as a pass.

    Blade efficiency against Kleshnev's 78.5% means nothing at tier 0:
    with no slip model the efficiency IS the lumped constant, so the
    target would be checking the constant against itself.
    """
    from coxswain.boats import catalog

    boat = physics.resolve(physics.SHIPPED).apply(
        catalog.build("4+", rate=28.0))
    runs = _fake_runs([0.086] * 4, [2.4, 3.3, 4.3, 5.1])
    scores = scorecard.measure(boat, "4+", physics.SHIPPED, runs=runs)
    got = {s.target.key: s for s in scores}
    assert got["blade_efficiency_level"].status == "n/a"
    assert got["blade_efficiency_level"].value is None
    assert "tier" in got["blade_efficiency_level"].detail


def test_pending_targets_appear_in_the_table_rather_than_vanishing():
    from coxswain.boats import catalog

    boat = physics.resolve(physics.SHIPPED).apply(
        catalog.build("4+", rate=28.0))
    runs = _fake_runs([0.086] * 4, [2.4, 3.3, 4.3, 5.1])
    scores = scorecard.measure(boat, "4+", physics.SHIPPED, runs=runs)
    assert len(scores) == len(validation.TARGETS)
    assert {s.status for s in scores} >= {"pending"}
    rendered = validation.table(scores)
    for target in validation.pending():
        assert target.key in rendered, target.key


def test_the_baseline_records_the_defect_as_a_failure():
    """What the whole battery exists to say about the current model."""
    from coxswain.boats import catalog

    boat = physics.resolve(physics.SHIPPED).apply(
        catalog.build("8+", rate=28.0))
    runs = _fake_runs([0.0849, 0.0872, 0.0861, 0.0871],
                      [2.81, 3.92, 5.02, 5.93])
    scores = {s.target.key: s
              for s in scorecard.measure(boat, "8+", physics.SHIPPED,
                                         runs=runs)}
    assert scores["blade_efficiency_zero_crossing"].status == "fail"
    assert scores["blade_efficiency_linearity"].status == "fail"


def test_the_table_renders_every_row():
    from coxswain.boats import catalog

    boat = physics.resolve(physics.SHIPPED).apply(
        catalog.build("4+", rate=28.0))
    runs = _fake_runs([0.083] * 4, [2.4, 3.3, 4.3, 5.1])
    rendered = validation.table(
        scorecard.measure(boat, "4+", physics.SHIPPED, runs=runs))
    lines = rendered.splitlines()
    # header, rule, one row per target
    assert len(lines) == len(validation.TARGETS) + 2
    assert "FAIL" in rendered


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_measured_baseline_is_a_line_through_the_origin():
    """Four real integrations, and the number this programme has to move.

    Slow on purpose: this is the measurement, not a mock of it. If it
    ever stops failing, the blade has gained a velocity term and the
    baseline needs re-recording.
    """
    from coxswain.boats import catalog

    boat = physics.resolve(physics.SHIPPED).apply(
        catalog.build("8+", rate=28.0))
    runs = scorecard.sweep(boat)
    slope, crossing, spread = scorecard._fit_efficiency(runs)

    assert slope > 0.0
    # Through the origin: the fitted line reaches zero efficiency within a
    # few percent of zero speed, where the racing range is 2.8-5.9 m/s.
    assert crossing < 0.15, (slope, crossing)
    # And eta/v barely moves across a 2x speed range.
    assert spread < 0.05, spread


@pytest.mark.slow
def test_the_scorecard_runs_end_to_end():
    rows = validation.run(physics.SHIPPED, boats=("4+",), rate=28.0)
    assert rows
    assert all(row.profile == physics.SHIPPED for row in rows)
    assert validation.table(rows).count("\n") >= len(validation.TARGETS)
