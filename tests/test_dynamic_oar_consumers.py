"""Everything that builds a boat through a profile, against the dynamic oar.

Repointing ``research`` at the dynamic oar changed what "the research
physics" means for every consumer at once.  A consumer that builds a boat
through the profile and then runs the ordinary simulator would silently
simulate the shipped oar under the research label -- the same trap the
collapse test nearly fell into.  Each consumer must either run the dynamic
oar or refuse, loudly and early.
"""

from __future__ import annotations

import os
import sys

import pytest

from coxswain import physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _research_boat():
    from coxswain.boats import catalog

    return physics.resolve("research").apply(catalog.build("4+", rate=30.0))


def _make_report():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_report

    return make_report


def test_the_report_refuses_unbuilt_physics_at_the_door(tmp_path):
    """Before the river is built, not forty minutes in.

    ``learned`` names a tier 2 blade and a trained policy, neither of which
    exists. It used to be ``research`` refused here; research is ported now,
    and the door refuses what is genuinely not there.
    """
    make_report = _make_report()
    out = tmp_path / "report"
    with pytest.raises(SystemExit) as raised:
        make_report.main(["--physics", "learned", "--out", str(out)])
    assert raised.value.code == 2
    assert not out.exists(), "it refused, so it must not have started"


def test_the_report_lets_research_through_the_door(tmp_path, monkeypatch):
    """Research gets past argument checking and into the first stage."""
    make_report = _make_report()

    class _PastTheDoor(Exception):
        pass

    def stop(*_args, **_kwargs):
        raise _PastTheDoor()

    monkeypatch.setattr(make_report, "progress", stop)
    with pytest.raises(_PastTheDoor):
        make_report.main(["--physics", "research",
                          "--out", str(tmp_path / "report")])


def test_the_masters_eight_is_refused_under_the_dynamic_oar():
    """No sourced wattage, so no simulated eight -- refused, not guessed."""
    make_report = _make_report()
    with pytest.raises(ValueError, match="handle power"):
        make_report.reference_eight(profile="research")
    with pytest.raises(ValueError, match="handle power"):
        make_report.quasi_steady_gap(5.2, profile="research")
    # The shipped eight is untouched.
    boat = make_report.reference_eight(profile=physics.SHIPPED)
    assert boat.physics_profile == physics.SHIPPED


def test_the_four_is_driven_at_its_own_erg_watts(monkeypatch):
    """Roster watts directly; the questionable conversion never consulted."""
    import numpy as np

    from coxswain.crew import exertion
    from coxswain.sim.dynamic_oar import (DynamicOarSimulator, _match_key,
                                          simulator_for)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mean_handle_power ran on a dynamic-oar four")

    monkeypatch.setattr(exertion, "mean_handle_power", forbidden)
    make_report = _make_report()
    boat, lineup, target = make_report.hocr_four(profile="research")

    watts = [r.watts for r in lineup.rowers if r.watts]
    assert target == np.mean(watts)
    assert boat.handle_watts == target
    assert float(np.mean(boat.power_scales)) == pytest.approx(1.0, abs=1e-12)
    # The research catch matches torque on a settle (84 s here); this test is
    # about routing, so a planted torque stands in for it and is removed after.
    monkeypatch.setitem(DynamicOarSimulator._MATCHED,
                        _match_key(boat, float(target), "sweep", "slip"), 150.0)
    assert isinstance(simulator_for(boat), DynamicOarSimulator)


@pytest.mark.slow
def test_the_reports_settle_runs_the_dynamic_oar_under_research(monkeypatch):
    """``settled_speed`` on the research four goes through the dynamic run."""
    from coxswain.sim.dynamic_oar import DynamicOarSimulator

    calls = []
    original = DynamicOarSimulator.run

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("duration", args[0] if args else None))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DynamicOarSimulator, "run", spy)
    make_report = _make_report()
    boat, _lineup, _target = make_report.hocr_four(profile="research")
    period = float(boat.timing.period)
    speed = make_report.settled_speed(boat, 3.6, duration=6.0 * period)
    assert calls, "the dynamic oar never ran"
    assert 1.5 < speed < 6.0, speed


def test_the_scorecard_prescribed_settle_refuses_a_dynamic_oar_boat():
    from coxswain.validation import scorecard

    with pytest.raises(ValueError, match="settle_dynamic"):
        scorecard.settle(_research_boat(), 0.5, 4.0)


def test_the_scorecard_routes_a_dynamic_oar_boat_to_stated_watts(monkeypatch):
    """Routed by the profile stamp; the prescribed settle is never called."""
    from coxswain.validation import scorecard

    calls = []

    def fake_dynamic(boat, watts, start, strokes=scorecard.SETTLE_STROKES):
        calls.append(("dynamic", watts, start))
        return scorecard.Settled(scale=watts, speed=start, surge_swing=0.5,
                                 crew_power=100.0, drag_power=60.0)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the prescribed settle ran on a research boat")

    monkeypatch.setattr(scorecard, "settle_dynamic", fake_dynamic)
    monkeypatch.setattr(scorecard, "settle", forbidden)

    runs = scorecard.sweep(_research_boat())
    assert [c[1:] for c in calls] == list(scorecard.DYNAMIC_POINTS)
    assert len(runs) == len(scorecard.DYNAMIC_POINTS)


def test_a_shipped_boat_still_takes_the_prescribed_route(monkeypatch):
    from coxswain.boats import catalog
    from coxswain.validation import scorecard

    calls = []

    def fake_settle(boat, scale, start, **_kwargs):
        calls.append(scale)
        return scorecard.Settled(scale=scale, speed=start, surge_swing=0.5,
                                 crew_power=100.0, drag_power=60.0)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the dynamic settle ran on a shipped boat")

    monkeypatch.setattr(scorecard, "settle", fake_settle)
    monkeypatch.setattr(scorecard, "settle_dynamic", forbidden)

    boat = physics.resolve(physics.SHIPPED).apply(
        catalog.build("4+", rate=30.0))
    scorecard.sweep(boat)
    assert calls == [scale for scale, _ in scorecard.OPERATING_POINTS]


@pytest.mark.slow
def test_research_passes_the_defect_targets_shipped_fails():
    """Phase 2 on the canonical harness, not on a bespoke sweep.

    The battery that recorded the defect, run against ``research``. Both
    defect targets -- where eta's fitted line crosses zero, and how far
    eta/v is from constant -- fail on ``shipped`` (0.020 and 0.027) and
    passed on the eight at 1.462 and 0.380 when this was written.

    The blade-efficiency LEVEL used to be n/a here, because it was read from
    ``boat.blade_model``, which a dynamic-oar boat does not carry. It is now
    measured on the run itself -- force-weighted over the drive, at the
    integrated oar states and the oarlock's instantaneous water speed -- so
    it must come back as a real score, never n/a and never a placeholder.
    It comes back a FAIL, about a quarter below the measured band, and that
    is pinned rather than loosened.
    """
    from coxswain.validation import scorecard

    scores = {s.target.key: s
              for s in scorecard.run("research", boats=("8+",), rate=28.0)}
    for key in ("blade_efficiency_zero_crossing",
                "blade_efficiency_linearity"):
        assert scores[key].status == "pass", (key, scores[key].value,
                                              scores[key].detail)
    assert scores["blade_efficiency_zero_crossing"].value > 1.0
    level = scores["blade_efficiency_level"]
    assert level.value is not None, level.detail
    assert "measured on the run" in level.detail, level.detail
    # Measured 2026-09-13: 0.586 at 5.51 m/s, against Kleshnev's measured
    # 0.754-0.816 (0.559 once the blade force moved to the blade centre). A
    # real FAIL, recorded in docs/TRACKING.md rather than tuned away. Pinned
    # here so the day it moves announces itself in this test instead of being
    # noticed in a table.
    #
    # It moved the same day: the research profile took [CR06]'s catch (the
    # blade enters at zero normal velocity instead of parked and loaded), and
    # at matched power the level is 0.714. Still a FAIL, now below the band
    # by less than half as much; re-pinned to that, not loosened.
    assert level.status == "fail", (level.value, level.detail)
    assert 0.65 < level.value < 0.754, level.value


def test_the_steering_caption_transcribes_nothing():
    """It used to quote one shipped run as prose -- an eight the research page
    does not have, and rms figures the table beside it contradicted. The only
    number it states now is the controller's own look-ahead, read from the
    controller."""
    from coxswain.sim.mpc import PREVIEW_DISTANCE

    caption = _make_report()._steering_caption()
    assert "%.0f m" % PREVIEW_DISTANCE in caption
    for stale in ("29 m", "16 m", "12.88", "0.80 m", "the eight"):
        assert stale not in caption, stale


# ---------------------------------------------------------------------------
# the blade-path figure on the report
# ---------------------------------------------------------------------------
def test_the_shipped_blade_figure_draws_the_schedule(tmp_path):
    """Shipped physics has only a schedule to draw -- no simulation needed."""
    from coxswain.boats import catalog

    make_report = _make_report()
    four = physics.resolve(physics.SHIPPED).apply(
        catalog.build("4+", rate=30.0))
    path = make_report.blade_path_figure(four, str(tmp_path),
                                         physics.SHIPPED, quick=True)
    assert os.path.getsize(path) > 10_000
    with open(path, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_the_blade_figure_says_what_each_physics_draws():
    """A shipped page must not describe the dynamic oar's finding, and a
    research page must not claim the loads are unused."""
    make_report = _make_report()
    shipped = " ".join(make_report._blade_path_words(
        physics.resolve(physics.SHIPPED)))
    research = " ".join(make_report._blade_path_words(
        physics.resolve("research")))
    assert "does not use a blade model" in shipped
    assert "re-anchor" not in shipped
    assert "re-anchors" in research and "dynamic oar" in research
    assert "does not use a blade model" not in research


@pytest.mark.slow
def test_the_research_blade_figure_draws_this_four_from_the_dynamic_oar(
        tmp_path, monkeypatch):
    from coxswain.sim.dynamic_oar import DynamicOarSimulator

    calls = []
    original = DynamicOarSimulator.run_strokes

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DynamicOarSimulator, "run_strokes", spy)
    make_report = _make_report()
    four, _lineup, _watts = make_report.hocr_four(profile="research")
    path = make_report.blade_path_figure(four, str(tmp_path), "research",
                                         quick=True)
    assert len(calls) == 2, "half and full erg power"
    assert os.path.getsize(path) > 10_000



# ---------------------------------------------------------------------------
# scoring tier 2 as an explicit study
# ---------------------------------------------------------------------------
def test_a_sweep_forwards_the_blade_law(monkeypatch):
    from coxswain.validation import scorecard

    seen = []

    def fake(boat, watts, start, strokes=scorecard.SETTLE_STROKES,
             blade_law="slip"):
        seen.append(blade_law)
        return scorecard.Settled(scale=watts, speed=start, surge_swing=0.4,
                                 crew_power=watts * 4, drag_power=watts * 2)

    monkeypatch.setattr(scorecard, "settle_dynamic", fake)
    scorecard.sweep(_research_boat(), blade_law="liftdrag")
    assert seen and all(law == "liftdrag" for law in seen)


def test_a_blade_law_needs_the_dynamic_oar():
    from coxswain.boats import catalog
    from coxswain.validation import scorecard

    shipped = physics.resolve(physics.SHIPPED).apply(
        catalog.build("4+", rate=30.0))
    with pytest.raises(ValueError, match="dynamic oar"):
        scorecard.sweep(shipped, blade_law="liftdrag")


def test_tier_two_scores_are_stamped_with_their_law(monkeypatch):
    """A table must not pass tier 2 numbers off as the profile own."""
    from coxswain.validation import scorecard

    def fake(boat, watts, start, strokes=scorecard.SETTLE_STROKES,
             blade_law="slip"):
        return scorecard.Settled(scale=watts, speed=2.0 + watts / 100.0,
                                 surge_swing=0.4, crew_power=watts * 4,
                                 drag_power=watts * (1.0 + watts / 500.0),
                                 blade_efficiency=0.6)

    monkeypatch.setattr(scorecard, "settle_dynamic", fake)
    boat = _research_boat()
    stamped = scorecard.measure(boat, "4+", "research", blade_law="liftdrag")
    plain = scorecard.measure(boat, "4+", "research")
    assert {s.profile for s in stamped} == {"research+liftdrag"}
    assert {s.profile for s in plain} == {"research"}


@pytest.mark.slow
def test_research_scored_with_the_tier_two_blade():
    """The canonical battery, run on tier 2 as a study.

    The level target rises from the tier 1 measurement (0.586 on the eight)
    and, on these provisional coefficients, still sits below Kleshnev band.
    Both defect targets must still pass: a better blade must not bring back
    the line through the origin.
    """
    from coxswain.validation import scorecard

    scores = {s.target.key: s
              for s in scorecard.run("research", boats=("8+",), rate=28.0,
                                     blade_law="liftdrag")}
    assert all(s.profile == "research+liftdrag" for s in scores.values())
    for key in ("blade_efficiency_zero_crossing",
                "blade_efficiency_linearity"):
        assert scores[key].status == "pass", (key, scores[key].value)
    level = scores["blade_efficiency_level"]
    assert level.value is not None and level.value > 0.6, level.value
