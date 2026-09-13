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


def test_the_report_refuses_a_dynamic_oar_profile_at_the_door(tmp_path):
    """Before the river is built, not forty minutes in.

    The prescribed oar block would refuse at the first derivative anyway,
    but only after the expensive stages had run.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_report

    out = tmp_path / "report"
    with pytest.raises(SystemExit) as raised:
        make_report.main(["--physics", "research", "--out", str(out)])
    assert raised.value.code == 2
    assert not out.exists(), "it refused, so it must not have started"


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
    # 0.754-0.816. A real FAIL, recorded in docs/TRACKING.md rather than
    # tuned away. Pinned here so the day it moves -- tier 2's lift, a better
    # pull shape, a forward-dynamic rower -- announces itself in this test
    # instead of being noticed in a table.
    assert level.status == "fail", (level.value, level.detail)
    assert 0.5 < level.value < 0.7, level.value
