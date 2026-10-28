"""The supplementary training-plan generator (research/training/plan_generator.py) on a synthetic athlete."""
import datetime as dt
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("plan_generator",
                                              os.path.join(ROOT, "research", "training", "plan_generator.py"))
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

PROFILE = {
    "id": "synthetic", "sex": "F", "birth_year": 2000, "body_mass_kg": 70.0,
    "tests": [{"date": "2025-12-10", "distance_m": 5000, "time_s": 1390.0},
              {"date": "2026-10-20", "distance_m": 5000, "time_s": 1340.0}],
    "team_week": [{"day": "Tue", "what": "steady", "zone": "Z3", "minutes": 75, "work_minutes": 40},
                  {"day": "Thu", "what": "intervals", "zone": "Z5", "minutes": 75, "work_minutes": 20},
                  {"day": "Fri", "what": "race pieces", "zone": "Z4", "minutes": 75, "work_minutes": 30}],
    "available_days": ["Mon", "Wed", "Sat", "Sun"],
    "own_aerobic_minutes_start": 100, "own_aerobic_minutes_cap": 300, "load_week_growth": 0.03,
    "long_session_cap_min": 90, "fall_key_race_md": [10, 17], "spring_key_race_md": [6, 1],
    "start": "2026-11-02", "target": {"distance_m": 5000, "time_s": 1080.0, "by": "2030-09-25"},
}


@pytest.fixture(scope="module")
def weeks(tmp_path_factory):
    path = tmp_path_factory.mktemp("plan") / "profile.json"
    path.write_text(json.dumps(PROFILE))
    p = pg.load_profile(str(path))
    return p, pg.plan(p, 60)


def _own(week, kind):
    return sum(s["minutes"] for ss in week["sessions"].values() for s in ss if s["kind"] == kind)


def test_every_key_race_week_is_tapered(weeks):
    """The week holding a key race tapers even when the race day falls in the next phase (1 June -> summer base)."""
    p, ws = weeks
    for race in (dt.date(2027, 6, 1), dt.date(2027, 10, 17)):
        wk = next(w for w in ws if dt.date.fromisoformat(w["monday"]) <= race < dt.date.fromisoformat(w["monday"])
                  + dt.timedelta(days=7))
        assert "(taper)" in wk["phase"], (race, wk["phase"])


def test_own_minutes_placed_add_up_to_the_target(weeks):
    """Rounding and per-session caps must not silently drop the week's planned volume."""
    p, ws = weeks
    for w in ws:
        if "base" in w["phase"] and "(" not in w["phase"]:
            assert _own(w, "aerobic") >= 95                 # the start is 100 min; nothing lost to caps
    assert max(_own(w, "aerobic") for w in ws) <= p["own_aerobic_minutes_cap"] + 5


def test_most_of_the_week_is_low_intensity(weeks):
    p, ws = weeks
    for w in ws:
        if "transition" not in w["phase"]:
            assert w["share_low"] >= 0.65, (w["monday"], w["share_low"])


def test_no_heavy_lift_the_day_before_the_interval_session(weeks):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    p, ws = weeks
    for w in ws:
        for d, ss in w["sessions"].items():
            if any(s["kind"] == "strength" for s in ss):
                nxt = days[(days.index(d) + 1) % 7]
                assert not any(s["kind"] == "team" and s["zone"] in ("Z5", "Z6") for s in w["sessions"][nxt])


def test_a_shared_day_lifts_first_and_rows_hours_later(weeks):
    """Same-session concurrent training blunts explosive strength; 3-6 h apart it does not (Schumann 2022,
    Robineau 2016). Any day holding both says so, and the lift is listed first."""
    p, ws = weeks
    shared = 0
    for w in ws:
        for d, ss in w["sessions"].items():
            kinds = [s["kind"] for s in ss if s["kind"] != "team"]
            if "strength" in kinds and "aerobic" in kinds:
                shared += 1
                assert kinds.index("strength") < kinds.index("aerobic")
                assert all("6 h after the lift" in s["what"] for s in ss if s["kind"] == "aerobic")
    assert shared > 0                                     # the synthetic week does produce shared days


def test_deload_every_fourth_week_cuts_volume(weeks):
    p, ws = weeks
    for prev, w in zip(ws, ws[1:]):
        if w["phase"].endswith("(deload)"):
            assert _own(w, "aerobic") < 0.75 * _own(prev, "aerobic")


def test_zones_follow_the_team_convention():
    z = {"time_s": 1337.7}
    w5 = pg.watts(z["time_s"] / 10.0)
    assert w5 == pytest.approx(146.2, abs=0.2)                # Concept2: P = 2.80 / pace^3
    assert pg.FTP_FROM_5K * w5 == pytest.approx(136.7, abs=0.3)
    assert pg.time5k_from_watts(w5) == pytest.approx(1337.7, abs=0.01)
