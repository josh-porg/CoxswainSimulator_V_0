"""coxswain.raceday: forecast parsing, wind labels and split arithmetic (no network, no river build)."""
import math

from coxswain.raceday import card as C
from coxswain.raceday import forecast as F


def test_mph_parsing_takes_upper_figure():
    assert math.isclose(F._mph("10 mph"), 10 * F.MPH)
    assert math.isclose(F._mph("5 to 15 mph"), 15 * F.MPH)
    assert math.isnan(F._mph(""))


def test_compass_points():
    assert F.COMPASS["N"] == 0.0 and F.COMPASS["E"] == 90.0 and F.COMPASS["SW"] == 225.0
    assert len(F.COMPASS) == 16


def test_wind_labels_relative_to_bow():
    assert C.label(0.0) == "headwind" and C.label(350.0) == "headwind"
    assert C.label(180.0) == "tailwind"
    assert C.label(90.0) == "cross from starboard"
    assert C.label(270.0) == "cross from port"
    assert C.label(float("nan")) == "calm"


def test_split_and_format():
    assert math.isclose(C.split_of(120.0, 500.0), 120.0)
    assert math.isclose(C.split_of(60.0, 250.0), 120.0)
    assert C.fmt(131.96) == "2:12.0"
    assert C.fmt(1276.0) == "21:16.0"
    assert C.fmt(float("nan")) == "-"
    assert C.fmt(839.96) == "14:00.0"


def test_goal_range_brackets_central_and_matches_chance():
    import numpy as np
    from coxswain.raceday import goals as G
    rng = np.random.default_rng(0)
    field = G.Field(others=rng.normal(1200.0, 30.0, (4000, 20)), day=np.ones(4000), n=21, nq=10,
                    own_mean=1210.0, own_sd=20.0, last_place=-1)
    g = G.goal(field, 3, "medal")
    assert g["fast"] < g["central"] < g["slow"]
    assert abs(field.chance(g["central"], 3) - 0.5) < 0.02
    assert abs(field.chance(g["fast"], 3) - 0.8) < 0.02
    f = G.forecast(field)
    assert math.isclose(f["slow"] - f["central"], 1.2816 * 20.0)


def test_plain_targets_become_goals_without_range():
    g = C._as_goal("requalify", 1276.0)
    assert g["central"] == g["fast"] == g["slow"] == 1276.0 and g["name"] == "requalify"


def test_turn_rate_of_a_circle_and_a_straight():
    import numpy as np
    r = 100.0
    a = np.linspace(0, np.pi, 400)
    arc = np.column_stack([r * np.sin(a), r * np.cos(a)])          # clockwise = to starboard
    om = C.turn_rate(arc, np.full(len(a), 4.0), smooth=0.0)
    mid = slice(50, 350)
    assert np.allclose(om[mid], np.degrees(4.0 / r), rtol=0.03)
    line = np.column_stack([np.linspace(0, 500, 200), np.zeros(200)])
    assert np.allclose(C.turn_rate(line, np.full(200, 4.0)), 0.0, atol=1e-9)


def test_key_decisions_verdicts_from_prices():
    from types import SimpleNamespace
    from coxswain.raceday import plan as P
    gates = [(SimpleNamespace(name=n), d) for n, d in (("River Street", 1444.0), ("Western Avenue", 1781.0),
                                                       ("Weeks Footbridge", 2278.0), ("Larz Anderson", 2711.0),
                                                       ("Eliot Bridge", 3979.0))]
    model = SimpleNamespace(gates=gates)

    def one(delta):
        return dict(planned="centre", alternative="Cambridge shore", delta_s=delta, delta_rudder_s=0.0,
                    delta_path_m=delta * 4)
    costs = [dict(single={"River Street": one(-3.0), "Western Avenue": one(3.0), "Weeks Footbridge": one(14.0)})]
    assert P.key_decisions(model, [dict(single={"River Street": one(-1.0)})])[0]["verdict"] == "either"
    d = {x["bridge"]: x for x in P.key_decisions(model, costs)}
    assert d["River Street"]["verdict"] == "quicker"
    assert d["Western Avenue"]["verdict"] == "cheap"
    assert d["Weeks Footbridge"]["verdict"] == "last resort"
    assert d["Eliot Bridge"]["verdict"] == "no choice" and d["Eliot Bridge"]["alternative"] is None
    assert [x["bridge"] for x in P.key_decisions(model, costs)][0] == "River Street"


def test_encounter_positions():
    import numpy as np
    from coxswain.raceday import traffic as T
    own = np.array([1000.0])
    # a crew starting 15 s behind that is 30 s faster draws level halfway; one 30 s slower never does
    where, sign = T.encounters(own, np.array([[970.0, 1030.0]]), 10, [11, 11], interval=15.0, length=4000.0)
    assert np.isclose(where[0, 0], 2000.0) and sign[0, 0] == -1
    assert np.isnan(where[0, 1]) and sign[0, 1] == 0
    # a slower crew starting 15 s ahead is caught where we have made up the 15 s
    where, sign = T.encounters(own, np.array([[1060.0]]), 10, [9], interval=15.0, length=4000.0)
    assert np.isclose(where[0, 0], 1000.0) and sign[0, 0] == 1


def test_day_factor_from_earlier_events(tmp_path):
    from coxswain.raceday import dayfactor as D
    p = tmp_path / "day.csv"
    p.write_text("# test\nevent,today,last_year\nA,20:24.0,20:00.0\nB,1236,1200\nB,1248,1200\nC,10:12,10:00\n")
    rows = D.read(str(p))
    assert len(rows) == 4 and D.seconds("1:00.5") == 60.5
    f = D.factor(rows)
    assert abs(f["factor"] - 1.02) < 1e-9 and f["events"] == 3
    assert f["low"] <= f["factor"] <= f["high"]


def test_rescale_moves_every_time_by_the_ratio():
    card = dict(wind_factor=1.10, targets=[dict(
        name="x", **{"today_" + k: 1100.0 for k in ("central", "fast", "slow")},
        **{"split_overall_" + k: 110.0 for k in ("central", "fast", "slow")},
        **{n + k: [10.0, 20.0] for n in ("splits_", "at_landmark_", "rudder_") for k in ("central", "fast", "slow")})])
    C.rescale(card, 1.21, "test")
    t = card["targets"][0]
    assert math.isclose(t["today_central"], 1210.0) and math.isclose(t["splits_slow"][1], 22.0)
    assert card["wind_factor"] == 1.21 and card["model_wind_factor"] == 1.10


def test_rowers_deck_check_catches_odds_and_times():
    from coxswain.raceday import decks as D
    ok = D.text_of('<section id="a"><h2>Settle, two moves</h2><p>Crews start 15 seconds apart.</p>'
                   '<aside>Hold the rhythm.</aside></section>')
    assert D.rowers_violations(ok) == []
    for bad in ("a 57% chance", "the odds are good", "we are likely to requalify", "split 2:11.9",
                "medal time 17:40.2", "unlikely to be passed"):
        assert D.rowers_violations(bad), bad


def test_rowers_deck_has_no_probabilities():
    """The rowers' deck kept in the repo (data/local, not committed) passes the check when present."""
    import os
    import pytest
    from coxswain.raceday import decks as D
    base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "local", "raceday", "decks")
    folders = [os.path.join(base, f) for f in ("rowers", "rowers_eight") if os.path.isdir(os.path.join(base, f))]
    if not folders:
        pytest.skip("rowers' decks not present (data/local is not committed)")
    for folder in folders:
        assert D.check_deck(folder) == [], folder


def test_plan_variants_from_a_card():
    from coxswain.raceday import plan as P
    card = dict(decisions=[dict(bridge="River Street", planned="Cambridge shore", alternative="centre", verdict="either"),
                           dict(bridge="Eliot Bridge", planned="centre", alternative=None, verdict="no choice")],
                sections=[dict(start="Powerhouse", end="River Street", push=True),
                          dict(start="Big turn", end="Eliot Bridge", push=False)],
                traffic=dict(neighbours=[dict(bow=22, club="X", starts_s=15.0, p_we_pass=0.0, p_they_pass=0.57,
                                              near="Start-Powerhouse")]))
    v = P.variants(card)
    assert [x["letter"] for x in v] == ["A", "B", "C"]
    assert v[1]["name"] == "Passed early" and "between Start and Powerhouse" in v[1]["when"]
    assert "River Street Cambridge" in v[0]["do"][0] and "Powerhouse to River Street" in v[0]["do"][1]


def test_heat_sheet_rows_parse_and_match():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    import heat_sheet as H
    rows = H.parse("Women's Veteran Fours\n21  Sammamish Rowing Association (Kelly, J.)\n22 | Gentle Giant\n"
                   "23. Newport\nnot a row\n")
    assert rows == [(21, "Sammamish Rowing Association"), (22, "Gentle Giant"), (23, "Newport")]
    clubs = ["Sammamish Rowing Association", "Gentle Giant", "Newport Rowing Club"]
    assert H.match("Sammamish RA", clubs) == "Sammamish Rowing Association" or H.match("Sammamish", clubs)
    assert H.match("Newport", clubs) == "Newport Rowing Club"


def test_daily_forecast_wind_is_the_middle_of_its_range():
    assert math.isclose(F._mid_mph("12 to 15 mph"), 13.5 * F.MPH)
    assert math.isclose(F._mid_mph("10 mph"), 10 * F.MPH)
    assert math.isnan(F._mid_mph(""))


def test_goal_chances_are_monotone_in_place():
    import numpy as np
    from coxswain.raceday import goals as G
    rng = np.random.default_rng(1)
    field = G.Field(others=rng.normal(1200.0, 30.0, (5000, 20)), day=np.ones(5000), n=21, nq=10,
                    own_mean=1200.0, own_sd=15.0, last_place=8)
    ch = G.chances(field, [1, 3, 10])
    assert ch[1] <= ch[3] <= ch[10] and 0.3 < ch[10] < 0.8 and 8 <= ch["median_place"] <= 13


def test_headwind_stretches_group_consecutive_sections():
    from coxswain.raceday import plan as P
    card = dict(sections=[dict(start="Start", end="A", wind_label="headwind"),
                          dict(start="A", end="B", wind_label="headwind"),
                          dict(start="B", end="C", wind_label="cross from starboard"),
                          dict(start="C", end="D", wind_label="headwind")])
    assert P.headwind_stretches(card) == "Start to B, C to D"


def _toy_table():
    """A two-section wind table whose time grows with the square of the wind (any direction)."""
    speeds, dirs = [0.0, 4.0, 8.0, 12.0], [22.5 * i for i in range(16)]
    rows = [dict(speed=v, direction=d, factor=1.0 + 0.002 * v * v, share=[0.5, 0.5], felt=[v, v], rudder=[0.0, 0.0])
            for v in speeds for d in dirs]
    return dict(reference_s=1000.0, speeds=speeds, directions=dirs, rows=rows)


def test_gust_intensity_from_gust_factor():
    from coxswain.raceday import gusts as GU
    assert GU.intensity(6.0, None) == 0.0 and GU.intensity(6.0, 5.0) == 0.0 and GU.intensity(0.0, 9.0) == 0.0
    assert math.isclose(GU.intensity(6.0, 6.0 * (1 + GU.PEAK_FACTOR * 0.2)), 0.2)
    assert GU.intensity(2.0, 40.0) == GU.MAX_INTENSITY


def test_gusts_cost_more_than_steady_wind_and_nothing_without_them():
    from coxswain.raceday import gusts as GU
    T = _toy_table()
    none = GU.allowance(T, 6.0, 315.0, None)
    assert none["ratio"] == 1.0 and none["added_s"] == 0.0
    a = GU.allowance(T, 6.0, 315.0, 11.0)
    sigma = a["sigma_ms"]
    # a quadratic cost averages to f(U) + 0.002 sigma^2, between the table's speeds and on them
    assert a["ratio"] > 1.0 and a["wind_ratio"] > 1.0
    assert math.isclose(a["added_s"], 0.002 * sigma ** 2 * 1000, rel_tol=1e-9)
    b = GU.allowance(T, 4.0, 90.0, 7.0)
    assert math.isclose(b["added_s"], 0.002 * b["sigma_ms"] ** 2 * 1000, rel_tol=1e-9)


def test_gridded_interval_parsing():
    from coxswain.raceday import gusts as GU
    t0, t1 = GU._interval("2026-10-17T14:00:00+00:00/PT3H")
    assert (t1 - t0).total_seconds() == 3 * 3600
    t0, t1 = GU._interval("2026-10-17T14:00:00+00:00/P1DT2H")
    assert (t1 - t0).total_seconds() == 26 * 3600
