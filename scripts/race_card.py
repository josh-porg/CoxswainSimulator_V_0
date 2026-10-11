r"""Build the race card for a crew: target splits by section, today's wind, and what it costs.

    python scripts/race_card.py --crew four --out data/local/raceday/card_four.json

    python scripts/race_card.py --crew eight --out data/local/raceday/card_eight.json

Sprint 1 of docs/RACEDAY.md. Targets are goals with a range (:mod:`coxswain.raceday.goals`) from the
simulated 2026 field (``data/local/raceday/field_<crew>.npz``, written by
``data/local/crew/feeders/charles_update_2026.py``), spread over the course by
:mod:`coxswain.raceday.card`. The wind is the NWS hourly forecast for
the race start when it is within the forecast range, otherwise the latest observation over the
basin (:func:`coxswain.river.stations.charles_reference`), labelled as such.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coxswain.raceday import card as C          # noqa: E402
from coxswain.raceday import forecast as F      # noqa: E402
from coxswain.river import stations             # noqa: E402

EDT = timezone(timedelta(hours=-4))
FIELDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local", "raceday")
CREWS = {
    # start times: the 2024 schedule's slots for these events (2026 not yet published)
    "four": dict(event="Women's Veteran Fours (60+)", boat="4+", rate=30.0,
                 entries="hocr2026_veteran_fours_entries.csv",
                 race_start=datetime(2026, 10, 17, 10, 32, tzinfo=EDT), start_confirmed=False,
                 goals=[("requalify", "requalify"), ("forecast", "forecast")],
                 # the cox's plan (2026-10-08): Cambridge arch at River Street and Western Avenue, centre at Weeks
                 arches={"River Street": "Cambridge shore", "Western Avenue": "Cambridge shore",
                         "Weeks Footbridge": "centre"}),
    "eight": dict(event="Women's Grand Master Eights (50+)", boat="8+", rate=30.0,
                  entries="hocr2026_grand_master_eights_entries.csv",
                  race_start=datetime(2026, 10, 17, 9, 43, tzinfo=EDT), start_confirmed=False,
                  goals=[("medal", "medal"), ("last_year", "beat 2025"), ("requalify", "requalify"),
                         ("forecast", "forecast")],
                  arches={"River Street": "centre", "Western Avenue": "centre", "Weeks Footbridge": "centre"}),
}


def crew_goals(crew: str):
    """The crew's goals from its simulated field: place goals at even odds with an 80-20% range."""
    from coxswain.raceday import goals as G
    field = G.Field.load(os.path.join(FIELDS, "field_%s.npz" % crew))
    out = []
    for kind, name in CREWS[crew]["goals"]:
        if kind == "medal":
            g = G.goal(field, 3, name)
        elif kind == "requalify":
            g = G.goal(field, field.nq, name)
        elif kind == "last_year":
            if field.last_place < 2:
                continue
            g = G.goal(field, field.last_place - 1, name)
            g["note"] = "place %d or better (%d of %d in 2025)" % (field.last_place - 1, field.last_place,
                                                                  field.n)
        else:
            g = G.forecast(field, name)
        if g.get("place"):
            g["place_text"] = "top %d of %d" % (g["place"], field.n)
        out.append(g)
    ch = G.chances(field, [1] + [g["place"] for g in out if g.get("place")])
    for g in out:
        if g.get("place"):
            g["chance"] = ch[g["place"]]
        else:
            g["median_place"], g["place_80"] = ch["median_place"], ch["place_80"]
    return out


def crew_plan(crew: str, model, boat, refresh: bool = False) -> dict:
    """The crew's planned line (optimised through its planned arches) and the arch alternatives priced
    against it, each with its own optimised line. Cached in ``plan_<crew>.json``: it takes minutes."""
    from coxswain.raceday import plan as P
    path = os.path.join(FIELDS, "plan_%s.json" % crew)
    spec = CREWS[crew]
    if not refresh and os.path.exists(path):
        cached = json.load(open(path))
        if cached.get("arches") == spec["arches"]:
            # the advice is rebuilt from the cached prices, so wording changes need no re-optimisation
            plan_pins, single = cached["arches"], {}
            for r in cached["combinations"]:
                moved = [n for n in plan_pins if r["pins"].get(n) != plan_pins[n]]
                if len(moved) == 1:
                    single[moved[0]] = dict(planned=plan_pins[moved[0]], alternative=r["pins"][moved[0]],
                                            delta_s=r["delta_s"], delta_path_m=r["delta_path_m"], delta_rudder_s=0.0)
            cached["decisions"] = P.key_decisions(model, [dict(single=single)])
            return cached
    reference = float(np.median([g["central"] for g in crew_goals(crew)]))
    k = C.RUDDER_K[spec["boat"]]
    costs = P.arch_costs(model, boat, reference, k, planned=spec["arches"], optimise=True)
    best = P.optimised_time(model, boat, reference, spec["arches"], k)
    out = dict(arches=spec["arches"], reference_s=reference, made=datetime.now(EDT).isoformat(timespec="minutes"),
               route=dict(stations=list(map(float, np.atleast_1d(best["route"].stations))),
                          offsets=list(map(float, np.atleast_1d(best["route"].offsets)))),
               decisions=P.key_decisions(model, [costs]),
               combinations=[dict(pins=r["pins"], delta_s=r["delta_s"], delta_path_m=r["delta_path_m"])
                             for r in costs["combinations"]])
    json.dump(out, open(path, "w"), indent=1)
    return out


OURS = "Sammamish Rowing Association"
FEEDERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local", "crew", "feeders")


def crew_traffic(crew: str, model, card: dict) -> dict:
    """Who we are likely to meet and where (R5), from the simulated field and the bow order."""
    from coxswain.raceday import goals as G, traffic as T
    field = G.Field.load(os.path.join(FIELDS, "field_%s.npz" % crew))
    entries = T.read_entries(os.path.join(FEEDERS, CREWS[crew]["entries"]))
    marks = [("Start", 0.0)] + [(s["end"], s["to_m"]) for s in card["sections"]]
    bridges = {g.name: float(d) for g, d in model.gates if d > C.START_CLEAR_M}
    summary = T.summarise(field, entries, OURS, marks, bridges, length=float(model.course.length)).as_dict()
    for n in summary["neighbours"]:
        at = n["median_at_m"]
        n["near"] = None if at is None else next((f"{a}-{b}" for (a, x), (b, y) in zip(marks[:-1], marks[1:])
                                                   if x <= at < y), marks[-1][0])
    summary["own_bow"] = next(e["bow"] for e in entries if e["club"] == OURS)
    summary["field_size"] = len(entries)
    return summary


def build_boat(crew: str, rate: float):
    if crew == "four":
        import hocr_crew
        boat, _ = hocr_crew.build(rate=rate)
        return boat
    if crew == "eight":
        # no rower-by-rower data for the eight yet: a masters women's eight at squad averages
        from coxswain.boats import catalog
        return catalog.eight(rate=rate, rower_mass=68.0, rower_stature=1.68, coxswain_mass=55.0)
    raise SystemExit("crew %r not built yet" % crew)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crew", default="four", choices=sorted(CREWS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=float, default=None, help="override wind, m/s at 10 m")
    ap.add_argument("--from", dest="direction", type=float, default=None)
    ap.add_argument("--day", default=os.path.join(FIELDS, "day_results.csv"),
                    help="earlier events' times today against last year (event,today,last_year); when the "
                         "file has rows, the measured day factor replaces the wind model's")
    ap.add_argument("--plan", action="store_true",
                    help="re-optimise the planned line and price the arch alternatives (a few minutes; the night "
                         "before); otherwise the cached plan_<crew>.json is used")
    a = ap.parse_args(argv)
    spec = dict(CREWS[a.crew])
    now = datetime.now(EDT)
    times_path = os.path.join(FIELDS, "start_times.json")       # from the heat sheet (scripts/heat_sheet.py)
    if os.path.exists(times_path):
        when = json.load(open(times_path)).get(a.crew)
        if when:
            day = "2026-10-17 " + when if len(when) <= 5 else when
            spec["race_start"] = datetime.strptime(day, "%Y-%m-%d %H:%M").replace(tzinfo=EDT)
            spec["start_confirmed"] = True

    conditions = dict(kind=None)
    if a.speed is not None:
        conditions = dict(kind="what-if", speed=a.speed,
                          direction=float("nan") if a.direction is None else a.direction, at=now.isoformat())
    else:
        try:
            hour = F.at(spec["race_start"])
        except Exception as error:                     # noqa: BLE001
            hour = None
            print("forecast unavailable: %s" % type(error).__name__)
        day_period = None
        if hour is None:
            try:
                day_period = F.daily_at(spec["race_start"])
            except Exception as error:                 # noqa: BLE001
                print("12-hour forecast unavailable: %s" % type(error).__name__)
        if hour is not None:
            conditions = dict(kind="forecast for the start", speed=hour.speed, direction=hour.direction,
                              gust=hour.gust, temperature_c=hour.temperature_c, sky=hour.short,
                              rain_pct=hour.precipitation, at=hour.start)
        elif day_period is not None and day_period.speed == day_period.speed:
            conditions = dict(kind="forecast for race day (12-hour period; hourly not yet in range)",
                              speed=day_period.speed, direction=day_period.direction,
                              temperature_c=day_period.temperature_c, sky=day_period.short,
                              rain_pct=day_period.precipitation, at=day_period.start)
        else:
            readings = []
            for ident in stations.STATIONS:
                try:
                    readings.append(stations.latest(ident))
                except Exception:                      # noqa: BLE001
                    pass
            speed, direction, note = stations.charles_reference(readings)
            conditions = dict(kind="observed now (race-time forecast not yet in range)", speed=float(speed),
                              direction=float(direction), at=readings[0].timestamp if readings else now.isoformat(),
                              note=note)
    wind = (conditions["speed"], conditions["direction"]) if conditions.get("speed") is not None else None

    model = C.CourseModel.build()
    boat = build_boat(a.crew, spec["rate"])
    plan = crew_plan(a.crew, model, boat, refresh=a.plan)
    from coxswain.river.route import Route
    model.route = Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]),
                        name="planned line")
    card = C.race_card(model, boat, crew_goals(a.crew), wind=wind, rudder_k=C.RUDDER_K[spec["boat"]])
    # the planned line and its key decisions: optimised in calm the night before, cached
    card["decisions"] = plan["decisions"]
    from coxswain.raceday import dayfactor as DF
    day = DF.factor(DF.read(a.day)) if os.path.exists(a.day) else None
    card["day"] = day
    if day:
        C.rescale(card, day["factor"], "%d earlier events today against last year" % day["events"])
    lake_path = os.path.join(FIELDS, "lake_equivalents.json")
    if os.path.exists(lake_path):            # Lake Sammamish training splits (scripts/lake_equivalents.py)
        lake = json.load(open(lake_path)).get(a.crew, {})
        for t in card["targets"]:
            if t["name"] in lake:
                t["lake_split"] = lake[t["name"]]["lake_split"]
    card["traffic"] = crew_traffic(a.crew, model, card)
    from coxswain.raceday import plan as P
    card["plan"] = dict(variants=P.variants(card), chosen=None)
    from coxswain.raceday import map as MP
    try:
        card["map"] = MP.course_map(model, model.route, card["sections"], card["decisions"])
    except Exception as error:                     # noqa: BLE001  (a map is a nicety; the card stands without it)
        card["map"] = None
        print("map unavailable: %s" % type(error).__name__)
    from coxswain.raceday import flow as FL
    reference = float(np.median([g["central"] for g in crew_goals(a.crew)]))
    card["flow"] = FL.card_entry(model, boat, reference, C._evaluate)
    card["arch_plan"] = plan["arches"]

    out = dict(generated_at=now.isoformat(timespec="minutes"), crew=a.crew, event=spec["event"],
               boat=spec["boat"], race_start=spec["race_start"].isoformat(),
               start_confirmed=spec["start_confirmed"], conditions=conditions, **card)

    print("%s  race %s%s" % (spec["event"], spec["race_start"].strftime("%a %d %b %H:%M"),
                             "" if spec["start_confirmed"] else " (assumed)"))
    if wind:
        r = card["wind_cost_pct_range"] or [float("nan")] * 2
        print("wind %.1f m/s from %03.0f (%s)  costs this line %+.1f%% (%+.1f to %+.1f%%)"
              % (wind[0], wind[1], conditions["kind"], card["wind_cost_pct"], r[0], r[1]))
    for t in card["targets"]:
        print("  %-16s %s today (%s-%s)  overall %s" % (t["name"], C.fmt(t["today_central"]),
              C.fmt(t["today_fast"]), C.fmt(t["today_slow"]), C.fmt(t["split_overall_central"])))
    for i, s in enumerate(card["sections"]):
        print("  %-16s -> %-16s %5.0f m  %-20s turn %3.1f deg/s %4.1f s  " % (
              s["start"], s["end"], s["path_m"], s["wind_label"], s["peak_turn"], card["targets"][0]["rudder_central"][i])
              + "  ".join("%s @%s" % (C.fmt(t["splits_central"][i]), C.fmt(t["at_landmark_central"][i]))
                          for t in card["targets"]))
    fl = card["flow"]
    if fl and fl.get("ok"):
        print("  river flow %.0f cfs (%.1f m3/s; October median %.1f): %+.1f s on this line"
              % (fl["discharge_cfs"], fl["discharge_m3s"], fl["october_median_m3s"], fl["effect_s"]))
    if card.get("day"):
        d = card["day"]
        print("  day factor %.3f (%.3f-%.3f) from %d earlier events; the wind model said %.3f"
              % (d["factor"], d["low"], d["high"], d["events"], card["model_wind_factor"]))
    t = card["traffic"]
    print("  traffic: bow %d of %d; passes made %.1f, passed by %.1f, clear race %.0f%%"
          % (t["own_bow"], t["field_size"], t["expected_passes_made"], t["expected_passed_by"], 100 * t["p_clear_race"]))
    for d in card["decisions"]:
        print("  %-16s %-12s %s" % (d["bridge"], d["verdict"], d["advice"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)

        def clean(x):
            if isinstance(x, float) and not np.isfinite(x):
                return None
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, list):
                return [clean(v) for v in x]
            if isinstance(x, (np.floating,)):
                return clean(float(x))
            return x
        json.dump(clean(out), open(a.out, "w"), indent=1)
        print("written", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
