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
    # start times: assumed 2024 slots; the published 2026 schedule (RegattaCentral events page, read
    # 12 Oct: eight 09:34, four 10:36, Saturday) is in data/local/raceday/start_times.json and overrides these
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


#: The two arch plans the coxswains have practised (Weeks in the centre in both). Each crew's own plan
#: (CREWS above) is one of them; the other is built with ``--arches`` and cached beside it.
ARCH_OPTIONS = {
    "cambridge": dict(name="Cambridge arches at River Street and Western Avenue",
                      arches={"River Street": "Cambridge shore", "Western Avenue": "Cambridge shore",
                              "Weeks Footbridge": "centre"}),
    "centre": dict(name="Centre arches at River Street and Western Avenue",
                   arches={"River Street": "centre", "Western Avenue": "centre", "Weeks Footbridge": "centre"}),
}


def arch_option_of(crew: str) -> str:
    """The key of the crew's own planned arches in ARCH_OPTIONS."""
    return next(k for k, v in ARCH_OPTIONS.items() if v["arches"] == CREWS[crew]["arches"])


def option_suffix(crew: str, option=None) -> str:
    """'' for the crew's own plan, '_<option>' for the other: plan_<crew><suffix>.json, card_..., wind_table_..."""
    return "" if option in (None, arch_option_of(crew)) else "_" + option


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


def crew_plan(crew: str, model, boat, refresh: bool = False, option=None) -> dict:
    """The crew's planned line (optimised through its planned arches, or ARCH_OPTIONS[option]) and the
    arch alternatives priced against it, each with its own optimised line. Cached in
    ``plan_<crew><suffix>.json``: it takes minutes."""
    from coxswain.raceday import plan as P
    path = os.path.join(FIELDS, "plan_%s%s.json" % (crew, option_suffix(crew, option)))
    spec = CREWS[crew]
    arches = ARCH_OPTIONS[option]["arches"] if option else spec["arches"]
    if not refresh and os.path.exists(path):
        cached = json.load(open(path))
        if cached.get("arches") == arches:
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
    costs = P.arch_costs(model, boat, reference, k, planned=arches, optimise=True)
    # the planned line by the full search (2026-10-22): bridge points free in their arches, a point
    # every 150 m, warm-started from the previous plan if there is one (TRACKING)
    from coxswain.river.route import Route
    starts = ()
    if os.path.exists(path):
        old = json.load(open(path))["route"]
        starts = (Route(np.asarray(old["stations"], float), np.asarray(old["offsets"], float), name="previous plan"),)
    best = P.optimise_line(model, boat, reference, arches, k, starts=starts)
    out = dict(arches=arches, reference_s=reference, made=datetime.now(EDT).isoformat(timespec="minutes"),
               route=dict(stations=list(map(float, np.atleast_1d(best["route"].stations))),
                          offsets=list(map(float, np.atleast_1d(best["route"].offsets)))),
               decisions=P.key_decisions(model, [costs]),
               combinations=[dict(pins=r["pins"], delta_s=r["delta_s"], delta_path_m=r["delta_path_m"])
                             for r in costs["combinations"]])
    json.dump(out, open(path, "w"), indent=1)
    return out


OURS = "Sammamish Rowing Association"
FEEDERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local", "crew", "feeders")


def crew_traffic(crew: str, model, card: dict, start_m: float = 0.0, scale: float = 1.0) -> dict:
    """Who we are likely to meet and where (R5), from the simulated field and the bow order.
    ``start_m``/``scale``: a shortened course starting ``start_m`` along the full one, every crew's time
    scaled by the share of the course that remains."""
    from coxswain.raceday import goals as G, traffic as T
    field = G.Field.load(os.path.join(FIELDS, "field_%s.npz" % crew))
    if scale != 1.0:
        field = G.Field(others=field.others * scale, day=field.day, n=field.n, nq=field.nq,
                        own_mean=field.own_mean * scale, own_sd=field.own_sd * scale, last_place=field.last_place)
    entries = T.read_entries(os.path.join(FEEDERS, CREWS[crew]["entries"]))
    first = card["sections"][0]["start"]
    marks = [(first, 0.0)] + [(s["end"], s["to_m"] - start_m) for s in card["sections"]]
    bridges = {g.name: float(d) - start_m for g, d in model.gates if d > max(C.START_CLEAR_M, start_m + 50.0)}
    summary = T.summarise(field, entries, OURS, marks, bridges,
                          length=float(model.course.length) - start_m).as_dict()
    for n in summary["neighbours"]:
        at = n["median_at_m"]
        n["near"] = None if at is None else next((f"{a}-{b}" for (a, x), (b, y) in zip(marks[:-1], marks[1:])
                                                   if x <= at < y), marks[-1][0])
    summary["own_bow"] = next(e["bow"] for e in entries if e["club"] == OURS)
    summary["field_size"] = len(entries)
    return summary


#: Physics profile the race card's boats are built on. "research" since 2026-10-22: depth moves the wave
#: term only (Sretenskii), not the whole resistance -- the old path overstated depth 3-6x and HOCR 2026 GPS
#: rejected it (PHYSICS_PROGRAMME, "Shallow water on the Charles"). ``None`` restores the old path.
PROFILE = "research"


def build_boat(crew: str, rate: float, profile: str = None):
    """The crew's boat. ``profile`` (e.g. ``"research"``) applies that physics profile; the race-day
    evaluator then prices depth with it (``card.depth_boat``: Sretenskii, wave term only). ``None``
    keeps the boat unstamped and the race card on its published path."""
    if crew == "four":
        import hocr_crew
        boat, _ = hocr_crew.build(rate=rate)
    elif crew == "eight":
        # no rower-by-rower data for the eight yet: a masters women's eight at squad averages
        from coxswain.boats import catalog
        boat = catalog.eight(rate=rate, rower_mass=68.0, rower_stature=1.68, coxswain_mass=55.0)
    else:
        raise SystemExit("crew %r not built yet" % crew)
    if profile is not None:
        from coxswain import physics
        physics.resolve(profile).apply(boat)
    return boat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crew", default="four", choices=sorted(CREWS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=float, default=None, help="override wind, m/s at 10 m")
    ap.add_argument("--from", dest="direction", type=float, default=None)
    ap.add_argument("--day", default=os.path.join(FIELDS, "day_results.csv"),
                    help="earlier events' times today against last year (event,today,last_year); when the "
                         "file has rows, the measured day factor replaces the wind model's")
    ap.add_argument("--arches", choices=sorted(ARCH_OPTIONS), default=None,
                    help="race card for this arch plan instead of the crew's own (its line is optimised and "
                         "cached in plan_<crew>_<arches>.json the first time, a few minutes)")
    ap.add_argument("--conditions-from", dest="conditions_from", default=None,
                    help="rebuild with the wind, gusts and river stored in this card JSON instead of fetching "
                         "today's (re-pricing a past card on new physics); the river is re-priced from the stored "
                         "discharge and basin level")
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
    stored = json.load(open(a.conditions_from)) if a.conditions_from else None
    if stored is not None:
        conditions = dict(stored["conditions"])
        conditions["rebuilt"] = "re-priced %s on the %s physics profile from %s" % (
            now.isoformat(timespec="minutes"), PROFILE, os.path.basename(a.conditions_from))
    elif a.speed is not None:
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
    from coxswain.raceday import gusts as GU
    if conditions.get("kind", "") and conditions["kind"].startswith("forecast") and not (conditions.get("gust") or 0) > 0:
        try:                                           # the hourly forecast leaves gusts empty; the grid has them
            conditions["gust"] = GU.gust_at(spec["race_start"])
        except Exception as error:                     # noqa: BLE001
            print("gust forecast unavailable: %s" % type(error).__name__)

    model = C.CourseModel.build()
    boat = build_boat(a.crew, spec["rate"], profile=PROFILE)
    plan = crew_plan(a.crew, model, boat, refresh=a.plan, option=a.arches)
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
    if stored is not None:
        sf = stored.get("flow") or {}
        card["flow"] = dict(sf, effect_s=FL.effect_s(model, boat, reference, sf["discharge_m3s"]))             if sf.get("ok") else sf
    else:
        card["flow"] = FL.card_entry(model, boat, reference, C._evaluate)
    # Saturday's river in the targets: forecast flow at the start (observed if beyond the forecast) and
    # the basin level, both paid by every crew; a measured day factor already contains them
    river = dict(effect_s=0.0)
    fc = None
    sr = (stored or {}).get("river") or {}
    if stored is not None:
        if sr.get("discharge_m3s") is not None:
            fc = (sr["discharge_m3s"], sr.get("source", "stored"))
    else:
        try:
            fc = FL.forecast_at(spec["race_start"])
        except Exception:                              # noqa: BLE001
            pass
    if fc is not None:
        q, source = fc
        river.update(discharge_m3s=q, discharge_cfs=q / FL.CFS, source=source,
                     flow_s=FL.effect_s(model, boat, reference, q))
    elif card["flow"] and card["flow"].get("ok"):
        river.update(discharge_m3s=card["flow"]["discharge_m3s"], discharge_cfs=card["flow"]["discharge_cfs"],
                     source="USGS observed now", flow_s=card["flow"]["effect_s"])
    lv_path = os.path.join(FIELDS, "river_conditions.json")
    try:
        offset_ft, lv_at = (sr["level_ft"], sr.get("level_at")) if stored is not None else FL.basin_level()
        table = json.load(open(lv_path))[a.crew]["level_s"]
        xs = sorted(float(k) for k in table)
        river.update(level_ft=offset_ft, level_at=lv_at,
                     level_s=float(np.interp(offset_ft, xs, [table[str(x) if str(x) in table else repr(x)] for x in xs])))
    except Exception:                                  # noqa: BLE001
        pass
    river["effect_s"] = river.get("flow_s", 0.0) + river.get("level_s", 0.0)
    river["factor"] = 1.0 + river["effect_s"] / reference
    card["river"] = river
    if not card.get("day"):
        C.scale_targets(card, river["factor"])
    # the shortened course (high wind): Riverside Boat Club to the same finish
    short = C.short_course(card)
    short["traffic"] = crew_traffic(a.crew, model, short, start_m=short["course"]["start_m"],
                                    scale=short["course"]["share"])
    card["short_card"] = short
    card["course"] = dict(name="full", start="Start", start_m=0.0, length_m=sum(x["path_m"] for x in card["sections"]),
                          share=1.0)
    card["arch_plan"] = plan["arches"]
    key = a.arches or arch_option_of(a.crew)
    card["arch_option"] = dict(key=key, name=ARCH_OPTIONS[key]["name"], own=key == arch_option_of(a.crew))
    # gusts: an allowance the pages switch on or off (off by default; the targets above are the steady wind's)
    table_path = os.path.join(FIELDS, "wind_table_%s.json" % a.crew)
    if wind and os.path.exists(table_path):
        card["gust"] = GU.allowance(json.load(open(table_path)), wind[0], wind[1], conditions.get("gust"))
        short["gust"] = card["gust"]

    out = dict(generated_at=now.isoformat(timespec="minutes"), crew=a.crew, event=spec["event"],
               boat=spec["boat"], race_start=spec["race_start"].isoformat(),
               start_confirmed=spec["start_confirmed"], conditions=conditions, **card)

    print("%s  race %s%s" % (spec["event"], spec["race_start"].strftime("%a %d %b %H:%M"),
                             "" if spec["start_confirmed"] else " (assumed)"))
    if wind:
        r = card["wind_cost_pct_range"] or [float("nan")] * 2
        print("wind %.1f m/s from %03.0f (%s)  costs this line %+.1f%% (%+.1f to %+.1f%%)"
              % (wind[0], wind[1], conditions["kind"], card["wind_cost_pct"], r[0], r[1]))
    gu = card.get("gust")
    if gu and gu["gust_ms"]:
        print("gusts to %.1f m/s (intensity %.2f): counting them adds %+.1f s at reference pace (x%.4f; wind cost x%.2f)"
              % (gu["gust_ms"], gu["intensity"], gu["added_s"], gu["ratio"], gu["wind_ratio"]))
    for t in card["targets"]:
        print("  %-16s %s today (%s-%s)  overall %s" % (t["name"], C.fmt(t["today_central"]),
              C.fmt(t["today_fast"]), C.fmt(t["today_slow"]), C.fmt(t["split_overall_central"])))
    for i, s in enumerate(card["sections"]):
        print("  %-16s -> %-16s %5.0f m  %-20s turn %3.1f deg/s %4.1f s  " % (
              s["start"], s["end"], s["path_m"], s["wind_label"], s["peak_turn"], card["targets"][0]["rudder_central"][i])
              + "  ".join("%s @%s" % (C.fmt(t["splits_central"][i]), C.fmt(t["at_landmark_central"][i]))
                          for t in card["targets"]))
    rv = card.get("river") or {}
    if rv:
        print("  river on race day: %s %.0f cfs -> %+.1f s; basin %+.2f ft -> %+.1f s; targets x%.4f"
              % (rv.get("source", "-"), rv.get("discharge_cfs", float("nan")), rv.get("flow_s", 0.0),
                 rv.get("level_ft", float("nan")), rv.get("level_s", 0.0), rv["factor"]))
    fl = card["flow"]
    if fl and fl.get("ok"):
        print("  river flow %.0f cfs (%.1f m3/s; October median %.1f): %+.1f s on this line"
              % (fl["discharge_cfs"], fl["discharge_m3s"], fl["october_median_m3s"], fl["effect_s"]))
    if card.get("day"):
        d = card["day"]
        print("  day factor %.3f (%.3f-%.3f) from %d earlier events; the wind model said %.3f"
              % (d["factor"], d["low"], d["high"], d["events"], card["model_wind_factor"]))
    sc = card["short_card"]
    print("  SHORT COURSE from %s (%.0f m): %s" % (sc["course"]["start"], sc["course"]["length_m"],
          "; ".join("%s %s" % (t["name"], C.fmt(t["today_central"])) for t in sc["targets"])))
    print("    traffic: passes made %.1f, passed by %.1f, clear race %.0f%%" % (
          sc["traffic"]["expected_passes_made"], sc["traffic"]["expected_passed_by"], 100 * sc["traffic"]["p_clear_race"]))
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
