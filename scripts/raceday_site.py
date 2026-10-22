r"""Build the self-updating race card site (static: GitHub Pages; no server, no scheduled job).

    python scripts/raceday_site.py                      # bundle: cards + wind and flow tables
    .venv-tools/Scripts/python.exe scripts/raceday_encrypt.py   # encrypt it into the site folder

The Claude-hosted page cannot fetch outside data, so it needs a job to push new cards. This site
can: the phone fetches the NWS forecast and the USGS river flow itself and applies them to tables
precomputed here, so the card stays current with nothing running anywhere.

The bundle per crew: the latest card (``data/local/raceday/card_<crew>.json``: goals, sections,
decisions, traffic, map, plan letters), its wind table (``scripts/wind_table.py``) and a flow table
(seconds the line gains or loses against the October median, for a range of discharges). Written to
``data/local/raceday/site_bundle.json`` (private; it is encrypted before it goes anywhere).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import race_card as RC                                   # noqa: E402
from coxswain.raceday import card as C, flow as FL      # noqa: E402
from coxswain.river.route import Route                  # noqa: E402

DISCHARGES = [1.0, 2.0, 2.8, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 45.0]   # m3/s


def flow_table(crew: str, model, option=None) -> dict:
    spec = RC.CREWS[crew]
    plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s%s.json" % (crew, RC.option_suffix(crew, option)))))
    model.route = Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]))
    boat = RC.build_boat(crew, spec["rate"], profile=RC.PROFILE)
    ref = float(plan["reference_s"])
    effect = [FL.effect_s(model, boat, ref, q) for q in DISCHARGES]
    return dict(discharge_m3s=DISCHARGES, effect_fraction=[e / ref for e in effect],
                october_median_m3s=FL.charles.monthly_discharge(10))


def still_air_s(crew: str, model, option=None) -> float:
    """The line's still-air time at our pace, turning included (the wind table's factors are relative to it)."""
    spec = RC.CREWS[crew]
    plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s%s.json" % (crew, RC.option_suffix(crew, option)))))
    model.route = Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]))
    model.flow = FL.charles.ContinuityFlow(model.course, discharge=FL.charles.monthly_discharge(10))
    boat = RC.build_boat(crew, spec["rate"], profile=RC.PROFILE)
    # every line at the crew's own plan's pace, so the lines compare like for like (an alternative plan
    # built later is optimised at that day's reference pace)
    ref = float(json.load(open(os.path.join(RC.FIELDS, "plan_%s.json" % crew)))["reference_s"])
    sec = C._sections(model, C._evaluate(model, boat, ref, None), None, C.RUDDER_K[spec["boat"]])
    return float(sum(s.time_s for s in sec))


def fields_of(crew):
    """The briefing map's water layers and line comparison (data/local/raceday/fields_2026.py), if built."""
    path = os.path.join(RC.FIELDS, "fields_%s.json" % crew)
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    model = C.CourseModel.build()
    bundle = dict(built=datetime.now(RC.EDT).isoformat(timespec="minutes"), crews={})
    for crew in RC.CREWS:
        card = json.load(open(os.path.join(RC.FIELDS, "card_%s.json" % crew)))
        table = json.load(open(os.path.join(RC.FIELDS, "wind_table_%s.json" % crew)))
        table["still_air_s"] = still_air_s(crew, model)
        river = json.load(open(os.path.join(RC.FIELDS, "river_conditions.json")))[crew]
        lv = sorted((float(k), v) for k, v in river["level_s"].items())
        an_path = os.path.join(RC.FIELDS, "analysis_2026.json")
        analysis = json.load(open(an_path))[crew] if os.path.exists(an_path) else None
        bundle["crews"][crew] = dict(card=card, wind=table, flow=flow_table(crew, model),
                                     level=dict(ft=[a for a, _ in lv], s=[b for _, b in lv]),
                                     analysis=analysis and dict(rows=analysis["rows"], flow_m3s=analysis["flow_m3s"]),
                                     fields=fields_of(crew))
        # the other practised arch plan, when its card and wind table have been built (race_card.py --arches)
        alts = []
        for key in RC.ARCH_OPTIONS:
            sfx = RC.option_suffix(crew, key)
            cp, wp = (os.path.join(RC.FIELDS, "%s_%s%s.json" % (n, crew, sfx)) for n in ("card", "wind_table"))
            if sfx and os.path.exists(cp) and os.path.exists(wp):
                wt = json.load(open(wp))
                wt["still_air_s"] = still_air_s(crew, model, key)
                alts.append(dict(key=key, card=json.load(open(cp)), wind=wt, flow=flow_table(crew, model, key)))
        bundle["crews"][crew]["alts"] = alts
        print("%s: card %s, wind table %d winds, flow table %d discharges; other arch plans: %s"
              % (crew, card["generated_at"], len(table["rows"]), len(DISCHARGES), ", ".join(x["key"] for x in alts) or "none"))
    path = os.path.join(RC.FIELDS, "site_bundle.json")
    json.dump(bundle, open(path, "w"), separators=(",", ":"))
    print("written %s (%.0f kB)" % (path, os.path.getsize(path) / 1024))


if __name__ == "__main__":
    main()
