r"""Precompute each crew's race card against wind, so a static page can apply any forecast itself.

    python scripts/wind_table.py            # both crews, ~3 min each

For every wind speed (m/s at 10 m) and direction (FROM, 16 points) the planned line is timed as the
race card times it (``coxswain.raceday.card``: the river model, sheltered wind, the measured turning
cost), and the result is stored relative to still air: the whole line's time factor and, per section,
its share of the time, the wind felt at 1.5 m, and the turning seconds. A page holding this table and
the card's still-air goals reproduces the card for any forecast by bilinear interpolation (speed
linearly, direction on the circle), with no server and no model. Output:
``data/local/raceday/wind_table_<crew>.json``.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import race_card as RC                                   # noqa: E402
from coxswain.raceday import card as C                  # noqa: E402
from coxswain.river.route import Route                  # noqa: E402

SPEEDS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
DIRECTIONS = [22.5 * i for i in range(16)]


def table(crew: str, model, option=None) -> dict:
    spec = RC.CREWS[crew]
    plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s%s.json" % (crew, RC.option_suffix(crew, option)))))
    model.route = Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]))
    boat = RC.build_boat(crew, spec["rate"], profile=RC.PROFILE)
    k = C.RUDDER_K[spec["boat"]]
    ref = float(plan["reference_s"])

    def run(wind):
        sec = C._sections(model, C._evaluate(model, boat, ref, wind), wind, k)
        return sec

    calm = run(None)
    calm_total = sum(s.time_s for s in calm)
    rows = []
    for v in SPEEDS:
        for d in DIRECTIONS:
            sec = calm if v == 0.0 else run((v, d))
            total = sum(s.time_s for s in sec)
            rows.append(dict(speed=v, direction=d, factor=total / calm_total,
                             share=[s.time_s / total for s in sec],
                             felt=[0.0 if not np.isfinite(s.wind_felt) else s.wind_felt for s in sec],
                             rudder=[s.rudder_s / total for s in sec]))
    return dict(crew=crew, reference_s=ref, speeds=SPEEDS, directions=DIRECTIONS,
                headings=[s.heading for s in calm], rows=rows)


def main():
    model = C.CourseModel.build()
    for arg in sys.argv[1:] or ["four", "eight"]:          # crew, or crew:arches (e.g. four:centre)
        crew, _, option = arg.partition(":")
        t0 = time.time()
        out = table(crew, model, option or None)
        path = os.path.join(RC.FIELDS, "wind_table_%s%s.json" % (crew, RC.option_suffix(crew, option or None)))
        json.dump(out, open(path, "w"))
        worst = max(out["rows"], key=lambda r: r["factor"])
        print("%s: %d winds in %.0f s; worst %.1f m/s from %03.0f costs %+.1f%% -> %s"
              % (crew, len(out["rows"]), time.time() - t0, worst["speed"], worst["direction"],
                 100 * (worst["factor"] - 1), path))


if __name__ == "__main__":
    main()
