"""Rung 1 on the two measured scullers with the population's hands and body. Nothing is fitted.

    python research/crew/athletes_population_body.py [--strokes 10]

The population test (SOURCES sec. 174) found the force shape set by the body as much as the
blade: with [K05]'s hands and legs and trunk, the slip law reproduces [K05]'s own force. Here the
same population hands (physical turning points) and body go onto [BR24] and [CR06], each in their
own rig and arc at their own rate, [K05]'s on-water rhythm. Their own measured bodies are one
athlete each and stay diagnostics; the population body is the default this tests.

Measures as research/crew/immersion_study.py (population definitions), plus speed at each
athlete's measured power (cube law) and IVV against theirs.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import handle_rung1 as R                                         # noqa: E402
import immersion_study as I                                      # noqa: E402
import k05_population_test as P                                 # noqa: E402

from coxswain.crew.blade_immersion import EntryImmersion       # noqa: E402

V = R.V

CONFIGS = [
    ("ergometer body, slip", dict(body=False)),
    ("K05 body, slip", dict(body=True)),
    ("K05 body, slip + immersion", dict(body=True, immersion=True)),
    ("K05 body, tier 2 Coppel + strips + immersion", dict(body=True, blade_law="liftdrag",
                                                         blade_coefficients="coppel", strips=True,
                                                         immersion=True)),
    ("K05 timing, own travels, slip", dict(body="own")),
    ("K05 timing, own travels, slip + immersion", dict(body="own", immersion=True)),
]

#: each athlete's own measured seat and back travel, m (measurements, not fits): [CR06] Fig. 3
#: leg and back displacement; [BR24] Ls and Lt (Lt at shoulder height, SOURCES sec. 156)
OWN_TRAVEL = {"cr06": (0.582, 0.408), "br24": (0.597, 0.435)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--athletes", default="br24,cr06")
    ap.add_argument("--match", default="", help="run only configurations whose label contains this")
    a = ap.parse_args()
    print("targets: IVV 49.1 / 49.4%; entry/peak 0.17 [LE26]; peak/mean 1.61-1.90; [H20] catch / finish "
          "slip 7.7-9.7 / 14.1-18.1 deg", flush=True)
    for name in a.athletes.split(","):
        measure, _build = V.ATHLETES[name]
        m = measure()
        print("\n%s measured: %.3f m/s, IVV %.1f%%, %.0f W" % (name, m["speed"], 100 * m["ivv"], m["power"]),
              flush=True)
        for label, opts in CONFIGS:
            if a.match and a.match not in label:
                continue
            boat = R.boat_for(name, m, "on-water", True)
            kw = {k: v for k, v in opts.items() if k not in ("strips", "immersion", "body")}
            if opts.get("strips"):
                kw["blade_span"] = float(boat.rig.seats[0].oarlocks[0].oar.blade_length)
            if opts.get("immersion"):
                kw["blade_immersion"] = EntryImmersion(4.0, 0.0)
            body = OWN_TRAVEL[name] if opts["body"] == "own" else opts["body"]
            r = P.predict(boat, a.strokes, body=body, **kw)
            d = I.descriptors(r)
            at_power = r["speed"] * (m["power"] / r["power"]) ** (1.0 / 3.0)
            print("  %-46s IVV %4.1f%% %4.0f W (%+.1f%% at his/her W) | entry/peak %.2f, peak/mean %.2f, "
                  "catch->90%% %.2f s | slips %.1f / %.1f deg"
                  % (label, 100 * r["ivv"], r["power"], 100 * (at_power / m["speed"] - 1),
                     d["entry_over_peak"], d["peak_over_mean"], d["catch_to_90"], d["catch_slip"],
                     d["finish_slip"]), flush=True)


if __name__ == "__main__":
    main()
