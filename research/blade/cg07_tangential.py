"""What Caplan & Gardner's measured points say about the tangential load: the force on a stalled blade is normal to it.

    .venv/Scripts/python.exe research/blade/cg07_tangential.py

SOURCES §201. [CG07]'s quarter-scale Big Blade in a steady flume (top edge at the surface), lift and drag measured
at nine sweep angles (data/literature/cg07_bigblade_via_coppel2010.csv, replotted by [CO10], +-0.02 / +-0.01).
In a steady flume the sweep angle is the angle of attack, so the measured points resolve onto the blade's own axes:

    C_N = C_L cos a + C_D sin a,      C_T = C_D cos a - C_L sin a      (C_T > 0: along the sliding, drag-like)

Compared with the three quasi-steady forms in the code: tier 2 (Atkinson's sin interpolation through [CG07]'s peaks,
LiftDragBlade.big_blade), and the review's cross-flow law at C_Dc 2.2 (LiftDragBlade.crossflow on a scull blade,
0.43 m along the shaft, 0.083 m^2), and with the on-water ratio [H10] measured mid-drive (§200).

Outputs: printed table; data/local/blade/cg07_tangential.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402

LIT = os.path.join(ROOT, "data", "literature", "cg07_bigblade_via_coppel2010.csv")
OUT = os.path.join(ROOT, "data", "local", "blade")


def main():
    rows = [l.strip().split(",") for l in open(LIT, encoding="utf-8") if not l.startswith("#")][1:]
    pts = [(float(a), float(l), float(d)) for a, l, d in rows if d and 5 < float(a) < 175]
    tier2 = LiftDragBlade.big_blade(outboard=1.8)
    cross = LiftDragBlade.crossflow(outboard=1.8, area=0.083, length=0.43, face_on=2.2)
    out = []
    print("angle   measured C_N  C_T(blade axis)  drag-like C_T/C_N |  tier 2 | cross-flow   (+ = drag-like)")
    for a_deg, cl, cd in pts:
        a = np.radians(a_deg)
        cn, ct = cl * np.cos(a) + cd * np.sin(a), cd * np.cos(a) - cl * np.sin(a)
        # the blade-axis laws are written for 0-90 deg; past 90 the flow arrives on the other edge (mirror)
        am = a if a_deg <= 90 else np.pi - a
        sign = 1.0 if a_deg <= 90 else -1.0
        n2, t2 = tier2.coefficients(am)
        nx, tx = cross.coefficients(am)
        # drag-like = along the flow's component on the blade, whose direction flips at 90 deg
        r = dict(angle=a_deg, cn=cn, ct=ct, ratio=sign * ct / cn, tier2_ratio=t2 / n2, crossflow_ratio=tx / nx)
        out.append(r)
        print("%5.0f   %6.2f      %+6.3f            %+6.3f        |  %+6.3f | %+6.3f" % (
            a_deg, cn, ct, r["ratio"], r["tier2_ratio"], r["crossflow_ratio"]))
    for lo, hi in ((20, 70), (110, 160)):
        g = [r for r in out if lo <= r["angle"] <= hi]
        print("%d-%d deg: measured C_T/C_N %+.3f; cross-flow law %+.3f; tier 2 %+.3f (+ = drag-like)" % (
            lo, hi, np.mean([r["ratio"] for r in g]), np.mean([r["crossflow_ratio"] for r in g]),
            np.mean([r["tier2_ratio"] for r in g])))
    ct = np.array([r["ct"] for r in out])
    print("measured C_T on the blade's own axis: %+.3f +- %.3f at every angle (digitisation ~+-0.02): a fixed bias,"
          " not a drag along the sliding" % (ct.mean(), ct.std()))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "cg07_tangential.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
