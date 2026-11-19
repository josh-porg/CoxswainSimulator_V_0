"""The flexible shaft's slow early loading: one candidate at a time, on the like-for-like population (§245).

The population stroke at rung 1 (the default research options, as population_kinematics.py; hands: the scullers' own,
`rbn2023-scull`), with one physical quantity moved to a value a source allows, scored against the data author's
sculling averages at rate 32 (force per 10 % of the drive's time, % of peak). A sensitivity map, not a fit: a candidate
is only taken further if its value is sourced and it also holds on the tank trial (B1) and the athletes' own strokes (C2).

    python research/validation/early_drive_candidates.py --label base
    python research/validation/early_drive_candidates.py --ei-scale 1.5 --label stiff1.5
    python research/validation/early_drive_candidates.py --surface-factor 0.3 --label kappa0.3
    python research/validation/early_drive_candidates.py --lift-scale 1.3 --label lift1.3
    python research/validation/early_drive_candidates.py --shaft rigid --label rigid

Appends one JSON line per run to data/local/validation/early_drive_candidates.jsonl.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"),
                os.path.join(ROOT, "research", "blade")]
import rung2_population as RP                                    # noqa: E402
import population_kinematics as PK                               # noqa: E402
import sculling_averages                                         # noqa: E402
import shaft_coupling_check as SCC                               # noqa: E402
from population_force_shape import population_run               # noqa: E402

Q = np.arange(0.0, 1.01, 0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shaft", default="medium", choices=["medium", "rigid"])
    ap.add_argument("--ei-scale", type=float, default=1.0, help="the scull shaft's EI over [L14]'s medium")
    ap.add_argument("--surface-factor", type=float, default=0.1, help="kappa: the blade's water near the surface")
    ap.add_argument("--lift-scale", type=float, default=1.0, help="the cross-flow lift slope over Helmbold's")
    ap.add_argument("--drive", default="rbn2023-scull", choices=["rbn2023", "rbn2023-scull"])
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    SCC.setup(a.shaft)
    RP.SURFACE_FACTOR, RP.LIFT_SCALE = a.surface_factor, a.lift_scale
    if a.ei_scale != 1.0:
        from coxswain.crew.oar_bending import L14_MEDIUM_EI
        RP.SHAFT_EI = L14_MEDIUM_EI * a.ei_scale
    sim, run = population_run(10, a.drive)
    tf, _dl, R = PK.table(sim, run)
    F = R[:, 3]
    f = 100 * np.interp(Q, tf, F) / F.max()
    ref = sculling_averages.reference(32)["force"]
    rms = float(np.sqrt(np.mean((f - ref) ** 2)))
    early = float(np.sqrt(np.mean((f[1:4] - ref[1:4]) ** 2)))
    out = dict(label=a.label, shaft=a.shaft, ei_scale=a.ei_scale, kappa=a.surface_factor, lift_scale=a.lift_scale,
               drive=a.drive, force_pct=[round(float(x), 1) for x in f], rms=round(rms, 2), early_rms=round(early, 2),
               peak_time=float(100 * tf[np.argmax(F)]), peak_N=float(F.max()), summary=RP.C2.summary(run))
    print("%-12s " % a.label + " ".join("%5.0f" % x for x in f) + " | rms %.1f, 10-30 %% rms %.1f, peak at %.0f %%"
          % (rms, early, out["peak_time"]))
    print("%-12s " % "sculling 32" + " ".join("%5.0f" % x for x in ref))
    with open(os.path.join(ROOT, "data", "local", "validation", "early_drive_candidates.jsonl"), "a") as fh:
        fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
