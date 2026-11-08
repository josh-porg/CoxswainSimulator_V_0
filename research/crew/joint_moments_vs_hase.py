"""The research body's hip and knee moments against Hase et al. (2002)'s musculoskeletal model (SOURCES §223).

[HKY02] (JSME Int. J. C 45(4) 1073-1081; data/local/literature/hase2002_rowing_msk/): 5 experienced men on a Concept2
at 26 spm, measured handle and foot forces, a 3-D whole-body musculoskeletal model. Read off its Figs. 2-3 (right
side): handle tension peak ~850 N; hip extension ~240 N.m; knee extension ~40 N.m early in the drive, then a flexor
moment ~60 N.m in its second half. Doubled for both sides, as the model's moments are totals (muscle_capacity.py: the
hip from the upper body's free body, the knee from both legs' stretcher force).

The rowers, boats and forces differ (ergometer men vs on-water athletes), so each moment is divided by the stroke's
own peak handle force: a lever arm, m, comparable across them. Reads muscle_capacity.py's rung-2 output.

    python research/crew/joint_moments_vs_hase.py [--json data/local/crew/muscle_capacity_rung2.json]
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: [HKY02] read off the plots, both sides: handle peak, hip extension peak, knee extension peak, knee flexion peak
HASE2002 = dict(handle=850.0, hip=2 * 240.0, knee_ext=2 * 40.0, knee_flex=2 * 60.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "data", "local", "crew", "muscle_capacity_rung2.json"))
    a = ap.parse_args()
    d = json.load(open(a.json))
    H = HASE2002
    print("lever arms, m (moment / peak handle force, both sides)")
    print("               hip ext   knee ext   knee flex   (peak handle N)")
    print("  [HKY02]      %6.3f    %6.3f     %6.3f      %5.0f" % (H["hip"] / H["handle"], H["knee_ext"] / H["handle"],
                                                                  H["knee_flex"] / H["handle"], H["handle"]))
    for name, v in d.items():
        s = v["series"]
        F = np.asarray(s["F"], float)
        hip = np.asarray(s["hip_ext"], float)
        knee = np.asarray(s["knee"], float)
        f = float(np.nanmax(F))
        ext_t = float(np.asarray(s["t"])[int(np.nanargmax(knee))])
        flex_t = float(np.asarray(s["t"])[int(np.nanargmin(knee))])
        print("  %-12s %6.3f    %6.3f     %6.3f      %5.0f   (knee ext peak at %.2f s, flex peak at %.2f s)"
              % (name, np.nanmax(hip) / f, np.nanmax(knee) / f, -np.nanmin(knee) / f, f, ext_t, flex_t))


if __name__ == "__main__":
    main()
