"""Anatomical hip, knee and spine angles through the rowing stroke from open motion-capture data (§268).

Source: Telfer et al. (2023), "The effect of foot-stretcher position and stroke rate on ergometer rowing kinematics",
PLoS ONE (PMC10174490); data github.com/Telfer/Rowing_Biomechanics (MIT licence), cloned to
data/local/literature/telfer2023_rowing_kinematics/. Eleven collegiate rowers (6 women, 5 men), Concept2 ergometer,
seven foot-stretcher positions x rates 22 / 26 / 32; Visual3D joint angles (link-model based) for both hips, both
knees, the lower back (lumbar on pelvis) and the upper back, each stroke normalised to 101 points finish to finish.

The model's trunk is one rigid segment, so its "hip" angle is the trunk-thigh angle with the spine lumped in (67-180
deg along rung 2's stroke; §267). This script gives the population's split: the anatomical hip's range and its share of
hip + lumbar + thoracic flexion, at the finish and at the catch (the stroke's largest hip flexion), per rower and
pooled, so that anatomical-hip data (Anderson 2007 strength, Silder 2007 passive) can be read at the right angle.

    python -I research/crew/telfer2023_kinematics.py [--rate 32]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "local", "literature", "telfer2023_rowing_kinematics", "Data")
OUT = os.path.join(ROOT, "data", "literature", "telfer2023_hip_spine_split.json")


def load(rower, name):
    rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(DATA, rower, name), encoding="latin-1")]
    files = rows[0][1:]
    data = np.array([[float(x) if x.strip() else np.nan for x in r[1:]] for r in rows[5:]])
    return files, data


def rate_of(path):
    base = os.path.basename(path.replace("\\", "/")).replace(".c3d", "")
    try:
        return int(base.split()[-1])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=None, help="only trials at this stroke rate (22, 26, 32)")
    a = ap.parse_args()
    per = {}
    for rower in sorted(os.listdir(DATA)):
        files, hipR = load(rower, "RightHip.txt")
        _, hipL = load(rower, "LeftHip.txt")
        _, low = load(rower, "LowerBack.txt")
        _, up = load(rower, "UpperBack.txt")
        _, kneeR = load(rower, "RightKnee.txt")
        keep = np.array([a.rate is None or rate_of(f) == a.rate for f in files])
        if not keep.any():
            continue
        n = min(hipR.shape[1], hipL.shape[1], low.shape[1], up.shape[1], kneeR.shape[1], keep.size)
        keep = keep[:n]
        hip = 0.5 * (hipR[:, :n] + hipL[:, :n])[:, keep]
        lowk, upk, knee = low[:, :n][:, keep], up[:, :n][:, keep], -kneeR[:, :n][:, keep]   # knee flexion positive
        good = np.all(np.isfinite(hip), axis=0) & np.all(np.isfinite(lowk), axis=0) & np.all(np.isfinite(upk), axis=0) \
            & np.all(np.isfinite(knee), axis=0)
        hip, lowk, upk, knee = hip[:, good], lowk[:, good], upk[:, good], knee[:, good]
        keep = np.zeros(1, bool) if not good.any() else keep
        if hip.shape[1] == 0:
            continue
        k_catch = np.nanargmax(hip, axis=0)                               # the catch: the largest hip flexion
        cols = np.arange(hip.shape[1])
        at = lambda x, idx: x[idx, cols]
        hip_f, hip_c = np.nanmean(hip[0]), np.nanmean(at(hip, k_catch))
        low_f, low_c = np.nanmean(lowk[0]), np.nanmean(at(lowk, k_catch))
        up_f, up_c = np.nanmean(upk[0]), np.nanmean(at(upk, k_catch))
        per[rower] = dict(strokes=int(hip.shape[1]), hip_finish=hip_f, hip_catch=hip_c, knee_catch=np.nanmean(at(knee, k_catch)),
                          lumbar_finish=low_f, lumbar_catch=low_c, thoracic_finish=up_f, thoracic_catch=up_c,
                          catch_at=float(np.nanmean(k_catch)) / 100.0)
    keys = ["hip_finish", "hip_catch", "knee_catch", "lumbar_finish", "lumbar_catch", "thoracic_finish", "thoracic_catch"]
    pooled = {k: (float(np.mean([p[k] for p in per.values()])), float(np.std([p[k] for p in per.values()])))
              for k in keys}
    print("Telfer et al. 2023, %d rowers%s (finish = stroke start; catch = largest hip flexion):"
          % (len(per), "" if a.rate is None else ", rate %d" % a.rate))
    print("%-8s %6s | %6s %6s %6s | %7s %7s | %7s %7s" % ("rower", "n", "hipF", "hipC", "kneeC", "lumbF", "lumbC",
                                                         "thorF", "thorC"))
    for r, p in per.items():
        print("%-8s %6d | %6.1f %6.1f %6.1f | %7.1f %7.1f | %7.1f %7.1f" % (
            r, p["strokes"], p["hip_finish"], p["hip_catch"], p["knee_catch"], p["lumbar_finish"], p["lumbar_catch"],
            p["thoracic_finish"], p["thoracic_catch"]))
    print("pooled mean (SD): " + "; ".join("%s %.1f (%.1f)" % (k, *pooled[k]) for k in keys))
    hip_range = pooled["hip_catch"][0] - pooled["hip_finish"][0]
    spine = abs(pooled["lumbar_catch"][0] - pooled["lumbar_finish"][0]) + abs(pooled["thoracic_catch"][0] - pooled["thoracic_finish"][0])
    print("anatomical hip range %.1f deg; spine (lumbar + thoracic) change %.1f deg; hip share %.2f"
          % (hip_range, spine, hip_range / (hip_range + spine)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(source="Telfer et al. 2023 PLoS ONE; github.com/Telfer/Rowing_Biomechanics (MIT)", rate=a.rate,
                   per_rower=per, pooled=pooled, hip_range=hip_range, spine_change=spine),
              open(OUT if a.rate is None else OUT.replace(".json", "_rate%d.json" % a.rate), "w"), indent=1)
    print("wrote", OUT if a.rate is None else OUT.replace(".json", "_rate%d.json" % a.rate))


if __name__ == "__main__":
    sys.exit(main())
