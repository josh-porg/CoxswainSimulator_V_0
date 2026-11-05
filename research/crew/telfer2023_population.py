"""The UW ergometer population's segment timing by stroke rate (Engstrom ... Telfer 2023, PLOS ONE; SOURCES §218).

Open data (github.com/Telfer/Rowing_Biomechanics, MIT), held in data/local/literature/telfer2023_data/: 11 college
rowers, seven foot-stretcher positions at 22, 26 and 32 spm, sagittal joint angles per stroke, each stroke time-
normalised to 101 points from finish to finish. Their paper: the catch at 60.6 / 57.0 / 53.2 % of the cycle.

This reads the neutral foot position (the ergometer's standard set-up), finds each stroke's catch as the knee's peak
flexion, re-normalises each drive from catch (0) to finish (1), and reports for each joint the drive progress at which
it has covered 10 %, 50 % and 90 % of its own drive excursion: the population's sequencing of legs, hips, back and
arms, rate by rate, as the body's segment clocks (SOURCES §211-215) are written.

    python research/crew/telfer2023_population.py [--position Neutral]

Outputs data/local/crew/telfer2023/drive_<position>_<rate>.csv (population mean course of each joint against drive
progress, each joint scaled 0 at the catch to 1 at the finish) and prints the landmarks.
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "local", "literature", "telfer2023_data", "Data")
OUT = os.path.join(ROOT, "data", "local", "crew", "telfer2023")
JOINTS = ("RightKnee", "RightHip", "LowerBack", "UpperBack", "RightShoulder", "RightElbow", "RightAnkle")
RATES = (22, 26, 32)
GRID = np.linspace(0.0, 1.0, 51)


def read_joint(rower, joint):
    """``{trial name: (101, strokes) array}`` for one rower's joint file."""
    lines = open(os.path.join(DATA, rower, joint + ".txt"), encoding="utf-8").read().split("\n")
    names = [p.split("\\")[-1].replace(".c3d", "") for p in lines[0].split("\t")[1:]]
    rows = [l.split("\t") for l in lines[5:] if l.strip()]
    def number(x):
        x = x.strip()
        return float(x) if x else np.nan                        # a blank cell: a sample the export did not fill
    values = np.array([[number(x) for x in (r[1:len(names) + 1] + [""] * len(names))[:len(names)]] for r in rows])
    out = {}
    for j, name in enumerate(names):
        if np.isnan(values[:, j]).any():
            continue                                            # a stroke with gaps is dropped, not filled
        out.setdefault(name, []).append(values[:, j])
    return {k: np.array(v).T for k, v in out.items()}


def drive_course(knee, joint):
    """One stroke: the catch at the knee's peak flexion (its extreme over the cycle's middle), the drive catch ->
    finish (the cycle's end) resampled to ``GRID``, and the joint scaled 0 at the catch to 1 at the finish."""
    n = knee.size
    middle = slice(int(0.3 * n), int(0.85 * n))
    k = knee[middle]
    # the knee is most flexed at the catch; the sign of flexion is whichever extreme sits further from the finish value
    i = int(np.argmax(np.abs(k - knee[-1]))) + middle.start
    seg = joint[i:]
    x = np.linspace(0.0, 1.0, seg.size)
    y = np.interp(GRID, x, seg)
    span = y[-1] - y[0]
    if abs(span) < 1e-6:
        return None, i / (n - 1)
    return (y - y[0]) / span, i / (n - 1)


def landmarks(course):
    out = []
    for level in (0.1, 0.5, 0.9):
        above = np.flatnonzero(course >= level)
        out.append(float(GRID[above[0]]) if above.size else float("nan"))
    return out


def _angle(a, b, c):
    """Angle at ``b`` between ``a`` and ``c`` (x, z pairs of arrays), degrees."""
    u = np.stack([a[0] - b[0], a[1] - b[1]])
    v = np.stack([c[0] - b[0], c[1] - b[1]])
    cos = (u * v).sum(0) / np.maximum(np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def model_landmarks(clocks=()):
    """The research body's knee angle, trunk inclination (hip to shoulder) and elbow angle over its drive, on
    [BR24]'s boat as the leg-landmark study builds it (SOURCES §214), scored as the population is."""
    import sys
    sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]
    import rung2_clock as C2
    m = C2.R.V.ATHLETES["br24"][0]()
    boat = C2.R.boat_for("br24", m, "on-water", drive_law=True, catch_posture="as-built", clocks=clocks)
    tm = boat.timing
    D = float(tm.drive_fraction) * float(tm.period)
    t = np.linspace(0.0, D, 400)
    ch = boat.crew[0].rower._chain(t)
    pos = {k: (np.asarray(v[0].value, float), np.asarray(v[1].value, float)) for k, v in ch.items()
           if k in ("ankle", "knee", "hip", "shoulder", "elbow", "hand")}
    series = {"knee": _angle(pos["ankle"], pos["knee"], pos["hip"]),
              "trunk": np.degrees(np.arctan2(pos["shoulder"][0] - pos["hip"][0], pos["shoulder"][1] - pos["hip"][1]))}
    if "elbow" in pos and "hand" in pos:
        series["elbow"] = _angle(pos["shoulder"], pos["elbow"], pos["hand"])
    x = np.linspace(0.0, 1.0, t.size)
    out = {}
    for name, y in series.items():
        y = np.interp(GRID, x, y)
        span = y[-1] - y[0]
        out[name] = landmarks((y - y[0]) / span) if abs(span) > 1e-6 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="Neutral")
    ap.add_argument("--model", action="store_true", help="also score the research body's drive the same way")
    args = ap.parse_args()
    if args.model:
        for label, clocks in (("research body (trunk clock)", ("pitto2024_trunk",)),
                              ("keyframes only", ()), ("[K05] legs + trunk clock", ("pitto2024_trunk", "k05_legs"))):
            print("%-28s %s" % (label, "  ".join("%s %s" % (k, "/".join("%.2f" % v for v in l) if l else "-")
                                                for k, l in model_landmarks(clocks).items())))
        return
    os.makedirs(OUT, exist_ok=True)
    rowers = sorted(d for d in os.listdir(DATA) if d.startswith("RFP"))
    print("UW ergometer population (%d rowers), foot position %s; drive progress catch 0 -> finish 1" %
          (len(rowers), args.position))
    for rate in RATES:
        trial = "%s %d" % (args.position, rate)
        per_joint = {j: [] for j in JOINTS}
        catches = []
        for rower in rowers:
            knee_all = read_joint(rower, "RightKnee").get(trial)
            if knee_all is None:
                continue
            for joint in JOINTS:
                data = read_joint(rower, joint).get(trial)
                if data is None:
                    continue
                strokes = []
                for s in range(min(knee_all.shape[1], data.shape[1])):
                    c, frac = drive_course(knee_all[:, s], data[:, s])
                    if c is not None:
                        strokes.append(c)
                    if joint == "RightKnee":
                        catches.append(frac)
                if strokes:
                    per_joint[joint].append(np.mean(strokes, axis=0))     # each rower's mean stroke
        print("\n%d spm: catch at %.1f %% of the cycle (paper: %s)" % (
            rate, 100 * float(np.mean(catches)), {22: "60.6", 26: "57.0", 32: "53.2"}[rate]))
        print("  joint           rowers   10 %%   50 %%   90 %% of its drive excursion (drive progress)")
        table = [GRID]
        header = ["progress"]
        for joint in JOINTS:
            if not per_joint[joint]:
                continue
            mean = np.mean(per_joint[joint], axis=0)
            spread = np.std([landmarks(c)[1] for c in per_joint[joint]])
            l10, l50, l90 = landmarks(mean)
            print("  %-14s  %5d   %.2f   %.2f   %.2f   (50 %% point sd across rowers %.2f)" % (
                joint, len(per_joint[joint]), l10, l50, l90, spread))
            table.append(mean)
            header.append(joint)
        np.savetxt(os.path.join(OUT, "drive_%s_%d.csv" % (args.position.replace(" ", "_"), rate)),
                   np.array(table).T, delimiter=",", header=",".join(header), comments="", fmt="%.5f")


if __name__ == "__main__":
    main()
