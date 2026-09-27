"""Do two on-water scullers share a seat and trunk time-law, and how far is the ergometer body?

    python research/biorow/timelaw_compare.py

Two measured on-water strokes: [BR24] (elite man, 32.4 spm; data/local, commercial) and
[CR06] Fig. 3 (woman, 30.9 spm; data/literature). For each, the leg channel (seat relative
to foot) and the back channel (shoulder relative to seat) are normalised to 0-1 and timed
from the catch (the oar's extreme catch angle), on two clocks:

  phase   t / T
  split   drive and recovery each scaled to their own length (catch -> finish -> catch,
          finish = the oar's extreme finish angle), which removes the rhythm difference

The research model's ergometer body (Caplan & Gardner keyframes) is timed the same way. The
question for an on-water driver: is the athletes' disagreement with each other much smaller
than either's with the ergometer body? Printed: rms differences (fraction of travel).
"""
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

G = np.linspace(0.0, 1.0, 400, endpoint=False)


def periodic(t, y, T):
    o = np.argsort(t)
    t, y = np.asarray(t)[o], np.asarray(y)[o]
    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    t, y = t[keep], y[keep]
    if t[-1] >= T - 1e-9:
        t, y = t[:-1], y[:-1]
    return CubicSpline(np.append(t, t[0] + T), np.append(y, y[0]), bc_type="periodic")


def clocks(t_catch, t_finish, T):
    """Map each normalised clock back to absolute time."""
    drive = (t_finish - t_catch) % T
    phase = lambda g: (t_catch + g * T) % T
    fd = drive / T

    def split(g):
        g = np.asarray(g)
        return np.where(g < 0.5, t_catch + (g / 0.5) * drive,
                        t_catch + drive + ((g - 0.5) / 0.5) * (T - drive)) % T
    return phase, split, fd


def norm(y):
    return (y - y.min()) / np.ptp(y)


def athlete_br24():
    D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"), delimiter=",", names=True)[:-1]
    T = 60.0 / 32.4093
    t = np.arange(len(D)) * T / len(D)
    A = 0.5 * (D["A1"] + D["A2"])                    # catch negative
    fine = np.linspace(0, T, 4000, endpoint=False)
    As = periodic(t, A, T)(fine)
    return dict(name="BR24 (M, 32.4 spm)", T=T, leg=periodic(t, D["Ls"], T), back=periodic(t, D["Lt"], T),
                t_catch=fine[np.argmin(As)], t_finish=fine[np.argmax(As)])


def athlete_cr06():
    import csv
    path = os.path.join(ROOT, "data", "literature", "cr06_fig3_measured.csv")
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    T = 1.94

    def series(name):
        pts = [(float(r["t_s"]), float(r["value"])) for r in rows if r["series"] == name]
        t, v = np.array(pts).T
        return np.mod(t, T), v
    fine = np.linspace(0, T, 4000, endpoint=False)
    ta, a = series("oar_angle_deg")                   # catch positive in this figure
    As = periodic(ta, a, T)(fine)
    tl, l = series("leg_disp_m")
    tb, b = series("back_disp_m")
    return dict(name="CR06 (W, 30.9 spm)", T=T, leg=periodic(tl, l, T), back=periodic(tb, b, T),
                t_catch=fine[np.argmax(As)], t_finish=fine[np.argmin(As)])


def model_body():
    import like_for_like as L
    boat = L.build("base")
    r = boat.crew[0].rower
    T = boat.timing.period
    t = np.linspace(0, T, 1200, endpoint=False)
    ch = r._chain(t)
    hip, ank, sh = ch["hip"][0].value, ch["ankle"][0].value, ch["shoulder"][0].value
    ang = np.array([float(boat.oar_sweep(x, boat.timing)) for x in t])   # catch positive
    return dict(name="model ergometer body", T=T, leg=periodic(t, hip - ank, T), back=periodic(t, sh - hip, T),
                t_catch=t[np.argmax(ang)], t_finish=t[np.argmin(ang)])


def curves(a):
    phase, split, fd = clocks(a["t_catch"], a["t_finish"], a["T"])
    out = {"drive_fraction": fd}
    for ch in ("leg", "back"):
        f = a[ch]
        y_phase = norm(f(phase(G)))
        y_split = norm(f(split(G)))
        # sign: both channels rise through the drive
        if y_split[int(0.45 * len(G))] < y_split[0]:
            y_phase, y_split = 1 - y_phase, 1 - y_split
        out[ch] = dict(phase=y_phase, split=y_split)
    return out


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    athletes = [athlete_br24(), athlete_cr06(), model_body()]
    cs = [curves(a) for a in athletes]
    for a, c in zip(athletes, cs):
        print("%-22s drive fraction %.3f" % (a["name"], c["drive_fraction"]))
    print("\nrms difference, fraction of travel      leg: phase / split      back: phase / split")
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        print("  %-20s vs %-22s  %.3f / %.3f          %.3f / %.3f"
              % (athletes[i]["name"][:20], athletes[j]["name"][:22],
                 rms(cs[i]["leg"]["phase"], cs[j]["leg"]["phase"]), rms(cs[i]["leg"]["split"], cs[j]["leg"]["split"]),
                 rms(cs[i]["back"]["phase"], cs[j]["back"]["phase"]), rms(cs[i]["back"]["split"], cs[j]["back"]["split"])))
    print("\n split clock, catch -> finish -> catch (0.5 = finish):  leg  BR24 / CR06 / model     back  BR24 / CR06 / model")
    for g in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        k = int(g * len(G))
        print("   %.2f                                                 %.2f / %.2f / %.2f          %.2f / %.2f / %.2f"
              % (g, *(c["leg"]["split"][k] for c in cs), *(c["back"]["split"][k] for c in cs)))


if __name__ == "__main__":
    main()
