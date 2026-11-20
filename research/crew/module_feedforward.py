"""Step 4's open-loop prediction: muscles fired on the on-water EMG timing, against what today's stroke needs (§254).

On rung 2's stroke as the muscle check logs it (`muscle_capacity.py --rung2`: joint moments, angles and speeds through
the drive), for the hip extensors and the elbow flexors:

* **needed** activation: the moment the stroke requires over the sourced maximum at the joint's angle and speed
  (Anderson et al. 2007 hip; Frey-Law et al. 2012 elbow; coxswain/crew/joint_muscles), as the capacity check computes;
* **predicted** activation: the module's excitation on the stroke's phase (coxswain/crew/excitation, Pitto et al. 2024
  on water) through the activation dynamics (Zajac 10 / 40 ms), scaled so its drive mean equals the needed mean (one
  amplitude per group: B4's placeholder; the *timing* is what is tested, not the level).

Both are read only inside the strength data's tested angles (blocker B1: outside them the needed activation is an
extrapolation). Reports the correlation and the lag that best aligns predicted with needed (positive: the stroke needs
the muscle later than rowers fire it, as an electromechanical delay would), per group and athlete. A lag far beyond
~50-100 ms, or a poor correlation, is the shape the body's kinematics must change by for step 4's gate.

    python research/crew/module_feedforward.py [muscle_capacity json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from coxswain.crew import joint_muscles as J                      # noqa: E402
from coxswain.crew.excitation import Modules, PITTO2024_FINISH    # noqa: E402

GROUP = {"cr06": "F18-25", "br24": "M18-25"}                      # as muscle_capacity.py
SEX = {"cr06": "F", "br24": "M"}
TESTED = {"hip": (-0.5, 1.3)}                                      # Anderson et al.'s tested hip range, rad (flexion +)


def activation_from_excitation(t, u, dyn):
    a = np.zeros_like(u)
    for i in range(1, t.size):
        a[i] = a[i - 1] + (t[i] - t[i - 1]) * float(dyn.rate(u[i - 1], a[i - 1]))
    return np.clip(a, 0.0, 1.0)


def best_lag(t, need, pred, lags):
    out = []
    for L in lags:
        p = np.interp(t - L, t, pred)
        m = np.isfinite(need)
        out.append(np.corrcoef(need[m], p[m])[0, 1] if m.sum() > 5 else np.nan)
    k = int(np.nanargmax(out))
    return float(lags[k]), float(out[k]), float(out[int(np.argmin(abs(lags)))])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "local", "crew", "muscle_capacity_rung2_s253.json")
    d = json.load(open(path))
    mods = Modules.pitto2024()
    dyn = J.ActivationDynamics.of("zajac")
    print("stroke from", os.path.basename(path), "; EMG modules: Pitto et al. 2024; activation dynamics: Zajac")
    for a in ("cr06", "br24"):
        if a not in d:
            continue
        s = {k: np.asarray(v, float) for k, v in d[a]["series"].items()}
        t, F = s["t"], s["F"]
        live = np.flatnonzero(F > 0.05 * F.max())
        t_drive = float(t[live[-1]])
        on = t <= t_drive
        T_cycle = t_drive / PITTO2024_FINISH                       # the drive scaled onto the EMG's cycle
        phase = t / T_cycle
        HE = J.anderson("HE", GROUP[a], d[a]["mass"], d[a]["stature"], passive=False)
        EF = J.elbow("flexion", SEX[a])
        groups = {
            "hip extensors": (HE.activation_needed(s["hip_ext"], s["th_hip"], s["w_hip"]), "trunk_extensors",
                              (s["th_hip"] >= TESTED["hip"][0]) & (s["th_hip"] <= TESTED["hip"][1])),
            "elbow flexors": (EF.activation_needed(np.maximum(s["elbow"], 0.0), s["th_elbow"], s["w_elbow"]), "arm_pullers",
                              np.ones_like(t, dtype=bool)),
        }
        print("\n%s: drive %.2f s" % (a, t_drive))
        for name, (need, mod, tested) in groups.items():
            need = np.where(np.isfinite(need) & tested, need, np.nan)
            u = mods(mod, phase)
            pred = activation_from_excitation(t, u, dyn)
            sel = on & np.isfinite(need)
            if sel.sum() < 6:
                print("  %-14s too few samples inside the strength data's tested angles (B1)" % name)
                continue
            scale = np.nanmean(need[sel]) / max(np.mean(pred[sel]), 1e-9)
            lag, r_best, r0 = best_lag(t[sel], need[sel], scale * pred[sel], np.linspace(-0.4, 0.4, 161))
            q = np.arange(0.0, 1.01, 0.1)
            print("  %-14s inside tested angles %.0f %% of the drive; needed vs predicted, per 10 %% of the drive:"
                  % (name, 100 * sel.sum() / on.sum()))
            print("    needed    " + " ".join("%5.2f" % np.interp(x * t_drive, t[sel], need[sel], left=np.nan, right=np.nan)
                                            for x in q))
            print("    predicted " + " ".join("%5.2f" % (scale * np.interp(x * t_drive, t, pred)) for x in q))
            print("    correlation %.2f at zero lag; best %.2f with the need %+.0f ms after the EMG-driven activation"
                  % (r0, r_best, 1000 * lag))


if __name__ == "__main__":
    main()
