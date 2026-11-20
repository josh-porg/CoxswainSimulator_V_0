"""The model's joint demands against on-water EMG timing: does the body load its groups when rowers' muscles fire? (§253)

A sourced check of the body's sequencing before the multi-clock body (review §12 step 4's gate is that the legs ->
trunk -> arms sequence *emerges*; this measures where today's body stands). The muscle check (`muscle_capacity.py
--rung2`) logs, inside rung 2's stroke, the hip-extensor and elbow-flexor moments the model's body needs. Pitto et al.
(2024)'s on-water EMG (coxswain/crew/excitation.Modules) gives when the trunk extensors and the arm pullers fire. EMG
leads force by the electromechanical delay and the activation dynamics (excitation to force ~50-100 ms; Zajac 1989,
Cavanagh & Komi 1979), so the EMG peak should *precede* the moment's peak by about that much.

Prints, per athlete: the drive-phase timing (fraction of the drive) of each demand's peak and of its rise through 50 %
of peak, against the module's, and the implied lead (EMG before moment, ms).

    python research/crew/sequencing_vs_emg.py [muscle_capacity json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from coxswain.crew.excitation import Modules, PITTO2024_FINISH    # noqa: E402

PAIRS = (("hip_ext", "trunk_extensors", "trunk extensors (hip moment)"),
         ("elbow", "arm_pullers", "arm pullers (elbow flexor moment)"))


def timing(x, y):
    """(peak, first 50 % rise) of ``y`` against ``x``."""
    y = np.asarray(y, float)
    k = int(np.argmax(y))
    half = np.flatnonzero(y[:k + 1] >= 0.5 * y[k])
    return float(x[k]), float(x[half[0]]) if half.size else float("nan")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "local", "crew", "muscle_capacity_rung2.json")
    d = json.load(open(path))
    mods = Modules.pitto2024()
    grid = np.linspace(0.0, PITTO2024_FINISH, 400)
    print("demands from", os.path.basename(path), "; EMG: Pitto et al. 2024 (on-water)")
    for a in ("cr06", "br24"):
        if a not in d:
            continue
        s = d[a]["series"]
        t = np.asarray(s["t"], float)
        F = np.asarray(s["F"], float)
        # the drive: from the catch (t = 0) to where the handle force last falls through zero after its peak
        kp = int(np.argmax(F))
        after = np.flatnonzero(F[kp:] <= 0.0)
        t_drive = float(t[kp + after[0]]) if after.size else float(t[-1])
        print("\n%s: drive %.2f s (from the handle force)" % (a, t_drive))
        print("  %-36s | demand peak / 50 %% rise (of drive) | EMG peak / 50 %% rise | EMG lead at peak" % "group")
        for key, mod, label in PAIRS:
            y = np.asarray(s[key], float)
            on = t <= t_drive
            pk, rise = timing(t[on] / t_drive, y[on])
            ek, erise = timing(grid / PITTO2024_FINISH, mods(mod, grid))
            print("  %-36s |        %.2f / %.2f                 |   %.2f / %.2f       | %+4.0f ms"
                  % (label, pk, rise, ek, erise, 1000 * (pk - ek) * t_drive))
    print("\nexpected: the EMG leads the moment by ~50-100 ms (excitation to force); a negative lead means the model's"
          " body loads the group before rowers' muscles fire")


if __name__ == "__main__":
    main()
