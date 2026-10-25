"""Step 4.3, rung 1 (rebuilt 2026-10-24): the stretcher force of [CR06]'s woman from her own data.

    .venv/Scripts/python.exe research/cr06/stretcher_cr06.py

The original (cr06/stretcher_from_her_data_cr06.py, 2026-09-14) was lost with a session scratchpad; its
results are in docs/TRACKING.md ("Step 4.3, first rung"). The force itself is her_stroke.Stroke.
stretcher_force (the body's momentum balance along the boat, frictionless seat, [CR06] eq. 5's centre of mass
with r = 0.4). This prints the record's checks across the smoothing band:
- at [LE26]'s catch landmark (minimum boat acceleration): the stretcher force, her absolute acceleration and
  its parts (boat, legs, 0.4 x back), against the record's 408-448 N and +5.2 to +5.7 m/s^2;
- the lead: the stretcher reaching 300 N before the landmark, the gate force after it (record 0.23 s);
- the peaks: stretcher against gate force (record 675-726 N against 792 N, ratio 0.85-0.92).
Gate force: the handles' force times (1 + inboard / blade lever), the oar's lever balance about the pin, with
[CR06]'s rig (force point 0.83 m from the lock, blade centre 1.805 m).

The model's own stretcher (rung 1's second half, cr06/stretcher_from_balance_cr06.py: 190 N at the catch after
its evaluation fix) needs the simulator and is not rebuilt here.
Writes data/local/cr06_study/stretcher_cr06.json.
"""
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import her_stroke as H                                         # noqa: E402

S_INBOARD, L_BLADE = 0.83, 1.805
RECORDED = dict(landmark_N=(408, 448), abs_acc=(5.2, 5.7), lead_s=0.23, peak_N=(675, 726), gate_peak_N=792, ratio=(0.85, 0.92))


def check(lam):
    S = H.Stroke(lam, lam)
    Fs = S.stretcher_force()
    k = S.k_landmark
    legs = S.legS.derivative(2)(S.t); back = S.backS.derivative(2)(S.t)
    gate = S.F_handle * (1 + S_INBOARD / L_BLADE)
    # first crossing of 300 N going into the drive, relative to the landmark (periodic)
    def first_up(y, thr=300.0):
        i = np.flatnonzero((y[:-1] < thr) & (y[1:] >= thr))
        ti = S.t[i] - S.t[k]
        ti = (ti + H.T / 2) % H.T - H.T / 2                    # nearest to the landmark
        return float(ti[np.argmin(np.abs(ti))]) if len(ti) else float("nan")
    return dict(lam=lam, t_landmark=float(S.t[k]), stretcher_landmark=float(Fs[k]), a_boat=float(S.a_boat[k]), legs=float(legs[k]),
                back=float(0.4 * back[k]), abs_acc=float(S.a_boat[k] + legs[k] + 0.4 * back[k]), gate_landmark=float(gate[k]),
                stretcher_300_s=first_up(Fs), gate_300_s=first_up(gate), stretcher_peak=float(Fs.max()), gate_peak=float(gate.max()),
                ratio=float(Fs.max() / gate.max()))


def main():
    rows = [check(lam) for lam in H.LAM_BAND]
    for r in rows:
        print("lam %.0e: landmark t %.3f s, stretcher %.0f N, abs. acceleration %+.2f m/s^2 (boat %+.2f, legs %+.2f, 0.4 x back %+.2f), gate %.0f N | "
              "300 N: stretcher %+.3f s, gate %+.3f s from the landmark (lead %.2f s) | peaks %.0f N vs gate %.0f N, ratio %.2f"
              % (r["lam"], r["t_landmark"], r["stretcher_landmark"], r["abs_acc"], r["a_boat"], r["legs"], r["back"], r["gate_landmark"],
                 r["stretcher_300_s"], r["gate_300_s"], r["gate_300_s"] - r["stretcher_300_s"], r["stretcher_peak"], r["gate_peak"], r["ratio"]))
    print("recorded: landmark 408-448 N, +5.2 to +5.7 m/s^2 (boat -8.8, legs +12.0, back 0.4 x 4.5), gate 34-36 N; stretcher 300 N 0.02 s before, "
          "gate 0.21 s after (lead 0.23 s); peaks 675-726 N vs 792 N, ratio 0.85-0.92")
    json.dump(dict(rows=rows, recorded=RECORDED), open(os.path.join(H.OUT, "stretcher_cr06.json"), "w"), indent=1)
    print("written", os.path.join(H.OUT, "stretcher_cr06.json"))


if __name__ == "__main__":
    main()
