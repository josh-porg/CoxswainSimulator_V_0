"""The fins' cross-flow law on three measured blade loads: its one open number, profiled per dataset.

    .venv/Scripts/python.exe research/blade/crossflow_profile.py

The law (SOURCES §190; research/blade/helmbold.crossflow): C_N = C_L_alpha sin a cos a + C_Dc sin^2 a, C_L_alpha the
fins' Polhamus/Helmbold slope at the blade's along-flow aspect ratio, tangential drag-like. Everything is fixed from
sources except the face-on coefficient C_Dc. This scans C_Dc on each dataset and reports where each one's error is
least and how sharply: a sensitivity, not a fit that enters the model. Each dataset's own error metric:

- [GTW21] half-scale robot (grift2021_quasisteady): F_n rms, N;
- [LJH12] 0.7-scale flat plate, rigid shaft (leroyer_jh12_quasisteady): F_x rms, N (data floor ~6.7 N);
- [RLBQV19] full size, bending oar from the measured moment (robert2019_quasisteady): F_x rms, N.

And one Froude number for all three, the same definition: the centre's normal slip at mid-stroke over sqrt(g h),
h the blade height (each paper defines its own, which do not compare).

Outputs: data/local/blade/crossflow_profile.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import grift2021_quasisteady as G                                 # noqa: E402
import helmbold                                                   # noqa: E402
import leroyer_jh12_quasisteady as J                              # noqa: E402
import robert2019_quasisteady as R                                # noqa: E402

G_ACC = 9.81
CDC = np.round(np.arange(1.0, 2.61, 0.1), 2)


def grift(c):
    g, th, thd, vtip = G.motion()
    star = g / G.T_DRIVE
    fn_meas, _ = G.measured(star)
    law = G.liftdrag_law(helmbold.crossflow(G.HEIGHT, G.SPAN, G.SPAN * G.HEIGHT, False, c))
    F = np.array([G.strip_loads(a, b, v, law) for a, b, v in zip(th, thd, vtip)])
    return float(np.sqrt(np.mean((-F[:, 0] - fn_meas) ** 2)))


def leroyer(c, D, theta):
    law = J.liftdrag_law(helmbold.crossflow(J.HEIGHT, J.SPAN, J.AREA, False, c))
    _cl, _cd, _inc, fx, _fy = J.reduce(J.run(law, theta), theta)
    td = np.degrees(theta)
    return float(np.sqrt(np.mean((-fx - np.interp(td, *D["Fx_N"].T)) ** 2)))


def robert(c, D, K, meas):
    law = R.liftdrag_law(helmbold.crossflow(R.HEIGHT, R.SPAN, R.AREA, False, c))
    tr = R.loads(K, law)
    return float(np.sqrt(np.mean((tr[:, 0] - meas[:, 0]) ** 2)))


def froude():
    g, th, thd, vtip = G.motion()
    n = np.stack([np.cos(th), np.sin(th)], 1)
    wn = ((vtip + 0.5 * G.SPAN * thd[:, None] * n) * n).sum(1)
    mid = np.abs(g / G.T_DRIVE - 0.5) < 0.1
    fr_g = float(np.mean(np.abs(wn[mid])) / np.sqrt(G_ACC * G.HEIGHT))
    fr_j = float(J.LE * J.K * (1 - J.ETA0) / np.sqrt(G_ACC * J.HEIGHT))          # at theta = 90 deg
    D = R.load()
    K = R.kinematics(D, rigid=False)
    i = int(np.argmin(np.abs(K["psi_deg"] - 90.0)))
    rc = R.L_E + 0.5 * R.SPAN
    wn_r = rc * np.cos(R.THETA1) * K["rate"][i] - K["f_dot"][i] - 0.5 * R.SPAN * K["phi_dot"][i] - R.VB * np.sin(K["psi"][i] - K["phi"][i])
    fr_r = float(abs(wn_r) / np.sqrt(G_ACC * R.HEIGHT))
    return dict(GTW21=fr_g, LJH12=fr_j, RLBQV19=fr_r)


def main():
    DJ = J.load()
    theta = np.radians(np.linspace(1.0, 179.0, 357))
    DR = R.load()
    KR = R.kinematics(DR, rigid=False)
    measR = np.stack([np.interp(KR["psi_deg"], *DR[k].T) for k in ("Fx_N", "Fy_N", "Mz_Nm")], 1)
    prof = {"GTW21": [grift(c) for c in CDC], "LJH12": [leroyer(c, DJ, theta) for c in CDC],
            "RLBQV19": [robert(c, DR, KR, measR) for c in CDC]}
    fr = froude()
    print("one Froude number for all three (centre normal slip at mid-stroke / sqrt(g h)): " +
          ", ".join("%s %.2f" % kv for kv in fr.items()))
    print("\nC_Dc  " + "  ".join("%7.1f" % c for c in CDC))
    best = {}
    for k, v in prof.items():
        v = np.array(v)
        j = int(np.argmin(v))
        ok = CDC[v <= 1.10 * v[j]]                                 # within 10% of the least error
        best[k] = dict(best=float(CDC[j]), rms=float(v[j]), within_10pct=[float(ok.min()), float(ok.max())])
        print("%-7s " % k + "  ".join("%7.1f" % x for x in v) + "    least at %.1f (rms %.1f); within 10%%: %.1f-%.1f"
              % (CDC[j], v[j], ok.min(), ok.max()))
    json.dump(dict(cdc=CDC.tolist(), profiles=prof, best=best, froude=fr),
              open(os.path.join(G.OUT, "crossflow_profile.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
