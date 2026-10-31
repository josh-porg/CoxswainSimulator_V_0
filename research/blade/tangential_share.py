"""How much of the blade's work goes into its tangential load, on the one measured on-water blade motion?

    .venv/Scripts/python.exe research/blade/tangential_share.py

SOURCES §196. [GTW21]'s robot drives a half-scale sweep blade along a measured on-water path with its load measured.
On that motion: the share of the blade's dissipated work |F_t w_a| / (|F_n w_n| + |F_t w_a|) for the measured load and
for the review blade (the fins' cross-flow law, C_Dc 2.05), and the blade-centre angle of attack and sliding fraction
|w_a|/|w| by fifths of the drive. Compared with the closed-loop cost of the same law in the boat (§193: 6-10 % of
speed), this separates a wrong law from a wrong blade motion.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grift2021_quasisteady as G  # noqa: E402
import helmbold  # noqa: E402


def main():
    g, th, thd, v = G.motion()
    star = g / G.T_DRIVE
    fn_m, ft_m = G.measured(star)
    centre = G.SPAN / 2
    wn = np.array([float(v[i] @ np.array([np.cos(th[i]), np.sin(th[i])])) + centre * thd[i] for i in range(len(g))])
    wa = np.array([float(v[i] @ np.array([-np.sin(th[i]), np.cos(th[i])])) for i in range(len(g))])

    def share(fn, ft):
        pn, pt = np.trapezoid(np.abs(fn * wn), g), np.trapezoid(np.abs(ft * wa), g)
        return pt / (pn + pt)
    law = G.liftdrag_law(helmbold.crossflow(G.HEIGHT, G.SPAN, G.SPAN * G.HEIGHT, False, 2.05))
    fl = np.array([G.strip_loads(th[i], thd[i], v[i], law) for i in range(len(g))])
    print("[GTW21] on-water path: tangential share of blade work, measured %.1f%%, review blade %.1f%%"
          % (100 * share(fn_m, ft_m), 100 * share(fl[:, 0], fl[:, 1])))
    alpha = np.degrees(np.arctan2(np.abs(wn), np.abs(wa)))
    for k in range(5):
        sel = (star >= k / 5) & (star < (k + 1) / 5)
        print("  fifth %d: centre angle of attack %3.0f deg, sliding |w_a|/|w| %.2f"
              % (k + 1, alpha[sel].mean(), np.mean(np.abs(wa[sel]) / np.hypot(wn[sel], wa[sel]))))


if __name__ == "__main__":
    main()
