"""What makes the trunk's hip moment peak early? Its terms, on measured motion, against the EMG (§271).

§270: on the athletes' own trunk motion, the hip moment rung2_trunk computes wants the trunk extensors 100-150 ms
before on-water EMG fires them. The moment (rung2_trunk._equations) is the upper body's free-body balance about the
hip:

    M = (z_hand - z_hip) F_handle                                  handle
      + sum_i r_x,i (-m_i g)                                       gravity
      - sum_i (r_x,i m_i a_z,i - r_z,i m_i a_x,i)   [a relative]   segment inertia (the body's own motion)
      - sum_i (r_x,i m_i A_z - r_z,i m_i A_x)       [A the hull's] hull recoil
      - I_spin theta''                                             segment spin

This script logs each term along the reference run (tau = clock, trunk on its path) on each athlete's own leg and
back channels, and reports each term's value and timing through the drive: where each peaks, and how the sum's
centroid compares with the trunk-extensor EMG's (Pitto et al. 2024, peak at 40 % of the drive; force follows
excitation by ~50-100 ms). The term whose timing pulls the sum early is the inverse-dynamics part to examine.

    python research/crew/trunk_hip_moment_terms.py [--strokes 8]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "biorow"), os.path.join(ROOT, "research", "blade")]
import rung2_trunk as RT                                         # noqa: E402

from coxswain.crew.excitation import Modules, PITTO2024_FINISH    # noqa: E402

C2, RP = RT.C2, RT.RP
G = RT.G


class TermsSim(RT.TrunkClockSim):
    """rung2_trunk's simulator with the hip moment's terms logged in reference mode."""

    def _equations(self, t, base, d, tau, td, tdd):
        resid, M = super()._equations(t, base, d, tau, td, tdd)
        if self.mode == "free":
            return resid, M
        mass, pos, _vel, acc = self.crew_field(t)
        mass = np.asarray(mass, float)
        hip = pos[RT.TRUNK[0], [0, 2]] - RT._rot(0.0) @ self._r_at(RT.TRUNK[0], tau)
        F = self._handle_force(t, base)
        hand_z = hip[1] + float(self._hand[1](float(tau) % self._P))
        handle = (hand_z - hip[1]) * F
        grav = inert = recoil = 0.0
        for i in RT.TRUNK + RT.ARMS:
            rx, rz = pos[i, 0] - hip[0], pos[i, 2] - hip[1]
            m = mass[i]
            grav += rx * (-m * G)
            inert -= rx * m * acc[i, 2] - rz * m * acc[i, 0]
            recoil -= rx * m * d[8] - rz * m * d[6]
        spin = M - (handle + grav + inert + recoil)
        if not hasattr(self, "termlog"):
            self.termlog = []
        if not self.termlog or t > self.termlog[-1][0]:
            self.termlog.append((t, t - self._stroke_start, handle, grav, inert, recoil, spin, M))
        return resid, M


def centroid(x, y):
    yp = np.clip(y, 0.0, None)
    return float(np.sum(x * yp) / max(np.sum(yp), 1e-9))


def athlete(name, strokes):
    RP.BLADE = "crossflow_normal"
    C2.TAKEOVER_DEG = C2.BIOROW_CATCH_TO_BURIED_DEG
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True, own_joints=RP.OWN_BODY_JOINTS)
    RT.TrunkClockSim, base = TermsSim, RT.TrunkClockSim
    try:
        sim = RT.make(boat, "reference", body=RP.own_body(name, boat))
    finally:
        RT.TrunkClockSim = base
    run = sim.run_strokes(strokes, surge_speed=4.3)
    start = float(run.last_time[0])
    L = np.array([r for r in sim.termlog if r[0] >= start])
    P = float(boat.timing.period)
    D = float(boat.timing.drive_fraction) * P
    c = L[:, 1] % P
    on = c <= D
    x = c[on] / D
    names = ["handle", "gravity", "segment inertia", "hull recoil", "spin", "total"]
    mods = Modules.pitto2024()
    grid = np.linspace(0.0, 1.0, 201)
    emg = mods("trunk_extensors", grid * PITTO2024_FINISH)
    emg_centroid = centroid(grid, emg)
    print("\n%s (own motion): drive %.2f s; trunk-extensor EMG centroid at %.0f %% of the drive (peak 40 %%); force "
          "follows by ~50-100 ms = %.0f-%.0f %% of the drive" % (name, D, 100 * emg_centroid, 100 * 0.05 / D,
                                                                  100 * 0.10 / D))
    out = {}
    print("  %-16s %9s %9s %9s %10s %10s" % ("term", "mean N m", "max N m", "peak at", "centroid", "at 10/30/50/70%"))
    o = np.argsort(x)
    for j, nm in enumerate(names):
        y = L[on, 2 + j][o]
        xs = x[o]
        k = int(np.argmax(y))
        vals = [float(np.interp(q, xs, y)) for q in (0.1, 0.3, 0.5, 0.7)]
        out[nm] = dict(mean=float(np.mean(y)), max=float(np.max(y)), peak_at=float(xs[k]), centroid=centroid(xs, y),
                       at=vals)
        print("  %-16s %9.0f %9.0f %8.0f%% %9.0f%% %s" % (nm, np.mean(y), np.max(y), 100 * xs[k], 100 * centroid(xs, y),
                                                          " ".join("%5.0f" % v for v in vals)))
    out["emg_centroid"] = emg_centroid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=8)
    ap.add_argument("--athletes", nargs="+", default=["cr06", "br24"])
    a = ap.parse_args()
    res = {n: athlete(n, a.strokes) for n in a.athletes}
    path = os.path.join(RT.OUT, "trunk_hip_moment_terms.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
