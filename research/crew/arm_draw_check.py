"""The drive the body generates, with the arm draw on its own clock (SOURCES §209): checks on both athletes' boats.

    .venv/Scripts/python.exe research/crew/arm_draw_check.py

Per boat (the held-arm catch, rung 1's on-water timing):

* the footboard shift and the solved finish elbow flexion (the population's is 100 deg, band ~80-120);
* the oar's peak angular acceleration through the drive, against the population law's own on the same boat (the
  blended hand-over of §208 reached 6300 deg/s^2 against the law's 1200);
* the angle and the rate where the drive meets the recovery (catch and finish), from both sides;
* the oar angle against the athlete's measured angle through the drive, with the population law alongside.

Outputs: data/local/crew/arm_draw_check.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]
import rung2_clock as C2                                         # noqa: E402
import rung2_population as RP                                    # noqa: E402

OUT = C2.OUT


def check(name):
    m = C2.R.V.ATHLETES[name][0]()
    law_boat = C2.R.boat_for(name, m, "on-water", drive_law=True, catch_posture="as-built")
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True, catch_posture="straight-arms")
    tm, sw, law = boat.timing, boat.oar_sweep, law_boat.oar_sweep
    P, D = float(tm.period), float(tm.drive_fraction) * float(tm.period)
    t = np.linspace(0.0, D, 4001)[1:-1]
    acc = lambda s: np.abs(np.gradient(np.degrees(s.rate(t, tm)), t))
    eps = 1e-4
    joins = {}
    for label, at in (("catch", P), ("finish", D)):
        joins[label] = dict(angle_jump_deg=float(np.degrees(sw(at + eps, tm) - sw(at - eps, tm))),
                            rate_before=float(np.degrees(sw.rate(at - eps, tm))),
                            rate_after=float(np.degrees(sw.rate(at + eps, tm))))
    ht, _hs, ha = RP.measured_track(name)
    rows = []
    for tt in np.arange(0.0, min(D, ht[-1]) + 1e-9, 0.05):
        rows.append((float(tt), float(np.degrees(sw(tt, tm))), float(np.degrees(law(tt, tm))),
                     float(np.interp(tt, ht, ha))))
    body, lw, meas = (np.array([r[k] for r in rows]) for k in (1, 2, 3))
    out = dict(footboard_shift=float(boat.footboard_shift), onset=float(boat.arm_draw_onset),
               finish_elbow_deg=float(boat.finish_elbow), peak_acc=float(acc(sw).max()),
               peak_acc_at=float(t[np.argmax(acc(sw))] / D), law_peak_acc=float(acc(law).max()), joins=joins,
               rms_body_vs_measured=float(np.sqrt(np.mean((body - meas) ** 2))),
               rms_law_vs_measured=float(np.sqrt(np.mean((lw - meas) ** 2))), rows=rows)
    print("%s: footboard %+.3f m; draw from %.3f of the drive; finish elbow %.1f deg (population 100)"
          % (name, out["footboard_shift"], out["onset"], out["finish_elbow_deg"]))
    print("  peak oar acceleration %.0f deg/s^2 at %.2f of the drive (population law on this boat: %.0f)"
          % (out["peak_acc"], out["peak_acc_at"], out["law_peak_acc"]))
    for k, j in joins.items():
        print("  %s: angle jump %.3f deg; rate %.1f -> %.1f deg/s" % (k, j["angle_jump_deg"], j["rate_before"],
                                                                     j["rate_after"]))
    print("  oar angle vs measured, rms over the drive: body %.1f deg, population law %.1f deg"
          % (out["rms_body_vs_measured"], out["rms_law_vs_measured"]))
    for tt, b, l_, me in rows[::2]:
        print("    t %.2f s: body %6.1f  law %6.1f  measured %6.1f" % (tt, b, l_, me))
    return out


def check_own(name):
    """The same arm clock on the athlete's own leg and back timing (straight_arm_check.own_shoulder, the §205
    own-body diagnostic): the footboard for a held-arm catch from those shoulders, the finish flexion solved at the
    measured finish, the oar angle against the measured one. Separates the arms from the keyframed body's sequencing."""
    import onwater_body as OB
    import straight_arm_check as SC
    from coxswain.crew import arm_draw as A
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True, catch_posture="as-built")
    rower = boat.crew[0].rower
    u, f = rower.upper_arm_length, rower.forearm_length
    rig = OB.rig_geometry(boat)
    sh, _K = SC.own_shoulder(name, boat)
    lo, hi = -0.6, 0.6
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        a = OB.straight_arm_angle(sh(0.0), rig, OB._hold_reach(rower), mid)
        lo, hi = (mid, hi) if (np.isnan(a) or a > rig["catch"]) else (lo, mid)
    shift = 0.5 * (lo + hi)
    ht, _hs, ha = RP.measured_track(name)
    Dm = float(ht[np.argmin(ha)])                           # the measured finish: the angle's turning point
    flex = np.radians(np.arange(10.4, 160.0, 0.05))
    ends = np.array([OB.straight_arm_angle(sh(Dm), rig, float(A.reach(u, f, x)), shift) for x in flex])
    ok = np.isfinite(ends)
    draw = A.ElbowDraw(finish=float(flex[ok][np.argmin(np.abs(ends[ok] - rig["finish"]))]))
    rows = []
    for tt in np.arange(0.0, Dm + 1e-9, 0.05):
        own = OB.straight_arm_angle(sh(tt), rig, float(A.reach(u, f, draw.flexion(tt / Dm))), shift)
        rows.append((float(tt), float(np.degrees(own)), float(np.interp(tt, ht, ha))))
    d = np.array([r[1] - r[2] for r in rows])
    out = dict(footboard_shift=shift, finish_elbow_deg=float(np.degrees(draw.finish)), rms=float(np.sqrt(np.mean(d**2))),
               worst=float(d[np.argmax(np.abs(d))]), worst_at=float(rows[int(np.argmax(np.abs(d)))][0]),
               late_drive_max=float(np.abs(d[[r[0] >= 0.7 for r in rows]]).max()), rows=rows)
    print("%s on its own leg and back timing: finish elbow %.1f deg (population 100, band ~80-120); oar vs measured "
          "rms %.1f deg, worst %+.1f deg at %.2f s, within %.1f deg from 0.7 s" % (
              name, out["finish_elbow_deg"], out["rms"], out["worst"], out["worst_at"], out["late_drive_max"]))
    return out


def main():
    out = {n: check(n) for n in ("br24", "cr06")}
    for n in ("br24", "cr06"):
        out[n]["own_timing"] = check_own(n)
    json.dump(out, open(os.path.join(OUT, "arm_draw_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
