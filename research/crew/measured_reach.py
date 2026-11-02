"""Is the 1.12-arm-length reach real, and what is it? Each athlete's shoulder rebuilt from their own channels.

    .venv/Scripts/python.exe research/crew/measured_reach.py

SOURCES §208 found the measured grip ~1.12 arm lengths from the shoulder in mid-drive, on both athletes, and
proposed the shoulder girdle. That shoulder was the own-body reconstruction (straight_arm_check.own_shoulder): the
model's hip on the leg-warped clock plus K times the model's whole shoulder-from-hip vector (height included). The
athletes' channels allow a reconstruction that keeps the trunk's length:

* the hip moves with the seat channel (seat height constant);
* the trunk channel is the horizontal displacement relative to the seat of a point a distance ``d_p`` along the
  trunk: [BR24]'s Lt at d_p = 0.583 m (fitted by whole-system momentum, SOURCES §156); [CR06]'s x_S/B is the
  shoulder itself (d_p = the trunk). So ``x_p - x_p,catch = d_p (sin th - sin th_catch)``, th the lean from vertical;
* the shoulder is the hip plus the model's own hip-to-shoulder length at angle th.

The catch lean is not in the channels: the model's own catch posture and Kleshnev's on-water elite catch
(24.5 deg forward, [K19]) are both tried. The footboard is placed so the catch closes with the held arm
(coxswain.crew.arm_draw). Then, through the drive:

* the grip-to-shoulder distance from the measured oar angle, in arm lengths;
* the excess over the held reach against the measured handle force per arm, while the arms are held (to 0.45 of
  the drive, Pitto et al. 2024): a straight-line fit gives the chain's apparent stiffness (an athlete measurement,
  a check on any sourced compliance, never an input).

Outputs: data/local/crew/measured_reach.json.
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
import onwater_body as OB                                        # noqa: E402

OUT = C2.OUT
BR24_D_P = 0.583                  # m, [BR24]'s trunk channel's tracked point along the trunk (SOURCES §156)
K19_CATCH_LEAN = np.radians(-24.5)
HELD_UNTIL = 0.45                 # drive fraction: the arms are held (Pitto et al. 2024: the draw starts at 0.465)


def channels(name):
    """From the catch to the finish: time (s), oar angle (model sense, rad), seat and trunk-point displacement (m,
    + toward the bow, zero at the catch), handle force per arm (N), and d_p (m, or None for the shoulder itself)."""
    T_grid = 2000
    if name == "br24":
        import like_for_like as L
        D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
        T = 60.0 / L.RATE
        t0 = np.arange(len(D)) * T / len(D)
        g = np.linspace(0.0, T, T_grid, endpoint=False)
        series = {k: np.interp(g, t0, v, period=T) for k, v in
                  (("ang", -0.5 * (D["A1"] + D["A2"])), ("seat", D["Ls"]), ("trunk", D["Lt"]),
                   ("force", 0.5 * (D["H1"] + D["H2"])))}                 # per scull = per arm
        d_p = BR24_D_P
    else:
        import validate_blade as V
        T = V.CR06_T
        g = np.linspace(0.0, T, T_grid, endpoint=False)
        series = dict(ang=V._cr06_series("oar_angle_deg")(g), seat=V._cr06_series("leg_disp_m")(g),
                      trunk=V._cr06_series("back_disp_m")(g),
                      force=0.5 * V._cr06_series("handle_force_N")(g))   # Fig. 3's F is both hands summed
        d_p = None
    i0 = int(np.argmax(series["ang"]))                                   # the catch
    s = {k: np.roll(v, -i0) for k, v in series.items()}
    j = int(np.argmin(s["ang"]))                                         # the finish
    s = {k: v[: j + 1] for k, v in s.items()}
    for k in ("seat", "trunk"):                                          # + toward the bow, zero at the catch
        v = s[k] - s[k][0]
        s[k] = v if v[-1] > 0 else -v
    return g[: j + 1], np.radians(s["ang"]), s["seat"], s["trunk"], s["force"], d_p


def reach_table(name, lean_catch=None):
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True, catch_posture="as-built")
    free, rower = OB._body_of(boat)
    rig = OB.rig_geometry(boat)
    L = rower.upper_arm_length + rower.forearm_length
    j = free.joint_positions(0.0)
    hip0, sh0 = np.asarray(j["hip"], float), np.asarray(j["shoulder"], float)
    v0 = sh0 - hip0
    trunk = float(np.hypot(v0[0], v0[2]))
    th_model = float(np.arctan2(v0[0], v0[2]))                           # + toward the bow (leaning back)
    th_c = th_model if lean_catch is None else float(lean_catch)
    t, phi, seat, tp, force, d_p = channels(name)
    s_th = np.sin(th_c) + tp / (d_p or trunk)
    if np.any(np.abs(s_th) > 1):
        raise ValueError("the trunk channel needs a lean past horizontal at this catch lean")
    th = np.arcsin(s_th)
    sh = np.stack([hip0[0] + seat + trunk * np.sin(th), np.full_like(th, 0.20), hip0[2] + trunk * np.cos(th)],
                  axis=1)
    hold = OB._hold_reach(rower)
    lo, hi = -0.8, 0.8
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        a = OB.straight_arm_angle(sh[0, [0, 1, 2]], rig, hold, mid)
        lo, hi = (mid, hi) if (np.isnan(a) or a > phi[0]) else (lo, mid)
    shift = 0.5 * (lo + hi)
    sh[:, 0] += shift
    grip = np.stack([rig["x"] - rig["r"] * np.sin(phi), rig["y"] - rig["r"] * np.cos(phi), np.full_like(phi, rig["z"])],
                    axis=1)
    dist = np.linalg.norm(grip - sh, axis=1)
    excess = dist - hold
    held = t <= HELD_UNTIL * t[-1]
    A = np.column_stack([force[held], np.ones(held.sum())])
    (c, b), *_ = np.linalg.lstsq(A, excess[held], rcond=None)
    r2 = 1.0 - np.sum((excess[held] - A @ [c, b]) ** 2) / np.sum((excess[held] - excess[held].mean()) ** 2)
    c0 = float(np.dot(force[held], excess[held]) / np.dot(force[held], force[held]))
    r20 = 1.0 - np.sum((excess[held] - c0 * force[held]) ** 2) / np.sum((excess[held] - excess[held].mean()) ** 2)
    return dict(t=t, reach=dist / L, excess=excess, force=force, lean_catch_deg=float(np.degrees(th_c)),
                lean_finish_deg=float(np.degrees(th[-1])), th_model_deg=float(np.degrees(th_model)), shift=shift,
                stiffness=float(1.0 / c), offset=float(b), r2=float(r2), stiffness_through_zero=float(1.0 / c0),
                r2_through_zero=float(r20), peak_excess=float(excess[held].max()),
                peak_force_held=float(force[held].max()), arm=L)


def full_drive(name, k_arm, lean_catch=None):
    """The whole drive from the athlete's own shoulders (rebuilt as in ``reach_table``): the arm on its clock
    (arm_draw, the finish flexion solved at the measured finish) stretched by ``F / k_arm`` along the arm
    (``k_arm`` None: no stretch). Returns ``(rms_deg, worst_deg, finish_flexion_deg)`` against the measured oar."""
    from coxswain.crew import arm_draw as A
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True, catch_posture="as-built")
    free, rower = OB._body_of(boat)
    rig = OB.rig_geometry(boat)
    u, f = rower.upper_arm_length, rower.forearm_length
    r = reach_table(name, lean_catch)
    t, phi, seat, tp, force, d_p = channels(name)
    j = free.joint_positions(0.0)
    hip0, sh0 = np.asarray(j["hip"], float), np.asarray(j["shoulder"], float)
    v0 = sh0 - hip0
    trunk = float(np.hypot(v0[0], v0[2]))
    th_c = float(np.arctan2(v0[0], v0[2])) if lean_catch is None else float(lean_catch)
    th = np.arcsin(np.sin(th_c) + tp / (d_p or trunk))
    sh = np.stack([hip0[0] + seat + trunk * np.sin(th) + r["shift"], np.full_like(th, 0.20),
                   hip0[2] + trunk * np.cos(th)], axis=1)
    stretch = np.zeros_like(force) if k_arm is None else np.clip(force, 0.0, None) / k_arm
    x = t / t[-1]

    def angles(draw):
        rr = A.reach(u, f, draw.flexion(x)) + stretch
        return np.array([OB.straight_arm_angle(s, rig, float(q), 0.0) for s, q in zip(sh, rr)])
    lo, hi = A.HOLD_FLEXION, np.radians(160.0)
    for _ in range(40):                      # the finish flexion that lands the hand on the measured finish
        mid = 0.5 * (lo + hi)
        end = OB.straight_arm_angle(sh[-1], rig, float(A.reach(u, f, mid) + stretch[-1]), 0.0)
        lo, hi = (mid, hi) if (not np.isnan(end) and end > phi[-1]) else (lo, mid)   # NaN: too bent to reach
    draw = A.ElbowDraw(finish=0.5 * (lo + hi))
    d = np.degrees(angles(draw) - phi)
    return float(np.sqrt(np.nanmean(d**2))), float(d[np.nanargmax(np.abs(d))]), float(np.degrees(draw.finish))


def main():
    out = {}
    for name in ("br24", "cr06"):
        for label, lean in (("model catch lean", None), ("K19 catch lean", K19_CATCH_LEAN)):
            r = reach_table(name, lean)
            out["%s / %s" % (name, label)] = {k: (list(map(float, v)) if isinstance(v, np.ndarray) else v)
                                               for k, v in r.items()}
            print("%s, %s: trunk %.1f -> %.1f deg, footboard %+.3f m" % (name, label, r["lean_catch_deg"],
                                                                         r["lean_finish_deg"], r["shift"]))
            print("   arms held (to %.2f s): excess up to %.3f m at %.0f N per arm; excess = F/k + c: k %.0f N/m per "
                  "arm, c %+.3f m, R2 %.3f; through zero k %.0f N/m, R2 %.3f"
                  % (HELD_UNTIL * r["t"][-1], r["peak_excess"], r["peak_force_held"], r["stiffness"], r["offset"],
                     r["r2"], r["stiffness_through_zero"], r["r2_through_zero"]))
            for tt in np.arange(0.0, r["t"][-1], 0.1):
                i = int(np.argmin(np.abs(r["t"] - tt)))
                print("      t %.1f s: reach %.3f L, excess %+.3f m, force %4.0f N" % (tt, r["reach"][i],
                                                                                  r["excess"][i], r["force"][i]))
    # the whole drive: no stretch, each athlete's own spring, and the other athlete's (out of sample)
    ks = {n: out["%s / model catch lean" % n]["stiffness_through_zero"] for n in ("br24", "cr06")}
    cross = {}
    for name, other in (("br24", "cr06"), ("cr06", "br24")):
        for label, k in (("no stretch", None), ("own k %.0f" % ks[name], ks[name]),
                         ("other athlete's k %.0f" % ks[other], ks[other])):
            rms, worst, fin = full_drive(name, k)
            cross["%s / %s" % (name, label)] = dict(rms=rms, worst=worst, finish_flexion=fin)
            print("%s whole drive, %s: oar vs measured rms %.1f deg (worst %+.1f), finish elbow %.0f deg"
                  % (name, label, rms, worst, fin))
    out["whole_drive"] = cross
    json.dump(out, open(os.path.join(OUT, "measured_reach.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
