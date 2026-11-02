"""Does a straight arm from the athlete's own shoulders reproduce the athlete's measured oar angle? (SOURCES §208)

    .venv/Scripts/python.exe research/crew/straight_arm_check.py

The body-coupled drive puts the hands where a straight arm from the shoulders meets the grip's arc, until the arm
draw. On the model's keyframed body that runs the oar 15-25° ahead of [BR24]'s measured angle at 0.3-0.5 s, because
the keyframed legs and trunk move the shoulders too early (§205's sequencing gap). This checks the coupling itself,
with nothing fitted, by asking what the oar does when the shoulders are re-timed on each athlete's own measured
channels, the §205 own-body diagnostic:

* the legs on the seat channel;
* the back on the trunk channel, scaled to the athlete's travel.

The footboard is placed for a straight-arm catch from those shoulders. "Straight" is the population's held elbow
flexion, 10.4 deg (coxswain.crew.arm_draw; until SOURCES 209 it was the elbow solve's 0.95 margin). The straight-arm angle is then printed against the measured angle through the drive.
Where the two agree, the arms are straight and the coupling holds; where the measured angle runs ahead, the arms
are drawing.

Outputs: data/local/crew/straight_arm_check.json.
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
import onwater_body as OB                                        # noqa: E402

OUT = C2.OUT


def _ns(name):
    if name == "br24":
        import measured_body as MB
        return dict(measured_channel=MB.measured_channel, model_channel=MB.model_channel, warp=MB.warp,
                    leg="Ls", back="Lt", K=None)
    path = os.path.join(ROOT, "research", "cr06", "self_consistent_body_cr06.py")
    src = open(path, encoding="utf-8").read().split("grid = np.linspace")[0]
    ns = {"__file__": path, "__name__": "cr06_body"}
    exec(compile(src, path, "exec"), ns)
    return dict(measured_channel=ns["measured_channel"], model_channel=ns["model_channel"], warp=ns["warp"],
                leg="leg_disp_m", back="back_disp_m", K=ns["K_BACK"])


def own_shoulder(name, boat):
    """The shoulder (x, z relative to the ankle) on the athlete's own leg and back time laws, as a function of time
    from the catch: the model's hip at the leg-warped time plus K times its shoulder-from-hip at the back-warped
    time (measured_body.body_field's construction, read at the shoulder)."""
    ns = _ns(name)
    rower = boat.crew[0].rower
    P = float(boat.timing.period)
    m_l, s_l, c_l = ns["measured_channel"](ns["leg"])[:3]
    mm_l, tau_l, cm_l = ns["model_channel"](boat, lambda j: j["hip"][0] - j["ankle"][0])[:3]
    mb_out = ns["measured_channel"](ns["back"])
    m_b, s_b, c_b = mb_out[:3]
    model_b = ns["model_channel"](boat, lambda j: j["shoulder"][0] - j["hip"][0])
    mm_b, tau_b, cm_b = model_b[:3]
    K = ns["K"] if ns["K"] is not None else mb_out[3] / model_b[3]    # his travel / the model's ([BR24])
    wl, wb = ns["warp"](s_l, c_l, tau_l, cm_l), ns["warp"](s_b, c_b, tau_b, cm_b)

    def tw(t, spline, meas_min, model_min):
        real = t / P - meas_min
        k = np.floor(real)
        return P * (k + float(spline(real - k)) + model_min)

    def shoulder(t):
        jl = rower.joint_positions(tw(t, wl, m_l, mm_l) % P)
        jb = rower.joint_positions(tw(t, wb, m_b, mm_b) % P)
        hip = jl["hip"] - jl["ankle"]
        return hip + K * (jb["shoulder"] - jb["hip"])
    return shoulder, K


def check(name):
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True, catch_posture="as-built")
    rower = boat.crew[0].rower
    arm = OB._hold_reach(rower)
    rig = OB.rig_geometry(boat)
    sh, K = own_shoulder(name, boat)
    lo, hi = -0.6, 0.6
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        a = OB.straight_arm_angle(sh(0.0), rig, arm, mid)
        lo, hi = (mid, hi) if (np.isnan(a) or a > rig["catch"]) else (lo, mid)
    shift = 0.5 * (lo + hi)
    ht, hs, ha = RP.measured_track(name)
    keyed, _ = OB._body_of(boat)
    rows = []
    for t in np.arange(0.0, ht[-1], 0.05):
        own = np.degrees(OB.straight_arm_angle(sh(t), rig, arm, shift))
        meas = float(np.interp(t, ht, ha))
        rows.append((float(t), own, meas))
    print("%s: K %.3f, footboard %+.3f m for a straight-arm catch on the athlete's own shoulders" % (name, K, shift))
    for t, own, meas in rows:
        print("  t %.2f s: straight arm from own shoulders %6.1f deg | measured %6.1f | measured - straight %+5.1f"
              % (t, own, meas, meas - own))
    return dict(K=K, footboard_shift=shift, rows=rows)


def main():
    out = {n: check(n) for n in ("br24", "cr06")}
    json.dump(out, open(os.path.join(OUT, "straight_arm_check.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
