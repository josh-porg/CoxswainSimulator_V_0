"""The blade's path through the water in the boat: measured athletes against the simulator, by fifths of the drive.

    .venv/Scripts/python.exe research/blade/blade_path_in_boat.py [--strokes 10]

SOURCES §196 left one question: the review blade's tangential share of blade work is right on [GTW21]'s measured
on-water motion (19.6 % against 15.7 %), yet in the boat the same law costs 6-10 % of speed, so the simulated blade
must move through the water differently from a real one (the measured sweep path turns the blade to 73 deg at the
centre in the fourth fifth of the drive). This script measures that path on both sides, with one formula:

    w = u_lock + l * theta_dot * n(theta),   attack angle a = atan(|w_n| / |w_a|),   sliding |w_a| / |w|

(coxswain.crew.liftdrag.LiftDragBlade.relative_velocity, l = the oar's outboard to the blade centre). Measured: the
athlete's own oar angle and boat velocity ([CR06] Fig. 3; [BR24]'s telemetry file), lock velocity = boat velocity
(no sway or yaw). Simulated: the last stroke of the same athlete's closed-loop run (research/crew/handle_rung1.py's
setup: hands on the handle, [K05]'s drive law, nothing fitted), oar angle, rate and lock velocity from the
integrator's states. The drive runs from the catch (largest angle) to the finish (smallest); fifths by time.

Also reported per fifth: the advance ratio J = |u| / (l |theta_dot|), boat speed, and the oar angle, so a difference
in attack angle can be read as the oar's sweep (angle, rate) or the boat's speed.

Outputs: data/local/blade/blade_path_in_boat.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"), ROOT]
import handle_rung1 as R                                          # noqa: E402
from coxswain.core.state import State                             # noqa: E402
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402

V = R.V
OUT = os.path.join(ROOT, "data", "local", "blade")
CONFIGS = {
    "research slip": {},
    "cross-flow C_Dc 2.2, strips": dict(blade_law="liftdrag", blade_coefficients="crossflow@2.2", strips=True),
}


def fifths(t, angle, rate, u, l, law=None):
    """Per fifth of the drive: attack angle (deg), sliding fraction, advance ratio (median), |u|, oar angle (deg);
    with ``law`` (a LiftDragBlade), the tangential share of the blade's dissipated work over the drive, at the centre."""
    i0 = int(np.argmax(angle))
    i1 = i0 + int(np.argmin(angle[i0:]))
    sel = slice(i0, i1 + 1)
    t, angle, rate, u = t[sel], angle[sel], rate[sel], u[sel]
    blade = LiftDragBlade(area=0.1, outboard=l)
    w = np.array([blade.relative_velocity(a, r, uu, 1) for a, r, uu in zip(angle, rate, u)])
    alpha = np.degrees(np.arctan2(np.abs(w[:, 0]), np.abs(w[:, 1])))
    slide = np.abs(w[:, 1]) / np.hypot(w[:, 0], w[:, 1])
    adv = np.linalg.norm(u, axis=1) / np.maximum(l * np.abs(rate), 1e-9)
    star = (t - t[0]) / (t[-1] - t[0])
    rows = []
    for k in range(5):
        m = (star >= k / 5) & (star < (k + 1) / 5 + (1e-9 if k == 4 else 0))
        rows.append(dict(alpha=float(alpha[m].mean()), slide=float(slide[m].mean()), advance=float(np.median(adv[m])),
                         speed=float(np.linalg.norm(u[m], axis=1).mean()), angle=float(np.degrees(angle[m]).mean())))
    share = None
    if law is not None:
        f = np.array([law.loads(a, r, uu, 1) for a, r, uu in zip(angle, rate, u)])
        pn, pt = np.trapezoid(np.abs(f[:, 0] * w[:, 0]), t), np.trapezoid(np.abs(f[:, 1] * w[:, 1]), t)
        share = float(pt / (pn + pt))
    return dict(drive_s=float(t[-1] - t[0]), catch_deg=float(np.degrees(angle[0])), finish_deg=float(np.degrees(angle[-1])),
                fifths=rows, tangential_share=share)


def measured_path(name):
    if name == "cr06":
        g = np.linspace(0.0, V.CR06_T, 4000, endpoint=False)
        ang = V._cr06_series("oar_angle_deg")
        th, thd = np.radians(ang(g)), np.radians(ang(g, 1))
        v = V._cr06_series("boat_velocity_mps")(g)
        # unwrap the cycle so the catch comes first
        k = int(np.argmax(th))
        roll = lambda x: np.roll(x, -k)
        return g, roll(th), roll(thd), np.column_stack([roll(v), np.zeros_like(v)])
    import like_for_like as L
    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    T = 60.0 / L.RATE
    t = np.arange(len(D)) * T / len(D)
    a = np.radians(-0.5 * (D["A1"] + D["A2"]))                    # file: catch -67, finish +36; flipped to catch +
    k = int(np.argmax(a))
    a, v = np.roll(a, -k), np.roll(D["Vs"], -k)
    ad = np.gradient(np.unwrap(a), t)
    return t, a, ad, np.column_stack([v, np.zeros_like(v)])


def simulated_path(r):
    sim, run = r["sim"], r["run"]
    t, Y = run.last_time, run.last_states
    n = sim.n_oar_states
    from coxswain.core.state import STATE_SIZE
    angle, rate = Y[STATE_SIZE], Y[STATE_SIZE + n]
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    u = np.array([sim._lock_velocity(State.from_vector(Y[:STATE_SIZE, k]), lock)[:2] for k in range(len(t))])
    side = int(lock.side)
    return t, angle, rate, u * np.array([1.0, side]), float(sim._oars[0].outboard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}

    def show(label, d):
        print("  %-30s drive %.2f s, catch %+.0f deg, finish %+.0f deg; %.0f W, %.3f m/s at the measured power; review"
              " blade's tangential share of blade work on this path %.1f%%" % (label, d["drive_s"], d["catch_deg"],
                                                                              d["finish_deg"], d["power"],
                                                                              d["speed_at_power"], 100 * d["tangential_share"]))
        for key, fmt in (("alpha", "%5.0f"), ("slide", "%5.2f"), ("advance", "%5.2f"), ("speed", "%5.2f"), ("angle", "%+5.0f")):
            print("      %-8s" % key + " ".join(fmt % f[key] for f in d["fifths"]))

    for name in ("cr06", "br24"):
        m = V.ATHLETES[name][0]()
        print("\n%s (measured %.3f m/s)" % (name, m["speed"]), flush=True)
        res = {}
        for label, cfg in CONFIGS.items():
            cfg = dict(cfg)
            boat = R.boat_for(name, m, "on-water", drive_law=True)
            if cfg.pop("strips", False):
                cfg["blade_span"] = float(boat.rig.seats[0].oarlocks[0].oar.blade_length)
            r = R.predict(boat, a.strokes, **cfg)
            t, ang, rate, u, l = simulated_path(r)
            oar = boat.rig.seats[0].oarlocks[0].oar
            law = LiftDragBlade.crossflow(outboard=l, area=float(oar.blade_area), length=float(oar.blade_length), face_on=2.2)
            res[label] = dict(speed=r["speed"], power=r["power"], speed_at_power=r["speed"] * (m["power"] / r["power"]) ** (1 / 3),
                              **fifths(t, ang, rate, u, l, law))
            res[label]["outboard"] = l
        t, ang, rate, u = measured_path(name)
        res["measured"] = dict(speed=m["speed"], power=m["power"], speed_at_power=m["speed"],
                               **fifths(t, ang, rate, u, res["research slip"]["outboard"], law))
        show("measured", res["measured"])
        for label in CONFIGS:
            show("%s (%.2f m/s)" % (label, res[label]["speed"]), res[label])
        out[name] = res
    json.dump(out, open(os.path.join(OUT, "blade_path_in_boat.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
