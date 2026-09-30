"""Steering study: the research eight on [WF58] eq. [1], reflection swept, Munk refitted.

SOURCES sec. 169.  For each reflection factor (how much of a reflection plane the hull is
to its fin; 1 none, 2 [WF58]'s ground board) the Munk factor is refitted by the procedure
of SOURCES sec. 36 as corrected in sec. 60: full-helm rudder authority -- the turn rate the
rudder buys over the rig's own zero-helm yaw, as ``scripts/validate.py`` measures it --
set to the coxswain's ~3 deg/s (about 15 degrees of heading in 5 s).  Then, at the fitted
factor: turn rate and steady radius against helm, the straight-line criterion, and the
skeg-loss drift that brackets the Munk factor from below (ACRAs: a lane in 20-30 s).

The research profile is the one run: dynamic oar, [CR06] sweep catch, the eight at rate 28
and 260 W per rower (a ``scorecard.DYNAMIC_POINTS`` point, ~5.0 m/s).  Nothing is fitted
to an athlete; the reflection factor is swept, not chosen by the fit.

    python scripts/steering_study.py --reflection 1.5      # one process per value
    python scripts/steering_study.py --law legacy          # the shipped fin law, for scale
    python scripts/steering_study.py --combine             # table from the JSON files

Each run writes ``data/local/steering_study/<tag>.json``.  Run the values in separate
processes: every Munk trial is a new matched-torque key and a new settle.
"""
import argparse
import copy
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np

from coxswain import physics
from coxswain.boats import catalog
from coxswain.sim.control import Coxswain
from coxswain.sim.dynamic_oar import simulator_for

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "local" / "steering_study"

RATE = 28.0
WATTS = 260.0
SPEED = 5.0
CYCLES, SETTLE = 12, 6
FULL_HELM = 45.0
#: The coxswain's full-rudder rate, deg/s, over the boat's own swing (SOURCES sec. 36).
TARGET = 3.0
HELMS = (-45.0, -25.0, -5.0, 0.0, 5.0, 10.0, 15.0, 25.0, 45.0)


def research(law, reflection, munk):
    base = physics.resolve("research")
    profile = dataclasses.replace(base, fin_law=law, fin_reflection=float(reflection),
                                  fin_crossflow=0.80, munk_factor=float(munk))
    boat = profile.apply(catalog.build("8+", rate=RATE))
    boat.handle_watts = WATTS
    boat.power_scales = np.ones(boat.n_seats)
    return boat


def turn(boat, helm_deg):
    """Mean yaw rate (deg/s, positive to port) and mean speed after the settle."""
    cox = Coxswain(rudder_override=lambda _t, _s: np.radians(helm_deg))
    sim = simulator_for(boat, coxswain=cox, fast=True)
    res = sim.run(duration=CYCLES * boat.timing.period, surge_speed=SPEED)
    tt = np.asarray(res.time)
    keep = tt >= SETTLE * boat.timing.period
    yaw = np.degrees(np.unwrap(np.asarray(res.attitude)[2]))
    return (float(np.polyfit(tt[keep], yaw[keep], 1)[0]),
            float(np.mean(np.asarray(res.speed, float)[keep])))


def authority(law, reflection, munk):
    boat = research(law, reflection, munk)
    neutral, _ = turn(boat, 0.0)
    full, _ = turn(boat, FULL_HELM)
    return abs(full - neutral), neutral, full


def fit_munk(law, reflection, log):
    """Secant on the Munk factor for full-helm authority = TARGET, kept inside [0, 1]."""
    pts = []
    for m in (0.35, 0.65):
        a, _, _ = authority(law, reflection, m)
        pts.append((m, a))
        log("  munk %.4f  authority %.3f" % (m, a))
    for _ in range(6):
        (m0, a0), (m1, a1) = pts[-2], pts[-1]
        if abs(a1 - TARGET) < 0.02 or a1 == a0:
            break
        m2 = float(np.clip(m1 + (TARGET - a1) * (m1 - m0) / (a1 - a0), 0.0, 1.0))
        a2, _, _ = authority(law, reflection, m2)
        pts.append((m2, a2))
        log("  munk %.4f  authority %.3f" % (m2, a2))
        if m2 in (0.0, 1.0):
            break
    best = min(pts, key=lambda p: abs(p[1] - TARGET))
    return best[0], pts


def directional_stability(boat, speed=SPEED):
    """``Nv`` and ``C = Yv Nr - Nv (Yr - m U)``, as ``scripts/validate.py`` measures them.

    Hull and fin loads only, so the oar is irrelevant: the boat is copied without its
    profile stamp to reach the ordinary simulator's breakdown.
    """
    from coxswain.core.frames import abs_to_hull, attitude_from_components
    from coxswain.core.state import State
    from coxswain.hydro.addedmass import AddedMass
    from coxswain.sim.control import BalanceController
    from coxswain.sim.simulator import RowingSimulator

    plain = copy.copy(boat)
    plain.__dict__.pop("physics_profile", None)
    sim = RowingSimulator(plain)
    sim.coxswain.balance = BalanceController(enabled=False)
    sim.coxswain.rudder_override = lambda _t, _s: 0.0

    def loads(sway=0.0, yaw_rate=0.0):
        state = State.create(attitude=attitude_from_components(roll=0.0),
                             velocity=(speed, sway, 0.0), omega=(0.0, 0.0, yaw_rate))
        parts = sim.breakdown(0.35, state)
        rot = abs_to_hull(state.attitude)
        force = rot @ (parts.resistance_force + parts.appendage_force)
        moment = rot @ (parts.resistance_moment + parts.appendage_moment)
        return float(force[1]), float(moment[2])

    y0, n0 = loads()
    yv, nv = loads(sway=0.05)
    yr, nr = loads(yaw_rate=0.01)
    Yv, Nv = (yv - y0) / 0.05, (nv - n0) / 0.05
    Yr, Nr = (yr - y0) / 0.01, (nr - n0) / 0.01
    added = AddedMass.from_offsets(boat.offsets, rho=boat.water.density)
    mass = boat.total_mass + float(added.matrix[1, 1])
    return Nv, (Yv * Nr - Nv * (Yr - mass * speed)) / 1e6


def skeg_lost(law, reflection, munk):
    """Heading change in 25 s with skeg and rudder gone, from 2 degrees off."""
    boat = research(law, reflection, munk)
    boat.appendages = ()
    sim = simulator_for(boat, coxswain=Coxswain(rudder_override=lambda _a, _b: 0.0),
                        fast=True)
    y0 = sim.initial_state(surge_speed=SPEED)
    y0[5] = np.radians(2.0)
    res = sim.run(duration=25.0, initial_state=y0, surge_speed=SPEED)
    return abs(float(np.degrees(np.unwrap(np.asarray(res.attitude)[2]))[-1]))


def sweep_helm(law, reflection, munk):
    boat = research(law, reflection, munk)
    rows = {}
    for helm in HELMS:
        rate, speed = turn(boat, helm)
        rows[helm] = {"rate": rate, "speed": speed,
                      "radius": speed / max(abs(np.radians(rate)), 1e-9)}
    base = rows[0.0]["rate"]
    for helm, row in rows.items():
        row["authority"] = abs(row["rate"] - base)
    nv, crit = directional_stability(boat)
    return {"munk": munk, "helm": {str(k): v for k, v in rows.items()},
            "ratio_25_5": rows[25.0]["authority"] / max(rows[5.0]["authority"], 1e-9),
            "ratio_45_5": rows[45.0]["authority"] / max(rows[5.0]["authority"], 1e-9),
            "Nv": nv, "C_millions": crit}


def study(law, reflection):
    tag = "legacy" if law == "legacy" else "wf_r%.2f" % reflection
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def log(msg):
        print("[%s %6.0fs] %s" % (tag, time.time() - started, msg), flush=True)

    fin = research(law, reflection, 0.5).appendages[0]
    out = {"law": law, "reflection": reflection, "rate": RATE, "watts": WATTS,
           "target_deg_s": TARGET, "full_helm_deg": FULL_HELM,
           "fin": {"aspect_ratio": fin.aspect_ratio, "area": fin.area,
                   "sweep_le_deg": float(np.degrees(fin.sweep)),
                   "sweep_qc_deg": float(np.degrees(fin.quarter_chord_sweep)),
                   "control_effectiveness": fin.control_effectiveness}}
    log("at the default Munk factor 0.50")
    out["at_default"] = sweep_helm(law, reflection, 0.5)
    log("  authority 45: %.3f" % out["at_default"]["helm"]["45.0"]["authority"])
    log("refitting")
    munk, pts = fit_munk(law, reflection, log)
    out["fit_points"] = pts
    log("fitted %.4f; sweeping helm" % munk)
    out["fitted"] = sweep_helm(law, reflection, munk)
    out["fitted"]["skeg_lost_deg_25s"] = skeg_lost(law, reflection, munk)
    out["skeg_lost_at_default"] = skeg_lost(law, reflection, 0.5)
    (OUT / (tag + ".json")).write_text(json.dumps(out, indent=1))
    log("done")


def combine():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    print("%-10s %6s %7s %7s %7s %7s %7s %7s %7s %7s"
          % ("case", "munk", "a5", "a25", "a45", "25/5", "R25 m", "R45 m", "C", "lost"))
    for r in rows:
        tag = "legacy" if r["law"] == "legacy" else "r=%.2f" % r["reflection"]
        for label, d in (("0.50", r["at_default"]), ("fit", r["fitted"])):
            h = d["helm"]
            lost = d.get("skeg_lost_deg_25s", r.get("skeg_lost_at_default"))
            print("%-10s %6.3f %7.3f %7.3f %7.3f %7.2f %7.0f %7.0f %7.2f %7.1f"
                  % (tag, d["munk"], h["5.0"]["authority"], h["25.0"]["authority"],
                     h["45.0"]["authority"], d["ratio_25_5"], h["25.0"]["radius"],
                     h["45.0"]["radius"], d["C_millions"], lost))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflection", type=float, default=2.0)
    parser.add_argument("--law", default="whicker_fehlner",
                        choices=("legacy", "whicker_fehlner"))
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.combine:
        combine()
        sys.exit(0)
    study(args.law, args.reflection)
