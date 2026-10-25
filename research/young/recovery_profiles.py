"""Young (2009), redone: does the recovery's velocity profile change shell speed? (sprint 2, S2.4)

    .venv/Scripts/python.exe research/young/recovery_profiles.py <profile> [--rates 20 28 36]
    .venv/Scripts/python.exe research/young/recovery_profiles.py report

    profiles: model (the simulator's own recovery), normal, constant, fast_pivot, slow_pivot
    Run each profile in its own process (the study swaps the recovery retiming for the whole process).

Question. Young, "Effects of Various Inefficiencies in Rowing on Shell Speed" (BSc thesis, MIT, 2009; [Y09],
data/local/literature/young2009/), ran three recovery profiles through a Simulink model of a heavyweight eight at
20-36 spm and found "little to no effect" on shell speed, with getting "the hands and body out of bow as fast as
possible" slowing the shell "a fraction of a percent" (his Fig. 11, p. 18). His model is a point-mass crew on a
slide, a constant-coefficient hull (R = C v^2 S, wave drag absorbed into C) and a prescribed blade force. Ours has a
segmented body on a joint chain, a 6-DOF hull with Michell wave drag, and a dynamic oar with a slip blade. Does his
result survive?

Profiles (his recovery.m and shell.m, pp. 28 and 31-32: T_rec normalised to 1, p_ave = 0.1, p_max = 0.4, slide speed
in units of the mean L_slide / T_rec; each integrates to exactly one slide length):
    normal       0 -> 1 over [0, 0.1], -> 1.25 at 0.4, -> 1 at 0.9, -> 0 at 1   (his 1st profile, "Normal")
    constant     0 -> 1.111 over [0, 0.1], held to 0.9, -> 0 at 1               (his 2nd profile)
    fast_pivot   0 -> 2 over [0, 0.1], then linearly -> 0 at 1                 (his 3rd profile, "Fast Pivot")
    slow_pivot   0 -> 2 linearly over [0, 0.9], -> 0 at 1                      (his 4th profile, "Slow Pivot")
    model        the simulator's own (ergometer-derived) recovery, no retiming
The seat is made to follow Young's profile exactly: the recovery is retimed by w(s) = P^-1(p_Y(s)), where P is the
seat's travel fraction against recovery progress on the model's own clock and p_Y Young's. The whole body is retimed
with it (Young's crew is one mass on the slide). The oar's blade-out recovery uses the same retiming.

Method. The research profile's heavyweight eight (catalog defaults: 88 kg, 1.90 m rowers), driven straight, settled
at a stated handle power per rower by `scorecard.settle_dynamic` (torque matched under the sweep catch, power
measured on the run). Young holds work per stroke constant across rates (his eq. 30-33), so the power per rower is
712.5 J per stroke (380 W at 32 spm): 237.5 W at 20, 332.5 W at 28, 427.5 W at 36. Each profile and rate reports
settled speed, surge swing, measured crew power, and the speed at equal measured power.

Assumptions: the research profile as it stands (ergometer-derived drive kinematics); the profile changes only the
recovery's timing, not its postures; Young's profiles are applied as printed (their corners are smoothed by the
body's Fourier refit).

Outputs: data/local/young/recovery_<profile>.json; report -> data/local/young/recovery_profiles.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CODE))
sys.path.insert(0, ROOT)
HERE = os.path.join(ROOT, "data", "local", "young")
os.makedirs(HERE, exist_ok=True)

from coxswain import physics                                    # noqa: E402
from coxswain.boats import catalog                              # noqa: E402
from coxswain.crew import stroke                                # noqa: E402
from coxswain.validation import scorecard                       # noqa: E402

WORK_PER_STROKE = 380.0 * 60.0 / 32.0                           # J per rower per stroke
START = {20: 4.4, 28: 5.0, 36: 5.5}
#: a distinct recovery_arrival per profile: it switches the retiming on and keeps every cache key distinct
SENTINEL = {"normal": 0.99991, "constant": 0.99992, "fast_pivot": 0.99993, "slow_pivot": 0.99994}


def young_velocity(name, s):
    """Young's slide speed (units of the mean) at normalised recovery time s."""
    s = np.asarray(s, float)
    if name == "normal":
        return np.interp(s, [0, 0.1, 0.4, 0.9, 1.0], [0, 1.0, 1.25, 1.0, 0])
    if name == "constant":
        return np.interp(s, [0, 0.1, 0.9, 1.0], [0, 1.0 / 0.9, 1.0 / 0.9, 0])
    if name == "fast_pivot":
        return np.interp(s, [0, 0.1, 1.0], [0, 2.0, 0])
    if name == "slow_pivot":
        return np.interp(s, [0, 0.9, 1.0], [0, 2.0, 0])
    raise ValueError(name)


def young_position(name, n=2001):
    s = np.linspace(0.0, 1.0, n)
    v = young_velocity(name, s)
    p = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(s))])
    return s, p / p[-1]


def seat_travel(rate):
    """The model's own seat travel fraction against recovery progress (uniform clock), monotone."""
    boat = catalog.eight(rate=rate)
    rower = boat.crew[0].rower
    P = float(boat.timing.period); d = float(boat.timing.drive_fraction)
    u = np.linspace(0.0, 1.0, 801)
    x = np.array([rower.joint_positions((d + (1.0 - d) * k) * P)["hip"][0] for k in u])
    f = (x - x[0]) / (x[-1] - x[0])
    f = np.maximum.accumulate(f) + 1e-9 * u                      # strictly increasing for the inverse
    return u, (f - f[0]) / (f[-1] - f[0])


def install(name, rate):
    """Replace the recovery retiming for this process with Young's profile, mapped through the model's seat."""
    u, P = seat_travel(rate)
    s, pY = young_position(name)
    w = np.interp(pY, P, u)                                     # w(s) = P^-1(p_Y(s))
    dw = np.gradient(w, s)

    def warp(progress, arrival):
        return np.interp(progress, s, w)

    def slope(progress, arrival):
        return np.interp(progress, s, dw)

    stroke.recovery_warp = warp
    stroke.recovery_warp_slope = slope
    return s, w


def run(name, rates):
    research = physics.resolve("research")
    out = dict(profile=name, work_per_stroke=WORK_PER_STROKE, rates={})
    for rate in rates:
        if name != "model":
            install(name, rate)
        boat = catalog.eight(rate=rate, recovery_arrival=SENTINEL.get(name, 1.0))
        research.apply(boat)
        watts = WORK_PER_STROKE * rate / 60.0
        got = scorecard.settle_dynamic(boat, watts, START[rate])
        rowers = sum(1 for seat in boat.rig.seats if seat.oarlocks)
        r = dict(rate=rate, watts_stated=watts, speed=got.speed, surge_swing=got.surge_swing,
                 crew_power_per_rower=got.crew_power / rowers, drag_power=got.drag_power,
                 blade_efficiency=got.blade_efficiency)
        out["rates"][str(rate)] = r
        print("%-11s %2d spm: %.4f m/s, swing %.3f, measured %.1f W/rower (stated %.1f), blade eff %s" % (
            name, rate, got.speed, got.surge_swing, r["crew_power_per_rower"], watts,
            "%.3f" % got.blade_efficiency if got.blade_efficiency is not None else "-"), flush=True)
        json.dump(out, open(os.path.join(HERE, "recovery_%s.json" % name), "w"), indent=1, default=float)


def report():
    rows = {}
    for f in glob.glob(os.path.join(HERE, "recovery_*.json")):
        if f.endswith("recovery_profiles.json"):
            continue
        d = json.load(open(f)); rows[d["profile"]] = d["rates"]
    if "constant" not in rows:
        raise SystemExit("run the 'constant' profile first: it is the reference")
    print("profile      | rate | speed m/s | vs constant | at equal power* | swing")
    out = {}
    for name in ("model", "normal", "constant", "fast_pivot", "slow_pivot"):
        if name not in rows:
            continue
        for rate, r in sorted(rows[name].items(), key=lambda kv: int(kv[0])):
            ref = rows["constant"].get(rate)
            if ref is None:
                continue
            dv = 100 * (r["speed"] / ref["speed"] - 1)
            # *equal power: correct for any mismatch in measured power with the local cube-root-like law
            dv_p = 100 * ((r["speed"] / ref["speed"]) * (ref["crew_power_per_rower"] / r["crew_power_per_rower"]) ** (1 / 3) - 1)
            out.setdefault(name, {})[rate] = dict(r, dv_pct=dv, dv_equal_power_pct=dv_p)
            print("%-12s | %4s | %.4f | %+6.2f%% | %+6.2f%% | %.3f" % (name, rate, r["speed"], dv, dv_p, r["surge_swing"]))
    print("\nYoung (2009, Fig. 11): little to no effect; fast pivot slower by 'a fraction of a percent'.")
    json.dump(out, open(os.path.join(HERE, "recovery_profiles.json"), "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", choices=("model", "normal", "constant", "fast_pivot", "slow_pivot", "report", "check"))
    ap.add_argument("--rates", type=int, nargs="+", default=[20, 28, 36])
    a = ap.parse_args()
    if a.profile == "report":
        report()
    elif a.profile == "check":                                  # print each profile's retiming at a few points
        for name in ("normal", "constant", "fast_pivot", "slow_pivot"):
            s, w = install(name, 28)
            print(name, " ".join("%.2f->%.3f" % (x, np.interp(x, s, w)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)))
    else:
        run(a.profile, a.rates)


if __name__ == "__main__":
    main()
