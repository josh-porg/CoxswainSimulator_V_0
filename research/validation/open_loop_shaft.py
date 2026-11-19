"""The blade and the shaft alone, on each athlete's measured motion: their handle force predicted open loop (§237).

No hull, no body, no muscles: the athlete's measured oar angle (the gate's) and boat speed are imposed, and the
simulator's own blade loads (strips, the research cross-flow law, BioRow's 4 deg wetting; ``_blade_loads_arm_rigid``,
the one load path) give the blade's normal force; the shaft is rigid, or [L14]'s measured scull shaft as the
quasi-static hinge (K phi = arm F_n, solved implicitly per step as ``DynamicOarSimulator._update_bending`` does; with
the blade's water at kappa 0.1 the dynamic shaft is within 0.2 deg of it, §228). The handle force is the torque
(blade's moment minus the oar's inertia) over the lever the measurement uses: [BR24] BioRow's 0.06 m from the handle's
top; [CR06] her paper's force point. The blade enters when its normal velocity through the water turns driving
([CR06] eq. 16) and leaves when it stops driving in the second half of the arc.

    python research/validation/open_loop_shaft.py [--athletes br24 cr06]

Prints the force per 10 % of the drive's time against the measured, its rms as a fraction of the measured peak, the
peak's time, and the impulse ratio.
"""
import argparse
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"),
                os.path.join(ROOT, "research", "blade")]
import rung2_population as RP                                    # noqa: E402

from coxswain.core.state import STATE_SIZE, State                # noqa: E402

N = 2400                                                          # samples per cycle


def measured(name):
    """(period, t grid, gate angle rad [catch positive, falling through the drive], rate, accel, boat speed,
    measured handle force per oar N, the measurement's lever m), the cycle starting at the catch."""
    if name == "br24":
        D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"), delimiter=",", names=True)[:-1]
        T = 60.0 / 32.4093
        n = len(D)
        t = np.arange(n) * T / n
        per = lambda y: make_smoothing_spline(np.concatenate([t - T, t, t + T]), np.concatenate([y, y, y]))  # noqa: E731
        ang = per(-np.radians(0.5 * (D["A1"] + D["A2"])))
        vel = per(D["Vs"])
        frc = per(0.5 * (D["H1"] + D["H2"]))
        lever = 0.875 - 0.06                                      # BioRow: the torque as a force 0.06 m from the top
    else:
        V = RP.V
        T = V.CR06_T
        g = np.linspace(0.0, T, 400, endpoint=False)
        per = lambda f: CubicSpline(np.append(g, T), np.append(f(g), f(g)[0]), bc_type="periodic")           # noqa: E731
        ang = per(lambda x: np.radians(V._cr06_series("oar_angle_deg")(x)))
        vel = per(V._cr06_series("boat_velocity_mps"))
        frc = per(lambda x: 0.5 * V._cr06_series("handle_force_N")(x))   # Fig. 3 sums both hands
        lever = V.CR06_FORCE_POINT
    fine = np.linspace(0.0, T, N, endpoint=False)
    t0 = fine[int(np.argmax(ang(fine)))]                          # the catch: the gate's largest angle
    tt = fine + t0
    return T, fine, ang(tt), ang(tt, 1), ang(tt, 2), vel(tt), frc(tt), lever


def predict(name, shaft, ei_scale=1.0, shift=0.0, inertia_value=None):
    """The handle force through the cycle on the athlete's measured motion; ``shaft`` "rigid", "hinge", or "locked":
    the hinge with a blade that does not slip at all (its normal velocity through the water held at zero from the
    entry, the shaft already bent there by the oar's own inertia under the hands' turn-round), which is the most
    force a shaft of that stiffness can carry on that motion whatever the blade's law -- a bound on the inputs, not
    a model (SOURCES §238). ``ei_scale``: the shaft's stiffness over [L14]'s medium; ``shift`` (s): the boat-speed
    channel delayed by this against the oar angle; ``inertia_value`` (kg m^2): the oar's inertia about the pin in
    place of the model's (sensitivities)."""
    RP.BLADE, RP.BODY = "crossflow_normal", "own"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    RP.ADDED_MASS, RP.SURFACE_FACTOR, RP.BEND_DYNAMICS = "none", 0.1, None
    RP.BENDING, RP.RIGID = shaft != "rigid", shaft == "rigid"
    m = RP.V.ATHLETES[name][0]()
    boat = RP.athlete_boat(name, m)
    sim = RP.make(boat, "reference")
    T, t, ang, rate, acc, v, f_meas, lever = measured(name)
    if shift:
        v = np.interp((t - float(shift)) % T, t, v, period=T)
    out = kinematic(sim, shaft, t, ang, rate, acc, lever, speeds=v, ei_scale=ei_scale, inertia_value=inertia_value)
    return dict(out, T=T, t=t, measured=f_meas, angle=ang, lever=lever)


def kinematic(sim, shaft, t, ang, rate, acc, lever, speeds=None, states=None, ei_scale=1.0, inertia_value=None):
    """The blade and shaft alone on an imposed motion (the one load path, ``_blade_loads_arm_rigid``): ``t`` from the
    catch (``ang[0]`` its angle, the largest), gate ``ang`` / ``rate`` / ``acc``, and the hull as ``speeds`` (surge
    only) or as ``states`` (a full ``State`` per sample, e.g. a closed-loop run's own; SOURCES §240). Returns
    ``force`` (handle force on ``lever``), ``bend``, ``wet`` and ``finish``. With ``shaft`` "hinge" the shaft is the
    quasi-static hinge solved implicitly per sample, and it unbends at once when the blade leaves."""
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    oar = sim._oars[0]
    inertia = float(getattr(oar.inertia, "oar_inertia", 0.0)) if inertia_value is None else float(inertia_value)
    n_s = len(t)
    dt = float(t[1] - t[0])
    k = float(sim.oar_bending.k_ang) * float(ei_scale) if shaft != "rigid" else None
    if shaft != "rigid":
        arm0 = float(oar.outboard)
        blever = (arm0 - sim.oar_bending.hinge_radius(arm0, float(lock.oar.blade_length))) / arm0
    span = ang.max() - ang.min()
    finish = int(np.argmin(ang))
    wet, phi, done = False, 0.0, False
    sim._entry_angle = np.array([np.nan])
    force = np.zeros(n_s)
    bend = np.zeros(n_s)
    wet_mask = np.zeros(n_s, dtype=bool)
    for i in range(n_s):
        state = (states[i] if states is not None
                 else State.from_vector(sim.augmented_initial_state(float(speeds[i]))[:STATE_SIZE]))
        a, r = float(ang[i]), float(rate[i])
        moment = 0.0
        if i <= finish and not done:
            if not wet and i > 0 and sim._normal_velocity(0, a, r, state) <= 0.0:
                wet = True                                        # entry at zero normal velocity ([CR06] eq. 16)
                sim._entry_angle[0] = a
                if shaft == "locked":                             # the shaft arrives bent by the oar's inertia
                    phi = max(0.0, -inertia * float(acc[i]) / k)
            if wet:
                if shaft == "rigid":
                    f_n, _ft, arm = sim._blade_loads_arm_rigid(0, a, r, state, lock)
                    moment = arm * f_n
                elif shaft == "locked":
                    slip = lambda pd: sim._normal_velocity(0, a + phi + dt * pd, r + blever * pd, state)  # noqa: E731
                    lo, hi = -60.0, 60.0
                    pd = brentq(slip, lo, hi, xtol=1e-7) if slip(lo) * slip(hi) < 0.0 else 0.0
                    phi = phi + dt * pd
                    moment = k * phi
                    if (ang[0] - a) / span > 0.5 and phi <= 0.0:
                        wet, done, phi, moment = False, True, 0.0, 0.0
                else:
                    def g(pd):
                        f_n, _ft, arm = sim._blade_loads_arm_rigid(0, a + phi + dt * pd, r + blever * pd, state, lock,
                                                                   gate=(a, r))
                        return k * (phi + dt * pd) - arm * f_n
                    lo, hi = -phi / dt - 60.0, 60.0
                    glo, ghi = g(lo), g(hi)
                    pd = brentq(g, lo, hi, xtol=1e-7) if glo * ghi < 0.0 else (lo if abs(glo) < abs(ghi) else hi)
                    phi = phi + dt * pd
                    moment = k * phi
                    wn_b = sim._normal_velocity(0, a + phi, r + blever * pd, state)
                    if (ang[0] - a) / span > 0.5 and wn_b >= 0.0:
                        wet, done, phi, moment = False, True, 0.0, 0.0
                if shaft == "rigid" and (ang[0] - a) / span > 0.5 and sim._normal_velocity(0, a, r, state) >= 0.0:
                    wet, done, moment = False, True, 0.0
        bend[i] = phi
        wet_mask[i] = wet
        # a locked blade holds the outboard oar, so its inertia no longer loads the handle separately
        force[i] = moment / lever if (shaft == "locked" and wet) else (moment - inertia * acc[i]) / lever
    return dict(force=force, bend=bend, finish=finish, wet=wet_mask)


def report(name, runs):
    r0 = next(iter(runs.values()))
    fin = r0["finish"]
    tf = r0["t"][:fin + 1] / r0["t"][fin]
    fm = r0["measured"][:fin + 1]
    peak = float(fm.max())
    print("\n%s: handle force per oar on the measured motion, N (drive %.2f s; measured peak %.0f N at %.0f %%; lever "
          "%.3f m)" % (name, r0["t"][fin], peak, 100 * tf[np.argmax(fm)], r0["lever"]))
    print(" time | measured | " + " | ".join("%-9s" % s for s in runs) + " | bend deg")
    for q in np.arange(0.0, 1.01, 0.1):
        i = int(np.argmin(abs(tf - q)))
        print(" %3.0f%% |  %6.0f  | " % (100 * q, fm[i]) + " | ".join("%9.0f" % r["force"][i] for r in runs.values())
              + " |  %5.2f" % np.degrees(list(runs.values())[-1]["bend"][i]))
    out = {}
    for s, r in runs.items():
        f = r["force"][:fin + 1]
        rms = float(np.sqrt(np.mean((f - fm) ** 2)))
        out[s] = dict(rms_frac=rms / peak, peak=float(f.max()), peak_time=float(100 * tf[np.argmax(f)]),
                      impulse=float(np.trapezoid(f, tf) / np.trapezoid(fm, tf)))
        print("%-6s rms %.1f %% of the measured peak; peak %.0f N at %.0f %% of the drive's time; impulse ratio %.2f"
              % (s, 100 * out[s]["rms_frac"], out[s]["peak"], out[s]["peak_time"], out[s]["impulse"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athletes", nargs="+", default=["br24", "cr06"])
    ap.add_argument("--bound", action="store_true",
                    help="the no-slip bound instead: the locked blade at shaft stiffness x 1, 1.5, 2, with the "
                         "boat-speed channel shifted +-20 ms, and with an oar inertia of 2.2 kg m^2 (nothing is written)")
    ap.add_argument("--ei-scale", type=float, default=1.0,
                    help="the flexible shaft's EI over [L14]'s medium for the ladder's C2 file (a what-if, §248)")
    ap.add_argument("--ei-scales", nargs="+", type=float, default=None,
                    help="a stiffness sensitivity instead: the flexible shaft at these multiples of [L14]'s medium EI, "
                         "beside rigid (§245; nothing is written)")
    a = ap.parse_args()
    if a.ei_scales:
        for name in a.athletes:
            runs = {"rigid": predict(name, "rigid")}
            for s in a.ei_scales:
                runs["EI x%g" % s] = predict(name, "hinge", ei_scale=s)
            report(name, runs)
        return
    import json
    res = {}
    for name in a.athletes:
        if a.bound:
            runs = {"hinge": predict(name, "hinge"), "locked": predict(name, "locked")}
            for s in (1.5, 2.0):
                runs["lock x%g" % s] = predict(name, "locked", ei_scale=s)
            for d in (-0.02, 0.02):
                runs["lock%+.0fms" % (1e3 * d)] = predict(name, "locked", shift=d)
            runs["lock I2.2"] = predict(name, "locked", inertia_value=2.2)
            report(name, runs)
            continue
        runs = {"rigid": predict(name, "rigid"), "hinge": predict(name, "hinge", ei_scale=a.ei_scale)}
        res[name] = report(name, runs)
    if not a.bound:
        tag = "" if a.ei_scale == 1.0 else "_ei%g" % a.ei_scale    # a what-if never overwrites the ladder's input
        json.dump(res, open(os.path.join(ROOT, "data", "local", "validation", "open_loop_shaft%s.json" % tag), "w"),
                  indent=1)


if __name__ == "__main__":
    main()
