"""Multi-clock body, step 4's first closed loop: rung 2's clock driven by joint muscles excited on the EMG modules (§256).

Rung 2 (rung2_clock.ClockSim, mode "free") makes the hands' time law an output: the stroke clock's acceleration solves
a virtual-work balance between the motion's needs (the body's inertia and weight along the clock, the oars' torque) and
the muscles' generalised force. Until now that force was a population's, borrowed as Q(tau) (rung2_population). Here it
is built from the body's own joint muscles:

    Q(tau, tau', t) = A * sum_g  m_g(phi(t - d)) * T_g,max(theta_g(tau), r_g(tau) tau') * r_g(tau)

* g: knee extensors (Anderson et al. 2007, KE), hip extensors (HE, passive term dropped), elbow flexors (Frey-Law et
  al. 2012); the joints' flexion angles and their rates along the clock from the model's own posture
  (muscle_capacity.geometry). r_g = d(angle)/d(tau) with the sign that makes the group's concentric work positive
  (extensors: the flexion falling; elbow flexors: rising). The torque generators carry the force-velocity, so no
  separate handle force-velocity or hold is applied (they would count it twice).
* m_g: the excitation modules on the cycle's phase (coxswain.crew.excitation: legs from Vieira et al. 2020 -- an
  **ergometer** source --, trunk extensors and arm pullers from Pitto et al. 2024 on water). The phase is time since the
  catch over the period (the oscillator with a full reset at each catch: rung 2 keeps the period).
* d: excitation to force, 0.075 s (the middle of the 50-100 ms used by the gate; activation 10/40 ms plus an
  electromechanical delay) -- placeholder, named.
* A: one amplitude per athlete (blocker B4's placeholder), set so the drive's muscle work equals the reference run's
  scaled to the athlete's measured power, as rung 2 scales its population.
* Angles beyond the strength data's tested range are held at the last tested angle (blocker B1's placeholder): the
  catch's deep knee and hip flexion would otherwise read zero strength.

The body is the model's (BODY "model": the research profile's clocks), so its postures are the ones the joint tables
come from. Nothing is fitted to an athlete except A. Reported: speed, IVV, power, the clock's rate range and overruns,
the oar's progress against the measured, and each group's torque peak against its EMG peak (the gate: 50-100 ms after).

    python research/crew/rung2_modules.py [--athletes cr06 br24] [--strokes 10] [--delay 0.075]
"""
import argparse
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "biorow"), os.path.join(ROOT, "research", "blade")]
import muscle_capacity as MC                                     # noqa: E402
import rung2_population as RP                                    # noqa: E402

from coxswain.crew import joint_muscles as J                     # noqa: E402
from coxswain.crew.excitation import Modules                     # noqa: E402

# (key, joint, module, sign): sign makes the group's concentric work positive (extensors: flexion falling)
AGONISTS = (("knee", "knee", "leg_extensors", -1.0), ("hip", "hip", "trunk_extensors", -1.0),
            ("elbow", "elbow", "arm_pullers", +1.0))
# the antagonists Pitto et al. (2024) record firing in the drive: the trunk flexors (rectus abdominis, obliques; peak
# at 88 % of the drive, braking the layback) and the arm extensor (triceps long head; 35 %, holding the arms straight).
# Off by default until tested (§257); under B4's placeholder they take the same amplitude as every other module.
ANTAGONISTS = (("hip_flex", "hip", "trunk_flexors", +1.0), ("elbow_ext", "elbow", "arm_extensors", -1.0))
GROUPS = AGONISTS
TESTED = {"knee": (0.0, 1.8), "hip": (-0.5, 1.3), "elbow": (0.0, 2.6)}   # rad, flexion + (Anderson; Frey-Law)


class Strength:
    """Each group's maximal torque at (angle, speed), with B1's clamp on the tested angles. Eccentric speeds: Anderson
    et al.'s eq. 8 at the hip and knee; the elbow held at isometric (Frey-Law has no eccentric cells)."""

    def __init__(self, name, mass, stature):
        self.gen = {"knee": J.anderson("KE", MC.GROUP[name], mass, stature),
                    "hip": J.anderson("HE", MC.GROUP[name], mass, stature, passive=False),
                    "elbow": J.elbow("flexion", MC.SEX[name]),
                    "hip_flex": J.anderson("HF", MC.GROUP[name], mass, stature, passive=False),
                    "elbow_ext": J.elbow("extension", MC.SEX[name])}
        self.joint = {k: j for k, j, _m, _s in AGONISTS + ANTAGONISTS}

    def __call__(self, g, theta, omega):
        lo, hi = TESTED[self.joint[g]]
        return float(np.asarray(self.gen[g].max_active(np.clip(theta, lo, hi), omega)).ravel()[0])


def tables(boat):
    P = float(boat.timing.period)
    grid, th, w = MC.geometry(boat.crew[0].rower, P)
    return P, grid, th, w


def module_force(tau, td, c, P, grid, th, w, strength, mods, delay, amp, parts=None, gscale=None):
    phase = ((c - delay) / P) % 1.0
    q = 0.0
    for g, joint, mod, sign in GROUPS:
        theta = float(np.interp(tau % P, grid, th[joint], period=P))
        r = sign * float(np.interp(tau % P, grid, w[joint], period=P))   # d(angle)/d(tau), concentric positive
        torque = float(mods(mod, phase)) * strength(g, theta, r * td) * float((gscale or {}).get(g, 1.0))
        if parts is not None:
            parts[g] = torque
        q += torque * r
    return amp * q


#: The joint demands the calibration matches: rung 2's stroke as the muscle check logs it (muscle_capacity.py --rung2;
#: knee and hip extensor-positive, elbow flexor-positive, N m, against time from the catch)
DEMANDS = os.path.join(ROOT, "data", "local", "crew", "muscle_capacity_rung2_s253.json")
#: each joint's demand convention: +1 where a group's torque adds to the demand, -1 where it opposes it
DEMAND_KEY = {"knee": "knee", "hip": "hip_ext", "elbow": "elbow"}
DEMAND_SIGN = {"knee": +1.0, "hip": +1.0, "elbow": +1.0, "hip_flex": -1.0, "elbow_ext": -1.0}


def calibrate(name, P, delay, mods, strength, path=DEMANDS):
    """EMG-assisted calibration (Lloyd & Besier 2003; Sartori et al. 2012, CEINMS; SOURCES §261): one non-negative
    gain per group so that the module-shaped torques reproduce the stroke's inverse-dynamics joint moments,

        M_j(t) ~ sum_{g at j} s_g * a_g * m_g(phi(t - d)) * T_g,max(theta_j(t), sigma_g theta_dot_j(t)),

    solved by non-negative least squares joint by joint over the logged stroke. The EMG fixes each group's *timing*;
    the gains fix each group's *level*, including the antagonists' (which a peak-normalised EMG cannot). Fitted to the
    model's own inverse dynamics, not to an athlete's measurement. Returns ({group: gain}, {joint: r^2})."""
    from scipy.optimize import nnls
    s = {k: np.asarray(v, float) for k, v in json.load(open(path))[name]["series"].items()}
    t = s["t"]
    phase = ((t - delay) / P) % 1.0
    gains, fit = {}, {}
    for joint in ("knee", "hip", "elbow"):
        y = s[DEMAND_KEY[joint]]
        cols, names = [], []
        for g, j, mod, sign in GROUPS:
            if j != joint:
                continue
            T = np.array([strength(g, th, sign * w) for th, w in zip(s["th_" + joint], s["w_" + joint])])
            cols.append(DEMAND_SIGN[g] * mods(mod, phase) * T)
            names.append(g)
        ok = np.isfinite(y) & np.all(np.isfinite(cols), axis=0)
        A = np.column_stack(cols)[ok]
        x, _ = nnls(A, y[ok])
        pred = A @ x
        fit[joint] = float(1.0 - np.sum((y[ok] - pred) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
        gains.update({n: float(v) for n, v in zip(names, x)})
    return gains, fit


def athlete(name, strokes, delay, iterate_power=0, amps=None, calibrated=False):
    """``iterate_power``: re-run that many times, rescaling the amplitude by measured / model power each time (B4's
    placeholder made to hit the power it is defined by). ``amps``: per-group multipliers on the amplitude.
    ``calibrated``: the per-group multipliers from :func:`calibrate` (their ratios kept; the amplitude still sets the
    level to the measured power)."""
    if calibrated:
        m = RP.V.ATHLETES[name][0]()
        b = RP.athlete_boat(name, m)
        P = float(b.timing.period)
        mass = float(np.sum(np.asarray(b.crew_field(0.0)[0], float)))
        st = Strength(name, mass, float(b.crew[0].rower.anthropometry.stature))
        gains, fit = calibrate(name, P, delay, Modules.pitto2024(), st)
        top = max(gains.values())
        amps = {g: v / top for g, v in gains.items()}
        print("%s: calibrated gains %s; joint-moment fit r^2 %s" % (
            name, {g: round(v, 3) for g, v in gains.items()}, {j: round(v, 2) for j, v in fit.items()}), flush=True)
    out = _athlete(name, strokes, delay, amps=amps)
    if calibrated:
        out["calibration"] = dict(gains=gains, r2=fit)
    for _ in range(int(iterate_power)):
        meas, got = out["measured"]["power"], out["free"]["power"]
        cal = out.get("calibration")
        out = _athlete(name, strokes, delay, amp_override=out["amp"] * meas / got, amps=amps)
        if cal:
            out["calibration"] = cal
    return out


def _athlete(name, strokes, delay, amp_override=None, amps=None):
    m = RP.V.ATHLETES[name][0]()
    boat = RP.athlete_boat(name, m)
    ref = RP.make(boat, "reference")
    run_r = ref.run_strokes(strokes, surge_speed=4.3)
    s_r = RP.C2.summary(run_r)
    P, grid, th, w = tables(boat)
    D = float(boat.timing.drive_fraction) * P
    mass = float(np.sum(np.asarray(boat.crew_field(0.0)[0], float)))
    stature = float(boat.crew[0].rower.anthropometry.stature)
    strength = Strength(name, mass, stature)
    mods = Modules.pitto2024()
    # the amplitude: the drive's muscle work equal to the reference's, scaled to the athlete's measured power
    start = float(run_r.last_time[0])
    lg = np.array([(c, q) for t, c, q in ref.qlog if t >= start - 1e-9 and 0.0 <= c <= D])
    lg = lg[np.argsort(lg[:, 0])]
    e_ref = float(np.trapezoid(lg[:, 1], lg[:, 0]))
    cs = np.linspace(0.0, D, 400)
    e_mod = float(np.trapezoid([module_force(c, 1.0, c, P, grid, th, w, strength, mods, delay, 1.0, gscale=dict(amps or {})) for c in cs], cs))
    amp = e_ref / e_mod * (m["power"] / s_r["power"]) if amp_override is None else float(amp_override)
    gscale = dict(amps or {})
    free_boat = RP.athlete_boat(name, m)
    free = RP.make(free_boat, "free", Q=lambda tau: 0.0)
    free.fv, free.hold = None, 0.0
    log = []

    def muscle(self, tau, td):
        c = float(getattr(self, "_t_now", 0.0)) - self._stroke_start
        parts = {}
        q = module_force(tau, td, c, P, grid, th, w, strength, mods, delay, amp, parts, gscale)
        if c >= 0.0:
            log.append((c, tau, td) + tuple(parts[g[0]] for g in GROUPS))
        return q
    free._muscle = types.MethodType(muscle, free)
    orig = free.derivative

    def derivative(t, y):
        free._t_now = t
        return orig(t, y)
    free.derivative = derivative
    free.start_state = np.concatenate([run_r.last_states[:, -1], [P, 1.0]])
    run_f = free.run_strokes(strokes, surge_speed=4.3)
    s_f = RP.C2.summary(run_f)
    ht, hs, _ha = RP.measured_track(name)
    tt, ss, _aa = RP.model_track(free, run_f, True)
    prog = RP.compare(tt, ss, ht, hs)
    tl = np.array(free.tlog)
    last = tl[tl[:, 0] >= float(run_f.last_time[0])]
    over = sum(1 for r in free.rlog if r[4] != "ok")
    # each group's torque peak in the last stroke's drive, against its EMG module's peak (fractions of the drive)
    L = np.array(log)
    out = dict(name=name, amp=amp, group_scales=gscale, reference=s_r, free=s_f, progress_rms=prog[0],
               progress_max=prog[1],
               tau_dot=[float(last[:, 2].min()), float(last[:, 2].max())] if last.size else None, overruns=over,
               measured=dict(speed=m["speed"], ivv=m["ivv"], power=m["power"]))
    if last.size:
        # the clock through the last free drive: where the rate peaks, as drive progress (the spike diagnosis, §257)
        sw, tm = free_boat.oar_sweep, free_boat.timing
        prog_l = [(sw.catch_angle - float(sw(float(r[1]), tm))) / (sw.catch_angle - sw.finish_angle) for r in last]
        k = int(np.argmax(last[:, 2]))
        out["clock_trace"] = dict(t=(last[:, 0] - float(run_f.last_time[0])).tolist(), tau_dot=last[:, 2].tolist(),
                                  tau_ddot=last[:, 3].tolist(), progress=prog_l)
        out["clock_peak"] = dict(tau_dot=float(last[k, 2]), progress=float(prog_l[k]),
                                 t=float(last[k, 0] - float(run_f.last_time[0])))
    if L.size:
        # the last stroke's samples: the log's clock restarts at each catch; take the final run of increasing c
        k0 = int(np.flatnonzero(np.diff(L[:, 0]) < -0.5)[-1] + 1) if np.any(np.diff(L[:, 0]) < -0.5) else 0
        Ls = L[k0:]
        drive_end = float(np.max(Ls[:, 0][Ls[:, 1] <= D])) if np.any(Ls[:, 1] <= D) else D
        timing = {}
        for j, (g, _joint, mod, _s) in enumerate(GROUPS):
            on = Ls[:, 0] <= drive_end
            kpk = int(np.argmax(np.abs(Ls[on, 3 + j])))
            emg = mods.peak_phase(mod) * P
            timing[g] = dict(torque_peak_s=float(Ls[on, 0][kpk]), emg_peak_s=float(emg),
                             lag_ms=float(1000 * (Ls[on, 0][kpk] - emg)))
        out["timing"] = timing
        out["drive_s"] = drive_end
    print("%s: amplitude %.3f; reference %.3f m/s, IVV %.1f %%, %.0f W" % (name, amp, s_r["speed"], 100 * s_r["ivv"], s_r["power"]))
    print("%s: MODULE-DRIVEN %.3f m/s, IVV %.1f %%, %.0f W; clock rate %s; overruns %d; oar progress rms %.3f (max %.3f)"
          % (name, s_f["speed"], 100 * s_f["ivv"], s_f["power"], out["tau_dot"], over, prog[0], prog[1]))
    print("%s: measured %.3f m/s, IVV %.1f %%, %.0f W" % (name, m["speed"], 100 * m["ivv"], m["power"]))
    for g, v in out.get("timing", {}).items():
        print("   %-5s torque peak %.3f s, EMG peak %.3f s: torque %+.0f ms after the EMG (gate 50-100)"
              % (g, v["torque_peak_s"], v["emg_peak_s"], v["lag_ms"]))
    if "clock_peak" in out:
        cp = out["clock_peak"]
        print("   the clock's rate peaks at %.2f, %.3f s into the stroke, at %.0f %% of the oar's drive progress"
              % (cp["tau_dot"], cp["t"], 100 * cp["progress"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athletes", nargs="+", default=["cr06", "br24"])
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--delay", type=float, default=0.075, help="excitation to force, s (placeholder)")
    ap.add_argument("--iterate-power", type=int, default=0, help="rescale the amplitude to the measured power N times")
    ap.add_argument("--group-scales", nargs=3, type=float, default=None, metavar=("KNEE", "HIP", "ELBOW"),
                    help="per-group multipliers on the amplitude (B4)")
    ap.add_argument("--antagonists", action="store_true",
                    help="add the trunk flexors and the arm extensor (Pitto et al. 2024 modules), same amplitude")
    ap.add_argument("--calibrate", action="store_true",
                    help="per-group gains from the stroke's joint demands (EMG-assisted calibration, §261)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global GROUPS
    if a.antagonists:
        GROUPS = AGONISTS + ANTAGONISTS
        a.tag = "_antag" + a.tag
    if a.calibrate:
        a.tag = "_cal" + a.tag
    # rung 2's research configuration, the body the model's own (its postures are the joint tables')
    RP.BODY, RP.BLADE = "model", "crossflow_normal"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    RP.ADDED_MASS, RP.SURFACE_FACTOR = "patton", 0.1
    RP.BENDING = True
    from coxswain.crew.oar_bending import BendDynamics
    RP.BEND_DYNAMICS = BendDynamics()
    RP.FV, RP.HOLD = None, 0.0
    amps = dict(zip(("knee", "hip", "elbow"), a.group_scales)) if a.group_scales else None
    res = {n: athlete(n, a.strokes, a.delay, a.iterate_power, amps, calibrated=a.calibrate) for n in a.athletes}
    out = os.path.join(ROOT, "data", "local", "crew", "rung2_modules_delay%g%s%s.json"
                       % (a.delay, "_pow%d" % a.iterate_power if a.iterate_power else "", a.tag))
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
