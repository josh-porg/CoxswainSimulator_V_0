"""Phase 4.3, rung 2's test (sprint 2, S2.5): the population's muscles on two athletes. Are the hands right?

    .venv/Scripts/python.exe research/crew/rung2_population.py [--strokes 10] [--athletes cr06 br24]

Rung 2a imposes [K05]'s on-water population drive law on the hands. Rung 2 (research/crew/rung2_clock.py) makes the
hands' time law an output: the stroke clock tau is solved from the body-and-oar virtual-work balance, driven by a
generalised muscle force Q, with a prescribed handover to the recovery just before the finish's turning point
(``rung2_clock.HANDOVER``) and the recovery retimed to keep the period.

The test, with nothing fitted:
  1. The population. [K05]'s crew (1.80 m, 72.2 kg, 32.3 spm, on-water rhythm) on [CR06]'s rig, with the population's
     body (research/crew/k05_body.py, legs at [K05]'s 0.51 m) and its drive law, run as rung 1 (tau = clock). Its
     inverse dynamics give Q_pop against drive progress s (the oar's fraction of its arc, catch 0 -> finish 1):
     the population's muscles, on the population's motion.
  2. Each athlete, on his or her own rig and rate with the model's body at their stature. Rung 1 with the population's
     drive law imposed is the baseline. Rung 2 drives the athlete with Q_pop at the same drive progress, scaled by the
     athlete's measured power over the population run's (equal work per stroke at their power). Posture, not cycle
     phase, carries the force across: by phase the population's late push landed after her blade had let go.
  3. The score: the athlete's measured oar angle as a fraction of the arc against time from the catch, rms over the
     drive (and in degrees for [CR06], whose rig the model shares); speed and IVV against theirs.

S2.5's gate: the athletes' IVV within about 5 points of measured, or stop and record.

Outputs: data/local/crew/rung2_population.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "biorow")]
import rung2_clock as C2                                         # noqa: E402
import k05_population_test as KP                                # noqa: E402
import k05_body                                                  # noqa: E402

from coxswain.crew.stroke import OnWaterTiming                  # noqa: E402
from coxswain.sim.control import Coxswain                        # noqa: E402

V = C2.R.V
OUT = C2.OUT


def make(boat, mode, Q=None, body=None):
    boat.power_scales = np.ones(boat.n_seats)
    sim = C2.ClockSim(boat, peak_torque=0.0, catch="sweep", crew="handle",
                      coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    sim.setup_clock(mode, Q, body=body)
    return sim


def q_table(sim, run):
    P = float(sim.boat.timing.period)
    start = float(run.last_time[0])
    lg = np.array([(c, q) for t, c, q in sim.qlog if t >= start])
    o = np.argsort(lg[:, 0]); c, q = lg[o, 0], lg[o, 1]
    keep = np.concatenate([[True], np.diff(c) > 1e-9])
    return c[keep] / P, q[keep]


def progress_of(boat, tau):
    sw, tm = boat.oar_sweep, boat.timing
    return (sw.catch_angle - np.array([float(sw(x, tm)) for x in np.atleast_1d(tau)])) / (sw.catch_angle - sw.finish_angle)


def model_track(sim, run, free):
    """The model's drive progress and oar angle (deg) against time from the catch, over the last stroke's drive."""
    t = run.last_time
    if free:
        tl = np.array(sim.tlog)
        tl = tl[tl[:, 0] >= t[0] - 1e-9]
        tt, tau = tl[:, 0] - t[0], tl[:, 1]
    else:
        tt = t - t[0]; tau = tt
        keep = tau <= float(sim.boat.timing.drive_fraction) * float(sim.boat.timing.period)
        tt, tau = tt[keep], tau[keep]
    sw, tm = sim.boat.oar_sweep, sim.boat.timing
    return tt, progress_of(sim.boat, tau), np.degrees([float(sw(x, tm)) for x in tau])


def measured_track(name):
    """The athlete's measured oar angle: (time from the catch, drive progress, angle deg) over the drive."""
    if name == "cr06":
        f = V._cr06_series("oar_angle_deg")
        T = V.CR06_T
        g = np.linspace(0.0, T, 2000, endpoint=False)
        a = np.array([float(f(x)) for x in g])
        i0 = int(np.argmax(a))                                   # her catch: the angle's maximum
    else:
        import like_for_like as L
        D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
        T = 60.0 / L.RATE
        t0 = np.arange(len(D)) * T / len(D)
        raw = 0.5 * (D["A1"] + D["A2"])
        g = np.linspace(0.0, T, 2000, endpoint=False)
        a = -np.interp(g, t0, raw, period=T)                     # his catch is his minimum: flip to the model's sense
        i0 = int(np.argmax(a))
    a2 = np.roll(a, -i0)
    tt = np.arange(a2.size) * T / a2.size
    j = int(np.argmin(a2))                                       # the finish: the angle's minimum
    a2, tt = a2[:j + 1], tt[:j + 1]
    s = (a2[0] - a2) / (a2[0] - a2[-1])
    return tt, s, a2


def compare(tt, x, ht, hx):
    end = min(tt[-1], ht[-1])
    grid = np.linspace(0.0, end, 200)
    m = np.interp(grid, tt, x); h = np.interp(grid, ht, hx)
    return float(np.sqrt(np.mean((m - h) ** 2))), float(np.max(np.abs(m - h)))


def athlete(name, strokes, s_pop, q_drive, ph, q, p_pop):
    m = V.ATHLETES[name][0]()
    ht, hs, ha = measured_track(name)
    out = dict(measured=dict(speed=m["speed"], ivv=m["ivv"], power=m["power"]))
    base_boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    base = make(base_boat, "reference")
    run_b = base.run_strokes(strokes, surge_speed=4.3)
    s_b = C2.summary(run_b)
    tt, ss, aa = model_track(base, run_b, False)
    prog_b = compare(tt, ss, ht, hs)
    ang_b = compare(tt, aa, ht, ha) if name == "cr06" else (float("nan"), float("nan"))
    out["rung1"] = dict(s_b, progress_rms=prog_b[0], progress_max=prog_b[1], angle_rms=ang_b[0], angle_max=ang_b[1])

    P = float(base_boat.timing.period)
    D = float(base_boat.timing.drive_fraction) * P
    k = m["power"] / p_pop

    def Q(tau):
        if tau <= D:
            return k * float(np.interp(float(progress_of(base_boat, tau)[0]), s_pop, q_drive))
        return k * float(np.interp(np.mod(tau / P, 1.0), ph, q, period=1.0))

    free_boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    free = make(free_boat, "free", Q=Q)
    free.start_state = np.concatenate([run_b.last_states[:, -1], [P, 1.0]])
    run_f = free.run_strokes(strokes, surge_speed=4.3)
    s_f = C2.summary(run_f)
    tt, ss, aa = model_track(free, run_f, True)
    prog_f = compare(tt, ss, ht, hs)
    ang_f = compare(tt, aa, ht, ha) if name == "cr06" else (float("nan"), float("nan"))
    tl = np.array(free.tlog); last = tl[tl[:, 0] >= float(run_f.last_time[0])]
    over = sum(1 for r in free.rlog if r[4] != "ok")
    out["rung2"] = dict(s_f, progress_rms=prog_f[0], progress_max=prog_f[1], angle_rms=ang_f[0], angle_max=ang_f[1],
                        scale=k, overruns=over, tau_dot_last=[float(last[:, 2].min()), float(last[:, 2].max())],
                        trace=dict(t=(last[:, 0] - float(run_f.last_time[0])).tolist(), tau_dot=last[:, 2].tolist()))
    for lab, s, pr, an in (("rung 1, population law imposed", s_b, prog_b, ang_b),
                           ("rung 2, population muscles x %.3f" % k, s_f, prog_f, ang_f)):
        print("%s, %s: %.3f m/s, IVV %.1f%%, %.0f W; oar progress vs measured: rms %.3f (max %.3f)%s" % (
            name, lab, s["speed"], 100 * s["ivv"], s["power"], pr[0], pr[1],
            "; angle rms %.1f deg (max %.1f)" % an if name == "cr06" else ""), flush=True)
    print("%s, measured: %.3f m/s, IVV %.1f%%, %.0f W; rung 2 drive tau' %.3f to %.3f, overruns %d" % (
        name, m["speed"], 100 * m["ivv"], m["power"], last[:, 2].min(), last[:, 2].max(), over), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--athletes", nargs="+", default=["cr06", "br24"])
    a = ap.parse_args()
    out = dict(handover=C2.HANDOVER)

    # 1. the population, rung 1 with its own body and drive law
    pop_boat = V.cr06_boat(dict(leg_travel=None), timing=OnWaterTiming(KP.K05_RATE), stature=KP.K05_STATURE,
                           drive_law=True, mass=KP.K05_MASS)
    body, _k = k05_body.body_field(pop_boat, leg_travel=0.51)
    pop = make(pop_boat, "reference", body=body)
    run_p = pop.run_strokes(a.strokes, surge_speed=4.3)
    s_p = C2.summary(run_p)
    ph, q = q_table(pop, run_p)
    P_pop = float(pop_boat.timing.period)
    on = ph * P_pop <= float(pop_boat.timing.drive_fraction) * P_pop
    s_pop = np.maximum.accumulate(progress_of(pop_boat, ph[on] * P_pop))
    q_drive = q[on]
    keep = np.concatenate([[True], np.diff(s_pop) > 1e-9])
    s_pop, q_drive = s_pop[keep], q_drive[keep]
    print("population (rung 1, [K05] body and law): %.3f m/s, IVV %.1f%%, %.0f W; Q_pop %.0f to %.0f"
          % (s_p["speed"], 100 * s_p["ivv"], s_p["power"], q.min(), q.max()), flush=True)
    out["population"] = s_p

    # 2. each athlete
    for name in a.athletes:
        out[name] = athlete(name, a.strokes, s_pop, q_drive, ph, q, s_p["power"])
    json.dump(out, open(os.path.join(OUT, "rung2_population.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
