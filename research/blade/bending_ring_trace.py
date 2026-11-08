"""Why does [BR24]'s rung-2 stroke ring with the bending shaft at the measured on-water force-velocity? (SOURCES §223)

At Nordez et al. 2025's on-water handle force-velocity (V0 4.53 m/s, SOURCES §222) the bending scull passes [CR06]'s
rung 2 but [BR24]'s stroke runs 0.4 m/s slow at 80 % of the measured power with IVV 61.5 %, while the rigid oar is right. This
runs one athlete's rung 2 rigid and bending (same muscles, same force-velocity) and prints the last stroke through the
drive: the stroke clock's speed tau', the gate angle, the shaft's bend, the blade's rate over the shaft's, the handle
force and the boat speed; then the bend's oscillation (period from its zero crossings about the drive's trend, and the
log decrement of successive peaks) against the shaft-crew mode sqrt(K / I) of SOURCES §218.

    python research/blade/bending_ring_trace.py [--athlete br24] [--fv 4.53] [--hold-own] [--strokes 10]

Output: printed, and data/local/validation/bending_ring_trace_<athlete>_fv<V0>.log by redirection.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow")]
import rung2_population as RP                                    # noqa: E402

from coxswain.core.state import STATE_SIZE                       # noqa: E402


def last_stroke(sim, run):
    """Per sample of the last stroke: time from its start, tau', gate angle (deg), bend (deg), blade/shaft rate,
    handle force per oar (N), boat speed (m/s), in-air flag."""
    t, y = run.last_time, run.last_states
    inboard = float(sim.boat.rig.seats[sim._seats[0]].oarlocks[0].oar.inboard)
    tl = np.array(sim.tlog)
    tl = tl[np.argsort(tl[:, 0])]
    hist = getattr(sim, "_bend_history", None) if sim.oar_bending is not None else None
    n = sim.n_oar_states
    saved = (sim._bend.copy(), sim._bend_rate.copy()) if hist is not None else None
    rows = []
    for k in range(t.size):
        bend, ratio = 0.0, 1.0
        if hist is not None:
            sim._bend[:], sim._bend_rate[:] = hist[0, :, k], hist[1, :, k]
            bend = float(hist[0, 0, k])
            rate = float(y[STATE_SIZE + n, k])
            if abs(rate) > 1e-6:
                ratio = (rate + float(sim._bend_lever[0]) * float(hist[1, 0, k])) / rate
        try:
            torque = float(np.asarray(sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k]))[0, 0])
        except Exception:                                         # noqa: BLE001 -- a diagnostic column only
            torque = float("nan")
        td = float(np.interp(t[k], tl[:, 0], tl[:, 2]))
        rows.append((t[k] - t[0], td, np.degrees(float(y[STATE_SIZE, k])), np.degrees(bend), ratio,
                     -torque / inboard, float(y[6, k]), bool(sim._air_mask[0, k])))
    if saved is not None:
        sim._bend[:], sim._bend_rate[:] = saved
    return np.array(rows, dtype=float)


def oscillation(tt, x):
    """Period (s) from zero crossings of x about its quadratic trend, and the log decrement of successive peaks."""
    if tt.size < 8:
        return float("nan"), float("nan")
    r = x - np.polyval(np.polyfit(tt, x, 2), tt)
    s = np.sign(r)
    cross = tt[1:][s[1:] * s[:-1] < 0]
    period = float(2.0 * np.mean(np.diff(cross))) if cross.size >= 3 else float("nan")
    peaks = [abs(r[i]) for i in range(1, r.size - 1) if abs(r[i]) >= abs(r[i - 1]) and abs(r[i]) >= abs(r[i + 1])]
    dec = float(np.mean(np.log(np.array(peaks[:-1]) / np.array(peaks[1:])))) if len(peaks) >= 3 else float("nan")
    return period, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athlete", default="br24", choices=["br24", "cr06"])
    ap.add_argument("--fv", type=float, default=4.53, help="handle force-velocity V0, m/s (SOURCES §222)")
    ap.add_argument("--hold", type=float, default=0.0, help="the crew's hold beta instead of / on top of the F-V")
    ap.add_argument("--hold-own", action="store_true")
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--step", type=float, default=0.04, help="print interval through the drive, s")
    ap.add_argument("--modes", nargs="+", default=["rigid", "bending"], choices=["rigid", "bending"],
                    help="run order within one process (a later run can inherit an earlier one's caches: S4)")
    ap.add_argument("--dt", type=float, default=None,
                    help="fixed integration step, s (default: integrators.estimate_step, 6.25 ms): a convergence test "
                         "of the bend's once-per-step hinge update")
    ap.add_argument("--added-mass", default="none", choices=["none", "patton", "labbe"])
    ap.add_argument("--bend-dynamics", action="store_true", help="the bend as a dynamic state (SOURCES §224)")
    a = ap.parse_args()
    if a.dt:
        import coxswain.core.integrators as I
        I.estimate_step = lambda *args, **kw: float(a.dt)
        print("integration step %.5f s" % a.dt, flush=True)
    RP.ADDED_MASS = a.added_mass
    if a.bend_dynamics:
        from coxswain.crew.oar_bending import BendDynamics
        RP.BEND_DYNAMICS = BendDynamics()
    from coxswain.crew.handle_fv import HandleForceVelocity
    RP.FV = HandleForceVelocity(a.fv, "--fv") if a.fv else None
    RP.HOLD, RP.HOLD_OWN = float(a.hold), bool(a.hold_own)
    RP.BLADE, RP.BODY = "crossflow_normal", "own"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    print("%s rung 2, own body, cross-flow normal blade, buried catch; F-V V0 %s, hold %g%s"
          % (a.athlete, a.fv, a.hold, " (own pace)" if a.hold_own else ""), flush=True)
    traces = {}
    for label in a.modes:
        # as rung2_population.main does: the population's muscle table is built on the same oar as the athlete's
        RP.BENDING = label == "bending"
        s_p, (s_pop, q_drive, ph, q) = RP.population_tables(a.strokes)
        RP.athlete(a.athlete, a.strokes, s_pop, q_drive, ph, q, s_p["power"])
        sim, run = RP.LAST_RUN[a.athlete]
        traces[label] = (last_stroke(sim, run), sim)
    first = traces[a.modes[0]][1]
    D = float(first.boat.timing.drive_fraction) * float(first.boat.timing.period)
    print("\n  t(s)  | tau' rigid  bending | angle rigid bending | bend deg  blade/shaft | force/oar rigid bending N"
          " | boat rigid bending")
    R = traces.get("rigid", traces[a.modes[0]])[0]
    B = traces.get("bending", traces[a.modes[0]])[0]
    for tq in np.arange(0.0, max(R[-1, 0], B[-1, 0]), a.step):
        i, j = int(np.argmin(abs(R[:, 0] - tq))), int(np.argmin(abs(B[:, 0] - tq)))
        print("  %5.2f | %6.3f  %6.3f   | %6.1f  %6.1f    | %6.2f   %6.3f     | %7.0f  %7.0f          | %5.2f  %5.2f%s"
              % (tq, R[i, 1], B[j, 1], R[i, 2], B[j, 2], B[j, 3], B[j, 4], R[i, 5], B[j, 5], R[i, 6], B[j, 6],
                 "  (air)" if B[j, 7] else ""))
    drive = B[B[:, 7] == 0]
    if drive.size:
        per, dec = oscillation(drive[:, 0], drive[:, 3])
        per_td, dec_td = oscillation(drive[:, 0], drive[:, 1])
        print("\nbending, blade in: bend oscillation period %.3f s, log decrement %.2f; tau' period %.3f s, decrement %.2f"
              % (per, dec, per_td, dec_td))
        print("bend range %.2f to %.2f deg; tau' range %.3f to %.3f (rigid %.3f to %.3f); drive %.2f s"
              % (drive[:, 3].min(), drive[:, 3].max(), drive[:, 1].min(), drive[:, 1].max(),
                 R[R[:, 7] == 0][:, 1].min(), R[R[:, 7] == 0][:, 1].max(), D))


if __name__ == "__main__":
    main()
