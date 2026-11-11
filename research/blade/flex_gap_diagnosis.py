"""Why does the dynamic shaft row rung 2 with 4-6 IVV points more than the rigid oar with the same added mass? (§227)

Runs one athlete's rung 2 twice in one process -- rigid oar + added mass, then the dynamic shaft (BendDynamics) + the
same added mass -- with the split hand-over and the second-half release, and prints for the dynamic run's last stroke:

* at the release: gate angle, the bent blade's angle (gate + bend), the finish angle, the bend, the spring's energy
  0.5 k_h phi^2 per blade and the gate's and the bent blade's normal velocity through the water;
* the stroke's handle work, the spring's energy at the release, and the energy the structural damper takes;
* boat speed through the cycle for both runs, so the reader sees where (drive or recovery) the two part.

    python research/blade/flex_gap_diagnosis.py [--athlete cr06] [--added-mass patton] [--strokes 10]

Output: printed; redirect to data/local/validation/flex_gap_<athlete>.log.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow")]
import rung2_population as RP                                    # noqa: E402

from coxswain.core.state import STATE_SIZE, State                # noqa: E402


def stroke_table(sim, run):
    """Per sample of the last stroke: t from its start, gate angle, bend, bend rate, boat speed, in-air, handle power."""
    t, y = run.last_time, run.last_states
    n = sim.n_oar_states
    hist = getattr(sim, "_bend_history", None) if sim.oar_bending is not None else None
    power = sim._handle_power(t, y)
    inboard = float(sim.boat.rig.seats[sim._seats[0]].oarlocks[0].oar.inboard)
    rows = []
    for k in range(t.size):
        phi, phid = (float(hist[0, 0, k]), float(hist[1, 0, k])) if hist is not None else (0.0, 0.0)
        force = -float(np.asarray(sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k]))[0, 0]) / inboard
        rows.append((t[k] - t[0], float(y[STATE_SIZE, k]), phi, phid, float(y[6, k]), bool(sim._air_mask[0, k]),
                     float(power[k]), float(y[STATE_SIZE + n, k]), force))
    return np.array(rows, dtype=float)


def force_shape(t, force, wet):
    """Handle force at fixed fractions of the time from the catch (first wet sample) to the release, its peak and
    when, and its impulse-weighted centre, as fractions of that time."""
    i0, i1 = int(np.flatnonzero(wet)[0]), int(np.flatnonzero(wet)[-1])
    tt, ff = t[i0:i1 + 1] - t[i0], force[i0:i1 + 1]
    span = tt[-1]
    at = [float(np.interp(q * span, tt, ff)) for q in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)]
    pos = np.clip(ff, 0.0, None)
    centre = float(np.sum(pos * tt) / np.sum(pos) / span) if pos.sum() else np.nan
    return span, at, float(ff.max()), float(tt[np.argmax(ff)] / span), centre


def measured_br24():
    """[BR24]'s measured cycle: time from the catch, mean handle force per oar (N), 'wet' while the force is
    positive after the catch."""
    sys.path.insert(0, os.path.join(ROOT, "research", "biorow"))
    import like_for_like as L
    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    T = 60.0 / L.RATE
    n = len(D)
    k = int(np.argmin(0.5 * (D["A1"] + D["A2"])))
    order = np.roll(np.arange(n), -k)
    t = np.arange(n) * T / n
    force = (0.5 * (D["H1"] + D["H2"]))[order]
    peak = int(np.argmax(force))
    after = np.flatnonzero(force[peak:] <= 0.0)
    end = peak + (int(after[0]) - 1 if after.size else n - 1 - peak)
    wet = np.arange(n) <= end                                    # the catch to the force's last positive sample
    return t, force, wet


def release_report(sim, run, tab):
    """The dynamic shaft at the release of the last stroke."""
    air = tab[:, 5] > 0.5
    wet = np.flatnonzero(~air)
    if wet.size == 0:
        print("  no blade in the water on the last stroke")
        return
    k = int(wet[-1]) + 1 if wet[-1] + 1 < tab.shape[0] else int(wet[-1])
    t, y = run.last_time, run.last_states
    oar = sim._oars[0]
    arm, e, k_h, c_h, m_dry = sim._bend_constants(0)
    locks = sim.boat.rig.seats[sim._seats[0]].oarlocks
    gate, phi, phid, rate = tab[k, 1], tab[k, 2], tab[k, 3], tab[k, 7]
    state = State.from_vector(y[:STATE_SIZE, k])
    v_gate = sim._normal_velocity(0, gate, rate, state)
    v_blade = sim._normal_velocity(0, gate + phi, rate + float(sim._bend_lever[0]) * phid, state)
    spring = 0.5 * k_h * phi ** 2 * len(locks)
    dt = np.diff(tab[:, 0])
    work = float(np.sum(0.5 * (tab[1:, 6] + tab[:-1, 6]) * dt))
    damper = float(np.sum(c_h * tab[:-1, 3] ** 2 * dt)) * len(locks)
    print("  release at t %.3f s: gate %.1f deg, blade %.1f deg, finish %.1f deg; bend %.2f deg, bend rate %.1f deg/s"
          % (tab[k, 0], np.degrees(gate), np.degrees(gate + phi), np.degrees(oar.finish_angle), np.degrees(phi),
             np.degrees(phid)))
    print("  normal velocity through the water (negative drives): gate %.3f m/s, bent blade %.3f m/s"
          % (v_gate, v_blade))
    print("  stroke: handle work %.1f J; spring energy at release %.1f J (%.1f %%); structural damper %.1f J"
          % (work, spring, 100 * spring / work if work else np.nan, damper))
    drive = tab[~air]
    print("  bend through the drive: max %.2f deg at %.0f %% of the drive; at the release %.0f %% of its max"
          % (np.degrees(drive[:, 2].max()), 100 * np.argmax(drive[:, 2]) / max(len(drive) - 1, 1),
             100 * phi / drive[:, 2].max() if drive[:, 2].max() else np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athlete", default="cr06", choices=["br24", "cr06"])
    ap.add_argument("--fv", type=float, default=4.53)
    ap.add_argument("--added-mass", default="patton", choices=["patton", "labbe"])
    ap.add_argument("--surface-factor", type=float, default=None, help="kappa x the added mass, wetting over 4 deg (§228)")
    ap.add_argument("--release-slip", type=float, default=None, help="the blade comes out over this much oar travel (§235)")
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--step", type=float, default=0.05)
    a = ap.parse_args()
    from coxswain.crew.handle_fv import HandleForceVelocity
    from coxswain.crew.oar_bending import BendDynamics
    RP.FV = HandleForceVelocity(a.fv, "--fv")
    RP.HOLD, RP.HOLD_OWN = 0.0, True
    RP.BLADE, RP.BODY = "crossflow_normal", "own"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    RP.ADDED_MASS, RP.Q_MAP, RP.RELEASE_AFTER, RP.SURFACE_FACTOR = a.added_mass, "split", 0.5, a.surface_factor
    RP.RELEASE_SLIP = a.release_slip
    out = {}
    for label in ("rigid", "dynamic"):
        RP.BENDING = label == "dynamic"
        RP.BEND_DYNAMICS = BendDynamics() if RP.BENDING else None
        s_p, (s_pop, q_drive, ph, q) = RP.population_tables(a.strokes)
        res = RP.athlete(a.athlete, a.strokes, s_pop, q_drive, ph, q, s_p["power"])
        sim, run = RP.LAST_RUN[a.athlete]
        out[label] = (sim, run, stroke_table(sim, run), res)
        print("%s: %s" % (label, {k: v for k, v in res.items() if not isinstance(v, (list, dict))}), flush=True)
    print("\n== %s, dynamic shaft at the release" % a.athlete)
    release_report(out["dynamic"][0], out["dynamic"][1], out["dynamic"][2])
    R, B = out["rigid"][2], out["dynamic"][2]
    print("\n  t(s) | boat rigid dynamic | gate rigid dynamic deg | bend deg | power rigid dynamic W")
    for tq in np.arange(0.0, max(R[-1, 0], B[-1, 0]), a.step):
        i, j = int(np.argmin(abs(R[:, 0] - tq))), int(np.argmin(abs(B[:, 0] - tq)))
        print("  %4.2f | %5.2f %5.2f | %6.1f %6.1f | %5.2f | %6.0f %6.0f%s%s"
              % (tq, R[i, 4], B[j, 4], np.degrees(R[i, 1]), np.degrees(B[j, 1]), np.degrees(B[j, 2]), R[i, 6],
                 B[j, 6], "  rigid-air" if R[i, 5] else "", "  dyn-air" if B[j, 5] else ""))
    for label, (_s, _r, tab, _res) in out.items():
        v = tab[:, 4]
        print("%s last stroke: mean %.3f, min %.3f, max %.3f, (max-min)/mean %.1f %%"
              % (label, v.mean(), v.min(), v.max(), 100 * (v.max() - v.min()) / v.mean()))
    print("\nhandle force per oar from the catch to the release (fractions of that time):")
    print("  %-9s | span s | F at 10 20 30 50 70 90 %%            | peak N at | centre" % "")
    shapes = [(label, tab[:, 0], tab[:, 8], tab[:, 5] < 0.5) for label, (_s, _r, tab, _res) in out.items()]
    if a.athlete == "br24":
        shapes.append(("measured",) + measured_br24())
    for label, t, f, wet in shapes:
        span, at, peak, when, centre = force_shape(t, f, wet)
        print("  %-9s | %5.3f | %s | %4.0f %4.2f | %4.2f" % (label, span, " ".join("%4.0f" % x for x in at), peak,
                                                             when, centre))
    np.savez(os.path.join(ROOT, "data", "local", "validation", "flex_gap_%s%s%s.npz" % (a.athlete, "" if a.surface_factor is None else "_kappa%g" % a.surface_factor, "" if a.release_slip is None else "_relslip%g" % a.release_slip)),
             **{label: tab for label, (_s, _r, tab, _res) in out.items()})


if __name__ == "__main__":
    main()
