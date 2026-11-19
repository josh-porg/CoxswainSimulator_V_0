"""Where the model's boat-speed swing is made: the crew's push on the hull against the one [BR24]'s accelerometer implies.

With the blade out the system's momentum changes only by the drag, so the crew's relative push on the hull is

    -m_c a_rel = M a_hull + D(v)           (M = hull + crew + oars, the whole system's mass)

From his measured hull acceleration (BioRow As, his velocity Vs) and the model boat's own drag this gives the push his
body must have made; the model's push is sum_i m_i a_i,x of his own-body field (research/crew/rung2_population.own_body,
his seat and trunk channels on the model's postures), by segment group, on the stroke clock at tau' = 1. Prints the
masses, and through the cycle the implied and modelled push (positive pushes the hull forward). SOURCES §231-232.

    python research/validation/recovery_momentum.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"),
                os.path.join(ROOT, "research", "blade")]
import rung2_population as RP                                    # noqa: E402

from coxswain.core.state import State                            # noqa: E402
from coxswain.crew.kinematics import SEGMENT_ORDER               # noqa: E402

GROUPS = {"head + trunk": ("head", "upper_trunk", "mid_trunk", "lower_trunk"),
          "arms": ("upper_arm_port", "upper_arm_starboard", "forearm_hand_port", "forearm_hand_starboard"),
          "thighs": ("thigh_port", "thigh_starboard"), "shanks + feet": ("shank_foot_port", "shank_foot_starboard")}


def measured_hull(athlete, P, n=200):
    """``(t, hull acceleration, hull speed)`` over one cycle from the catch. [BR24]: BioRow's As / Vs (catch at the least
    oar angle, BioRow's sign); [CR06]: her digitised boat velocity (catch at the largest oar angle, the paper's sign),
    smoothed by GCV and differentiated (§241)."""
    if athlete == "br24":
        D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"), delimiter=",", names=True)[:-1]
        k = int(np.argmin(0.5 * (D["A1"] + D["A2"])))
        return np.arange(len(D)) * P / len(D), np.roll(D["As"], -k), np.roll(D["Vs"], -k)
    from scipy.interpolate import make_smoothing_spline
    V = RP.V
    T = V.CR06_T
    g = np.linspace(0.0, T, 400, endpoint=False)
    t0 = g[int(np.argmax(V._cr06_series("oar_angle_deg")(g)))]
    v = V._cr06_series("boat_velocity_mps")(np.mod(g + t0, T))
    sp = make_smoothing_spline(np.concatenate([g - T, g, g + T]), np.concatenate([v, v, v]))
    tm = np.linspace(0.0, T, n, endpoint=False)
    return tm * P / T, sp(tm, 1), sp(tm)


def main():
    athlete = "cr06" if "cr06" in sys.argv else "br24"
    RP.POP_RECOVERY = "sculling" if "sculling" in sys.argv else RP.POP_RECOVERY
    print("planned recovery:", RP.POP_RECOVERY)
    RP.BODY, RP.BLADE = "own", "crossflow_normal"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    m = RP.V.ATHLETES[athlete][0]()
    boat = RP.athlete_boat(athlete, m)
    body = RP.own_body(athlete, boat)
    print("athlete:", athlete)
    P = float(boat.timing.period)
    mass, _p, _v, _a = body(0.0)
    mass = np.asarray(mass, float)
    hull = float(getattr(boat, "hull_mass", np.nan))
    crew = float(mass.sum())
    print("masses: crew field %.1f kg (%d points); hull %s; rower stated %.1f kg" % (
        crew, mass.size, "%.1f" % hull if np.isfinite(hull) else "n/a", float(m.get("mass", np.nan))))
    try:
        M_total = float(boat.total_mass)
    except Exception:                                             # noqa: BLE001 -- name differs by build
        M_total = crew + (hull if np.isfinite(hull) else 18.0)
    print("system mass used: %.1f kg" % M_total)
    tm, ah, vh = measured_hull(athlete, P)
    sim = RP.make(boat, "reference", body=body)


    # the model boat's drag at speed v, from the hull's resistance curve (as the scorecard reads it)
    from coxswain.validation.scorecard import _drag_at
    print("\n phase | measured a_h  implied push N | model push N:  total  " + "  ".join("%s" % g for g in GROUPS))
    names = list(SEGMENT_ORDER)
    for q in np.arange(0.0, 1.0, 0.05):
        t = q * P
        mm, _pp, _vv, aa = body(t)
        mm, aa = np.asarray(mm, float), np.asarray(aa, float)
        per = mm * aa[:, 0]
        groups = []
        for g, segs in GROUPS.items():
            idx = [i for i in range(mm.size) if names[i % len(names)] in segs]
            groups.append(-per[idx].sum())
        a_meas = float(np.interp(t, tm, ah, period=P))
        v_meas = float(np.interp(t, tm, vh, period=P))
        try:
            d = float(_drag_at(sim, boat, v_meas))
        except Exception:                                         # noqa: BLE001
            d = np.nan
        implied = abs(d) + M_total * a_meas              # -sum m a_rel = M a_hull + D (blade out)
        print("  %.2f | %7.2f     %8.0f       |  %7.0f  " % (q, a_meas, implied, -per.sum()) +
              "  ".join("%8.0f" % x for x in groups))
    # the crew's centre of mass relative to the boat: its travel and its greatest speeds (the data author agreed [BR24]'s
    # travels ~0.72 m, from the seat and trunk channels with segment fractions; 2026-10-28, SOURCES §233)
    k = 400
    xc, vc = np.zeros(k), np.zeros(k)
    for j in range(k):
        mm, pp, vv, _aa = body(j * P / k)
        mm = np.asarray(mm, float)
        xc[j] = float(np.sum(mm * np.asarray(pp, float)[:, 0]) / mm.sum())
        vc[j] = float(np.sum(mm * np.asarray(vv, float)[:, 0]) / mm.sum())
    print("\ncrew centre of mass relative to the boat: travel %.3f m; greatest speeds %+.2f / %+.2f m/s"
          % (np.ptp(xc), vc.max(), vc.min()))
    seat = np.array([np.asarray(body(j * P / k)[1], float)[3, 0] for j in range(k)])
    print("seat (lower trunk) travel %.3f m; centre of mass travel / seat travel %.3f" % (np.ptp(seat), np.ptp(xc) / np.ptp(seat)))
    # the momentum the body takes out of the hull over the catch approach, model against implied: impulses are robust
    # to the sampling and to small timing offsets that pointwise values near a steep peak are not (§242). Window: from
    # the recovery's speed maximum region to just after the oar's turning point, the blade out throughout.
    lo, hi = (0.80, 1.03)
    q = np.linspace(lo, hi, 300)
    tq = np.mod(q, 1.0) * P
    pm = np.array([-float(np.sum(np.asarray(body(float(u))[0], float) * np.asarray(body(float(u))[3], float)[:, 0]))
                   for u in tq])
    am = np.interp(tq, tm, ah, period=P)
    vm = np.interp(tq, tm, vh, period=P)
    dm = np.array([abs(float(_drag_at(sim, boat, float(v)))) for v in vm])
    imp = M_total * am + dm
    dt_q = (hi - lo) * P / (q.size - 1)
    neg = lambda x: np.trapezoid(np.minimum(x, 0.0), dx=dt_q)    # noqa: E731
    print("\ncatch approach, %.2f-%.2f of the cycle: impulse of the push, N s: model %.1f, implied %.1f (decelerating "
          "part: model %.1f, implied %.1f); model - implied = %.1f N s = %+.3f m/s of hull speed"
          % (lo, hi, np.trapezoid(pm, dx=dt_q), np.trapezoid(imp, dx=dt_q), neg(pm), neg(imp),
             np.trapezoid(pm - imp, dx=dt_q), np.trapezoid(pm - imp, dx=dt_q) / M_total))
    print("  greatest decelerating push: model %.0f N at %.3f, implied %.0f N at %.3f of the cycle"
          % (pm.min(), q[np.argmin(pm)] % 1.0, imp.min(), q[np.argmin(imp)] % 1.0))
    for end in (0.99, 1.00, 1.01, 1.02, 1.03, 1.04):            # the window's end on a steep part: its sensitivity
        m_ = q <= end
        d_ = np.trapezoid((pm - imp)[m_], dx=dt_q)
        print("  window %.2f-%.2f: model - implied %+.1f N s = %+.3f m/s" % (lo, end, d_, d_ / M_total))
    names = list(SEGMENT_ORDER)
    parts = {g: np.zeros(q.size) for g in GROUPS}
    for j, u in enumerate(tq):
        mm, _pp, _vv, aa = body(float(u))
        per = np.asarray(mm, float) * np.asarray(aa, float)[:, 0]
        for g, segs in GROUPS.items():
            parts[g][j] = -sum(per[i] for i in range(per.size) if names[i % len(names)] in segs)
    crew_mass = float(np.asarray(body(0.0)[0], float).sum())
    print("  by segment group, impulse N s (per kg of crew, m/s): " + "; ".join(
        "%s %.1f (%.3f)" % (g, np.trapezoid(parts[g], dx=dt_q), np.trapezoid(parts[g], dx=dt_q) / crew_mass)
        for g in GROUPS))
    if "--fine" in sys.argv:
        # the model's push finely (the 0.05 table alternates sign through the drive: real ripple or numerics? §240),
        # with the parts carried by the two measured channels' own second derivatives for comparison
        k = 400
        push = np.zeros(k)
        for j in range(k):
            mm, _pp, _vv, aa = body(j * P / k)
            push[j] = -float(np.sum(np.asarray(mm, float) * np.asarray(aa, float)[:, 0]))
        spec = np.abs(np.fft.rfft(push - push.mean())) / k * 2
        f = np.fft.rfftfreq(k, d=P / k)
        print("\nmodel push, N, every 0.01 of the cycle:")
        for j in range(0, k, 4):
            print("  %.2f %7.0f" % (j / k, push[j]))
        print("largest harmonics (Hz: amplitude N): " + ", ".join(
            "%.1f: %.0f" % (f[i], spec[i]) for i in np.argsort(spec)[::-1][:8]))
        print("rms of the push above 3 Hz: %.0f N" % np.sqrt(np.sum(spec[f > 3.0] ** 2) / 2))


if __name__ == "__main__":
    main()
