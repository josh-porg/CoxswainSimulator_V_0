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


def main():
    RP.BODY, RP.BLADE = "own", "crossflow_normal"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    m = RP.V.ATHLETES["br24"][0]()
    boat = RP.athlete_boat("br24", m)
    body = RP.own_body("br24", boat)
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
    # measured: catch at the least oar angle
    D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"), delimiter=",", names=True)[:-1]
    n = len(D)
    k = int(np.argmin(0.5 * (D["A1"] + D["A2"])))
    ah, vh = np.roll(D["As"], -k), np.roll(D["Vs"], -k)
    tm = np.arange(n) * P / n
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


if __name__ == "__main__":
    main()
