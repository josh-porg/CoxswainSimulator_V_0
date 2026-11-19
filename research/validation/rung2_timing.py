"""Where rung 2's muscles pull the athletes' oar timing off rung 1's (§243).

Rung 1 imposes the population's drive law on the athlete's arc (the stroke clock tau = t); rung 2 makes the clock a
state, tau' solved from the body-and-oar virtual-work balance with the population's muscles scaled to the athlete's
power. Both are scored by drive progress against the athlete's measured oar angle (rung2_population.compare). This
reads a rung-2 output (its last stroke's tau' trace), rebuilds tau(t) and the progress, and prints, per 10 % of the
measured drive's time: the measured progress, rung 1's, rung 2's, and rung 2's clock rate; and the drive's duration.

    python research/validation/rung2_timing.py <rung2 json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"),
                os.path.join(ROOT, "research", "blade")]
import rung2_population as RP                                    # noqa: E402


def main():
    d = json.load(open(sys.argv[1]))
    RP.BODY = "own"
    for name in ("cr06", "br24"):
        m = RP.V.ATHLETES[name][0]()
        boat = RP.athlete_boat(name, m)
        P, D = float(boat.timing.period), float(boat.timing.drive_fraction)
        ht, hs, _ha = RP.measured_track(name)
        tr = d[name]["rung2"]["trace"]
        t, td = np.array(tr["t"]), np.array(tr["tau_dot"])
        o = np.argsort(t)
        t, td = t[o], td[o]
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (td[1:] + td[:-1]) * np.diff(t))])
        on = tau <= D * P
        t2, s2 = t[on], RP.progress_of(boat, tau[on])
        t1 = np.linspace(0.0, D * P, 400)
        s1 = RP.progress_of(boat, t1)
        T_meas = ht[-1]
        # the clock's trace covers the drive up to the hand-over to the finish's turning point (rung2_clock HANDOVER),
        # so rung 2 is read only where it exists, as rung2_population.compare does
        print("\n== %s: drive duration measured %.3f s, rung 1 %.3f s; rung 2's trace ends at progress %.3f, %.3f s "
              "from the catch; stroke period %.3f s" % (name, T_meas, D * P, s2[-1], t2[-1], P))
        mean_rate = float(np.trapezoid(td[t <= t2[-1]], t[t <= t2[-1]]) / t2[-1])
        print("  rung 2's mean clock rate over that span %.3f (>1: the hands run ahead of the planned law)" % mean_rate)
        print("  measured-drive time | measured | rung 1 | rung 2 | rung 2 - measured | rung 2 clock rate")
        for q in np.arange(0.0, 1.01, 0.1):
            u = q * T_meas
            if u > t2[-1]:
                break
            sm, r1, r2 = (np.interp(u, ht, hs), np.interp(u, t1, s1), np.interp(u, t2, s2))
            print("        %3.0f %%        |  %.3f   | %.3f  | %.3f  |   %+.3f          |  %.3f"
                  % (100 * q, sm, r1, r2, r2 - sm, np.interp(u, t, td)))


if __name__ == "__main__":
    main()
