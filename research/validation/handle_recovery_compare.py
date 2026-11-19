"""The hands through the recovery and into the catch: the model's imposed handle law against [BR24]'s measured (§242).

rung 2's athlete boats impose the population's drive law (RBN 2023/12 Fig. 1) and, for the recovery, the parent sweep's
raised-cosine law, on the athlete's arc; the arms follow the handle. [BR24]'s BioRow file has his handle velocity (Vh1,
Vh2, at the force point) and his oar angle, so the two can be compared directly, as handle speed along the boat (the gate
rate times the inboard to the force point) from the catch. Also printed: the data author's sculling averages' recovery
(sculling_averages.py, rate 32, the nearest), scaled to his peak drive handle speed.

    python research/validation/handle_recovery_compare.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"),
                os.path.join(ROOT, "research", "blade")]
import rung2_population as RP                                    # noqa: E402
import open_loop_shaft as OL                                     # noqa: E402
import sculling_averages                                         # noqa: E402


def main():
    RP.BODY, RP.BLADE = "own", "crossflow_normal"
    RP.POP_RECOVERY = "sculling" if "sculling" in sys.argv else "cosine"
    print("planned recovery:", RP.POP_RECOVERY)
    name = "cr06" if "cr06" in sys.argv else "br24"
    m = RP.V.ATHLETES[name][0]()
    boat = RP.athlete_boat(name, m)
    P = float(boat.timing.period)
    D = float(boat.timing.drive_fraction)
    sw = boat.oar_sweep
    lever = OL.measured(name)[7]
    T, t, ang, rate, acc, v, f, _lv = OL.measured(name)
    i1 = int(np.argmin(ang))
    D_meas = t[i1] / T
    q = np.arange(0.0, 1.0, 0.025)
    model_rate = np.array([float(sw.rate(u * P, boat.timing)) for u in q])
    meas_rate = np.interp(q * T, t, rate)
    peak_m = np.abs(model_rate[q < D]).max()
    peak_d = np.abs(meas_rate[q < D_meas]).max()
    ref = sculling_averages.reference(32, "recovery")
    print("[BR24] drive fraction: model %.3f, measured %.3f; handle speed at the force point (lever %.3f m), m/s"
          % (D, D_meas, lever))
    print(" cycle | model imposed | measured | measured / its drive peak | model / its drive peak")
    for j, u in enumerate(q):
        print("  %.3f |   %+6.2f      |  %+6.2f  |   %+5.2f   |   %+5.2f" % (
            u, -model_rate[j] * lever, -meas_rate[j] * lever, -meas_rate[j] / peak_d, -model_rate[j] / peak_m))
    # the recovery on its own clock (time fraction of the recovery), against the sculling averages
    print("\nrecovery, per 10 % of its time: handle speed / drive peak (negative: towards the stern)")
    tr = np.linspace(0.0, 1.0, 11)
    mod = np.array([float(sw.rate((D + x * (1 - D)) * P, boat.timing)) for x in tr]) / peak_m
    mea = np.interp((D_meas + tr * (1 - D_meas)) * T, t, rate) / peak_d
    print("  model imposed      " + " ".join("%5.2f" % -x for x in mod))
    print("  %-18s " % ("[%s] measured" % name.upper()) + " ".join("%5.2f" % -x for x in mea))
    print("  sculling avg (32)  " + " ".join("%5.2f" % (x / 100) for x in ref["handle_v"]))
    rms = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))     # noqa: E731
    print("  rms vs measured: model %.3f, sculling averages %.3f" % (rms(-mod, -mea), rms(ref["handle_v"] / 100, -mea)))


if __name__ == "__main__":
    main()
