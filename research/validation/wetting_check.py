"""The blade's wetting through the catch: the model's entry rule against the population's measured vertical angle (§247).

The model's blade enters when its normal velocity through the water turns driving ([CR06] eq. 16) and wets over 4 deg
of oar travel (BioRow's target; EntryImmersion). The data author's sculling averages give the vertical oar angle per
10 % of the drive's time (the blade's centre at the water at 0 deg, covered at -3 deg), so the population's wetted
fraction is about clip(0.5 - vertical / 6 deg, 0, 1) of the blade's height. This prints both through the first half of
the drive, with the model's blade normal slip, on the population stroke at rung 1 (scullers' hands, flexible shaft).

    python research/validation/wetting_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"),
                os.path.join(ROOT, "research", "blade")]
import handle_force_shape as HFS                                 # noqa: E402
import lag_decomposition as LD                                   # noqa: E402
import sculling_averages                                         # noqa: E402
import shaft_coupling_check as SCC                               # noqa: E402
from population_force_shape import population_run               # noqa: E402

from coxswain.core.state import STATE_SIZE, State                # noqa: E402


def main():
    SCC.setup("medium")
    sim, run = population_run(10, "rbn2023-scull")
    t, f, wet = HFS.model_stroke(sim, run)
    y = run.last_states
    n = sim.n_oar_states
    ang, rate = y[STATE_SIZE], y[STATE_SIZE + n]
    tf, i1 = LD.drive_fraction_map(t, ang)
    w_model = np.array([sim._wetted(0, float(ang[k])) if wet[k] else 0.0 for k in range(t.size)])
    slip = np.array([sim._normal_velocity(0, float(ang[k]), float(rate[k]), State.from_vector(y[:STATE_SIZE, k]))
                     for k in range(t.size)])
    ref = sculling_averages.reference(32)
    w_meas = np.clip(0.5 - ref["vertical"] / 6.0, 0.0, 1.0)
    entry = tf[np.flatnonzero(wet)[0]]
    print("model entry at %.1f %% of the drive's time (gate %.1f deg)" % (100 * entry, np.degrees(ang[np.flatnonzero(wet)[0]])))
    print(" time | population vertical deg -> wetted | model wetted | model slip m/s (gate track; <0 drives) | force N")
    for j, q in enumerate(np.arange(0.0, 0.51, 0.05)):
        k = int(np.argmin(abs(tf - q)))
        vert = float(np.interp(100 * q, ref["time"], ref["vertical"]))
        wm = float(np.clip(0.5 - vert / 6.0, 0.0, 1.0))
        print(" %3.0f%% |      %+5.1f   ->   %.2f          |    %.2f      |   %+.2f   | %4.0f" % (
            100 * q, vert, wm, w_model[k], slip[k], f[k]))


if __name__ == "__main__":
    main()
