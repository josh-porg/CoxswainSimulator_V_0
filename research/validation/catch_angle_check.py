"""How much of the early-drive force deficit is the catch angle the population is rowed at? (§245)

The population stroke borrows [CR06]'s rig: catch 60.5 deg, finish -44.3 deg (an arc split 58 / 42). Early in the
drive the blade's normal slip is the small difference of the gate-driven blade speed and the hull's v cos(theta), so a
few degrees of catch angle move it a lot. Here the blade and shaft alone (``open_loop_shaft.kinematic``) on the
population's own motion and hull, with the whole arc rotated towards the catch by ``delta`` (the same rates; the arc's
length unchanged), flexible and rigid, scored against the sculling averages at rate 32. [BR24]'s measured split is
67.8 / 37.3 deg; [CR06]'s 60.5 / 44.3.

    python research/validation/catch_angle_check.py
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
import open_loop_shaft as OL                                     # noqa: E402
import sculling_averages                                         # noqa: E402
import shaft_coupling_check as SCC                               # noqa: E402
from population_force_shape import population_run               # noqa: E402

from coxswain.core.state import STATE_SIZE                       # noqa: E402


def main():
    SCC.setup("medium")
    sim, run = population_run(10, "rbn2023-scull")
    t, _f, _w = HFS.model_stroke(sim, run)
    y = run.last_states
    n = sim.n_oar_states
    ang, rate = y[STATE_SIZE], y[STATE_SIZE + n]
    acc = np.array([sim._sweep_motion(float(tt))[1] for tt in t])
    v_pop = y[6]
    tf, i1 = LD.drive_fraction_map(t, ang)
    inboard = float(sim.boat.rig.seats[sim._seats[0]].oarlocks[0].oar.inboard)
    ref = sculling_averages.reference(32)["force"]
    print("population arc: catch %.1f, finish %.1f deg" % (np.degrees(ang[0]), np.degrees(ang[i1])))
    print("%-22s" % "" + " ".join("%5.0f%%" % (100 * q) for q in LD.Q) + " | rms | 10-30 % rms")
    print("%-22s" % "sculling averages 32" + " ".join("%6.0f" % v for v in ref))
    for shaft in ("hinge", "rigid"):
        for delta in (0.0, 3.0, 6.0, 7.3):
            a = ang + np.radians(delta)
            saved = np.array(sim._entry_angle, dtype=float)
            f = OL.kinematic(sim, shaft, t, a, rate, acc, inboard, speeds=v_pop)["force"]
            sim._entry_angle = saved
            g = np.interp(LD.Q, tf[:i1 + 1], f[:i1 + 1])
            s = 100 * g / g.max()
            print("%-22s" % ("%s, catch +%.1f deg" % (shaft, delta)) + " ".join("%6.0f" % v for v in s)
                  + " | %4.1f | %4.1f" % (np.sqrt(np.mean((s - ref) ** 2)), np.sqrt(np.mean((s[1:4] - ref[1:4]) ** 2))))


if __name__ == "__main__":
    main()
