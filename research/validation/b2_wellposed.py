"""Is B2 a valid test of the shaft? Mean of forces against the force of the mean hands (§244).

B2 imposes the population's *mean* hand motion and compares the resulting force with the population's *mean* force
curve, which the data's author forms by normalising each stroke's force to its own peak and averaging (RBN 2023/12).
If the force depends on the hands nonlinearly -- and early in the drive it does steeply (§240) -- the force of the mean
hands need not equal the mean of the forces (Jensen's inequality), and B2 would test the averaging as much as the oar.

Here, on the blade and shaft alone (``open_loop_shaft.kinematic``; the hull speed the population's simulated one), an
ensemble of hand motions is built from the curves at hand -- the population's imposed sculling hands and the two
athletes' measured hands mapped onto the population's arc (lag_decomposition.athlete_on_population) -- and, for each
shaft (flexible, rigid):

* F(mean hands): the force of the ensemble's mean drive progress (what B2 does);
* mean F: the mean of each member's force, each normalised to its own peak (what the reference curve is).

Each is scored against the sculling averages at rate 32 (rms of % of peak, per 10 % of the drive's time). If the two
differ by as much as the shafts differ, B2 cannot discriminate the shafts as constructed. A small ensemble (three
members spanning two athletes and the population) bounds, it does not estimate, the effect.

    python research/validation/b2_wellposed.py
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

Q = LD.Q


def main():
    SCC.setup("medium")
    sim, run = population_run(10, "rbn2023-scull")
    t, _f, _w = HFS.model_stroke(sim, run)
    y = run.last_states
    n = sim.n_oar_states
    ang = y[STATE_SIZE]
    v_pop = y[6]
    tf, i1 = LD.drive_fraction_map(t, ang)
    inboard = float(sim.boat.rig.seats[sim._seats[0]].oarlocks[0].oar.inboard)
    dt = float(t[1] - t[0])
    c, fin = float(ang[0]), float(ang[i1])
    ref = sculling_averages.reference(32)["force"]

    def progress(a):
        return (c - a) / (c - fin)

    members = {"population hands": np.array(ang, float)}
    for name in ("br24", "cr06"):
        members["%s hands" % name] = LD.athlete_on_population(name, tf, i1, ang, t)[0]
    on = np.arange(t.size) <= i1
    mean_prog = np.mean([progress(a)[on] for a in members.values()], axis=0)
    mean_ang = np.array(ang, float)
    mean_ang[on] = c - (c - fin) * mean_prog

    def force(a, shaft):
        r = np.gradient(a, dt)
        acc = np.gradient(r, dt)
        saved = np.array(sim._entry_angle, dtype=float)
        f = OL.kinematic(sim, shaft, t, a, r, acc, inboard, speeds=v_pop)["force"]
        sim._entry_angle = saved
        g = np.interp(Q, tf[:i1 + 1], f[:i1 + 1])
        return 100 * g / g.max()

    rms = lambda x: float(np.sqrt(np.mean((x - ref) ** 2)))      # noqa: E731
    print("ensemble: " + ", ".join(members) + " (early hand speed differs most: §240)")
    print("%-28s" % "" + " ".join("%5.0f%%" % (100 * q) for q in Q) + " | rms vs sculling 32")
    print("%-28s" % "sculling averages, rate 32" + " ".join("%6.0f" % v for v in ref) + " |")
    out = {}
    for shaft in ("hinge", "rigid"):
        each = {k: force(a, shaft) for k, a in members.items()}
        f_mean_hands = force(mean_ang, shaft)
        mean_f = np.mean(list(each.values()), axis=0)
        for k, s in each.items():
            print("%-28s" % ("%s, %s" % (shaft, k))[:28] + " ".join("%6.0f" % v for v in s) + " | %5.1f" % rms(s))
        print("%-28s" % ("%s, F(mean hands)" % shaft) + " ".join("%6.0f" % v for v in f_mean_hands)
              + " | %5.1f" % rms(f_mean_hands))
        print("%-28s" % ("%s, mean F" % shaft) + " ".join("%6.0f" % v for v in mean_f) + " | %5.1f" % rms(mean_f))
        gap = float(np.sqrt(np.mean((f_mean_hands - mean_f) ** 2)))
        print("  %s: F(mean hands) against mean F, rms %.1f points; largest %.0f points at %.0f %% of the drive"
              % (shaft, gap, np.abs(f_mean_hands - mean_f).max(), 100 * Q[np.argmax(np.abs(f_mean_hands - mean_f))]))
        out[shaft] = (rms(f_mean_hands), rms(mean_f), gap)
    print("\nB2's verdict (flexible against rigid, rms vs the sculling averages):")
    print("  as B2 is built, F(mean hands): flexible %.1f, rigid %.1f -> %s" % (
        out["hinge"][0], out["rigid"][0], "flexible better" if out["hinge"][0] < out["rigid"][0] else "rigid better"))
    print("  as the reference is formed, mean F: flexible %.1f, rigid %.1f -> %s" % (
        out["hinge"][1], out["rigid"][1], "flexible better" if out["hinge"][1] < out["rigid"][1] else "rigid better"))
    print("  averaging gap (F(mean hands) vs mean F): flexible %.1f, rigid %.1f points; the shafts differ by %.1f"
          % (out["hinge"][2], out["rigid"][2], abs(out["hinge"][0] - out["rigid"][0])))


if __name__ == "__main__":
    main()
