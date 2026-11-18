"""Why does the flexible shaft load on time on the athletes' measured strokes but lag on the population's? (§240)

The blade and shaft alone (``open_loop_shaft.kinematic``: the one load path, the quasi-static hinge, which the closed
loop matches to 1-3 N; shaft_coupling_check.py), on four motions built from two sources, one input swapped at a time:

* hands: the population's closed-loop gate motion (rung 1, the default research options) or an athlete's measured gate
  angle, mapped onto the population's arc by drive progress at the same fraction of the drive's time;
* hull: the population's simulated surge speed, or the athlete's measured boat speed at the same fraction of the
  drive's time, scaled to the population's cycle mean (its shape kept).

Prints the handle force per 10 % of the drive's time (% of each run's peak) against the data author's sculling averages
at rate 32 and against the athlete's own measured force shape.

    python research/validation/lag_decomposition.py [--athletes br24 cr06] [--drive rbn2023-scull]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"),
                os.path.join(ROOT, "research", "blade")]
import rung2_population as RP                                    # noqa: E402
import handle_force_shape as HFS                                 # noqa: E402
import open_loop_shaft as OL                                     # noqa: E402
import sculling_averages                                         # noqa: E402
import shaft_coupling_check as SCC                               # noqa: E402
from population_force_shape import population_run               # noqa: E402

from coxswain.core.state import STATE_SIZE                       # noqa: E402

Q = np.arange(0.0, 1.01, 0.1)


def drive_fraction_map(t, ang):
    """``(tf, i1)``: the time fraction of the drive from the catch (largest angle, index 0) to the finish."""
    i1 = int(np.argmin(ang))
    return t / t[i1], i1


def athlete_on_population(name, tf_pop, i1_pop, ang_pop, t_pop):
    """The athlete's measured gate angle (as drive progress) and boat speed (its shape), at the population's samples
    of drive-time fraction; the angle on the population's arc. Returns (angle, rate, acc, speed, measured force % shape)."""
    T, t, ang, rate, acc, v, f, lever = OL.measured(name)
    tf_a, i1_a = drive_fraction_map(t, ang)
    prog_a = (ang[0] - ang[:i1_a + 1]) / (ang[0] - ang[i1_a])
    c, fin = float(ang_pop[0]), float(ang_pop[i1_pop])
    x = np.clip(tf_pop, 0.0, None)
    on = x <= 1.0
    prog = np.interp(x[on], tf_a[:i1_a + 1], prog_a)
    a_new = np.array(ang_pop, float)
    a_new[on] = c - (c - fin) * prog
    dt = float(t_pop[1] - t_pop[0])
    r_new = np.gradient(a_new, dt)
    acc_new = np.gradient(r_new, dt)
    v_cycle_mean = float(np.mean(v))
    v_on = np.interp(x[on], tf_a[:i1_a + 1], v[:i1_a + 1]) / v_cycle_mean
    f_meas = np.interp(Q, tf_a[:i1_a + 1], f[:i1_a + 1])
    return a_new, r_new, acc_new, v_on, on, 100 * f_meas / f_meas.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athletes", nargs="+", default=["br24", "cr06"])
    ap.add_argument("--drive", default="rbn2023-scull", choices=["rbn2023", "rbn2023-scull"])
    ap.add_argument("--strokes", type=int, default=10)
    a = ap.parse_args()
    SCC.setup("medium")
    sim, run = population_run(a.strokes, a.drive)
    t, f_cl, _wet = HFS.model_stroke(sim, run)
    y = run.last_states
    n = sim.n_oar_states
    ang, rate = y[STATE_SIZE], y[STATE_SIZE + n]
    acc = np.array([sim._sweep_motion(float(tt))[1] for tt in t])
    v_pop = y[6]                                                  # the hull's surge speed (population_kinematics)
    tf, i1 = drive_fraction_map(t, ang)
    inboard = float(sim.boat.rig.seats[sim._seats[0]].oarlocks[0].oar.inboard)
    v_mean = float(np.mean(v_pop))
    ref = sculling_averages.reference(32)["force"]

    def shape(f):
        g = np.interp(Q, tf[:i1 + 1], f[:i1 + 1])
        return 100 * g / g.max(), float(100 * tf[np.argmax(f[:i1 + 1])])

    def run_open(ang_, rate_, acc_, speed_, shaft="hinge"):
        saved = np.array(sim._entry_angle, dtype=float)
        out = OL.kinematic(sim, shaft, t, ang_, rate_, acc_, inboard, speeds=speed_)
        sim._entry_angle = saved
        return out["force"]

    print("population closed loop, %s hands, medium shaft: %s" % (a.drive, RP.C2.summary(run)))
    print("force per 10 %% of the drive's time, %% of each run's own peak; sculling averages at 32 for reference")
    print("%-34s" % "" + " ".join("%5.0f%%" % (100 * q) for q in Q) + " | peak at | rms vs sculling")
    rows = [("sculling averages, rate 32", ref, float("nan"))]
    s, pk = shape(run_open(ang, rate, acc, v_pop))
    rows.append(("pop hands + pop hull (open, hinge)", s, pk))
    s, pk = shape(run_open(ang, rate, acc, v_pop, shaft="rigid"))
    rows.append(("pop hands + pop hull (open, RIGID)", s, pk))
    for name in a.athletes:
        a_new, r_new, acc_new, v_shape, on, f_meas = athlete_on_population(name, tf, i1, ang, t)
        v_ath = np.array(v_pop, float)
        v_ath[on] = v_shape * v_mean
        rows.append(("%s measured force" % name, f_meas, float("nan")))
        s, pk = shape(run_open(ang, rate, acc, v_ath))
        rows.append(("pop hands + %s hull" % name, s, pk))
        s, pk = shape(run_open(a_new, r_new, acc_new, v_pop))
        rows.append(("%s hands + pop hull" % name, s, pk))
        s, pk = shape(run_open(a_new, r_new, acc_new, v_ath))
        rows.append(("%s hands + %s hull" % (name, name), s, pk))
        s, pk = shape(run_open(a_new, r_new, acc_new, v_ath, shaft="rigid"))
        rows.append(("%s hands + %s hull (RIGID)" % (name, name), s, pk))
    for label, s, pk in rows:
        rms = float(np.sqrt(np.mean((s - ref) ** 2)))
        print("%-34s" % label[:34] + " ".join("%6.0f" % v for v in s) + " |  %5.1f  |  %5.1f" % (pk, rms))
    # the hull speed through the drive, as a fraction of each source's cycle mean
    print("\nhull speed / cycle mean, per 10 %% of the drive's time:")
    print("%-34s" % "population (simulated)" + " ".join("%6.3f" % v for v in np.interp(Q, tf[:i1 + 1], v_pop[:i1 + 1] / v_mean)))
    for name in a.athletes:
        T, tt, aa, rr, ac, vv, ff, lv = OL.measured(name)
        tfa, i1a = drive_fraction_map(tt, aa)
        print("%-34s" % ("%s (measured)" % name) + " ".join("%6.3f" % v for v in np.interp(Q, tfa[:i1a + 1], vv[:i1a + 1] / np.mean(vv))))
    print("\nhandle (gate) speed / its drive peak, per 10 %% of the drive's time:")
    print("%-34s" % "population (imposed)" + " ".join("%6.3f" % v for v in np.interp(Q, tf[:i1 + 1], np.abs(rate[:i1 + 1]) / np.abs(rate[:i1 + 1]).max())))
    for name in a.athletes:
        T, tt, aa, rr, ac, vv, ff, lv = OL.measured(name)
        tfa, i1a = drive_fraction_map(tt, aa)
        print("%-34s" % ("%s (measured)" % name) + " ".join("%6.3f" % v for v in np.interp(Q, tfa[:i1a + 1], np.abs(rr[:i1a + 1]) / np.abs(rr[:i1a + 1]).max())))


if __name__ == "__main__":
    main()
