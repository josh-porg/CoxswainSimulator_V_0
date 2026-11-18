"""Is the flexible shaft's lag on the population stroke the coupling code or the blade-and-shaft physics? (§240)

The population stroke is run closed loop (rung 1: hands imposed, hull and dynamic shaft simulated together; the
default research options). Its own last stroke -- gate angle, rate, acceleration and the hull's full state at every
sample -- is then fed through the open-loop path (``open_loop_shaft.kinematic``: the same blade loads, the shaft as the
quasi-static hinge, no hull). Same motion, two implementations:

* if the open loop reproduces the closed loop's force through the drive, the lag is the blade-and-shaft physics;
* if the open loop loads on time and the closed loop lags, the closed-loop coupling (the dynamic bend's row in the
  added-mass system, the release, the bookkeeping) is where to look.

Also printed: a fine trace through the release (handle force, bend, wet flag, the blade's normal velocity), for the
late drive's negative force.

    python research/validation/shaft_coupling_check.py [--drive rbn2023-scull] [--strokes 10]
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
from population_force_shape import population_run               # noqa: E402

from coxswain.core.state import STATE_SIZE, State                # noqa: E402


def setup(shaft):
    """The research options population_kinematics.py uses (rung 1 on the population)."""
    from coxswain.crew.handle_fv import HandleForceVelocity
    from coxswain.crew.oar_bending import BendDynamics
    RP.FV = HandleForceVelocity(4.53, "--fv")
    RP.HOLD, RP.HOLD_OWN = 0.0, True
    RP.BLADE, RP.BODY = "crossflow_normal", "model"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    RP.ADDED_MASS, RP.SURFACE_FACTOR = "patton", 0.1
    RP.BENDING = shaft != "rigid"
    RP.RIGID = shaft == "rigid"
    RP.BEND_DYNAMICS = BendDynamics() if RP.BENDING else None


def per_tenth(tf, x, i1):
    q = np.arange(0.0, 1.01, 0.1)
    return np.interp(q, tf[:i1 + 1], x[:i1 + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="rbn2023-scull", choices=["rbn2023", "rbn2023-scull"])
    ap.add_argument("--strokes", type=int, default=10)
    a = ap.parse_args()
    setup("medium")
    sim, run = population_run(a.strokes, a.drive)
    t, f_cl, wet_cl = HFS.model_stroke(sim, run)
    y = run.last_states
    n = sim.n_oar_states
    ang, rate = y[STATE_SIZE], y[STATE_SIZE + n]
    t0 = float(run.last_time[0])
    acc = np.array([sim._sweep_motion(float(tt))[1] for tt in t])      # stroke time from the catch
    states = [State.from_vector(y[:STATE_SIZE, k]) for k in range(t.size)]
    inboard = float(sim.boat.rig.seats[sim._seats[0]].oarlocks[0].oar.inboard)
    bend_cl = sim._bend_history[0, 0] if sim._bend_history is not None else np.zeros(t.size)
    if abs(ang[0] - ang.max()) > 1e-6:
        raise RuntimeError("the last stroke does not start at the catch")
    # the open loop on the closed loop's own motion and hull states; the same simulator object supplies the loads
    saved_entry = np.array(sim._entry_angle, dtype=float)
    ol = {s: OL.kinematic(sim, s, t, ang, rate, acc, inboard, states=states) for s in ("hinge", "rigid")}
    sim._entry_angle = saved_entry
    i1 = int(np.argmin(ang))
    tf = t / t[i1]
    peak = max(f_cl[:i1 + 1].max(), 1e-9)
    ref = sculling_averages.reference(32)
    rows = dict(closed=per_tenth(tf, f_cl, i1), hinge=per_tenth(tf, ol["hinge"]["force"], i1),
                rigid=per_tenth(tf, ol["rigid"]["force"], i1))
    print("population, %s hands, medium shaft (closed loop: dynamic shaft + Patton x0.1); last stroke %s"
          % (a.drive, RP.C2.summary(run)))
    print("handle force per oar, N, per 10 %% of the drive's time (closed-loop peak %.0f N)" % peak)
    print(" time | closed loop | open loop hinge | open loop rigid || %% of own peak: closed / hinge / rigid / "
          "sculling 32")
    for j, q in enumerate(np.arange(0.0, 1.01, 0.1)):
        c, h, r = rows["closed"][j], rows["hinge"][j], rows["rigid"][j]
        print(" %3.0f%% |   %6.0f    |     %6.0f      |     %6.0f      ||   %4.0f / %4.0f / %4.0f / %4.0f" % (
            100 * q, c, h, r, 100 * c / rows["closed"].max(), 100 * h / rows["hinge"].max(),
            100 * r / rows["rigid"].max(), ref["force"][j]))
    for s, f in (("closed", f_cl), ("hinge", ol["hinge"]["force"]), ("rigid", ol["rigid"]["force"])):
        ff = f[:i1 + 1]
        pos = np.clip(ff, 0, None)
        print("%-7s peak %5.0f N at %4.1f %% of the drive's time; centre %.3f; impulse %.1f N s"
              % (s, ff.max(), 100 * tf[np.argmax(ff)], np.sum(pos * tf[:i1 + 1]) / np.sum(pos),
                 np.trapezoid(ff, t[:i1 + 1])))
    bend_ol = ol["hinge"]["bend"]
    print("\nbend, deg, per 10 %: closed " + " ".join("%5.2f" % np.degrees(v) for v in per_tenth(tf, bend_cl, i1)))
    print("                     open   " + " ".join("%5.2f" % np.degrees(v) for v in per_tenth(tf, bend_ol, i1)))
    # the release, finely: from 70 % of the drive's time to 20 % into the recovery
    print("\nthrough the release (closed loop): time %, handle force N, bend deg, wet")
    t_end = 1.2 * t[i1]
    for k in range(t.size):
        if not (0.7 * t[i1] <= t[k] <= t_end) or k % 3:
            continue
        print("  %5.1f%%  %6.0f   %6.2f   %s" % (100 * t[k] / t[i1], f_cl[k], np.degrees(bend_cl[k]),
                                                "wet" if wet_cl[k] else "out"))
    print("open-loop hinge release at %.1f %% of the drive's time; closed-loop release at %.1f %%"
          % (100 * tf[np.flatnonzero(ol["hinge"]["wet"])[-1]], 100 * tf[np.flatnonzero(wet_cl)[-1]]))


if __name__ == "__main__":
    main()
