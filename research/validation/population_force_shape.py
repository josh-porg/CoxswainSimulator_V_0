"""The population's stroke on three shafts: how far does the shaft's stiffness move the force peak? (§236, §237)

Rung 1 for the population (rung2_population.population_sim: body, rhythm and handle velocity imposed, hands on the
handle), so the handle force is the oar's and the blade's response alone. Rigid, [L14]'s medium scull shaft and its
extra-soft shaft (BendDynamics, kappa 0.1, the research blade); the time of the force peak in the drive, against
[L14] Table 12's on-water trial (four scullers x 20 strokes: 34-47 % of the drive; extra-soft minus medium +0.2 points,
rowers +1.4 / +0.4 / +0.3 / -1.3), and the force per 10 % of drive time against RBN 2023/12 (rbn2023.py; the figure's
axis is time: SOURCES §237 -- §236 compared it on drive length, which was wrong).

    python research/validation/population_force_shape.py [--drive k05|rbn2023]
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
import rbn2023                                                   # noqa: E402

from coxswain.core.state import STATE_SIZE                       # noqa: E402


def population_run(strokes, drive="k05", body_drive=None):
    """The population at rung 1. ``drive``: "k05" -- [K05] Fig. 1's handle speed (as built); "rbn2023" -- the handle
    velocity of RBN 2023/12's reference curves (rbn2023.drive_sweep; SOURCES §237). ``body_drive``: the body's legs
    and trunk through the drive, the same two choices (default: as ``drive``)."""
    _boat, sim = RP.population_sim(drive, body_drive)             # one builder, rung 2's own (S1)
    return sim, sim.run_strokes(strokes, surge_speed=4.3)


def shape(sim, run):
    """(time fraction, force / peak) from the catch (largest gate angle) to the finish."""
    t, force, _wet = HFS.model_stroke(sim, run)
    gate = run.last_states[STATE_SIZE]
    i0, i1 = int(np.argmax(gate)), int(np.argmin(gate))
    F, tt = force[i0:i1 + 1], t[i0:i1 + 1] - t[i0]
    return tt / tt[-1], F / F.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--drive", default=RP.POP_DRIVE, choices=["k05", "rbn2023"])
    a = ap.parse_args()
    from coxswain.crew.handle_fv import HandleForceVelocity
    from coxswain.crew.oar_bending import L14_EXTRA_SOFT_EI, L14_MEDIUM_EI, BendDynamics
    RP.FV = HandleForceVelocity(4.53, "--fv")
    RP.HOLD, RP.HOLD_OWN = 0.0, True
    RP.BLADE, RP.BODY = "crossflow_normal", "model"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    RP.ADDED_MASS, RP.SURFACE_FACTOR = "patton", 0.1
    rows, peaks = {}, {}
    for label, ei in (("rigid", None), ("medium", L14_MEDIUM_EI), ("extra-soft", L14_EXTRA_SOFT_EI)):
        RP.BENDING = ei is not None
        RP.SHAFT_EI = ei
        RP.BEND_DYNAMICS = BendDynamics() if RP.BENDING else None
        sim, run = population_run(a.strokes, a.drive)
        rows[label] = shape(sim, run)
        peaks[label] = 100 * rows[label][0][np.argmax(rows[label][1])]
        print("%s: %s" % (label, RP.C2.summary(run)), flush=True)
    print("\n%s hands; peak force by time, %% of the drive: %s" % (a.drive, ", ".join("%s %.1f" % kv for kv in peaks.items())))
    print("[L14] Table 12: 34-47 %%; extra-soft minus medium +0.2 (rowers +1.4 / +0.4 / +0.3 / -1.3); model %+.1f"
          % (peaks["extra-soft"] - peaks["medium"]))
    ref = rbn2023.reference()
    print("\nhandle force, % of peak, per 10 % of the drive's time:")
    print(" time | RBN 2023/12 | " + " | ".join("%-10s" % n for n in rows))
    for q, f in zip(ref["time"], ref["force"]):
        print(" %3.0f%% |    %5.0f    | " % (q, f) +
              " | ".join("%10.0f" % (100 * np.interp(q / 100.0, x, y)) for x, y in rows.values()))


if __name__ == "__main__":
    main()
