"""The population's own stroke against the population's force curve: is the late peak the oar and blade? (§236)

Rung 1 for [K05]'s population (rung2_population.population_tables' run: their body, rhythm and drive law imposed, hands
on the handle), so the handle force is purely the oar's and the blade's response to a population hand motion. Its
force against drive length is compared with Kleshnev's population curve (RBN 2023/12): 13 / 22 / 60 / 100 / 74 / 0 %
of peak at 0 / 10 / 20 / 40 / 60 / 90 % of drive length. Drive length is on the gate (as BioRow measure it) and, for
the flexible shaft, also on the bent blade's own angle. Rigid and flexible (BendDynamics, κ 0.1), the research blade.

    python research/validation/population_force_shape.py [--release-slip 16.6]
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
from force_vs_drive_length import BIOROW_2023                    # noqa: E402

from coxswain.core.state import STATE_SIZE                       # noqa: E402


def population_run(strokes):
    pop_boat = RP.V.cr06_boat(dict(leg_travel=None), timing=RP.OnWaterTiming(RP.KP.K05_RATE),
                              stature=RP.KP.K05_STATURE, drive_law=True, mass=RP.KP.K05_MASS)
    body, _k = RP.k05_body.body_field(pop_boat, leg_travel=0.51)
    sim = RP.make(pop_boat, "reference", body=body)
    return sim, sim.run_strokes(strokes, surge_speed=4.3)


def shape(sim, run):
    """(drive length on the gate, on the blade, force / peak) from the catch (largest gate angle) to the finish."""
    t, force, wet = HFS.model_stroke(sim, run)
    y = run.last_states
    gate = y[STATE_SIZE]
    hist = getattr(sim, "_bend_history", None) if sim.oar_bending is not None else None
    blade = gate + (hist[0, 0] if hist is not None else 0.0)
    i0, i1 = int(np.argmax(gate)), int(np.argmin(gate))
    g, b, F = gate[i0:i1 + 1], blade[i0:i1 + 1], force[i0:i1 + 1]
    span = g[0] - g[-1]
    tt = t[i0:i1 + 1] - t[i0]
    return (g[0] - g) / span, (b[0] - b) / span, F / F.max(), tt / tt[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--release-slip", type=float, default=None)
    a = ap.parse_args()
    from coxswain.crew.handle_fv import HandleForceVelocity
    from coxswain.crew.oar_bending import BendDynamics
    RP.FV = HandleForceVelocity(4.53, "--fv")
    RP.HOLD, RP.HOLD_OWN = 0.0, True
    RP.BLADE, RP.BODY = "crossflow_normal", "model"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    RP.ADDED_MASS, RP.SURFACE_FACTOR, RP.RELEASE_SLIP = "patton", 0.1, a.release_slip
    from coxswain.crew.oar_bending import L14_EXTRA_SOFT_EI, L14_MEDIUM_EI
    rows, peaks_t = {}, {}
    for label, ei in (("rigid", None), ("medium", L14_MEDIUM_EI), ("extra-soft", L14_EXTRA_SOFT_EI)):
        RP.BENDING = ei is not None
        RP.SHAFT_EI = ei
        RP.BEND_DYNAMICS = BendDynamics() if RP.BENDING else None
        sim, run = population_run(a.strokes)
        dg, db, f, ft = shape(sim, run)
        rows["%s (gate)" % label] = (dg, f)
        if ei is not None:
            rows["%s (blade)" % label] = (db, f)
        peaks_t[label] = 100 * ft[np.argmax(f)]
        print("%s: %s" % (label, RP.C2.summary(run)), flush=True)
    print("\npeak force by time, %% of the drive (catch to finish): %s" %
          ", ".join("%s %.1f" % kv for kv in peaks_t.items()))
    print("[L14] Table 12 (4 scullers, 20 strokes each): 34-47 %%; extra-soft minus medium +0.2 (rowers +1.4 / +0.4 / "
          "+0.3 / -1.3); model extra-soft minus medium %+.1f" % (peaks_t["extra-soft"] - peaks_t["medium"]))
    print("\n[K05] population at rung 1 (hand motion imposed): handle force, % of peak, by drive length")
    print("drive length | RBN 2023/12 | " + " | ".join(rows))
    for q, b in BIOROW_2023:
        print("   %3.0f %%     |   %3.0f %%    | " % (100 * q, b) +
              " | ".join("%7.0f %%" % (100 * np.interp(q, x, y)) for x, y in rows.values()))
    for name, (x, y) in rows.items():
        print("%s: peak at %.0f %% of drive length (population 40 %%)" % (name, 100 * x[np.argmax(y)]))


if __name__ == "__main__":
    main()
