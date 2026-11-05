"""Does the bending oar bring the handle force closer to [CR06]'s measured one? (SOURCES §218)

Rung 1 imposes the oar's motion (her arc, the population drive law, the on-water rhythm; hands on the handle), so the
handle force is an output: the constraint's reaction, the oar's inertia minus the blade's moment. [CR06] measured her
handle force (Fig. 3, both hands summed) and her oar angle, so the comparison is force against oar angle through the
drive -- no timing alignment needed. Rigid against [L14]'s measured scull shaft; the bend is read per sample from the
run's own history (STRUCTURAL_ISSUES S1).

    python research/blade/bending_handle_force.py [--strokes 10]

Prints, per oar angle bin through the drive, the measured force per hand and the model's, and the rms difference over
the drive, its rms as a fraction of the measured peak, and the angle of the peak.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow")]
import rung2_population as RP                                    # noqa: E402

from coxswain.core.state import STATE_SIZE                       # noqa: E402
from coxswain.crew.oar_bending import OarBending                 # noqa: E402

V = RP.V


def measured_curve():
    """Her handle force per hand against oar angle through the drive (catch to finish), degrees and N."""
    g = np.linspace(0.0, V.CR06_T, 4000, endpoint=False)
    angle = V._cr06_series("oar_angle_deg")(g)
    force = 0.5 * V._cr06_series("handle_force_N")(g)          # Fig. 3 sums both hands
    catch = int(np.argmax(angle))
    order = np.roll(np.arange(g.size), -catch)
    angle, force = angle[order], force[order]
    finish = int(np.argmin(angle))
    return angle[:finish + 1], force[:finish + 1]


def model_curve(bending, strokes):
    m = V.ATHLETES["cr06"][0]()
    boat = RP.C2.R.boat_for("cr06", m, "on-water", drive_law=True)
    boat.oar_bending = bending
    sim = RP.make(boat, "reference")
    run = sim.run_strokes(strokes, surge_speed=4.3)
    t, y = run.last_time, run.last_states
    inboard = float(boat.rig.seats[sim._seats[0]].oarlocks[0].oar.inboard)
    hist = getattr(sim, "_bend_history", None)
    saved = (sim._bend.copy(), sim._bend_rate.copy())
    angle, force = [], []
    for k in range(t.size):
        if sim._air_mask[0, k]:
            continue
        if hist is not None and sim.oar_bending is not None:
            sim._bend[:], sim._bend_rate[:] = hist[0, :, k], hist[1, :, k]
        torque = float(np.asarray(sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k]))[0, 0])
        angle.append(np.degrees(float(y[STATE_SIZE, k])))      # the gate angle, as she measured it
        force.append(-torque / inboard)
    sim._bend[:], sim._bend_rate[:] = saved
    angle, force = np.array(angle), np.array(force)
    keep = np.concatenate([[True], np.diff(angle) < 0])           # the drive: the angle falls
    return angle[keep], force[keep], RP.C2.summary(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    args = ap.parse_args()
    RP.BLADE = "crossflow_normal"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    ma, mf = measured_curve()
    oar = None
    rows = {}
    for label in ("rigid", "bending"):
        m = V.ATHLETES["cr06"][0]()
        boat = RP.C2.R.boat_for("cr06", m, "on-water", drive_law=True)
        oar = boat.rig.seats[0].oarlocks[0].oar
        law = (OarBending.laschowski2014_scull(float(oar.blade_centre_outboard), float(oar.blade_length))
               if label == "bending" else None)
        rows[label] = model_curve(law, args.strokes)
    lo, hi = max(min(ma), max(min(r[0]) for r in rows.values())), min(max(ma), min(max(r[0]) for r in rows.values()))
    bins = np.linspace(hi, lo, 21)
    mref = np.interp(bins, ma[::-1], mf[::-1])
    print("[CR06] handle force per hand against oar angle, catch to finish (N); rung 1, her rig and rhythm")
    print("  angle  measured   rigid  bending")
    curves = {lab: np.interp(bins, r[0][::-1], r[1][::-1]) for lab, r in rows.items()}
    for i, b in enumerate(bins):
        print("  %5.1f  %7.0f  %7.0f  %7.0f" % (b, mref[i], curves["rigid"][i], curves["bending"][i]))
    peak = float(mref.max())
    for lab, c in curves.items():
        s = rows[lab][2]
        print("%-8s rms %.0f N (%.1f%% of her peak %.0f N); peak at %.1f deg (hers %.1f); %.3f m/s, IVV %.1f%%, %.0f W"
              % (lab, np.sqrt(np.mean((c - mref) ** 2)), 100 * np.sqrt(np.mean((c - mref) ** 2)) / peak, peak,
                 bins[int(np.argmax(c))], bins[int(np.argmax(mref))], s["speed"], 100 * s["ivv"], s["power"]))


if __name__ == "__main__":
    main()
