"""The population stroke against Kleshnev's population reference curves, by drive time, and the blade's slip (§237).

RBN 2023/12 Fig. 1 (>50 000 samples; extract_rbn2023.py -> rbn2023_fig1.csv) gives, per 10 % of the **time** of the
drive: handle force and handle velocity as % of their peaks, and seat velocity as % of the peak handle velocity. The
model's population run ([K05] body, rhythm and drive law imposed; rung 1) is read the same way: the drive from the
catch (largest gate angle) to the finish (smallest), handle velocity = gate rate x inboard, seat velocity = the lower
trunk's fore-and-aft velocity in the boat, drive length = the gate angle's fraction. Also the blade centre's normal slip
split into the gate's motion, the boat's and the shaft's unbending.

    python research/validation/population_kinematics.py [--shaft rigid|medium]
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
from population_force_shape import population_run               # noqa: E402

from coxswain.core.state import STATE_SIZE, State                # noqa: E402

from rbn2023 import reference                                   # noqa: E402


def table(sim, run):
    """Per sample of the last stroke's drive: (time fraction, drive-length fraction, columns R)."""
    t, force, _wet = HFS.model_stroke(sim, run)
    y = run.last_states
    n = sim.n_oar_states
    gate, rate = y[STATE_SIZE], y[STATE_SIZE + n]
    hist = getattr(sim, "_bend_history", None) if sim.oar_bending is not None else None
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    oar = lock.oar
    inboard, arm = float(oar.inboard), float(sim._oars[0].outboard)
    i0, i1 = int(np.argmax(gate)), int(np.argmin(gate))
    rows = []
    t0 = float(run.last_time[0])
    for k in range(i0, i1 + 1):
        state = State.from_vector(y[:STATE_SIZE, k])
        phi, phid = (float(hist[0, 0, k]), float(hist[1, 0, k])) if hist is not None else (0.0, 0.0)
        lever = float(sim._bend_lever[0]) if hist is not None else 0.0
        blade = sim._liftdrag[0]
        u = sim._lock_velocity(state, lock)[:2]
        wn_blade, wa_blade = blade.relative_velocity(gate[k] + phi, rate[k] + lever * phid, u, int(lock.side))
        wn_boat, _ = blade.relative_velocity(gate[k] + phi, 0.0, u, int(lock.side))
        _m, _p, vel, _a = sim._body(t0 + t[k])
        seat = abs(float(np.asarray(vel)[3, 0]))
        rows.append((gate[k], abs(rate[k]) * inboard, seat, force[k], arm * rate[k], wn_boat,
                     arm * lever * phid, wn_blade, np.degrees(np.arctan2(abs(wn_blade), abs(wa_blade))),
                     float(y[6, k]), np.degrees(phi)))
    R = np.array(rows)
    tt = t[i0:i1 + 1] - t[i0]
    return tt / tt[-1], (R[0, 0] - R[:, 0]) / (R[0, 0] - R[-1, 0]), R


def compare(tf, dl, R, label):
    """Print the model against RBN 2023/12 per 10 % of drive time; returns the rms differences (force, handle v)."""
    ref = reference()
    vh, vs, F = R[:, 1], R[:, 2], R[:, 3]
    q = ref["time"] / 100.0
    g = lambda x: np.interp(q, tf, x)                            # noqa: E731
    mF, mvh, mvs, mdl = 100 * g(F) / F.max(), 100 * g(vh) / vh.max(), 100 * g(vs) / vh.max(), 100 * g(dl)
    print("\n%s against RBN 2023/12, per 10 %% of the drive's time (population | model):" % label)
    print(" time | force %      | handle v %   | seat v % of peak handle | drive length %")
    for i in range(q.size):
        print(" %3.0f%% | %5.0f  %5.0f | %5.0f  %5.0f | %5.0f  %5.0f            | %5.0f  %5.0f" % (
            ref["time"][i], ref["force"][i], mF[i], ref["handle_v"][i], mvh[i], ref["seat_v"][i], mvs[i],
            ref["length"][i], mdl[i]))
    rms = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))    # noqa: E731
    out = dict(force_rms=rms(mF, ref["force"]), handle_rms=rms(mvh, ref["handle_v"]),
               seat_rms=rms(mvs, ref["seat_v"]), force_peak_time=float(100 * tf[np.argmax(F)]),
               handle_peak_time=float(100 * tf[np.argmax(vh)]))
    print("rms difference: force %.1f, handle velocity %.1f, seat velocity %.1f points; force peak at %.0f %% of the "
          "drive's time (population 40-50), handle velocity peak at %.0f %% (60)"
          % (out["force_rms"], out["handle_rms"], out["seat_rms"], out["force_peak_time"], out["handle_peak_time"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shaft", default="medium", choices=["rigid", "medium"])
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--drive", default="k05", choices=["k05", "rbn2023"],
                    help="the imposed handle velocity: [K05] Fig. 1 (as built) or RBN 2023/12's reference curve")
    a = ap.parse_args()
    from coxswain.crew.handle_fv import HandleForceVelocity
    from coxswain.crew.oar_bending import BendDynamics
    RP.FV = HandleForceVelocity(4.53, "--fv")
    RP.HOLD, RP.HOLD_OWN = 0.0, True
    RP.BLADE, RP.BODY = "crossflow_normal", "model"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    RP.ADDED_MASS, RP.SURFACE_FACTOR = "patton", 0.1
    RP.BENDING = a.shaft == "medium"
    RP.BEND_DYNAMICS = BendDynamics() if RP.BENDING else None
    sim, run = population_run(a.strokes, a.drive)
    tf, dl, R = table(sim, run)
    print("population, rung 1, %s shaft, %s hands: %s" % (a.shaft, a.drive, RP.C2.summary(run)))
    out = compare(tf, dl, R, "population, %s shaft, %s hands" % (a.shaft, a.drive))
    import json
    json.dump(dict(out, shaft=a.shaft, drive=a.drive, summary=RP.C2.summary(run)),
              open(os.path.join(ROOT, "data", "local", "validation", "population_kinematics_%s_%s.json"
                                % (a.shaft, a.drive)), "w"), indent=1)
    print("\n time | gate deg | blade speed from the gate | boat's part | unbending | slip w_n | incidence deg | boat "
          "m/s | bend deg | force N")
    for q in np.arange(0.0, 1.01, 0.1):
        g = lambda j: float(np.interp(q, tf, R[:, j]))           # noqa: E731
        print(" %3.0f%% |  %6.1f  |        %6.2f             |   %6.2f    |  %6.2f   |  %6.2f  |    %5.1f      |  %5.2f   |"
              "  %5.2f   | %5.0f" % (100 * q, np.degrees(g(0)), g(4), g(5), g(6), g(7), g(8), g(9), g(10), g(3)))


if __name__ == "__main__":
    main()
