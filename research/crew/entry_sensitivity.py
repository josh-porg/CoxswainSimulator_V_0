"""Is the handle force at blade entry set by the drive law's smoothing, or by the hull?

    python research/crew/entry_sensitivity.py [--strokes 10]

The [K05] scan does not resolve the handle speed at the catch, so the hands' acceleration there
comes from the smoothing spline (SOURCES sec. 171). Blade added mass loads the blade by its normal
acceleration, w_n_dot = g . X_h + l phi_ddot + c: the hands' part l phi_ddot and the hull's part
g . X_h + c (surge deceleration at the catch, rotation). This splits it at entry and over the
first 0.1 s of blade-in, and reruns rung 1 with the smoothing weighted by 0.02, 0.03 (the
digitisation's stated error) and 0.05 m/s, with and without Patton's added mass.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import handle_rung1 as R                                         # noqa: E402

from coxswain.core.rigid_body import solve_accelerations       # noqa: E402
from coxswain.core.state import STATE_SIZE, State              # noqa: E402
from coxswain.crew.drive_law import PopulationDriveSweep       # noqa: E402

V = R.V


def split_entry(result):
    """Per wet sample in the first 0.1 s: (tau, w_n, hands' part, hull's part), m/s and m/s^2."""
    sim, run = result["sim"], result["run"]
    t, y = run.last_time, run.last_states
    n = sim.n_oar_states
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    wet = np.flatnonzero(~sim._air_mask[0])
    rows = []
    for k in wet:
        tau = t[k] - t[wet[0]]
        if tau > 0.1:
            break
        state = State.from_vector(y[:STATE_SIZE, k])
        angle = float(y[STATE_SIZE, k])
        rate, accel = sim._sweep_motion(t[k] - sim._stroke_start)
        saved = sim._in_air
        sim._in_air = sim._air_mask[:, k].copy()
        try:
            system, rhs = sim._coupled_system(t[k], state, y[STATE_SIZE:STATE_SIZE + n, k],
                                              y[STATE_SIZE + n:STATE_SIZE + 2 * n, k])
        finally:
            sim._in_air = saved
        hull = solve_accelerations(system, rhs)[:6]
        _m, arm, g, c, w_n, _h = sim._blade_mass_terms(
            0, lock, angle, rate, state, state.rot_hull_to_abs,
            np.asarray(state.omega, float), np.asarray(state.omega_hull, float))
        rows.append((tau, w_n, arm * accel, float(g @ hull) + c))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    a = ap.parse_args()
    for name in ("br24", "cr06"):
        measure, _build = V.ATHLETES[name]
        m = measure()
        print("\n%s" % name, flush=True)
        for error in (0.02, 0.03, 0.05):
            for mass in ("none", "patton"):
                boat = R.boat_for(name, m, "on-water", True)
                sw = boat.oar_sweep
                boat.oar_sweep = PopulationDriveSweep(catch_angle=sw.catch_angle, finish_angle=sw.finish_angle,
                                                      speed_error=error)
                span = float(boat.rig.seats[0].oarlocks[0].oar.blade_length)
                r = R.predict(boat, a.strokes, blade_law="liftdrag", blade_span=span, blade_added_mass=mass)
                d = R.force_descriptors(r)
                print("  error %.2f %-6s IVV %.1f%%, %.0f W; entry/peak %.2f, catch-to-peak %.2f s, peak/mean %.2f, "
                      "peak %.0f N" % (error, mass, 100 * r["ivv"], r["power"], d["entry_fraction"],
                                       d["catch_to_peak"], d["peak_over_mean"], d["peak"]), flush=True)
                if mass == "patton":
                    for tau, w_n, hands, hull in split_entry(r)[::2]:
                        print("      +%.3f s  w_n %+.3f m/s  w_n_dot: hands %+6.2f, hull %+6.2f m/s^2"
                              % (tau, w_n, hands, hull), flush=True)


if __name__ == "__main__":
    main()
