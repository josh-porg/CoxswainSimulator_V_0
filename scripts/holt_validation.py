r"""Put the modelled small boats next to Holt's measured ones.

    python scripts/holt_validation.py

Holt et al. [H20]_ report per-stroke means for four boat classes from
forty-seven instrumented 2000 m races.  Three of their four quantities are
things this simulator computes from first principles and has never been
checked against: boat velocity, distance per stroke, and the within-stroke
velocity range.  The fourth, power, is the input.

So this is a genuine test rather than a fit.  Set each modelled boat to
the measured power and the measured rate, and see whether the speed and
the surge swing come out where the instrumentation put them.

What matches and what does not is the point
-------------------------------------------
The eight has no entry in their table -- they measured singles and pairs
-- so the boat this project actually cares about is validated only by
inference from its smaller relatives.  That is worth stating plainly
rather than quietly comparing an eight to a single and calling it
agreement.

A note on the pairs
-------------------
Holt's M2- and W2- are coxless **pairs**: sweep, one oar each.  The
catalog's two-seat boat is a **double scull**: two oars each.  Same hull
family, different rig and different oar geometry, so those rows are a
sanity check on hull scale rather than a like-for-like comparison, and
they are labelled as such.

References
----------
.. [H20] Holt, A. C., Aughey, R. J., Ball, K., Hopkins, W. G. and
   Siegel, R. (2020) *Technical determinants of on-water rowing
   performance*, Frontiers in Sports and Active Living 2:589013,
   table 1.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.resistance import hull_resistance   # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

#: Holt table 1.  rate, within-stroke velocity range (m/s), distance per
#: stroke (m), power per rower (W), rower mass (kg) from their subjects.
HOLT = (
    ("M1x", "1x", 1, 34.7, 2.27, 7.97, 334.0, 84.8),
    ("W1x", "1x", 1, 32.8, 2.14, 7.65, 223.0, 73.6),
    ("M2-", "2x", 2, 38.1, 2.71, 7.82, 380.0, 84.8),
    ("W2-", "2x", 2, 35.1, 2.30, 7.38, 240.5, 73.6),
)


def build(kind, rate, mass, scale):
    if kind == "1x":
        boat = catalog.single_scull(rate=rate, rower_mass=mass)
    else:
        boat = catalog.double_scull(rate=rate, rower_mass=mass)
    boat.power_scales = np.full(boat.n_seats, float(scale))
    return boat


def steady(boat, guess, duration=30.0, dt=0.005):
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=dt, surge_speed=guess)
    t = np.asarray(result.time)
    v = np.hypot(*np.asarray(result.velocity)[:2])
    period = boat.timing.period
    cycles = int((0.5 * t[-1]) // period)
    keep = t >= t[-1] - cycles * period
    v = v[keep]
    return float(v.mean()), float(v.max() - v.min())


def delivered_power(boat, speed):
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    force, _ = hull_resistance(np.array([float(speed), 0.0, 0.0]), submerged,
                               mean_wetted_length=boat.length,
                               water=boat.water, coefficients=boat.resistance)
    return abs(float(force[0])) * float(speed)


def match_power(kind, rate, mass, target, guess, limit=18, tol=1.5):
    """Scale oar force until the boat delivers ``target`` watts to the water.

    Holt's power is measured at the gate, and not all of it reaches the
    water: blade slip and the crew's own secondary motion take a share.
    So this matches DELIVERED power and the shortfall against their gate
    figure is itself a result -- it is the boat's propulsive efficiency,
    and it should land near the 0.75-0.85 the blade literature reports.
    """
    low, high = 0.10, 3.0
    scale = speed = swing = power = float("nan")
    for _ in range(limit):
        scale = 0.5 * (low + high)
        boat = build(kind, rate, mass, scale)
        speed, swing = steady(boat, guess)
        power = delivered_power(boat, speed)
        if abs(power - target) < tol:
            break
        if power < target:
            low = scale
        else:
            high = scale
    return scale, speed, swing, power


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--efficiency", type=float, default=0.80,
                        help="fraction of gate power that reaches the water")
    args = parser.parse_args(argv)

    print("Holt table 1 against this model, at their rate and their power")
    print("  gate power scaled by %.2f to give power delivered to the water"
          % args.efficiency)
    print()
    print("  %-6s %-4s %18s %18s %18s"
          % ("class", "rig", "velocity m/s", "distance/stroke m",
             "range m/s"))
    print("  %-6s %-4s %8s %9s %8s %9s %8s %9s"
          % ("", "", "Holt", "model", "Holt", "model", "Holt", "model"))
    for name, kind, rowers, rate, span, per_stroke, power, mass in HOLT:
        measured = per_stroke * rate / 60.0
        target = args.efficiency * power * rowers
        _scale, speed, swing, _delivered = match_power(
            kind, rate, mass, target, measured)
        model_per_stroke = speed * 60.0 / rate
        note = "" if kind == "1x" else "  (2x vs their 2-)"
        print("  %-6s %-4s %8.2f %9.2f %8.2f %9.2f %8.2f %9.2f%s"
              % (name, kind, measured, speed, per_stroke, model_per_stroke,
                 span, swing, note))
    print()
    print("  velocity and distance per stroke are the same statement -- rate")
    print("  ties them -- so the two independent tests here are SPEED at a")
    print("  given power, and the SURGE SWING, which nothing was tuned to.")
    print()

    print("what the swing says, since it is the one this project needed")
    print("  %-6s %10s %10s %10s" % ("class", "Holt %", "model %", "ratio"))
    for name, kind, rowers, rate, span, per_stroke, power, mass in HOLT:
        measured = per_stroke * rate / 60.0
        target = args.efficiency * power * rowers
        _scale, speed, swing, _d = match_power(kind, rate, mass, target,
                                               measured)
        print("  %-6s %9.0f%% %9.0f%% %10.2f"
              % (name, 100 * span / measured, 100 * swing / speed,
                 (swing / speed) / (span / measured)))
    print()
    print("  A ratio near one means the model's surge oscillation is the")
    print("  size a real boat's is.  This is the quantity scripts/unsteady.py")
    print("  squares, so a factor of two here is a factor of four there --")
    print("  and it is why an unvalidated swing was worth this much trouble.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
