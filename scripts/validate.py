"""Validation report: every model output with an independent anchor.

Run it and read the FAIL rows.  The point of this file is that the
failures stay visible -- a model is only worth optimising if you know
which of its numbers you can spend.

    python scripts/validate.py
"""
import sys

import numpy as np

from coxswain.boats import catalog
from coxswain.crew.kinematics import SEGMENT_ORDER
from coxswain.hydro.addedmass import AddedMass, surge_coefficient
from coxswain.sim.control import Coxswain
from coxswain.sim.simulator import RowingSimulator

ROWS = []


def check(group, quantity, model, low, high, source):
    ok = (model is not None) and (low <= model <= high)
    ROWS.append((group, quantity, model, low, high, ok, source))


def steady(boat, rate_speed=4.0, cycles=9):
    period = boat.timing.period
    res = RowingSimulator(boat).run(duration=cycles * period, dt=0.005,
                                    surge_speed=rate_speed)
    keep = res.time >= (cycles - 2) * period
    return res, np.asarray(res.speed, float)[keep]


# -- speed, by class --------------------------------------------------------
two = catalog.double_scull(rate=24.0)
_, v2 = steady(two)
check("speed", "2x @ 24 spm, m/s", float(v2.mean()), 3.6, 4.1,
      "DGPS, same session: 3.82 (7 logs, 3.39-4.19)")

eight = catalog.eight(rate=32.0)
_, v8 = steady(eight, 4.8)
check("speed", "8+ @ 32 spm, m/s", float(v8.mean()), 4.9, 5.6,
      "club eight cruises ~5.2 at r32")

single = catalog.single_scull(rate=30.0)
_, v1 = steady(single, 4.3)
check("speed", "1x @ 30 spm, m/s", float(v1.mean()), 4.0, 4.7,
      "club sculler ~4.2-4.6")

check("speed", "8+ faster than 2x", float(v8.mean() - v2.mean()), 0.5, 3.0,
      "an eight is the fastest boat afloat")

# -- fluctuation ------------------------------------------------------------
ivv2 = 100.0 * float(np.ptp(v2) / v2.mean())
check("fluctuation", "2x IVV, %", ivv2, 33.0, 45.0,
      "DGPS 37.3; accel-derived 41.1; Day et al. ~20% about mean")

# -- crew kinematics --------------------------------------------------------
rower = two.crew[0].rower
period = rower.timing.period
t = np.linspace(0.0, period, 801)
trunk = np.degrees([rower.joint_angles.trunk(x).value for x in t])
check("crew", "trunk link swing, deg", float(np.ptp(trunk)), 45.0, 58.0,
      "Kleshnev 50.8; pelvis IMU 37 + spinal flexion")
check("crew", "seat travel, m", float(rower.slide_travel()), 0.58, 0.72,
      "0.60-0.70 across the literature")

masses = np.asarray(rower.segment_masses, float)
X = np.array([np.asarray(rower.segment_state(x)[0])[:, 0] for x in t])
com = (masses[None, :] * X).sum(axis=1) / masses.sum()
check("crew", "crew CoM travel, m", float(np.ptp(com)), 0.55, 0.68,
      "momentum balance from measured hull swing implies ~0.60")

share = dict(zip(SEGMENT_ORDER, masses / masses.sum()))
check("crew", "trunk mass fraction", float(sum(v for k, v in share.items()
                                               if "trunk" in k)),
      0.40, 0.47, "de Leva 1996: 0.435")

# -- added mass -------------------------------------------------------------
check("added mass", "Lamb k1, sphere", surge_coefficient(1.0, 1.0),
      0.48, 0.52, "Lamb 1932 art.71: 0.500")
check("added mass", "Lamb k1, 5:1 spheroid", surge_coefficient(5.0, 1.0),
      0.05, 0.07, "Lamb 1932 art.71: 0.059")
am8 = AddedMass.from_offsets(eight.offsets, rho=eight.water.density)
check("added mass", "8+ sway added / mass",
      float(am8.matrix[1, 1] / eight.total_mass), 0.4, 1.2,
      "strip theory; comparable to displacement for a slender hull")

# -- steering ---------------------------------------------------------------
def turn_rate(boat, rudder_deg, cycles=6):
    cox = Coxswain(rudder_override=lambda tt, s: np.radians(rudder_deg))
    sim = RowingSimulator(boat, coxswain=cox)
    res = sim.run(duration=cycles * boat.timing.period, dt=0.008,
                  surge_speed=5.0)
    tt = res.time
    yaw = np.degrees(np.unwrap(np.asarray(res.attitude)[2]))
    keep = tt >= 3 * boat.timing.period
    return abs(float(np.polyfit(tt[keep], yaw[keep], 1)[0]))

e28 = catalog.eight(rate=28.0)
check("steering", "8+ full rudder, deg/s", turn_rate(e28, 25.0), 2.0, 4.0,
      "coxswain: ~15 deg in 5 s, so about 3 deg/s at most")
check("steering", "8+ typical rudder, deg/s", turn_rate(e28, 8.0), 0.7, 1.6,
      "coxswain: about 1 deg/s")

import copy
bare = copy.copy(e28)
object.__setattr__(bare, "appendages", ())
sim = RowingSimulator(bare, coxswain=Coxswain(rudder_override=lambda a, b: 0.0))
y0 = sim.initial_state(surge_speed=5.0)
y0[5] = np.radians(2.0)
res = sim.run(duration=25.0, dt=0.008, initial_state=y0)
drift = abs(float(np.degrees(np.unwrap(np.asarray(res.attitude)[2]))[-1]))
check("steering", "skeg+rudder lost, deg in 25 s", drift, 20.0, 90.0,
      "ACRAs: crossed a lane and hit another boat in 20-30 s")

# -- attitude ---------------------------------------------------------------
res8, _ = steady(eight, 4.8)
keep = res8.time >= 7 * eight.timing.period
check("attitude", "8+ roll swing, deg",
      float(np.ptp(np.degrees(np.asarray(res8.roll)[keep]))), 0.0, 5.0,
      "reported <5 deg within-stroke for racing shells")
check("attitude", "8+ pitch swing, deg",
      float(np.ptp(np.degrees(np.asarray(res8.pitch)[keep]))), 0.0, 1.5,
      "reported <1 deg within-stroke")

# -- report -----------------------------------------------------------------
width = max(len(r[1]) for r in ROWS) + 2
print()
print("%-13s %-*s %10s %16s  %-6s %s"
      % ("group", width, "quantity", "model", "expected", "verdict", "anchor"))
print("-" * 132)
last = None
fails = 0
for group, quantity, model, low, high, ok, source in ROWS:
    if group != last:
        print()
        last = group
    fails += (not ok)
    print("%-13s %-*s %10.3f %7.2f-%-8.2f %-6s %s"
          % (group, width, quantity, model, low, high,
             "ok" if ok else "FAIL", source))
print()
print("%d of %d checks pass; %d FAIL" % (len(ROWS) - fails, len(ROWS), fails))
sys.exit(0)
