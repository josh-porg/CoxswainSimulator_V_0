"""The on-water crew driver (sprint 1 #3): what two on-water scullers share, in the model.

    python research/biorow/onwater_driver.py [--strokes 16]

Two settings, both from what [BR24] and [CR06] share against the ergometer body
(SOURCES sec. 160):

  drive length       OnWaterTiming: Kleshnev (2005)'s on-water single-scull rhythm at two
                     rates, which predicts both athletes to 0.006 (sourced)
  recovery swing     the recovery traverse's arrival parameter (crew.stroke.recovery_warp),
                     FITTED here to the mean of the two athletes' recovery curves on the
                     split clock: the one fitted number in the driver, and marked so

Then the like-for-like on [BR24] at his 432 W: ergometer body, on-water timing only, and
the full on-water driver. The model's catalogue arc and posture are unchanged, so any
change is the timing alone.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import like_for_like as L                                        # noqa: E402
import timelaw_compare as TC                                     # noqa: E402

from coxswain.crew.kinematics import SegmentSequencing          # noqa: E402
from coxswain.crew.stroke import OnWaterTiming, StrokeTiming    # noqa: E402
from coxswain.validation.scorecard import settle_dynamic        # noqa: E402

ARRIVALS = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4)


def boat(timing, arrival, trunk_lag=0.0):
    """His boat (like_for_like's), rebuilt with a given timing and recovery arrival."""
    from coxswain import physics
    from coxswain.boats import catalog
    from coxswain.boats.rig import Oarlock, Rig, Seat
    from coxswain.crew.anthropometry import PORT, STARBOARD, RowerAnthropometry
    ref = catalog.single_scull(rate=L.RATE, rower_mass=L.MASS, rower_stature=L.STATURE)
    locks = tuple(Oarlock(position=np.array([-0.35 + L.WORK_THROUGH, side * 0.80, 0.32]),
                          side=side, oar=L.OAR) for side in (PORT, STARBOARD))
    b = type(ref)(name="1x [BR24] on-water", offsets=ref.offsets,
                  rig=Rig(seats=(Seat(station_x=-0.35, oarlocks=locks, label="stroke"),)),
                  hull_mass=L.HULL, hull_inertia=ref.hull_inertia, timing=timing,
                  appendages=ref.appendages, water=ref.water, force_profile=ref.force_profile,
                  oar_sweep=catalog.SCULLING_ARC,
                  default_anthropometry=RowerAnthropometry(mass=L.MASS, stature=L.STATURE),
                  recovery_arrival=arrival,
                  sequencing=(SegmentSequencing(trunk=-trunk_lag) if trunk_lag else None))
    physics.resolve("research").apply(b)
    return b


def model_curves(b):
    r = b.crew[0].rower
    T = b.timing.period
    t = np.linspace(0, T, 1200, endpoint=False)
    ch = r._chain(t)
    hip, ank, sh = ch["hip"][0].value, ch["ankle"][0].value, ch["shoulder"][0].value
    ang = np.array([float(b.oar_sweep(x, b.timing)) for x in t])
    a = dict(name="model", T=T, leg=TC.periodic(t, hip - ank, T), back=TC.periodic(t, sh - hip, T),
             t_catch=t[np.argmax(ang)], t_finish=t[np.argmin(ang)])
    return TC.curves(a)


def fit_arrival():
    his, her = TC.curves(TC.athlete_br24()), TC.curves(TC.athlete_cr06())
    rec = TC.G >= 0.5
    target = {ch: 0.5 * (his[ch]["split"] + her[ch]["split"]) for ch in ("leg", "back")}
    rows = []
    for a in ARRIVALS:
        try:
            c = model_curves(boat(OnWaterTiming(L.RATE), a))
        except ValueError as e:                      # unreachable handle at this timing
            rows.append((a, np.nan, np.nan, str(e)[:60]))
            continue
        e_leg = TC.rms(c["leg"]["split"][rec], target["leg"][rec])
        e_back = TC.rms(c["back"]["split"][rec], target["back"][rec])
        rows.append((a, e_leg, e_back, ""))
    print("recovery arrival fit (rms against the two athletes' mean, recovery half of the split clock)")
    for a, el, eb, note in rows:
        print("  arrival %.1f   leg %.3f   back %.3f   %s" % (a, el, eb, note))
    ok = [r for r in rows if np.isfinite(r[1])]
    best = min(ok, key=lambda r: r[1] + r[2])
    print("best arrival %.1f" % best[0])
    return best[0]


def settle(b, strokes):
    s = settle_dynamic(b, L.POWER, start=4.6, strokes=strokes)
    kin = L.kinematics(b)
    return s.speed, s.surge_swing, s.crew_power, kin["com_travel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=16)
    ap.add_argument("--arrival", type=float, default=None)
    ap.add_argument("--trunk-lag", type=float, default=0.1,
                    help="trunk sequencing lag (fraction of cycle); 0.1 fits the shared recovery")
    a = ap.parse_args()
    arrival = a.arrival if a.arrival is not None else fit_arrival()
    lag = a.trunk_lag
    print("\n[BR24] at %.0f W: speed 4.641, IVV 49.1%%, CoM travel 0.71-0.74" % L.POWER)
    for name, b in (("ergometer body", boat(StrokeTiming(L.RATE), 1.0)),
                    ("on-water drive length", boat(OnWaterTiming(L.RATE), 1.0)),
                    ("on-water driver (+ arrival %.1f)" % arrival, boat(OnWaterTiming(L.RATE), arrival)),
                    ("drive length + trunk lag %.1f" % lag, boat(OnWaterTiming(L.RATE), 1.0, lag)),
                    ("ergometer timing + trunk lag %.1f" % lag, boat(StrokeTiming(L.RATE), 1.0, lag))):
        v, ivv, p, com = settle(b, a.strokes)
        print("  %-34s speed %.3f  IVV %.1f%%  power %.0f W  CoM travel %.3f  drive %.3f"
              % (name, v, 100 * ivv, p, com, b.timing.drive_fraction), flush=True)


if __name__ == "__main__":
    main()
