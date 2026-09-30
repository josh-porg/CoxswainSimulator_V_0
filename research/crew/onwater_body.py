"""Rung 2a: an on-water body that catches with straight arms. Nothing fitted to an athlete.

    python research/crew/onwater_body.py

The ergometer keyframes (Caplan & Gardner 2010) travel ~20% too far in the legs and ~10% in the
trunk against on-water rowers, which leaves the arms bent at the catch (SOURCES sec. 169). Here:

  1. **Stature** — [BR24]'s measured 1.91 m; [CR06]'s athlete is not given, so [LE26]'s
     world-class women's mean, 1.787 m (stated, not fitted).
  2. **Segment travel** — [K05]'s on-water means (five women, 1.80 m, racing rate: legs 0.515,
     trunk 0.49 m), scaled by stature. The shank and trunk excursions are scaled about their
     means (``scale_leg_amplitude``, ``scale_trunk_amplitude``) until the body travels that far.
  3. **Footboard** — moved (within a stretcher's +-0.3 m) until straight arms reach the rig's
     catch angle, the user's direction when the catch angle cannot be derived as is.
  4. **Check, not fit** — the finish: how far the hands end from the shoulders, and each
     athlete's own measured travels, printed beside the model's.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]

from coxswain.crew.anthropometry import RowerAnthropometry                 # noqa: E402
from coxswain.crew.kinematics import JointDrivenRower, RowerStation         # noqa: E402
from coxswain.crew.stroke import OnWaterTiming                              # noqa: E402
from coxswain.crew.stroke_data import (CAPLAN_GARDNER_2010, scale_leg_amplitude,  # noqa: E402
                                       scale_trunk_amplitude)

K05 = dict(stature=1.80, legs=0.515, trunk=0.49)          # on-water means, racing rate
LE26_WOMEN_STATURE = 1.787


def travels(rower, period):
    t = np.linspace(0.0, period, 600, endpoint=False)
    J = [rower.joint_positions(float(x)) for x in t]
    hip = np.array([j["hip"][0] for j in J])
    sh = np.array([j["shoulder"][0] for j in J])
    return float(np.ptp(hip)), float(np.ptp(sh - hip))


def onwater_dataset(anthro, station, timing, stature, base=CAPLAN_GARDNER_2010):
    """Scale shank then trunk excursion until legs and trunk travel [K05]'s, scaled by stature."""
    k = stature / K05["stature"]
    target_legs, target_trunk = K05["legs"] * k, K05["trunk"] * k

    def solve(make, measure, target):
        lo, hi = 0.4, 1.3
        for _ in range(30):
            f = 0.5 * (lo + hi)
            value = measure(make(f))
            lo, hi = (lo, f) if value > target else (f, hi)
        return 0.5 * (lo + hi)

    probe = lambda ds: JointDrivenRower(anthro, station, timing, dataset=ds)
    f_legs = solve(lambda f: scale_leg_amplitude(base, f),
                   lambda ds: travels(probe(ds), timing.period)[0], target_legs)
    legs = scale_leg_amplitude(base, f_legs)
    f_trunk = solve(lambda f: scale_trunk_amplitude(legs, f),
                    lambda ds: travels(probe(ds), timing.period)[1], target_trunk)
    return scale_trunk_amplitude(legs, f_trunk), f_legs, f_trunk, (target_legs, target_trunk)


def rig_geometry(boat):
    """The starboard scull's lock relative to the footboard, the grip radius and height."""
    lock = [k for k in boat.rig.seats[0].oarlocks if int(k.side) > 0][0]
    ankle = float(boat.crew[0].rower.station.x_ankle)
    p = np.asarray(lock.position, dtype=float)
    return dict(x=p[0] - ankle, y=p[1], z=p[2], r=float(lock.oar.inboard),
                catch=float(boat.oar_sweep.catch_angle), finish=float(boat.oar_sweep.finish_angle))


def straight_arm_angle(shoulder, rig, arm, shift=0.0):
    """Oar angle (rad) at which the starboard grip is exactly one arm length from the shoulder;
    NaN where no angle does. ``shoulder`` relative to the footboard, centreline x/z; the scull's
    grip at (x_L - r sin(phi), y_L - r cos(phi), z_L)."""
    phi = np.radians(np.linspace(-80.0, 89.9, 3400))
    grip = np.stack([rig["x"] - rig["r"] * np.sin(phi), rig["y"] - rig["r"] * np.cos(phi),
                     np.full_like(phi, rig["z"])], axis=1)
    sh = np.array([shoulder[0] + shift, 0.20, shoulder[2]])       # starboard shoulder
    d = np.linalg.norm(grip - sh, axis=1) - arm
    k = np.flatnonzero(np.diff(np.sign(d)) != 0)
    if k.size == 0:
        return np.nan
    k = k[-1]                                                     # the sternward (catch-side) root
    return float(phi[k] - d[k] * (phi[k + 1] - phi[k]) / (d[k + 1] - d[k]))


def free_body(anthro, timing, dataset):
    rower = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing, dataset=dataset)
    t = np.linspace(0.0, timing.period, 400, endpoint=False)
    J = [rower.joint_positions(float(x)) for x in t]
    return t, np.array([j["shoulder"] for j in J]), float(rower.upper_arm_length + rower.forearm_length)


def athlete(name):
    import like_for_like as L
    import validate_blade as V
    if name == "br24":
        stature, mass, sex, rate = L.STATURE, L.MASS, "male", L.RATE
        build = lambda ds, shift: L.build("arc", timing=OnWaterTiming(rate), dataset=ds,
                                          footboard_shift=shift)
        D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
        measured = dict(legs=float(np.ptp(D["Ls"])), trunk=float(np.ptp(D["Lt"])))
    else:
        stature, mass, sex, rate = LE26_WOMEN_STATURE, V.CR06_MASS, "female", 60.0 / V.CR06_T
        m = V.cr06_measured()
        build = lambda ds, shift: V.cr06_boat(m, OnWaterTiming(rate), dataset=ds,
                                              footboard_shift=shift or 0.0, stature=stature)
        measured = dict(legs=m["leg_travel"], trunk=float(np.ptp(V._cr06_series("back_disp_m")(
            np.linspace(0, V.CR06_T, 400)))))
    return stature, mass, sex, rate, build, measured


def main():
    import kinematic_drive as K
    for name in ("br24", "cr06"):
        stature, mass, sex, rate, build, measured = athlete(name)
        anthro = RowerAnthropometry(mass=mass, stature=stature, sex=sex)
        timing = OnWaterTiming(rate)
        ds, f_legs, f_trunk, (tl, tt) = onwater_dataset(anthro, RowerStation(x_ankle=0.0), timing, stature)
        rig = rig_geometry(build(None, None))           # the rig as the like-for-like builds it
        print("\n%s (stature %.3f m): K05 targets legs %.3f, trunk %.3f; measured legs %.3f, trunk %.3f"
              % (name, stature, tl, tt, measured["legs"], measured["trunk"]))
        print("\nscale factors: shank excursion %.3f, trunk excursion %.3f; rig catch %.1f, finish %.1f deg"
              % (f_legs, f_trunk, np.degrees(rig["catch"]), np.degrees(rig["finish"])))
        for label, dataset in (("ergometer body", None), ("on-water body ", ds)):
            t, sh, arm = free_body(anthro, timing, dataset or CAPLAN_GARDNER_2010)
            # footboard shift for a straight-arm catch at the rig's catch angle
            lo, hi = -0.6, 0.6
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                a = straight_arm_angle(sh[0], rig, arm, mid)
                lo, hi = (mid, hi) if (np.isnan(a) or a > rig["catch"]) else (lo, mid)
            shift = 0.5 * (lo + hi)
            kf = int(np.argmax(sh[:, 0]))                 # the body's finish: shoulders furthest bow
            fin_straight = straight_arm_angle(sh[kf], rig, arm, shift)
            draw = rig["r"] * (np.sin(fin_straight) - np.sin(rig["finish"])) if np.isfinite(fin_straight) else np.nan
            print("  %s: footboard %+.3f m for a straight-arm catch; at the body's finish (%.2f s) straight "
                  "arms give %.1f deg, so the arms must draw %.2f m (fore-aft) to reach %.1f"
                  % (label, shift, t[kf], np.degrees(fin_straight), draw, np.degrees(rig["finish"])))
            if dataset is not None and name == "br24":
                ang, *_ = K.his_traces()
                for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                    i = int(x / timing.period * len(t))
                    print("\nt %.1f s: straight-arm oar %5.1f deg, his %5.1f" % (
                        x, np.degrees(straight_arm_angle(sh[i], rig, arm, shift)), np.degrees(float(ang(x)))))
        print("\nK05 arm travel scaled to this stature: %.2f m (along the handle's path)" % (0.615 * stature / 1.80), flush=True)


if __name__ == "__main__":
    main()
