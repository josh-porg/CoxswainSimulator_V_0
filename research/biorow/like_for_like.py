"""The research model, run as [BR24]'s athlete, and scored against his stroke.

    python research/biorow/like_for_like.py [--variants base,arc,trunk,...] [--strokes 16]

[BR24] (docs/SOURCES.md sec. 156) is one elite single sculler's ensemble-averaged
stroke: 97 kg, 1.91 m, an 18 kg boat, 2.885 m sculls at 0.875 m inboard, 32.4 spm,
432 W handle power, 4.64 m/s. It is the first target where the athlete, rig, rate,
power and boat response all belong to one person, so the model is set up as him
and driven at his measured power, and everything it predicts is compared with
what he did.

The measured stroke is commercial data held in data/local/biorow/ (gitignored);
only derived quantities are printed. Without the file the script still runs the
model and prints its side of the table.

Variants isolate one suspect at a time:

  base    research profile, his body, rig, rate and power; model's own crew
          kinematics (Caplan & Gardner ergometer keyframes) and drive timing
  arc     + his measured oar arc (catch 67.2, finish 36.4 deg)
  trunk   + his trunk sweep: the trunk channel, calibrated to shoulder height
          (sec. 156), implies sin(finish) + sin(catch) = 0.718, split in the
          proportion of Kleshnev's on-water catch and finish angles
  legs    + seat travel scaled to his measured 0.597 m
  drive   + his drive fraction (0.50 of the cycle, force above 10% of peak)
  all     every substitution above together
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from coxswain import physics                                   # noqa: E402
from coxswain.boats import catalog                             # noqa: E402
from coxswain.boats.rig import Oar, Oarlock, Rig, Seat          # noqa: E402
from coxswain.crew.anthropometry import PORT, STARBOARD         # noqa: E402
from coxswain.crew.anthropometry import RowerAnthropometry     # noqa: E402
from coxswain.crew.oarlock import OarAngleSweep                  # noqa: E402
from coxswain.crew.stroke import StrokeTiming                   # noqa: E402
from coxswain.crew.stroke_data import (CAPLAN_GARDNER_2010,     # noqa: E402
                                       scale_leg_amplitude)
from coxswain.validation.scorecard import settle_dynamic      # noqa: E402

DATA = os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv")

# ---- [BR24]: the athlete and rig as stated in the file's header
RATE, MASS, STATURE, HULL = 32.4093, 97.0, 1.91, 18.0
OAR = Oar(length=2.885, inboard=0.875, blade_area=0.083, blade_length=0.43)
POWER = 432.0                       # W, mean of H1*Vh1 + H2*Vh2 over his cycle
DRIVE_FRACTION = 0.50               # force above 10% of peak (0.56 at 5%, 0.54 by angle)
CATCH_DEG, FINISH_DEG = 67.2, 36.4
SEAT_TRAVEL = 0.597
TRUNK_SIN_SWEEP = 0.435 / 0.606     # trunk channel / shoulder height above hip (sec. 156)
WORK_THROUGH = 0.30                 # catalogue: oarlock 0.30 m bow of the footboard station


class MeasuredTiming(StrokeTiming):
    """StrokeTiming with a stated drive fraction in place of the Telfer fit."""

    def __init__(self, rate, drive_fraction):
        object.__setattr__(self, "_fraction", float(drive_fraction))
        super().__init__(rate)

    @property
    def drive_fraction(self) -> float:
        return self._fraction


def trunk_dataset(base):
    """Base keyframes with the catch and finish trunk angles set to his sweep.

    The channel gives only the sweep, sin(finish) + sin(catch); it is split in
    the ratio of Kleshnev's on-water angles (24.5 forward, 26.3 back). The
    mid-stroke keyframes are scaled by the same factor about the base's catch
    and finish, so the shape of the traverse is the base's own.
    """
    k_c, k_f = np.sin(np.radians(24.5)), np.sin(np.radians(26.3))
    catch = -np.degrees(np.arcsin(TRUNK_SIN_SWEEP * k_c / (k_c + k_f)))
    finish = np.degrees(np.arcsin(TRUNK_SIN_SWEEP * k_f / (k_c + k_f)))
    old = np.asarray(base.trunk, float)
    lo, hi = old[0], old[2]
    new = catch + (old - lo) * (finish - catch) / (hi - lo)
    return dataclasses.replace(base, name=base.name + "_br24trunk",
                               trunk=tuple(new.tolist()))


def build(variant, timing=None, dataset=None, footboard_shift=None):
    """His boat. ``timing`` (a StrokeTiming) overrides the variant's, e.g. [K05]'s on-water rhythm;
    ``dataset`` the body keyframes; ``footboard_shift`` (m, + toward the bow) fixes the rig instead of
    searching the nearest reachable oarlock."""
    subs = {"base": set(), "arc": {"arc"}, "trunk": {"trunk"}, "legs": {"legs"},
            "drive": {"drive"}, "all": {"arc", "trunk", "legs", "drive"}}[variant]
    dataset = dataset or CAPLAN_GARDNER_2010
    if "trunk" in subs:
        dataset = trunk_dataset(dataset)
    if timing is None:
        timing = MeasuredTiming(RATE, DRIVE_FRACTION) if "drive" in subs else StrokeTiming(RATE)
    sweep = (OarAngleSweep(catch_angle=np.radians(CATCH_DEG), finish_angle=np.radians(-FINISH_DEG))
             if "arc" in subs else catalog.SCULLING_ARC)
    anthro = RowerAnthropometry(mass=MASS, stature=STATURE)

    def make(ds, work=WORK_THROUGH):
        ref = catalog.single_scull(rate=RATE, rower_mass=MASS, rower_stature=STATURE)
        station = -0.35
        locks = tuple(Oarlock(position=np.array([station + work, side * 0.80, 0.32]),
                              side=side, oar=OAR) for side in (PORT, STARBOARD))
        rig = Rig(seats=(Seat(station_x=station, oarlocks=locks, label="stroke"),))
        boat = type(ref)(name="1x [BR24]", offsets=ref.offsets, rig=rig, hull_mass=HULL,
                         hull_inertia=ref.hull_inertia, timing=timing,
                         appendages=ref.appendages, water=ref.water,
                         force_profile=ref.force_profile, oar_sweep=sweep,
                         default_anthropometry=anthro, stroke_dataset=ds)
        physics.resolve("research").apply(boat)
        return boat

    def feasible(ds):
        # The catalogue's oarlock sits 0.30 m bow of the footboard station. [BR24] does not
        # state his work-through, so where his arc or posture cannot be reached from the
        # model's, the nearest feasible oarlock position is used and reported.
        for dw in sorted(np.arange(-0.20, 0.2001, 0.02), key=abs):
            try:
                return make(ds, WORK_THROUGH + dw), WORK_THROUGH + dw
            except ValueError:
                continue
        raise ValueError("no oarlock position within +-0.20 m lets the rower reach the handle")

    if "legs" in subs:
        # scale the shank excursion until the model's slide matches his; slide travel is
        # a property of the legs alone, so probe it on a rower that need not reach an oar
        from coxswain.crew.kinematics import JointDrivenRower
        probe_station = catalog.single_scull(rate=RATE, rower_mass=MASS,
                                             rower_stature=STATURE).crew[0].rower.station
        lo_f, hi_f = 0.5, 1.2
        for _ in range(25):
            f = 0.5 * (lo_f + hi_f)
            probe = JointDrivenRower(anthro, probe_station, timing,
                                     dataset=scale_leg_amplitude(dataset, f))
            if probe.slide_travel() > SEAT_TRAVEL:
                hi_f = f
            else:
                lo_f = f
        dataset = scale_leg_amplitude(dataset, 0.5 * (lo_f + hi_f))
    if footboard_shift is not None:
        boat, work = make(dataset, WORK_THROUGH - footboard_shift), WORK_THROUGH - footboard_shift
        boat.work_through = work
        return boat
    boat, work = feasible(dataset)
    boat.work_through = work
    return boat


def kinematics(boat, n=400):
    rower = boat.crew[0].rower
    T = boat.timing.period
    t = np.linspace(0.0, T, n, endpoint=False)
    com = np.array([rower.centre_of_mass(tt)[0] for tt in t])
    chain = rower._chain(t)
    hip = chain["hip"][0].value
    sh_rel = chain["shoulder"][0].value - hip
    trunk_len = rower.trunk_length
    return dict(com_travel=float(com.max() - com.min()),
                slide=float(hip.max() - hip.min()),
                shoulder_rel_hip=float(sh_rel.max() - sh_rel.min()),
                trunk_sin_sweep=float((sh_rel.max() - sh_rel.min()) / trunk_len))


def run(variant, strokes):
    t0 = time.time()
    try:
        boat = build(variant)
    except ValueError as e:
        return dict(variant=variant, error=str(e)[:160])
    kin = kinematics(boat)
    s = settle_dynamic(boat, POWER, start=4.6, strokes=strokes)
    out = dict(variant=variant, speed=s.speed, ivv=s.surge_swing, power=s.crew_power,
               blade_eff=s.blade_efficiency, drive_fraction=boat.timing.drive_fraction,
               drive_s=boat.timing.drive_duration, work_through=boat.work_through, seconds=round(time.time() - t0, 1), **kin)
    return out


def measured():
    if not os.path.exists(DATA):
        return None
    d = np.genfromtxt(DATA, delimiter=",", names=True)[:-1]   # last row repeats the first
    v = d["Vs"]
    return dict(speed=float(v.mean()), ivv=float((v.max() - v.min()) / v.mean()),
                power=float((d["H1"] * d["Vh1"] + d["H2"] * d["Vh2"]).mean()),
                com_travel="0.71-0.74", slide=float(d["Ls"].max() - d["Ls"].min()),
                trunk_sin_sweep=TRUNK_SIN_SWEEP, drive_fraction=DRIVE_FRACTION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,arc,trunk,legs,drive,all")
    ap.add_argument("--strokes", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for v in a.variants.split(","):
        r = run(v, a.strokes)
        rows.append(r)
        if "error" in r:
            print("%-6s infeasible: %s" % (v, r["error"]), flush=True)
            continue
        print("%-6s speed %.3f  IVV %5.1f%%  power %5.1f W  CoM travel %.3f  slide %.3f  "
              "trunk sin-sweep %.3f  drive %.2f  work-through %.2f  blade eff %s  (%ss)"
              % (v, r["speed"], 100 * r["ivv"], r["power"], r["com_travel"], r["slide"],
                 r["trunk_sin_sweep"], r["drive_fraction"], r["work_through"],
                 "%.3f" % r["blade_eff"] if r["blade_eff"] else "-", r["seconds"]), flush=True)
    m = measured()
    if m:
        print("BR24   speed %.3f  IVV %5.1f%%  power %5.1f W  CoM travel %s  slide %.3f  "
              "trunk sin-sweep %.3f  drive %.2f"
              % (m["speed"], 100 * m["ivv"], m["power"], m["com_travel"], m["slide"],
                 m["trunk_sin_sweep"], m["drive_fraction"]))
    if a.out:
        json.dump(dict(model=rows, measured=m), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
