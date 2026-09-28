"""Blade configurations against two measured on-water single scullers. Nothing is fitted.

    python research/blade/validate_blade.py [--configs research,cg07,coppel+labbe+strips,...]
                                            [--athletes br24,cr06] [--strokes 16]

Each athlete's boat is built from their own published or measured rig and driven at their own
measured handle power by the model's own crew (the research profile's prescribed body and
pull). The blade configuration is the only thing that changes between rows, and no number in
any configuration is taken from either athlete:

  research       the research profile as it stands: slip law with the sculling C2 fitted to
                 [CR06] (so her row is not independent of it -- marked)
  cg07           tier 2, Caplan & Gardner's measured lift and drag amplitudes (A_l 1.25, A_d 2.07)
  cg07+patton    tier 2 with Patton's AR-2 potential-flow added mass (13 kg on a scull blade)
  cg07+labbe     tier 2 with Labbe et al.'s measured C_m on their cylinder volume (22 kg)
  coppel, plates tier 2 corrected to full size: [CO10]'s CFD of the Big Blade (drag x0.65) or
                 Sliasas & Tullis's plates as [CO10] reports them (lift x0.8, drag x0.7)
  ...+strips     the same, integrated across the blade's span (the rig's blade length) instead
                 of read at its centre (coxswain.crew.blade_strips; derived, nothing fitted)

The two added masses bracket the sourced range (coxswain.crew.blade_added_mass).

Athletes (validation targets, never tuning inputs):
  br24   elite man, 32.4 spm, 432 W handle power, 4.641 m/s, IVV 49.1% (commercial, local)
  cr06   woman, 30.9 spm, [CR06] Fig. 3 re-extracted (data/literature), 4.19 m/s; her handle
         power (~260 W) is computed here from her measured handle force (both hands summed),
         force point and oar rate
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "biorow"))

from coxswain import physics                                     # noqa: E402
from coxswain.validation.scorecard import settle_dynamic         # noqa: E402

#: A configuration is "+"-joined tokens: one base, then any modifiers.
BASES = {                       # blade law, tier 2 amplitudes
    "research": ("slip", "cg07"),         # slip law, sculling C2 fitted to [CR06]
    "cg07": ("liftdrag", "cg07"),         # quarter-scale flume amplitudes
    "coppel": ("liftdrag", "coppel"),     # full-size correction, [CO10] CFD of the Big Blade
    "plates": ("liftdrag", "sliasas_tullis"),   # full-size correction, Sliasas & Tullis plates
}
MODIFIERS = ("patton", "labbe", "strips")


def parse(config):
    base, *mods = config.split("+")
    if base not in BASES or any(m not in MODIFIERS for m in mods):
        raise SystemExit("unknown configuration %r: bases %s, modifiers %s"
                         % (config, ", ".join(BASES), ", ".join(MODIFIERS)))
    law, coefficients = BASES[base]
    mass = "patton" if "patton" in mods else ("labbe" if "labbe" in mods else "none")
    return law, coefficients, mass, "strips" in mods

# [CR06] Table 1 and appendix A.5, singles
CR06_T = 1.94
CR06_MASS = 75.0
CR06_HULL = 19.7                  # 15.8 rigged + 3.9 telemetry
CR06_FORCE_POINT = 0.83           # s, handle force point from the lock
CR06_BLADE_CENTRE = 1.805         # l
CR06_ARC_DEG = (60.49, -44.35)    # her measured catch and finish angles


def _cr06_series(name):
    path = os.path.join(ROOT, "data", "literature", "cr06_fig3_measured.csv")
    rows = [r for r in csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#"))
            if r["series"] == name]
    t = np.mod(np.array([float(r["t_s"]) for r in rows]), CR06_T)
    v = np.array([float(r["value"]) for r in rows])
    o = np.argsort(t)
    t, v = t[o], v[o]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, v = t[keep], v[keep]
    return CubicSpline(np.append(t, t[0] + CR06_T), np.append(v, v[0]), bc_type="periodic")


def cr06_measured():
    g = np.linspace(0.0, CR06_T, 2000, endpoint=False)
    v = _cr06_series("boat_velocity_mps")(g)
    # Fig. 3's F_hand is both hands summed, not per oar (SOURCES, "The measured single's rig":
    # the power and magnitude checks of 2026-09-14), so the handle power is F s |phi_dot|
    force = _cr06_series("handle_force_N")(g)
    rate = np.radians(_cr06_series("oar_angle_deg")(g, 1))
    power = np.clip(force, 0.0, None) * CR06_FORCE_POINT * np.abs(rate)
    leg = _cr06_series("leg_disp_m")(g)
    i = int(np.argmin(v[g < 0.4 * CR06_T]))
    return dict(speed=float(v.mean()), ivv=float(np.ptp(v) / v.mean()), v_min=float(v[i]),
                t_min=float(g[i]), power=float(power.mean()), leg_travel=float(np.ptp(leg)))


def cr06_boat(measured):
    """Her rig from [CR06]; her stature is not published, so it is set so the model's leg
    travel equals her measured leg travel (her own measurement, not a blade parameter)."""
    from coxswain.boats import catalog
    from coxswain.boats.boat import Boat
    from coxswain.boats.rig import Oar, build_sculling_rig
    oar = Oar(length=CR06_FORCE_POINT + CR06_BLADE_CENTRE + 0.215, inboard=CR06_FORCE_POINT,
              blade_area=0.0903, blade_length=0.43, mass=1.2)
    arc = catalog.SCULLING_ARC.__class__(catch_angle=np.radians(CR06_ARC_DEG[0]),
                                         finish_angle=np.radians(CR06_ARC_DEG[1]))

    def build(stature):
        base = catalog.single_scull(rate=60.0 / CR06_T, rower_mass=CR06_MASS, rower_stature=stature)
        rig = build_sculling_rig(n_seats=1, spacing=1.22, stern_station=-0.35, span=0.80,
                                 oarlock_height=0.32, oar=oar)
        return Boat(name="1x [CR06]", offsets=base.offsets, rig=rig, hull_mass=CR06_HULL,
                    hull_inertia=base.hull_inertia, timing=base.timing, appendages=base.appendages,
                    water=base.water, force_profile=base.force_profile, oar_sweep=arc,
                    default_anthropometry=catalog.RowerAnthropometry(mass=CR06_MASS, stature=stature,
                                                                     sex="female"))

    def leg_travel(b):
        r = b.crew[0].rower
        t = np.linspace(0.0, b.timing.period, 600, endpoint=False)
        ch = r._chain(t)
        return float(np.ptp(ch["hip"][0].value - ch["ankle"][0].value))

    lo, hi = 1.55, 2.00
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if leg_travel(build(mid)) < measured["leg_travel"] else (lo, mid)
    b = build(0.5 * (lo + hi))
    physics.resolve("research").apply(b)
    return b


def br24_measured():
    import like_for_like as L
    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    T = 60.0 / L.RATE
    t = np.arange(len(D)) * T / len(D)
    k = int(np.argmin(0.5 * (D["A1"] + D["A2"])))
    ph = np.mod(t - t[k], T)
    v = D["Vs"]
    early = ph < 0.4 * T
    i = int(np.argmin(np.where(early, v, np.inf)))
    return dict(speed=float(v.mean()), ivv=float(np.ptp(v) / v.mean()), v_min=float(v[i]),
                t_min=float(ph[i]), power=float(L.POWER))


def br24_boat(_measured):
    import like_for_like as L
    return L.build("arc")


ATHLETES = {"br24": (br24_measured, br24_boat), "cr06": (cr06_measured, cr06_boat)}


def row(boat, power, config, strokes):
    law, coefficients, mass, strips = parse(config)
    span = float(boat.rig.seats[0].oarlocks[0].oar.blade_length) if strips else None
    s = settle_dynamic(boat, power, start=4.4, strokes=strokes, blade_law=law,
                       blade_added_mass=mass, blade_span=span, blade_coefficients=coefficients)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="research,cg07,cg07+patton,cg07+labbe")
    ap.add_argument("--athletes", default="br24,cr06")
    ap.add_argument("--strokes", type=int, default=16)
    a = ap.parse_args()
    for name in a.athletes.split(","):
        measure, build = ATHLETES[name]
        m = measure()
        print("\n%s measured: speed %.3f m/s  IVV %.1f%%  dip %.2f at %.3f s  handle power %.0f W"
              % (name, m["speed"], 100 * m["ivv"], m["v_min"], m["t_min"], m["power"]), flush=True)
        for config in a.configs.split(","):
            t0 = time.time()
            try:
                s = row(build(m), m["power"], config, a.strokes)
            except ValueError as e:
                print("  %-12s refused: %s" % (config, str(e)[:100]), flush=True)
                continue
            note = "  (C2 fitted to this athlete)" if (config == "research" and name == "cr06") else ""
            print("  %-12s speed %.3f (%+.1f%%)  IVV %.1f%%  power %.0f W  [%.0f s]%s"
                  % (config, s.speed, 100 * (s.speed / m["speed"] - 1), 100 * s.surge_swing,
                     s.crew_power, time.time() - t0, note), flush=True)


if __name__ == "__main__":
    main()
