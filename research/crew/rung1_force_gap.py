"""Phase 4.3: where the hands' load goes missing. [BR24]'s measured handle force against rung 1's, on his kinematics.

    .venv/Scripts/python.exe research/crew/rung1_force_gap.py [--strokes 12]

Rung 2 (research/crew/rung2_population.py) needs [K05]'s population muscles scaled 1.70x to reach [BR24]'s 432 W, and
his drive then runs 14% fast. Rung 1 says why the scale is that large: with [K05]'s drive law imposed on his own arc
(103.6 against his 105.1 deg) and drive time (1.001 against 1.000 s), the model's hands take 302 W where his took 432.
The arc and timing match, so the load does not: on the same hand motion the model's blade pushes back less.

This script puts the two side by side over the drive, in fifths from the catch: handle force per scull, hand speed
(the model's: inboard x oar angular rate, the lever its force is defined on), handle power, and the boat's speed.
A deficit spread evenly through the drive points at the blade's grip (force per slip); one at the catch points at
entry (immersion, added mass); one with the hand speeds apart points at the drive law, not the load.

``--span 0.43`` integrates the law across his blade's span instead of reading it at the centre.
``--blade-law liftdrag --coefficients cg07|coppel|sliasas_tullis`` runs the tier 2 lift-and-drag blade instead of the
normal-slip law. The default run also takes the hull out of the loop: his own motion through every sourced law
(``open_loop``). Result (2026-10-27, SOURCES §183): no quasi-steady blade holds his late-drive load.

Nothing is fitted. Outputs: data/local/crew/rung1_force_gap_<blade>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "biorow"), os.path.join(ROOT, "research", "blade")]
import handle_rung1 as R                                         # noqa: E402
import like_for_like as L                                         # noqa: E402

from coxswain.crew.blade_strips import slip_strips               # noqa: E402
from coxswain.crew.oarlock import BladeModel                      # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "research", "blade"))
import helmbold as HB                                             # noqa: E402

V = R.V
OUT = os.path.join(ROOT, "data", "local", "crew")
BINS = 5


def measured():
    """His cycle from the catch: time, force per scull (N), hand speed (m/s), power per scull (W), boat speed."""
    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    T = 60.0 / L.RATE
    n = len(D)
    a = 0.5 * (D["A1"] + D["A2"])
    i0 = int(np.argmin(a))                                       # his catch: the angle's minimum
    roll = lambda x: np.roll(np.asarray(x, float), -i0)          # noqa: E731
    t = np.arange(n) * T / n
    a = roll(a)
    f = roll(0.5 * (D["H1"] + D["H2"]))
    vh = roll(0.5 * (D["Vh1"] + D["Vh2"]))
    p = roll(D["H1"] * D["Vh1"] + D["H2"] * D["Vh2"]) / 2.0
    vb = roll(D["Vs"])
    j = int(np.argmax(a))                                        # his finish: the angle's maximum
    return dict(t=t, force=f, hand=np.abs(vh), power=p, boat=vb, drive=float(t[j]), period=T,
                speed=float(np.mean(D["Vs"])), total_power=float(np.mean(D["H1"] * D["Vh1"] + D["H2"] * D["Vh2"])))


def open_loop(boat, c2, blade_length=0.43, speed_offset=0.0, quiet=False):
    """His own motion through each sourced blade law, the hull not simulated.

    Per sample over his drive: his oar angle and rate (each scull's own channel) and his boat speed give the blade's
    motion through the water; each law gives the blade's normal load and so the torque about the pin; his handle
    power over his oar rate is the torque he applied. Per fifth of the drive, the law's torque over his. A law that
    holds through the drive gives one ratio throughout; the level is a separate question (the slip law's C2 is
    [CR06]'s fit, 140.9, against her computed 58.7).

    ``speed_offset``: added to his boat speed, m/s. The slip is the small difference of two ~4 m/s terms, so the
    verdict is tested against an error in his speed through the water (a current, the speed channel's calibration;
    SOURCES, "the slip law at his kinematics", 2026-09-27).
    """
    from coxswain.crew.liftdrag import LiftDragBlade
    from coxswain.crew.blade_strips import liftdrag_strips

    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    T = 60.0 / L.RATE
    n = len(D)
    dt = T / n
    i0 = int(np.argmin(D["A1"] + D["A2"]))
    locks = boat.rig.seats[0].oarlocks
    outboard = float(locks[0].oar.length - locks[0].oar.inboard - 0.5 * blade_length)
    area = float(locks[0].oar.blade_area)
    slip = BladeModel(c2=c2, outboard=outboard)
    laws = {"slip, centre": lambda th, rate, v, side: float(slip.normal_force(th, rate, v)),
            "slip, span": lambda th, rate, v, side: _span(slip_strips(slip, th, rate, v, blade_length), outboard)}
    for name, make in (("cg07", lambda: LiftDragBlade.big_blade(outboard=outboard, area=area)),
                       ("coppel", lambda: LiftDragBlade.big_blade_full_scale(outboard=outboard, area=area,
                                                                              source="coppel")),
                       ("sliasas_tullis", lambda: LiftDragBlade.big_blade_full_scale(
                           outboard=outboard, area=area, source="sliasas_tullis")),
                       # the blade review's sourced laws (docs/BLADE_HYDRODYNAMICS_REVIEW.md, SOURCES 188-189): a scull
                       # Big Blade 0.43 x 0.215 m (coxswain/crew/blade_added_mass.py), along-flow AR 0.50, no mirror
                       ("Polhamus-Lamar cap 1.3", lambda: HB.polhamus_lamar(0.215, blade_length, area, False, 1.3, outboard)),
                       ("Polhamus-Lamar cap 2.05", lambda: HB.polhamus_lamar(0.215, blade_length, area, False, 2.05, outboard)),
                       ("cross-flow C_Dc 1.3", lambda: HB.crossflow(0.215, blade_length, area, False, 1.3, outboard)),
                       ("cross-flow C_Dc 2.2", lambda: HB.crossflow(0.215, blade_length, area, False, 2.2, outboard))):
        ld = make()
        laws["lift-drag %s, centre" % name] = (lambda ld: lambda th, rate, v, side: float(
            ld.loads(th, rate, np.array([v, 0.0]), side)[0]))(ld)
        laws["lift-drag %s, span" % name] = (lambda ld: lambda th, rate, v, side: _span(
            liftdrag_strips(ld, th, rate, np.array([v, 0.0]), side, blade_length)[::2], outboard))(ld)
    rows = []
    for ch, lock in (("1", locks[0]), ("2", locks[-1])):
        a = np.radians(np.roll(D["A" + ch], -i0))
        w = (np.roll(a, -1) - np.roll(a, 1)) / (2.0 * dt)       # his oar rate, catch to finish positive
        torque = np.roll(D["H" + ch] * D["Vh" + ch], -i0) / np.where(np.abs(w) > 0.2, w, np.nan)
        vb = np.roll(D["Vs"], -i0) + float(speed_offset)
        j = int(np.argmax(a))
        for k in range(1, j):
            th, rate = -a[k], -w[k]                              # the model's signs: catch positive, drive negative
            model = [law(th, rate, vb[k], int(lock.side)) * outboard for law in laws.values()]
            rows.append([k / j, torque[k]] + model)
    R = np.array(rows, float)
    names = list(laws)
    if quiet:
        fifths = [(R[:, 0] >= i / BINS) & (R[:, 0] < (i + 1) / BINS) for i in range(BINS - 1)]
        return {nm: [float(np.nanmean(R[g, 2 + c]) / np.nanmean(R[g, 1])) for g in fifths]
                for c, nm in enumerate(names)}
    print("\nopen loop: each law's torque over his, on his own motion (blade centre %.3f m, span %.2f m, area %.3f)"
          % (outboard, blade_length, area))
    print("  %-30s" % "law" + "".join("  fifth %d" % (i + 1) for i in range(BINS - 1)) + "   drive")
    out = {}
    for c, nm in enumerate(names):
        vals = []
        for i in range(BINS - 1):                                # the last fifth is the release: no load to compare
            g = (R[:, 0] >= i / BINS) & (R[:, 0] < (i + 1) / BINS)
            vals.append(float(np.nanmean(R[g, 2 + c]) / np.nanmean(R[g, 1])))
        g = R[:, 0] < (BINS - 1) / BINS
        whole = float(np.nanmean(R[g, 2 + c]) / np.nanmean(R[g, 1]))
        out[nm] = dict(fifths=vals, drive=whole)
        print("  %-30s" % nm + "".join("  %7.2f" % v for v in vals) + "  %6.2f" % whole)
    return out


def _span(result, outboard):
    """A strip result ``(F_n, r_cp)`` as the equivalent load at the centre (same torque)."""
    f_n, r_cp = result
    return float(f_n) * float(r_cp) / outboard


def model(strokes, **options):
    m = V.ATHLETES["br24"][0]()
    boat = R.boat_for("br24", m, "on-water", drive_law=True)
    r = R.predict(boat, strokes, **options)
    sim, run = r["sim"], r["run"]
    t, y = run.last_time, run.last_states
    t0 = float(t[0])
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    inboard = float(lock.oar.inboard)
    sw, tm = boat.oar_sweep, boat.timing
    ang = np.array([float(sw(x - t0, tm)) for x in t])
    omega = np.gradient(ang, t)
    force = np.array([-sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k])[0, 0] / inboard for k in range(t.size)])
    hand = np.abs(omega) * inboard
    power = force * hand * np.sign(-omega)                       # the drive sweeps the angle down
    return dict(blade=sim._oars[0].blade, boat_obj=boat, t=t - t0, force=force, hand=hand, power=power, boat=np.asarray(run.last_speed, float),
                drive=float(tm.drive_fraction * tm.period), period=float(tm.period), speed=r["speed"],
                total_power=r["power"])


def by_fifths(x, drive):
    """Means over fifths of the drive (time from the catch)."""
    edges = np.linspace(0.0, drive, BINS + 1)
    out = {}
    for key in ("force", "hand", "power", "boat"):
        vals = []
        for a, b in zip(edges[:-1], edges[1:]):
            g = np.linspace(a, b, 40)
            vals.append(float(np.mean(np.interp(g, x["t"], x[key]))))
        out[key] = vals
    g = np.linspace(0.0, drive, 400)
    out["drive_mean"] = {k: float(np.mean(np.interp(g, x["t"], x[k]))) for k in ("force", "hand", "power")}
    out["peak_force"] = float(np.max(np.interp(g, x["t"], x["force"])))
    out["peak_at"] = float(g[int(np.argmax(np.interp(g, x["t"], x["force"])))] / drive)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=12)
    ap.add_argument("--blade-law", default="slip", choices=["slip", "liftdrag"])
    ap.add_argument("--coefficients", default="cg07", help="tier 2 amplitudes: cg07, coppel, sliasas_tullis")
    ap.add_argument("--span", type=float, default=None,
                    help="integrate the law across this blade span, m (his sculls: 0.43); default: read at the centre")
    a = ap.parse_args()
    h = measured()
    opts = dict(blade_law=a.blade_law)
    if a.blade_law == "liftdrag":
        opts["blade_coefficients"] = a.coefficients
    tag = a.blade_law if a.blade_law == "slip" else "liftdrag_" + a.coefficients
    if a.span is not None:
        opts["blade_span"] = a.span
        tag += "_span%.2f" % a.span
    print("blade: %s" % tag)
    mo = model(a.strokes, **opts)
    if a.blade_law == "slip":
        mo["open_loop"] = open_loop(mo["boat_obj"], float(mo["blade"].c2))
        print("\nthe same, his boat speed offset (m/s): fourth fifth over first-to-third best, per law")
        sweep = {}
        for dv in (-0.4, -0.2, 0.0, 0.2, 0.4):
            r = open_loop(mo["boat_obj"], float(mo["blade"].c2), speed_offset=dv, quiet=True)
            sweep[dv] = r
            print("  %+.1f  " % dv + "  ".join("%s %.2f" % (nm.replace("lift-drag ", "").replace(", ", "/"),
                                                             v[3] / max(v[:3])) for nm, v in r.items()))
        mo["speed_sweep"] = {str(k): v for k, v in sweep.items()}
    H, M = by_fifths(h, h["drive"]), by_fifths(mo, mo["drive"])
    print("[BR24] measured: %.3f m/s, %.0f W (both sculls); drive %.3f s; drive means: force %.0f N, hand %.2f m/s, "
          "power %.0f W per scull; peak %.0f N at %.0f%% of the drive"
          % (h["speed"], h["total_power"], h["drive"], H["drive_mean"]["force"], H["drive_mean"]["hand"],
             H["drive_mean"]["power"], H["peak_force"], 100 * H["peak_at"]))
    print("rung 1, [K05] law:  %.3f m/s, %.0f W (both sculls); drive %.3f s; drive means: force %.0f N, hand %.2f m/s, "
          "power %.0f W per scull; peak %.0f N at %.0f%% of the drive"
          % (mo["speed"], mo["total_power"], mo["drive"], M["drive_mean"]["force"], M["drive_mean"]["hand"],
             M["drive_mean"]["power"], M["peak_force"], 100 * M["peak_at"]))
    print("\nfifth of the drive    force N (his / model / ratio)    hand m/s (his / model)    power W (his / model)")
    for i in range(BINS):
        print("  %d                    %5.0f / %5.0f / %.2f              %.2f / %.2f                %4.0f / %4.0f"
              % (i + 1, H["force"][i], M["force"][i], M["force"][i] / H["force"][i] if H["force"][i] else np.nan,
                 H["hand"][i], M["hand"][i], H["power"][i], M["power"][i])
              + "      %.2f / %.2f  (%.3f / %.3f)" % (H["boat"][i], M["boat"][i], H["boat"][i] / h["speed"],
                                                    M["boat"][i] / mo["speed"]))
    json.dump(dict(measured=H, model=M, measured_speed=h["speed"], model_speed=mo["speed"],
                   measured_power=h["total_power"], model_power=mo["total_power"],
                   measured_drive=h["drive"], model_drive=mo["drive"], open_loop=mo.get("open_loop"), speed_sweep=mo.get("speed_sweep")),
              open(os.path.join(OUT, "rung1_force_gap_%s.json" % tag), "w"), indent=1)


if __name__ == "__main__":
    main()
