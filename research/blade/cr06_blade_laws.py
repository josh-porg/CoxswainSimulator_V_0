"""Every project blade law against [CR06]'s blade load on her own motion (an athlete on the water, a scull).

    .venv/Scripts/python.exe research/blade/cr06_blade_laws.py

Rebuilds the lost `cr06/blade_law_vs_her.py` (2026-09-18; SOURCES "Tested against measured blade loads": tier 2 rms
82.3 N against tier 1's 25.1 N; her drive "would choose A_l = 0.25"), on the committed Fig. 3 data. Scored over fifths 1-4 of the angle's drive: her blade is
out before the angle's end (release at about -38.7 deg against her -44.35 deg finish), where her load is ~0
(data/literature/cr06_fig3_measured.csv) and her rig ([CR06] Table 1: handle force point s = 0.83 m and blade centre
l = 1.805 m from the lock; blade 0.43 m, 0.0903 m^2). Her normal blade load from her oar's moment balance about the
pin: F_n = (s F_hand / 2 - I theta_dd) / l (F_hand is both handles together; I the oar's inertia about the lock,
~1.3 kg m^2 for a 1.2 kg scull as a rod, swept 0 to 1.3). Her blade's slip at radius r: w_n = r theta_dot + v cos(theta)
(the angle decreasing through the drive), along the shaft v sin(theta). Laws in strips across the 0.43 m span.

Not a fair test of the research profile's fitted C2 (140.88): it was fitted to her. The fins' cross-flow law
(SOURCES §190) is run with each dataset's face-on values (1.3 and 2.2; crossflow_profile.py).

Outputs: data/local/blade/cr06_blade_laws.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import helmbold                                                   # noqa: E402
from coxswain.crew.blade_strips import liftdrag_strips, slip_strips   # noqa: E402
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402
from coxswain.crew.oarlock import BladeModel                      # noqa: E402

LIT = os.path.join(ROOT, "data", "literature", "cr06_fig3_measured.csv")
OUT = os.path.join(ROOT, "data", "local", "blade")
T, S, L, SPAN, AREA = 1.94, 0.83, 1.805, 0.43, 0.0903
HEIGHT = AREA / SPAN
RHO = 1000.0
BINS = 5


def load():
    rows = {}
    with open(LIT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("series"):
                continue
            s, a, b = line.strip().split(",")
            rows.setdefault(s, []).append((float(a), float(b)))
    return {k: np.array(v) for k, v in rows.items()}


def periodic_smooth(t, y, grid, harmonics=12):
    """Fourier series through the stroke (the rebuilt rungs' rule: splines break derivatives at the seam)."""
    w = 2 * np.pi / T
    A = np.column_stack([np.ones_like(t)] + [f(k * w * t) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    G = np.column_stack([np.ones_like(grid)] + [f(k * w * grid) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)])
    dG = np.column_stack([np.zeros_like(grid)] + [d for k in range(1, harmonics + 1)
                                                  for d in (-k * w * np.sin(k * w * grid), k * w * np.cos(k * w * grid))])
    d2G = np.column_stack([np.zeros_like(grid)] + [d for k in range(1, harmonics + 1)
                                                   for d in (-(k * w) ** 2 * np.cos(k * w * grid), -(k * w) ** 2 * np.sin(k * w * grid))])
    return G @ c, dG @ c, d2G @ c


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load()
    g = np.linspace(0.0, T, 1000, endpoint=False)
    th, thd, thdd = periodic_smooth(D["oar_angle_deg"][:, 0], np.radians(D["oar_angle_deg"][:, 1]), g)
    v = np.interp(g, *D["boat_velocity_mps"].T, period=T)
    fh = np.interp(g, *D["handle_force_N"].T, period=T)
    i0 = int(np.argmax(th)); i1 = int(np.argmin(th))
    drive = np.arange(i0, i1 if i1 > i0 else i1 + g.size) % g.size
    prog = (np.arange(drive.size)) / drive.size
    slip = BladeModel(c2=1.0, outboard=L)
    laws = {
        "slip, fitted C2 140.88, centre (as fitted to her)": lambda a, r, u: 140.88 * _centre_slip(slip, a, r, u),
        "slip, computed C2 58.7, centre": lambda a, r, u: 58.7 * _centre_slip(slip, a, r, u),
        "slip, fitted C2 140.88, strips (refused in the sim)": lambda a, r, u: 140.88 * _span_slip(slip, a, r, u),
        "slip, computed C2 58.7 (Hoerner C0 1.3)": lambda a, r, u: 58.7 * _span_slip(slip, a, r, u),
        "lift-drag CG07": _ld(LiftDragBlade.big_blade(outboard=L, area=AREA)),
        "lift-drag Coppel": _ld(LiftDragBlade.big_blade_full_scale(outboard=L, area=AREA, source="coppel")),
        "cross-flow (fins' law), C_Dc 1.3": _ld(helmbold.crossflow(HEIGHT, SPAN, AREA, False, 1.3, outboard=L)),
        "cross-flow (fins' law), C_Dc 2.2": _ld(helmbold.crossflow(HEIGHT, SPAN, AREA, False, 2.2, outboard=L)),
    }
    out = {}
    for inertia in (0.0, 1.3):
        her = (S * 0.5 * fh - inertia * thdd) / L
        print("\noar inertia %.1f kg m^2: her normal blade load by fifths of the drive: %s N"
              % (inertia, " ".join("%5.0f" % np.mean(her[drive][(prog >= i / BINS) & (prog < (i + 1) / BINS)]) for i in range(BINS))))
        print("  %-42s %s   rms N (1-4)  impulse (1-4)" % ("law", "  ".join("fifth %d" % (i + 1) for i in range(BINS - 1))))
        for name, law in laws.items():
            F = np.array([law(th[k], thd[k], v[k]) for k in drive])
            h = her[drive]
            fifths = [float(np.mean(F[(prog >= i / BINS) & (prog < (i + 1) / BINS)]) /
                            np.mean(h[(prog >= i / BINS) & (prog < (i + 1) / BINS)])) for i in range(BINS - 1)]
            wet = prog < 0.8                                          # fifths 1-4: her blade leaves before the angle's end
            rms = float(np.sqrt(np.mean((F[wet] - h[wet]) ** 2)))
            imp = float(np.sum(F[wet]) / np.sum(h[wet]))
            out.setdefault(name, {})["I=%.1f" % inertia] = dict(fifths=fifths, rms=rms, impulse=imp)
            print("  %-42s %s   %6.1f   %5.2f" % (name, "  ".join("%7.2f" % x for x in fifths), rms, imp))
    wn = L * thd + v * np.cos(th)
    mid = drive[(prog > 0.4) & (prog < 0.6)]
    fr = float(np.mean(np.abs(wn[mid])) / np.sqrt(9.81 * HEIGHT))
    print("\nher Froude number, the review's definition (centre normal slip at mid-drive / sqrt(g h)): %.2f" % fr)
    json.dump(dict(results=out, froude=fr), open(os.path.join(OUT, "cr06_blade_laws.json"), "w"), indent=1)


def _centre_slip(slip, a, r, u):
    """Normal load per unit C2 read at the blade centre (how [CR06]'s 2.4x fit was made)."""
    return float(slip.normal_force(a, r, u))


def _span_slip(slip, a, r, u):
    """Normal load per unit C2 (the law's own sign: driving positive), integrated over the span."""
    f, rcp = slip_strips(slip, a, r, u, SPAN)
    return float(f) * float(rcp) / L


def _ld(blade):
    return lambda a, r, u: float(np.prod(liftdrag_strips(blade, a, r, np.array([u, 0.0]), 1, SPAN)[::2])) / L


if __name__ == "__main__":
    main()
