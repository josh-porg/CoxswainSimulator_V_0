"""Every quasi-steady blade law in the project, on Grift et al. (2021)'s measured blade motion, against their forces.

    .venv/Scripts/python.exe research/blade/grift2021_quasisteady.py

The test [BR24] could not give cleanly (SOURCES §183): there, the blade's slip is the small difference of his oar's
speed and his boat's speed through the water, and an error in the latter moves the verdict. Here the motion is
known exactly: a robot drove a 1:2 sweep blade (span 0.275 m, height 0.125 m) along an on-water blade path in still
water and measured the force ([GTW21], data/literature/grift2021_*.csv, extracted by extract_grift2021.py).

The blade's motion: tip position x(t), y(t) (Fig. 7b) and orientation theta(t) (Fig. 4b), timing as on the water
(kappa = 1). Their decomposition (sec. 1.4) fixes the axes: n = (cos theta, sin theta), along-span t = (-sin theta,
cos theta), tip outward. A point s metres in from the tip moves at v_tip + s theta_dot n. Each law is integrated in
strips across the span with each strip's own velocity (area l_b ds):

- tier 1, slip: normal load 1/2 rho C0 w_n |w_n|, C0 = 1.3 ([CR06]'s computed plate) and 3.12 (their fitted
  sculling C2 140.88 over 1/2 rho A, A = 0.0903 m^2: the research profile's level);
- tier 2, lift and drag on the angle of attack (coxswain.crew.liftdrag.LiftDragBlade.coefficients): [CG07] Big
  Blade amplitudes, and with Coppel's and Sliasas-Tullis's full-size corrections.

Scored on F_n and F_t through the drive (t* = t / 0.80 s): by fifths, impulse, rms, and the time of the peak. The
sign of the measured force is as printed; it is the opposite of the water's force on the blade along +n (checked
here: the correlation's sign is printed), so the laws are compared as the force the blade exerts.

Outputs: data/local/blade/grift2021_quasisteady.json and .png.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402
import helmbold                                                   # noqa: E402
from coxswain.crew.blade_added_mass import labbe_added_mass, patton_added_mass   # noqa: E402

LIT = os.path.join(ROOT, "data", "literature")
OUT = os.path.join(ROOT, "data", "local", "blade")
SPAN, HEIGHT = 0.275, 0.125                                       # model blade, Fig. 6
RHO = 1000.0
T_DRIVE = 0.80                                                    # t* = 1 at the release (Figs 4b, 7b end at 0.80 s)
STRIPS = 24
BINS = 5


def load(name):
    rows = {}
    with open(os.path.join(LIT, name), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("series"):
                continue
            s, a, b = line.strip().split(",")
            rows.setdefault(s, []).append((float(a), float(b)))
    return {k: np.array(v) for k, v in rows.items()}


def smooth_derivative(t, x, grid, window=0.04):
    """Value and derivative on ``grid``: local quadratic fits over +-window/2 (the digitised curve is a polyline)."""
    val, der = np.empty_like(grid), np.empty_like(grid)
    for i, g in enumerate(grid):
        m = np.abs(t - g) <= 0.5 * window
        if m.sum() < 5:
            m = np.argsort(np.abs(t - g))[:7]
        c = np.polyfit(t[m] - g, x[m], 2)
        val[i], der[i] = c[2], c[1]
    return val, der


def motion(n=400):
    k = load("grift2021_kinematics.csv")
    g = np.linspace(0.0, T_DRIVE, n)
    th, thd = smooth_derivative(k["theta_deg"][:, 0], np.radians(k["theta_deg"][:, 1]), g, 0.06)
    x, xd = smooth_derivative(*k["tip_x_m_scaled"].T, g)
    y, yd = smooth_derivative(*k["tip_y_m_scaled"].T, g)
    return g, th, thd, np.stack([xd, yd], 1)


def measured(grid_star):
    f = load("grift2021_forces.csv")
    fn = np.mean([np.interp(grid_star, *f["Fn_N_run%d" % i].T) for i in (1, 2, 3)], 0)
    ft = np.mean([np.interp(grid_star, *f["Ft_N_run%d" % i].T) for i in (1, 2, 3)], 0)
    return fn, ft


def strip_loads(th, thd, vtip, law):
    """Water's force on the blade, (along n, along t), integrated over the span."""
    n = np.array([np.cos(th), np.sin(th)])
    t = np.array([-np.sin(th), np.cos(th)])
    ds = SPAN / STRIPS
    s = (np.arange(STRIPS) + 0.5) * ds
    fn = ft = 0.0
    for si in s:
        v = vtip + si * thd * n
        wn, wa = float(v @ n), float(v @ t)
        q = 0.5 * RHO * HEIGHT * ds * (wn * wn + wa * wa)
        a, b = law(wn, wa, q)
        fn += a; ft += b
    return fn, ft


def slip_law(c0):
    return lambda wn, wa, q: (-np.sign(wn) * 0.5 * RHO * HEIGHT * (SPAN / STRIPS) * c0 * wn * wn, 0.0)


def liftdrag_law(blade):
    def law(wn, wa, q):
        if q <= 0.0:
            return 0.0, 0.0
        c_n, c_t = blade.coefficients(np.arctan2(abs(wn), abs(wa)))
        return -np.sign(wn) * q * c_n, -np.sign(wa) * q * c_t
    return law


def added_mass_loads(g, th, thd, vtip, m_a):
    """Water's force on the blade from its added mass, (along n, along t), Kirchhoff's form for a plate whose only
    added mass is normal to it: F = -m_a dU_n/dt n - m_a U_n Omega t, U_n the blade centre's normal velocity in the
    blade's own axes and Omega = theta_dot. The second term is a tangential load no quasi-steady law carries."""
    n = np.stack([np.cos(th), np.sin(th)], 1)
    vc = vtip + 0.5 * SPAN * thd[:, None] * n
    un = (vc * n).sum(1)
    dun = np.gradient(un, g)
    return -m_a * dun, -m_a * un * thd


def score(star, model, meas):
    out = dict(fifths=[], impulse_ratio=None, rms=None, peak_at=None)
    for i in range(BINS):
        m = (star >= i / BINS) & (star < (i + 1) / BINS)
        mm, hh = np.mean(model[m]), np.mean(meas[m])
        out["fifths"].append(float(mm / hh) if abs(hh) > 1e-6 else float("nan"))
    out["impulse_ratio"] = float(np.trapezoid(model, star) / np.trapezoid(meas, star))
    out["rms"] = float(np.sqrt(np.mean((model - meas) ** 2)))
    out["peak_at"] = float(star[np.argmax(np.abs(model))])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    g, th, thd, vtip = motion()
    star = g / T_DRIVE
    fn_meas, ft_meas = measured(star)
    laws = {
        "slip C0 1.3 ([CR06] computed)": slip_law(1.3),
        "slip C0 3.12 (research profile's fitted level)": slip_law(140.88 / (0.5 * RHO * 0.0903)),
        "lift-drag CG07": liftdrag_law(LiftDragBlade.big_blade(outboard=1.0, area=SPAN * HEIGHT)),
        "lift-drag Coppel full size": liftdrag_law(LiftDragBlade.big_blade_full_scale(outboard=1.0, area=SPAN * HEIGHT,
                                                                                       source="coppel")),
        "lift-drag Sliasas-Tullis": liftdrag_law(LiftDragBlade.big_blade_full_scale(outboard=1.0, area=SPAN * HEIGHT,
                                                                                    source="sliasas_tullis")),
        "lift-drag, Helmbold lift, no mirror (AR %.2f)" % helmbold.blade(HEIGHT, SPAN, 1.0, SPAN * HEIGHT, False)[1]:
            liftdrag_law(helmbold.blade(HEIGHT, SPAN, 1.0, SPAN * HEIGHT, False)[0]),
        "lift-drag, Helmbold lift, surface mirror (AR %.2f)" % helmbold.blade(HEIGHT, SPAN, 1.0, SPAN * HEIGHT, True)[1]:
            liftdrag_law(helmbold.blade(HEIGHT, SPAN, 1.0, SPAN * HEIGHT, True)[0]),
    }
    res = {}
    traces = {}
    for name, law in laws.items():
        F = np.array([strip_loads(a, b, c, law) for a, b, c in zip(th, thd, vtip)])
        fn, ft = -F[:, 0], -F[:, 1]                               # the force the blade exerts, as printed
        traces[name] = (fn, ft)
        res[name] = dict(Fn=score(star, fn, fn_meas), Ft=score(star, ft, ft_meas))
    # quasi-steady plus added mass, both sourced coefficients ([LB19] measured on a blade, Patton potential flow)
    for qs in ("lift-drag CG07", "lift-drag Coppel full size"):
        for lab, m_a in (("Patton", patton_added_mass(SPAN, HEIGHT, RHO)), ("[LB19]", labbe_added_mass(SPAN, HEIGHT, RHO))):
            an, at = added_mass_loads(g, th, thd, vtip, m_a)
            fn, ft = traces[qs][0] - an, traces[qs][1] - at
            name = "%s + added mass %s (%.1f kg)" % (qs.replace("lift-drag ", ""), lab, m_a)
            traces[name] = (fn, ft)
            res[name] = dict(Fn=score(star, fn, fn_meas), Ft=score(star, ft, ft_meas))
    corr = float(np.corrcoef(traces["lift-drag CG07"][0], fn_meas)[0, 1])
    print("[GTW21] measured: F_n peak %.1f N at t* %.2f, impulse %.2f N (t* units); F_t impulse %.2f; sign check: "
          "CG07 F_n correlates %+.2f with the printed F_n" % (fn_meas.min(), star[np.argmin(fn_meas)],
                                                             np.trapezoid(fn_meas, star), np.trapezoid(ft_meas, star), corr))
    print("\n%-46s  F_n by fifths (law / measured)            impulse  peak t*   rms N   F_t impulse ratio"
          % "law")
    for name, r in res.items():
        print("%-46s  %s   %5.2f    %4.2f    %5.1f    %5.2f" % (
            name, " ".join("%5.2f" % v for v in r["Fn"]["fifths"]), r["Fn"]["impulse_ratio"], r["Fn"]["peak_at"],
            r["Fn"]["rms"], r["Ft"]["impulse_ratio"]) + "       %4.1f" % r["Ft"]["rms"])
    # what a single constant would need: the measured over the slip law's shape, per fifth
    base = traces["slip C0 1.3 ([CR06] computed)"][0]
    need = [float(np.mean(fn_meas[(star >= i / BINS) & (star < (i + 1) / BINS)]) /
                  np.mean(base[(star >= i / BINS) & (star < (i + 1) / BINS)]) * 1.3) for i in range(BINS)]
    print("\nC0 the slip law would need, per fifth: " + " ".join("%.2f" % v for v in need))
    json.dump(dict(results=res, c0_needed_by_fifth=need, measured_peak=float(fn_meas.min()),
                   sign_correlation=corr), open(os.path.join(OUT, "grift2021_quasisteady.json"), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        ax[0].plot(star, fn_meas, "k", lw=2, label="measured [GTW21]")
        ax[1].plot(star, ft_meas, "k", lw=2, label="measured [GTW21]")
        for name, (fn, ft) in traces.items():
            ax[0].plot(star, fn, lw=1, label=name); ax[1].plot(star, ft, lw=1)
        for a, lab in zip(ax, ("F_n (N)", "F_t (N)")):
            a.set_xlabel("t* (catch 0, release 1)"); a.set_ylabel(lab); a.grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.savefig(os.path.join(OUT, "grift2021_quasisteady.png"), dpi=130)
    except Exception as e:                                         # the figure is a convenience
        print("figure skipped: %s" % e)


if __name__ == "__main__":
    main()
