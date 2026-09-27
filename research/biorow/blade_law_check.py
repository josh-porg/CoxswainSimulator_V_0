"""The slip law evaluated on [BR24]'s own blade: his kinematics in, his force out?

    python research/biorow/blade_law_check.py

No simulation. Every input is his measurement: oar angle and rate, boat speed, handle force,
and the vertical oar angle for the blade's wetted fraction. Two blade normal forces per oar,
through the drive:

  his      from the oar's balance about the pin: l F_n = r_h H + I_oar phi_ddot (angle
           coordinate, catch positive), with r_h his handle radius read from the data as
           handle speed / oar rate (no chosen number)
  model    the research profile's slip law, C2 (l phi_dot + v cos phi)^2, at his angle, rate
           and boat speed, times his wetted fraction (BioRow's zero, z_0 = 0)

and the C2 his blade implies, F_n / slip^2, where the slip is driving. If the model's force
falls short at his kinematics, the kinematic-drive reference cannot reach his speed at his
oar motion whatever the body does -- the shortfall is in the blade law.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import like_for_like as L                                        # noqa: E402

from coxswain.crew.liftdrag import LiftDragBlade                 # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402

T = 60.0 / L.RATE
TIMES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def periodic(t, y):
    return CubicSpline(np.append(t, t[0] + T), np.append(y, y[0]), bc_type="periodic")


def main(lever_override=None, verbose=True):
    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    n = len(D)
    t = np.arange(n) * T / n
    k = int(np.argmin(0.5 * (D["A1"] + D["A2"])))          # catch
    ph = np.mod(t - t[k], T)
    o = np.argsort(ph)
    ph = ph[o]
    boat = L.build("arc")
    sim = DynamicOarSimulator(boat, peak_torque=1.0, fast=True)
    blade = sim._oars[0].blade
    lever = float(sim._oars[0].outboard) if lever_override is None else float(lever_override)
    blade = __import__("dataclasses").replace(blade, outboard=lever) if lever_override else blade
    rig_oar = boat.rig.seats[0].oarlocks[0].oar
    inertia = float(rig_oar.inertia_about_lock)
    width = float(rig_oar.blade_area) / float(rig_oar.blade_length)
    speed = periodic(ph, D["Vs"][o])
    g = np.linspace(0.0, T, 2400, endpoint=False)
    v = speed(g)
    out = {}
    for side in ("1", "2"):
        ang = periodic(ph, np.radians(-D["A" + side][o]))        # model sign: catch positive
        hf = periodic(ph, D["H" + side][o])
        vh = periodic(ph, D["Vh" + side][o])
        vert = periodic(ph, np.radians(D["V" + side][o]))
        a, r, acc = ang(g), ang(g, 1), ang(g, 2)
        drive = r < -0.3
        r_h = float(np.median(np.abs(vh(g)[drive] / r[drive])))
        f_his = (r_h * hf(g) + inertia * acc) / lever
        centre = lever * np.sin(vert(g))
        wet = np.where(r < 0, np.clip(-(centre - 0.5 * width) / width, 0.0, 1.0), 0.0)
        slip = blade.slip_velocity(a, r, v)
        f_model = wet * blade.normal_force(a, r, v)
        ld = LiftDragBlade.big_blade(outboard=lever, area=float(rig_oar.blade_area),
                                     density=float(boat.water.density))
        wn_wa = np.array([ld.relative_velocity(a[j], r[j], (v[j], 0.0), 1) for j in range(len(g))])
        alpha = np.arctan2(np.abs(wn_wa[:, 0]), np.abs(wn_wa[:, 1]))
        q = 0.5 * ld.density * ld.area * (wn_wa ** 2).sum(axis=1)
        cn_tier2 = np.array([ld.coefficients(x)[0] for x in alpha])
        out[side + "ld"] = dict(alpha=alpha, q=q, cn2=cn_tier2, wn=wn_wa[:, 0])
        out[side] = dict(r_h=r_h, f_his=f_his, f_model=f_model, f_dry=blade.normal_force(a, r, v),
                         slip=slip, wet=wet, drive=drive)
    f_his = 0.5 * (out["1"]["f_his"] + out["2"]["f_his"])
    f_model = 0.5 * (out["1"]["f_model"] + out["2"]["f_model"])
    f_dry = 0.5 * (out["1"]["f_dry"] + out["2"]["f_dry"])
    slip = 0.5 * (out["1"]["slip"] + out["2"]["slip"])
    wet = 0.5 * (out["1"]["wet"] + out["2"]["wet"])
    drive = out["1"]["drive"]
    dt = T / len(g)
    fin = g < g[drive].max()
    good = fin & (slip < -0.3) & (wet > 0.9)
    fit = float(np.sum(f_his[good] * slip[good] ** 2) / np.sum(slip[good] ** 4))
    if verbose == "alpha":
        return g, f_his, wet, drive, fin, out
    if not verbose:
        f_fit = f_model * fit / float(blade.c2)
        rms = float(np.sqrt(np.mean((f_fit[fin] - f_his[fin]) ** 2)))
        return dict(lever=lever, c2=fit, rms=rms, imp_his=np.sum(f_his[fin]) * dt,
                    imp_fit=np.sum(f_fit[fin]) * dt, early=float(f_fit[int(0.1 / T * len(g))]),
                    late=float(f_fit[int(0.8 / T * len(g))]))
    print("handle radius from the data: %.3f / %.3f m (inboard %.3f); blade lever %.3f; research C2 %.2f"
          % (out["1"]["r_h"], out["2"]["r_h"], float(rig_oar.inboard), lever, float(blade.c2)))
    print("\n  t s    his F_n   model F_n (wet)   model dry   slip m/s   wet    implied C2")
    for x in TIMES:
        i = int(round(x / T * len(g))) % len(g)
        c2 = f_his[i] / slip[i] ** 2 if slip[i] < -0.05 else np.nan
        print("  %.2f   %7.0f   %9.0f         %7.0f     %+6.2f    %.2f   %8.0f"
              % (x, f_his[i], f_model[i], f_dry[i], slip[i], wet[i], c2))
    dt = T / len(g)
    fin = g < g[drive].max()
    print("\ndrive impulse per oar, N s: his %.0f   model (wet) %.0f   model dry %.0f"
          % (np.sum(f_his[fin]) * dt, np.sum(f_model[fin]) * dt, np.sum(f_dry[fin]) * dt))
    print("least-squares C2 over the fully wet, driving drive: %.0f  (research %.1f, x%.2f)"
          % (fit, float(blade.c2), fit / float(blade.c2)))
    print("mean normal slip while fully wet and driving: %.2f m/s" % float(np.mean(slip[good])))


def lever_sweep():
    """The blade's centre of pressure is not measured: sweep it from the blade centre to
    the tip (a named choice), refit C2 at each, and see whether any constant C2 follows him."""
    print("\ncentre-of-pressure sweep (C2 refitted at each; his drive impulse per oar in brackets)")
    print("\nlever m   fitted C2   rms N   impulse fit / his   F at 0.1 s   F at 0.8 s (his 70, 37)")
    for lev in (1.795, 1.85, 1.90, 1.95, 2.01):
        r = main(lev, verbose=False)
        print("  %.3f     %7.0f    %5.0f     %5.0f / %.0f      %6.0f       %6.0f"
              % (r["lever"], r["c2"], r["rms"], r["imp_fit"], r["imp_his"], r["early"], r["late"]))


def attack_angle_table(lever=None):
    """His blade's normal-force coefficient against its angle of attack, measured: C_N =
    F_n / (1/2 rho A |w|^2), with w the blade centre's velocity through the water. Against
    the tier 2 Big Blade curve ([CG06a] amplitudes, Atkinson's shape), which was blocked on
    a measured blade load and velocity together; this is one."""
    g, f_his, wet, drive, fin, out = main(lever, verbose="alpha")
    alpha = 0.5 * (out["1ld"]["alpha"] + out["2ld"]["alpha"])
    q = 0.5 * (out["1ld"]["q"] + out["2ld"]["q"])
    cn2 = 0.5 * (out["1ld"]["cn2"] + out["2ld"]["cn2"])
    wn = 0.5 * (out["1ld"]["wn"] + out["2ld"]["wn"])
    use = fin & (wet > 0.9) & (q > 1.0)
    cn = f_his / np.where(q > 0, q, np.nan)
    print("\nhis blade's C_N against attack angle (fully wet; lever %s)" % ("blade centre" if lever is None else "%.3f" % lever))
    print("\nalpha deg   n    his C_N   tier 2 C_N   when (s)       normal slip")
    for lo in range(0, 90, 10):
        m = use & (np.degrees(alpha) >= lo) & (np.degrees(alpha) < lo + 10)
        if m.sum() < 5:
            continue
        print("  %2d-%2d     %4d   %6.2f     %6.2f      %.2f-%.2f    %+.2f"
              % (lo, lo + 10, m.sum(), np.median(cn[m]), np.median(cn2[m]), g[m].min(), g[m].max(),
                 np.median(wn[m])))
    for x in (0.10, 0.20, 0.50, 0.70, 0.80):
        i = int(x / T * len(g))
        print("\nat %.2f s: alpha %4.1f deg, his C_N %.2f, tier 2 %.2f" % (x, np.degrees(alpha[i]), cn[i], cn2[i]))


def added_mass_fit():
    """Sprint 1 #5, first test: F_n = C2 slip|slip| (driving sign) + m_a * d(slip)/dt, both
    fitted by least squares to his blade over the wetted drive, the lever swept. Labbe et al.
    (2019) give m_a ~ 10 kg for a scull blade (C_m 0.7); if the fit wants that order and
    follows his catch, an acceleration term is the missing load."""
    print("\nslip drag + added mass, both fitted (his blade, wetted drive)")
    print("\nlever m   C2 fit   m_a fit kg   rms N   impulse fit / his   F at 0.1 / 0.8 s (his 70 / 37)")
    for lev in (1.795, 1.90, 2.01):
        g, f_his, wet, drive, fin, out = main(lev, verbose="alpha")
        slip = 0.5 * (out["1"]["slip"] + out["2"]["slip"])
        dslip = np.gradient(slip, g)
        use = fin & (wet > 0.05)
        X = np.column_stack([-slip * np.abs(slip) * wet, -dslip * wet])[use]
        coef, *_ = np.linalg.lstsq(X, f_his[use], rcond=None)
        fit = np.zeros_like(f_his)
        fit[use] = X @ coef
        dt = T / len(g)
        rms = float(np.sqrt(np.mean((fit[fin] - f_his[fin]) ** 2)))
        i1, i8 = int(0.1 / T * len(g)), int(0.8 / T * len(g))
        print("  %.3f    %6.0f     %6.1f      %5.0f     %5.0f / %.0f          %4.0f / %4.0f"
              % (lev, coef[0], coef[1], rms, np.sum(fit[fin]) * dt, np.sum(f_his[fin]) * dt,
                 fit[i1], fit[i8]))


def tier2_fit():
    """Tier 2's form, C_N = A_l sin(2a) cos(a) + A_d sin^3(a), with both amplitudes fitted to
    his blade (linear least squares on F_n = q C_N, wetted drive), the lever swept. Against
    [CG07]'s Big Blade, A_l 1.25 and A_d 2.07."""
    print("\ntier 2 amplitudes fitted to his blade (wetted drive)")
    print("\nlever m   A_l fit   A_d fit   rms N   impulse fit / his   F at 0.1 / 0.8 s (his 70 / 37)")
    for lev in (1.795, 1.90, 2.01):
        g, f_his, wet, drive, fin, out = main(lev, verbose="alpha")
        cols, use = [], fin & (wet > 0.05)
        basis = []
        for side in ("1", "2"):
            a, q = out[side + "ld"]["alpha"], out[side + "ld"]["q"]
            basis.append(np.column_stack([q * np.sin(2 * a) * np.cos(a), q * np.sin(a) ** 3]) * wet[:, None])
        X = 0.5 * (basis[0] + basis[1])
        coef, *_ = np.linalg.lstsq(X[use], f_his[use], rcond=None)
        fit = np.where(use, X @ coef, 0.0)
        dt = T / len(g)
        rms = float(np.sqrt(np.mean((fit[fin] - f_his[fin]) ** 2)))
        i1, i8 = int(0.1 / T * len(g)), int(0.8 / T * len(g))
        print("  %.3f     %5.2f     %5.2f     %5.0f     %5.0f / %.0f          %4.0f / %4.0f"
              % (lev, coef[0], coef[1], rms, np.sum(fit[fin]) * dt, np.sum(f_his[fin]) * dt,
                 fit[i1], fit[i8]))


def strip_theory(strips=41):
    """The slip law integrated across the blade's span instead of read at its centre.

    The normal slip l*phi_dot + v cos(phi) varies along the blade (near zero at the root when
    the centre's is small, largest at the tip), and the load goes as its square, so the
    centre-point law understates the load and puts it too far in. Per unit span the research
    C2 is spread evenly over the blade length (a rectangular blade: the one chosen shape
    here); each strip carries C2/L_b * slip(r)^2, signed to oppose its slip, and its moment
    about the pin is r times that. Nothing is fitted."""
    g, f_his, wet, drive, fin, out = main(None, verbose="alpha")
    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    boat = L.build("arc")
    sim = DynamicOarSimulator(boat, peak_torque=1.0, fast=True)
    blade = sim._oars[0].blade
    centre = float(sim._oars[0].outboard)
    span = float(boat.rig.seats[0].oarlocks[0].oar.blade_length)
    r = np.linspace(centre - 0.5 * span, centre + 0.5 * span, strips)
    n = len(D)
    t = np.arange(n) * T / n
    k = int(np.argmin(0.5 * (D["A1"] + D["A2"])))
    ph = np.mod(t - t[k], T)
    o = np.argsort(ph)
    ph = ph[o]
    v = periodic(ph, D["Vs"][o])(g)
    moment = np.zeros_like(g)
    force = np.zeros_like(g)
    for side in ("1", "2"):
        ang = periodic(ph, np.radians(-D["A" + side][o]))
        a, w = ang(g), ang(g, 1)
        slip = r[None, :] * w[:, None] + (v * np.cos(a))[:, None]
        dF = -np.sign(slip) * blade.c2 / span * slip ** 2
        force += 0.5 * np.trapezoid(dF, r, axis=1)
        moment += 0.5 * np.trapezoid(dF * r[None, :], r, axis=1)
    force *= wet
    moment *= wet
    f_equiv = moment / centre                  # the load at the centre with the same moment
    dt = T / len(g)
    cop = np.where(np.abs(force) > 5, moment / np.where(force != 0, force, 1), np.nan)
    print("\nstrip theory, research C2 %.1f spread over %.2f m of blade, nothing fitted" % (float(blade.c2), span))
    print("\nt s    his F_n (at centre)   strip load   same-moment load at centre   centre of pressure m")
    for x in (0.10, 0.20, 0.30, 0.50, 0.70, 0.80):
        i = int(x / T * len(g))
        print("  %.2f       %5.0f               %5.0f          %5.0f                        %s"
              % (x, f_his[i], force[i], f_equiv[i], "%.2f" % cop[i] if np.isfinite(cop[i]) else "  -"))
    print("\ndrive impulse per oar, N s: his %.0f   strip %.0f   same-moment %.0f   (centre-point law 37)"
          % (np.sum(f_his[fin]) * dt, np.sum(force[fin]) * dt, np.sum(f_equiv[fin]) * dt))
    print("\nrms against his, N: %.0f" % float(np.sqrt(np.mean((f_equiv[fin] - f_his[fin]) ** 2))))


if __name__ == "__main__":
    strip_theory()
    main()
    lever_sweep()
    attack_angle_table()
    added_mass_fit()
    tier2_fit()
