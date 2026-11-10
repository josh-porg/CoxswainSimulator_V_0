"""The catch's build-up, tank first: a rotating plate's rising/falling asymmetry on the symmetric cross-flow law (§230).

Kobayashi & Kinoshita (2003) ([KK03], SOURCES §229) rotated a scull-sized plate through angle of attack 0 -> pi in a
stream, top edge at the surface: the normal-force coefficient rises above steady while alpha rises and falls below it
as alpha falls, more so at higher reduced frequency f_r = 2 pi c / (T U). The project's cross-flow law (face-on set by
the free-surface regime, §191) is symmetric in alpha. No new level is introduced: the factor is [KK03]'s dynamic curve
over its own mirror average,

    m(alpha, f_r) = C_dyn(alpha) / (0.5 [C_dyn(alpha) + C_dyn(pi - alpha)]),

> 1 on the rising side, < 1 on the falling side, interpolated in f_r between the digitised curves (f_r 0 -> m = 1;
data/literature/kobayashi2003_fig10_caseA.csv). The signed angle is alpha_s = atan2(|w_n|, s0 w_a), s0 the sign of the
along-blade flow at the start of the drive, so the catch side is alpha_s < pi/2.

Scored open loop against the three measured loads on known motions, each with its own scorer and its own face-on
value: [GTW21] F_n by fifths (2.2), [RLBQV19] F_x by windows of sweep angle with the measured bend (1.3), [LJH12] C_d by
pseudo-incidence and F_x (2.05). The factor passes the tank test only if it improves all three; only then does it go
to the athletes and the whole validation ladder (docs/VALIDATION_LADDER.md).

    python research/blade/asymmetric_angle_law.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import helmbold                                                   # noqa: E402

KK03 = os.path.join(ROOT, "data", "literature", "kobayashi2003_fig10_caseA.csv")


def kk03_curves():
    d = np.loadtxt(KK03, delimiter=",", comments="#", skiprows=8)
    return d[:, 0], {0.246: d[:, 2], 0.899: d[:, 3], 1.074: d[:, 4]}


def asymmetry(fr):
    """m(alpha) on [0, pi] at reduced frequency ``fr`` (linear in f_r between curves; 1 at f_r = 0; held above 1.074)."""
    alpha, curves = kk03_curves()
    ms = {0.0: np.ones_like(alpha)}
    for f, c in curves.items():
        mirror = c[::-1]
        mean = 0.5 * (c + mirror)
        ms[f] = np.where(mean > 0.05, c / np.maximum(mean, 1e-9), 1.0)
    fs = sorted(ms)
    fr = float(np.clip(fr, 0.0, fs[-1]))
    hi = next(f for f in fs if f >= fr)
    lo = max(f for f in fs if f <= fr)
    w = 0.0 if hi == lo else (fr - lo) / (hi - lo)
    m = (1 - w) * ms[lo] + w * ms[hi]
    return lambda a: float(np.interp(a, alpha, m))


def asymmetric(law, fr, s0):
    """Wrap a project law ``law(w_n, w_a, x) -> (F_n, F_t)``: the normal load times m(alpha_s)."""
    m = asymmetry(fr)

    def wrapped(wn, wa, x):
        fn, ft = law(wn, wa, x)
        return fn * m(float(np.arctan2(abs(wn), s0 * wa))), ft
    return wrapped


def reduced_frequency(chord, drive_time, speed):
    return 2.0 * np.pi * chord / (drive_time * speed)


def gtw21():
    import grift2021_quasisteady as G
    g, th, thd, vtip = G.motion()
    star = g / G.T_DRIVE
    fn_meas, _ft = G.measured(star)
    n0 = np.array([np.cos(th[0]), np.sin(th[0])]); t0 = np.array([-np.sin(th[0]), np.cos(th[0])])
    v0 = vtip[0] + 0.5 * G.SPAN * thd[0] * n0
    s0 = float(np.sign(v0 @ t0))
    speeds = [np.linalg.norm(vtip[i] + 0.5 * G.SPAN * thd[i] * np.array([np.cos(th[i]), np.sin(th[i])]))
              for i in range(len(g))]
    fr = reduced_frequency(G.SPAN, G.T_DRIVE, float(np.mean(speeds)))
    base = G.liftdrag_law(helmbold.crossflow(G.HEIGHT, G.SPAN, G.SPAN * G.HEIGHT, False, 2.2))
    out = {}
    for name, law in (("symmetric", base), ("asymmetric", asymmetric(base, fr, s0))):
        F = np.array([G.strip_loads(a, b, c, law) for a, b, c in zip(th, thd, vtip)])
        out[name] = G.score(star, -F[:, 0], fn_meas)
    print("[GTW21] f_r %.2f (c %.3f m, T %.2f s, U %.2f m/s), catch-side sign %+d; F_n law / measured by fifths"
          % (fr, G.SPAN, G.T_DRIVE, np.mean(speeds), s0))
    for name, r in out.items():
        print("  %-11s %s   impulse %.2f  rms %.1f N" % (name, " ".join("%5.2f" % v for v in r["fifths"]),
                                                       r["impulse_ratio"], r["rms"]))
    return out


def rlbqv19():
    import robert2019_quasisteady as R
    D = R.load()
    K = R.kinematics(D, rigid=False, n=400)                       # the measured bend, open loop
    meas = np.stack([np.interp(K["psi_deg"], *D[k].T) for k in ("Fx_N", "Fy_N", "Mz_Nm")], 1)
    rc = R.L_E + 0.5 * R.SPAN
    un = rc * np.cos(R.THETA1) * K["rate"] - R.VB * np.sin(K["psi"] - K["phi"])
    speed = float(np.mean(np.hypot(un, R.VB * np.cos(K["psi"] - K["phi"]))))
    fr = reduced_frequency(R.SPAN, float(K["t"][-1]), speed)
    s0 = float(np.sign(np.cos(K["psi"][0] - K["phi"][0])))
    base = R.liftdrag_law(helmbold.crossflow(R.HEIGHT, R.SPAN, R.AREA, False, 1.3))
    out = {}
    for name, law in (("symmetric", base), ("asymmetric", asymmetric(base, fr, s0))):
        out[name] = R.score(K, R.loads(K, law), meas)
    print("[RLBQV19] f_r %.2f (c %.2f m, T %.2f s, U %.2f m/s), catch-side sign %+d; F_x mean error / peak by window %s"
          % (fr, R.SPAN, K["t"][-1], speed, s0, "/".join("%d-%d" % w for w in R.WINDOWS)))
    for name, r in out.items():
        print("  %-11s %s   F_x impulse %.2f  rms %.1f N; M_z rms %.1f" % (
            name, " ".join("%+5.2f" % v for v in r["Fx"]["windows"]), r["Fx"]["impulse_ratio"], r["Fx"]["rms"],
            r["Mz"]["rms"]))
    return out


def ljh12():
    import leroyer_jh12_quasisteady as J
    D = J.load()
    theta = np.radians(np.linspace(1.0, 179.0, 357))
    t = np.concatenate([[0.0], np.cumsum(np.diff(theta) / (J.K * np.sin(0.5 * (theta[1:] + theta[:-1]))))])
    speed = float(np.mean(np.hypot(J.LE * J.K * np.sin(theta) - J.VB * np.sin(theta), J.VB * np.cos(theta))))
    fr = reduced_frequency(J.SPAN, float(t[-1]), speed)
    s0 = 1.0                                                      # w_t = V_b cos(theta) > 0 at the start
    base = J.liftdrag_law(helmbold.crossflow(J.HEIGHT, J.SPAN, J.AREA, False, 2.05))
    td = np.degrees(theta)
    signs = None
    out = {}
    for name, law in (("symmetric", base), ("asymmetric", asymmetric(base, fr, s0))):
        cl, cd, inc, fx, fy = J.reduce(J.run(law, theta), theta)
        if signs is None:                                         # one sign convention, from the symmetric law
            signs = np.sign(np.corrcoef(fx, np.interp(td, *D["Fx_N"].T))[0, 1])
        fx = signs * fx
        o = np.argsort(inc)
        w = J.windows(inc[o], cd[o], np.interp(inc[o], *D["Cd"].T))
        mfx = np.interp(td, *D["Fx_N"].T)
        out[name] = dict(cd_windows=w, fx_rms=float(np.sqrt(np.mean((fx - mfx) ** 2))),
                         fx_impulse=float(np.trapezoid(fx, td) / np.trapezoid(mfx, td)))
    print("[LJH12] f_r %.2f (c %.3f m, T %.2f s for 1-179 deg, U %.2f m/s); C_d error / peak by incidence 0-45/45-90/"
          "90-135/135-180" % (fr, J.SPAN, t[-1], speed))
    for name, r in out.items():
        print("  %-11s %s   F_x rms %.1f N, impulse %.2f" % (name, " ".join("%+5.2f" % v for v in r["cd_windows"]),
                                                            r["fx_rms"], r["fx_impulse"]))
    return out


def main():
    import json
    res = dict(gtw21=gtw21(), rlbqv19=rlbqv19(), ljh12=ljh12())
    json.dump(res, open(os.path.join(ROOT, "data", "local", "blade", "asymmetric_angle_law.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
