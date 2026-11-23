"""The optimisation step on the trunk: excitations solved, their timing an output, validated against on-water EMG (§274).

The literature's sequence (docs/OPEN_ITEMS_REVIEW.md §10) goes from computed muscle control to optimal control: the
excitations are solved rather than prescribed. Here, for the trunk's hip moment over one stroke:

    minimise   sum_k [ (M_mus(k) + P(k) - M_ref(k))^2 / s^2 ] + eps sum_k (e_E,k^2 + e_F,k^2)
    subject to M_mus(k) = a_E,k T_HE(theta_k, -w_k) - a_F,k T_HF(theta_k, w_k)
               a = L e        (first-order activation dynamics, tau 25 ms, periodic over the stroke)
               0 <= e <= 1

* M_ref: the reference run's hip moment (the moment that keeps the trunk on its path, rung2_trunk);
* P: the passive hip (Silder et al. 2007) and lumbar (Dolan et al. 1994) moments; theta on the anatomical hip (Telfer et
  al. 2023 split); T_HE, T_HF: Anderson et al. 2007 (§267-273's sourced trunk muscle model);
* the excitations e_E (trunk extensors), e_F (trunk flexors) are free at every instant: their timing is the output.

Validation (nothing fitted to EMG): the solved excitations' peak and centroid in the drive against Pitto et al. 2024's
trunk-extensor (peak 40 %) and trunk-flexor (88 %) EMG, and their correlation with it at the best lag. Then the fair
gate (rung2_trunk_modules --clock reference): the solved activations as the trunk's feedforward.

    python research/crew/trunk_excitation_optimisation.py [--strokes 8] [--delays 0 0.1]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import lsq_linear

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "biorow"), os.path.join(ROOT, "research", "blade")]
import rung2_trunk_modules as TM                                 # noqa: E402

from coxswain.crew.excitation import Modules, PITTO2024_FINISH    # noqa: E402

RT, C2, RP = TM.RT, TM.C2, TM.RP
TAU_ACT = 0.025          # s, a first-order activation time constant between Zajac's 10 ms rise and 40 ms decay
N = 240                  # nodes over one stroke


def activation_matrix(n, dt, tau):
    """a = L e for a_{k+1} = a_k + dt (e_k - a_k) / tau, periodic: the steady-state circulant response."""
    r = dt / tau
    g = (1.0 - r) ** np.arange(n)                  # impulse response, wrapped (periodic over the stroke)
    g = r * g / (1.0 - (1.0 - r) ** n)
    L = np.zeros((n, n))
    for k in range(n):
        L[k] = np.roll(g[::-1], k + 1)
    return L


def solve(name, boat, M):
    tm = TM.TrunkMuscles(name, boat, M)            # carries the angle tables, strength, passive (flags set in main)
    P = tm.P
    taus = np.linspace(0.0, P, N, endpoint=False)
    dt = P / N
    TE, TF, PAS, Y = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for k, tau in enumerate(taus):
        x = float(tau)
        theta = float(np.interp(x, tm.grid, tm.th, period=P))
        w = float(np.interp(x, tm.grid, tm.w, period=P))
        if TM.ANATOMICAL:
            theta, w = tm.anatomical_hip(theta), w * tm._anat_slope()
        theta = float(np.clip(theta, *TM.RM.TESTED["hip"]))
        TE[k] = float(np.ravel(tm.HE.max_active(theta, -w))[0])
        TF[k] = float(np.ravel(tm.HF.max_active(theta, w))[0])
        PAS[k] = tm.passive(x)
        Y[k] = tm.sigma * float(M(x))
    L = activation_matrix(N, dt, TAU_ACT)
    s = max(np.std(Y), 1.0)
    eps = 1e-3
    A = np.vstack([np.hstack([TE[:, None] * L, -TF[:, None] * L]) / s,
                   np.sqrt(eps) * np.eye(2 * N)])
    b = np.concatenate([(Y - PAS) / s, np.zeros(2 * N)])
    res = lsq_linear(A, b, bounds=(0.0, 1.0), method="trf", lsmr_tol="auto", max_iter=2000)
    eE, eF = res.x[:N], res.x[N:]
    aE, aF = L @ eE, L @ eF
    Mfit = aE * TE - aF * TF + PAS
    r2 = 1.0 - np.sum((Y - Mfit) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return dict(taus=taus, eE=eE, eF=eF, aE=aE, aF=aF, TE=TE, TF=TF, PAS=PAS, Y=Y, Mfit=Mfit, r2=float(r2), sigma=tm.sigma,
                P=P, D=float(boat.timing.drive_fraction) * P)


def validate(sol):
    """The solved excitations against Pitto et al. 2024's EMG: peak, centroid, correlation at the best lag."""
    mods = Modules.pitto2024()
    P, D = sol["P"], sol["D"]
    phase = sol["taus"] / P
    out = {}
    for key, mod in (("eE", "trunk_extensors"), ("eF", "trunk_flexors")):
        e = sol[key]
        emg = mods(mod, phase)
        on = sol["taus"] <= D
        x = sol["taus"][on] / D
        k = int(np.argmax(e[on]))
        cen = float(np.sum(x * e[on]) / max(np.sum(e[on]), 1e-9))
        ke = int(np.argmax(emg[on]))
        cene = float(np.sum(x * emg[on]) / max(np.sum(emg[on]), 1e-9))
        best = (-2.0, 0)
        for lag in range(-N // 4, N // 4 + 1):
            c = np.corrcoef(np.roll(emg, lag), e)[0, 1]
            if np.isfinite(c) and c > best[0]:
                best = (float(c), lag)
        out[mod] = dict(peak=float(x[k]), centroid=cen, emg_peak=float(x[ke]), emg_centroid=cene,
                        r_zero=float(np.corrcoef(emg, e)[0, 1]), r_best=best[0], lag_ms=1000.0 * best[1] * P / N)
    return out


class SolvedFeedforward:
    """The solved activations as the trunk's feedforward, on the stroke's phase."""

    def __init__(self, sol):
        self.sol = sol

    def __call__(self, tau, c, td):
        s = self.sol
        x = float(tau) % s["P"]
        tq = s["taus"]
        aE = float(np.interp(x, tq, s["aE"], period=s["P"]))
        aF = float(np.interp(x, tq, s["aF"], period=s["P"]))
        TE = float(np.interp(x, tq, s["TE"], period=s["P"]))
        TF = float(np.interp(x, tq, s["TF"], period=s["P"]))
        PAS = float(np.interp(x, tq, s["PAS"], period=s["P"]))
        return s["sigma"] * (aE * TE - aF * TF + PAS)


def athlete(name, strokes, delays):
    RP.BLADE = "crossflow_normal"
    C2.TAKEOVER_DEG = C2.BIOROW_CATCH_TO_BURIED_DEG
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    ref, run_r, Qr, M = RT.reference(boat, strokes)
    sol = solve(name, boat, M)
    val = validate(sol)
    print("%s: solved excitations reproduce the hip moment to r^2 %.3f; peak activations extensors %.2f, flexors %.2f"
          % (name, sol["r2"], sol["aE"].max(), sol["aF"].max()), flush=True)
    for mod, v in val.items():
        print("   %-16s solved peak %3.0f %% / centroid %3.0f %% of the drive; EMG peak %3.0f %% / centroid %3.0f %%; "
              "r %.2f at zero lag, %.2f at %+.0f ms" % (mod, 100 * v["peak"], 100 * v["centroid"], 100 * v["emg_peak"],
                                                       100 * v["emg_centroid"], v["r_zero"], v["r_best"], v["lag_ms"]),
              flush=True)
    ff = SolvedFeedforward(sol)
    gate = {}
    for dly in delays:
        boat2 = C2.R.boat_for(name, m, "on-water", drive_law=True)
        P = float(boat2.timing.period)
        holder = {}

        def tau_ff(tau, holder=holder):
            v = ff(tau, holder.get("c", 0.0), holder.get("td", 1.0))
            sim = holder.get("sim")
            if sim is not None:
                sim._ff_now = v
            return v
        free = RT.make(boat2, "free", Q=Qr, tau_ff=tau_ff, delay=dly)
        holder["sim"] = free
        orig = free.derivative

        def derivative(t, y, free=free, orig=orig, holder=holder):
            holder["c"], holder["td"] = t - free._stroke_start, float(y[-1])
            return orig(t, y)
        free.derivative = derivative
        free.start_state = np.concatenate([run_r.last_states[:, -1], [0.0, 0.0, P, 1.0]])
        run_f = free.run_strokes(strokes, surge_speed=4.3)
        s = RT.summary(free, run_f)
        gate["fb%.0fms" % (1000 * dly)] = s
        print("   fair gate, PD at %3.0f ms: trunk wander max %.2f deg; %.3f m/s, IVV %.1f%%"
              % (1000 * dly, s.get("delta_max_deg", np.nan), s["speed"], 100 * s["ivv"]), flush=True)
    return dict(r2=sol["r2"], validation=val, gate=gate,
                excitations=dict(taus=sol["taus"].tolist(), eE=sol["eE"].tolist(), eF=sol["eF"].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=8)
    ap.add_argument("--delays", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--athletes", nargs="+", default=["cr06", "br24"])
    a = ap.parse_args()
    TM.ANATOMICAL, TM.PASSIVE, TM.PASSIVE_LUMBAR = True, True, True
    res = {n: athlete(n, a.strokes, a.delays) for n in a.athletes}
    path = os.path.join(RT.OUT, "trunk_excitation_optimisation.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
