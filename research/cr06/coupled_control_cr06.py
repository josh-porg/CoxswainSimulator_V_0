"""Sprint 2, S2.1: the trunk controller designed on the coupled plant, and the internal model re-run on it.

    .venv/Scripts/python.exe research/cr06/coupled_control_cr06.py prepare
    .venv/Scripts/python.exe research/cr06/coupled_control_cr06.py run pd:0 pd:45 pd:75 pd:100 pd:117 pred:117 obs:117
    .venv/Scripts/python.exe research/cr06/coupled_control_cr06.py report

Question. Rung 5 rebuilt two-way (SOURCES §179) found that the hull's recoil lowers the trunk's effective inertia
from 9.12 to ~6.3 kg m^2 and raises its toppling rate, so the record's gains, designed on the one-way plant, hold the
trunk with half the intended stiffness (10 N.m bias: 8.96 deg). What does the same design rule give when it is
applied to the plant the trunk actually has, and do the record's internal-model results (a finite-horizon state
predictor over the delay, and a slow disturbance observer) hold on it?

Method.
  prepare  The prescribed-trunk run (12 strokes) gives the feedforward torque, as coupled_trunk_cr06.py does. A free,
           undisturbed run (3 strokes) then gives the coupled plant along its last stroke at 40 phases: the slope
           s = dM/d(alpha) with the hull's response included (minus the effective inertia), the one-way slope (hull
           acceleration held), and the toppling stiffness k = dM0/da by central difference (0.5 deg). The trunk error
           obeys e'' = lambda^2 e + u + d with lambda^2 = k/|s|, u = feedback/s and d = bias/s.
  run      Each case, 12 strokes, 10 N.m bias, last stroke reported:
             pd:<ms>    delayed PD, Kp = wn^2 + lambda_c^2, Kd = 2 zeta wn (wn 5 rad/s, zeta 0.7), lambda_c^2 the
                        stroke mean of the coupled plant: the record's design rule on the coupled plant
             pred:<ms>  the same gains on a finite-horizon state predictor: x_hat = A^L x(t - delay) + sum A^(L-1-i)
                        B u_i over the delay window (exact zero-order-hold matrices, 1 ms grid, the commands held
                        between samples, powers precomputed and summed directly), internal model lambda_c
             obs:<ms>   the predictor plus a slow disturbance observer (K_OBS 2 /s) on the mismatch between the
                        delayed measured trunk acceleration (5 ms difference of the logged rate) and the model's
                        lambda_c^2 e + u + d_hat; d_hat is added to the commands the model is driven with
  report   Collects the cases with the linear predictions: static error (d/|s|)/(Kp - lambda^2) and the delay
           equation's stability boundary (trunk_forward_cr06.dde_root) for both gain designs.

Assumptions. Everything rung 5 assumes (coupled_trunk_cr06.py). The internal model is linear with constant
coefficients (the stroke mean); the real plant's slope and toppling stiffness vary through the stroke.

Outputs: data/local/cr06_study/coupled_control_prep.json, coupled_control_<case>.json, coupled_control_cr06.json.
"""
import argparse
import bisect
import glob
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import coupled_trunk_cr06 as C                                  # noqa: E402
import trunk_forward_cr06 as TF                                 # noqa: E402

OUT = C.OUT
PREP = os.path.join(OUT, "coupled_control_prep.json")
WN, ZETA, K_OBS, GRID, BIAS = 5.0, TF.ZETA, 2.0, 0.001, 10.0
TAU_ONEWAY = 0.274                                              # rung 3's one-way e-folding time (SOURCES §178)


class CoupledControl(C.TrunkSim):
    """Rung 5's trunk with a choice of controller: delayed PD, a state predictor, or a predictor with an observer."""

    def setup_trunk(self, *args, lam2_model=None, predictor=False, observer=0.0, **kw):
        super().setup_trunk(*args, **kw)
        self.lam2, self.predictor, self.k_obs = lam2_model, predictor, observer
        self.uhist = ([], [])                                   # accepted times, commands (rad/s^2)
        self.d_hat, self._t_obs = 0.0, None
        self.obs_log = []
        if predictor and self.delay > 0:
            L = max(1, int(round(self.delay / GRID)))
            h = self.delay / L
            M = np.zeros((3, 3)); M[0, 1] = 1.0; M[1, 0] = lam2_model; M[1, 2] = 1.0
            E = expm(M * h)                                     # exact zero-order hold
            Ah, Bh = E[:2, :2], E[:2, 2]
            powers = [np.eye(2)]
            for _ in range(L):
                powers.append(powers[-1] @ Ah)
            self.L, self.h = L, h
            self.AL = powers[L]
            self.G = np.array([powers[L - 1 - i] @ Bh for i in range(L)])      # L x 2, summed directly

    def u_at(self, tq):
        """The command applied at time ``tq`` (held between samples; 0 before the first)."""
        ut, uu = self.uhist
        k = bisect.bisect_right(ut, tq) - 1
        return uu[k] if k >= 0 else 0.0

    def _observe(self, t):
        ht, _, hed = self.hist
        td = t - self.delay
        if len(ht) < 3 or td - 0.005 <= ht[0]:
            return
        he = self.hist[1]
        e_d = float(np.interp(td, ht, he))
        a_meas = (float(np.interp(td, ht, hed)) - float(np.interp(td - 0.005, ht, hed))) / 0.005
        resid = a_meas - (self.lam2 * e_d + self.u_at(td) + self.d_hat)
        if self._t_obs is not None:
            self.d_hat += self.k_obs * resid * (t - self._t_obs)
        self._t_obs = t
        self.obs_log.append((t, self.d_hat))

    def feedback(self, t, e, ed, s):
        ht, he, hed = self.hist
        if len(ht) > 8000:                                      # a sliding window: the lookups only reach back
            keep = bisect.bisect_left(ht, t - self.delay - 0.2)   # delay + 5 ms, and a full list made them O(n^2)
            del ht[:keep]; del he[:keep]; del hed[:keep]
        e_d, ed_d = self.delayed(t, e, ed)
        if self.predictor and self.delay > 0:
            ts = t - self.delay + self.h * np.arange(self.L)
            us = np.array([self.u_at(x) for x in ts]) + self.d_hat
            e_d, ed_d = self.AL @ np.array([e_d, ed_d]) + self.G.T @ us
        u = -self.Kp * e_d - self.Kd * ed_d
        ut, uu = self.uhist
        if not ut or t > ut[-1]:
            ut.append(t); uu.append(u)
            if len(ut) > 6000:                                  # keep a window well past the delay
                keep = bisect.bisect_left(ut, t - self.delay - 0.2)
                del ut[:keep]; del uu[:keep]
            if self.k_obs > 0 and self.delay > 0:
                self._observe(t)
        return s * u


def measure_plant(tau_ff, strokes=3, n=40):
    """Coupled and one-way slope, and toppling stiffness, along a free undisturbed run's last stroke."""
    Base = C.DynamicOarSimulator
    sim = C.build("free", tau_ff=tau_ff)
    try:
        run = sim.run_strokes(strokes, surge_speed=4.191, dt=C.DT_RUN)
        times, states = run.last_time, run.last_states
        P = float(sim.boat.timing.period)
        frac = np.mod(times - times[0], P) / P
        dl = np.radians(0.5)

        def ev(t, base, a, ad):
            sim._trunk = (a, ad, 0.0); d0 = Base.derivative(sim, t, base)
            sim._trunk = (a, ad, 1.0); d1 = Base.derivative(sim, t, base); sim._trunk = None
            M0 = sim.hip_moment(t, a, ad, 0.0, d0[6])
            return M0, sim.hip_moment(t, a, ad, 1.0, d1[6]) - M0, sim.hip_moment(t, a, ad, 1.0, d0[6]) - M0, d1[6] - d0[6]

        rows = []
        for q in np.linspace(0.0, 1.0, n, endpoint=False):
            j = int(np.argmin(np.abs(frac - q))); t = float(times[j]); y = states[:, j]
            base, (a, ad) = y[:-2], y[-2:]
            _, s, s_ow, recoil = ev(t, base, a, ad)
            k = (ev(t, base, a + dl, ad)[0] - ev(t, base, a - dl, ad)[0]) / (2 * dl)
            rows.append(dict(phase=float(q), s=float(s), s_oneway=float(s_ow), k=float(k), recoil=float(recoil),
                             lam2=float(-k / s), lam2_oneway=float(-k / s_ow)))
        return rows
    finally:
        C.oardynamics.InertiaProfile.of = C.REAL_OF


def prepare():
    m, sim, run = C.run_case("prescribed trunk", "prescribed", 12)
    tau_ff, peak = C.feedforward(sim, run)
    ph = np.linspace(0.0, 1.0, 2000, endpoint=False)
    P = float(sim.boat.timing.period)
    tq = np.array([tau_ff(x * P) for x in ph])
    plant = measure_plant(tau_ff)
    lam2 = np.array([r["lam2"] for r in plant]); s = np.array([r["s"] for r in plant])
    out = dict(prescribed=m, ff_phase=ph.tolist(), ff_torque=tq.tolist(), period=P, ff_peak=peak, plant=plant,
               lam2_mean=float(lam2.mean()), lam2_range=[float(lam2.min()), float(lam2.max())],
               s_mean=float(s.mean()), s_range=[float(s.min()), float(s.max())],
               lam2_oneway_mean=float(np.mean([r["lam2_oneway"] for r in plant])),
               s_oneway_mean=float(np.mean([r["s_oneway"] for r in plant])))
    json.dump(out, open(PREP, "w"), indent=1)
    print("coupled plant: slope %.2f (%.2f to %.2f) kg m^2, lambda^2 %.1f (%.1f to %.1f) /s^2; one-way slope %.2f, "
          "lambda^2 %.1f" % (out["s_mean"], *out["s_range"], out["lam2_mean"], *out["lam2_range"],
                             out["s_oneway_mean"], out["lam2_oneway_mean"]), flush=True)
    print("written", PREP)


def load_prep():
    d = json.load(open(PREP))
    ph, tq, P = np.array(d["ff_phase"]), np.array(d["ff_torque"]), d["period"]
    return d, (lambda t: float(np.interp(np.mod(t, P) / P, ph, tq, period=1.0)))


def run_cases(cases, strokes):
    d, tau_ff = load_prep()
    lam2 = d["lam2_mean"]
    tau_c = 1.0 / np.sqrt(lam2)                                 # Kp = wn^2 + 1/tau^2 = wn^2 + lambda_c^2
    for case in cases:
        kind, ms = case.split(":"); delay = float(ms) / 1000.0
        kw = dict(tau_ff=tau_ff, bias=BIAS, delay=delay, wn=WN, tau=tau_c, lam2_model=lam2,
                  predictor=kind in ("pred", "obs"), observer=K_OBS if kind == "obs" else 0.0)
        label = {"pd": "coupled-gain PD", "pred": "state predictor", "obs": "predictor + observer"}[kind]
        try:
            m, sim, _ = C.run_case("%s, %s ms" % (label, ms), "free", strokes, cls=CoupledControl, **kw)
            if kind == "obs" and sim.obs_log:
                m["d_hat_end"] = float(sim.obs_log[-1][1]); m["d_true"] = float(BIAS / d["s_mean"])
        except (OverflowError, FloatingPointError, ValueError) as e:
            m = dict(diverged=True, error=repr(e)[:200]); print("%s %s ms DIVERGED" % (label, ms), flush=True)
        m.update(case=case, kind=kind, delay_ms=float(ms), Kp=float(WN ** 2 + lam2), Kd=float(2 * ZETA * WN))
        json.dump(m, open(os.path.join(OUT, "coupled_control_%s_%s.json" % (kind, ms)), "w"), indent=1)


def report():
    d, _ = load_prep()
    lam2_c, s_c = d["lam2_mean"], abs(d["s_mean"])
    Kd = 2 * ZETA * WN
    designs = {"one-way gains (record)": WN ** 2 + 1 / TAU_ONEWAY ** 2, "coupled gains": WN ** 2 + lam2_c}
    lin = {}
    print("coupled plant: |s| %.2f kg m^2, lambda^2 %.1f /s^2 (range %.1f-%.1f)" % (s_c, lam2_c, *d["lam2_range"]))
    for name, Kp in designs.items():
        static = np.degrees((BIAS / s_c) / (Kp - lam2_c))
        lo, hi = 0.0, 0.4
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if TF.dde_root(np.sqrt(lam2_c), Kp, Kd, mid) < 0:
                lo = mid
            else:
                hi = mid
        lin[name] = dict(Kp=Kp, static_error_deg=float(static), delay_boundary_ms=float(lo * 1000))
        print("  %-24s Kp %.1f: static bias error %.2f deg, delay boundary %.0f ms" % (name, Kp, static, lo * 1000))
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "coupled_control_*_*.json"))):
        if f.endswith("_prep.json"):
            continue
        r = json.load(open(f)); rows.append(r)
    rows.sort(key=lambda r: ({"pd": 0, "pred": 1, "obs": 2}[r["kind"]], r["delay_ms"]))
    print("\ncase                     | error max / end deg | feedback N.m | velocity rms | speed")
    for r in rows:
        if r.get("diverged"):
            print("%-24s | DIVERGED" % r["case"]); continue
        print("%-24s | %7.2f / %+6.2f | %6.1f | %.4f | %.4f%s" % (r["case"], r["err_max"], r["err_end"], r["fb_max"], r["rms"],
              r["speed"], ("  d_hat %.2f vs %.2f rad/s^2" % (r["d_hat_end"], r["d_true"])) if "d_hat_end" in r else ""))
    print("\nfor comparison (SOURCES §179): one-way gains on the coupled plant, bias 8.96 / -7.78 deg, 38.9 N.m (0 ms);"
          " 10.42 deg at 117 ms. Record: 3.71 / -3.43 deg, 22.2 N.m; predictor + observer 4.44 / -3.61 deg at 117 ms.")
    json.dump(dict(prep={k: v for k, v in d.items() if not k.startswith("ff_")}, linear=lin, cases=rows),
              open(os.path.join(OUT, "coupled_control_cr06.json"), "w"), indent=1)
    print("written", os.path.join(OUT, "coupled_control_cr06.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "run", "report"))
    ap.add_argument("cases", nargs="*", default=[])
    ap.add_argument("--strokes", type=int, default=12)
    a = ap.parse_args()
    if a.stage == "prepare":
        prepare()
    elif a.stage == "run":
        run_cases(a.cases or ["pd:0", "pd:45", "pd:75", "pd:100", "pd:117", "pred:117", "obs:117"], a.strokes)
    else:
        report()


if __name__ == "__main__":
    main()
