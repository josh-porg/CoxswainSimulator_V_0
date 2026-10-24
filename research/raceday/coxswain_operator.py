"""HOCR 2026 club coxswains as human operators: McRuer's crossover model with a Pade delay, and three checks.

    .venv/Scripts/python.exe research/raceday/coxswain_operator.py

The coxswain closes a loop: sees the boat's error from the line they mean to steer (line_smoothness.py's
split: lateral offset y and heading psi of the steering part, per stroke) and moves the rudder. Four
models of that loop, each fitted per coxswain:

A. Crossover / precision model (McRuer & Jex 1967), fitted by maximum likelihood (Kalman filter on the
   irregular stroke times, exact discretisation). The coxswain aims at a point D metres ahead, so the error
   is e = psi + y/D; the command is u = -K e^{-tau s} (T_L s + 1)/(T_I s + 1) e, the delay by a first-order
   Pade approximant (1 - tau s/2)/(1 + tau s/2); the boat turns at u with a first-order lag T_b; the
   lateral disturbance (wind, wash, stream) is an Ornstein-Uhlenbeck drift; the remnant is white noise on
   the turn. From the fit: crossover frequency, phase margin and effective delay of the open loop (exact
   delay), and a profile likelihood over tau for its interval. Stability: refitted on each half of the race.
B. ARX kernel (model-free): the turn per stroke regressed on the error at lags of 0-6 strokes (ridge).
C. Inverse optimal control (the optimal-control model's idea, Kleinman, Baron & Levison 1970): the LQR
   weights on lateral error and heading error, rudder effort = 1, whose gains best match A's static gains:
   how hard each coxswain weighs being off the line against using the rudder.
D. Intermittent control (Gawthrop et al. 2011): steering as discrete corrections; the error at which a
   correction starts (a logistic threshold), and the time between corrections.
Writes operator.json (crew keys; compare_page.py anonymises).
"""
import json, os, sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from numba import njit

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C                                       # noqa: E402
import wind_perturbation as WP                                               # noqa: E402
from review_crews import CREWS                                               # noqa: E402

START_SKIP, END_SKIP = 250.0, 30.0
NAMES = ["K", "D", "tau", "T_L", "T_I", "T_b", "T_w", "q_w", "q_u", "r_y", "r_psi"]
LO = np.log([0.01, 5.0, 0.05, 0.02, 0.02, 0.3, 2.0, 1e-4, 1e-7, 1e-3, 1e-6])
HI = np.log([10.0, 400.0, 4.0, 15.0, 15.0, 6.0, 300.0, 1.0, 1e-1, 4.0, 1e-2])
X0 = np.log([0.3, 40.0, 1.0, 1.0, 1.0, 1.5, 30.0, 0.01, 1e-4, 0.05, 1e-4])


def system(p, v):
    """Continuous A (6x6) and process-noise Qc. States: y, psi, omega, w, z (Pade), q (lead-lag)."""
    K, D, tau, TL, TI, Tb, Tw, qw, qu = p[:9]
    e = np.array([1.0 / D, 1.0, 0, 0, 0, 0])
    z = np.eye(6)[4]; q = np.eye(6)[5]
    ed = 2 * z - e                                           # Pade(1,1) output: 2z - e
    u = -K * ((TL / TI) * ed + (1 - TL / TI) * q)
    A = np.zeros((6, 6))
    A[0, 1] = v; A[0, 3] = 1.0                               # y' = v psi + w
    A[1, 2] = 1.0                                            # psi' = omega
    A[2] = (u - np.eye(6)[2]) / Tb                           # omega' = (u - omega)/T_b
    A[3, 3] = -1.0 / Tw                                      # w' = -w/T_w + noise
    A[4] = (2.0 / tau) * (e - z)                             # z' = (2/tau)(e - z)
    A[5] = (ed - q) / TI                                     # q' = (e_d - q)/T_I
    Qc = np.zeros((6, 6)); Qc[3, 3] = qw; Qc[2, 2] = qu
    return A, Qc


def discretise(A, Qc, dt):
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n)); M[:n, :n] = -A; M[:n, n:] = Qc; M[n:, n:] = A.T
    E = expm(M * dt)
    Ad = E[n:, n:].T
    return Ad, Ad @ E[:n, n:]


@njit(cache=True)
def kf_loop(idx, Ads, Qds, Y, Psi, R):
    """Kalman filter negative log-likelihood; measurements y and psi (the first two states)."""
    x = np.zeros(6); x[0] = Y[0]; x[1] = Psi[0]
    P = np.diag(np.array([1.0, 1e-3, 1e-3, 0.01, 1e-3, 1e-3]))
    out = 0.0
    for k in range(1, len(Y)):
        Ad = Ads[idx[k]]; Qd = Qds[idx[k]]
        x = Ad @ x
        P = Ad @ P @ Ad.T + Qd
        r0 = Y[k] - x[0]; r1 = Psi[k] - x[1]
        s00 = P[0, 0] + R[0]; s01 = P[0, 1]; s11 = P[1, 1] + R[1]
        det = s00 * s11 - s01 * s01
        if det <= 0:
            return 1e9
        i00 = s11 / det; i01 = -s01 / det; i11 = s00 / det
        out += 0.5 * (np.log(det) + r0 * (i00 * r0 + i01 * r1) + r1 * (i01 * r0 + i11 * r1))
        Kg = np.empty((6, 2))
        for i in range(6):
            Kg[i, 0] = P[i, 0] * i00 + P[i, 1] * i01
            Kg[i, 1] = P[i, 0] * i01 + P[i, 1] * i11
        x = x + Kg[:, 0] * r0 + Kg[:, 1] * r1
        KH = np.zeros((6, 6))
        KH[:, 0] = Kg[:, 0]; KH[:, 1] = Kg[:, 1]
        P = (np.eye(6) - KH) @ P
    return out


def nll(theta, data, fixed=None):
    p = np.exp(theta)
    if fixed:
        for i, val in fixed.items():
            p[i] = val
    A, Qc = system(p, float(np.mean(data["v"])))
    if np.max(np.real(np.linalg.eigvals(A))) > 0.5:
        return 1e9
    Ads = np.empty((len(data["dts"]), 6, 6)); Qds = np.empty((len(data["dts"]), 6, 6))
    for i, dt in enumerate(data["dts"]):
        Ads[i], Qds[i] = discretise(A, Qc, dt)
    out = kf_loop(data["idx"], Ads, Qds, data["y"], data["psi"], np.array([p[9], p[10]]))
    return out if np.isfinite(out) else 1e9


def prepare(d):
    """Strokes' time steps rounded to 0.05 s, so each distinct step is discretised once."""
    dt = np.round(np.diff(d["t"]) / 0.05) * 0.05
    dts, inv = np.unique(dt, return_inverse=True)
    d = dict(d); d["dts"] = dts; d["idx"] = np.concatenate([[0], inv]).astype(np.int64)
    return d


def fit_A(data, starts=4, rng=None, fixed=None):
    rng = rng or np.random.default_rng(3)
    best = None
    for i in range(starts):
        x0 = X0 + (0 if i == 0 else rng.normal(0, 0.6, len(X0)))
        x0 = np.clip(x0, LO + 1e-3, HI - 1e-3)
        r = minimize(nll, x0, args=(data, fixed), method="L-BFGS-B", bounds=list(zip(LO, HI)), options=dict(maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    return best


def open_loop(p, v, w):
    K, D, tau, TL, TI, Tb = p[:6]
    s = 1j * w
    Yp = K * np.exp(-tau * s) * (TL * s + 1) / (TI * s + 1)
    G = (1.0 / (s * (Tb * s + 1))) * (1 + v / (D * s))
    return Yp * G


def crossover(p, v):
    w = np.logspace(-3, 1.5, 4000)
    L = open_loop(p, v, w)
    m = np.abs(L)
    i = np.flatnonzero((m[:-1] >= 1) & (m[1:] < 1))
    if not len(i):
        return dict(wc=None, pm_deg=None, tau_e=None)
    i = i[-1]
    wc = float(np.exp(np.interp(0.0, [np.log(m[i + 1]), np.log(m[i])], [np.log(w[i + 1]), np.log(w[i])])))
    ph = float(np.degrees(np.angle(open_loop(p, v, np.array([wc]))[0])))
    ph = ((ph + 360.0) % 360.0) - 360.0                      # in (-360, 0]
    return dict(wc=round(wc, 4), pm_deg=round(180.0 + ph, 1), tau_e=round(float(-np.radians(ph + 90.0) / wc), 2))


def step_response(p, v, offset=5.0, T=90.0, dt=0.25):
    """Closed loop, no noise: the boat starts `offset` m off the line, heading along it."""
    A, _ = system(p, v)
    A[0, 3] = 0.0
    Ad = expm(A * dt)
    x = np.zeros(6); x[0] = offset
    out = []
    for k in range(int(T / dt) + 1):
        out.append([round(k * dt, 2), round(float(x[0]), 3), round(float(np.degrees(x[1])), 3)])
        x = Ad @ x
    return out


def arx(data, lags=7, lam=1.0):
    t, y, psi, v = data["t"], data["y"], data["psi"], data["v"]
    D = 40.0
    e = psi + y / D
    turn = np.diff(psi) / np.diff(t)                          # rad/s between strokes
    X = np.column_stack([np.roll(e, L)[:-1] for L in range(lags)])[lags:]
    yv = turn[lags:]
    Xs = X / X.std(0)
    b = np.linalg.solve(Xs.T @ Xs + lam * np.eye(lags), Xs.T @ yv) / X.std(0)
    r2 = 1 - np.var(yv - X @ b) / np.var(yv)
    dt = float(np.median(np.diff(t)))
    return dict(kernel=[round(float(x), 5) for x in b], lag_s=[round(L * dt, 2) for L in range(lags)],
                gain=round(float(-np.sum(b)), 4), peak_lag_s=round(float(np.argmin(b)) * dt, 2), r2=round(float(r2), 3))


def inverse_lqr(p, v):
    """Optimal-control check. Keep the coxswain's gain on lateral offset (K/D, from A) and ask what gain on
    heading an optimal (LQR) controller with that lateral weight would use, for the plant y' = v psi,
    psi' = omega, omega' = (u - omega)/T_b with rudder effort weight 1. anticipation = the coxswain's heading
    gain (K) over the optimal one: below 1, they correct the offset with too little regard for the heading
    already taking them back (overshoot, weave); above 1, more heading damping than optimal (slow, sluggish)."""
    from scipy.linalg import solve_continuous_are
    from scipy.optimize import brentq
    K, D, Tb = p[0], p[1], p[5]
    A = np.array([[0, v, 0], [0, 0, 1], [0, 0, -1 / Tb]]); B = np.array([[0], [0], [1 / Tb]])

    def gains(lq):
        P = solve_continuous_are(A, B, np.diag([np.exp(lq), 1e-9, 1e-9]), np.eye(1))
        return (B.T @ P)[0]
    lq = brentq(lambda z: np.log(abs(gains(z)[0])) - np.log(K / D), -40.0, 20.0)
    g = gains(lq)
    return dict(q_lateral=float(np.exp(lq)), optimal_heading_gain=round(float(g[1]), 4), heading_gain=round(float(K), 4),
                anticipation=round(float(K / g[1]), 3), optimal_omega_gain=round(float(g[2]), 4))


def intermittent(data):
    """Corrections: reversals of the turn beyond 0.5 deg of heading; the error at their start (logistic)."""
    t, y, psi = data["t"], data["y"], data["psi"]
    e = np.degrees(psi + y / 40.0)
    turn = np.diff(psi)
    sgn = np.sign(turn)
    start = np.zeros(len(t), bool)
    last = 0
    for k in range(1, len(sgn)):
        if sgn[k] != sgn[k - 1] and abs(np.degrees(psi[k] - psi[last])) > 0.5:
            start[k] = True; last = k
    from scipy.optimize import minimize as mn
    a = np.abs(e[:-1]); s = start[:-1].astype(float)

    def nl(b):
        z = b[0] + b[1] * a
        return float(np.sum(np.logaddexp(0, z) - s * z))
    b = mn(nl, [-1.0, 0.5], method="Nelder-Mead").x
    ivl = np.diff(t[start])
    return dict(corrections=int(start.sum()), threshold_deg=round(float(-b[0] / b[1]), 2) if b[1] > 0 else None,
                sharpness=round(float(b[1]), 3), interval_s=round(float(np.median(ivl)), 1) if len(ivl) else None,
                interval_iqr_s=[round(float(np.percentile(ivl, 25)), 1), round(float(np.percentile(ivl, 75)), 1)] if len(ivl) else None)


def main():
    CM = C.CourseModel.build()
    course = CM.course
    W = WP.asos()
    shelter = WP.Shelter(CM.raster)
    keys = [k for k, c in CREWS.items() if os.path.exists(os.path.join(HERE, c["json"]))]
    out = dict(params=NAMES, crews={})
    for k in keys:
        Z = WP.crew_series(k, course, W, shelter)
        m = (Z["st"] > START_SKIP) & (Z["st"] < course.length - END_SKIP)
        o = np.argsort(Z["t"][m])
        t = Z["t"][m][o]; keep = np.concatenate([[True], np.diff(t) > 0.3])
        data = prepare(dict(t=t[keep], y=Z["d"][m][o][keep], psi=Z["fast"][m][o][keep], v=Z["v"][m][o][keep]))
        vm = float(np.mean(data["v"]))
        r = fit_A(data)
        # profile likelihood over the delay, which doubles as a global search: if a fixed delay fits better
        # than the free fit, the free fit restarts from there (twice at most)
        for _round in range(3):
            prof, sols = [], []
            for tau in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
                th = r.x.copy(); th[2] = np.log(tau)
                rr = minimize(lambda x: nll(np.insert(x, 2, np.log(tau)), data), np.delete(th, 2), method="L-BFGS-B",
                              bounds=list(zip(np.delete(LO, 2), np.delete(HI, 2))), options=dict(maxiter=300))
                prof.append([tau, round(float(rr.fun - r.fun), 2)]); sols.append(np.insert(rr.x, 2, np.log(tau)))
            j = int(np.argmin([d for _, d in prof]))
            if prof[j][1] > -0.5 or _round == 2:
                break
            r2 = minimize(nll, np.clip(sols[j], LO + 1e-6, HI - 1e-6), args=(data, None), method="L-BFGS-B", bounds=list(zip(LO, HI)), options=dict(maxiter=600))
            r = r2 if r2.fun < r.fun else type(r)(x=sols[j], fun=r.fun + prof[j][1])
        p = np.exp(r.x)
        okp = [a for a, d in prof if d <= 1.92] + [float(p[2])]
        halves = {}
        n = len(data["t"])
        for name, sl in (("first", slice(0, n // 2)), ("second", slice(n // 2, n))):
            dh = prepare({kk: data[kk][sl] for kk in ("t", "y", "psi", "v")})
            rh = fit_A(dh, starts=2)
            ph = np.exp(rh.x)
            halves[name] = dict(K=round(float(ph[0]), 3), D=round(float(ph[1]), 1), tau=round(float(ph[2]), 2), **crossover(ph, vm))
        # one-step prediction quality: innovation share of the heading variance
        A_ = dict(params={nm: float(x) for nm, x in zip(NAMES, p)}, nll=round(float(r.fun), 2), n=n,
                  tau_ci=[round(min(okp), 2), round(max(okp), 2)], profile=prof, halves=halves, **crossover(p, vm),
                  step=step_response(p, vm))
        out["crews"][k] = dict(speed=round(vm, 2), A=A_, B=arx(data), C=inverse_lqr(p, vm), D=intermittent(data))
        c = out["crews"][k]
        print("%-7s A: K %.3f  D %.0f m  tau %.2f s [%.2f, %.2f]  T_L %.2f  T_I %.2f  T_b %.2f | wc %s rad/s, PM %s deg, tau_e %s s | halves tau %.2f / %.2f, wc %s / %s"
              % (k, p[0], p[1], p[2], *A_["tau_ci"], p[3], p[4], p[5], A_["wc"], A_["pm_deg"], A_["tau_e"],
                 halves["first"]["tau"], halves["second"]["tau"], halves["first"]["wc"], halves["second"]["wc"]))
        print("        B: gain %.4f, peak lag %.1f s, R^2 %.3f | C: q_lat %.2e, heading gain %.3f vs optimal %.3f (anticipation %.2f) | D: %d corrections, threshold %s deg, interval %s s"
              % (c["B"]["gain"], c["B"]["peak_lag_s"], c["B"]["r2"], c["C"]["q_lateral"], c["C"]["heading_gain"], c["C"]["optimal_heading_gain"], c["C"]["anticipation"],
                 c["D"]["corrections"], c["D"]["threshold_deg"], c["D"]["interval_s"]))
    json.dump(out, open(os.path.join(HERE, "operator.json"), "w"), indent=1)
    print("written operator.json")


if __name__ == "__main__":
    main()
