"""Frozen-time linear stability of rung 2 with the bending shaft: where does the swing's energy come from? (SOURCES §224)

§223 found [BR24]'s bending rung-2 drive swinging at ~0.25 s with a growing amplitude that does not converge with the
time step. This takes the simulator's own continuous-time equations for a small disturbance, at each instant of the
last stroke's drive, on three states: the stroke clock tau, its speed tau', and the shaft's hinge bend phi (the hull's
velocity held at its value there).

    tau''  = (Q(tau, tau') - R0) / (R1 - R0)          rung2_clock: the virtual-work solve, R(a) the residual at tau''=a
    phi'   : K phi = l F_n(S(tau) + phi, S'(tau) tau' + k phi')    the massless hinge (dt -> 0 of _update_bending)

1. **Numerical Jacobian** of (tau', tau'', phi') in (tau, tau', phi) by central differences through the simulator's
   own functions (ClockSim.residual, _muscle, _blade_loads_arm_rigid): its eigenvalues are the model's local modes.
2. **The analytic reduced form.** Linearising the hinge gives K y = G_a (S' dtau + y) + G_r (S' dtau' + k y'), with
   G_a = d(l F_n)/d(blade angle), G_r = d(l F_n)/d(blade rate); substituted into the clock's equation the blade terms
   collapse to the spring, M dtau'' = Q_v dtau' + N S' K y (N oarlocks), so

       A = [[0, 1, 0], [0, Q_v/M, N S' K/M], [-G_a S'/(G_r k), -S'/k, (K - G_a)/(G_r k)]]

   (Q_v = dQ/dtau' less the body's velocity terms, M = R1 - R0). Its eigenvalues are printed with terms switched off:
   G_a = 0 (no hydrodynamic stiffness: the bend does not turn the blade's load), Q_v = 0 (no force-velocity damping).
   Classical reading: K - G_a < 0 is static divergence (the load grows with the bend faster than the spring); a
   complex pair with a positive real part is a coupled-mode (flutter-type) instability between the crew and the bend.

    python research/blade/bending_stability.py [--athlete br24] [--fv 4.53] [--hold-own] [--strokes 10] [--every 4]
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow")]
import rung2_population as RP                                    # noqa: E402

from coxswain.core.state import STATE_SIZE, State               # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402


class Point:
    """The simulator frozen at one sample of the last stroke, with the reduced right-hand side in (tau, tau', phi)."""

    def __init__(self, sim, t, y, air, slot=0):
        self.sim, self.t, self.slot = sim, float(t), slot
        self.base = np.asarray(y[:-2], float).copy()
        self.tau0, self.td0 = float(y[-2]), float(y[-1])
        self.air = np.asarray(air, bool)
        self.n = sim.n_oar_states
        self.lock = sim.boat.rig.seats[sim._seats[slot]].oarlocks[0]
        self.locks = len(sim.boat.rig.seats[sim._seats[slot]].oarlocks)
        self.K = float(sim.oar_bending.k_ang)
        self.lev = float(sim._bend_lever[slot])
        self.state = State.from_vector(self.base[:STATE_SIZE])

    def arm_fn(self, angle, rate, state=None):
        f_n, _f_t, arm = self.sim._blade_loads_arm_rigid(self.slot, angle, rate, state or self.state, self.lock)
        return arm * f_n

    def sweep(self, tau):
        sw, tm = self.sim.boat.oar_sweep, self.sim.boat.timing
        return float(sw(tau, tm)), float(sw.rate(tau, tm))

    def phi_dot(self, tau, td, phi, state=None):
        S, Sp = self.sweep(tau)
        g = lambda pd: self.K * phi - self.arm_fn(S + phi, Sp * td + self.lev * pd, state)   # noqa: E731
        lo, hi = -400.0, 400.0
        if g(lo) * g(hi) > 0.0:
            return float("nan")
        return float(brentq(g, lo, hi, xtol=1e-10))

    def rhs(self, X):
        """(tau', tau'', phi'[, u']) at X = (tau, tau', phi[, u]); with a 4th entry the hull's surge velocity u
        (state index 6) is a state too, so the blade's slip and the hull's recoil enter the Jacobian."""
        tau, td, phi = (float(v) for v in X[:3])
        surge = len(X) > 3
        sim, n = self.sim, self.n
        base = self.base.copy()
        if surge:
            base[6] = float(X[3])
        state = State.from_vector(base[:STATE_SIZE])
        pd = self.phi_dot(tau, td, phi, state)
        S, Sp = self.sweep(tau)
        base[STATE_SIZE + self.slot], base[STATE_SIZE + n + self.slot] = S, Sp * td
        saved = (sim._bend.copy(), sim._bend_rate.copy(), sim._in_air, sim._clock)
        try:
            sim._bend[:], sim._bend_rate[:] = phi, pd
            sim._in_air = self.air.copy()
            Rs, D = [], []
            for a in (0.0, 1.0):
                sim._clock = (tau, td, a)
                d = DynamicOarSimulator.derivative(sim, self.t, base)
                Rs.append(sim.residual(self.t, base, d, tau, td, a))
                D.append(d)
            sim._clock = None
            q = sim._muscle(tau, td)
        finally:
            sim._bend[:], sim._bend_rate[:] = saved[0], saved[1]
            sim._in_air, sim._clock = saved[2], saved[3]
        alpha = (q - Rs[0]) / (Rs[1] - Rs[0])
        out = [td, alpha, pd]
        if surge:
            out.append(float(D[0][6] + alpha * (D[1][6] - D[0][6])))
        return np.array(out), Rs[1] - Rs[0], q

    def jacobian(self, X, h=(1e-4, 1e-4, 1e-5, 1e-4)):
        m = len(X)
        J = np.zeros((m, m))
        for j in range(m):
            e = np.zeros(m); e[j] = h[j]
            J[:, j] = (self.rhs(X + e)[0] - self.rhs(X - e)[0]) / (2 * h[j])
        return J

    def step_map(self, phi, p, dt):
        """The simulator's own partitioned step: one RK4 step of the clock and hull with the hinge frozen at
        (phi, p), then ``_update_bending`` -- returns the hinge rate it hands to the next step."""
        import coxswain.core.integrators as I
        sim = self.sim
        y0 = np.concatenate([self.base, [self.tau0, self.td0]])
        saved = (sim._bend.copy(), sim._bend_rate.copy(), sim._in_air, sim._clock, sim._last, sim._placing, sim._rec)
        try:
            sim._bend[:], sim._bend_rate[:] = phi, p
            sim._in_air, sim._last, sim._placing, sim._rec = self.air.copy(), np.inf, False, None
            try:
                y1 = I.rk4_step(sim.derivative, self.t, y0, dt)
            except (ZeroDivisionError, FloatingPointError):         # the clock's generalised inertia hit zero
                return float("nan")
            sim._update_bending(y1[:-2], dt)
            return float(sim._bend_rate[self.slot])
        finally:
            sim._bend[:], sim._bend_rate[:] = saved[0], saved[1]
            sim._in_air, sim._clock, sim._last, sim._placing, sim._rec = saved[2:]

    def coupling(self, phi, p0, dt, iters=8):
        """ATLAS's coupling layer (Poznanski 2026, ch. ATLAS / App. added-mass benchmark): the interface map
        p -> H(S(p)) of the staggered scheme. Returns its gain dp'/dp (|gain| >= 1: the one-pass stagger amplifies the
        interface error every step), the residual amplification rho of the fixed-point iteration (geometric mean of
        successive residual ratios, ATLAS ``residual_amplification``), and the strongly coupled fixed point p*."""
        h = max(1e-3, 1e-3 * abs(p0))
        gain = (self.step_map(phi, p0 + h, dt) - self.step_map(phi, p0 - h, dt)) / (2 * h)
        res, pj = [], p0
        for _ in range(iters):
            pn = self.step_map(phi, pj, dt)
            res.append(abs(pn - pj))
            if not np.isfinite(pn):
                break
            pj = pn
        r = np.asarray(res, float)
        r = r[np.isfinite(r) & (r > 0)]
        ratios = r[1:] / r[:-1] if r.size >= 3 else np.array([])
        rho = float(np.exp(np.mean(np.log(ratios[-5:])))) if ratios.size else float("nan")
        pstar = float("nan")
        for half in (1.0, 5.0, 20.0):                              # the nearest bracket that holds a sign change
            lo, hi = p0 - half, p0 + half
            flo, fhi = self.step_map(phi, lo, dt) - lo, self.step_map(phi, hi, dt) - hi
            if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0.0:
                pstar = float(brentq(lambda q: self.step_map(phi, q, dt) - q, lo, hi, xtol=1e-8))
                break
        return gain, rho, pstar

    def body_velocity_slope(self, X, h=1e-4):
        """d R0 / d tau' with every blade out: the body's (and oar inertia's) velocity terms along tau."""
        tau, td, _phi = (float(v) for v in X)
        sim, n = self.sim, self.n
        saved = (sim._in_air, sim._clock)
        out = []
        try:
            sim._in_air = np.ones_like(self.air)
            for v in (td + h, td - h):
                S, Sp = self.sweep(tau)
                base = self.base.copy()
                base[STATE_SIZE + self.slot], base[STATE_SIZE + n + self.slot] = S, Sp * v
                sim._clock = (tau, v, 0.0)
                d = DynamicOarSimulator.derivative(sim, self.t, base)
                out.append(sim.residual(self.t, base, d, tau, v, 0.0))
        finally:
            sim._in_air, sim._clock = saved
        return (out[0] - out[1]) / (2 * h)

    def partials(self, X):
        """M, Q_v (muscle and body velocity terms, from the Jacobian's tau'' row with phi held), G_a, G_r, S'."""
        tau, td, phi = (float(v) for v in X)
        S, Sp = self.sweep(tau)
        pd = self.phi_dot(tau, td, phi)
        a0, r0 = S + phi, Sp * td + self.lev * pd
        ha, hr = 1e-4, 1e-3
        G_a = (self.arm_fn(a0 + ha, r0) - self.arm_fn(a0 - ha, r0)) / (2 * ha)
        G_r = (self.arm_fn(a0, r0 + hr) - self.arm_fn(a0, r0 - hr)) / (2 * hr)
        return Sp, G_a, G_r, pd


def analytic(M, Qv, Sp, K, G_a, G_r, k, N):
    return np.array([[0.0, 1.0, 0.0],
                     [0.0, Qv / M, N * Sp * K / M],
                     [-G_a * Sp / (G_r * k), -Sp / k, (K - G_a) / (G_r * k)]])


def worst(eig):
    i = int(np.argmax(eig.real))
    return eig[i].real, abs(eig[i].imag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athlete", default="br24", choices=["br24", "cr06"])
    ap.add_argument("--fv", type=float, default=4.53)
    ap.add_argument("--hold", type=float, default=0.0)
    ap.add_argument("--hold-own", action="store_true")
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--every", type=int, default=4, help="analyse every n-th drive sample")
    ap.add_argument("--coupling", action="store_true", help="also ATLAS's coupling-layer test of the staggered hinge")
    ap.add_argument("--surge", action="store_true", help="also the Jacobian with the hull's surge velocity as a state")
    ap.add_argument("--dt", type=float, default=None, help="integration step for the run and the coupling test, s")
    a = ap.parse_args()
    import coxswain.core.integrators as I
    if a.dt:
        I.estimate_step = lambda *args, **kw: float(a.dt)
    a.dt = a.dt or I.estimate_step(1.0)
    from coxswain.crew.handle_fv import HandleForceVelocity
    RP.FV = HandleForceVelocity(a.fv, "--fv") if a.fv else None
    RP.HOLD, RP.HOLD_OWN = float(a.hold), bool(a.hold_own)
    RP.BLADE, RP.BODY, RP.BENDING = "crossflow_normal", "own", True
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    print("%s rung 2 bending, own body; F-V V0 %s, hold %g%s" % (a.athlete, a.fv, a.hold,
                                                                 " (own pace)" if a.hold_own else ""), flush=True)
    s_p, (s_pop, q_drive, ph, q) = RP.population_tables(a.strokes)
    RP.athlete(a.athlete, a.strokes, s_pop, q_drive, ph, q, s_p["power"])
    sim, run = RP.LAST_RUN[a.athlete]
    sim._stroke_start = float(run.last_time[0])
    hist = sim._bend_history
    t, Y = run.last_time, run.last_states
    print("K = %.0f N m/rad, hinge lever k = %.3f, oarlocks %d"
          % (sim.oar_bending.k_ang, float(sim._bend_lever[0]), len(sim.boat.rig.seats[sim._seats[0]].oarlocks)))
    print("\n  t(s)  tau'   phi deg |   M      Q_v      S'    G_a/K   G_r   | numerical: sigma  omega | analytic: "
          "sigma omega | G_a=0: sigma omega | Q_v=0: sigma omega")
    rows, surge_rows = [], []
    for k in range(0, t.size, a.every):
        air = sim._air_mask[:, k]
        if air[0]:
            continue
        y = Y[:, k]
        p = Point(sim, t[k], y, air)
        phi0 = float(hist[0, 0, k])
        X = np.array([p.tau0, p.td0, phi0])
        try:
            f0, M, _q = p.rhs(X)
            if not np.all(np.isfinite(f0)):
                continue
            J = p.jacobian(X)
            Sp, G_a, G_r, _pd = p.partials(X)
        except Exception as exc:                                  # noqa: BLE001 -- report and move on
            print("  %5.2f  (skipped: %s)" % (t[k] - t[0], exc))
            continue
        # Q_v: the muscles' force-velocity slope less the body's velocity terms (the residual's d/dtau' with the blade
        # out, so the blade's share, carried by the hinge row, is not counted twice)
        Qv = (p.sim._muscle(p.tau0, p.td0 + 1e-4) - p.sim._muscle(p.tau0, p.td0 - 1e-4)) / 2e-4
        Qv -= p.body_velocity_slope(X)
        N = p.locks
        A = analytic(M, Qv, Sp, p.K, G_a, G_r, p.lev, N)
        En, Ea = np.linalg.eigvals(J), np.linalg.eigvals(A)
        E0 = np.linalg.eigvals(analytic(M, Qv, Sp, p.K, 0.0, G_r, p.lev, N))
        Eq = np.linalg.eigvals(analytic(M, 0.0, Sp, p.K, G_a, G_r, p.lev, N))
        if a.coupling:
            gain, rho, pstar = p.coupling(phi0, float(hist[1, 0, k]), a.dt)
        else:
            gain = rho = pstar = float("nan")
        if a.surge:                                               # the hull's surge velocity as a 4th state
            J4 = p.jacobian(np.array([p.tau0, p.td0, phi0, float(y[6])]))
            E4 = np.linalg.eigvals(J4)
            surge_rows.append((t[k] - t[0],) + worst(E4) + worst(En))
            print("  %5.2f  surge Jacobian: worst sigma %7.2f omega %6.1f   (3-state %7.2f %6.1f)   eig %s"
                  % (surge_rows[-1] + (" ".join("%.1f%+.1fi" % (e.real, e.imag) for e in E4),)), flush=True)
        r = (t[k] - t[0], p.td0, np.degrees(phi0), M, Qv, Sp, G_a / p.K, G_r) + worst(En) + worst(Ea) + worst(E0) \
            + worst(Eq) + (gain, rho, float(hist[1, 0, k]), pstar)
        rows.append(r)
        print("  %5.2f  %5.3f  %6.2f  | %7.1f %8.1f %6.3f  %6.3f %7.1f | %8.2f %6.1f | %7.2f %6.1f | %7.2f %6.1f | %7.2f %6.1f"
              " | gain %7.3f rho %7.3f  p %7.2f p* %7.2f" % r, flush=True)
    if rows:
        R = np.array(rows)
        print("\nnumerical Jacobian: max growth rate %.2f 1/s (at %.2f s, omega %.1f rad/s); fraction of drive unstable %.0f%%"
              % (R[:, 8].max(), R[int(np.argmax(R[:, 8])), 0], R[int(np.argmax(R[:, 8])), 9], 100 * np.mean(R[:, 8] > 0)))
        print("G_a/K range %.3f to %.3f (divergence where > 1)" % (R[:, 6].min(), R[:, 6].max()))
        if a.coupling:
            print("staggered interface gain dp'/dp: %.3f to %.3f (|gain| >= 1 amplifies); residual amplification rho "
                  "%.3f to %.3f" % (np.nanmin(R[:, 16]), np.nanmax(R[:, 16]), np.nanmin(R[:, 17]), np.nanmax(R[:, 17])))
    if surge_rows:
        S4 = np.array(surge_rows)
        print("with surge: max growth rate %.2f 1/s (omega %.1f rad/s at %.2f s); oscillatory unstable (omega > 5) at "
              "%d of %d samples" % (S4[:, 1].max(), S4[int(np.argmax(S4[:, 1])), 2], S4[int(np.argmax(S4[:, 1])), 0],
                                     int(np.sum((S4[:, 1] > 0) & (S4[:, 2] > 5))), len(S4)))


if __name__ == "__main__":
    main()
