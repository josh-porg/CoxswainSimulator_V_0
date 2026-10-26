"""Phase 4.3, rung 2 prototype (sprint 2, S2.5): the stroke clock as a dynamic coordinate.

    .venv/Scripts/python.exe research/crew/rung2_clock.py [--strokes 10]

Question. Rung 1 (``DynamicOarSimulator(crew="handle")``) puts the hands on the handle and moves the body and the
oar on a prescribed clock: the sweep angle and every body posture are read at c = t - stroke_start. Rung 2 makes the
hands' time law an output. Here the clock becomes a state tau, with tau' = 1 reproducing rung 1 exactly: the sweep and
the postures are read at tau, velocities scale by tau' and accelerations by tau'^2 plus tau'' times the velocity.

Equation. Virtual work along tau on the body-and-oar subsystem. The seat rolls without friction and the stretcher and
pin are fixed to the hull, so the contact forces do no work in a displacement along tau that holds the hull fixed:

    Q_muscle(tau) = sum_i m_i (a_hull + a_i,rel - g) . dx_i/dtau  +  sum_oars T_handle(phi'') dphi/dtau

with a_i,rel = x_i''(tau) tau'^2 + x_i'(tau) tau'' and T_handle the oar's own balance (``handle_torques``: I_oar phi''
minus the blade's normal load times its arm). Hull and handle torque are affine in tau'', so each integrator stage
evaluates the simulator's derivative at tau'' = 0 and 1 and solves exactly (as rung 5 does for the trunk); the hull
feels the body through the crew field at tau.

The first check (this script): the round trip. The reference run (tau = c) gives Q_ref(tau) by inverse dynamics on its
last stroke; driven by Q_ref, the free run must return tau' = 1. Rung 1's boat: [CR06]'s single, the on-water rhythm
and [K05]'s drive law (research/crew/handle_rung1.py).

Known simplifications of the prototype: the hand positions used for the oar's moment arms are read on the clock, not at
tau; the hull's rotation is left out of the body's absolute acceleration (it is small); tau stays dynamic through the
recovery, and the period is not yet enforced (a drifting tau would shift the stroke boundaries).

Outputs: data/local/crew/rung2_clock.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]
import handle_rung1 as R                                         # noqa: E402

from coxswain.sim.control import Coxswain                        # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402

OUT = os.path.join(ROOT, "data", "local", "crew")
os.makedirs(OUT, exist_ok=True)
G = 9.81


class ClockSim(DynamicOarSimulator):
    """Rung 1 with the stroke clock a state (mode 'free') or the clock itself (mode 'reference')."""

    def setup_clock(self, mode, Q=None):
        self.mode, self.Q = mode, Q
        self._clock = None
        self.qlog = []                                           # reference: (c, Q) at accepted times
        self.tlog = []                                           # free: (t, tau, tau')
        self.rlog = []                                           # free: each release (stroke start, c, tau, tau')
        self._rec = None
        self._last = -np.inf

    # -- the clock -------------------------------------------------------------------------------------
    def _tau(self, c):
        return (float(c), 1.0, 0.0) if self._clock is None else self._clock

    def _sweep_pose(self, c):
        tau, td, _ = self._tau(c)
        sw, tm = self.boat.oar_sweep, self.boat.timing
        return float(sw(tau, tm)), float(sw.rate(tau, tm)) * td

    def _sweep_motion(self, c):
        tau, td, tdd = self._tau(c)
        sw, tm = self.boat.oar_sweep, self.boat.timing
        step = 1e-4 * float(tm.period)
        low, high = max(tau - step, 0.0), tau + step
        rate = float(sw.rate(tau, tm))
        accel = (float(sw.rate(high, tm)) - float(sw.rate(low, tm))) / (high - low)
        return rate * td, accel * td * td + rate * tdd

    def crew_field(self, t, exact=False):
        tau, td, tdd = self._tau(t - self._stroke_start)
        mass, pos, vel, acc = self.boat.crew_field(self._stroke_start + tau)
        vel = np.asarray(vel, float)
        return mass, np.asarray(pos, float), vel * td, np.asarray(acc, float) * td * td + vel * tdd

    # -- the state ---------------------------------------------------------------------------------------
    def augmented_initial_state(self, surge_speed=4.0):
        if getattr(self, "start_state", None) is not None:      # continue from a settled run
            return np.array(self.start_state, float)
        y = super().augmented_initial_state(surge_speed)
        return np.concatenate([y, [0.0, 1.0]]) if self.mode == "free" else y

    def _catch(self, t0, y, index):
        if self.mode != "free":
            return super()._catch(t0, y, index)
        tau, td = float(y[-2]), float(y[-1])
        P = float(self.boat.timing.period)
        if tau > 0.5 * P:
            tau -= P                                             # a new stroke: the clock wraps
        self._rec = None                                         # the drive is dynamic again
        self._clock = (tau, td, 0.0)
        try:
            after = super()._catch(t0, np.asarray(y[:-2]), index)
        finally:
            self._clock = None
        return np.concatenate([after, [tau, td]])

    def _integrate_sweep_catch(self, t_span, y0, dt):
        # the base loop resets every oar onto the sweep after each step through _sweep_pose(c); read the clock
        # from the step's own state there
        if self.mode != "free":
            return super()._integrate_sweep_catch(t_span, y0, dt)
        import coxswain.core.integrators as I
        orig = I.rk4_step

        def step(f, t, y, h):
            y2 = orig(f, t, y, h)
            c = t + h - self._stroke_start
            done = getattr(self, "_handle_done", None)
            if self._rec is None and done is not None and np.all(done):
                self._start_recovery(c, float(y2[-2]), float(y2[-1]))
            if self._rec is not None:                            # the recovery follows its curve exactly
                y2[-2], y2[-1] = self._recovery_clock(c)[:2]
            self._clock = (float(y2[-2]), float(y2[-1]), 0.0)    # used by the reset that follows
            return y2
        I.rk4_step = step
        try:
            return super()._integrate_sweep_catch(t_span, y0, dt)
        finally:
            I.rk4_step = orig
            self._clock = None

    # -- the period: the recovery retimed to end at the catch on the clock ------------------------------
    def _start_recovery(self, c0, tau0, td0):
        """From the release (clock c0, tau0, tau0') to the catch at (P, P, 1): a cubic Hermite in clock time,
        so tau and tau' are continuous (no impulse on the hull) and the next stroke starts on time."""
        P = float(self.boat.timing.period)
        if c0 >= P - 1e-3 or tau0 >= P - 1e-3:
            self.rlog.append((self._stroke_start, c0, tau0, td0, "overrun"))
            return
        self._rec = (c0, tau0, td0, P)
        self.rlog.append((self._stroke_start, c0, tau0, td0, "ok"))

    def _recovery_clock(self, c):
        c0, t0, d0, P = self._rec
        H = P - c0
        u = min(max((c - c0) / H, 0.0), 1.0)
        h00, h10, h01, h11 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u, -2*u**3 + 3*u**2, u**3 - u**2
        d00, d10, d01, d11 = 6*u**2 - 6*u, 3*u**2 - 4*u + 1, -6*u**2 + 6*u, 3*u**2 - 2*u
        s00, s10, s01, s11 = 12*u - 6, 6*u - 4, -12*u + 6, 6*u - 2
        tau = h00 * t0 + h10 * H * d0 + h01 * P + h11 * H * 1.0
        td = (d00 * t0 + d10 * H * d0 + d01 * P + d11 * H) / H
        tdd = (s00 * t0 + s10 * H * d0 + s01 * P + s11 * H) / H ** 2
        return tau, td, tdd

    # -- the equation ------------------------------------------------------------------------------------
    def residual(self, t, base, d, tau, td, a):
        """The generalised muscle force along tau the motion needs at tau'' = a (W per unit tau')."""
        m, _p, vf, af = self.boat.crew_field(self._stroke_start + tau)
        m = np.asarray(m, float); vf = np.asarray(vf, float); af = np.asarray(af, float)
        a_rel = af * td * td + vf * a
        a_abs = a_rel + np.array([d[6], d[7], d[8]]) + np.array([0.0, 0.0, G])
        body = float(np.sum(m[:, None] * a_abs * vf))
        n = self.n_oar_states
        air = self._in_air if self._in_air is not None else np.zeros(n, dtype=bool)
        torque = self.handle_torques(t, base, air)
        dphi = float(self.boat.oar_sweep.rate(tau, self.boat.timing))
        return body + float(np.sum(torque)) * dphi

    def derivative(self, t, y):
        y = np.asarray(y, float)
        if self.mode != "free":
            self._clock = None
            d = super().derivative(t, y)
            if t > self._last:
                c = t - self._stroke_start
                self._clock = (c, 1.0, 0.0)
                try:
                    self.qlog.append((t, c, self.residual(t, y, d, c, 1.0, 0.0)))
                finally:
                    self._clock = None
                self._last = t
            return d
        base, tau, td = y[:-2], float(y[-2]), float(y[-1])
        if self._rec is not None:                                # the recovery: the clock is prescribed
            tau, td, tdd = self._recovery_clock(t - self._stroke_start)
            self._clock = (tau, td, tdd)
            try:
                d = super().derivative(t, base)
            finally:
                self._clock = None
            return np.concatenate([d, [td, tdd]])
        D, Rs = [], []
        for a in (0.0, 1.0):
            self._clock = (tau, td, a)
            try:
                d = super().derivative(t, base)
                Rs.append(self.residual(t, base, d, tau, td, a))
            finally:
                self._clock = None
            D.append(d)
        alpha = (self.Q(tau) - Rs[0]) / (Rs[1] - Rs[0])
        if t > self._last:
            self.tlog.append((t, tau, td, alpha)); self._last = t
        return np.concatenate([D[0] + alpha * (D[1] - D[0]), [td, alpha]])


def build(mode, Q=None):
    measured = R.V.ATHLETES["cr06"][0]()                      # her measured stroke (leg travel sets her stature)
    boat = R.boat_for("cr06", measured, "on-water", drive_law=True)
    boat.power_scales = np.ones(boat.n_seats)
    sim = ClockSim(boat, peak_torque=0.0, catch="sweep", crew="handle",
                   coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    sim.setup_clock(mode, Q)
    return sim


def summary(run):
    last = run.strokes[-3:]
    return dict(speed=float(np.mean([s.mean_speed for s in last])), ivv=float(np.mean([s.surge_swing for s in last])),
                power=float(np.mean([s.handle_power for s in last])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    a = ap.parse_args()
    ref = build("reference")
    run = ref.run_strokes(a.strokes, surge_speed=4.3)
    s_ref = summary(run)
    P = float(ref.boat.timing.period)
    start = float(run.last_time[0])
    lg = np.array([(c, q) for t, c, q in ref.qlog if t >= start])
    o = np.argsort(lg[:, 0]); c, q = lg[o, 0], lg[o, 1]
    keep = np.concatenate([[True], np.diff(c) > 1e-9]); c, q = c[keep], q[keep]
    Q = lambda tau: float(np.interp(np.mod(tau, P), c, q, period=P))      # noqa: E731
    print("reference (rung 1): %.4f m/s, IVV %.2f%%, %.0f W; Q_ref %.0f to %.0f" % (
        s_ref["speed"], 100 * s_ref["ivv"], s_ref["power"], q.min(), q.max()), flush=True)
    releases_ref = [s.drive_duration for s in run.strokes[-3:]]
    out = dict(reference=dict(s_ref, release_clock=releases_ref))
    # (1) from the reference's settled state (the clock at the end of its stroke): the equation alone;
    # (2) from the same unsettled start as the reference: the equation with the period enforced
    for label, start_state in (("settled start", np.concatenate([run.last_states[:, -1], [P, 1.0]])),
                               ("unsettled start (4.3 m/s)", None)):
        free = build("free", Q)
        free.start_state = start_state
        run2 = free.run_strokes(a.strokes, surge_speed=4.3)
        s_free = summary(run2)
        tl = np.array(free.tlog)
        last = tl[tl[:, 0] >= float(run2.last_time[0])]
        rel = [r for r in free.rlog if r[0] >= float(run2.last_time[0]) - 1e-9]
        over = sum(1 for r in free.rlog if r[4] != "ok")
        print("free clock, %s: %.4f m/s, IVV %.2f%%, %.0f W; drive tau' on the last stroke %.4f to %.4f (all %.4f to"
              " %.4f); release on the clock %s s (reference %.3f); overruns %d" % (
                  label, s_free["speed"], 100 * s_free["ivv"], s_free["power"], last[:, 2].min(), last[:, 2].max(),
                  tl[:, 2].min(), tl[:, 2].max(), ", ".join("%.3f" % r[1] for r in rel), releases_ref[-1], over),
              flush=True)
        out[label] = dict(s_free, tau_dot_last=[float(last[:, 2].min()), float(last[:, 2].max())],
                          tau_dot_all=[float(tl[:, 2].min()), float(tl[:, 2].max())],
                          release_clock=[float(r[1]) for r in rel], overruns=over)
    json.dump(out, open(os.path.join(OUT, "rung2_clock.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
