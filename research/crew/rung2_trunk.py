"""Phase 4.3, rung 2 with the trunk in the loop: the stroke clock and the trunk's angle both dynamic.

    .venv/Scripts/python.exe research/crew/rung2_trunk.py roundtrip [--strokes 10]
    .venv/Scripts/python.exe research/crew/rung2_trunk.py bias [--delay 0 0.1]
    .venv/Scripts/python.exe research/crew/rung2_trunk.py athletes [--strokes 10]

Rung 2 (rung2_clock.py) solves the hands' time law from a virtual-work balance along the stroke clock tau, with every
posture read on tau. Rungs 3-5 (research/cr06/coupled_trunk_cr06.py, coupled_control_cr06.py) made the trunk a state
driven by the hip moment on [CR06]'s own stroke, with a delayed PD designed on the coupled plant (SOURCES §178-182).
Here the two are joined on rung 2's crew (hands on the handle, the population's muscles, the research profile's blade,
the blade-placement takeover of §203):

* State: the simulator's, then [delta, delta', tau, tau']. delta is the trunk's rotation about the hip, in the
  sagittal plane, away from the posture the clock prescribes; delta = 0 is rung 2 exactly.
* The upper body is the head and the three trunk segments: their offsets from the hip, r_i(tau), are tabulated from
  the rower's own chain (segment_state and joint_positions, one solve), and rotated by delta. The arms stay on the
  hands' path, as in rung 5 (the elbow takes the few centimetres the shoulder moves).
* Two unknown accelerations, tau'' and delta''. The simulator's derivative, the clock's virtual-work residual and the
  hip moment are all affine in both, so each stage evaluates the derivative three times and solves a 2x2 system
  exactly (rung 2 and rung 5 each solved one).
* The hip moment the upper body needs, on its free body (trunk, head and arms, the handle force at the hands):
      M = (z_hand - z_hip) F_handle + sum_i [ r_x,i (-m_i g) - (r_x,i m_i a_z,i - r_z,i m_i a_x,i) ]
  (coupled_trunk_cr06.hip_moment's form, point masses, the handle force horizontal), and the muscles supply
  T = T_ff(tau) + bias + s (-Kp e(t - d) - Kd e'(t - d)), s = dM/d(delta'') (the effective inertia), Kp = wn^2 +
  lambda_c^2, Kd = 2 zeta wn: rung 5's design on the coupled plant (wn 5 rad/s, zeta 0.7, lambda_c^2 24.7 s^-2).
* The feedforward T_ff is the hip moment along the reference run (tau = clock, delta = 0) on the same boat: the trunk
  is asked to hold the posture it was given, as in rung 5.

Checks: ``roundtrip`` -- driven by the reference run's own Q and T_ff, the free run must return tau' = 1 and delta = 0
([CR06]'s boat, rung 1's on-water rhythm and [K05]'s drive law). ``bias`` -- a 10 N.m hip bias, with and without a
sensory delay: the trunk's error and the boat. ``athletes`` -- §181's test with the trunk free: the population's
muscles on the clock, each athlete's own reference hip moment as the trunk's feedforward.

Outputs: data/local/crew/rung2_trunk_<check>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import rung2_clock as C2                                         # noqa: E402
import rung2_population as RP                                    # noqa: E402

from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402

OUT = C2.OUT
G = 9.81
TRUNK = (0, 1, 2, 3)                                             # head, upper, mid, lower trunk (SEGMENT_ORDER)
ARMS = (4, 5, 6, 7)
WN, ZETA, LAM2 = 5.0, 0.7, 24.7                                  # rung 5's coupled-plant design (SOURCES §182)


def _rot(d):
    c, s = np.cos(d), np.sin(d)
    return np.array([[c, -s], [s, c]])


class TrunkClockSim(C2.ClockSim):
    """Rung 2 with the trunk's rotation about the hip a state."""

    def setup_trunk(self, mode, Q=None, body=None, tau_ff=None, bias=0.0, delay=0.0, wn=WN, zeta=ZETA, lam2=LAM2,
                    spin=True, ff_delay=0.0, efference=True, ff_model="efference"):
        """``spin``: the upper-body segments' own rotational inertias (de Leva, sagittal axis) in the hip moment.
        ``ff_delay``: the internal model reads the clock's motion this many seconds old, with no prediction.
        ``efference``: True -- it knows the clock's acceleration from its own current command (an efference copy, no
        delay) and only the clock's speed tau' is perceived late; False -- tau'' is read late too (a delayed
        acceleration feedback, which is unstable by construction: SOURCES sec. 206)."""
        self.setup_clock(mode, Q, body=body)
        self.tau_ff, self.bias, self.delay = tau_ff, float(bias), float(delay)
        self.spin, self.ff_delay, self.efference = bool(spin), float(ff_delay), bool(efference)
        #: ``ff_model``: how the internal model knows the clock (SOURCES sec. 217). ``"efference"``: as before (tau''
        #: from the current command when ``efference``). ``"learned"``: the previous stroke's clock speed against stroke
        #: position, tau'_L(tau), corrected by the delayed perception of the current stroke's departure from it:
        #:     tau'_hat = tau'_L(tau) + [tau'(t - d) - tau'_L(tau(t - d))],   tau''_hat = dtau'_L/dtau (tau) tau'_hat
        #: no knowledge of the true tau''. Until a stroke has been rowed it falls back to the delayed reading.
        self.ff_model = str(ff_model)
        self._learned, self._stroke_clock, self._stroke_mark = None, ([], []), None
        self.tau_hist = []
        self.clock_hist = ([], [], [])                           # t, tau', tau'' (accepted samples)
        self.Kp, self.Kd = wn ** 2 + lam2, 2.0 * zeta * wn
        self._trunk = None                                       # (delta, delta', delta'') while evaluating
        self.mlog = []                                           # reference: (t, c, M)
        self.trlog = []                                          # free: (t, tau, tau', delta, delta', feedback, s)
        self.hist = ([], [], [])
        self._tabulate()

    # -- the upper body's geometry on the clock ---------------------------------------------------------
    def _tabulate(self):
        rower = self.boat.crew[0].rower
        P = float(self.boat.timing.period)
        grid = np.linspace(0.0, P, 801)
        R, H, A = [], [], []
        for c in grid:
            pos, _v, _a, hand = rower.segment_state(float(c % P), with_hand=True)
            hip = rower.joint_positions(float(c % P))["hip"]
            R.append(np.asarray(pos, float)[:8][:, [0, 2]] - hip[[0, 2]])
            H.append(np.asarray(hand, float)[[0, 2]] - hip[[0, 2]])
            sh = rower.joint_positions(float(c % P))["shoulder"] - hip
            A.append(np.arctan2(sh[2], sh[0]))
        R, H, A = np.array(R), np.array(H), np.unwrap(np.array(A))
        R[-1], H[-1], A[-1] = R[0], H[0], A[0]
        self._aref = CubicSpline(grid, A, bc_type="periodic")     # the trunk's reference angle on the clock
        from coxswain.crew.kinematics import SEGMENT_ORDER
        segs = rower._segments
        self._spin_I = [float(segs[SEGMENT_ORDER[i]].inertia[0]) if SEGMENT_ORDER[i] in segs else 0.0 for i in TRUNK]
        self._r = [[CubicSpline(grid, R[:, i, j], bc_type="periodic") for j in range(2)] for i in range(8)]
        self._hand = [CubicSpline(grid, H[:, j], bc_type="periodic") for j in range(2)]
        self._P = P

    def _r_at(self, i, tau, k=0):
        x = float(tau) % self._P
        return np.array([self._r[i][0](x, k), self._r[i][1](x, k)])

    def crew_field(self, t, exact=False):
        mass, pos, vel, acc = C2.ClockSim.crew_field(self, t, exact)
        if self._trunk is None:
            return mass, pos, vel, acc
        tau, td, tdd = self._tau(t - self._stroke_start)
        dl, dld, dldd = self._trunk
        _m, _p, vf, af = self._body(self._stroke_start + tau)   # path velocity / acceleration per unit tau
        vf, af = np.asarray(vf, float), np.asarray(af, float)
        pos, vel, acc = pos.copy(), vel.copy(), acc.copy()
        Rm, Rd = _rot(dl), _rot(dl + np.pi / 2)                  # R and dR/d(delta); d2R/d(delta)2 = -R
        for i in TRUNK:
            r0, r1, r2 = self._r_at(i, tau), self._r_at(i, tau, 1), self._r_at(i, tau, 2)
            h0 = pos[i, [0, 2]] - r0                             # the hip on the path
            h1, h2 = vf[i, [0, 2]] - r1, af[i, [0, 2]] - r2
            p = h0 + Rm @ r0
            v = (h1 + Rm @ r1) * td + (Rd @ r0) * dld
            a = (h2 + Rm @ r2) * td * td + (h1 + Rm @ r1) * tdd + 2.0 * (Rd @ r1) * td * dld \
                - (Rm @ r0) * dld * dld + (Rd @ r0) * dldd
            pos[i, [0, 2]], vel[i, [0, 2]], acc[i, [0, 2]] = p, v, a
        return mass, pos, vel, acc

    def _dpath(self, tau):
        """d(position)/d(tau) for every row at the current delta: the path, the trunk rows rotated."""
        _m, _p, vf, _af = self._body(self._stroke_start + tau)
        vf = np.asarray(vf, float).copy()
        if self._trunk is not None:
            Rm = _rot(self._trunk[0])
            for i in TRUNK:
                r1 = self._r_at(i, tau, 1)
                vf[i, [0, 2]] = (vf[i, [0, 2]] - r1) + Rm @ r1
        return vf

    # -- the two equations ---------------------------------------------------------------------------
    def _handle_force(self, t, base):
        n = self.n_oar_states
        air = self._in_air if self._in_air is not None else np.zeros(n, dtype=bool)
        torque = float(np.sum(self.handle_torques(t, base, air)))
        lever = float(self.boat.rig.seats[0].oarlocks[0].oar.inboard)
        return -torque / lever                                   # the pull on the handle, positive in the drive

    def _equations(self, t, base, d, tau, td, tdd):
        """(clock residual, hip moment) at the current clock and trunk accelerations."""
        mass, pos, _vel, acc = self.crew_field(t)
        mass = np.asarray(mass, float)
        a_abs = acc + np.array([d[6], d[7], d[8]]) + np.array([0.0, 0.0, G])
        dp = self._dpath(tau)
        body = float(np.sum(mass[:, None] * a_abs * dp))
        air = self._in_air if self._in_air is not None else np.zeros(self.n_oar_states, dtype=bool)
        torque = self.handle_torques(t, base, air)
        dphi = float(self.boat.oar_sweep.rate(tau, self.boat.timing))
        resid = body + float(np.sum(torque)) * dphi
        hip = pos[TRUNK[0], [0, 2]] - _rot(self._trunk[0] if self._trunk else 0.0) @ self._r_at(TRUNK[0], tau)
        F = self._handle_force(t, base)
        hand_z = hip[1] + float(self._hand[1](float(tau) % self._P))
        M = (hand_z - hip[1]) * F
        if self.spin:                                            # each segment spins with the trunk: I theta''
            x = float(tau) % self._P
            dldd = self._trunk[2] if self._trunk else 0.0
            th_dd = float(self._aref(x, 2)) * td * td + float(self._aref(x, 1)) * tdd + dldd
            M -= sum(self._spin_I) * th_dd
        for i in TRUNK + ARMS:
            rx, rz = pos[i, 0] - hip[0], pos[i, 2] - hip[1]
            m = mass[i]
            ax, az = acc[i, 0] + d[6], acc[i, 2] + d[8]
            M += rx * (-m * G) - (rx * m * az - rz * m * ax)
        return resid, M

    def _feedback(self, t, e, ed, s):
        ht, he, hed = self.hist
        if self.delay > 0:
            if ht and t - self.delay > ht[0]:
                e, ed = float(np.interp(t - self.delay, ht, he)), float(np.interp(t - self.delay, ht, hed))
            else:
                e, ed = 0.0, 0.0
        return s * (-self.Kp * e - self.Kd * ed)

    def _path_moment(self, t, base, d, tau, td, a1):
        """The hip moment that keeps the trunk on its path (delta = delta' = delta'' = 0) at this clock motion, on the
        hull acceleration ``d`` the body feels: the internal model's feedforward (``tau_ff="internal"``)."""
        self._clock, self._trunk = (tau, td, a1), (0.0, 0.0, 0.0)
        try:
            return self._equations(t, base, d, tau, td, a1)[1]
        finally:
            self._clock, self._trunk = None, None

    def _evaluate(self, t, base, tau, td, a1, dl, dld, a2):
        self._clock, self._trunk = (tau, td, a1), (dl, dld, a2)
        try:
            d = DynamicOarSimulator.derivative(self, t, base)
            r, M = self._equations(t, base, d, tau, td, a1)
        finally:
            self._clock, self._trunk = None, None
        return d, r, M

    # -- state plumbing: [base, delta, delta', tau, tau'] --------------------------------------------
    def augmented_initial_state(self, surge_speed=4.0):
        if getattr(self, "start_state", None) is not None:
            return np.array(self.start_state, float)
        y = DynamicOarSimulator.augmented_initial_state(self, surge_speed)
        return np.concatenate([y, [0.0, 0.0, 0.0, 1.0]]) if self.mode == "free" else y

    def _catch(self, t0, y, index):
        if self.mode != "free":
            return super()._catch(t0, y, index)
        y = np.asarray(y, float)
        trunk = y[-4:-2].copy()
        after = C2.ClockSim._catch(self, t0, np.concatenate([y[:-4], y[-2:]]), index)
        return np.concatenate([after[:-2], trunk, after[-2:]])

    def _handle_power(self, times, states):
        if self.mode != "free":
            return super()._handle_power(times, states)
        return C2.ClockSim._handle_power(self, times, np.vstack([states[:-4], states[-2:]]))

    def derivative(self, t, y):
        y = np.asarray(y, float)
        if self.mode != "free":
            d = C2.ClockSim.derivative(self, t, y)
            if not self.mlog or t > self.mlog[-1][0]:
                c = t - self._stroke_start
                self._clock, self._trunk = (c, 1.0, 0.0), (0.0, 0.0, 0.0)
                try:
                    dd = DynamicOarSimulator.derivative(self, t, y)
                    _r, M = self._equations(t, y, dd, c, 1.0, 0.0)
                finally:
                    self._clock, self._trunk = None, None
                self.mlog.append((t, c, M))
            return d
        base, dl, dld, tau, td = y[:-4], float(y[-4]), float(y[-3]), float(y[-2]), float(y[-1])
        c = t - self._stroke_start
        clock_given = self._rec is not None or self._placing
        if self._rec is not None:
            tau, td, tdd_given = self._recovery_clock(c)
        else:
            tdd_given = 0.0
        e, ed = dl, dld
        d00, r00, M00 = self._evaluate(t, base, tau, td, tdd_given if clock_given else 0.0, dl, dld, 0.0)
        d01, r01, M01 = self._evaluate(t, base, tau, td, tdd_given if clock_given else 0.0, dl, dld, 1.0)
        s = M01 - M00
        fb = self._feedback(t, e, ed, s)
        internal = isinstance(self.tau_ff, str) and self.tau_ff == "internal"
        if internal and self.ff_delay > 0:
            ct, ctd, ctdd = self.clock_hist
            if ct and t - self.ff_delay > ct[0]:
                td_seen = float(np.interp(t - self.ff_delay, ct, ctd))
                tdd_seen = float(np.interp(t - self.ff_delay, ct, ctdd))
            else:
                td_seen, tdd_seen = td, 0.0
            if self.ff_model == "learned" and self._learned is not None and ct and t - self.ff_delay > ct[0]:
                lt, ltd, ldd = self._learned
                tau_d = float(np.interp(t - self.ff_delay, ct, self.tau_hist))
                x, xd = float(tau) % self._P, tau_d % self._P
                td_seen = float(np.interp(x, lt, ltd)) + (td_seen - float(np.interp(xd, lt, ltd)))
                tdd_seen = float(np.interp(x, lt, ldd)) * td_seen
        if clock_given:
            a1 = tdd_given
            if internal and self.ff_delay > 0:
                ff = self._path_moment(t, base, d00, tau, td_seen,
                                       a1 if (self.efference and self.ff_model != "learned") else tdd_seen)
            else:
                ff = self._path_moment(t, base, d00, tau, td, a1) if internal else self.tau_ff(tau)
            T = ff + self.bias + fb
            a2 = (T - M00) / s
            out = d00 + a2 * (d01 - d00)
        else:
            d10, r10, M10 = self._evaluate(t, base, tau, td, 1.0, dl, dld, 0.0)
            if internal and self.ff_delay > 0 and self.efference and self.ff_model != "learned":   # tau'' from the command
                p0 = self._path_moment(t, base, d00, tau, td_seen, 0.0)
                p1 = self._path_moment(t, base, d10, tau, td_seen, 1.0)
            elif internal and self.ff_delay > 0:                 # the clock's motion as perceived, d seconds old
                p0 = p1 = self._path_moment(t, base, d00, tau, td_seen, tdd_seen)
            elif internal:                                       # feedforward affine in the clock's acceleration
                p0 = self._path_moment(t, base, d00, tau, td, 0.0)
                p1 = self._path_moment(t, base, d10, tau, td, 1.0)
            else:
                p0 = p1 = self.tau_ff(tau)
            A = np.array([[r10 - r00, r01 - r00], [(M10 - M00) - (p1 - p0), M01 - M00]])
            b = np.array([self.Q(tau) - r00, p0 + self.bias + fb - M00])
            a1, a2 = np.linalg.solve(A, b)
            T = p0 + a1 * (p1 - p0) + self.bias + fb
            out = d00 + a1 * (d10 - d00) + a2 * (d01 - d00)
        ht, he, hed = self.hist
        if not ht or t > ht[-1]:
            ht.append(t); he.append(e); hed.append(ed)
            self.trlog.append((t, tau, td, dl, dld, float(fb), float(s)))
            self.tlog.append((t, tau, td, float(a1)))
            self.clock_hist[0].append(t); self.clock_hist[1].append(td); self.clock_hist[2].append(float(a1))
            self.tau_hist.append(float(tau))
            if self.ff_model == "learned":
                self._learn(tau, td)
        return np.concatenate([out, [dld, a2, td, a1]])

    def _learn(self, tau, td):
        """Collect this stroke's clock speed against stroke position; at the next stroke it becomes the model."""
        if self._stroke_mark != self._stroke_start:
            st, sd = self._stroke_clock
            if len(st) > 50:
                x = np.mod(np.asarray(st), self._P)
                o = np.argsort(x)
                x, v = x[o], np.asarray(sd)[o]
                keep = np.concatenate([[True], np.diff(x) > 1e-6])
                x, v = x[keep], v[keep]
                grid = np.linspace(0.0, self._P, 400, endpoint=False)
                vg = np.interp(grid, x, v, period=self._P)
                dv = np.gradient(np.concatenate([vg[-2:], vg, vg[:2]]), grid[1] - grid[0])[2:-2]
                self._learned = (grid, vg, dv)
            self._stroke_clock, self._stroke_mark = ([], []), self._stroke_start
        self._stroke_clock[0].append(float(tau))
        self._stroke_clock[1].append(float(td))

    def _integrate_sweep_catch(self, t_span, y0, dt):
        # rung2_clock's step hook reads tau, tau' from the last two entries, which they still are
        return C2.ClockSim._integrate_sweep_catch(self, t_span, y0, dt)


def make(boat, mode, **kw):
    boat.power_scales = np.ones(boat.n_seats)
    from coxswain.sim.control import Coxswain
    sim = TrunkClockSim(boat, peak_torque=0.0, catch="sweep", crew="handle",
                        coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True, **RP.blade_options(boat))
    sim.setup_trunk(mode, **kw)
    return sim


def table(log, P):
    """A periodic function of clock phase from (t, c, value) samples of the last stroke."""
    c = np.array([r[1] for r in log]) % P
    v = np.array([r[2] for r in log])
    o = np.argsort(c)
    c, v = c[o], v[o]
    keep = np.concatenate([[True], np.diff(c) > 1e-9])
    c, v = c[keep], v[keep]
    return lambda tau: float(np.interp(float(tau) % P, c, v, period=P))


def summary(sim, run):
    s = C2.summary(run)
    start = float(run.last_time[0])
    tr = np.array([r for r in sim.trlog if r[0] >= start]) if sim.trlog else None
    if tr is not None and len(tr):
        s.update(delta_max_deg=float(np.degrees(np.max(np.abs(tr[:, 3])))), delta_end_deg=float(np.degrees(tr[-1, 3])),
                 feedback_max=float(np.max(np.abs(tr[:, 5]))), tau_dot=[float(tr[:, 2].min()), float(tr[:, 2].max())],
                 inertia=[float(tr[:, 6].min()), float(tr[:, 6].max())])
    return s


def reference(boat, strokes):
    ref = make(boat, "reference")
    run = ref.run_strokes(strokes, surge_speed=4.3)
    P = float(boat.timing.period)
    start = float(run.last_time[0])
    Q = table([(t, c, q) for t, c, q in ref.qlog if t >= start], P)
    M = table([(t, c, m) for t, c, m in ref.mlog if t >= start], P)
    return ref, run, Q, M


def roundtrip(strokes, bias=0.0, delays=(0.0,)):
    RP.BLADE = "crossflow_normal"
    C2.TAKEOVER_DEG = C2.BIOROW_CATCH_TO_BURIED_DEG
    m = C2.R.V.ATHLETES["cr06"][0]()
    boat = C2.R.boat_for("cr06", m, "on-water", drive_law=True)
    ref, run_r, Q, M = reference(boat, strokes)
    s_r = C2.summary(run_r)
    Ms = [M(x) for x in np.linspace(0, float(boat.timing.period), 50)]
    print("reference (tau = clock, trunk on its path): %.4f m/s, IVV %.2f%%, %.0f W; hip moment %.0f to %.0f N.m"
          % (s_r["speed"], 100 * s_r["ivv"], s_r["power"], min(Ms), max(Ms)), flush=True)
    out = dict(reference=s_r)
    for dly in delays:
        boat2 = C2.R.boat_for("cr06", m, "on-water", drive_law=True)
        free = make(boat2, "free", Q=Q, tau_ff=M, bias=bias, delay=dly)
        P = float(boat2.timing.period)
        free.start_state = np.concatenate([run_r.last_states[:, -1], [0.0, 0.0, P, 1.0]])
        run_f = free.run_strokes(strokes, surge_speed=4.3)
        s_f = summary(free, run_f)
        key = "bias%.0f_delay%.0fms" % (bias, 1000 * dly)
        out[key] = s_f
        print("free, bias %.0f N.m, delay %.0f ms: %.4f m/s, IVV %.2f%%, %.0f W; tau' %.3f-%.3f; trunk error max %.2f deg"
              " (end %+.2f), feedback max %.1f N.m; effective inertia %.2f-%.2f kg m^2" % (
                  bias, 1000 * dly, s_f["speed"], 100 * s_f["ivv"], s_f["power"], *s_f.get("tau_dot", [np.nan, np.nan]),
                  s_f.get("delta_max_deg", np.nan), s_f.get("delta_end_deg", np.nan), s_f.get("feedback_max", np.nan),
                  *s_f.get("inertia", [np.nan, np.nan])), flush=True)
    return out


def athletes(strokes, ff="reference", ff_delay=0.0, delay=0.0, spin=True, efference=True, ff_model="efference"):
    """§181's test with the trunk free: population muscles on the clock; each athlete's own reference hip moment."""
    RP.BLADE = "crossflow_normal"
    C2.TAKEOVER_DEG = C2.BIOROW_CATCH_TO_BURIED_DEG
    from coxswain.crew.stroke import OnWaterTiming
    import k05_body
    import k05_population_test as KP
    V = C2.R.V
    pop_boat = V.cr06_boat(dict(leg_travel=None), timing=OnWaterTiming(KP.K05_RATE), stature=KP.K05_STATURE,
                           drive_law=True, mass=KP.K05_MASS)
    body, _k = k05_body.body_field(pop_boat, leg_travel=0.51)
    pop = RP.make(pop_boat, "reference", body=body)
    run_p = pop.run_strokes(strokes, surge_speed=4.3)
    s_p = C2.summary(run_p)
    ph, q = RP.q_table(pop, run_p)
    P_pop = float(pop_boat.timing.period)
    on = ph * P_pop <= float(pop_boat.timing.drive_fraction) * P_pop
    s_pop = np.maximum.accumulate(RP.progress_of(pop_boat, ph[on] * P_pop))
    q_drive = q[on]
    keep = np.concatenate([[True], np.diff(s_pop) > 1e-9])
    s_pop, q_drive = s_pop[keep], q_drive[keep]
    out = dict(population=s_p)
    for name in ("cr06", "br24"):
        m = V.ATHLETES[name][0]()
        boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
        ref, run_r, _Qr, M = reference(boat, strokes)
        P = float(boat.timing.period)
        D = float(boat.timing.drive_fraction) * P
        k = m["power"] / s_p["power"]

        def Q(tau, boat=boat, k=k, D=D, P=P):
            if tau <= D:
                return k * float(np.interp(float(RP.progress_of(boat, tau)[0]), s_pop, q_drive))
            return k * float(np.interp(np.mod(tau / P, 1.0), ph, q, period=1.0))
        boat2 = C2.R.boat_for(name, m, "on-water", drive_law=True)
        free = make(boat2, "free", Q=Q, tau_ff=("internal" if ff == "internal" else M), ff_delay=ff_delay,
                    delay=delay, spin=spin, efference=efference, ff_model=ff_model)
        free.start_state = np.concatenate([run_r.last_states[:, -1], [0.0, 0.0, P, 1.0]])
        try:
            run_f = free.run_strokes(strokes, surge_speed=4.3)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            tr = np.array(free.trlog) if free.trlog else np.zeros((1, 7))
            stroke = int(tr[-1, 0] // P) + 1
            out[name] = dict(diverged=True, stroke=stroke, delta_last_deg=float(np.degrees(tr[-1, 3])),
                             delta_max_deg=float(np.degrees(np.nanmax(np.abs(tr[:, 3])))), error=str(exc))
            print("%s, trunk free (%s feedforward, ff delay %.0f ms, fb delay %.0f ms): DIVERGED in stroke %d (trunk error"
                  " reached %.1f deg; %s)" % (name, ff, 1000 * ff_delay, 1000 * delay, stroke,
                                              out[name]["delta_max_deg"], exc), flush=True)
            continue
        s_f = summary(free, run_f)
        ht, hs, ha = RP.measured_track(name)
        tt, ss, aa = RP.model_track(free, run_f, True)
        prog = RP.compare(tt, ss, ht, hs)
        at = s_f["speed"] * (m["power"] / s_f["power"]) ** (1 / 3) / m["speed"] - 1
        s_f.update(progress_rms=prog[0], speed_error=at, measured=dict(speed=m["speed"], ivv=m["ivv"], power=m["power"]))
        out[name] = s_f
        print("%s, trunk free (%s feedforward, ff delay %.0f ms, fb delay %.0f ms): %.3f m/s, IVV %.1f%% (measured %.1f%%), %.0f W; at the measured power %+.1f%%; hands rms %.3f;"
              " trunk error max %.2f deg; tau' %.3f-%.3f" % (name, ff, 1000 * ff_delay, 1000 * delay, s_f["speed"], 100 * s_f["ivv"], 100 * m["ivv"],
                                                             s_f["power"], 100 * at, prog[0], s_f.get("delta_max_deg", np.nan),
                                                             *s_f.get("tau_dot", [np.nan, np.nan])), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["roundtrip", "bias", "athletes"])
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--delay", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--ff-delay", type=float, default=0.0, help="the internal model's lag on the clock's motion, s")
    ap.add_argument("--fb-delay", type=float, default=0.0, help="the PD's sensory delay in the athletes check, s")
    ap.add_argument("--no-spin", action="store_true", help="leave out the segments' own rotational inertias")
    ap.add_argument("--no-efference", action="store_true",
                    help="the internal model reads the clock's acceleration late too (no efference copy)")
    ap.add_argument("--ff", default="reference", choices=["reference", "internal"],
                    help="the trunk's feedforward: the reference run's hip moment on the clock (posture only) or an "
                         "internal model, the moment that holds the path at the current clock motion")
    a = ap.parse_args()
    if a.check == "roundtrip":
        out = roundtrip(a.strokes)
    elif a.check == "bias":
        out = roundtrip(a.strokes, bias=10.0, delays=a.delay)
    else:
        out = athletes(a.strokes, a.ff, a.ff_delay, a.fb_delay, not a.no_spin, not a.no_efference)
    json.dump(out, open(os.path.join(OUT, "rung2_trunk_%s%s.json" % (a.check, ("" if a.ff == "reference" else "_internal") + ("_ffd%.0fms" % (1000 * a.ff_delay) if a.ff_delay else "")
                        + ("_fbd%.0fms" % (1000 * a.fb_delay) if a.fb_delay else "")
                        + ("_noeff" if a.no_efference else ""))),
                        "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
