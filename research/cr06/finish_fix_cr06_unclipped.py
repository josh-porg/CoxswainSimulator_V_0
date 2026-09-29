"""Finish defect: the sourced fix, as a study on her stroke (no committed code touched).

1. Release by [CR06]'s rule: once the drive is under way (blade normal velocity below -0.05 m/s), the blade comes out when
   its normal velocity returns to zero.
2. After release the oar stays a dynamic state with the blade out: theta'' = -tau / I_lock, tau her handle force per oar
   times s, UNCLIPPED (her negative recovery force). Before release the torque is clipped at zero, as in the validated runs.
3. When the oar's rate reverses (turn-round), it is held for the recovery (that part is 4.3's hands). The rate dropped at
   the hold is recorded.
Her rig, female table, whole measured body (leg clock + back law & travel), oar-only balance, T/320, 8 strokes.
Checks: release near her 0.894 s / -39.1 deg / -80 deg/s; turn-round near her 1.025 s / -44.34 deg; rate at the hold ~0
(the unfixed model froze at 114 deg/s); hull speed and velocity rms unchanged (4.11 m/s, ~0.03 m/s)."""
import os, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = "\n" + "grid = np.linspace(0.0, 1.0, 400"
exec(open(os.path.join(HERE, "female_segments_cr06.py")).read().split(MARKER)[0])
from coxswain.core.state import State  # noqa: E402
K = 0.398 / back


class FinishFixSim(DynamicOarSimulator):
    released = None
    turned = None
    driving = None
    events = None

    def _catch(self, t0, y, index):
        n = self.n_oar_states
        self.released = np.zeros(n, dtype=bool); self.turned = np.zeros(n, dtype=bool); self.driving = np.zeros(n, dtype=bool)
        if self.events is None:
            self.events = []
        return super()._catch(t0, y, index)

    def _oar_loads(self, t, state):
        saved = self._in_air
        if self.released is not None and self.released.any():
            self._in_air = (np.zeros(self.n_oar_states, dtype=bool) if saved is None else saved.copy()) | self.released
        try:
            return super()._oar_loads(t, state)
        finally:
            self._in_air = saved

    def _oar_rates(self, t, hull, angles, rates):
        ar, rr = super()._oar_rates(t, hull, angles, rates)
        if self.released is None:
            return ar, rr
        state = State.from_vector(hull)
        for slot in range(self.n_oar_states):
            if self.released[slot] and not self.turned[slot]:
                ar[slot] = rates[slot]
                rr[slot] = -self._torque(slot, float(angles[slot]), state, t) / I_LOCK
            elif self.released[slot] and self.turned[slot]:
                ar[slot] = 0.0; rr[slot] = 0.0
        return ar, rr

    def _integrate_sweep_catch(self, t_span, y0, dt):
        t_start, t_end = float(t_span[0]), float(t_span[1])
        n_steps = int(np.ceil((t_end - t_start) / dt))
        n = self.n_oar_states
        times = np.empty(n_steps + 1); states = np.empty((len(y0), n_steps + 1)); air = np.zeros((n, n_steps + 1), dtype=bool)
        t, y = t_start, np.array(y0, dtype=float)
        if self._in_air is None:
            self._in_air = np.zeros(n, dtype=bool)
        times[0], states[:, 0], air[:, 0] = t, y, self._in_air
        for i in range(n_steps):
            step = min(dt, t_end - t)
            y = integrators.rk4_step(self.derivative, t, y, step)
            t += step
            state = State.from_vector(y[:STATE_SIZE])
            if self._in_air.any():
                angle, rate = self._sweep_pose(t - self._stroke_start)
                for slot in np.flatnonzero(self._in_air):
                    y[STATE_SIZE + slot] = angle; y[STATE_SIZE + n + slot] = rate
                    if self._normal_velocity(slot, angle, rate, state) <= 0.0:
                        self._in_air[slot] = False
                        self.entries.append((int(slot), float(t), float(angle), float(rate)))
            for slot in range(n):
                if self._in_air[slot]:
                    continue
                angle, rate = float(y[STATE_SIZE + slot]), float(y[STATE_SIZE + n + slot])
                if not self.released[slot]:
                    nv = self._normal_velocity(slot, angle, rate, state)
                    if nv < -0.05:
                        self.driving[slot] = True
                    elif self.driving[slot] and nv >= 0.0:
                        self.released[slot] = True
                        self.events.append(("release", int(slot), float(t - self._stroke_start), angle, rate))
                elif not self.turned[slot] and rate >= 0.0:
                    self.turned[slot] = True
                    self.events.append(("turn", int(slot), float(t - self._stroke_start), angle, rate))
                    y[STATE_SIZE + n + slot] = 0.0
            times[i + 1], states[:, i + 1] = t, y
            air[:, i + 1] = self._in_air
        self._air_mask = air
        return times, states


oardynamics.InertiaProfile.of = staticmethod(oar_only)
try:
    boat = research.apply(her_boat(stature))
    boat.power_scales = np.ones(boat.n_seats)
    lock = boat.rig.seats[0].oarlocks[0]
    r_h, I_LOCK = float(lock.oar.inboard), float(lock.oar.inertia_about_lock)
    sim = FinishFixSim(boat, peak_torque=1.0, catch=research.catch, coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    # Second attempt (2026-09-14): the fixed finish angle froze the oar at -44.35 deg before the slip release could act
    # (the blade still drives there). Put each dynamic oar's finish angle out of reach so the release rule governs; the
    # boat's prescribed sweep (pre-entry path, hand tracks) is untouched.
    import dataclasses as _dc
    FINISH_BEFORE = [float(o.finish_angle) for o in sim._oars]
    sim._oars = [_dc.replace(o, finish_angle=np.radians(-80.0)) for o in sim._oars]
    # C2 study (2026-09-14): [CR06] p. 208, "the value of C2 which minimizes the net error is about 2.4 times the nominal
    # value", and with it their oar-angle error halves. Env C2_SCALE (default 1.0 = attempt 3 exactly).
    C2_SCALE = float(os.environ.get("C2_SCALE", "1.0"))
    C2_BEFORE = [float(o.blade.c2) for o in sim._oars]
    sim._oars = [_dc.replace(o, blade=_dc.replace(o.blade, c2=o.blade.c2 * C2_SCALE)) for o in sim._oars]
    print("c2 scale %.2f: blade c2 %s -> %s" % (C2_SCALE, C2_BEFORE, [round(float(o.blade.c2), 2) for o in sim._oars]), flush=True)
    # Third attempt (2026-09-14): with the torque clipped at zero before release, quadratic slip drag makes the slip
    # decay toward zero without ever crossing it, so the release never fired (0 in 8 strokes). Her measured force goes
    # negative near the finish: use it UNCLIPPED throughout, as measured, so her push can carry the slip through zero.
    # [LO00] reconciliation (2026-09-19): her handle force scaled uniformly, to ask what speed
    # the model gives at the force level three measured international W1x scullers applied.
    # Default 1.0, so the recorded end-to-end gate is byte-for-byte unchanged.
    F_SCALE = float(os.environ.get("F_SCALE", "1.0"))
    if F_SCALE != 1.0:
        print("handle force scaled x%.3f" % F_SCALE, flush=True)

    def torque(slot, ang, state, t):
        F = float(np.interp(np.mod(t, T_CR06), t_F, F_hand, period=T_CR06))
        return 0.5 * F * F_SCALE * r_h
    sim._torque = torque
    sim.crew_field = body_field(boat, True, K)
    # 4.3a candidate (2026-09-19): during the air phase put the handle where the seat
    # has carried it, instead of on the prescribed arc. Predictive -- dx comes from the
    # model's own hip, never from her oar. HIP_ENTRY=0 (default) leaves the gate as recorded.
    if os.environ.get("HIP_ENTRY", "0") == "1":
        _rower = boat.crew[0].rower
        _th0 = float(boat.oar_sweep.catch_angle)
        _hip0 = float(_rower.joint_positions(0.0)["hip"][0])
        _eps = 1e-4 * T_CR06

        def _hip_pose(tau):
            x = float(np.mod(float(tau), T_CR06))
            dx = float(_rower.joint_positions(x)["hip"][0]) - _hip0
            v = (float(_rower.joint_positions(min(x + _eps, T_CR06))["hip"][0])
                 - float(_rower.joint_positions(max(x - _eps, 0.0))["hip"][0])) / (2 * _eps)
            sn = float(np.clip(np.sin(_th0) - dx / r_h, -0.999, 0.999))
            th = float(np.arcsin(sn))
            return th, float(-v / (r_h * np.cos(th)))
        sim._sweep_pose = _hip_pose
        print("HIP_ENTRY: the seat carries the hands through the air phase", flush=True)
    run = sim.run_strokes(int(os.environ.get("N_STROKES", "8")), surge_speed=4.11, dt=integrators.estimate_step(T_CR06) / 4.0)
finally:
    oardynamics.InertiaProfile.of = REAL_OF

last = [e for e in sim.events if e[0] == "release"][-2:] + [e for e in sim.events if e[0] == "turn"][-2:]
print("finish angle before %s deg -> out of reach; releases logged %d, turns logged %d over %d strokes" % (
    np.round(np.degrees(FINISH_BEFORE), 2), sum(e[0] == "release" for e in sim.events), sum(e[0] == "turn" for e in sim.events), len(run.strokes)))
print("I_lock %.3f kg m^2; events on the last stroke (kind, slot, stroke time s, angle deg, rate deg/s):" % I_LOCK)
for e in last:
    print("  %-7s slot %d  t %.3f  %.2f deg  %.1f deg/s" % (e[0], e[1], e[2], np.degrees(e[3]), np.degrees(e[4])))
print("  her: release 0.894 s, -39.1 deg, -80 deg/s; turn-round 1.025 s, -44.34 deg | unfixed model: frozen at -44.35 deg, 0.870 s, -114 deg/s")
t, y = run.last_time, run.last_states
n = sim.n_oar_states
ph = np.mod(t - t[0], T_CR06)
tA, A = series("oar_angle_deg", MEAS, "value")
a_m = np.degrees(y[STATE_SIZE]); her = np.interp(ph, tA, A, period=T_CR06)
win = (ph > 0.80) & (ph < 1.10)
o_ph = np.argsort(ph)
for tq in (0.30, 0.50, 0.70, 0.80, 0.85, 0.894, 0.909, 0.95, 1.025):
    print("  t %.3f s: model %.2f deg, her %.2f deg, diff %+.2f" % (tq, np.interp(tq, ph[o_ph], a_m[o_ph]), np.interp(tq, tA, A, period=T_CR06),
                                                             np.interp(tq, ph[o_ph], a_m[o_ph]) - np.interp(tq, tA, A, period=T_CR06)))
print("oar angle rms vs her, 0.80-1.10 s: %.2f deg (max |diff| %.2f)" % (np.sqrt(np.mean((a_m - her)[win] ** 2)), np.max(np.abs(a_m - her)[win])))
grid = np.linspace(0.0, 1.0, 400, endpoint=False)
v_meas_g = np.interp(grid * T_CR06, t_v, v_boat, period=T_CR06)
o = np.argsort(ph)
v_m = np.interp(grid, ph[o] / T_CR06, np.hypot(y[6], y[7])[o], period=1.0)
rms = float(np.sqrt(np.mean(((v_m - v_m.mean()) - (v_meas_g - v_meas_g.mean())) ** 2)))
print("hull: speed %.4f m/s (%+.1f%%), velocity rms %.4f m/s, power %.1f W, last strokes %s" % (
    run.settled_speed(), 100 * (run.settled_speed() / 4.191 - 1), rms, run.settled_power(), np.round([s.mean_speed for s in run.strokes[-3:]], 4)))
