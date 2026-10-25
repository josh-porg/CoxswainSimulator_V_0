"""Step 4.3, rung 5 (rebuilt 2026-10-24): her controlled trunk inside the simulation, two-way coupled.

    .venv/Scripts/python.exe research/cr06/coupled_trunk_cr06.py [--strokes 12]

The originals (cr06/coupled_trunk_cr06.py, coupled_trunk_delayed_cr06.py, 2026-09-14) were lost with a
session scratchpad; their results are in docs/TRACKING.md ("Step 4.3, fifth rung"). This rebuild follows the
record's method and prints the recorded numbers beside its own.

Boat and drive: [CR06]'s single as self_consistent_cr06.py builds it (her rig, 19.7 kg hull, [CR06] oar,
sweep catch, the research physics profile), with the women's table at her 1.758 m as female_segments_cr06.py
does; the oar driven by her measured handle force, nothing fitted; dt = the integrator's estimate / 8 (see DT_RUN).

Body: her legs on her clock (the leg channel warped onto the model's, self_consistent_body_cr06.py), and above
the hip the rigid body of rungs 2-4: head and the three trunk segments at the package rower's distances along a
0.570 m link, both arms as point masses at the shoulder. The trunk angle is either prescribed (her back channel,
as rungs 2-4, the Fourier reference of trunk_forward_cr06.py) or a state.

Two-way coupling. With the trunk a state, the simulator's state gains [angle, rate]. At every integrator
stage the simulator's own derivative is evaluated with the trunk's angular acceleration at 0 and at 1; the
hull's response and the hip moment are both affine in it, so the hip-moment equation is solved exactly and
the hull feels the trunk while the trunk feels the hull. The catch reset (which zeroes everything after the
oar angles) keeps the trunk's state.

Torque: the feedforward is the hip moment along the prescribed-trunk run on the simulated hull (rung 2's
torque, as the record used: so the undisturbed coupled run must reproduce the prescribed one), plus a PD on
the trunk-angle error, 5 rad/s, gains from the plant (trunk_forward_cr06.py), fed the error d seconds old.

Cases (record's table): prescribed trunk; coupled undisturbed; 0.1 deg start offset; 10 N.m torque bias;
the bias with 45, 75, 100 and 117 ms delay. Writes data/local/cr06_study/coupled_trunk_cr06.json.

Audit options (each runs the prescribed trunk and the bias cases at 0, 100 and 117 ms):
    --one-way      the hull sees only the prescribed trunk; the trunk still feels the simulated hull
    --hull-blind   the hull feels the trunk, but the trunk's own equation leaves out the hull's recoil
They write coupled_trunk_cr06_one_way.json / _hull_blind.json.

Finding (2026-10-25, SOURCES §179). Two-way coupled, the hull recoils from the trunk's swing (0.17 m/s^2 per
rad/s^2, the upper body's 16.5 kg m over the 97 kg system), so the trunk's effective inertia falls from 9.12 to
6.0-6.8 kg m^2 and its toppling rate rises (lambda^2 ~ 26 against 18 /s^2). The gains, designed on the one-way
plant, then hold it with half the net stiffness: the 10 N.m bias costs ~8 deg, not the record's 3.7, and the
delay margin rises to ~147 ms, so 117 ms holds. The record's bias numbers are reproduced only by --one-way
(3.28 / -3.02 deg, 20.2 N.m); its 117 ms divergence by no variant.
"""
import argparse
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
HERE = CODE                                                     # the exec'd study files look for siblings here
_src = open(os.path.join(CODE, "self_consistent_body_cr06.py"), encoding="utf-8").read().split("grid = np.linspace")[0]
exec(_src)                                                      # cr06_single, body_field, DynamicOarSimulator, research, ...
sys.path.insert(0, CODE)
import her_stroke as HS                                         # noqa: E402
import hip_topdown_cr06 as HT                                   # noqa: E402
import trunk_forward_cr06 as TF                                 # noqa: E402

OUT = HS.OUT
STATURE = HS.STATURE
TRUNK_IDX = {"head": 0, "upper_trunk": 1, "mid_trunk": 2, "lower_trunk": 3}
ARM_IDX = (4, 5, 6, 7)
# The step. The research profile's fitted scull C2 (140.9, adopted 2026-09-18, after the original rung 5 ran) makes the
# torque-driven oar with the oar's own inertia unstable at the study's half step (estimate_step / 2): it diverges
# mid-drive. A quarter of that step runs, and gives the same speed to 0.1% as the default C2 at the half step.
DT_RUN = DT / 4
RECORDED = dict(prescribed=dict(speed=4.115, rms=0.030), undisturbed=dict(speed=4.1173, rms=0.0297, catch_min=2.98, err=(0.0017, -0.0010), fb=0.0),
                bias=dict(speed=4.1183, rms=0.0299, err=(3.71, -3.43), fb=22.2),
                delays={45: (3.80, 3.35, 23.0, 0.0303), 75: (3.88, 3.21, 23.7, 0.0312), 100: (4.17, 2.89, 27.9, 0.0331), 117: (45.6, -31.4, 485, 0.64)})


def her_boat():
    base = catalog.single_scull(rate=RATE, rower_mass=75.0, rower_stature=STATURE)
    rig = build_sculling_rig(n_seats=1, spacing=1.22, stern_station=-0.35, span=0.80, oarlock_height=0.32, oar=CR06_OAR)
    return Boat(name="CR06 women's single", offsets=base.offsets, rig=rig, hull_mass=19.7, hull_inertia=base.hull_inertia,
                timing=base.timing, appendages=base.appendages, water=base.water,
                default_anthropometry=catalog.RowerAnthropometry(mass=75.0, stature=STATURE, sex="female"),
                force_profile=base.force_profile, oar_sweep=ARC)


class TrunkSim(DynamicOarSimulator):
    """The simulator with the trunk's angle and rate appended to its state (mode 'free'), or the trunk on its
    reference (mode 'prescribed')."""

    def setup_trunk(self, leg_field, hip_fn, ang, dist, mode, tau_ff=None, wn=5.0, delay=0.0, bias=0.0, a0_offset=0.0, tau=0.274,
                    hull_blind=False, one_way=False):
        self.leg_field, self.hip_fn, self.ang, self.dist, self.mode = leg_field, hip_fn, ang, dist, mode
        self.tau_ff, self.bias, self.a0_offset, self.delay = tau_ff, bias, np.radians(a0_offset), delay
        self.Kp, self.Kd = wn ** 2 + 1 / tau ** 2, 2 * TF.ZETA * wn
        self.one_way = one_way                                  # audit: the hull sees the prescribed trunk only
        self.hull_blind = hull_blind                            # audit: the trunk's equation ignores the hull's recoil (see main)
        self._trunk = None
        self.hist = ([], [], [])                                 # t, error, error rate (for the delay)
        self.log = []                                            # (t, error, feedback, hull accel)

    # the state: hull, oar states, then [angle, rate] when the trunk is free
    def augmented_initial_state(self, surge_speed=4.0):
        y = super().augmented_initial_state(surge_speed)
        if self.mode == "free":
            y = np.concatenate([y, [self.ang(0.0) + self.a0_offset, self.ang(0.0, 1)]])
        return y

    def _catch(self, t0, y, index):
        if self.mode != "free":
            return super()._catch(t0, y, index)
        trunk = np.array(y[-2:], float)
        after = super()._catch(t0, np.asarray(y[:-2]), index)
        return np.concatenate([after, trunk])

    def _upper(self, t, a, ad, add):
        """Positions, velocities, accelerations (hull frame) of the upper body for trunk (a, a', a'')."""
        hp, hv, ha = self.hip_fn(t)
        out = {}
        for name, d in self.dist.items():
            u = np.array([np.cos(a), 0.0, np.sin(a)]); n = np.array([-np.sin(a), 0.0, np.cos(a)])
            out[name] = (hp + d * u, hv + d * ad * n, ha + d * (add * n - ad ** 2 * u))
        return out

    def field(self, t, exact=False):
        mass, pos, vel, acc = self.leg_field(t)
        pos, vel, acc = np.array(pos, float), np.array(vel, float), np.array(acc, float)
        a, ad, add = self._trunk if (self._trunk is not None and not self.one_way) else (self.ang(t), self.ang(t, 1), self.ang(t, 2))
        up = self._upper(t, a, ad, add)
        for name, i in TRUNK_IDX.items():
            p, v, ac = up[name]; y_lat = pos[i, 1]
            pos[i], vel[i], acc[i] = p, v, ac; pos[i, 1] = y_lat
        p, v, ac = up["arms"]
        for i in ARM_IDX:
            y_lat = pos[i, 1]; pos[i], vel[i], acc[i] = p, v, ac; pos[i, 1] = y_lat
        return mass, pos, vel, acc

    def hip_moment(self, t, a, ad, add, a_hull):
        """Rung 2's extension moment of both hips on the simulated hull (hip_topdown_cr06.moment's terms)."""
        hp, hv, ha = self.hip_fn(t)
        F = float(np.interp(t % T_CR06, t_F, F_hand, period=T_CR06))
        Mx = (HT.Z_HAND - hp[2]) * F                            # handle: horizontal, at the oarlock height
        for name, m, d, I in HT.upper_body():
            rx, rz = d * np.cos(a), d * np.sin(a)
            ax = a_hull + ha[0] + d * (-np.sin(a) * add - np.cos(a) * ad ** 2)
            az = ha[2] + d * (np.cos(a) * add - np.sin(a) * ad ** 2)
            Mx += rx * (-m * 9.81) - (rx * m * az - rz * m * ax) - I * add
        return Mx

    def derivative(self, t, y):
        y = np.asarray(y, float)
        if self.mode != "free":
            self._trunk = None
            d = super().derivative(t, y)
            # the feedforward: rung 2's hip moment along this run, on the simulated hull
            if not self.log or t > self.log[-1][0]:
                self.log.append((t, self.hip_moment(t, self.ang(t), self.ang(t, 1), self.ang(t, 2), float(d[6]))))
            return d
        base, (a, ad) = y[:-2], y[-2:]
        d = []
        for alpha in (0.0, 1.0):
            self._trunk = (a, ad, alpha)
            d.append(super().derivative(t, base))
        self._trunk = None
        acc0, acc1 = d[0][6], d[1][6]                           # hull's x acceleration, affine in alpha
        M0 = self.hip_moment(t, a, ad, 0.0, acc0)
        M1 = self.hip_moment(t, a, ad, 1.0, acc0 if self.hull_blind else acc1)
        e, ed = a - self.ang(t), ad - self.ang(t, 1)
        if self.delay > 0:
            ht, he, hed = self.hist
            if ht and t - self.delay > ht[0]:
                e_d, ed_d = float(np.interp(t - self.delay, ht, he)), float(np.interp(t - self.delay, ht, hed))
            else:
                e_d, ed_d = 0.0, 0.0
        else:
            e_d, ed_d = e, ed
        s = M1 - M0                                             # d(moment)/d(alpha): minus the effective inertia
        fb = s * (-self.Kp * e_d - self.Kd * ed_d)
        torque = self.tau_ff(t) + self.bias + fb
        alpha = (torque - M0) / s
        out = d[0] + alpha * (d[1] - d[0])
        # the delay line and the log, in time order (RK4's stages run forward within a step)
        ht, he, hed = self.hist
        if not ht or t > ht[-1]:
            ht.append(t); he.append(e); hed.append(ed)
            self.log.append((t, e, fb))
        return np.concatenate([out, [ad, alpha]])


def build(mode, **kw):
    oardynamics.InertiaProfile.of = staticmethod(oar_only)
    boat = research.apply(her_boat())
    boat.power_scales = np.ones(boat.n_seats)
    r_h = float(boat.rig.seats[0].oarlocks[0].oar.inboard)
    sim = TrunkSim(boat, peak_torque=1.0, catch=research.catch, coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    sim._torque = lambda slot, ang, state, t: 0.5 * max(float(np.interp(np.mod(t, T_CR06), t_F, F_hand, period=T_CR06)), 0.0) * r_h
    leg = body_field(boat, False, 1.0)                          # legs on her clock; the upper body is replaced
    # the hip on the same warp as the legs: the rower's hip joint at the warped model time
    m_leg, s_leg, c_leg = measured_channel("leg_disp_m")
    mm_leg, tau_leg, cm_leg = model_channel(boat, lambda j: j["hip"][0] - j["ankle"][0])
    wsp = warp(s_leg, c_leg, tau_leg, cm_leg); d1, d2 = wsp.derivative(1), wsp.derivative(2)
    P = float(boat.timing.period); rower = boat.crew[0].rower
    # the rower's hip over one model stroke, tabulated once and splined (periodic): solving the joint chain at every
    # call made the run hours long
    from scipy.interpolate import CubicSpline
    Tg = np.linspace(0.0, P, 1601)
    Hg = np.array([rower.joint_positions(float(x % P))["hip"] for x in Tg], float)
    Hg[-1] = Hg[0]
    hip_sp = [CubicSpline(Tg, Hg[:, j], bc_type="periodic") for j in range(3)]
    def hip_fn(t):
        real = t / P - m_leg; k = np.floor(real); s = real - k
        Tm = (P * (k + float(wsp(s)) + mm_leg)) % P; tp, tpp = float(d1(s)), float(d2(s))
        p0 = np.array([sp(Tm) for sp in hip_sp]); dp = np.array([sp(Tm, 1) for sp in hip_sp]); ddp = np.array([sp(Tm, 2) for sp in hip_sp])
        return p0, dp * tp, ddp * tp ** 2 + dp * tpp / P
    # the trunk reference: her back channel (the Fourier reference of rungs 3-4) on the rower's catch value
    S = HS.Stroke(); x_catch, _ = HT.setup(S); ang = TF.her_angle(S, x_catch)
    dist = {name: d for name, m, d, I in HT.upper_body()}
    sim.setup_trunk(leg, hip_fn, ang, dist, mode, **kw)
    sim.crew_field = sim.field
    return sim


def stroke_metrics(sim, run):
    times, states = run.last_time, run.last_states
    P = float(sim.boat.timing.period)
    grid = np.linspace(0.0, 1.0, 400, endpoint=False)
    v_meas = np.interp(grid * T_CR06, t_v, v_boat, period=T_CR06)
    frac = np.mod(times - times[0], P) / P; o = np.argsort(frac)
    v_m = np.interp(grid, frac[o], np.hypot(states[6], states[7])[o], period=1.0)
    rms = float(np.sqrt(np.mean(((v_m - v_m.mean()) - (v_meas - v_meas.mean())) ** 2)))
    return dict(speed=float(run.settled_speed()), rms=rms, catch_min=float(v_m[grid < 0.3].min()))


def run_case(label, mode, strokes, **kw):
    sim = build(mode, **kw)
    try:
        run = sim.run_strokes(strokes, surge_speed=4.191, dt=DT_RUN)
    finally:
        oardynamics.InertiaProfile.of = REAL_OF
    m = stroke_metrics(sim, run)
    if mode == "free":
        last_start = run.last_time[0]
        lg = [(t, e, fb) for t, e, fb in sim.log if t >= last_start]
        errs = np.degrees([e for _, e, _ in lg]); fbs = [abs(fb) for _, _, fb in lg]
        m.update(err_max=float(np.max(np.abs(errs))) if len(errs) else float("nan"), err_end=float(errs[-1]) if len(errs) else float("nan"),
                 fb_max=float(np.max(fbs)) if fbs else 0.0, err_max_all=float(np.degrees(np.max(np.abs([e for _, e, _ in sim.log])))))
    print("%-28s speed %.4f m/s | velocity rms vs her %.4f | catch min %.2f%s" % (label, m["speed"], m["rms"], m["catch_min"],
          (" | trunk error max %.4f / end %+.4f deg (all strokes %.3f), feedback %.1f N.m" % (m["err_max"], m["err_end"], m["err_max_all"], m["fb_max"])) if mode == "free" else ""), flush=True)
    return m, sim, run


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--strokes", type=int, default=12)
    ap.add_argument("--hull-blind", action="store_true", help="audit: solve the trunk without the hull's recoil in its own equation")
    ap.add_argument("--one-way", action="store_true", help="audit: the hull sees only the prescribed trunk (one-way, on the simulated hull)")
    a = ap.parse_args()
    out = dict(strokes=a.strokes, hull_blind=a.hull_blind, one_way=a.one_way, cases={}, recorded=RECORDED)
    # 1. prescribed trunk, and the feedforward torque along it on the simulated hull
    m, sim, run = run_case("prescribed trunk", "prescribed", a.strokes)
    out["cases"]["prescribed"] = m
    P = float(sim.boat.timing.period)
    last_start = float(run.last_time[0])
    lg = [(t, M) for t, M in sim.log if t >= last_start]                 # the last (settled) stroke
    ph = np.array([np.mod(t, P) / P for t, _ in lg]); tq = np.array([M for _, M in lg])
    o = np.argsort(ph); ph, tq = ph[o], tq[o]
    keep = np.concatenate([[True], np.diff(ph) > 1e-9]); ph, tq = ph[keep], tq[keep]
    tau_ff = lambda t: float(np.interp(np.mod(t, P) / P, ph, tq, period=1.0))      # noqa: E731
    out["feedforward_peak"] = float(np.max(tq))
    print("feedforward (rung-2 hip moment on the simulated hull): peak %.0f N.m" % np.max(tq), flush=True)
    cases = (("coupled, undisturbed", {}), ("coupled, 0.1 deg offset", dict(a0_offset=0.1)), ("coupled, 10 N.m bias", dict(bias=10.0)),
             ("bias, 45 ms delay", dict(bias=10.0, delay=0.045)), ("bias, 75 ms delay", dict(bias=10.0, delay=0.075)),
             ("bias, 100 ms delay", dict(bias=10.0, delay=0.100)), ("bias, 117 ms delay", dict(bias=10.0, delay=0.117)))
    if a.hull_blind or a.one_way:
        cases = tuple((lab, dict(kw, hull_blind=a.hull_blind, one_way=a.one_way)) for lab, kw in cases if "bias" in lab and ("117" in lab or "delay" not in lab or "100" in lab))
    for label, kw in cases:
        try:
            m, _, _ = run_case(label, "free", a.strokes, tau_ff=tau_ff, **kw)
        except (OverflowError, FloatingPointError, ValueError) as e:      # a diverging trunk can blow up the integrator
            m = dict(diverged=True, error=repr(e)[:200])
            print("%-28s DIVERGED (%s)" % (label, repr(e)[:120]), flush=True)
        out["cases"][label] = m
    print("recorded: prescribed 4.115 / 0.030; undisturbed 4.1173, 0.0297, catch 2.98, error 0.0017/-0.0010, 0 N.m; 10 N.m bias 4.1183, "
          "0.0299, 3.71/-3.43, 22.2 N.m; delays 45/75/100/117 ms: 3.80/3.35 23.0 0.0303; 3.88/3.21 23.7 0.0312; 4.17/2.89 27.9 0.0331; 45.6/-31.4 485 0.64")
    name = "coupled_trunk_cr06%s%s.json" % ("_hull_blind" if a.hull_blind else "", "_one_way" if a.one_way else "")
    json.dump(out, open(os.path.join(OUT, name), "w"), indent=1, default=float)
    print("written", os.path.join(OUT, name))


if __name__ == "__main__":
    main()
