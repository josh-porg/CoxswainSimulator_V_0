"""Drive the model's body on [BR24]'s measured seat and trunk time-laws.

    python research/biorow/measured_body.py [--strokes 16]

shape_br24.py found that the model's extra speed fluctuation comes from the crew,
not the pull: early in the drive the model's seat reaches 1.5 m/s and its trunk opens
with the legs, while his seat holds about 1.0 m/s and his trunk waits (legs first);
early in the recovery the model's body swings forward at once while his is slower.
Driving the model with his force shape changed IVV by only 1.5 points.

This keeps the model's postures and re-times them, as the [CR06] one-athlete study did
(TRACKING, finish defect; self_consistent_body_cr06):

  leg  channel   hip - ankle        monotone time-warp onto his seat position Ls
  back channel   shoulder - hip     onto his trunk channel Lt (shoulder height, sec. 156),
                                    and scaled by K = his travel / the model's

The whole body follows the leg channel; the head, trunk above the lower trunk and the
arms take their motion relative to the lower trunk from the back channel. Each
channel's minimum is placed at his time on the catch clock. Nothing is fitted.

Builds, all at his 432 W through the research pull (and, last, his force shape):
  model body                         (like_for_like "arc")
  legs on his clock
  legs + back on his clock and travel
  legs + back, and his force shape
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import like_for_like as L                                        # noqa: E402
import shape_br24 as S                                           # noqa: E402

from coxswain import physics                                     # noqa: E402
from coxswain.sim.control import Coxswain                        # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402

T = 60.0 / L.RATE
WARP_LAM = 1e-4                        # as the [CR06] study
UPPER, REF = [0, 1, 2, 4, 5, 6, 7], 3  # SEGMENT_ORDER: upper body rows, lower trunk
D = np.genfromtxt(L.DATA, delimiter=",", names=True)
i_catch = int(np.argmin(D["A1"][:-1]))


#: Whose time-law drives the body: "br24" (his own curves) or "cr06" (the [CR06] athlete's,
#: carried onto his catch -> finish -> catch clock, with his own travel). The second is the
#: transfer test: does another on-water sculler's timing do what his own does?
TIMELAW = "br24"


def transferred_channel(name):
    import timelaw_compare as TC
    his = TC.curves(TC.athlete_br24())
    her = TC.curves(TC.athlete_cr06())
    ch = "leg" if name == "Ls" else "back"
    a = TC.athlete_br24()
    drive = (a["t_finish"] - a["t_catch"]) % a["T"]
    s = np.linspace(0.0, 1.0, 4001, endpoint=False)
    t = s * T                                              # time since his catch
    g = np.where(t < drive, 0.5 * t / drive, 0.5 + 0.5 * (t - drive) / (T - drive))
    c = np.interp(g, TC.G, her[ch]["split"], period=1.0)
    k = int(np.argmin(c))
    m = s[k]                                               # phase of the minimum, catch clock
    c = np.roll(c, -k)
    y = D[name][:-1]
    return m, s, (c - c.min()) / np.ptp(c), float(np.ptp(y))


def measured_channel(name):
    """His curve on the catch clock: (phase of its minimum, phase grid since it, normalised curve)."""
    if TIMELAW == "cr06":
        return transferred_channel(name)
    y = D[name][:-1]
    n = len(y)
    t = np.mod(np.arange(n) * T / n - np.arange(n)[i_catch] * T / n, T)
    o = np.argsort(t)
    t, y = t[o], y[o]
    sp = CubicSpline(np.append(t, t[0] + T), np.append(y, y[0]), bc_type="periodic")
    fine = np.linspace(0.0, T, 4000, endpoint=False)
    m = float(fine[np.argmin(sp(fine))]) / T
    s = np.linspace(0.0, 1.0, 4001, endpoint=False)
    c = sp(np.mod(m + s, 1.0) * T)
    return m, s, (c - c.min()) / np.ptp(c), float(np.ptp(c))


def model_channel(boat, fn):
    P = float(boat.timing.period)
    rower = boat.crew[0].rower
    t = np.linspace(0.0, P, 1601, endpoint=False)
    c = np.array([fn(rower.joint_positions(float(x))) for x in t])
    k = int(np.argmin(c))
    tau = np.mod(t - t[k], P) / P
    o = np.argsort(tau)
    return t[k] / P, tau[o], (c[o] - c.min()) / np.ptp(c), float(np.ptp(c))


def warp(s_meas, c_meas, tau_grid, c_model):
    k_max_m, k_max_c = int(np.argmax(c_model)), int(np.argmax(c_meas))
    drive = np.maximum.accumulate(c_model[:k_max_m + 1]) + np.arange(k_max_m + 1) * 1e-9
    rec = np.minimum.accumulate(c_model[k_max_m:])[::-1] + np.arange(c_model.size - k_max_m) * 1e-9
    tau = np.where(s_meas <= s_meas[k_max_c], np.interp(c_meas, drive, tau_grid[:k_max_m + 1]),
                   np.interp(c_meas, rec, tau_grid[k_max_m:][::-1]))
    tau = np.maximum.accumulate(tau)
    return make_smoothing_spline(np.concatenate([s_meas - 1, s_meas, s_meas + 1]),
                                 np.concatenate([tau - 1, tau, tau + 1]), lam=WARP_LAM)


def channel_field(boat, spline, meas_min, model_min):
    P = float(boat.timing.period)
    d1, d2 = spline.derivative(1), spline.derivative(2)

    def field(t):
        real = t / P - meas_min
        k = np.floor(real)
        s = real - k
        tm = P * (k + float(spline(s)) + model_min)
        tp, tpp = float(d1(s)), float(d2(s))
        mass, pos, vel, acc = boat.crew_field(tm, exact=True)
        vel = np.array(vel, float)
        return (mass, np.array(pos, float), vel * tp,
                np.array(acc, float) * tp ** 2 + vel * tpp / P)
    return field


def body_field(boat, back):
    m_l, s_l, c_l, _ = measured_channel("Ls")
    mm_l, tau_l, cm_l, _ = model_channel(boat, lambda j: j["hip"][0] - j["ankle"][0])
    leg = channel_field(boat, warp(s_l, c_l, tau_l, cm_l), m_l, mm_l)
    if not back:
        return lambda t, exact=False: leg(t), None
    m_b, s_b, c_b, his_travel = measured_channel("Lt")
    mm_b, tau_b, cm_b, model_travel = model_channel(boat, lambda j: j["shoulder"][0] - j["hip"][0])
    bk = channel_field(boat, warp(s_b, c_b, tau_b, cm_b), m_b, mm_b)
    K = his_travel / model_travel

    def field(t, exact=False):
        mass, pos, vel, acc = leg(t)
        _, pb, vb, ab = bk(t)
        for off in range(0, vel.shape[0] - vel.shape[0] % 12, 12):
            idx = [off + i for i in UPPER]
            r = off + REF
            pos[idx] = pos[r] + K * (pb[idx] - pb[r])
            vel[idx] = vel[r] + K * (vb[idx] - vb[r])
            acc[idx] = acc[r] + K * (ab[idx] - ab[r])
        return mass, pos, vel, acc
    return field, K


def run(body, force, strokes, catch=None, blade_law="slip", ld_scale=1.0, added_mass="none"):
    boat = L.build("arc")
    boat.power_scales = np.ones(boat.n_seats)
    if added_mass != "none":
        if blade_law != "liftdrag":
            raise ValueError("added mass is studied with the lift-drag blade only")
        boat.scull_c2 = None      # the slip coefficient is unused by the lift-drag law
    catch = catch or physics.resolve("research").catch
    r_h = float(boat.rig.seats[0].oarlocks[0].oar.inboard)
    hp, _hv, hF, _hA = S.his()
    field, K = (None, None) if body == "model" else body_field(boat, back=(body == "legs+back"))

    def make(scale=None, torque=None):
        if torque is None:
            torque = DynamicOarSimulator.torque_for_power(boat, L.POWER, catch=catch, start=4.6,
                                                          blade_law=blade_law)
        sim = DynamicOarSimulator(boat, peak_torque=torque, catch=catch, blade_law=blade_law,
                                  blade_added_mass=added_mass,
                                  coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
        if field is not None:
            sim.crew_field = field
        if blade_law == "liftdrag" and ld_scale != 1.0:
            # one scale on both tier 2 amplitudes: a study, never a profile change
            import dataclasses
            sim._liftdrag = [dataclasses.replace(b, lift_amplitude=b.lift_amplitude * ld_scale,
                                                 drag_amplitude=b.drag_amplitude * ld_scale)
                             for b in sim._liftdrag]
        if scale is not None:
            sim._torque = lambda slot, ang, state, t: scale * max(
                float(np.interp(np.mod(t - sim._stroke_start, T), hp, hF, period=T)), 0.0) * r_h
        return sim, sim.run_strokes(int(strokes), surge_speed=4.6)

    # the re-timed body changes the power a given pull delivers, so rematch on the run
    scale = 1.0 if force == "his" else None
    sim, out = make(scale)
    for _ in range(4):
        p = out.settled_power()
        if abs(p / L.POWER - 1) < 0.005:
            break
        f = (L.POWER / p) ** (2.0 / 3.0)
        if force == "his":
            scale *= f
            sim, out = make(scale)
        else:
            sim, out = make(torque=sim.peak_torque * f)
    t, y = out.last_time, out.last_states
    v = np.hypot(y[6], y[7])
    ph = t - t[0]
    i_min = int(np.argmin(v))
    return dict(body=body, force=force, K=K, speed=out.settled_speed(), power=out.settled_power(),
                ivv=float(np.mean([s.surge_swing for s in out.strokes[-4:]])),
                v_min=float(v.min()), t_min=float(ph[i_min]), v_max=float(v.max()), ph=ph, v=v, sim=sim, boat=boat, field=field, states=y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=16)
    ap.add_argument("--builds", default="model:model,legs:model,legs+back:model,legs+back:his")
    ap.add_argument("--catch", default=None, help="override the research profile's catch rule")
    ap.add_argument("--blade-law", default="slip", help="slip (tier 1) or liftdrag (tier 2)")
    ap.add_argument("--ld-scale", type=float, default=1.0, help="scale on both tier 2 amplitudes")
    ap.add_argument("--added-mass", default="none", help="none or patton (lift-drag only)")
    ap.add_argument("--timelaw", default="br24", choices=["br24", "cr06"],
                    help="whose seat and trunk timing drives the body (his travel either way)")
    a = ap.parse_args()
    global TIMELAW
    TIMELAW = a.timelaw
    hp, hv, _hF, _hA = S.his()
    print("BR24                         speed 4.641  IVV 49.1%%  v min %.2f at %.3f s  v max %.2f"
          % (hv.min(), hp[np.argmin(hv)], hv.max()))
    for b in a.builds.split(","):
        body, force = b.split(":")
        r = run(body, force, a.strokes, a.catch, a.blade_law, a.ld_scale, a.added_mass)
        print("%-10s body, %-5s force  speed %.3f  IVV %.1f%%  v min %.2f at %.3f s  v max %.2f  "
              "power %.0f W%s" % (body, force, r["speed"], 100 * r["ivv"], r["v_min"], r["t_min"],
                                  r["v_max"], r["power"], "  K %.3f" % r["K"] if r["K"] else ""), flush=True)


if __name__ == "__main__":
    main()
