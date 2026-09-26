"""Where in the stroke does the model's extra speed fluctuation come from?

    python research/biorow/shape_br24.py [--variant arc] [--strokes 16]

Runs a like_for_like variant at his power and compares, against [BR24], the
within-stroke boat speed and handle force, both timed from the catch: the speed
minimum and maximum (value and time), the speed at fixed points of the cycle, the
handle-force peak (value and time), and how long the force stays above 10% of peak.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import like_for_like as L                                        # noqa: E402

from coxswain import physics                                     # noqa: E402
from coxswain.core.state import STATE_SIZE, State                # noqa: E402
from coxswain.sim.control import Coxswain                        # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402

T = 60.0 / L.RATE


def his():
    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    n = len(D)
    t = np.arange(n) * T / n
    k = int(np.argmin(D["A1"]))
    ph = np.mod(t - t[k], T)
    o = np.argsort(ph)
    return ph[o], D["Vs"][o], (0.5 * (D["H1"] + D["H2"]))[o], (0.5 * (D["A1"] + D["A2"]))[o]


def model(variant, strokes, force="model"):
    """force="model": the research pull shape at the torque for his power.
    force="his": his measured handle-force shape (clipped at zero, as the default
    clips), scaled until the run's handle power is his 432 W."""
    boat = L.build(variant)
    boat.power_scales = np.ones(boat.n_seats)
    catch = physics.resolve("research").catch
    r_h = float(boat.rig.seats[0].oarlocks[0].oar.inboard)

    def make(scale=None):
        torque = DynamicOarSimulator.torque_for_power(boat, L.POWER, catch=catch, start=4.6)
        sim = DynamicOarSimulator(boat, peak_torque=torque, catch=catch,
                                  coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
        if scale is not None:
            hp, _hv, hF, _hA = his()
            sim._torque = lambda slot, ang, state, t: scale * max(
                float(np.interp(np.mod(t - sim._stroke_start, T), hp, hF, period=T)), 0.0) * r_h
        return sim, sim.run_strokes(int(strokes), surge_speed=4.6)

    if force == "model":
        sim, run = make()
    else:
        scale = 1.0
        for _ in range(4):
            sim, run = make(scale)
            p = run.settled_power()
            print("  his force x %.3f -> %.1f W" % (scale, p), flush=True)
            if abs(p / L.POWER - 1) < 0.005:
                break
            scale *= (L.POWER / p) ** (2.0 / 3.0)
    t, y = run.last_time, run.last_states
    ph = t - t[0]
    v = np.hypot(y[6], y[7])
    ang = np.degrees(y[STATE_SIZE])
    F = np.array([sim._torque(0, float(y[STATE_SIZE, i]), State.from_vector(y[:STATE_SIZE, i]), float(t[i]))
                  for i in range(len(t))]) / r_h
    # the model's stroke starts at the catch; confirm from the oar angle
    return ph, v, F, -ang, run


def summary(label, ph, v, F, A):
    i_min, i_max = int(np.argmin(v)), int(np.argmax(v))
    k_pk = int(np.argmax(F))
    on = F > 0.1 * F.max()
    dt = np.gradient(ph)
    print("%-6s mean %.3f  IVV %.1f%%  v min %.2f at %.3f s  v max %.2f at %.3f s  "
          "F peak %.0f N at %.3f s  force>10%% for %.3f s  catch angle %.1f  finish %.1f"
          % (label, np.average(v, weights=dt), 100 * np.ptp(v) / np.average(v, weights=dt),
             v[i_min], ph[i_min], v[i_max], ph[i_max], F[k_pk], ph[k_pk], float(np.sum(dt[on])),
             A.min(), A.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="arc")
    ap.add_argument("--strokes", type=int, default=16)
    ap.add_argument("--force", default="model", choices=["model", "his"])
    a = ap.parse_args()
    hp, hv, hF, hA = his()
    mp, mv, mF, mA, run = model(a.variant, a.strokes, a.force)
    summary("BR24", hp, hv, hF, hA)
    summary("model", mp, mv, mF, mA)
    print("\n  t from catch   speed his / model (dev. from mean)      force his / model (N)")
    hmean = np.mean(hv)
    mmean = np.average(mv, weights=np.gradient(mp))
    for q in np.arange(0.0, T, 0.1):
        print("  %.2f s         %+.2f / %+.2f                          %4.0f / %4.0f"
              % (q, np.interp(q, hp, hv, period=T) - hmean, np.interp(q, mp, mv) - mmean,
                 np.interp(q, hp, hF, period=T), np.interp(q, mp, mF)))


if __name__ == "__main__":
    main()
