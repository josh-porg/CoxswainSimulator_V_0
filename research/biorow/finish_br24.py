"""The finish fix, on a second athlete: [BR24]'s measured handle force and oar angle.

    python research/biorow/finish_br24.py [--strokes 12]

TRACKING's finish defect was validated on one athlete, [CR06]'s sculler: with her
measured handle force driving the oar (unclipped, so her push at the finish is
present), [CR06]'s release rule -- the blade comes out when its normal velocity
returns to zero -- fires on every stroke, and the blade-out oar turns round on her
time. Promotion was held because the push was one athlete's measurement.

[BR24] measures the same things on a different, elite, athlete at 32.4 spm: oar angle
and handle force on both oars through the release. This runs the same study on him:

  - the research profile, as his boat (like_for_like.build("arc"): his body, hull,
    oars, rate and arc);
  - the oar balance is the oar alone, because the measured handle force already is
    what the rower applies (as in the [CR06] study);
  - his handle force per oar, unclipped, as the handle torque;
  - three runs: the committed default (blade out at the finish angle, force clipped
    at zero, as the research profile runs), the default with his push unclipped, and
    the fix (finish angle out of reach, blade out at zero normal velocity, blade-out
    oar under his force, held at turn-round).

Scored against his release (handle force crossing zero at the finish) and his
turn-round (the oar's extreme angle). Only derived numbers are printed.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import like_for_like as L                                        # noqa: E402

from coxswain import physics                                     # noqa: E402
from coxswain.core import integrators                            # noqa: E402
from coxswain.core.state import STATE_SIZE, State                # noqa: E402
from coxswain.crew import oardynamics                            # noqa: E402
from coxswain.sim.control import Coxswain                        # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402

REAL_OF = oardynamics.InertiaProfile.__dict__["of"]


def oar_only(boat, seat=0, **kwargs):
    return float(boat.rig.seats[seat].oarlocks[0].oar.inertia_about_lock)


# ---- his stroke, re-timed so t = 0 is his catch
D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
N = len(D)
T = 60.0 / L.RATE
t_file = np.arange(N) * T / N
i_catch = int(np.argmin(D["A1"]))
t_his = np.mod(t_file - t_file[i_catch], T)
o = np.argsort(t_his)
t_his = t_his[o]
F_oar = 0.5 * (D["H1"] + D["H2"])[o]                 # per oar, N
A_his = 0.5 * (D["A1"] + D["A2"])[o]                 # deg, his convention: catch negative


def his_events():
    # release: handle force falls through zero after its peak
    k_pk = int(np.argmax(F_oar))
    k = k_pk + int(np.argmax(F_oar[k_pk:] <= 0.0))
    f0, f1 = F_oar[k - 1], F_oar[k]
    t_rel = t_his[k - 1] + (t_his[k] - t_his[k - 1]) * f0 / (f0 - f1)
    a_rel = float(np.interp(t_rel, t_his, A_his))
    rate = np.gradient(A_his, t_his)
    k_turn = int(np.argmax(A_his))
    return dict(release_t=t_rel, release_deg=a_rel, release_rate=float(np.interp(t_rel, t_his, rate)),
                turn_t=float(t_his[k_turn]), turn_deg=float(A_his[k_turn]))


class FinishFixSim(DynamicOarSimulator):
    """The [CR06] study's slip release and blade-out phase (TRACKING, finish defect)."""

    released = turned = driving = events = None
    fix = True

    def _catch(self, t0, y, index):
        n = self.n_oar_states
        self.released = np.zeros(n, bool)
        self.turned = np.zeros(n, bool)
        self.driving = np.zeros(n, bool)
        if self.events is None:
            self.events = []
        return super()._catch(t0, y, index)

    def _oar_loads(self, t, state):
        saved = self._in_air
        if self.fix and self.released is not None and self.released.any():
            self._in_air = (np.zeros(self.n_oar_states, bool) if saved is None else saved.copy()) | self.released
        try:
            return super()._oar_loads(t, state)
        finally:
            self._in_air = saved

    def _oar_rates(self, t, hull, angles, rates):
        ar, rr = super()._oar_rates(t, hull, angles, rates)
        if not self.fix or self.released is None:
            return ar, rr
        state = State.from_vector(hull)
        for slot in range(self.n_oar_states):
            if self.released[slot] and not self.turned[slot]:
                ar[slot] = rates[slot]
                rr[slot] = -self._torque(slot, float(angles[slot]), state, t) / self.I_LOCK
            elif self.released[slot] and self.turned[slot]:
                ar[slot] = 0.0
                rr[slot] = 0.0
        return ar, rr

    def _integrate_sweep_catch(self, t_span, y0, dt):
        if not self.fix:
            return super()._integrate_sweep_catch(t_span, y0, dt)
        t_start, t_end = float(t_span[0]), float(t_span[1])
        n_steps = int(np.ceil((t_end - t_start) / dt))
        n = self.n_oar_states
        times = np.empty(n_steps + 1)
        states = np.empty((len(y0), n_steps + 1))
        air = np.zeros((n, n_steps + 1), bool)
        t, y = t_start, np.array(y0, float)
        if self._in_air is None:
            self._in_air = np.zeros(n, bool)
        times[0], states[:, 0], air[:, 0] = t, y, self._in_air
        for i in range(n_steps):
            step = min(dt, t_end - t)
            y = integrators.rk4_step(self.derivative, t, y, step)
            t += step
            state = State.from_vector(y[:STATE_SIZE])
            if self._in_air.any():
                angle, rate = self._sweep_pose(t - self._stroke_start)
                for slot in np.flatnonzero(self._in_air):
                    y[STATE_SIZE + slot] = angle
                    y[STATE_SIZE + n + slot] = rate
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
                        self.events.append(("release", slot, float(t - self._stroke_start), angle, rate))
                elif not self.turned[slot] and rate >= 0.0:
                    self.turned[slot] = True
                    self.events.append(("turn", slot, float(t - self._stroke_start), angle, rate))
                    y[STATE_SIZE + n + slot] = 0.0
            times[i + 1], states[:, i + 1] = t, y
            air[:, i + 1] = self._in_air
        self._air_mask = air
        return times, states


def run(mode, strokes, c2_scale=1.0):
    research = physics.resolve("research")
    oardynamics.InertiaProfile.of = staticmethod(oar_only)
    try:
        boat = L.build("arc")
        boat.power_scales = np.ones(boat.n_seats)
        lock = boat.rig.seats[0].oarlocks[0]
        r_h = float(lock.oar.inboard)
        sim = FinishFixSim(boat, peak_torque=1.0, catch=research.catch,
                           coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
        sim.I_LOCK = float(lock.oar.inertia_about_lock)
        sim.fix = mode == "fix"
        clip = mode == "default"
        if sim.fix:
            sim._oars = [dataclasses.replace(o, finish_angle=np.radians(-80.0)) for o in sim._oars]
        if c2_scale != 1.0:
            # one-parameter study, as [CR06]'s own C2 fit: never a change to the profile
            sim._oars = [dataclasses.replace(o, blade=dataclasses.replace(o.blade, c2=o.blade.c2 * c2_scale))
                         for o in sim._oars]

        def torque(slot, ang, state, t):
            F = float(np.interp(np.mod(t, T), t_his, F_oar, period=T))
            return (max(F, 0.0) if clip else F) * r_h
        sim._torque = torque
        out = sim.run_strokes(int(strokes), surge_speed=4.6, dt=integrators.estimate_step(T) / 4.0)
    finally:
        oardynamics.InertiaProfile.of = REAL_OF
    t, y = out.last_time, out.last_states
    ph = np.mod(t - t[0], T)
    ang = -np.degrees(y[STATE_SIZE])                 # model finish is negative; his is positive
    oo = np.argsort(ph)
    his = np.interp(ph, t_his, A_his, period=T)
    win = (ph > 0.75 * T * 0.5) & (ph < 0.75 * T)
    res = dict(mode=mode, speed=out.settled_speed(), power=out.settled_power(),
               ivv=float(np.mean([s.surge_swing for s in out.strokes[-4:]])),
               angle_rms_late=float(np.sqrt(np.mean((ang - his)[win] ** 2))),
               angle_max=float(ang.max()), t_angle_max=float(ph[np.argmax(ang)]))
    if sim.fix:
        rel = [e for e in sim.events if e[0] == "release"]
        turn = [e for e in sim.events if e[0] == "turn"]
        res.update(n_release=len(rel), n_turn=len(turn))
        if rel:
            res.update(release_t=rel[-1][2], release_deg=-np.degrees(rel[-1][3]), release_rate=-np.degrees(rel[-1][4]))
        if turn:
            res.update(turn_t=turn[-1][2], turn_deg=-np.degrees(turn[-1][3]))
    res["lead_deg"] = {tq: float(np.interp(tq, ph[oo], ang[oo]) - np.interp(tq, t_his, A_his, period=T))
                       for tq in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=12)
    ap.add_argument("--modes", default="default,unclipped,fix")
    ap.add_argument("--c2-scale", default="1.0", help="comma list; multiplies the research C2")
    a = ap.parse_args()
    h = his_events()
    print("BR24: release %.3f s at %.1f deg (%.0f deg/s); turn-round %.3f s at %.1f deg; "
          "force minimum after the finish %.1f N per oar (%.1f%% of peak)"
          % (h["release_t"], h["release_deg"], h["release_rate"], h["turn_t"], h["turn_deg"],
             F_oar[np.argmax(F_oar):].min(), 100 * F_oar[np.argmax(F_oar):].min() / F_oar.max()), flush=True)
    for mode, c2s in [(m, float(c)) for m in a.modes.split(",") for c in a.c2_scale.split(",")]:
        r = run(mode, a.strokes, c2s)
        line = ("%-9s C2 x%.2f " % (mode, c2s) + "%-0s speed %.3f  IVV %.1f%%  power %.0f W  oar max %.1f deg at %.3f s  late-drive angle rms %.1f deg"
                % ("", r["speed"], 100 * r["ivv"], r["power"], r["angle_max"], r["t_angle_max"], r["angle_rms_late"]))
        if "n_release" in r:
            line += ("\n          releases %d, turns %d; release %s; turn %s"
                     % (r["n_release"], r["n_turn"],
                        "%.3f s at %.1f deg (%.0f deg/s)" % (r["release_t"], r["release_deg"], r["release_rate"]) if "release_t" in r else "none",
                        "%.3f s at %.1f deg" % (r["turn_t"], r["turn_deg"]) if "turn_t" in r else "none"))
        line += "\n          model - his angle: " + "  ".join("%.1fs %+.1f" % (k, v) for k, v in r["lead_deg"].items())
        print(line, flush=True)


if __name__ == "__main__":
    main()
