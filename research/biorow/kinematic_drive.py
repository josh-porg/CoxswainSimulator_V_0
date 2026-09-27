"""The reference for sprint 1 #1: [BR24] driven by his measured kinematics.

    python research/biorow/kinematic_drive.py [--strokes 12] [--body his|model]

Everything the rower does is prescribed from his measurements -- the oar angle through the
whole stroke, the blade's depth from his vertical oar angle (so it enters and leaves the
water where his did), and the body re-timed onto his seat and trunk curves -- and the
model's blade and hull turn it into force, power and boat speed. This is [CR06]'s own
architecture (body and oar prescribed, forces as outputs), so no oar balance, reflected
inertia or effort split is involved.

If his boat speed, handle power and speed fluctuation come out right, the blade and hull are
right and the remaining gap (SOURCES sec. 158-161) is the split of his effort between body and
handle, which #1's constraint has to produce. If they do not, the gap is in the blade or hull.

Handle power is an output: per oar, torque = I_oar * phi_ddot - l * w * F_n (the oar alone,
since the body's inertia is carried by the prescribed body; w the wetted fraction), times
phi_dot.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import like_for_like as L                                        # noqa: E402
import measured_body as MB                                       # noqa: E402

from coxswain.core.state import STATE_SIZE, State               # noqa: E402
from coxswain.sim.control import Coxswain                        # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402

T = 60.0 / L.RATE


def his_traces():
    D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
    n = len(D)
    t = np.arange(n) * T / n
    k = int(np.argmin(D["A1"]))
    ph = np.mod(t - t[k], T)
    o = np.argsort(ph)
    ph = ph[o]
    per = lambda y: CubicSpline(np.append(ph, ph[0] + T), np.append(y[o], y[o][0]), bc_type="periodic")
    angle = per(np.radians(-0.5 * (D["A1"] + D["A2"])))       # model sign: catch positive
    vert = per(np.radians(0.5 * (D["V1"] + D["V2"])))          # blade up positive
    speed = per(D["Vs"])
    power = per(D["H1"] * D["Vh1"] + D["H2"] * D["Vh2"])       # both oars, W
    return angle, vert, speed, power


class KinematicOarSim(DynamicOarSimulator):
    """Oar angle prescribed from his trace; blade force scaled by his measured wetted width."""

    def setup(self, angle, vert, lever, width, zero=0.0):
        self._ang, self._vert, self._lever, self._width = angle, vert, lever, width
        self._zero = zero                      # z_0, m: blade centre height at V = 0 (BladeDepth.zero_offset)
        self.work = []                         # (t, handle torque * rate) samples, per oar

    def _tau(self, t):
        return float(np.mod(t - self._stroke_start, T))

    def _wetted(self, t):
        """Submerged fraction of the squared blade's width. The blade is squared only while the
        oar turns in the drive direction (rate < 0); feathered, it carries no load. At z_0 = 0
        his feathered blade clears the water by its own half-width (V >= 3.1 deg on the
        recovery), so BioRow's zero is consistent; deeper z_0 would dip a squared blade."""
        if float(self._ang(self._tau(t), 1)) >= 0.0:
            return 0.0
        centre = self._lever * np.sin(float(self._vert(self._tau(t)))) + self._zero
        bottom = centre - 0.5 * self._width
        return float(np.clip(-bottom / self._width, 0.0, 1.0))

    def _catch(self, t0, y, index):
        y = np.array(y, dtype=float)
        n = self.n_oar_states
        y[STATE_SIZE:STATE_SIZE + n] = float(self._ang(0.0))
        y[STATE_SIZE + n:STATE_SIZE + 2 * n] = float(self._ang(0.0, 1))
        self._in_air = None
        return y

    def _oar_rates(self, t, hull, angles, rates):
        tau = self._tau(t)
        n = self.n_oar_states
        return np.full(n, float(self._ang(tau, 1))), np.full(n, float(self._ang(tau, 2)))

    def blade(self, t, state, slot, lock, angle, rate):
        """``(F_n, F_t)`` on one blade, times the wetted fraction: tier 1 (slip, no tangential
        load) or tier 2 (lift and drag on the angle of attack), per ``blade_law``."""
        if self.his_force is not None:          # his measured blade load, no law at all
            return float(self.his_force(self._tau(t))), 0.0
        wet = self._wetted(t)
        if wet <= 0.0:
            return 0.0, 0.0
        if self.blade_law == "slip":
            speed = self._lock_speed_on_normal(state, lock, angle)
            return wet * float(self._oars[slot].blade.normal_force(angle, rate, speed)), 0.0
        velocity = self._lock_velocity(state, lock)
        f_n, f_t = self._liftdrag[slot].loads(angle, rate, velocity[:2], int(lock.side))
        return wet * f_n, (wet * f_t if self.tangential else 0.0)

    #: tier 2's load along the shaft; [BR24] measures only the normal load, so it can be
    #: switched off to see what the normal load alone does
    tangential = True

    #: his measured blade normal load against time from the catch (blade_law_check.py), in
    #: place of any blade law: the hull then answers to his force and the model's body alone
    his_force = None

    def _oar_loads(self, t, state):
        force = np.zeros(3)
        moment = np.zeros(3)
        tau = self._tau(t)
        angle, rate = float(self._ang(tau)), float(self._ang(tau, 1))
        for slot, seat in enumerate(self._seats):
            oar = self._oars[slot]
            for lock in self.boat.rig.seats[seat].oarlocks:
                side = int(lock.side)
                normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
                axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
                f_n, f_t = self.blade(t, state, slot, lock, angle, rate)
                load = f_n * normal + f_t * axis         # tangential: along the shaft
                point = np.asarray(lock.position, dtype=float) + oar.outboard * axis
                force += load
                moment += np.cross(point, load)
        return force, moment


def run(strokes, body, c2_scale=1.0, zero=0.0, blade_law="slip", amplitudes=None, tangential=True, lever=None):
    boat = L.build("arc")
    boat.power_scales = np.ones(boat.n_seats)
    angle, vert, speed, power = his_traces()
    field = None if body == "model" else MB.body_field(boat, back=True)[0]
    sim = KinematicOarSim(boat, peak_torque=1.0, catch="rest", blade_law="slip" if blade_law == "his" else blade_law,
                          coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    lock = boat.rig.seats[0].oarlocks[0]
    if c2_scale != 1.0:
        import dataclasses
        sim._oars = [dataclasses.replace(o, blade=dataclasses.replace(o.blade, c2=o.blade.c2 * c2_scale))
                     for o in sim._oars]
    if amplitudes is not None:                   # tier 2 with (A_l, A_d) fitted, e.g. to his blade
        import dataclasses
        sim._liftdrag = [dataclasses.replace(b, lift_amplitude=amplitudes[0], drag_amplitude=amplitudes[1])
                         for b in sim._liftdrag]
    sim.tangential = tangential
    if blade_law == "his":
        import blade_law_check as BL
        g, f_his, _w, _d, fin, _o = BL.main(lever, verbose="alpha")   # load = torque / lever
        f = np.where(fin, f_his, 0.0)            # his load through the drive, none on the recovery
        sim.his_force = CubicSpline(np.append(g, T), np.append(f, f[0]), bc_type="periodic")
    centre = float(sim._oars[0].outboard)       # wetted fraction: the blade's centre, always
    if lever is not None:                        # centre of pressure (chosen): slip and force point
        import dataclasses
        sim._oars = [dataclasses.replace(o, outboard=lever, blade=dataclasses.replace(o.blade, outboard=lever))
                     for o in sim._oars]
        sim._liftdrag = [dataclasses.replace(b, outboard=lever) for b in sim._liftdrag]
    oar = sim._oars[0]
    sim.setup(angle, vert, centre, float(lock.oar.blade_area) / float(lock.oar.blade_length), zero)
    if field is not None:
        sim.crew_field = field
    out = sim.run_strokes(int(strokes), surge_speed=4.6)
    t, y = out.last_time, out.last_states
    v = np.hypot(y[6], y[7])
    ph = t - t[0]
    # handle power from the oar alone. On the angle coordinate (catch positive, so the rate is
    # negative through the drive) the blade's torque on each oar is +l * w * F_n, resisting the
    # stroke; the rower's torque is Q = I phi_ddot - l w F_n per oar and the power is Q * phi_dot.
    I = float(lock.oar.inertia_about_lock)
    l = float(oar.outboard)
    p = []
    for i in range(len(t)):
        tau = float(np.mod(ph[i], T))
        a, r, acc = float(angle(tau)), float(angle(tau, 1)), float(angle(tau, 2))
        state = State.from_vector(y[:STATE_SIZE, i])
        q = 0.0
        for lk in boat.rig.seats[0].oarlocks:
            f_n, _f_t = sim.blade(t[i], state, 0, lk, a, r)   # the shaft load turns nothing
            q += I * acc - l * f_n
        p.append(q * r)
    p = np.array(p)
    dt = np.gradient(ph)
    mean_v = float(np.average(v, weights=dt))
    return dict(speed=mean_v, ivv=float(np.ptp(v) / mean_v), power=float(np.average(p, weights=dt)),
                v_min=float(v.min()), t_min=float(ph[np.argmin(v)]), v_max=float(v.max()),
                his_power=float(np.mean(power(np.linspace(0, T, 400, endpoint=False)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=12)
    ap.add_argument("--body", default="his", choices=["his", "model"])
    ap.add_argument("--c2-scale", default="1.0", help="comma list; multiplies the research scull C2")
    ap.add_argument("--blade", default="slip", choices=["slip", "liftdrag", "his"],
                    help="tier 1 slip law (C2 scaled by --c2-scale) or tier 2 lift/drag")
    ap.add_argument("--amplitudes", default=None,
                    help="A_l,A_d for --blade liftdrag (default [CG07]'s 1.25,2.07); "
                         "blade_law_check.py fits 0.76,3.39 to [BR24] at the blade centre")
    ap.add_argument("--lever", type=float, default=None,
                    help="centre of pressure, m from the pin (chosen; default the blade centre)")
    ap.add_argument("--no-tangential", action="store_true", help="tier 2 without its shaft load")
    ap.add_argument("--zero", default="0.0",
                    help="comma list, m: blade centre height at V = 0, BladeDepth.zero_offset (chosen; 0 = BioRow convention)")
    a = ap.parse_args()
    for sc, z in [(float(x), float(y)) for x in a.c2_scale.split(",") for y in a.zero.split(",")]:
        amp = tuple(float(x) for x in a.amplitudes.split(",")) if a.amplitudes else None
        r = run(a.strokes, a.body, sc, z, a.blade, amp, not a.no_tangential, a.lever)
        print(("%-8s" % (a.blade + ("" if not amp else " %.2f/%.2f" % amp) + (" n-only" if a.no_tangential else "") + ("" if a.lever is None else " l %.2f" % a.lever))) + "C2 x%.2f zero %+.2f " % (sc, z) + "kinematic drive, %-5s body speed %.3f  IVV %.1f%%  v min %.2f at %.3f s  v max %.2f  handle power %.0f W"
              % (a.body, r["speed"], 100 * r["ivv"], r["v_min"], r["t_min"], r["v_max"], r["power"]), flush=True)
    print("BR24                                        speed 4.641  IVV 49.1%%  v min 3.46 at 0.111 s  v max 5.74  handle power %.0f W"
          % r["his_power"])


if __name__ == "__main__":
    main()
