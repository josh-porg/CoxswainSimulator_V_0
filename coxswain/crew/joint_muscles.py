r"""The body's actuators and its lower control layers, from sourced population data.

This is the first layer of the multi-clock rower (docs/NEUROMUSCULAR_CONTROL_REVIEW.md §12; SOURCES §207). Joint-level
muscle torque generators carry Hill's force-length and force-velocity relations to the joint (Anderson et al. 2007;
Frey-Law et al. 2012). Activation dynamics turn neural excitation into activation (Zajac 1989; Winters 1990; Thelen
2003). Two lower control layers, measured on the human trunk (Moorhouse & Granata 2007), are scaled by effort:

* the **intrinsic layer**, which acts with no delay;
* the **spinal reflex layer**, which acts at 40 ms.

Nothing here is fitted to a rower. Every number is a population mean from its source, and the data files are in
``data/literature``.

Joint torque generator (Anderson, Madigan & Nussbaum 2007, eqs 1, 8, 9)
------------------------------------------------------------------------
Angle :math:`\theta` (rad; 0 anatomical, + flexion) and angular velocity :math:`\omega` (rad/s; + concentric for the
exertion considered):

.. math::

    T_{max}(\theta, \omega) = C_1 \cos(C_2(\theta - C_3))\, f(\omega), \qquad
    f(\omega) = \frac{2C_4C_5 + \omega(C_5 - 3C_4)}{2C_4C_5 + \omega(2C_5 - 4C_4)} \quad (\omega \ge 0)

C4 and C5 are the speeds at which torque falls to 75 % and 50 % of isometric. Eccentric torque is the concentric
torque at the same speed scaled by :math:`(1 - C_6\omega)` with :math:`\omega < 0` (eq. 8). C1 is normalised by body
weight times height. Passive torque is :math:`B_1 e^{k_1\theta} + B_2 e^{k_2\theta}`.

Caveats that travel with every result:

* **Range of the data.** The tested ranges were −0.5 to 1.3 rad (hip) and 0 to 1.8 rad (knee). A rower's catch flexes
  both further (hip ~2.2 rad, knee ~2.2 rad), so there the cosine is *extrapolated*.
* **Who was tested.** The subjects were active adults, not rowers.

The elbow uses Frey-Law et al. (2012) Tables 4–5 directly (bilinear in angle and speed). Shoulder extension, which
carries much of the arm draw, has no source yet.

Activation dynamics
-------------------
:math:`\dot a = (u - a)/\tau`, with :math:`\tau = \tau_{act}` when :math:`u > a` and :math:`\tau_{deact}` otherwise. The
time constants are 10 / 40 ms (Zajac 1989; Winters 1990) or 15 / 50 ms (Thelen 2003).

Trunk impedance (Moorhouse & Granata 2007, 11 men, isometric extension at 20, 35 and 50 % MVE)
-----------------------------------------------------------------------------------------------
Their stiffnesses are linear (N/m at T10, pelvis fixed). Each one is carried to a model as a **ratio to their
gravitational term** (1695 N/m), multiplied by the model's own gravitational stiffness :math:`m g h` of the upper
body. No lever length is assumed: an identification at a point maps to rotation through the same lever for every term.

* Intrinsic stiffness, :math:`k_{INT}`: 1280 / 1738 / 2116 N/m.
* Reflex: a 40 ms delay, then a differentiator (velocity), a half-wave rectifier (stretch only), and
  :math:`G_R/(s^2 + 2\zeta\omega_n s + \omega_n^2)` with :math:`G_R` 197 / 221 / 248 N·s/m, ζ 0.66 / 0.55 / 0.53 and
  ω_n 8.69 / 9.54 / 9.16 Hz. Read literally, that transfer function's units do not close: G_R in N·s/m would need a
  unity-gain low-pass. The paper's k_REFLEX values also do not follow from G_R/(2ζω_n). So this module reads it as
  :math:`G_R\,\omega_n^2/(s^2 + 2\zeta\omega_n s + \omega_n^2)` acting on the stretch velocity, which is an
  interpretation.

Here the velocity gain carries units of s after the same normalisation. Effort is interpolated linearly in % MVE and
held flat outside 20–50 %; holding it flat is a choice, and it is flagged where it binds. The source reports no
intrinsic damping. In a muscle-driven model, damping comes from the generator's own force-velocity slope
(:meth:`JointTorque.damping`).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

__all__ = ["JointTorque", "ElbowTable", "ActivationDynamics", "TrunkImpedance", "anderson", "elbow",
           "ANDERSON_FILE", "FREYLAW_FILE"]

G = 9.81
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "data", "literature")
ANDERSON_FILE = os.path.join(_LIT, "anderson2007_joint_torque.csv")
FREYLAW_FILE = os.path.join(_LIT, "freylaw2012_elbow_strength.csv")


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@lru_cache(maxsize=None)
def _anderson_table() -> Dict[Tuple[str, str], Dict[str, float]]:
    return {(r["exertion"], r["group"]): {k: float(v) for k, v in r.items() if k not in ("exertion", "group")}
            for r in _rows(ANDERSON_FILE)}


@dataclass(frozen=True)
class JointTorque:
    """Maximum voluntary torque at one joint in one direction (Anderson et al. 2007), in N·m.

    ``C1`` here is already in N·m (the table's normalised value times body weight times height); ``sides`` multiplies
    for both limbs together."""

    exertion: str
    C1: float
    C2: float
    C3: float
    C4: float
    C5: float
    C6: float
    B1: float = 0.0
    k1: float = 0.0
    B2: float = 0.0
    k2: float = 0.0
    sides: int = 1

    def angle_factor(self, theta):
        """The torque-angle factor cos(C2 (theta - C3)), floored at zero."""
        return np.maximum(np.cos(self.C2 * (np.asarray(theta, float) - self.C3)), 0.0)

    def velocity_factor(self, omega):
        """Concentric hyperbola (eq. 9) for omega >= 0; eccentric scaling (eq. 8) for omega < 0. Floored at zero."""
        w = np.asarray(omega, float)
        v = np.abs(w)
        a, b = self.C4, self.C5
        f = (2 * a * b + v * (b - 3 * a)) / (2 * a * b + v * (2 * b - 4 * a))
        f = np.maximum(f, 0.0)
        return np.where(w >= 0.0, f, f * (1.0 + self.C6 * v))

    def max_active(self, theta, omega):
        """Maximum active torque at (theta, omega), N·m, both sides."""
        return self.sides * self.C1 * self.angle_factor(theta) * self.velocity_factor(omega)

    def passive(self, theta):
        """Passive torque in this exertion's direction, N·m, both sides."""
        th = np.asarray(theta, float)
        return self.sides * (self.B1 * np.exp(self.k1 * th) + self.B2 * np.exp(self.k2 * th))

    def activation_needed(self, torque, theta, omega):
        """The activation (0-1 = maximum voluntary) a required torque needs, passive torque counted. > 1: infeasible."""
        cap = self.max_active(theta, omega)
        need = np.asarray(torque, float) - self.passive(theta)
        return np.where(cap > 1e-9, need / np.maximum(cap, 1e-9), np.inf)

    def damping(self, activation, theta, omega, step=1e-3):
        """The intrinsic damping the force-velocity slope gives at this activation: -dT/d(omega), N·m·s/rad."""
        up = self.max_active(theta, np.asarray(omega, float) + step)
        dn = self.max_active(theta, np.asarray(omega, float) - step)
        return -np.asarray(activation, float) * (up - dn) / (2 * step)


def anderson(exertion: str, group: str, mass: float, stature: float, sides: int = 2,
             passive: bool = True) -> JointTorque:
    """Anderson et al. (2007) Table 3 mean parameters for ``exertion`` (HE, HF, KE, KF, PF, DF) and ``group``
    (M18-25, F18-25, M55-65, F55-65, M65+, F65+), C1 scaled to this body (weight x height).

    ``passive=False`` drops the passive terms. The hip's were measured standing, with the knee held nearly straight
    (hamstrings stretched): its mean flexion term 0.476 exp(5.91 theta) gives ~176 N·m per side at 1 rad of hip
    flexion. A rower's knee is flexed whenever the hip is, so that term does not carry over to rowing."""
    p = dict(_anderson_table()[(exertion, group)])
    p["C1"] = p["C1"] * mass * G * stature
    if not passive:
        p.update(B1=0.0, B2=0.0)
    return JointTorque(exertion=exertion, sides=sides, **p)


@dataclass(frozen=True)
class ElbowTable:
    """Frey-Law et al. (2012) peak elbow torque, bilinear in angle (deg, 0 = extended) and concentric speed (deg/s).

    Unreported cells (speeds most subjects could not reach) are filled by linear extrapolation in speed from the two
    fastest reported cells at that angle, floored at zero; eccentric speeds are held at the isometric value (no data).
    """

    direction: str
    sex: str
    angles: Tuple[float, ...]
    speeds: Tuple[float, ...]
    grid: Tuple[Tuple[float, ...], ...]
    sides: int = 2

    def max_active(self, theta, omega):
        """theta rad (flexion), omega rad/s (+ concentric); N·m for both arms."""
        ang = np.degrees(np.asarray(theta, float))
        spd = np.degrees(np.maximum(np.asarray(omega, float), 0.0))
        A, S, Z = np.array(self.angles), np.array(self.speeds), np.array(self.grid)
        # extrapolate in speed beyond the table, per angle (linear from the last two speeds)
        out = np.empty(np.broadcast(ang, spd).shape)
        for idx, (a_, s_) in enumerate(np.broadcast(ang, spd)):
            col = np.array([np.interp(a_, A, Z[i]) for i in range(len(S))])
            if s_ <= S[-1]:
                v = np.interp(s_, S, col)
            else:
                v = col[-1] + (col[-1] - col[-2]) / (S[-1] - S[-2]) * (s_ - S[-1])
            out.flat[idx] = max(v, 0.0)
        return self.sides * out.reshape(np.broadcast(ang, spd).shape)

    def activation_needed(self, torque, theta, omega):
        cap = self.max_active(theta, omega)
        return np.where(cap > 1e-9, np.asarray(torque, float) / np.maximum(cap, 1e-9), np.inf)


def elbow(direction: str = "flexion", sex: str = "M", sides: int = 2) -> ElbowTable:
    rows = [r for r in _rows(FREYLAW_FILE) if r["direction"] == direction and r["sex"] == sex]
    angles = (15.0, 40.0, 65.0, 90.0, 110.0)
    speeds = tuple(float(r["velocity_dps"]) for r in rows)
    raw = [[(float(r[k]) if r[k] else np.nan) for k in ("a15", "a40", "a65", "a90", "a110")] for r in rows]
    Z = np.array(raw)
    for j in range(Z.shape[1]):                                  # fill unreported fast cells, per angle
        ok = np.where(np.isfinite(Z[:, j]))[0]
        for i in np.where(~np.isfinite(Z[:, j]))[0]:
            i1, i2 = ok[-1], ok[-2]
            slope = (Z[i1, j] - Z[i2, j]) / (speeds[i1] - speeds[i2])
            Z[i, j] = max(Z[i1, j] + slope * (speeds[i] - speeds[i1]), 0.0)
    return ElbowTable(direction, sex, angles, speeds, tuple(map(tuple, Z)), sides)


@dataclass(frozen=True)
class ActivationDynamics:
    """First-order excitation-to-activation dynamics. ``source``: 'zajac' (10 / 40 ms) or 'thelen' (15 / 50 ms)."""

    tau_act: float = 0.010
    tau_deact: float = 0.040
    floor: float = 0.01

    @classmethod
    def of(cls, source: str = "zajac") -> "ActivationDynamics":
        return {"zajac": cls(0.010, 0.040), "thelen": cls(0.015, 0.050)}[source]

    def rate(self, u, a):
        u = np.clip(u, self.floor, 1.0)
        tau = np.where(u > a, self.tau_act, self.tau_deact)
        return (u - a) / tau


# Moorhouse & Granata (2007) Tables 2-3, means at 20 / 35 / 50 % MVE (N/m, N·s/m, -, Hz) and their gravity term (N/m)
MG07_EFFORT = (0.20, 0.35, 0.50)
MG07_K_INT = (1280.0, 1738.0, 2116.0)
MG07_G_R = (197.0, 221.0, 248.0)
MG07_ZETA = (0.66, 0.55, 0.53)
MG07_WN_HZ = (8.69, 9.54, 9.16)
MG07_GRAVITY = 1695.0
MG07_DELAY = 0.040


@dataclass(frozen=True)
class TrunkImpedance:
    """The trunk's intrinsic and reflex layers (Moorhouse & Granata 2007), normalised to a model's gravity term.

    ``gravity_stiffness``: the model's own m g h for the upper body about its rotation axis, N·m/rad."""

    gravity_stiffness: float

    @staticmethod
    def _at(effort, table):
        return float(np.interp(np.clip(effort, MG07_EFFORT[0], MG07_EFFORT[-1]), MG07_EFFORT, table))

    def intrinsic_stiffness(self, effort):
        """k_INT at this effort (fraction of MVE), N·m/rad."""
        return self._at(effort, MG07_K_INT) / MG07_GRAVITY * self.gravity_stiffness

    def reflex_gain(self, effort):
        """G_R at this effort, normalised: N·m·s/rad, the reflex torque per unit stretch velocity at low frequency."""
        return self._at(effort, MG07_G_R) / MG07_GRAVITY * self.gravity_stiffness

    def reflex_filter(self, effort):
        """(zeta, w_n rad/s) of the reflex's second-order element."""
        return self._at(effort, MG07_ZETA), 2 * np.pi * self._at(effort, MG07_WN_HZ)

    delay: float = MG07_DELAY


class ReflexState:
    """The reflex pathway as a running filter: delayed stretch velocity -> half-wave rectifier -> G_R / (s^2 + 2 z w s
    + w^2). Integrated with the caller's accepted steps (semi-implicit Euler), it holds its own history."""

    def __init__(self, impedance: TrunkImpedance):
        self.imp = impedance
        self.t = []
        self.v = []
        self.y, self.yd = 0.0, 0.0
        self.last = None

    def record(self, t, stretch_velocity):
        if not self.t or t > self.t[-1]:
            self.t.append(float(t)); self.v.append(float(stretch_velocity))

    def stretch_seen(self, t):
        d = self.imp.delay
        if not self.t or t - d < self.t[0]:
            return 0.0
        return float(np.interp(t - d, self.t, self.v))

    def output(self, t, effort):
        """Advance the filter to ``t`` and return the reflex torque (N·m, opposing stretch)."""
        if self.last is None:
            self.last = t
        h = t - self.last
        if h > 0:
            zeta, wn = self.imp.reflex_filter(effort)
            x = max(self.stretch_seen(t), 0.0) * self.imp.reflex_gain(effort)
            n = max(1, int(np.ceil(h / 5e-4)))
            dt = h / n
            for _ in range(n):
                ydd = wn * wn * (x - self.y) - 2 * zeta * wn * self.yd
                self.yd += dt * ydd
                self.y += dt * self.yd
            self.last = t
        return self.y
