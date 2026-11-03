"""Multi-clock body, steps 2-3 (review §12): the trunk's zero-delay and spinal layers in place of one delayed PD.

    .venv/Scripts/python.exe research/crew/rung2_trunk_layers.py [--layers intrinsic reflex ll] [--ff-delay 0.1]

§206 found the trunk loop failing in two ways. A single PD acting 100 ms late, with an efference-copy internal model
that perceives the clock's speed 100 ms late, let the trunk wander 29-39° off its path. The review (§11b) traced this
to a missing structure: a stabilising human trunk has two lower layers (Moorhouse & Granata 2007):

* **intrinsic**: activation-scaled stiffness acting with no delay;
* **reflex**: a velocity reflex on stretch at 40 ms.

This script replaces rung 5's PD with those layers. Everything else is rung2_trunk's ``athletes`` check, unchanged:

* the population's muscles on the stroke clock;
* the efference-copy internal model as the trunk's feedforward, which perceives tau' ``--ff-delay`` late and
  knows tau'' from its own command;
* the research blade, with the catch at burial.

Layers (``--layers``, any of):

* ``intrinsic``: T = -k_INT(e) delta - b(e) delta'.
  - k_INT is Moorhouse & Granata's at effort e, as a ratio to their gravity term times this upper body's own upright
    gravitational stiffness (coxswain/crew/joint_muscles.TrunkImpedance).
  - b = e C1 |f'(0)|, the force-velocity slope of Anderson et al.'s hip extensors at their peak angle.
  - e = |feedforward hip moment| / C1, the isometric peak of both hips. The angle factor is not used because the
    catch lies outside the strength data (muscle_capacity.py).
  - Effort is held at the measured 20-50 % range's ends outside it.
* ``reflex``: the stretch velocity of the trunk extensors (delta' > 0 is forward flexion) passes, in order, through
  - a 40 ms delay;
  - a half-wave rectifier;
  - G_R with Moorhouse & Granata's second-order element.
  It acts on the deviation from the intended path, i.e. with alpha-gamma co-activation, which is an interpretation.
* ``ll``: rung 5's PD (w_n 5, zeta 0.7, lambda_c^2 24.7) at ``--fb-delay`` (default 100 ms), as the long-latency
  layer on top; with ``--predict``, on the state predicted over the delay (rung 5's finite-horizon predictor, its
  internal model including the zero-delay layer; SOURCES §215).

No gain is tuned. Outputs: data/local/crew/rung2_trunk_layers_<layers>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import rung2_trunk as RT                                         # noqa: E402

from coxswain.crew import joint_muscles as J                     # noqa: E402

G = 9.81
_Base = RT.TrunkClockSim


class LayersSim(_Base):
    LAYERS = ("intrinsic", "reflex")

    def _tabulate(self):
        super()._tabulate()
        rower = self.boat.crew[0].rower
        anth = rower.anthropometry
        group = ("F" if str(anth.sex).startswith("f") else "M") + "18-25"
        self._HE = J.anderson("HE", group, float(rower.total_mass), float(anth.stature), passive=False)
        masses = np.asarray(rower.segment_masses, float)
        grid = np.linspace(0.0, self._P, 200, endpoint=False)
        reach = [np.mean([np.linalg.norm(self._r_at(i, c)) for c in grid]) for i in RT.TRUNK]
        self._k_grav = float(sum(masses[i] * G * r for i, r in zip(RT.TRUNK, reach)))   # upright m g h, N.m/rad
        self._imp = J.TrunkImpedance(self._k_grav)
        self._reflex = J.ReflexState(self._imp)
        he = self._HE
        self._fv_slope = abs((he.C5 - he.C4) / (2 * he.C4 * he.C5))  # |f'(0)| per rad/s
        self.layerlog = []
        self._ff_now = 0.0

    def _effort(self):
        return float(np.clip(abs(self._ff_now) / (self._HE.sides * self._HE.C1), 0.0, 1.0))

    def _feedback(self, t, e, ed, s):
        sg = 1.0 if s >= 0 else -1.0
        eff = self._effort()
        T = 0.0
        if "intrinsic" in self.LAYERS:
            k = self._imp.intrinsic_stiffness(eff)
            b = eff * self._HE.sides * self._HE.C1 * self._fv_slope
            T += sg * (-k * e - b * ed)
        if "reflex" in self.LAYERS:
            self._reflex.record(t, ed)                           # stretch of the extensors: forward flexion, delta' > 0
            T += -sg * self._reflex.output(t, eff)
        if "ll" in self.LAYERS:
            T += self._long_latency(t, e, ed, s, eff)
        if not self.layerlog or t > self.layerlog[-1][0]:
            self.layerlog.append((t, eff, T))
        return T

    #: the long-latency layer on a state predicted over its delay (rung 5's finite-horizon predictor, SOURCES §182)
    PREDICT = False
    PRED_STEPS = 20

    def _u_at(self, tq):
        ut, uu = getattr(self, "_uhist", ([], []))
        k = int(np.searchsorted(ut, tq, side="right")) - 1
        return uu[k] if k >= 0 else 0.0

    def _long_latency(self, t, e, ed, s, eff):
        """Rung 5's PD at ``delay``; with PREDICT, on the state projected over the delay by the internal model

            e'' = (lambda_c^2 - kappa) e - beta e' + u,   kappa = k_INT(eff)/|s|,  beta = b(eff)/|s|

        (the trunk's toppling rate as rung 5 measured it on the coupled plant, less the zero-delay layer's own stiffness
        and damping, which are properties of the body) driven by the layer's own past commands u (an efference copy,
        held between samples; exact zero-order-hold steps). The reflex is not in the model: it is a disturbance to it."""
        if not self.PREDICT or self.delay <= 0:
            return _Base._feedback(self, t, e, ed, s)
        from scipy.linalg import expm
        ht, he, hed = self.hist
        if ht and t - self.delay > ht[0]:
            x = np.array([float(np.interp(t - self.delay, ht, he)), float(np.interp(t - self.delay, ht, hed))])
        else:
            x = np.zeros(2)
        a_s = max(abs(s), 1e-9)
        kappa = self._imp.intrinsic_stiffness(eff) / a_s if "intrinsic" in self.LAYERS else 0.0
        beta = eff * self._HE.sides * self._HE.C1 * self._fv_slope / a_s if "intrinsic" in self.LAYERS else 0.0
        h = self.delay / self.PRED_STEPS
        M = np.zeros((3, 3))
        M[0, 1], M[1, 0], M[1, 1], M[1, 2] = 1.0, RT.LAM2 - kappa, -beta, 1.0
        E = expm(M * h)
        Ah, Bh = E[:2, :2], E[:2, 2]
        if self.OBSERVER > 0:
            self._observe(t, x, kappa, beta)
        d_hat = getattr(self, "_d_hat", 0.0)
        for i in range(self.PRED_STEPS):
            x = Ah @ x + Bh * (self._u_at(t - self.delay + i * h) + d_hat)
        u = -self.Kp * x[0] - self.Kd * x[1]
        if not hasattr(self, "_uhist"):
            self._uhist = ([], [])
        ut, uu = self._uhist
        if not ut or t > ut[-1]:
            ut.append(t)
            uu.append(u)
            if len(ut) > 6000:
                keep = int(np.searchsorted(ut, t - self.delay - 0.2))
                del ut[:keep]
                del uu[:keep]
        return s * u

    #: a slow disturbance observer on the predictor (rung 5's, K_OBS 2 /s; SOURCES §182): it learns the steady part of
    #: what the internal model misses (the reflex, the clock's misread speed, the model's constant coefficients)
    OBSERVER = 0.0

    def _observe(self, t, x_delayed, kappa, beta):
        """d_hat += K (a_measured - a_model) dt, on the delayed state: a_measured from a 5 ms difference of the logged
        rate, a_model = (lambda_c^2 - kappa) e - beta e' + u + d_hat at the same instant (rung 5's observer)."""
        ht, he, hed = self.hist
        td = t - self.delay
        if len(ht) < 3 or td - 0.005 <= ht[0]:
            return
        a_meas = (float(np.interp(td, ht, hed)) - float(np.interp(td - 0.005, ht, hed))) / 0.005
        d_hat = getattr(self, "_d_hat", 0.0)
        a_model = (RT.LAM2 - kappa) * x_delayed[0] - beta * x_delayed[1] + self._u_at(td) + d_hat
        t_prev = getattr(self, "_t_obs", None)
        if t_prev is not None and t > t_prev:
            self._d_hat = d_hat + self.OBSERVER * (a_meas - a_model) * (t - t_prev)
        self._t_obs = t

    def _path_moment(self, t, base, d, tau, td, a1):
        m = _Base._path_moment(self, t, base, d, tau, td, a1)
        self._ff_now = m
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["intrinsic", "reflex"], choices=["intrinsic", "reflex", "ll"])
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--ff-delay", type=float, default=0.1, help="the internal model's perception lag on tau', s")
    ap.add_argument("--fb-delay", type=float, default=0.1, help="the long-latency PD's delay ('ll' layer), s")
    ap.add_argument("--predict", action="store_true",
                    help="the 'll' layer acts on the state predicted over its delay (rung 5's predictor)")
    ap.add_argument("--ff-model", default="efference", choices=["efference", "learned"],
                    help="how the trunk's internal model knows the clock: the efference copy of tau'' (as before) or "
                         "the previous stroke's clock, corrected by delayed perception (SOURCES sec. 217)")
    ap.add_argument("--observer", type=float, default=0.0,
                    help="with the predictor, a disturbance observer of this gain, /s (rung 5's: 2)")
    ap.add_argument("--rigid", action="store_true", help="the rigid oar (the research profile bends it since §217)")
    a = ap.parse_args()
    RT.RP.RIGID = bool(a.rigid)
    LayersSim.LAYERS = tuple(a.layers)
    LayersSim.PREDICT = bool(a.predict or a.observer)
    LayersSim.OBSERVER = float(a.observer)
    RT.TrunkClockSim = LayersSim                                 # rung2_trunk.make builds this class
    try:
        out = RT.athletes(a.strokes, ff="internal", ff_delay=a.ff_delay, delay=a.fb_delay, spin=True, efference=True,
                          ff_model=a.ff_model)
    finally:
        RT.TrunkClockSim = _Base
    out["layers"] = list(a.layers)
    tag = "_".join(a.layers) + ("_pred" if a.predict or a.observer else "") + ("_obs%g" % a.observer if a.observer else "") + ("_learned" if a.ff_model == "learned" else "") + ("_rigid" if a.rigid else "") + ("_ffd%.0fms" % (1000 * a.ff_delay))
    json.dump(out, open(os.path.join(RT.OUT, "rung2_trunk_layers_%s.json" % tag), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
