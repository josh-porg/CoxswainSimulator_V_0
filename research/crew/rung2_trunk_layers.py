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
  layer on top.

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
            T += _Base._feedback(self, t, e, ed, s)
        if not self.layerlog or t > self.layerlog[-1][0]:
            self.layerlog.append((t, eff, T))
        return T

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
    a = ap.parse_args()
    LayersSim.LAYERS = tuple(a.layers)
    RT.TrunkClockSim = LayersSim                                 # rung2_trunk.make builds this class
    try:
        out = RT.athletes(a.strokes, ff="internal", ff_delay=a.ff_delay, delay=a.fb_delay, spin=True, efference=True)
    finally:
        RT.TrunkClockSim = _Base
    out["layers"] = list(a.layers)
    tag = "_".join(a.layers) + ("_ffd%.0fms" % (1000 * a.ff_delay))
    json.dump(out, open(os.path.join(RT.OUT, "rung2_trunk_layers_%s.json" % tag), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
