"""[CR06]'s Model 1, as printed, reproduces their own printed result on their own data.

Their trial (c): leg, back and oar motion fitted to the measurements, predicted mean boat speed
3.83 m/s against 4.18 measured. Driven by the measured curves (data/literature, re-extracted
from their Fig. 3) with their Table 2.7 constants, the same equations give 3.80.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "research", "cr06"))
import reproduce_model1 as M                                        # noqa: E402

from coxswain.crew.oarlock import BladeModel                        # noqa: E402


def test_model_1_reproduces_their_trial_c_speed():
    r = M.run(harmonics=12, periods=15)
    assert r["speed"] == pytest.approx(M.PRINTED["trial_c_speed"], abs=0.06)
    assert r["measured_speed"] == pytest.approx(M.PRINTED["measured_speed"], abs=0.02)


def test_their_fitted_c2_brings_the_speed_to_the_measurement():
    """p. 1045: the C2 that minimises their error is ~2.4x nominal."""
    r = M.run(harmonics=12, c2=2.4 * M.C2, periods=15)
    assert r["speed"] == pytest.approx(M.PRINTED["measured_speed"], abs=0.08)


def test_the_models_blade_law_is_their_eq_11():
    """Same magnitude, C2 (l phi' + v cos phi)^2. [CR06] apply it only while the normal velocity
    drives (their drive window, eq. 16); ours signs it to oppose the slip, so outside that
    window it brakes -- the angle release's documented behaviour, refused by release="slip"."""
    blade = BladeModel(c2=M.C2, outboard=M.L)
    for phi, rate, v in ((0.3, -2.0, 4.0), (-0.4, -1.5, 4.5), (0.9, -1.2, 3.0)):
        normal = M.L * rate + v * np.cos(phi)
        force = float(blade.normal_force(phi, rate, v))
        assert abs(force) == pytest.approx(M.C2 * normal ** 2)
        assert np.sign(force) == -np.sign(normal)
