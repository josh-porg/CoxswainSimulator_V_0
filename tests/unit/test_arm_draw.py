"""The arm draw's clock (SOURCES §209): fitted to the population elbow curve, smooth where it starts."""
from __future__ import annotations

import numpy as np

from coxswain.crew import arm_draw as A


def test_the_constants_are_the_population_fit():
    hold, onset, finish, rms = A.fit_population()
    assert abs(hold - np.degrees(A.HOLD_FLEXION)) < 0.15
    assert abs(onset - A.DRAW_ONSET) < 0.005
    assert abs(finish - np.degrees(A.POPULATION_FINISH_FLEXION)) < 0.5
    assert rms < 1.5                        # a held angle and one minimum-jerk movement describe the curve


def test_minimum_jerk_starts_and_ends_at_rest():
    m, dm, d2m = A.minimum_jerk(np.array([0.0, 1.0]))
    np.testing.assert_allclose(m, [0.0, 1.0])
    np.testing.assert_allclose(dm, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(d2m, [0.0, 0.0], atol=1e-12)


def test_the_draw_is_c2_at_its_onset_and_held_before():
    d = A.ElbowDraw(finish=np.radians(95.0))
    x = np.array([0.0, 0.3, d.onset])
    np.testing.assert_allclose(d.flexion(x), d.hold)
    np.testing.assert_allclose(d.rate(x), 0.0, atol=1e-12)
    assert abs(d.flexion(1.0) - np.radians(95.0)) < 1e-12 and abs(d.rate(1.0)) < 1e-12
    xs = np.linspace(0.3, 0.99, 300)
    h = 1e-6
    np.testing.assert_allclose(d.rate(xs), (d.flexion(xs + h) - d.flexion(xs - h)) / (2 * h), atol=1e-5)


def test_reach_is_the_law_of_cosines():
    assert abs(A.reach(0.3, 0.35, 0.0) - 0.65) < 1e-12
    assert abs(A.reach(0.3, 0.3, np.pi / 2) - 0.3 * np.sqrt(2)) < 1e-12
    # the held 10.4 deg leaves the grip within half a percent of the full arm: "straight" to the eye
    assert A.reach(0.28, 0.355, A.HOLD_FLEXION) / 0.635 > 0.995


def test_the_return_is_the_population_fit_and_starts_at_rest_at_the_finish():
    end, rms = A.fit_population_return()
    assert abs(end - A.RETURN_END) < 0.005 and rms < 3.5
    d = A.ElbowDraw(finish=np.radians(90.0))
    assert abs(d.recovery_flexion(0.0) - d.flexion(1.0)) < 1e-12        # continuous at the finish
    h = 1e-6
    assert abs((d.recovery_flexion(h) - d.recovery_flexion(0.0)) / h) < 1e-3   # at rest there, as the draw ended
    np.testing.assert_allclose(d.recovery_flexion(np.array([d.return_end, 0.9, 1.0])), d.hold)
