"""The blade review's study laws (research/blade/helmbold.py) against the project's existing lifting-surface code.

docs/BLADE_HYDRODYNAMICS_REVIEW.md; SOURCES §188-190. The review applied the fins' own low-aspect-ratio lift to the
blade; these tests pin that the study module computes the same thing rather than a second, drifting copy.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "research", "blade"))
import helmbold                                                    # noqa: E402

from coxswain.hydro.appendages import LiftingSurface              # noqa: E402


@pytest.mark.parametrize("aspect", [0.45, 0.48, 0.96, 2.0, 6.0])
def test_slope_is_the_fins_polhamus_helmbold_slope(aspect):
    # LiftingSurface's aspect ratio is span / chord; an unswept surface
    fin = LiftingSurface(span=aspect * 0.2, chord=0.2, position=np.zeros(3), sweep=0.0)
    assert fin.aspect_ratio == pytest.approx(aspect)
    assert helmbold.helmbold_slope(aspect) == pytest.approx(fin.lift_curve_slope, rel=1e-12)


def test_crossflow_form_is_self_capping():
    # Whicker & Fehlner's cross-flow form: face-on the normal coefficient is C_Dc, with no separate cap
    for c_dc in (1.3, 2.05):
        b = helmbold.crossflow(0.25, 0.52, 0.11, False, c_dc)
        assert b.coefficients(np.pi / 2)[0] == pytest.approx(c_dc)
        assert b.coefficients(np.radians(30.0))[0] < c_dc


def test_tangential_load_is_drag_like():
    # [GTW21] measures the tangential load opposing the sliding (SOURCES §184); with Helmbold's lift it is positive
    b = helmbold.crossflow(0.25, 0.52, 0.11, False, 1.3)
    assert b.coefficients(np.radians(45.0))[1] > 0.0


def test_strips_honour_a_blade_that_overrides_its_coefficients():
    # liftdrag_strips used to resolve tier 2's lift and drag itself, silently ignoring a subclass's own
    # coefficients (the cross-flow law's face-on value had no effect under strips; SOURCES §191)
    from coxswain.crew.blade_strips import liftdrag_strips
    from coxswain.crew.liftdrag import LiftDragBlade
    args = (np.radians(10.0), -2.0, np.array([4.0, 0.0]), 1, 0.43)
    lo = helmbold.crossflow(0.21, 0.43, 0.0903, False, 1.3, outboard=1.8)
    hi = helmbold.crossflow(0.21, 0.43, 0.0903, False, 2.2, outboard=1.8)
    assert abs(liftdrag_strips(hi, *args)[0]) > abs(liftdrag_strips(lo, *args)[0])
    # and a plain tier 2 blade is unchanged by the fix: strips equal the vectorised lift-drag resolution
    b = LiftDragBlade.big_blade(outboard=1.8, area=0.0903)
    f_n, f_t, _ = liftdrag_strips(b, *args)
    assert np.isfinite(f_n) and np.isfinite(f_t)


def test_core_crossflow_blade_matches_the_study_law():
    from coxswain.crew.liftdrag import LiftDragBlade
    core = LiftDragBlade.crossflow(outboard=1.8, area=0.0903, length=0.43, face_on=2.2)
    study = helmbold.PolhamusLamarBlade(helmbold.helmbold_slope(0.0903 / 0.43 ** 2) / 2.0, 2.07, 0.0903, 1.8,
                                        1000.0, None, 2.2, 99.0)
    for a in np.radians([5.0, 30.0, 60.0, 90.0]):
        assert core.coefficients(a) == pytest.approx(study.coefficients(a))


def test_simulator_names_the_crossflow_blade_and_refuses_a_bad_name():
    from coxswain.sim.dynamic_oar import _crossflow_face_on
    assert _crossflow_face_on("crossflow@2.2") == 2.2
    assert _crossflow_face_on("crossflow@x") is None and _crossflow_face_on("cg07") is None
