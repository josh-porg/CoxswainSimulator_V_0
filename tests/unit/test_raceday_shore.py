"""The opt-in Cambridge shore corridor (TRACKING, "The Charles corridor stops at the 1.22 m contour").

Two promises: it never narrows the corridor it replaces, and the evaluator refuses it without a research
boat, whose depth term the HOCR 2026 GPS supports (the default factor-on-total term would price the newly
reachable shallow water at 3-6x what the boats lost).
"""
import numpy as np
import pytest

from coxswain.raceday import card as C


@pytest.fixture(scope="module")
def models():
    return C.CourseModel.build(), C.CourseModel.build(starboard="shore")


def test_shore_limit_never_narrows_the_corridor(models):
    navigable, shore = models
    assert np.all(shore.course.starboard_limit >= navigable.course.starboard_limit - 1e-9)
    assert np.allclose(shore.course.port_limit, navigable.course.port_limit)


def test_shore_limit_reaches_the_water_the_crews_use(models):
    # every HOCR 2026 track rows the start stretch 61-78 m out; the chart's water ends 70-91 m out there
    _, shore = models
    stations = np.arange(300.0, 951.0, 50.0)
    stbd = np.array([shore.course.limits_at(s)[1] for s in stations])
    assert stbd.min() >= 65.0 and stbd.max() <= 95.0


def test_shore_needs_a_research_boat(models):
    _, shore = models
    from coxswain.boats import catalog
    shipped = catalog.coxed_four(rate=30.0)
    with pytest.raises(ValueError):
        C._evaluate(shore, shipped, 1300.0, None)


def test_bad_option_refused():
    with pytest.raises(ValueError):
        C.CourseModel.build(starboard="bank")
