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



def test_arch_point_clips_to_each_side(models):
    # arch windows clip each side to its own limit: under the shore corridor a deck point at the Cambridge
    # end keeps the extra starboard water instead of being cut back to the half-width (2026-10-27)
    from coxswain.river import lines
    _, shore = models
    course = shore.course
    for gate, _m in shore.gates:
        for along in (-500.0, 500.0):                     # past either end of the deck: clipped
            st, off = lines.arch_point(course, gate, along, 4.0)
            port, stbd = course.limits_at(st)
            assert -(stbd - 4.0) - 1e-9 <= off <= port - 4.0 + 1e-9
            assert np.isclose(off, -(stbd - 4.0)) or np.isclose(off, port - 4.0) or abs(off) < port - 4.0


def test_optimiser_bounds_follow_the_corridor(models, monkeypatch):
    # the line search's bounds are the corridor's own sides, not the symmetric half-width that kept it out
    # of the shore water while the crews' own lines used it (2026-10-27)
    import scipy.optimize
    from coxswain.raceday import plan as P
    from coxswain import physics
    from coxswain.boats import catalog

    class Captured(Exception):
        pass

    seen = {}

    def stop(cost, x0, **kw):
        seen["lo"] = np.array([b[0] for b in kw["bounds"]])
        raise Captured

    monkeypatch.setattr(scipy.optimize, "minimize", stop)
    monkeypatch.setattr(P, "_time", lambda *a, **k: (0.0, 0.0))
    _, shore = models
    course = shore.course
    free = (450.0, 2500.0)                                   # clear of the bridges (a point near one is dropped)
    boat = catalog.coxed_four(rate=30.0)
    physics.resolve("research").apply(boat)
    with pytest.raises(Captured):
        P.optimise_line(shore, boat, 1300.0, {}, free=free, margin=4.0, method="Powell")
    for f in free:
        want = -(course.limits_at(f)[1] - 4.0)
        assert want < -(course.half_width_at(f) - 4.0) - 1.0            # the shore really is wider here
        assert np.any(np.abs(seen["lo"] - want) < 1e-6)
