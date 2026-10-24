"""Lines through the arches are legal where they claim to be (2026-10-22).

A bridge control point used to carry the arch centre's offset at the bridge's own station; on the
skewed Grand Junction trestle that crossed the deck metres away, outside the narrow mandated arch,
and ``legalise`` (which checked the offset at the station) disagreed with the evaluator (which checks
the true crossing). Now the point sits on the deck and both use the true crossing.
"""
import itertools

import numpy as np
import pytest

from coxswain.raceday import card as C
from coxswain.river import bridges, lines


@pytest.fixture(scope="module")
def model():
    return C.CourseModel.build()


def _illegal_by_evaluator(model, route):
    model.route = route
    return C._evaluate(model, None, 1400.0, None).illegal_arches


def test_default_line_is_legal_both_ways(model):
    assert lines.illegal_bridges(model.route, model.course, model.raster, model.gates) == []
    assert _illegal_by_evaluator(model, model.route) == 0


def test_every_pinned_arch_choice_is_legal_both_ways(model):
    choices = {g.name: [a.label for a in bridges.candidate_arches(g, model.raster)]
               for g, _ in model.gates}
    names = [n for n, labels in choices.items() if len(labels) > 1]
    for combo in itertools.product(*[choices[n] for n in names]):
        pins = dict(zip(names, combo))
        route = lines.pinned_arch_route(model.course, model.raster, model.gates, pins)
        assert lines.illegal_bridges(route, model.course, model.raster, model.gates) == [], pins
        assert _illegal_by_evaluator(model, route) == 0, pins


def test_legalise_leaves_a_legal_line_alone(model):
    route = model.route
    again = lines.legalise(route, model.course, model.raster, model.gates)
    assert np.allclose(again.stations, route.stations)
    assert np.allclose(again.offsets, route.offsets)


def test_legalise_fixes_the_centreline(model):
    """The centreline goes through a trestle pier; legalised, it is legal both ways."""
    route = lines.legalise(lines.centreline_route(model.course), model.course, model.raster, model.gates)
    assert lines.illegal_bridges(route, model.course, model.raster, model.gates) == []
    assert _illegal_by_evaluator(model, route) == 0
