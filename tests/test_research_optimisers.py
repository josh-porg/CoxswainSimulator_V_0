"""The route and pacing optimisers see the research model's wave drag.

A boat carrying a depth-aware wave table -- what ``research`` and ``learned``
profiles give it -- has depth change only its wave term, by Sretenskii's
integral, in :class:`~coxswain.crew.pacing.CoursePacing` and
:class:`~coxswain.river.route.RouteEvaluator`.  Anything else keeps the
chosen shallow-water factor exactly as before.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from coxswain.boats import catalog
from coxswain.crew.pacing import CoursePacing, CourseSegment
from coxswain.hydro.finite_depth_michell import FiniteDepthWaveTable
from coxswain.hydro.michell import MichellWave, elliptical_offsets
from coxswain.hydro.resistance import hull_resistance
from coxswain.river.course import Course, DepthField
from coxswain.river.route import RouteEvaluator


@pytest.fixture(scope="module")
def boats():
    """The same eight twice: a plain deep table, and a depth-aware one."""
    plain_boat = catalog.build("8+", rate=32.0)
    research_boat = catalog.build("8+", rate=32.0)
    x, z, half = elliptical_offsets(plain_boat.offsets, stations=161, levels=21)
    research_boat.wave_table = FiniteDepthWaveTable(station=x, level=z,
                                                    half_beam=half)
    plain_boat.wave_table = MichellWave(station=x, level=z, half_beam=half,
                                        quadrature="trapezoid").tabulate()
    return plain_boat, research_boat


def drag_for(boat):
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)

    def drag(speed):
        force, _ = hull_resistance(
            np.array([float(speed), 0.0, 0.0]), submerged,
            mean_wetted_length=boat.length, water=boat.water,
            coefficients=boat.resistance, wave_table=boat.wave_table)
        return abs(float(force[0]))
    return drag


# --------------------------------------------------------------------------
# pacing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [float("inf"), 2.2])
def test_pacing_with_a_plain_table_is_unchanged(boats, depth):
    plain_boat, _ = boats
    drag = drag_for(plain_boat)
    segment = CourseSegment(500.0, depth=depth)
    before = CoursePacing([segment], drag, shallow_model=plain_boat.shallow)
    after = CoursePacing([segment], drag, shallow_model=plain_boat.shallow,
                         wave_table=plain_boat.wave_table)
    assert after.speed_for_power(330.0, segment) == \
        before.speed_for_power(330.0, segment)


def test_pacing_deep_segment_is_the_same_either_way(boats):
    _, research_boat = boats
    drag = drag_for(research_boat)
    segment = CourseSegment(500.0)
    factor = CoursePacing([segment], drag)
    research = CoursePacing([segment], drag,
                            wave_table=research_boat.wave_table)
    assert research.speed_for_power(330.0, segment) == \
        factor.speed_for_power(330.0, segment)


def test_pacing_shallow_segment_uses_sretenskii_not_the_factor(boats):
    _, research_boat = boats
    table = research_boat.wave_table
    drag = drag_for(research_boat)
    depth, power = 2.6, 420.0
    segment = CourseSegment(500.0, depth=depth)
    model = CoursePacing([segment], drag, wave_table=table)
    speed = model.speed_for_power(power, segment)

    delivered = model.efficiency * power * model.rowers

    def balance(v):
        air = 0.5 * model.air_density * model.drag_area * v * abs(v)
        wave = table.at_depth(v, depth) - float(table(v))
        return (drag(v) + wave + air) * v - delivered

    assert speed == pytest.approx(brentq(balance, 1.0, 9.0), abs=1e-3)
    factor = CoursePacing([segment], drag).speed_for_power(power, segment)
    assert speed != pytest.approx(factor, abs=1e-3)


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def straight_course():
    station = np.linspace(0.0, 2000.0, 40)
    centreline = np.column_stack([station, np.zeros_like(station)])
    offsets = np.linspace(-50.0, 50.0, 21)
    points, depths = [], []
    for s in station:
        for o in offsets:
            points.append([s, o])
            depths.append(6.0 - 4.5 * (o / 50.0) ** 2)
    return Course(centreline=centreline, half_width=50.0,
                  depth=DepthField(points=np.array(points),
                                   depths=np.array(depths), is_survey=True),
                  name="test reach")


@pytest.fixture(scope="module")
def research_route(boats, straight_course):
    _, research_boat = boats
    return RouteEvaluator(straight_course, boat=research_boat,
                          reference_speed=5.2)


def test_route_speed_is_monotone_and_bounded(research_route):
    depths = np.array([50.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0])
    speeds = research_route.speed_through_water(depths)
    assert np.all(np.diff(speeds) <= 1e-12), speeds
    assert np.all(speeds <= 5.2 + 1e-9)
    assert research_route.speed_through_water(200.0) == pytest.approx(
        5.2, rel=1e-6)


def test_route_speed_is_the_boats_own_power_balance(boats, research_route,
                                                    straight_course):
    _, research_boat = boats
    table = research_boat.wave_table
    drag = drag_for(research_boat)
    # On a grid node, so the check is of the solve and not the interpolation.
    research_route.speed_through_water(3.0)
    grid = research_route._speed_table[0]
    depth = float(grid[np.argmin(np.abs(grid - 3.0))])
    speed = research_route.speed_through_water(depth)
    target = drag(5.2) * 5.2
    residual = (drag(speed) + table.at_depth(speed, depth)
                - float(table(speed))) * speed - target
    assert abs(residual) < 1e-3 * target
    shipped = RouteEvaluator(straight_course, reference_speed=5.2)
    assert speed != pytest.approx(shipped.speed_through_water(depth),
                                  abs=1e-3)
