"""The regatta's traffic pattern.

The racing course is not the whole river: crews returning to the start
come down the Boston side, and the water a boat is *allowed* to use is
narrower than the water it could physically use.  None of that is visible
in a depth survey, which is the point of keeping it separate and keeping
the source of every width attached to it.
"""

import numpy as np
import pytest

from coxswain.river import traffic
from coxswain.river.bridges import EIGHT_ROWED_WIDTH


@pytest.fixture(scope="module")
def geometry():
    from coxswain.river.charts import CourseGeometry
    return CourseGeometry()


def test_a_lane_one_boat_wide_is_the_width_of_a_boat():
    assert traffic.ONE_BOAT == pytest.approx(EIGHT_ROWED_WIDTH)
    assert traffic.PASSING_WIDTH == pytest.approx(2 * EIGHT_ROWED_WIDTH)


def test_every_width_declares_where_it_came_from():
    """No width may be published-looking when it is not published.

    The regatta publishes the pattern and none of the widths, so anything
    with a number on it has to say whether that number came from the rules
    or from someone who has raced it.
    """
    for lane in traffic.HOCR_LANES:
        assert lane.source in ("rules", "local"), lane.name
        if lane.width is not None:
            assert lane.note, lane.name


def test_the_cambridge_boat_club_bend_is_one_boat_wide():
    """The pinch, and the one width that is known by report only."""
    bend = [lane for lane in traffic.HOCR_LANES
            if lane.name == "Cambridge Boat Club bend"]
    assert len(bend) == 1
    assert bend[0].width == pytest.approx(traffic.ONE_BOAT)
    assert bend[0].source == "local"
    assert bend[0].shore == "Boston"


def test_the_travel_lane_does_not_reach_below_weeks():
    """Below Weeks there is no travel lane; returning crews rejoin the
    ordinary pattern, so nothing may be subtracted down there."""
    lane = [entry for entry in traffic.HOCR_LANES
            if entry.name == "travel lane to the start"][0]
    assert lane.begins == pytest.approx(2278.0)
    assert not lane.contains(1500.0)
    assert lane.contains(3000.0)


def test_the_bend_takes_a_boat_width_out_of_the_course(geometry):
    usable, total, lane = traffic.usable_width(geometry, 3372.0)
    assert lane == pytest.approx(traffic.ONE_BOAT)
    assert usable == pytest.approx(total - traffic.ONE_BOAT)


def test_nothing_is_subtracted_where_no_width_is_known(geometry):
    """Most of the course has a travel lane of unknown width, and an
    unknown width must not be silently treated as zero *or* as a guess."""
    usable, total, lane = traffic.usable_width(geometry, 800.0)
    assert lane == 0.0
    assert usable == pytest.approx(total)


def test_the_course_is_never_narrower_than_a_boat(geometry):
    rows = traffic.lane_report(geometry, step=50.0)
    _, total, _, usable = rows.T
    assert total.min() > EIGHT_ROWED_WIDTH
    assert usable.min() > EIGHT_ROWED_WIDTH


def test_the_narrowest_water_is_in_the_bend_below_eliot(geometry):
    """The tightest point on the course is the one the travel lane also
    squeezes, which is why the two have to be looked at together."""
    rows = traffic.lane_report(geometry, step=25.0)
    metres, total, _, _ = rows.T
    tightest = metres[int(np.argmin(total))]
    assert 3200.0 < tightest < 3700.0
    assert total.min() < 60.0
