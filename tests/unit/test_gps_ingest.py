"""GPS ingest: does a track come back as the line it was made from?

The projection from lat/lon to ``(station, offset)`` is the only part of
the ingest that can be wrong in a way nobody notices, because a plausible
number comes out either way.  So it is tested by round trip: build a GPX
from a line whose offset is known exactly, and require the ingest to
recover it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "tools"))

from coxswain.river.charles import CHARLES_ORIGIN, charles_course  # noqa: E402
from coxswain.river.route import Route                             # noqa: E402

ingest = pytest.importorskip("ingest_gps")

EARTH = 6378137.0


@pytest.fixture(scope="module")
def course():
    return charles_course()


@pytest.fixture(scope="module")
def projector():
    return ingest.Projector()


def to_latlon(east, north):
    """Inverse of the local tangent plane, for building test fixtures."""
    lat0, lon0 = CHARLES_ORIGIN
    lat = lat0 + np.degrees(np.asarray(north) / EARTH)
    lon = lon0 + np.degrees(np.asarray(east)
                            / (EARTH * np.cos(np.radians(lat0))))
    return lat, lon


def gpx_for(course, offset, n=300, reverse=False):
    route = Route.constant_offset(course, offset).clip_to_channel(
        course, margin=1.5)
    points = route.path(course, n=n)
    if reverse:
        points = points[::-1]
    lat, lon = to_latlon(points[:, 0], points[:, 1])
    body = ["<?xml version='1.0'?>",
            "<gpx version='1.1' "
            "xmlns='http://www.topografix.com/GPX/1/1'>",
            "<trk><name>offset %+.0f</name><trkseg>" % offset]
    for index, (a, b) in enumerate(zip(lat, lon)):
        body.append("<trkpt lat='%.7f' lon='%.7f'>"
                    "<time>2025-10-18T13:%02d:%02dZ</time></trkpt>"
                    % (a, b, (index * 3) // 60, (index * 3) % 60))
    body += ["</trkseg></trk>", "</gpx>"]
    return "\n".join(body)


def summarise(course, projector, offset, reverse=False):
    parsed = list(ingest.parse_gpx(gpx_for(course, offset, reverse=reverse)))
    assert parsed, "the GPX did not parse"
    name, points = parsed[0]
    return ingest.summarise("test", name, points, projector)


# --------------------------------------------------------------------------
def test_gpx_parses(course):
    parsed = list(ingest.parse_gpx(gpx_for(course, 0.0)))
    assert len(parsed) == 1
    _name, points = parsed[0]
    assert len(points) == 300
    assert all("lat" in p and "lon" in p for p in points)


def test_a_centreline_track_projects_to_zero_offset(course, projector):
    record = summarise(course, projector, 0.0)
    assert record is not None
    assert np.median(record["offset"]) == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize("offset", [15.0, -20.0])
def test_a_known_offset_is_recovered(course, projector, offset):
    """Magnitude and, critically, SIGN.

    A sign error here would silently mirror every track about the
    centreline, and the result would still look like a plausible racing
    line.
    """
    record = summarise(course, projector, offset)
    median = float(np.median(record["offset"]))
    assert np.sign(median) == np.sign(offset)
    # Narrow sections clip the line inboard, so the recovered offset is a
    # little smaller in magnitude than asked for.  That is the channel,
    # not the projection.
    assert abs(median) <= abs(offset) + 0.5
    assert abs(median) > 0.9 * abs(offset)


def test_the_offset_sign_matches_the_route_convention(course, projector):
    """Positive to port, the same as Route, so both plot on one axis."""
    port = float(np.median(summarise(course, projector, 15.0)["offset"]))
    starboard = float(np.median(summarise(course, projector, -15.0)["offset"]))
    assert port > 0.0 > starboard


def test_direction_is_detected(course, projector):
    """A paddle home must not be mistaken for a race piece."""
    assert summarise(course, projector, 0.0)["direction"] == "upstream"
    assert summarise(course, projector, 0.0,
                     reverse=True)["direction"] == "downstream"


def test_the_whole_reach_is_spanned(course, projector):
    record = summarise(course, projector, 0.0)
    assert record["station_span_m"] > 0.9 * course.length


def test_metadata_survives(course, projector):
    record = summarise(course, projector, 0.0)
    assert record["year"] == 2025
    assert record["month"] == 10
    assert record["elapsed_s"] > 0
    assert record["mean_speed_ms"] > 0
    assert record["fraction_on_water"] > 0.9


def test_boat_type_is_left_blank_rather_than_guessed(course, projector):
    """GPS cannot know the boat, and inventing it would be worse than null."""
    record = summarise(course, projector, 0.0)
    assert record["boat_type"] is None
    assert record["crew"] is None


def test_a_track_off_the_water_is_rejected(projector):
    """A run along the towpath is not a rowing line."""
    lat = np.linspace(42.360, 42.368, 60)
    lon = np.full(60, -71.155)          # well outside the channel
    points = [{"lat": float(a), "lon": float(b), "time": None}
              for a, b in zip(lat, lon)]
    assert ingest.summarise("test", "towpath", points, projector) is None
