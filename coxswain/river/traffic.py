"""The regatta's traffic pattern: what the river is shared with.

The racing course is not the whole river.  While a crew races up it,
other crews are coming *down* it to the start, in a travel lane on the
Boston shore, and the two are separated by a line of buoys rather than by
any physical feature.  That matters to a steering model for one reason:
the water a boat is allowed to use is narrower than the water it could
physically use, and it is narrowest exactly where the river bends.

What is published and what is not
---------------------------------
The **pattern is published**.  The regatta's rules state that the course
is bounded by orange buoys to port on the Boston side and an intermittent
line of green buoys to starboard on the Cambridge side, that where there
are no green buoys the Cambridge shore is the boundary, and that returning
crews travel down the Boston side from the finish to just below the Weeks
footbridge.  Blades may cross a buoy; the hull may not.

The **widths are not published**.  No figure for the width of the travel
lane, or of the racing course, appears in the rules, the regatta's
competitor material, or the Charles River Rowing Committee's year-round
traffic pattern.  Buoys are laid each year and their positions are not
surveyed.  So the widths here come from two places, both marked in
:attr:`TrafficLane.source`, and neither is a survey:

``rules``
    Read directly out of the published rules -- extent, which shore, which
    direction.  Reliable.
``local``
    Reported by a competitor who has raced the course.  This is how the
    one-boat pinch at the Cambridge Boat Club bend is known.  It is not
    published anywhere and it is not measured, but it is a first-hand
    observation of a constraint the geometry alone would not reveal: the
    channel there is 53 m wide, which sounds ample and is not, because the
    bend is double-buoyed and most of that width is out of bounds.

The double-buoyed reaches are the reason the two disagree.  At the
Cambridge Boat Club bend below Eliot, and at the Weeks turn, the Boston
side carries *two* buoy lines, orange and white, and the water between
them is out of bounds -- so the navigable width the bathymetry reports is
not the usable width, and no amount of depth data would show it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .bridges import EIGHT_ROWED_WIDTH

__all__ = ["TrafficLane", "HOCR_LANES", "ONE_BOAT", "PASSING_WIDTH",
           "usable_width", "lane_report"]


#: Width of a single rowed eight, metres.  A lane "one boat wide" is this
#: wide, and a crew in it cannot be passed and cannot turn.
ONE_BOAT = EIGHT_ROWED_WIDTH

#: Width needed for one boat to pass another, which the regatta's rules
#: require of the racing course: the passing crew must leave room for both
#: boats to stay inside it.
PASSING_WIDTH = 2.0 * EIGHT_ROWED_WIDTH


@dataclass(frozen=True)
class TrafficLane:
    """A lane of the river, as distance along the racing course.

    ``begins`` and ``ends`` are metres from the start line, measured up the
    course, so they read in the order a racing crew meets them.
    """

    name: str
    shore: str                      # "Boston" or "Cambridge"
    direction: str                  # "up" (racing) or "down" (to the start)
    begins: Optional[float]
    ends: Optional[float]
    width: Optional[float] = None
    source: str = "rules"
    note: str = ""

    def contains(self, metres) -> bool:
        low = -np.inf if self.begins is None else self.begins
        high = np.inf if self.ends is None else self.ends
        return bool(low <= float(metres) <= high)


#: The lanes of the Head of the Charles course.
#:
#: Distances are from the start line off the DeWolfe Boathouse, using the
#: bridge stations the channel model puts at Weeks 2278 m, Anderson 2711 m
#: and Eliot 3979 m.
HOCR_LANES = (
    TrafficLane(
        name="racing course", shore="both", direction="up",
        begins=0.0, ends=None, width=None, source="rules",
        note="orange buoys to port on the Boston side, intermittent green "
             "to starboard on the Cambridge side; where there are no green "
             "buoys the Cambridge shore is the boundary. Blades may cross "
             "a buoy, the hull may not."),
    TrafficLane(
        name="travel lane to the start", shore="Boston", direction="down",
        begins=2278.0, ends=None, width=None, source="rules",
        note="returning crews come down the Boston side from the finish to "
             "just below the Weeks footbridge, without stopping and without "
             "power strokes. Below Weeks there is no travel lane and "
             "returning crews rejoin the ordinary pattern."),
    TrafficLane(
        name="Cambridge Boat Club bend", shore="Boston", direction="down",
        begins=3300.0, ends=3979.0, width=ONE_BOAT, source="local",
        note="the pinch. Double-buoyed orange and white with the water "
             "between the lines out of bounds, on the tightest bend of the "
             "course, where the channel narrows to 50 m at 3372 m. The "
             "travel lane through it is one boat wide: a crew in it cannot "
             "be passed, cannot turn, and cannot give way."),
    TrafficLane(
        name="Weeks turn", shore="Boston", direction="down",
        begins=2100.0, ends=2450.0, width=None, source="rules",
        note="also double-buoyed orange and white on the Boston side. The "
             "abrupt bend crowds racing crews into the centre span and is "
             "where the collisions happen."),
    TrafficLane(
        name="warm-up loop", shore="both", direction="loop",
        begins=None, ends=0.0, width=None, source="rules",
        note="in the basin below the start funnel, worked counter-clockwise. "
             "Power strokes are allowed only from just below Weeks down to "
             "the BU Bridge, and only when it is safe."),
)


def usable_width(geometry, metres, navigable_width: Optional[float] = None):
    """Racing width left at ``metres``, after the travel lane is taken out.

    Returns ``(usable, total, lane)`` in metres.  ``lane`` is the width
    given over to downstream traffic, which is only known where a lane in
    :data:`HOCR_LANES` states one; elsewhere it is ``0.0`` and ``usable``
    is just the navigable width, which will be an over-estimate wherever
    buoys narrow the course without anyone having measured by how much.
    """
    total = (channel_width(geometry, metres) if navigable_width is None
             else float(navigable_width))
    lane = 0.0
    for entry in HOCR_LANES:
        if entry.direction == "down" and entry.width and entry.contains(metres):
            lane = max(lane, float(entry.width))
    return (max(total - lane, 0.0), total, lane)


def channel_width(geometry, metres, half_width: float = 140.0,
                  samples: int = 281) -> float:
    """Navigable width of the channel at ``metres`` from the start line."""
    offsets = np.linspace(-half_width, half_width, samples)
    index = geometry.index_at(metres)
    normal = geometry.normal_at(index)
    points = geometry.line[index][None, :] + offsets[:, None] * normal[None, :]
    inside = np.array([bool(geometry.channel.is_navigable(p[0], p[1]))
                       for p in points])
    return float(np.ptp(offsets[inside])) if inside.any() else 0.0


def lane_report(geometry, step: float = 25.0):
    """Width along the course, as ``(metres, total, lane, usable)`` rows."""
    rows = []
    for metres in np.arange(0.0, geometry.length + step, step):
        metres = float(min(metres, geometry.length))
        total = channel_width(geometry, metres)
        usable, total, lane = usable_width(geometry, metres, total)
        rows.append((metres, total, lane, usable))
    return np.array(rows)
