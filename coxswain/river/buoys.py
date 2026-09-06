"""Turning a regatta's buoys into one-sided limits on a course.

"Keep the red buoys to port and the green to starboard" is a rule about
sides, not widths.  A buoy forbids one side of the lane and says nothing
about the other, so it narrows :attr:`~coxswain.river.course.Course.port_limit`
*or* :attr:`~coxswain.river.course.Course.starboard_limit`, never both.

A line of buoys is a line
-------------------------
The first version of this bound each buoy over a boat length either side
of the mark and nothing in between.  With marks a hundred metres apart on
a turn that left the corridor wide open between them, and the optimiser
did what optimisers do: it threaded between the marks, "saved" 322 s on
Head of the Lake, and would have collected a 10 s penalty per buoy or a
disqualification for two.  Consecutive marks of one colour within
:data:`CHAIN` of each other along the course are now joined, the limit
interpolated by station between them, which is what "keep all the orange
buoys to port" means on the water.

Which buoys count
-----------------
A regatta map carries marks that are not lane limits: marshalling buoys
a kilometre off the course, a line-up chute either side of the start, the
finish gate, and the odd mark set for traffic other than the racing lane.
A buoy sets a limit only if it is within :data:`LIMIT_REACH` of the lane,
outside the :data:`END_ZONE` at either end, and on the side its colour
says.  A mark that contradicts the regatta's own drawn lane is reported
and ignored: the drawn lane is the regatta's statement of where the race
goes, and this code cannot settle which of the two is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

#: How far either side of a lone mark its limit binds, m.
REACH = 40.0
#: Clearance a shell needs off a mark, m -- blade plus nerves.
MARGIN = 6.0
#: Consecutive marks of one colour closer than this along the course are
#: one line, m.  400 because the three orange marks along the Montlake
#: Cut are 357 m apart, and with them unjoined the optimiser put the
#: line 5 m off the north wall between them, on the wrong side of the
#: orange line.
CHAIN = 400.0
#: A mark further off the lane than this is not a lane limit, m.
LIMIT_REACH = 80.0
#: Marks within this of the start or finish are chute and gate, m.
END_ZONE = 100.0
#: Never let a limit close the lane entirely, m.
FLOOR = 2.0


@dataclass
class Mark:
    """One buoy, placed against the lane."""
    keep_to_port: bool
    east: float
    north: float
    #: Index of the nearest lane station.
    index: int
    #: Signed lateral offset from the lane, m; positive to port.
    offset: float
    #: Distance from the lane, m.
    gap: float
    #: Why it does not set a limit, or ``""`` if it does.
    ignored: str = ""

    @property
    def colour(self) -> str:
        return "port" if self.keep_to_port else "starboard"


def place(line: np.ndarray, buoys: np.ndarray, station: np.ndarray,
          limit_reach: float = LIMIT_REACH,
          end_zone: float = END_ZONE) -> List[Mark]:
    """Every buoy against the lane, with the reason any is not a limit.

    ``buoys`` rows are ``(keep_to_port, east, north)``, the layout the
    tracing tools write.
    """
    heading = np.arctan2(np.gradient(line[:, 1]), np.gradient(line[:, 0]))
    marks = []
    for keep_to_port, bx, by in np.asarray(buoys, dtype=float):
        gap = np.hypot(line[:, 0] - bx, line[:, 1] - by)
        index = int(np.argmin(gap))
        normal = np.array([-np.sin(heading[index]), np.cos(heading[index])])
        offset = float(np.dot([bx - line[index, 0], by - line[index, 1]],
                              normal))
        mark = Mark(bool(keep_to_port), float(bx), float(by), index, offset,
                    float(gap[index]))
        if gap[index] >= limit_reach:
            mark.ignored = "off the lane"
        elif not end_zone < station[index] < station[-1] - end_zone:
            mark.ignored = "start chute or finish gate"
        elif (offset > 0) != bool(keep_to_port):
            mark.ignored = "on the wrong side of the drawn lane"
        marks.append(mark)
    return marks


def one_sided_limits(line: np.ndarray, half: np.ndarray, buoys: np.ndarray,
                     reach: float = REACH, margin: float = MARGIN,
                     chain: float = CHAIN, limit_reach: float = LIMIT_REACH,
                     end_zone: float = END_ZONE
                     ) -> Tuple[np.ndarray, np.ndarray, List[Mark]]:
    """``(port_limit, starboard_limit, marks)`` for a lane and its buoys.

    ``half`` is the corridor half-width the limits start from.  Each
    limit that a mark sets is the mark's lateral offset less ``margin``;
    a lone mark binds over ``reach`` either side of itself, and marks of
    one colour within ``chain`` of each other along the course bind
    continuously between them.
    """
    line = np.asarray(line, dtype=float)
    station = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(line, axis=0).T))])
    port = np.array(half, dtype=float).copy()
    starboard = np.array(half, dtype=float).copy()
    marks = place(line, buoys, station, limit_reach, end_zone)

    for keep_to_port in (True, False):
        # Limits are distances to the side they bound, so a port mark's
        # offset is used as is and a starboard mark's is negated.
        sign = 1.0 if keep_to_port else -1.0
        target = port if keep_to_port else starboard
        active = sorted((m for m in marks
                         if m.keep_to_port == keep_to_port and not m.ignored),
                        key=lambda m: station[m.index])
        for mark in active:
            near = np.abs(station - station[mark.index]) < reach
            target[near] = np.minimum(target[near],
                                      sign * mark.offset - margin)
        for before, after in zip(active[:-1], active[1:]):
            s0, s1 = station[before.index], station[after.index]
            if s1 - s0 >= chain:
                continue
            between = (station >= s0) & (station <= s1)
            along = np.interp(station[between], [s0, s1],
                              [sign * before.offset, sign * after.offset])
            target[between] = np.minimum(target[between], along - margin)

    return (np.maximum(port, FLOOR), np.maximum(starboard, FLOOR), marks)


def summary(marks: List[Mark]) -> str:
    """One line for a script to print."""
    used = sum(1 for m in marks if not m.ignored)
    reasons = {}
    for m in marks:
        if m.ignored:
            reasons[m.ignored] = reasons.get(m.ignored, 0) + 1
    detail = ", ".join("%d %s" % (n, why) for why, n in sorted(reasons.items()))
    return "%d of %d buoys set a limit%s" % (
        used, len(marks), (" (" + detail + ")") if detail else "")
