r"""A second boat on the course, and what its wake does to your line.

Every line this project has optimised so far assumed an empty river. A
Head of the Charles entry never has one: boats start at a fixed interval
and the water ahead is full of the crew in front.

The geometry that makes this tractable
--------------------------------------
A general two-boat problem is nasty -- the wake you meet depends on where
the leader was when they made it, which depends on their speed history.
One assumption collapses it: **if both boats hold the same speed, the wake
you meet is always the same age.**

Leader starts ``dt`` ahead. You reach station ``s`` at time ``t``; they
reached it at ``t - dt``. So whatever they left at ``s`` has been ageing
for exactly ``dt``, everywhere, for the whole race. The wake field is then
a *static* property of the leader's track, and the follower's problem
becomes a plain line optimisation over a course with an extra drag layer
painted along the leader's path.

That assumption is worth stating loudly because it is doing a lot of work,
and it fails exactly when the race gets interesting -- when you are
actually catching them, ``dt`` is shrinking and the wake is younger and
stronger than this says. It is right for the common case of a crew of
similar speed sitting the same distance ahead all the way down, and it is
optimistic for a crew you are closing on.

What is in the wake, laterally
------------------------------
:class:`~coxswain.hydro.wake.PuddleWake` gives the decay with *age* but
carries no lateral structure at all, because it was built to answer "how
much does sitting directly behind cost". A line-choice question is
entirely about lateral structure, so this module adds it:

**Two puddle lines**, one per side, at the leader's blade track
(:func:`~coxswain.hydro.wake.blade_track`, about 2.5 m out for an eight).
Your blades meet them when *your* blade track lands on *their* puddle
track, which happens at a lateral offset near zero **and** at offsets near
twice the blade track -- so there are three bad places to be, not one.

**One hull wake**, on the leader's centreline, spreading as the
self-similar axisymmetric wake does. This is the term that can *help*
(:meth:`PuddleWake.hull_benefit`), and the module keeps its sign.

References
----------
.. [HOCR] Head Of The Charles Regatta, *Rules and Guidelines*: the Passer
   declares a side within one boat length of open water and the Passee
   must have yielded by half a boat length; failing to yield costs 60 s,
   then 120 s, then disqualification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..hydro.wake import PuddleWake, blade_track

__all__ = ["TrafficWake", "LeadBoat"]

#: HOCR sends boats at roughly this interval; the chute is queued at 2-3
#: lengths of open water but the on-course spacing is set by the starter.
DEFAULT_START_INTERVAL = 15.0


@dataclass
class TrafficWake:
    """Lateral and longitudinal structure of one boat's wake.

    ``puddle`` supplies the decay with age; everything here is about
    *where* the wake is, which that class does not model.
    """

    puddle: PuddleWake
    #: Lateral offset of the leader's puddle lines from their centreline, m.
    track: float = 2.5
    #: Lateral offset of the FOLLOWER's blades from their own centreline, m.
    follower_track: float = 2.5
    #: Extra lateral tolerance on a puddle encounter, m.  A blade is not a
    #: point and neither is a puddle; this is the sum of their half-widths
    #: and it sets how sharply the penalty falls off with offset.
    blade_half_width: float = 0.45
    #: Hull half-beam used for the hull-wake overlap, m.
    hull_half_beam: float = 0.29

    def puddle_radius(self, age: float) -> float:
        """Puddle radius at this age, m -- the wake's own spreading."""
        return float(np.atleast_1d(self.puddle.radius(age))[0])

    def _lateral_weight(self, separation, width):
        """Gaussian overlap factor, 1 on the line and 0 well off it."""
        separation = np.asarray(separation, dtype=float)
        return np.exp(-0.5 * (separation / max(width, 1e-6)) ** 2)

    def blade_exposure(self, offset, age: float):
        """Fraction of the full head-on puddle penalty seen at this offset.

        ``offset`` is the follower's lateral position relative to the
        leader's centreline, positive to port.

        Your two blade tracks sit at ``offset +/- follower_track``; their
        two puddle lines sit at ``+/- track``. Any of the four pairings can
        coincide, and the worst offsets are therefore **0** (both sides
        line up at once) and **+/-(track + follower_track)** (your inside
        blade in their far puddle line).
        """
        offset = np.asarray(offset, dtype=float)
        width = self.puddle_radius(age) + self.blade_half_width
        total = np.zeros_like(offset)
        for mine in (+self.follower_track, -self.follower_track):
            for theirs in (+self.track, -self.track):
                total = total + self._lateral_weight(
                    offset + mine - theirs, width)
        # Two of the four pairings coincide when offset is zero, so
        # normalise by that worst case rather than by the pairing count.
        peak = 0.0
        for mine in (+self.follower_track, -self.follower_track):
            for theirs in (+self.track, -self.track):
                peak += float(np.exp(-0.5 * ((mine - theirs) / width) ** 2))
        return total / max(peak, 1e-9)

    def hull_exposure(self, offset, gap: float):
        """Fraction of the centreline hull-wake benefit seen at this offset.

        The self-similar wake spreads as ``x^(1/3)``, so how far off the
        line you can sit and still feel it grows slowly with distance
        astern.
        """
        offset = np.asarray(offset, dtype=float)
        scale = np.sqrt(self.puddle.momentum_area())
        width = max(scale * (max(gap, 1.0) / scale) ** (1.0 / 3.0),
                    self.hull_half_beam)
        return self._lateral_weight(offset, width)

    def drag_factor(self, offset, gap: float):
        """Multiplier on the follower's resistance at this lateral offset.

        Above one is a penalty, below one a benefit. Both terms are
        present and they act in opposite directions, which is the point:
        directly astern is the worst place for your blades and the best
        place for your hull, and which wins is a quantitative question
        rather than a matter of received wisdom.
        """
        age = max(float(gap), 1.0) / max(self.puddle.speed, 1e-6)
        blades = self.blade_exposure(offset, age)
        hull = self.hull_exposure(offset, gap)
        penalty = blades * float(np.atleast_1d(
            self.puddle.power_penalty(gap))[0])
        benefit = hull * float(np.atleast_1d(
            self.puddle.hull_benefit(gap))[0])
        return 1.0 + penalty - benefit


@dataclass
class LeadBoat:
    """A boat ahead, its track, and the wake field it paints on the river."""

    route: object                      # Route
    course: object
    #: Seconds of head start. With equal speeds this is also the age of
    #: every puddle the follower meets, which is what makes it a static
    #: field rather than a chase.
    interval: float = DEFAULT_START_INTERVAL
    speed: float = 4.23
    wake: Optional[TrafficWake] = None
    _samples: int = 900
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def gap(self) -> float:
        """Metres of water between the boats, from the interval."""
        return float(self.interval) * float(self.speed)

    @classmethod
    def build(cls, route, course, boat, drag: float, interval: float = None,
              speed: float = 4.23, **kwargs):
        """Construct with a wake derived from the leader's own boat."""
        track = blade_track(boat)
        puddle = PuddleWake(drag=float(drag), speed=float(speed),
                            period=float(boat.timing.period),
                            n_blades=boat.n_seats)
        wake = TrafficWake(puddle=puddle, track=track, follower_track=track)
        return cls(route=route, course=course,
                   interval=DEFAULT_START_INTERVAL if interval is None
                   else float(interval),
                   speed=float(speed), wake=wake, **kwargs)

    def offset_at(self, station):
        """The leader's own offset from the course centreline, m."""
        return np.asarray(self.route.offset_at(station), dtype=float)

    def drag_factor_along(self, station, offset):
        """Resistance multiplier for a follower at ``(station, offset)``.

        The leader's track is itself an offset from the centreline, so
        what matters is the **difference** between the two lines, not the
        follower's offset alone. Getting that wrong makes the wake sit on
        the centreline regardless of where the leader actually rowed.
        """
        separation = (np.asarray(offset, dtype=float)
                      - self.offset_at(station))
        return self.wake.drag_factor(separation, self.gap)
