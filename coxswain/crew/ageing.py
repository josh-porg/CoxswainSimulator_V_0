r"""Physiology that depends on how old the rower is.

What age does NOT change
------------------------
The power.  If a 62-year-old pulls a 22:20 5k then she pulls a 22:20
5k, and scaling that by an age factor would be counting the same thing
twice -- the erg score already *is* the measurement.  Age grading exists
to say how impressive a score is, not what it is, and this module never
touches :func:`~coxswain.viz.rigview.erg_watts`.

What age does change
--------------------
The split between the two things that make up a race.  The
critical-power model has an aerobic asymptote ``CP`` and a finite store
``W'`` that can be spent above it.  A 5 km piece is nearly all CP; the
sprint at the end, and every push a coxswain calls, comes out of ``W'``.
``W'`` falls with age faster than ``CP`` does, so an older crew is not
uniformly slower -- they are closer to *flat*: what they can hold is
most of what they have, and there is less to spend on a move.  That is
the thing a coxswain feels and the thing a single erg score cannot say.

Getting CP from the erg instead of from a book
----------------------------------------------
The trainer used a literature ``CP`` of 302.7 W and ``W'`` of 11.4 kJ
for every crew, which are means for young male athletes and roughly
twice what this project's own Women's Veteran four pulls.  With a 5 km
time and the same two-parameter model, ``CP`` follows:

.. math::

    P_{5k} = CP + W' / t_{5k}
    \quad\Longrightarrow\quad
    CP = P_{5k} - W' / t_{5k}

which is not a new assumption -- it is the model already used for
pacing, read backwards.  For a 20-minute 5k it takes about 10 W off the
erg power, so ``CP`` lands just under it, which is what the definition
says it should.

The number that is not measured
-------------------------------
:data:`W_PRIME_DECLINE_PER_DECADE`.  The literature is consistent about
the *direction* and the rough size -- anaerobic capacity falls with age
while aerobic capacity holds up comparatively well -- but the studies
are on cyclists and runners, in the main, and not on masters rowers.
0.10 a decade from 30 is judgement inside a range the literature
supports; it is in ``docs/TRACKING.md`` with the other placeholders,
and a crew with no ages given gets no ageing at all.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .exertion import ROWER_ANAEROBIC_WORK, ROWER_CRITICAL_POWER

#: The age the literature means describe: young trained adults.
AGE_REFERENCE = 30.0

#: Fraction of ``W'`` lost per decade past :data:`AGE_REFERENCE`.
#: **Judgement**, from the direction and rough magnitude in the
#: literature; nobody here has measured it.  See the module docstring.
W_PRIME_DECLINE_PER_DECADE = 0.10

#: ``W'`` never falls below this fraction of the reference: an eighty
#: year old still has a sprint, and a linear decline taken far enough
#: would say otherwise.
W_PRIME_FLOOR = 0.40


def w_prime_factor(age: Optional[float]) -> float:
    """Multiplier on ``W'`` for a rower of this age.

    ``None`` or a nonsense age gives ``1.0`` -- no age, no ageing.
    Below the reference age it is also 1.0: a twenty-year-old is not
    modelled as having *more* than the reference, because the reference
    already describes trained adults at their peak.
    """
    if age is None:
        return 1.0
    try:
        age = float(age)
    except (TypeError, ValueError):
        return 1.0
    if not 10.0 <= age <= 110.0:
        return 1.0
    decades = max(0.0, (age - AGE_REFERENCE) / 10.0)
    return float(max(W_PRIME_FLOOR,
                     1.0 - W_PRIME_DECLINE_PER_DECADE * decades))


def w_prime_for_age(age: Optional[float],
                    reference: float = ROWER_ANAEROBIC_WORK) -> float:
    """``W'`` in joules for a rower of this age."""
    return float(reference) * w_prime_factor(age)


def critical_power_from_erg(watts: float, seconds: float,
                            w_prime: float = ROWER_ANAEROBIC_WORK) -> float:
    """``CP`` implied by holding ``watts`` for ``seconds``.

    The two-parameter model read backwards; see the module docstring.
    Clamped at half the erg power, which only matters for a piece short
    enough that ``W'`` dominates -- a 5 km never is.
    """
    watts = float(watts)
    seconds = max(float(seconds), 1.0)
    return float(max(0.5 * watts, watts - float(w_prime) / seconds))


def crew_physiology(watts: Sequence[float] = None,
                    seconds: Sequence[float] = None,
                    ages: Sequence[Optional[float]] = None):
    """``(critical_power, w_prime)`` for a crew, per rower averaged.

    Every argument may be ``None`` or contain ``None``, because a lineup
    is typed in by hand and half of it is often blank.  With no erg
    scores this returns the literature pair unchanged, so a boat that
    knows nothing about its crew behaves exactly as it did before.

    The average is over rowers rather than over the boat, because that
    is what the reserve model is per: one rower's ``W'``, spent by one
    rower.  The crew rows together, so the mean is what the whole boat
    can do divided by the number of people doing it.
    """
    ages = list(ages or [])
    factors = [w_prime_factor(a) for a in ages] or [1.0]
    w_prime = float(ROWER_ANAEROBIC_WORK) * float(np.mean(factors))

    pairs = [(float(w), float(s))
             for w, s in zip(watts or [], seconds or [])
             if w and s and float(w) > 0.0 and float(s) > 0.0]
    if not pairs:
        return float(ROWER_CRITICAL_POWER), w_prime
    cps = [critical_power_from_erg(w, s, w_prime) for w, s in pairs]
    return float(np.mean(cps)), w_prime
