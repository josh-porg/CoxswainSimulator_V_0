"""OpenStreetMap geometry that is not about any particular course.

One function, and it is here rather than in
:mod:`coxswain.river.seattle` because it is a fact about OpenStreetMap
rather than about Lake Union -- and because the moment a second course
needed it, the choice was to import a Seattle module to process Boston
data or to copy the function.  This project has already been bitten by
the second option: the same shoreline filter existed in the extractor and
in the loader, and fixing either one alone changed nothing.
"""

from __future__ import annotations

import numpy as np

__all__ = ["stitch_rings", "is_closed"]


def stitch_rings(fragments, tolerance: float = 5.0):
    """Join open ways into closed rings.

    OpenStreetMap returns a large lake as a **multipolygon relation whose
    members are open ways**, each a piece of the shoreline.  Running
    point-in-polygon on a fragment treats it as if its two ends were
    joined, which fills whatever that chord happens to enclose.

    That is not a subtle failure.  Doing it to Lake Union produced a
    narrow Y-shape of 0.592 km2 against the lake's real 2.1 km2 -- and it
    survived every numeric check in this module, because a wrong mask
    still yields a plausible-looking lap, fetch and dock fraction.  It
    fell over the moment somebody drew a picture of it.

    ``tolerance`` is in the units of the points given, so metres for
    tangent-plane coordinates and degrees for raw latitude and longitude.
    """
    remaining = [np.asarray(f, dtype=float) for f in fragments
                 if len(f) >= 2]
    rings = []
    while remaining:
        chain = remaining.pop(0)
        extended = True
        while extended and not is_closed(chain, tolerance):
            extended = False
            for index, candidate in enumerate(remaining):
                for piece in (candidate, candidate[::-1]):
                    if np.hypot(*(piece[0] - chain[-1])) <= tolerance:
                        chain = np.vstack([chain, piece[1:]])
                    elif np.hypot(*(piece[-1] - chain[0])) <= tolerance:
                        chain = np.vstack([piece[:-1], chain])
                    else:
                        continue
                    remaining.pop(index)
                    extended = True
                    break
                if extended:
                    break
        if len(chain) >= 4:
            rings.append(chain)
    return rings


def is_closed(ring: np.ndarray, tolerance: float) -> bool:
    return len(ring) >= 4 and np.hypot(*(ring[-1] - ring[0])) <= tolerance
