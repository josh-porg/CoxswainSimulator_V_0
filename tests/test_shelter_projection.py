r"""The wind shelter is the hull's projection into the wind, exactly.

A hull lying beam-on shelters a strip as wide as it is long; bow-on it
shelters a strip as wide as its beam.  In between, the sheltered width
is the shadow the hull rectangle casts on the axis perpendicular to the
wind -- not an interpolation between the two, not a guess.  This checks
the band's half-width against that projection computed from the
rectangle's four corners, over a sweep of headings, and that the band
is centred on the boat and runs along the wind.
"""

from __future__ import annotations

import numpy as np


def _projected_half_width(length, beam, heading, wind_from):
    """Half the hull rectangle's extent across the wind, from its corners."""
    blow = np.array([np.cos(wind_from + np.pi), np.sin(wind_from + np.pi)])
    across = np.array([-blow[1], blow[0]])
    axis = np.array([np.cos(heading), np.sin(heading)])
    side = np.array([-axis[1], axis[0]])
    corners = [sx * 0.5 * length * axis + sy * 0.5 * beam * side
               for sx in (-1, 1) for sy in (-1, 1)]
    extent = [float(c @ across) for c in corners]
    return 0.5 * (max(extent) - min(extent))


def test_the_band_is_the_hulls_projection_across_the_wind():
    from coxswain.viz.water import hull_shelter

    length, beam = 13.4, 0.5
    wind_from = np.radians(200.0)
    blow = np.array([np.cos(wind_from + np.pi), np.sin(wind_from + np.pi)])
    across = np.array([-blow[1], blow[0]])
    for heading in np.radians(np.arange(0.0, 360.0, 15.0)):
        want = _projected_half_width(length, beam, heading, wind_from)
        # Walk across the wind, 12 m downwind of the boat, and find where
        # the shelter stops: the band edge is a 1.5 m ramp, so take the
        # point where the factor is halfway back to 1.
        lateral = np.linspace(-12.0, 12.0, 4801)
        points = 12.0 * blow[None, :] + lateral[:, None] * across[None, :]
        factor = hull_shelter(points[:, 0], points[:, 1], 0.0, 0.0,
                              heading, wind_from, length, beam)
        depth = 1.0 - factor
        deepest = depth.max()
        inside = lateral[depth > 0.5 * deepest]
        got = 0.5 * (inside.max() - inside.min())
        # Half the 1.5 m ramp either side is inside the half-depth
        # contour, which is where the width lands by construction.
        assert abs(got - (want + 0.75)) < 0.12, (
            np.degrees(heading), got, want)
        # Centred on the boat, across the wind.
        assert abs(0.5 * (inside.max() + inside.min())) < 0.05


def test_the_band_runs_downwind_from_the_boat():
    """Walking along the wind through the boat: lee behind, pile ahead."""
    from coxswain.viz.water import hull_shelter

    wind_from = np.radians(200.0)
    blow = np.array([np.cos(wind_from + np.pi), np.sin(wind_from + np.pi)])
    heading = wind_from + np.pi / 2.0                        # beam-on
    along = np.linspace(-30.0, 60.0, 901)
    factor = hull_shelter(along * blow[0], along * blow[1], 0.0, 0.0,
                          heading, wind_from, 13.4, 0.5)
    assert factor[along > 3.0].min() < 0.6      # calm to leeward
    assert factor[along < -1.5].max() > 1.05    # piled to windward
    assert abs(factor[0] - 1.0) < 0.02          # gone far upwind
    assert factor[-1] > 0.9                     # recovered far downwind
