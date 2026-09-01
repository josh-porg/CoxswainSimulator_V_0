r"""Wave resistance from Michell's integral, so ``C_D`` stops being a number.

The resistance model carried wave drag as a constant coefficient on the
wetted area.  That is not a small approximation, it is the wrong *shape*:
a constant coefficient makes wave drag a fixed fraction of a ``v^2`` law,
so it can never have the humps and hollows a real hull's wave resistance
has -- and a racing shell is designed to sit in one of the hollows.

Pulman puts a VIII at racing speed at Froude 0.35 and calls that a local
minimum of the wave-drag curve.  A constant coefficient cannot represent
a local minimum of anything.  With ``C_dw = 0.00084`` on the wetted area
the model made wave drag **28% of total** at 6 m/s and, being a fixed
fraction, 28% at every other speed too.

Michell (1898)
--------------
For a thin ship -- and a hull thirty times longer than it is wide is as
thin as ships get -- the wave resistance follows from the hull's own
offsets with no fitted coefficient at all [M1898]_, [T87]_:

.. math::

    R_w = \frac{4\rho g^2}{\pi U^2}
          \int_1^\infty \frac{\lambda^2}{\sqrt{\lambda^2-1}}
          \left(I^2 + J^2\right) \, d\lambda

.. math::

    I(\lambda) = \iint \frac{\partial f}{\partial x}\,
                 e^{\lambda^2 g z / U^2}
                 \cos\!\left(\frac{\lambda g x}{U^2}\right) dx\, dz

with ``J`` the same with a sine, and ``f(x, z)`` the hull half-beam over
the centreplane.  The humps and hollows are not put in: they come out of
the interference between the bow and stern contributions to those
integrals, which is exactly the mechanism Pulman describes.

Why not the two-point shortcut
------------------------------
Bow and stern as two point sources gives interference at
``1/Fn^2 = (2n+1)\pi`` for humps -- Pulman's own equations 11 and 12 --
and it is tempting because it is a one-liner.  It is also useless here:
undamped, it oscillates a hundredfold with speed and puts a masters eight
at 4.23 m/s on a **hump**.  Real hulls have their sources distributed over
the length, and the distribution is what damps the oscillation to the
gentle ripple a real curve shows.  Doing the integral over the actual
offsets is not gold-plating; it is the difference between a curve with the
right shape and one with the wrong one.

What it still assumes
---------------------
Thin-ship linearisation -- the boundary condition is applied on the
centreplane rather than the hull, and the free surface is linearised.  For
a beam-to-length ratio of 1:30 that is about as favourable as the
assumption ever gets.  It is *inviscid*: no boundary layer, so no
sheltering of the stern wave by the wake, which is the main reason
Michell over-predicts for full ships.  For a shell, less so.

References
----------
.. [M1898] Michell, J. H. (1898) *The wave resistance of a ship*,
   Philosophical Magazine 45, 106-123.
.. [T87] Tuck, E. O. (1987) *Wave resistance of thin ships and
   catamarans*, and the review of Michell's formula and its numerical
   evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MichellWave", "wigley_offsets"]

GRAVITY = 9.80665
WATER_DENSITY = 1000.0


def wigley_offsets(length: float = 1.0, beam: float = 0.1,
                   draft: float = 0.0625, stations: int = 81,
                   levels: int = 41):
    """The Wigley parabolic hull -- the standard thin-ship test case.

    .. math::

        f(x, z) = \\frac{B}{2}
                  \\left(1 - \\left(\\frac{2x}{L}\\right)^2\\right)
                  \\left(1 - \\left(\\frac{z}{T}\\right)^2\\right)

    Included because Michell's integral needs validating on a hull whose
    answer is published, and the Wigley form is the one everybody uses for
    exactly that.
    """
    x = np.linspace(-0.5 * length, 0.5 * length, stations)
    z = np.linspace(-draft, 0.0, levels)
    half = (0.5 * beam
            * (1.0 - (2.0 * x[:, None] / length) ** 2)
            * (1.0 - (z[None, :] / draft) ** 2))
    return x, z, half


def elliptical_offsets(offsets, stations: int = 121, levels: int = 41):
    """Centreplane half-beam ``f(x, z)`` from a boat's own offsets.

    The parametric hulls in :mod:`coxswain.hydro.hull` have
    semi-elliptical sections, so the half-beam at depth ``z`` on a station
    of waterline beam ``b`` and draft ``T`` is
    ``(b/2) sqrt(1 - (z/T)^2)``.
    """
    x_raw = np.asarray(offsets.station, dtype=float)
    beam_raw = np.asarray(offsets.beam, dtype=float)
    draft_raw = np.asarray(offsets.depth, dtype=float)

    x = np.linspace(x_raw.min(), x_raw.max(), stations)
    beam = np.interp(x, x_raw, beam_raw)
    draft = np.maximum(np.interp(x, x_raw, draft_raw), 1e-6)
    z = np.linspace(-float(draft_raw.max()), 0.0, levels)

    ratio = np.clip(z[None, :] / draft[:, None], -1.0, 1.0)
    half = 0.5 * beam[:, None] * np.sqrt(np.maximum(1.0 - ratio ** 2, 0.0))
    # Below the local keel there is no hull.
    half = np.where(np.abs(z)[None, :] <= draft[:, None], half, 0.0)
    return x, z, half


@dataclass
class MichellWave:
    """Michell's wave resistance for one hull, as a function of speed."""

    station: np.ndarray            # (nx,) longitudinal, m
    level: np.ndarray              # (nz,) vertical, m, negative down
    half_beam: np.ndarray          # (nx, nz) centreplane offsets, m
    density: float = WATER_DENSITY
    #: Quadrature points in the ``lambda = sec(theta)`` substitution.  The
    #: integrand decays through ``exp(lambda^2 g z / U^2)``, so the tail is
    #: killed by the hull's own draft rather than by truncation.
    angles: int = 400
    #: Integrate lambda only while ``exp(-k0 lambda^2 T)`` is above
    #: ``exp(-decay_cutoff)``.  Beyond that the hull's own draft has
    #: removed the contribution and everything further is quadrature noise.
    decay_cutoff: float = 25.0

    @classmethod
    def from_offsets(cls, offsets, **kwargs):
        x, z, half = elliptical_offsets(offsets)
        return cls(station=x, level=z, half_beam=half, **kwargs)

    @classmethod
    def wigley(cls, **kwargs):
        x, z, half = wigley_offsets()
        return cls(station=x, level=z, half_beam=half, **kwargs)

    def __post_init__(self):
        self.station = np.asarray(self.station, dtype=float)
        self.level = np.asarray(self.level, dtype=float)
        self.half_beam = np.asarray(self.half_beam, dtype=float)
        # d(half-beam)/dx on the centreplane -- the source strength.
        self.slope = np.gradient(self.half_beam, self.station, axis=0)
        self._dx = float(np.mean(np.diff(self.station)))
        self._dz = float(np.mean(np.diff(self.level)))

    def resistance(self, speed) -> np.ndarray:
        """Wave resistance at each speed, N.

        Two things have to be right or the quadrature produces noise that
        looks like physics.

        **The lambda range is physical, not fixed.**  The integrand is
        killed by ``exp(lambda^2 g z / U^2)`` over the draft, so it dies
        beyond ``lambda ~ sqrt(C / (k0 T))``.  Integrating to a fixed
        large lambda instead -- or worse, to infinity through
        ``lambda = sec(theta)`` with evenly spaced theta -- adds a tail
        that is pure discretisation error: the first version did that and
        returned wave resistance coefficients of 18 rather than 0.002,
        with spikes at whatever Froude number the aliasing happened to
        land on.

        **The station grid has to resolve the phase.**  The integrand
        carries ``cos(k0 lambda x)``, so the grid needs several points per
        wavelength at the LARGEST lambda that still contributes.  That is
        checked here and reported, because silently aliasing an
        oscillatory integral is how a wave-resistance curve grows humps
        that are not there.

        The substitution is ``lambda = cosh(u)``, which turns
        ``lambda^2 / sqrt(lambda^2 - 1) d(lambda)`` into ``cosh^2(u) du``
        and removes the singularity at ``lambda = 1`` outright.
        """
        speeds = np.atleast_1d(np.asarray(speed, dtype=float))
        out = np.zeros(len(speeds))
        draft = float(np.max(np.abs(self.level)))
        self.resolution = np.zeros(len(speeds))

        for index, speed_value in enumerate(speeds):
            if speed_value <= 1e-6:
                continue
            k0 = GRAVITY / speed_value ** 2
            # Where the draft decay has removed the integrand entirely.
            lam_max = max(float(np.sqrt(self.decay_cutoff
                                        / max(k0 * draft, 1e-9))), 2.0)
            u_max = float(np.arccosh(lam_max))
            u = (np.arange(self.angles) + 0.5) * (u_max / self.angles)
            lam = np.cosh(u)
            weight = np.cosh(u) ** 2 * (u_max / self.angles)

            # Points per wavelength of the fastest phase we integrate.
            self.resolution[index] = (2.0 * np.pi
                                      / max(k0 * lam.max() * self._dx, 1e-12))

            decay = np.exp(np.clip(
                k0 * lam[:, None] ** 2 * self.level[None, :], -700.0, 0.0))
            phase = k0 * lam[:, None] * self.station[None, :]
            weighted = np.einsum("lz,xz->lx", decay, self.slope)
            cosine = np.einsum("lx,lx->l", weighted, np.cos(phase))
            sine = np.einsum("lx,lx->l", weighted, np.sin(phase))
            integral = (cosine ** 2 + sine ** 2) * (self._dx * self._dz) ** 2
            out[index] = (4.0 * self.density * GRAVITY ** 2
                          / (np.pi * speed_value ** 2)
                          * float(np.sum(integral * weight)))
        return out

    def tabulate(self, low: float = 0.5, high: float = 8.0,
                 points: int = 64):
        """Precompute the curve and return an interpolator, N against m/s.

        The integral takes about a tenth of a second at the resolution it
        needs, and the 6-DOF simulator asks for resistance every
        derivative evaluation.  So it is solved once on a grid and
        interpolated, which is the same trick
        :class:`~coxswain.river.route.RouteEvaluator` uses for the
        shallow-water factor and for the same reason: the curve is smooth
        in speed even though it has humps, and the humps are resolved by
        the grid rather than smoothed by it.
        """
        speeds = np.linspace(float(low), float(high), int(points))
        values = self.resistance(speeds)

        def interpolate(speed):
            return np.interp(np.abs(np.asarray(speed, dtype=float)),
                             speeds, values)

        interpolate.speeds = speeds
        interpolate.values = values
        return interpolate

    def coefficient(self, speed, wetted_area: float) -> np.ndarray:
        """``R_w / (0.5 rho U^2 S)``, the usual non-dimensional form."""
        speeds = np.atleast_1d(np.asarray(speed, dtype=float))
        return (self.resistance(speeds)
                / (0.5 * self.density * speeds ** 2 * float(wetted_area)))

    def froude(self, speed) -> np.ndarray:
        length = float(self.station[-1] - self.station[0])
        return np.atleast_1d(np.asarray(speed, float)) / np.sqrt(GRAVITY
                                                                 * length)
