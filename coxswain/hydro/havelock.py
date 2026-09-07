r"""The hull's wave pattern from the free-surface Green's function.

:mod:`coxswain.viz.water` drew the wake as a Kelvin *wedge*: exact in its
angle and its transverse wavelength, empirical in everything else, and
bolted on beside a separately computed near field with no guarantee the
two did not count the same water twice.  This replaces the wedge with
the wave field the hull actually radiates, computed from the same
thin-ship source sheet and the same free-surface Green's function that
:mod:`coxswain.hydro.michell` uses for wave resistance.

One integral, two uses
----------------------
Michell's wave resistance is

.. math::
   R_w = \frac{4\rho g^2}{\pi U^2}\int_1^\infty
         |P(\lambda)|^2 \frac{\lambda^2}{\sqrt{\lambda^2-1}}\,d\lambda,
   \qquad
   P(\lambda) = \iint \frac{\partial b}{\partial x}\,
                e^{k_0\lambda^2 z}\, e^{i k_0 \lambda x}\, dx\,dz

with :math:`k_0 = g/U^2`.  Each :math:`\lambda = \sec\theta` is a free
plane wave travelling at angle :math:`\theta` to the track, with
wavenumber :math:`k_0\sec^2\theta` and components
:math:`(k_0\lambda,\; k_0\lambda\sqrt{\lambda^2-1})` along and across.
:math:`|P|^2` is how much of each the hull makes.  The **elevation** is
those same waves, added up with their phases kept instead of squared
away:

.. math::
   \eta(x,y) = C \int_1^\infty \frac{\lambda^2}{\sqrt{\lambda^2-1}}
   \sum_i D_i(\lambda)\, H(\xi_i - x)\,
   \sin\!\big(k_0\lambda(\xi_i - x)\big)\,
   \cos\!\big(k_0\lambda\sqrt{\lambda^2-1}\,y\big)\, d\lambda

where :math:`D_i` is the depth-integrated source strength of station
:math:`\xi_i` and the Heaviside is the radiation condition: a source
leaves waves *behind* it and none ahead.  The stationary-phase points of
this integral in :math:`\lambda` are the Kelvin cusp -- the 19.47 degree
wedge falls out, it is not put in -- and the :math:`\lambda \to 1` end is
the transverse system with wavelength :math:`2\pi U^2/g`.

The constant
------------
Everything about the *shape* of the pattern is fixed by the integral:
the wedge, the wavelengths, the relative phase of every panel, the
interference between bow and stern, how all of it changes with Froude
number.  The single overall constant :math:`C` is not derived here; it is
fixed by an energy closure -- the wave energy crossing the wedge per
unit length of track equals the wave resistance -- against
:class:`~coxswain.hydro.michell.MichellWave`.  That is the same closure
the wedge amplitude already used, so the scale is continuous with what
was accepted before, and it is tied to a wave resistance that has its
own validation.  The closure is done once per hull and reported.

What this depends on that the near field did not
------------------------------------------------
The near-field pile-up scales as :math:`U^2/g` exactly, which is why it
bakes to one texture.  The wave field does not: :math:`k_0` sets the
wavelengths, so the pattern reshapes with speed.  It is therefore baked
over a set of speeds and interpolated between neighbouring slices at run
time -- a 3-D texture rather than a 2-D one, and still a single lookup.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

GRAVITY = 9.80665
WATER_DENSITY = 1000.0

__all__ = ["HavelockField", "closure_scale", "bake_wave"]


class HavelockField:
    """Free-surface elevation radiated by a thin-ship source sheet."""

    def __init__(self, offsets, stations: int = 81, levels: int = 21,
                 angles: int = 240, decay_cutoff: float = 25.0):
        from .michell import elliptical_offsets

        self.offsets = offsets
        x, z, half = elliptical_offsets(offsets, stations=stations,
                                        levels=levels)
        self.station = np.asarray(x, dtype=float)
        self.level = np.asarray(z, dtype=float)
        half = np.asarray(half, dtype=float)
        # +x is the bow and the flow runs toward -x, so the thin-ship
        # slope (written for a stream along +x) changes sign -- the same
        # correction :mod:`coxswain.hydro.nearfield` needed.
        self.slope = -np.gradient(half, self.station, axis=0)
        self._dx = float(np.mean(np.diff(self.station)))
        self._dz = float(np.mean(np.diff(self.level)))
        self.draft = float(-self.level.min())
        self.angles = int(angles)
        self.decay_cutoff = float(decay_cutoff)

    # -- the lambda quadrature, shared with Michell --------------------
    def _quadrature(self, speed: float):
        """``(lam, weight)`` with ``lam = cosh(u)`` and the draft cutoff.

        Identical to :meth:`MichellWave.resistance`: the integrand dies
        through ``exp(-k0 lam^2 T)`` over the draft, so the range is set
        by the hull rather than truncated; and ``lam = cosh(u)`` turns
        ``lam^2 / sqrt(lam^2 - 1) dlam`` into ``cosh^2(u) du`` and removes
        the singularity at ``lam = 1`` outright.
        """
        k0 = GRAVITY / max(speed, 0.05) ** 2
        lam_max = max(float(np.sqrt(self.decay_cutoff
                                    / max(k0 * self.draft, 1e-9))), 2.0)
        u_max = float(np.arccosh(lam_max))
        u = (np.arange(self.angles) + 0.5) * (u_max / self.angles)
        lam = np.cosh(u)
        weight = np.cosh(u) ** 2 * (u_max / self.angles)
        return k0, lam, weight

    def elevation(self, east, north, speed: float,
                  scale: float = 1.0) -> np.ndarray:
        """Wave elevation on a grid, in the **boat frame**, unscaled.

        ``east`` along the hull (positive forward, origin amidships),
        ``north`` across.  Returns ``(len(north), len(east))``.  The
        result is ``scale`` times the raw integral; see
        :func:`closure_scale` for what ``scale`` should be.
        """
        east = np.asarray(east, dtype=float)
        north = np.asarray(north, dtype=float)
        k0, lam, weight = self._quadrature(speed)
        # Depth-integrated strength of each station at each lambda.
        decay = np.exp(np.clip(k0 * lam[:, None] ** 2 * self.level[None, :],
                               -700.0, 0.0))                # (L, Z)
        strength = np.einsum("lz,xz->lx", decay, self.slope) \
            * self._dz                                       # (L, X)
        across = np.cos(k0 * lam[:, None] * np.sqrt(lam[:, None] ** 2 - 1.0)
                        * north[None, :])                    # (L, Y)
        out = np.zeros((len(north), len(east)))
        for e_index, xe in enumerate(east):
            astern = self.station - xe                       # xi_i - x
            behind = astern > 0.0
            if not behind.any():
                continue
            phase = np.sin(k0 * lam[:, None] * astern[None, behind])  # (L, Xb)
            along = np.einsum("lx,lx->l", strength[:, behind], phase) \
                * self._dx                                   # (L,)
            out[:, e_index] = (weight * along) @ across      # (Y,)
        return scale * out


def closure_scale(field: HavelockField, speed: float,
                  resistance: Optional[float] = None,
                  astern: float = 45.0) -> float:
    r"""The constant that makes the wave energy match the wave resistance.

    The work done against wave resistance over a metre of track is the
    energy left in the wave system per metre of wake.  Across a
    transverse cut far astern, that energy is
    :math:`\int \tfrac12 \rho g\,\overline{\eta^2}\, dy`, the overbar an
    average over one transverse wavelength in :math:`x` so the
    interference pattern does not bias the cut.  Equating the two fixes
    the scale, exactly as the wedge amplitude was fixed before.  With
    ``resistance`` omitted it is taken from :class:`MichellWave`.
    """
    from .michell import MichellWave

    if resistance is None:
        # The hull's own Michell model, from the same offsets -- not a
        # reconstruction from the slope, which would only add error.
        michell = MichellWave.from_offsets(field.offsets)
        resistance = float(michell.resistance(np.array([speed]))[0])
    wavelength = 2.0 * np.pi * speed ** 2 / GRAVITY
    xs = -astern - np.linspace(0.0, wavelength, 9, endpoint=False)
    half_width = 1.6 * np.tan(np.arcsin(1.0 / 3.0)) * astern + 6.0
    ys = np.linspace(-half_width, half_width, 241)
    raw = field.elevation(xs, ys, speed, scale=1.0)
    mean_square = float(np.mean(raw ** 2, axis=1) @ np.ones(len(ys))
                        * (ys[1] - ys[0]))
    energy_per_metre = 0.5 * WATER_DENSITY * GRAVITY * mean_square
    if energy_per_metre <= 0.0:
        return 0.0
    return float(np.sqrt(max(resistance, 0.0) / energy_per_metre))


def bake_wave(boat, speeds=None, reach_astern: float = 110.0,
              reach_ahead: float = 12.0, reach_side: float = 42.0,
              nx: int = 244, ny: int = 84
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray]:
    """``(speeds, east, north, waves, scales)`` for a hull.

    ``waves`` is ``(len(speeds), ny, nx)`` in metres, already scaled by
    the closure at each speed; ``scales`` records the constant used so
    it can be inspected.  Speeds below the lowest slice are treated as
    still water by the caller.
    """
    speeds = (np.linspace(1.5, 7.5, 13) if speeds is None
              else np.asarray(speeds, dtype=float))
    field = HavelockField(boat.offsets)
    half = 0.5 * float(boat.length)
    east = np.linspace(-half - reach_astern, half + reach_ahead, nx)
    north = np.linspace(-reach_side, reach_side, ny)
    waves = np.zeros((len(speeds), ny, nx))
    scales = np.zeros(len(speeds))
    for index, speed in enumerate(speeds):
        scale = closure_scale(field, float(speed))
        scales[index] = scale
        waves[index] = field.elevation(east, north, float(speed),
                                       scale=scale)
    return speeds, east, north, waves, scales
