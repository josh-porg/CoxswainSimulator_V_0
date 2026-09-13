r"""Michell's integral in water of finite depth, so depth stops being a factor.

Wave resistance in shallow water is today Michell's integral for *deep*
water multiplied by a depth factor (:mod:`coxswain.hydro.shallow`):
Schlichting's construction to depth Froude 0.92, then three chosen numbers
-- a blend, a cap of 3.0 at ``Fr_h = 1`` and a relaxation back to 1 by 1.6.
On the surveyed Charles at race speed 35-60% of the rowable water sits in
the chosen bands.  A thin-ship integral that knows the bed replaces all
three with the hull's offsets and the depth.

Sretenskii (1937)
-----------------
Printed in Wehausen & Laitone (1960), *Surface Waves*, section 20,
eq. (20.69), with the published errata applied ("in numerator of first
coefficient eliminate c") [WL60]_.  In this module's axes -- ``x`` along the
boat, ``z`` vertical and negative down, ``f(x, z)`` the half-beam on the
centreplane, ``nu = g / U^2``, depth ``h``:

.. math::

    R_w = \frac{2 \rho g}{\pi}
          \int_{\mu_h}^{\infty} \left(P^2 + Q^2\right)
          \sqrt{\frac{\mu}{\mu - \nu \tanh \mu h}} \, d\mu

.. math::

    P + iQ = \iint \frac{\partial f}{\partial x}\,
             \frac{\cosh \mu (z + h)}{\cosh \mu h}\,
             e^{\,i x \sqrt{\nu \mu \tanh \mu h}} \, dx \, dz

``mu_h`` is the nonzero root of ``mu = nu tanh(mu h)`` when ``U^2/gh < 1``
and zero otherwise.  Above the critical speed ``sqrt(g h)`` the root is gone:
the transverse waves cannot keep up with the boat and drop out of the
integral, which is the ``Fr_h > 1`` behaviour the depth factor had to choose.

The deep limit is the check that it is the same integral.  With ``h`` large
the depth weight is ``exp(mu z)``, ``mu_h -> nu``, and ``mu = nu lambda^2``
turns it into

.. math::

    \frac{4 \rho g^2}{\pi U^2} \int_1^\infty
    \frac{\lambda^2}{\sqrt{\lambda^2 - 1}} \left(I^2 + J^2\right) d\lambda

which is W&L eq. (20.68) and exactly what :class:`MichellWave` computes.

Numerics
--------
At ``mu_h`` the square root is singular but integrable.  Substituting
``mu = mu_h + s^2`` removes it: ``d mu = 2 s ds`` and, near the root,
``mu - nu tanh(mu h)`` is proportional to ``s^2``, so the integrand is finite
at ``s = 0``.  The midpoint rule in ``s`` never evaluates there.  The upper
limit is the draft decay :class:`MichellWave` already uses.  Station and
level weights are the parent's, so ``quadrature`` means the same thing.

What it still assumes
---------------------
Everything :class:`MichellWave` does -- thin ship, linear free surface,
inviscid -- plus a flat bed and a steady speed.  It is steady resistance at
the instantaneous speed: Day et al. (2011) found that quasi-steady use is
the larger error near ``Fr_h = 1`` (TRACKING).  No sinkage or trim, which in
shallow water are not small.  An open channel: no banks.

References
----------
.. [WL60] Wehausen, J. V. and Laitone, E. V. (1960) *Surface Waves*,
   Handbuch der Physik IX, section 20, eqs. (20.67)-(20.69), pp. 579-581,
   and the online edition's errata (Regents of the University of
   California, 2002).  After Sretenskii, L. N. (1937).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .michell import GRAVITY, MichellWave

__all__ = ["FiniteDepthMichell", "FiniteDepthWaveTable", "stationary_root",
           "wave_table_for"]

#: Depth Froude numbers each depth row is solved at: dense through
#: critical, where the resistance peak is a few hundredths wide in ``Fr_h``,
#: and dense below it too, where the hull's own humps and hollows go by
#: faster in speed than a 0.05 step resolves (a 0.05 step left 9-13% at
#: ``Fr_h`` 0.575 on the eight).
FROUDE_NODES = np.unique(np.concatenate([
    np.linspace(0.30, 0.80, 41),
    np.linspace(0.80, 1.20, 41),
    np.linspace(1.20, 3.00, 19),
]))

#: Below this depth Froude number the answer is deep water's.  [D11] puts
#: the boundary at 0.5; Sretenskii's integral on the eight was within 0.14%
#: of deep water there, so 0.4 leaves margin.
DEEP_BELOW_FROUDE = 0.4


def stationary_root(nu: float, depth: float) -> float:
    """Nonzero root of ``mu = nu tanh(mu h)``, or 0 when there is none.

    A root exists only when ``nu h > 1``, that is below the critical speed
    ``sqrt(g h)``.  It lies in ``(0, nu)``: the function ``mu - nu tanh(mu h)``
    is negative just above zero and positive at ``nu``.
    """
    nu = float(nu)
    depth = float(depth)
    if nu * depth <= 1.0:
        return 0.0
    low, high = 0.0, nu
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle - nu * np.tanh(middle * depth) < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def depth_weight(mu, level, depth):
    """``cosh(mu (z + h)) / cosh(mu h)`` for ``-h <= z <= 0``, overflow-safe."""
    mu = np.asarray(mu, dtype=float)[:, None]
    level = np.asarray(level, dtype=float)[None, :]
    return (np.exp(mu * level)
            * (1.0 + np.exp(-2.0 * mu * (level + depth)))
            / (1.0 + np.exp(-2.0 * mu * depth)))


@dataclass
class FiniteDepthMichell(MichellWave):
    """Sretenskii's thin-ship wave resistance in water of depth ``depth``."""

    #: Water depth, m.  Must exceed the hull's draft.
    depth: float = 1.0e3

    def __post_init__(self):
        super().__post_init__()
        draft = float(np.max(np.abs(self.level)))
        if not self.depth > draft:
            raise ValueError("depth %.3f m does not clear the draft %.3f m"
                             % (self.depth, draft))

    def resistance(self, speed) -> np.ndarray:
        """Wave resistance at each speed in water of depth ``depth``, N."""
        speeds = np.atleast_1d(np.asarray(speed, dtype=float))
        out = np.zeros(len(speeds))
        draft = float(np.max(np.abs(self.level)))
        depth = float(self.depth)
        self.resolution = np.zeros(len(speeds))
        self.root = np.zeros(len(speeds))

        for index, speed_value in enumerate(speeds):
            if speed_value <= 1e-6:
                continue
            nu = GRAVITY / speed_value ** 2
            mu_h = stationary_root(nu, depth)
            self.root[index] = mu_h
            # The same draft decay the deep-water integral stops at.
            mu_max = max(self.decay_cutoff / max(draft, 1e-9), 4.0 * nu)
            s_max = float(np.sqrt(mu_max - mu_h))
            step = s_max / self.angles
            s = (np.arange(self.angles) + 0.5) * step
            mu = mu_h + s ** 2
            gap = mu - nu * np.tanh(mu * depth)
            jacobian = 2.0 * s * np.sqrt(mu / gap) * step

            wavenumber = np.sqrt(nu * mu * np.tanh(mu * depth))
            self.resolution[index] = (2.0 * np.pi
                                      / max(wavenumber.max() * self._dx,
                                            1e-12))
            phase = wavenumber[:, None] * self.station[None, :]
            weighted = np.einsum("mz,xz,z->mx",
                                 depth_weight(mu, self.level, depth),
                                 self.slope, self._level_weight)
            p = np.einsum("mx,mx,x->m", weighted, np.cos(phase),
                          self._station_weight)
            q = np.einsum("mx,mx,x->m", weighted, np.sin(phase),
                          self._station_weight)
            out[index] = (2.0 * self.density * GRAVITY / np.pi
                          * float(np.sum((p ** 2 + q ** 2) * jacobian)))
        return out


class FiniteDepthWaveTable:
    """One hull's wave resistance against speed and water depth.

    Called with a speed it is the deep-water table, a drop-in for
    :meth:`MichellWave.tabulate`, so anything that only knows deep water
    keeps working.  :meth:`at_depth` answers in water of a given depth from
    Sretenskii's integral, and :func:`coxswain.hydro.resistance.hull_resistance`
    uses it in place of multiplying by the shallow-water factor.

    Tabulated as ``C = R / U^2`` against depth Froude number at depth nodes
    ``h_j = h_0 r^j``, each row solved the first time a depth next to it is
    asked for.  Interpolation between rows is linear in ``log h`` at fixed
    ``Fr_h``: the resistance peak sits just below ``Fr_h = 1`` at every depth,
    so holding ``Fr_h`` fixed keeps it aligned from row to row.  Within a
    row it is linear in ``Fr_h``, on nodes dense through critical.
    """

    def __init__(self, station, level, half_beam, quadrature="trapezoid",
                 depth_ratio: float = 1.10, max_speed: float = 12.0,
                 **kwargs):
        self._arguments = dict(station=station, level=level,
                               half_beam=half_beam, quadrature=quadrature,
                               **kwargs)
        # 301 samples, not tabulate()'s 64.  Linear interpolation between
        # 64 samples (0.119 m/s apart) reads up to 14% off the integral at
        # 2-3 m/s and 1.7% at 3-4, where the hull's humps fall between
        # samples; 301 (0.025 m/s) holds it to 1.05% and 0.08% on the eight.
        # Race speed was never the problem: 0.5% and under above 4 m/s.
        deep = MichellWave(**self._arguments).tabulate(points=301)
        self._deep = deep
        self.speeds = deep.speeds
        self.values = deep.values
        self.draft = float(np.max(np.abs(np.asarray(level, dtype=float))))
        self.depth_ratio = float(depth_ratio)
        self.max_speed = float(max_speed)
        #: The shallowest row.  The integral needs the bed below the keel,
        #: and anything shallower is a boat aground, not a wave problem.
        self.base_depth = 1.25 * self.draft
        self._rows = {}

    def __call__(self, speed):
        return self._deep(speed)

    def node_depth(self, index: int) -> float:
        return self.base_depth * self.depth_ratio ** int(index)

    def _row(self, index: int) -> np.ndarray:
        row = self._rows.get(index)
        if row is None:
            depth = self.node_depth(index)
            speeds = FROUDE_NODES * np.sqrt(GRAVITY * depth)
            wanted = speeds <= 1.25 * self.max_speed
            wanted[:2] = True
            row = np.full(len(speeds), np.nan)
            model = FiniteDepthMichell(depth=depth, **self._arguments)
            row[wanted] = (model.resistance(speeds[wanted])
                           / speeds[wanted] ** 2)
            self._rows[index] = row
        return row

    def at_depth(self, speed, depth) -> float:
        """Wave resistance at ``speed`` in water ``depth`` deep, N."""
        u = abs(float(speed))
        h = float(depth)
        if u <= 1e-6 or not np.isfinite(h):
            return float(self._deep(u))
        h = max(h, self.base_depth)
        froude = u / np.sqrt(GRAVITY * h)
        if froude <= DEEP_BELOW_FROUDE:
            return float(self._deep(u))
        position = np.log(h / self.base_depth) / np.log(self.depth_ratio)
        low = int(np.floor(position))
        weight = position - low
        # Which quantity to hold fixed between the two depth rows.  Through
        # critical the peak sits at a fixed ``Fr_h``, so hold that.  Below
        # it the hull's own humps sit at fixed SPEED and depth changes
        # things gently, so hold the speed: at fixed ``Fr_h`` neighbouring
        # rows are ``sqrt(r)`` apart in speed and land on different humps
        # (+7.7% at ``Fr_h`` 0.5 in 2.58 m on the eight).  Blended over
        # ``Fr_h`` 0.8-0.9.
        blend = float(np.clip((froude - 0.8) / 0.1, 0.0, 1.0))
        blend = blend * blend * (3.0 - 2.0 * blend)
        coefficient = 0.0
        for index, share in ((low, 1.0 - weight), (low + 1, weight)):
            row = self._row(index)
            valid = np.isfinite(row)
            nodes = FROUDE_NODES[valid]
            at_speed = u / np.sqrt(GRAVITY * self.node_depth(index))
            wanted = max(blend * froude + (1.0 - blend) * at_speed,
                         float(nodes[0]))
            if wanted > nodes[-1]:
                # Past the last solved node: far supercritical, where the
                # integral has already fallen back towards deep water.
                return float(self._deep(u))
            coefficient += share * float(np.interp(wanted, nodes,
                                                   row[valid]))
        return coefficient * u * u


_TABLES = {}


def wave_table_for(offsets, stations: int = 641, levels: int = 81,
                   **kwargs) -> FiniteDepthWaveTable:
    """The depth-aware wave table for a hull, built once per hull shape."""
    from .michell import elliptical_offsets

    key = None
    try:
        key = (np.asarray(offsets.station, dtype=float).tobytes(),
               np.asarray(offsets.beam, dtype=float).tobytes(),
               np.asarray(offsets.depth, dtype=float).tobytes(),
               stations, levels, tuple(sorted(kwargs.items())))
    except Exception:                                    # pragma: no cover
        key = None
    if key is not None and key in _TABLES:
        return _TABLES[key]
    x, z, half = elliptical_offsets(offsets, stations=stations, levels=levels)
    table = FiniteDepthWaveTable(station=x, level=z, half_beam=half, **kwargs)
    if key is not None:
        _TABLES[key] = table
    return table
