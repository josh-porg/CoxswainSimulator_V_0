r"""A water surface the boat sits in, and disturbs.

Two things are happening on the water and they are physically different,
so they are computed differently.

The sea state
-------------
Wind chop is a **random** surface, and the honest description of one is a
spectrum rather than a shape.  :class:`~coxswain.hydro.chop.FetchLimitedSea`
already turns wind speed and fetch into a significant height and a peak
period by the JONSWAP relations, and that module is used here rather than
reimplemented: the trainer's water is the same water the conditions
analysis reports on, so a 14 m/s day looks like what the report says it
is.  :class:`WaveField` samples that spectrum into a handful of
directional components whose sum is the surface.

Linear superposition is not an approximation of convenience here -- to
first order in wave steepness the free-surface problem *is* linear, and a
sum of components each satisfying it satisfies it too.  It stops being
true when the waves get steep, and :attr:`FetchLimitedSea.steepness`
says when: above about 1/7 a wave breaks, and none of this applies.

What the boat does to it
------------------------
A hull moving at speed drags a **Kelvin wake**, and that is not random at
all -- it is a stationary-phase superposition of the same elementary
waves, and it has exact geometry:

* the wake is confined to a wedge of half-angle
  :data:`KELVIN_HALF_ANGLE` = arcsin(1/3) = 19.47 degrees, and this is
  independent of speed, which is why every boat's wake looks the same
  shape;
* the transverse waves inside it have wavelength
  :math:`2\pi V^2 / g` exactly, which is why a fast boat's wake is
  longer-waved;
* amplitude falls off along the wedge, roughly as one over the square
  root of distance, because the energy spreads.

So the wake is drawn from those, not from a texture -- and it responds to
the boat's actual speed each frame, which means a crew can *see* the
puddles and the wake stretch when they lengthen the stroke.

The puddles
-----------
Each catch drops a pair of vortices where the blades went in, and they
sit there and decay while the boat runs away from them.
:class:`~coxswain.hydro.wake.PuddleWake` already models their strength
for the passing study; here they are placed at the catch -- the same
stroke phase that fires the catch in :mod:`coxswain.viz.strokeaudio` --
and faded on the same timescale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["KELVIN_HALF_ANGLE", "WaveField", "kelvin_wavelength",
           "PuddleTrail", "sea_for", "wake_amplitude", "wake_table",
           "hull_wake", "load_nearfield", "hull_shelter"]

GRAVITY = 9.80665

#: Half-angle of the Kelvin wedge, radians.  ``arcsin(1/3)``: a constant
#: of the deep-water dispersion relation, not of the boat.
KELVIN_HALF_ANGLE = float(np.arcsin(1.0 / 3.0))

#: Components used to represent the sea.  Eight is enough that the
#: surface does not visibly repeat over a boat length and few enough to
#: evaluate per vertex on an Intel iGPU.
COMPONENTS = 8

#: Spread of component directions either side of the wind, radians.  Wind
#: sea is not unidirectional; this is the usual cosine-squared spread
#: truncated to something a handful of components can represent.
DIRECTION_SPREAD = 0.9


def kelvin_wavelength(speed: float) -> float:
    """Transverse wake wavelength, m: ``2 pi V^2 / g``, exactly."""
    return float(2.0 * np.pi * max(speed, 0.05) ** 2 / GRAVITY)


@dataclass(frozen=True)
class WaveField:
    """A sea state as a handful of superposed components.

    ``amplitude``, ``wavenumber``, ``direction`` and ``phase`` are
    parallel arrays; the surface at ``(x, y, t)`` is the sum over
    components of ``a cos(k (x cos d + y sin d) - w t + p)`` with
    ``w = sqrt(g k)`` -- the deep-water dispersion relation, which is the
    same one :mod:`coxswain.hydro.chop` assumes and checks.
    """

    amplitude: np.ndarray
    wavenumber: np.ndarray
    direction: np.ndarray
    phase: np.ndarray
    significant_height: float = 0.0
    peak_period: float = 0.0

    @property
    def frequency(self) -> np.ndarray:
        """Angular frequency of each component, rad/s."""
        return np.sqrt(GRAVITY * self.wavenumber)

    def height_at(self, east, north, t: float):
        """Surface elevation, m.  Vectorised over ``east``/``north``."""
        east = np.asarray(east, dtype=float)
        north = np.asarray(north, dtype=float)
        total = np.zeros(np.broadcast(east, north).shape)
        for a, k, d, p, w in zip(self.amplitude, self.wavenumber,
                                 self.direction, self.phase, self.frequency):
            total = total + a * np.cos(k * (east * np.cos(d)
                                            + north * np.sin(d)) - w * t + p)
        return total

    def as_uniform(self):
        """``(COMPONENTS, 4)`` of ``(amplitude, k, direction, phase)``.

        Laid out for a shader, which evaluates the same sum per vertex.
        """
        rows = np.zeros((COMPONENTS, 4), dtype="f4")
        count = min(COMPONENTS, len(self.amplitude))
        rows[:count, 0] = self.amplitude[:count]
        rows[:count, 1] = self.wavenumber[:count]
        rows[:count, 2] = self.direction[:count]
        rows[:count, 3] = self.phase[:count]
        return rows


def sea_for(wind: float, fetch: float, bearing: float = 0.0,
            seed: int = 7) -> WaveField:
    """Build a :class:`WaveField` from wind and fetch.

    Goes through :class:`~coxswain.hydro.chop.FetchLimitedSea` so the
    height and period are the analysis's numbers, then distributes the
    energy over :data:`COMPONENTS` components around the peak.  The
    component amplitudes are scaled so the sum has the right significant
    height: for a narrow-band sea ``H_s = 4 sqrt(m0)`` and ``m0`` is the
    sum of ``a^2 / 2``, which fixes the scale without a free parameter.
    """
    from ..hydro.chop import FetchLimitedSea

    sea = FetchLimitedSea(wind=float(wind), fetch=float(fetch))
    height = sea.significant_height
    period = sea.peak_period
    if height <= 1e-4 or period <= 1e-3:
        empty = np.zeros(0)
        return WaveField(empty, empty, empty, empty, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    peak_k = (2.0 * np.pi / period) ** 2 / GRAVITY
    # Spread either side of the peak: half an octave each way is enough
    # to stop the surface looking like a single sine.
    scale = np.geomspace(0.55, 1.9, COMPONENTS)
    wavenumber = peak_k * scale
    direction = bearing + np.linspace(-DIRECTION_SPREAD, DIRECTION_SPREAD,
                                      COMPONENTS)
    # JONSWAP-ish weighting about the peak, then normalised to H_s.
    weight = np.exp(-1.25 * (scale ** -2)) * scale ** -2.5
    weight /= weight.sum()
    variance = (height / 4.0) ** 2
    amplitude = np.sqrt(2.0 * variance * weight)
    phase = rng.uniform(0.0, 2.0 * np.pi, COMPONENTS)
    return WaveField(amplitude, wavenumber, direction, phase,
                     significant_height=height, peak_period=period)


def load_nearfield(shell: str = "four"):
    """``(east, north, F)`` for a shell, or ``None`` if not baked.

    ``F`` is the geometric part of the hull's own surface disturbance:
    multiply by ``U^2 / g`` for metres.  Produced by
    ``tools/bake_nearfield.py`` from :mod:`coxswain.hydro.nearfield`.
    """
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "nearfield.npz")
    if not os.path.exists(path):
        return None
    blob = np.load(path)
    if ("%s_field" % shell) not in blob:
        return None
    return (blob["%s_east" % shell], blob["%s_north" % shell],
            blob["%s_field" % shell])


def wake_amplitude(wave_resistance: float, half_angle: float = None) -> float:
    r"""Wake amplitude coefficient ``A``, where the crest height is
    ``A / sqrt(x)`` at ``x`` metres astern.

    Derived rather than chosen.  The work the hull does against **wave
    resistance** over a metre of track is exactly the energy it leaves
    behind in the wave system, so the energy per unit length of wake is
    :math:`R_w`.  That energy is spread across the Kelvin wedge, whose
    width at ``x`` astern is :math:`2 x 	an	heta`, and a deep-water
    wave of amplitude ``a`` carries :math:`	frac12 
ho g a^2` per unit
    area.  Equating them,

    .. math::
        	frac12 
ho g a^2 \cdot 2 x 	an	heta = R_w
        \quad\Rightarrow\quad
        a(x) = \sqrt{rac{R_w}{
ho g 	an	heta}} \; x^{-1/2}

    which also **predicts the one-over-root-x falloff** that was
    previously put in by hand, so the shape and the scale now come from
    the same place.

    ``wave_resistance`` comes from
    :class:`~coxswain.hydro.michell.MichellWave` -- the same thin-ship
    theory the drag validation uses.  For this four it runs 5.7 N at
    2 m/s to 38 N at 6, giving crests of 1.3 to 3.3 cm ten metres
    astern; the constant it replaced was about 40% larger.
    """
    if half_angle is None:
        half_angle = KELVIN_HALF_ANGLE
    denominator = 1000.0 * GRAVITY * np.tan(half_angle)
    return float(np.sqrt(max(float(wave_resistance), 0.0) / denominator))


def wake_table(boat, speeds=None):
    """``(speeds, amplitude)`` for a hull, ready to interpolate per frame.

    Michell's integral is not expensive but it is not free either, and
    the wake only depends on speed, so it is tabulated once.
    """
    from ..hydro.michell import MichellWave

    speeds = (np.linspace(0.0, 8.0, 33) if speeds is None
              else np.asarray(speeds, dtype=float))
    model = MichellWave.from_offsets(boat.offsets)
    safe = np.maximum(speeds, 0.05)
    resistance = np.asarray(model.resistance(safe), dtype=float)
    return speeds, np.array([wake_amplitude(r) for r in resistance])


def kelvin_height(along, across, speed: float, amplitude: float = 0.06,
                  reach: float = 60.0):
    """Wake elevation behind a hull, in the **boat frame**.

    ``along`` is distance astern (positive behind the transom) and
    ``across`` is lateral offset.  Outside the Kelvin wedge the result is
    zero, which is the one thing about a wake everybody recognises.

    The transverse system is exact in wavelength; the divergent system is
    represented by the wedge envelope rather than resolved, because
    resolving it needs the stationary-phase integral and at the size a
    wake appears in a seat view the envelope is what reads.
    """
    along = np.asarray(along, dtype=float)
    across = np.asarray(across, dtype=float)
    inside = (along > 0.2) & (np.abs(across)
                              <= np.tan(KELVIN_HALF_ANGLE) * along)
    if not np.any(inside):
        return np.zeros(np.broadcast(along, across).shape)
    wavelength = kelvin_wavelength(speed)
    k = 2.0 * np.pi / max(wavelength, 0.5)
    # Energy spreads along the wedge and the pattern fades astern.
    fade = np.exp(-along / max(reach, 1.0)) / np.sqrt(np.maximum(along, 0.5))
    # Ride up toward the cusp lines, which is where a real wake is
    # steepest and where the eye picks the wedge out.
    edge = np.abs(across) / np.maximum(np.tan(KELVIN_HALF_ANGLE) * along, 1e-6)
    crest = 0.45 + 0.55 * edge ** 2
    return np.where(inside,
                    amplitude * fade * crest * np.cos(k * along), 0.0)


def hull_wake(along, across, speed: float, amplitude: float,
              length: float, stern_share: float = 0.55):
    """The wake of a **hull**, not of a point: bow source and stern sink.

    ``along`` is distance astern of the boat's centre and ``across`` the
    lateral offset, both in the boat frame.

    A single Kelvin system centred on the boat has no V leaving the stem
    -- its wedge simply begins under the hull.  What a coxswain sees is a
    pair of crests thrown from the bow, running slightly wider than the
    hull and opening out behind it.  Havelock's model of a ship is a
    pressure **source at the bow** and a **sink at the stern**, separated
    by the waterline length; superposing their two Kelvin systems gives
    the bow V, and the interference between them is what puts the humps
    into a wave-resistance curve against Froude number.

    So the two wedge apexes sit a hull length apart, which is the check
    in the tests: one at the bow and one at the stern, not one at the
    middle.
    """
    half = 0.5 * float(length)
    bow = kelvin_height(np.asarray(along) - half, across, speed,
                        amplitude=amplitude)
    stern = kelvin_height(np.asarray(along) + half, across, speed,
                          amplitude=stern_share * amplitude)
    return bow - stern


#: How much of the short-wave energy the hull and crew take out of the
#: air directly to leeward, and over what distance it comes back, m.
SHELTER_DEPTH = 0.55
SHELTER_RECOVERY = 14.0
#: Amplitude gain against the windward side, and how far it reaches, m.
WINDWARD_GAIN = 0.35
WINDWARD_REACH = 1.6


def hull_shelter(east, north, boat_east, boat_north, heading, wind_from,
                 length, beam):
    """Factor on the local wave amplitude near the hull, from wind shelter.

    A shell sitting in wind chop does two things to the water around it,
    and neither is wave diffraction: at 0.5 m of beam against a two-metre
    wave the hull is very nearly transparent, so scattering is not the
    mechanism.

    What it does do is **block the wind**.  The hull, the riggers and
    four bodies stand about a metre out of the water, and short wind
    waves are sustained by the wind acting on them continuously -- take
    the air away and they decay within metres.  That is the patch of calm
    that sits to leeward of a boat.  It recovers downwind over
    :data:`SHELTER_RECOVERY`, as the wind re-establishes the short-wave
    field over its own fetch.

    To windward the surface piles against the side: partly the waves
    meeting an obstacle, partly the hull's own displacement.  It is a
    much smaller effect than the lee and reaches only
    :data:`WINDWARD_REACH`.

    **The shadow's width depends on the boat's aspect to the wind.**  A
    hull lying beam-on to the breeze shelters a strip as long as it is;
    bow-on it shelters almost nothing.  That is the projected width of
    the hull across the wind, and it is what makes the effect look right
    when a crew turns.
    """
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    # Unit vector the wind blows TOWARD.
    blow = np.array([np.cos(wind_from + np.pi), np.sin(wind_from + np.pi)])
    axis = np.array([np.cos(heading), np.sin(heading)])
    across_wind = abs(axis[0] * blow[1] - axis[1] * blow[0])   # |a x w|
    along_wind = abs(axis[0] * blow[0] + axis[1] * blow[1])    # |a . w|
    half_width = 0.5 * (length * across_wind + beam * along_wind)

    dx = east - float(boat_east)
    dy = north - float(boat_north)
    downwind = dx * blow[0] + dy * blow[1]
    lateral = np.abs(-dx * blow[1] + dy * blow[0])

    inside = np.clip(1.0 - (lateral - half_width) / 1.5, 0.0, 1.0)
    lee = np.where(downwind > 0.0,
                   SHELTER_DEPTH * np.exp(-downwind / SHELTER_RECOVERY), 0.0)
    windward = np.where(downwind < 0.0,
                        WINDWARD_GAIN * np.exp(downwind / WINDWARD_REACH),
                        0.0)
    return 1.0 - inside * lee + inside * windward


@dataclass
class PuddleTrail:
    """Where the blades went in, and how long ago.

    A ring buffer, because this is read by a shader every frame and the
    count has to be fixed.  Puddles are dropped at the catch -- the same
    phase that fires the catch sound -- and fade over
    :attr:`lifetime`, which is set from
    :class:`~coxswain.hydro.wake.PuddleWake`'s decay rather than picked
    to look right.
    """

    capacity: int = 16
    lifetime: float = 7.0
    points: List[Tuple[float, float, float]] = field(default_factory=list)

    def drop(self, east: float, north: float, t: float) -> None:
        self.points.append((float(east), float(north), float(t)))
        if len(self.points) > self.capacity:
            self.points = self.points[-self.capacity:]

    def as_uniform(self, now: float):
        """``(capacity, 4)`` of ``(east, north, age fraction, unused)``."""
        rows = np.zeros((self.capacity, 4), dtype="f4")
        for slot, (east, north, when) in enumerate(self.points[-self.capacity:]):
            age = (now - when) / max(self.lifetime, 1e-6)
            if age < 0.0 or age > 1.0:
                continue
            rows[slot] = (east, north, 1.0 - age, 0.0)
        return rows
