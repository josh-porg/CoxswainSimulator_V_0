r"""What the stroke sounds like, so a coxswain can feel where it is.

The seat view has no clock and no rate meter a cox would actually look
at, and from the bow of a four you cannot see the blades enter.  What
you *have* is sound: the catch, the run of the seats up the slide, the
release.  Watching the boat without it, there is no way to tell drive
from recovery, which makes calling anything impossible -- and calling is
the thing this is meant to train.

Nothing here is a recording
---------------------------
The samples are synthesised at start-up from noise and decaying
sinusoids.  That is a deliberate trade: a real recording of a four would
sound better, but it would be an asset to license, ship and keep in step
with the rig, and what matters for orientation is **when** the sound
happens and how it is shaped, not its timbre.  Everything is derived
from :class:`~coxswain.crew.stroke.StrokeTiming`, so a rate change moves
the sounds with it.

The events
----------
``catch``     blade in: a short broadband knock, the loudest thing in the
              cycle and the one a crew rows to.
``release``   blade out: lighter, with a little water in it.
``slide``     the seats running up the recovery, a low rumble whose
              level follows how fast the crew is moving.

Phase, not wall clock
---------------------
:func:`events_between` takes the simulated interval a frame covered and
returns the events inside it.  Driving it from simulated time rather
than from the audio callback means the sound stays locked to the physics
when the frame rate wobbles, and it is why this is testable without a
sound card: the timing is arithmetic, and only the playing is hardware.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["events_between", "synthesise", "StrokeAudio"]

#: Sample rate for the synthesised clips, Hz.
RATE = 44100


def events_between(t0: float, t1: float, period: float,
                   drive: float) -> List[Tuple[float, str]]:
    """``(time, name)`` for every stroke event in ``(t0, t1]``.

    ``drive`` is the drive as a fraction of the whole cycle, so the
    release falls at ``drive * period`` after each catch.  Returns them
    in time order; an interval longer than a stroke returns them all,
    which is what makes a stall recover rather than silently drop a
    catch.
    """
    if period <= 0.0 or t1 <= t0:
        return []
    found = []
    drive = min(max(float(drive), 0.05), 0.95)
    for offset, name in ((0.0, "catch"), (drive * period, "release")):
        # First occurrence strictly after t0.
        index = np.floor((t0 - offset) / period) + 1.0
        when = offset + index * period
        while when <= t1:
            if when > t0:
                found.append((float(when), name))
            when += period
    return sorted(found)


def _noise_knock(length: float, tone: float, decay: float,
                 noise: float = 0.7, seed: int = 0) -> np.ndarray:
    """A percussive click: filtered noise plus a decaying tone."""
    n = max(int(RATE * length), 16)
    t = np.arange(n) / RATE
    envelope = np.exp(-t * decay)
    rng = np.random.default_rng(seed)
    grain = rng.standard_normal(n)
    # A cheap one-pole low pass, so it is a knock and not a hiss.
    for _ in range(3):
        grain = np.convolve(grain, np.ones(6) / 6.0, mode="same")
    grain /= max(np.abs(grain).max(), 1e-9)
    body = np.sin(2.0 * np.pi * tone * t)
    return (noise * grain + (1.0 - noise) * body) * envelope


def synthesise():
    """``{name: float array in [-1, 1]}`` for the three stroke sounds."""
    catch = _noise_knock(0.22, tone=150.0, decay=26.0, noise=0.72, seed=1)
    # A touch of water: a second, softer knock just behind the first.
    tail = _noise_knock(0.22, tone=95.0, decay=14.0, noise=0.9, seed=2)
    catch = np.clip(catch + 0.45 * np.roll(tail, int(0.012 * RATE)), -1, 1)

    release = 0.55 * _noise_knock(0.16, tone=320.0, decay=34.0,
                                  noise=0.85, seed=3)

    # The slide: a loop of low rumble, level shaped by the caller.
    n = int(RATE * 0.5)
    t = np.arange(n) / RATE
    rng = np.random.default_rng(4)
    rumble = rng.standard_normal(n)
    for _ in range(9):
        rumble = np.convolve(rumble, np.ones(12) / 12.0, mode="same")
    rumble /= max(np.abs(rumble).max(), 1e-9)
    # Fade the ends into each other so the loop does not tick.
    edge = int(0.02 * RATE)
    ramp = np.ones(n)
    ramp[:edge] = np.linspace(0.0, 1.0, edge)
    ramp[-edge:] = np.linspace(1.0, 0.0, edge)
    slide = 0.30 * rumble * ramp * (0.7 + 0.3 * np.sin(2 * np.pi * 1.5 * t))
    return {"catch": catch, "release": release, "slide": slide}


class StrokeAudio:
    """Plays the stroke through ``pygame.mixer``.

    Degrades to silence rather than failing: a machine with no sound
    device should still run the trainer, so every hardware call is
    guarded and :attr:`available` says what happened.
    """

    def __init__(self, boat, volume: float = 0.85):
        self.boat = boat
        self.available = False
        self._sounds = {}
        self._slide = None
        self._last = None
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=RATE, size=-16, channels=1,
                                  buffer=512)
            # Match whatever the mixer actually opened.  ``pygame.init()``
            # brings the mixer up stereo before we get here, and a mono
            # array is refused outright -- which is how this first came
            # back "unavailable" on a machine with a working sound card.
            opened = pygame.mixer.get_init()
            channels = int(opened[2]) if opened else 1
            clips = synthesise()
            for name, wave in clips.items():
                mono = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
                data = (mono if channels == 1
                        else np.repeat(mono[:, None], channels, axis=1))
                self._sounds[name] = pygame.sndarray.make_sound(
                    np.ascontiguousarray(data))
                self._sounds[name].set_volume(volume)
            self._slide = self._sounds.get("slide")
            if self._slide is not None:
                self._slide.set_volume(0.0)
                self._slide.play(loops=-1)
            self.available = True
        except Exception as error:                # pragma: no cover
            self.reason = str(error)[:80]

    @property
    def timing(self):
        return self.boat.timing

    def update(self, t: float) -> List[str]:
        """Play whatever falls between the last call and ``t``.

        Returns the names fired, so a caller can show them or a test can
        assert on them without a sound card.
        """
        if self._last is None:
            self._last = float(t)
            return []
        period = float(self.timing.period)
        drive = float(getattr(self.timing, "drive_fraction", 0.0) or
                      (float(getattr(self.timing, "drive_duration", 0.0))
                       / max(period, 1e-9)) or 0.4)
        fired = events_between(self._last, float(t), period, drive)
        self._last = float(t)
        names = []
        for _when, name in fired:
            names.append(name)
            clip = self._sounds.get(name)
            if clip is not None and self.available:
                try:
                    clip.play()
                except Exception:                 # pragma: no cover
                    pass
        return names

    def set_slide_level(self, level: float) -> None:
        """Volume of the recovery rumble, ``level`` in ``[0, 1]``."""
        if self._slide is None or not self.available:
            return
        try:
            self._slide.set_volume(float(min(max(level, 0.0), 1.0)) * 0.5)
        except Exception:                         # pragma: no cover
            pass

    def stop(self) -> None:
        if self._slide is not None and self.available:
            try:
                self._slide.stop()
            except Exception:                     # pragma: no cover
                pass
