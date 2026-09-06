r"""What the stroke sounds like, so a coxswain can feel where it is.

The seat view has no clock and no rate meter a cox would actually look
at, and from the bow of a four you cannot see the blades enter.  What
you *have* is sound: the catch, the run of the seats up the slide, the
release.  Watching the boat without it, there is no way to tell drive
from recovery, which makes calling anything impossible -- and calling is
the thing this is meant to train.

Nothing here is a recording, but the shape of it is measured
-----------------------------------------------------------
The samples are synthesised at start-up: noise taken into the frequency
domain, multiplied by the spectral envelope of a real catch, and brought
back.  No audio is shipped -- only eighteen numbers describing a shape --
so there is no asset to license and nothing of what anyone said on the
water is in the repository.  Timing comes from
:class:`~coxswain.crew.stroke.StrokeTiming`, so a rate change moves the
sounds with it.

The events
----------
``catch``     blade in: the loudest thing in the cycle and the one a crew
              rows to.  Filtered to a spectral envelope measured off a
              real eight, not invented -- see :data:`CATCH_BANDS`.
``release``   blade out: the same water, brighter and much quieter.
``slide``     the seats running up the recovery, looped underneath so the
              cycle is continuous.  Without it the two events sit in
              silence and the result is a stomp rather than a boat.

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


#: Measured spectral envelope of a catch, ``(low Hz, high Hz, dB)``.
#:
#: Averaged over 30 catches from a masters eight
#: (``tools/scan_rowing.py``).  A catch is **two lobes**, not a click: a
#: thump peaking at 120-190 Hz -- the hull and the puddle -- and a splash
#: at 3-5 kHz, rolling off hard above that.  Synthesising broadband noise
#: at a single centre frequency, which is what this used to do, gets
#: neither, and with a percussive envelope on top it came out as a
#: drum-machine stomp.
#:
#: **The 306-3137 Hz band is interpolated, not measured.**  That is
#: exactly where :data:`~tools.scan_rowing.VOICE_BAND` is notched out to
#: keep a coxswain's call from being measured as part of the stroke, so
#: the recording cannot say what is there.  A log-linear ramp between the
#: two lobes is assumed; reproducing the measured hole would put a notch
#: in the synthesis that the real boat does not have.
CATCH_BANDS = (
    (60, 76, -12.4), (76, 96, -9.5), (96, 121, -7.7), (121, 152, 0.0),
    (152, 192, -3.6), (192, 242, -7.7), (242, 306, -13.7),
    # interpolated across the voice notch
    (306, 620, -17.0), (620, 1250, -21.0), (1250, 2100, -24.0),
    (2100, 3137, -26.5),
    (3137, 3959, -27.9), (3959, 4997, -29.1), (4997, 6307, -42.8),
    (6307, 7959, -51.3), (7959, 10045, -57.9), (10045, 12678, -61.7),
    (12678, 16000, -65.4),
)

#: The release is the same water an instant later and much less of it.
RELEASE_TILT = 6.0          # dB a decade, brighter than the catch
RELEASE_LEVEL = 0.30        # and quieter


def _shaped_noise(length: float, bands, decay: float, attack: float,
                  seed: int = 0, tilt: float = 0.0) -> np.ndarray:
    """Noise filtered to a measured spectral envelope, then enveloped.

    This is the whole change in approach: instead of inventing a timbre
    from a centre frequency and a filter order, white noise is taken into
    the frequency domain, multiplied by the shape an actual catch has,
    and brought back.  ``tilt`` adds a slope in dB per decade for the
    release, which is the same event with less water in it.
    """
    n = max(int(RATE * length), 64)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n, 1.0 / RATE)

    centres = np.array([0.5 * (a + b) for a, b, _ in bands], dtype=float)
    levels = np.array([d for _a, _b, d in bands], dtype=float)
    if tilt:
        levels = levels + tilt * np.log10(centres / centres[0])
    # Interpolate in log frequency, which is how the bands were spaced.
    safe = np.maximum(freq, 1.0)
    gain_db = np.interp(np.log10(safe), np.log10(centres), levels,
                        left=levels[0] - 12.0, right=levels[-1] - 12.0)
    wave = np.fft.irfft(spectrum * 10.0 ** (gain_db / 20.0), n=n)

    t = np.arange(n) / RATE
    rise = np.clip(t / max(attack, 1e-6), 0.0, 1.0)
    wave = wave * rise * np.exp(-t * decay)
    return wave / max(np.abs(wave).max(), 1e-9)


def synthesise():
    """``{name: float array in [-1, 1]}`` for the three stroke sounds."""
    # Decay 1.5/s and an attack of a few milliseconds, both measured.
    catch = _shaped_noise(0.90, CATCH_BANDS, decay=1.6, attack=0.007, seed=1)
    release = RELEASE_LEVEL * _shaped_noise(
        0.55, CATCH_BANDS, decay=3.8, attack=0.006, seed=3,
        tilt=RELEASE_TILT)

    # The slide: the same envelope with the lobes flattened out, looped
    # under everything so the cycle is a continuous sound with two events
    # in it rather than two events in silence -- which is what made it a
    # stomp.
    n = int(RATE * 0.5)
    flat = tuple((a, b, min(d, -6.0) * 0.35 - 8.0) for a, b, d in CATCH_BANDS)
    rumble = _shaped_noise(0.5, flat, decay=0.0, attack=0.001, seed=4)
    edge = int(0.02 * RATE)
    ramp = np.ones(n)
    ramp[:edge] = np.linspace(0.0, 1.0, edge)
    ramp[-edge:] = np.linspace(1.0, 0.0, edge)
    slide = 0.5 * rumble[:n] * ramp
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
