r"""The world's own sound: wind, water at the bank, a gull.

The stroke is the subject of the trainer's audio and it stays that way.
This is the scenery behind it -- synthesised rather than recorded, kept
to a fifth of the stroke's level, and following the wind setting so a
dead calm is very nearly silent, because it is.
"""

from __future__ import annotations

import numpy as np

from .strokeaudio import RATE

__all__ = ["AmbientAudio", "synthesise_ambient", "AMBIENT_VOLUME"]

#: Seconds of loop synthesised.  Long enough that the gusting and the
#: lapping do not audibly repeat; short enough to build in a moment.
AMBIENT_SECONDS = 24.0

#: The bed's ceiling against the stroke's 0.85.  Scenery, not subject.
AMBIENT_VOLUME = 0.16


def _one_pole(signal: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """A first-order low-pass, which is enough to turn white into wind."""
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / RATE)
    try:
        from scipy.signal import lfilter

        return lfilter([alpha], [1.0, -(1.0 - alpha)], signal)
    except Exception:                                    # pragma: no cover
        out = np.empty_like(signal)
        acc = 0.0
        for i, v in enumerate(signal):
            acc += alpha * (v - acc)
            out[i] = acc
        return out


def _band(signal: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return _one_pole(signal, high_hz) - _one_pole(signal, low_hz)


def _slow_noise(seconds: float, hz: float, rng) -> np.ndarray:
    """Smooth random modulation at about ``hz``, in [0, 1]."""
    knots = max(int(seconds * hz) + 2, 4)
    values = rng.uniform(0.0, 1.0, knots)
    x = np.linspace(0.0, knots - 1, int(seconds * RATE))
    return np.interp(x, np.arange(knots), values)


def synthesise_ambient(wind_speed: float, seed: int = 7):
    """A seamless loop of wind, water on the bank, and a gull or two.

    Three layers:

    * **wind** -- low-passed noise gusting on a slow random envelope,
      louder and brighter with wind speed.
    * **lapping** -- band-limited noise shaped into soft, irregular
      bursts at about a wash a second: the water working at a bank.
      Present in a calm, more in a breeze.
    * **a gull** -- a quiet formant sweep, once or twice a loop.

    Returns a float array in [-1, 1] at :data:`RATE`.
    """
    rng = np.random.default_rng(seed)
    n = int(AMBIENT_SECONDS * RATE)
    breeze = float(np.clip(wind_speed / 8.0, 0.0, 1.5))

    wind = _one_pole(rng.standard_normal(n), 180.0 + 420.0 * breeze)
    wind /= max(float(np.std(wind)), 1e-9)
    gust = 0.55 + 0.45 * _slow_noise(AMBIENT_SECONDS, 0.18, rng)
    wind *= gust * (0.08 + 0.55 * breeze ** 1.2)

    wash = _band(rng.standard_normal(n), 300.0, 2600.0)
    wash /= max(float(np.std(wash)), 1e-9)
    envelope = np.zeros(n)
    when = 0.0
    while when < AMBIENT_SECONDS:
        start = int(when * RATE)
        length = int(RATE * rng.uniform(1.2, 2.6))
        ramp = np.linspace(0.0, 1.0, length)
        shape = np.sin(np.pi * ramp) ** 1.6 * rng.uniform(0.4, 1.0)
        stop = min(start + length, n)
        envelope[start:stop] = np.maximum(envelope[start:stop],
                                          shape[:stop - start])
        # Slow and irregular, deliberately.  The first version washed
        # every 0.6-1.6 s, which is a stroke rate, and under a crew at
        # 30 it was heard as more stroke -- a second set of blades going
        # in.  Water at a bank is a longer, lazier thing than that, and
        # it never keeps time.
        when += rng.uniform(1.8, 4.5)
    wash *= envelope * (0.05 + 0.13 * breeze)

    gull = np.zeros(n)
    for _ in range(int(rng.integers(1, 3))):
        at = int(rng.uniform(2.0, AMBIENT_SECONDS - 2.0) * RATE)
        length = int(0.42 * RATE)
        tt = np.arange(length) / RATE
        pitch = 2100.0 - 900.0 * (tt / tt[-1])
        cry = np.sin(2.0 * np.pi * np.cumsum(pitch) / RATE)
        cry += 0.4 * np.sin(2.0 * np.pi * np.cumsum(2.0 * pitch) / RATE)
        cry *= np.sin(np.pi * tt / tt[-1]) ** 0.7 * 0.08
        gull[at:at + length] += cry[:n - at]

    mix = wind + wash + gull
    # Seamless.  The last ``fade`` samples are blended INTO the first
    # ``fade`` samples, and the loop is then the middle of the signal
    # followed by that blend: it ends on the head's final sample and
    # starts on the sample after it, so the join is continuous.  (The
    # first version blended and then trimmed half the blend off the
    # end, which closed the loop on the middle of a crossfade -- a
    # click every 24 seconds, which is exactly what this is for.)
    fade = int(0.6 * RATE)
    ramp = np.linspace(0.0, 1.0, fade)
    cross = mix[-fade:] * (1.0 - ramp) + mix[:fade] * ramp
    loop = np.concatenate([mix[fade:-fade], cross])
    # Levelled to a target RMS that follows the breeze, NOT to the peak.
    # Peak-normalising each loop threw the absolute gains away: a dead
    # calm came out within a decibel of a fresh breeze, because both
    # were scaled to the same peak and only their spectra differed.
    # The gains above set the balance between the layers; this sets
    # how loud the whole thing is, and a calm is meant to be quiet.
    target = 0.035 + 0.16 * breeze ** 1.2
    rms = float(np.sqrt(np.mean(loop * loop)))
    loop = loop * (target / max(rms, 1e-9))
    return np.clip(loop, -0.95, 0.95)


class AmbientAudio:
    """The world under the stroke, on its own mixer channel.

    Built with the same guards as :class:`StrokeAudio`: no device, no
    sound, no complaint.  Follows the wind setting live.
    """

    def __init__(self, wind_speed: float = 5.0,
                 volume: float = AMBIENT_VOLUME):
        self.available = False
        self.reason = ""
        self.volume = float(volume)
        self._channel = None
        self._sound = None
        self._wind = float(wind_speed)
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=RATE, size=-16, channels=1,
                                  buffer=512)
            if pygame.mixer.get_num_channels() < 12:
                pygame.mixer.set_num_channels(12)
            self._sound = self._make(pygame)
            self._channel = pygame.mixer.find_channel(True)
            self._channel.set_volume(self._level())
            self._channel.play(self._sound, loops=-1)
            self.available = True
        except Exception as error:
            self.reason = "%s: %s" % (type(error).__name__, error)

    def _make(self, pygame):
        opened = pygame.mixer.get_init()
        channels = int(opened[2]) if opened else 1
        wave = synthesise_ambient(self._wind)
        mono = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        data = (mono if channels == 1
                else np.repeat(mono[:, None], channels, axis=1))
        return pygame.sndarray.make_sound(np.ascontiguousarray(data))

    def _level(self) -> float:
        # Quieter in a dead calm, when there is less to hear.
        return self.volume * (0.55 + 0.45 * min(1.0, self._wind / 8.0))

    def set_wind(self, wind_speed: float) -> None:
        """Follow the wind: a large change rebuilds the loop's spectrum,
        a small one only re-levels, so a slider does not stutter."""
        wind_speed = float(wind_speed)
        if not self.available:
            return
        big_change = abs(wind_speed - self._wind) > 2.5
        self._wind = wind_speed
        if big_change and self._channel is not None:
            try:
                import pygame

                self._sound = self._make(pygame)
                self._channel.play(self._sound, loops=-1)
            except Exception:
                pass
        if self._channel is not None:
            self._channel.set_volume(self._level())

    def stop(self) -> None:
        if self._channel is not None:
            try:
                self._channel.fadeout(400)
            except Exception:
                pass
