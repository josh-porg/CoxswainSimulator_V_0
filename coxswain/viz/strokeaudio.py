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
``finish``    the loudest thing in the cycle: the blades coming out and
              feathering in the oarlocks, a hard knock of wood on metal.
              This was called the catch for a while, on the assumption
              that the loudest event must be the one a crew rows to.  A
              coxswain sitting in the boat says otherwise, and the sound
              itself agrees -- a catch is an entry into water and does
              not knock.  Everything the decomposition measures is
              referred to this event, and
              :attr:`StrokeAudio.anchored_phases` moves it into the
              model's cycle, which starts at the catch.
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
                   phases) -> List[Tuple[float, int]]:
    """``(time, index)`` for every stroke event in ``(t0, t1]``.

    ``phases`` are the fractions of the cycle at which events happen,
    **measured per shell class** rather than assumed -- see
    :func:`load_envelope`.  Returns them in time order; an interval
    longer than a stroke returns them all, which is what makes a stall
    recover rather than silently drop a catch.
    """
    if period <= 0.0 or t1 <= t0:
        return []
    found = []
    for index, fraction in enumerate(phases):
        offset = float(fraction) * period
        step = np.floor((t0 - offset) / period) + 1.0
        when = offset + step * period
        while when <= t1:
            if when > t0:
                found.append((float(when), index))
            when += period
    return sorted(found)


#: Where the measured per-shell envelopes live.
ENVELOPE_FILE = "stroke_envelope.npz"


def shell_of(boat) -> str:
    """``"four"`` or ``"eight"`` for a boat, by how many are pulling."""
    seats = int(getattr(boat, "n_seats", 4) or 4)
    return "eight" if seats >= 6 else "four"


def load_envelope(shell: str = "four"):
    """The measured description of one shell's stroke, or ``None``.

    Per shell class: the band edges, the spectrum against stroke phase,
    the phases at which **events** happen, and the spectrum of the bed
    between them -- all recovered by :mod:`tools.dmd_stroke` from real
    recordings, with no voice notch, by phase-locked DMD, and **pooled
    over every recording of that shell** by
    :mod:`tools.build_stroke_envelope`: 14 outings for the four, 3 for
    the eight, at rates from 18 to 40 spm.

    Pooling changed the answer, which is the argument for it.  A single
    outing of the four put its events at phases 0.00, 0.50 and 0.88;
    across fourteen the consensus is 0.00, 0.12 and 0.50, so two of the
    three phases fitted from one recording were that morning's and not
    the boat's.

    The third event is the weak one and is left in knowingly: across the
    fourteen fours it lands anywhere from 0.50 to 0.98, a spread of
    0.17, where the first two agree in almost every recording.  Read the
    catch and the second event as measured, and the third as a
    placeholder for something that is not yet resolved.

    A four and an eight are different boats to listen to and both are
    stored.  The eight is far brighter (980-1237 Hz at -13.9 dB against
    the four's -23.3) and its events fall at different points of the
    cycle, so guessing one from the other sounds like the wrong crew.
    """
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", ENVELOPE_FILE)
    if not os.path.exists(path):
        return None
    blob = np.load(path)
    key = "%s_%%s" % shell
    if (key % "bands") not in blob:
        return None
    return {"bands": blob[key % "bands"],
            "envelope": blob[key % "envelope"].astype(float),
            "phase": blob[key % "phase"].astype(float),
            "events": blob[key % "events"].astype(float),
            "bed": blob[key % "bed"].astype(float),
            "period": float(blob[key % "period"])}


def _from_bands(bands, level, length, decay, attack, seed):
    """Noise shaped to one measured band spectrum, then enveloped."""
    n = max(int(RATE * length), 128)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n, 1.0 / RATE)
    centres = 0.5 * (bands[:, 0] + bands[:, 1])
    db = 20.0 * np.log10(np.maximum(level / max(level.max(), 1e-12), 1e-6))
    gain = np.interp(np.log10(np.maximum(freq, 1.0)), np.log10(centres), db,
                     left=db[0] - 12.0, right=db[-1] - 12.0)
    wave = np.fft.irfft(spectrum * 10.0 ** (gain / 20.0), n=n)
    t = np.arange(n) / RATE
    wave = wave * np.clip(t / max(attack, 1e-6), 0.0, 1.0) * np.exp(-t * decay)
    return wave / max(np.abs(wave).max(), 1e-9)


def synthesise(shell: str = "four"):
    """``{"bed", "events", "phases"}`` for one shell class, or ``None``.

    The events are what multi-resolution DMD found inside the cycle --
    for a four: the catch at phase 0, a second event at the finish where
    the blades come out and are feathered, and a third late in the
    recovery.  Each is synthesised from the spectrum measured **at that
    phase**, so the click at the finish differs from the catch because
    it measured as different, not because it was made to.

    Why mrDMD and not the plain decomposition: within a cycle these are
    transients, and a single stationary spectrum per phase describes a
    transient as a smear.  Splitting the phase axis recursively puts the
    bed at the coarse level and the events at the fine ones, localised
    to where they happen.
    """
    loaded = load_envelope(shell)
    if loaded is None:
        return None
    bands = loaded["bands"]
    envelope = loaded["envelope"]
    phases = loaded["phase"]
    catch_energy = max(envelope[:, 0].sum(), 1e-9)
    clips = []
    for index, at in enumerate(loaded["events"]):
        column = int(np.argmin(np.abs(phases - at)))
        level = envelope[:, column]
        share = float(level.sum() / catch_energy)
        # The catch rings; a feather click is short and sharp.
        decay = 1.6 if index == 0 else 5.5
        span = 0.9 if index == 0 else 0.30
        clips.append(_from_bands(bands, level, span, decay, 0.006,
                                 11 + index) * min(share, 1.0))
    bed = _from_bands(bands, loaded["bed"], 0.5, 0.0, 0.001, 41)
    edge = int(0.02 * RATE)
    ramp = np.ones(len(bed))
    ramp[:edge] = np.linspace(0.0, 1.0, edge)
    ramp[-edge:] = np.linspace(1.0, 0.0, edge)
    floor = float(loaded["bed"].sum() / catch_energy)
    return {"bed": bed * ramp * min(floor, 0.8), "events": clips,
            "phases": loaded["events"]}


def synthesise_cycle(period: float, shell: str = "four", seed: int = 11):
    """One whole stroke as a single clip, spectrum following the phase.

    Where the event synthesis plays a few fixed timbres, this
    overlap-adds a noise grain for every phase point, each shaped to the
    envelope measured there, so the spectrum moves continuously through
    the cycle.

    **The grain length follows the structure**, and that is the whole
    difference from the first version.  A uniform grain of two
    forty-eighths of a stroke is about 110 ms: far wider than a catch,
    so transients came out smeared, and far too stationary elsewhere, so
    the rest came out as undifferentiated hiss -- audibly a good catch
    followed by noise.  The multi-resolution decomposition says which
    phases carry transients, so those get short grains that can hold an
    attack and the bed gets long ones that sound like water rather than
    like a noise generator.

    "Transient" here is measured, not marked: it is how far the envelope
    at a phase stands above the bed spectrum, which is the same quantity
    mrDMD isolates into its fine levels.
    """
    loaded = load_envelope(shell)
    if loaded is None:
        return None
    bands = loaded["bands"]
    envelope = loaded["envelope"]
    bed = loaded["bed"]
    n = max(int(RATE * period), 1024)
    phases = envelope.shape[1]
    centres = 0.5 * (bands[:, 0] + bands[:, 1])
    logc = np.log10(centres)
    rng = np.random.default_rng(seed)

    # How much each phase stands above the bed: 1 at the catch, ~0 in the
    # recovery.  This is what decides the grain.
    floor = max(bed.sum(), 1e-9)
    excess = np.array([max(envelope[:, k].sum() / floor - 1.0, 0.0)
                       for k in range(phases)])
    if excess.max() > 0:
        excess = excess / excess.max()

    #: Grain lengths, seconds: short enough to hold a catch, long enough
    #: that the bed is not a stutter.
    short, long = 0.020, 0.150
    out = np.zeros(n + int(RATE * long) + 16)
    for k in range(phases):
        sharp = float(excess[k])
        span = long + (short - long) * sharp
        grain = max(int(RATE * span), 128)
        level = 20.0 * np.log10(np.maximum(envelope[:, k], 1e-6))
        freq = np.fft.rfftfreq(grain, 1.0 / RATE)
        gain = 10.0 ** (np.interp(np.log10(np.maximum(freq, 1.0)), logc,
                                  level, left=level[0] - 12.0,
                                  right=level[-1] - 12.0) / 20.0)
        piece = np.fft.irfft(np.fft.rfft(rng.standard_normal(grain)) * gain,
                             n=grain)
        # A transient wants an attack and a decay, not a symmetric bell.
        t = np.arange(grain) / RATE
        if sharp > 0.05:
            shape = np.clip(t / 0.004, 0.0, 1.0) * np.exp(-t * (4.0 + 26.0
                                                                * sharp))
        else:
            shape = np.hanning(grain)
        piece = piece * shape
        # Level follows the envelope, so the cycle has the dynamics the
        # measurement found rather than a flat loudness.
        piece = piece / max(np.abs(piece).max(), 1e-9)
        piece = piece * float(envelope[:, k].sum() / max(
            envelope[:, 0].sum(), 1e-9))
        at = int(k * n / phases)
        out[at:at + grain] += piece
    cycle = out[:n] + np.concatenate([
        out[n:], np.zeros(max(n - len(out[n:]), 0))])[:n]
    return (cycle / max(np.abs(cycle).max(), 1e-9)).astype(float)


class StrokeAudio:
    """Plays the stroke through ``pygame.mixer``.

    Degrades to silence rather than failing: a machine with no sound
    device should still run the trainer, so every hardware call is
    guarded and :attr:`available` says what happened.
    """

    def __init__(self, boat, volume: float = 0.85, mode: str = "events"):
        """``mode`` is ``"events"`` or ``"full"``.

        ``"events"`` plays the measured transients at their measured
        phases over a continuous bed.  ``"full"`` plays one clip a stroke
        whose spectrum follows the whole envelope; it is truer to the
        measurement in principle and smears the transients in practice,
        which is why it is not the default.  Switching is one word.
        """
        self.boat = boat
        self.mode = mode
        self.shell = shell_of(boat)
        self.available = False
        self._events = []
        self._phases = []
        self._bed = None
        self._cycle = None
        self._last = None
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=RATE, size=-16, channels=1,
                                  buffer=512)
            opened = pygame.mixer.get_init()
            channels = int(opened[2]) if opened else 1

            def to_sound(wave):
                mono = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
                data = (mono if channels == 1
                        else np.repeat(mono[:, None], channels, axis=1))
                clip = pygame.sndarray.make_sound(np.ascontiguousarray(data))
                clip.set_volume(volume)
                return clip

            built = synthesise(self.shell)
            if built is None:
                self.reason = "no measured envelope for %s" % self.shell
                return
            self._events = [to_sound(w) for w in built["events"]]
            self._phases = list(built["phases"])
            self._bed = to_sound(built["bed"])
            if mode == "full":
                cycle = synthesise_cycle(float(self.timing.period),
                                         self.shell)
                if cycle is not None:
                    self._cycle = to_sound(cycle)
                else:
                    self.mode = "events"
            if self._bed is not None and self.mode != "full":
                self._bed.set_volume(0.0)
                self._bed.play(loops=-1)
            self.available = True
        except Exception as error:                # pragma: no cover
            self.reason = str(error)[:80]

    @property
    def timing(self):
        return self.boat.timing

    @property
    def drive_fraction(self) -> float:
        """Drive as a fraction of the whole cycle."""
        period = float(self.timing.period)
        drive = float(getattr(self.timing, "drive_fraction", 0.0) or 0.0)
        if drive <= 0.0:
            drive = float(getattr(self.timing, "drive_duration", 0.0))                 / max(period, 1e-9)
        return float(min(max(drive, 0.05), 0.95)) or 0.4

    @property
    def anchored_phases(self):
        """The measured event phases, moved into the model's cycle.

        **The loudest transient in a recording is the finish, not the
        catch.**  A coxswain sitting in the boat identifies it as the
        blades coming out and feathering in the oarlocks -- a hard
        wood-on-metal knock -- where the catch is a softer entry into
        water.  ``tools/dmd_stroke.py`` anchors its phase grid on that
        loudest transient, so everything it measures is referred to the
        **finish**.

        The simulator's cycle starts at the catch, so the two frames are
        offset by the drive.  Without this shift the finish sound was
        being played at the model's catch, roughly a third of a stroke
        early -- audible as a boat whose sound does not match its blades.
        """
        return [(float(p) + self.drive_fraction) % 1.0
                for p in self._phases]

    def update(self, t: float) -> List[str]:
        """Play whatever falls between the last call and ``t``.

        Returns the event names fired, so a caller can show them or a
        test can assert on them without a sound card.
        """
        if self._last is None:
            self._last = float(t)
            return []
        period = float(self.timing.period)
        fired = events_between(self._last, float(t), period,
                               self.anchored_phases)
        self._last = float(t)
        names = []
        for _when, index in fired:
            names.append("finish" if index == 0 else "event%d" % index)
            if not self.available:
                continue
            try:
                if self.mode == "full" and self._cycle is not None:
                    if index == 0:
                        self._cycle.stop()
                        self._cycle.play()
                elif index < len(self._events):
                    self._events[index].play()
            except Exception:                     # pragma: no cover
                pass
        return names

    def set_slide_level(self, level: float) -> None:
        """Volume of the bed between the events, ``level`` in ``[0, 1]``."""
        if self._bed is None or not self.available or self.mode == "full":
            return
        try:
            self._bed.set_volume(float(min(max(level, 0.0), 1.0)) * 0.6)
        except Exception:                         # pragma: no cover
            pass

    def stop(self) -> None:
        if self._bed is not None and self.available:
            try:
                self._bed.stop()
            except Exception:                     # pragma: no cover
                pass
        if self._cycle is not None and self.available:
            try:
                self._cycle.stop()
            except Exception:                     # pragma: no cover
                pass
