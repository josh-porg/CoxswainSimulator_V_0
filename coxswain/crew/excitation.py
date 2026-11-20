"""Neural excitation for the multi-clock body: muscle modules on a phase oscillator (review §12 step 4; SOURCES §253).

The rower's segments are to be driven by excitations, not by per-segment time laws, so that the legs -> trunk -> arms
sequence comes out of the segments' own dynamics (docs/NEUROMUSCULAR_CONTROL_REVIEW.md §11c, §12). Two parts:

* **Modules** — excitation waveforms for groups of muscles, from on-water EMG: Pitto, Ertel, Simon, Gauchard & Mornieux
  (2024), Appl. Sci. 14:9055 (CC BY), 12 national-level male scullers at race rate, digitised from Fig. 3 to
  ``data/literature/pitto2024_emg.csv`` (research/crew/pitto2024_emg.py; ±0.02). Each module is the mean of its
  muscles' peak-normalised envelopes, renormalised to peak 1, as a periodic function of the stroke's phase. **The leg
  extensors** are not in Pitto et al.'s set (no knee extensor recorded; blocker B2): since §255 they come from Vieira et
  al. (2020), Sensors 20:1666 (CC BY), the vasti on an **indoor rower** (``data/literature/vieira2020_leg_emg.csv``),
  drive and recovery mapped separately onto the on-water cycle. ``Modules.sources`` carries each module's provenance;
  ``Modules.pitto2024(legs=False)`` gives the on-water modules alone.
* **The phase oscillator** — one phase variable for the stroke, advancing at its own rate and pulled towards zero by
  sensed events (the catch: the handle's load), a CPG used as a feedback predictor (Kuo 2002; Dzeladini, van den Kieboom &
  Ijspeert 2014). Phase resetting, ``phi <- phi - g * wrap(phi - phi_event)``, with gain ``g`` in [0, 1] (0: free
  running; 1: reset to the event). Its period adapts with gain ``h`` towards the interval between events.

Nothing here is fitted to an athlete. The modules' amplitudes are the controller's to set (review §12 step 5, or matched
to power as rung 2 does).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

_LIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "literature")
PITTO2024_FILE = os.path.join(_LIT, "pitto2024_emg.csv")
#: the finish's cycle fraction in Pitto et al.'s Fig. 3 (the dashed line)
PITTO2024_FINISH = 0.521

#: module -> muscles (each recorded by Pitto et al. 2024)
MODULE_MUSCLES: Dict[str, Tuple[str, ...]] = {
    "trunk_extensors": ("thoracic_erector_spinae", "lumbar_erector_spinae", "gluteus_maximus"),
    "arm_pullers": ("biceps_brachii", "brachioradialis", "deltoideus_posterior"),
    "trunk_flexors": ("rectus_abdominis", "external_obliquus", "internal_obliquus"),
    "arm_return": ("deltoideus_anterior",),
    "arm_extensors": ("triceps_brachii_longus",),
}


VIEIRA2020_FILE = os.path.join(_LIT, "vieira2020_leg_emg.csv")
#: the finish's cycle fraction on Vieira et al.'s indoor rower (18-28 spm)
VIEIRA2020_FINISH = 0.379


def vieira2020_leg_extensors(cycle, path: str = VIEIRA2020_FILE) -> np.ndarray:
    """The leg extensors' module on ``cycle`` (the on-water phase, finish at ``PITTO2024_FINISH``): the mean of Vieira et
    al. (2020)'s vastus lateralis and medialis envelopes (indoor rower; research/crew/vieira2020_emg.py), renormalised to
    peak 1, its drive mapped onto the on-water drive and its recovery onto the on-water recovery (a time normalisation
    by phase; the ergometer's drive is 37.9 % of its cycle against 52.1 % on water)."""
    vc, env = _read(path)
    w = np.mean([env["vastus_lateralis"], env["vastus_medialis"]], axis=0)
    w = w / w.max()
    c = np.asarray(cycle, float)
    src = np.where(c <= PITTO2024_FINISH, c / PITTO2024_FINISH * VIEIRA2020_FINISH,
                   VIEIRA2020_FINISH + (c - PITTO2024_FINISH) / (1.0 - PITTO2024_FINISH) * (1.0 - VIEIRA2020_FINISH))
    return np.interp(src, vc, w, period=1.0)


def _read(path=PITTO2024_FILE):
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    cyc = np.array([float(r["cycle"]) for r in rows])
    return cyc, {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "cycle"}


@dataclass(frozen=True)
class Modules:
    """The excitation modules as periodic functions of the stroke phase (0 at the catch)."""
    cycle: np.ndarray = field(repr=False)
    waveforms: Dict[str, np.ndarray] = field(repr=False)
    finish: float = PITTO2024_FINISH
    source: str = "Pitto et al. (2024), Appl. Sci. 14:9055, Fig. 3 (on-water, 12 scullers)"
    #: module -> where its waveform comes from (every module's provenance travels with it)
    sources: Dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def pitto2024(cls, path: str = PITTO2024_FILE, legs: bool = True) -> "Modules":
        """Pitto et al.'s on-water modules; with ``legs``, the leg extensors from Vieira et al. (2020)'s indoor-rowing
        vasti (``vieira2020_leg_extensors``; an **ergometer** source, flagged in ``sources``)."""
        cyc, env = _read(path)
        waves, src = {}, {}
        for m, muscles in MODULE_MUSCLES.items():
            w = np.mean([env[k] for k in muscles], axis=0)
            waves[m] = w / w.max()
            src[m] = "Pitto et al. 2024, on water: " + ", ".join(muscles)
        if legs:
            waves["leg_extensors"] = vieira2020_leg_extensors(cyc)
            src["leg_extensors"] = ("Vieira et al. 2020, Sensors 20:1666, indoor rower (ERGOMETER): vastus lateralis and "
                                    "medialis, drive and recovery mapped separately onto the on-water cycle")
        return cls(cyc, waves, sources=src)

    @property
    def names(self):
        return tuple(self.waveforms)

    def __call__(self, name: str, phase) -> np.ndarray:
        """Module ``name`` at ``phase`` (fraction of the cycle from the catch; wrapped)."""
        if name not in self.waveforms:
            raise KeyError("no module %r (have %s; the leg extensors were not recorded -- docs/MULTI_CLOCK_BODY_PLAN.md B2)"
                           % (name, ", ".join(self.waveforms)))
        return np.interp(np.mod(phase, 1.0), self.cycle, self.waveforms[name], period=1.0)

    def peak_phase(self, name: str) -> float:
        return float(self.cycle[int(np.argmax(self.waveforms[name]))])

    def excitations(self, phase, amplitudes: Dict[str, float]) -> Dict[str, float]:
        """``{module: amplitude x waveform(phase)}`` for the modules named in ``amplitudes``."""
        return {m: float(a) * float(self(m, phase)) for m, a in amplitudes.items()}


def _wrap(x: float) -> float:
    """A phase difference wrapped to [-0.5, 0.5)."""
    return float((x + 0.5) % 1.0 - 0.5)


@dataclass
class PhaseOscillator:
    """One phase variable for the stroke, entrained to sensed events by phase resetting (see the module notes)."""
    period: float
    reset_gain: float = 0.5
    period_gain: float = 0.3
    phase: float = 0.0
    _last_event: float = field(default=None, repr=False)

    def __post_init__(self):
        if not self.period > 0.0:
            raise ValueError("the period must be positive")
        if not (0.0 <= self.reset_gain <= 1.0 and 0.0 <= self.period_gain <= 1.0):
            raise ValueError("the gains lie in [0, 1]")

    def step(self, dt: float) -> float:
        """Advance by ``dt`` (s) at the oscillator's own rate; returns the phase in [0, 1)."""
        self.phase = (self.phase + float(dt) / self.period) % 1.0
        return self.phase

    def event(self, t: float, event_phase: float = 0.0) -> float:
        """A sensed event at time ``t`` whose phase should be ``event_phase`` (the catch: 0): the phase is pulled towards
        it by ``reset_gain``, and the period towards the interval since the last event by ``period_gain``. Returns the
        phase error before the correction."""
        err = _wrap(self.phase - float(event_phase))
        self.phase = (self.phase - self.reset_gain * err) % 1.0
        if self._last_event is not None:
            interval = float(t) - self._last_event
            if interval > 0.0:
                self.period += self.period_gain * (interval - self.period)
        self._last_event = float(t)
        return err
