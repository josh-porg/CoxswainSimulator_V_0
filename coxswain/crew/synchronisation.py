"""Crew synchronisation as coupled oscillators.

Section 21 gave every rower their own stroke phase but left those phases
as inputs -- prescribed offsets, or draws from a distribution.  Real crews
do not have prescribed phases: they *converge* on one, imperfectly, and
that convergence is a dynamical process with its own stability, its own
timescale, and its own failure modes.

This is the framework Naomi Ehrich Leonard's group works in [L-PUB]: phase
models of coupled oscillators and their connection to collective motion,
including a collaboration using improvisational dance as a model system
for in-the-moment collective decision making.  The transfer to rowing is
unusually clean, and it goes both ways.

Two coupling channels, not one
------------------------------
Most human-synchronisation models have a single perceptual coupling.  A
rowing crew has two, and they are physically different:

**Mechanical.**  Rower *i*'s motion moves the shell, and every other rower
is rigidly attached to that shell.  This is involuntary, immediate, and
computable -- the hull acceleration is already a state of this simulator.
It is a genuine mean-field term through a *shared rigid body*, not a
modelling convenience, which is what a rowing crew contributes back to the
coupled-oscillator literature rather than merely borrowing from it.

**Sensory.**  Rowers watch the blade or the back in front of them and hear
the catch.  This is voluntary, *delayed* by reaction time, and
**directional**: a rower faces the stern and can only see the crew ahead
of them in that direction.  The coupling graph is therefore a directed
chain from the stroke seat toward bow, not all-to-all.

Which dominates is not established.  ``sensory_gain`` and ``hull_gain``
are separate so the question can be answered rather than assumed.

Two phases, not one
-------------------
An early extraction changes the vertical timing without changing the
horizontal sweep, and the two reach the boat through different terms:

* ``phi`` -- horizontal: oar sweep angle and handle force; surge and yaw;
* ``psi`` -- vertical: blade immersion and extraction; roll and blade
  forces.

A rower who washes out early has ``psi`` leading ``phi``.  That is a
common fault, invisible to a single-phase model, and a roll disturbance
arriving exactly when §15-16 say roll authority is lowest.

Why the delay matters
---------------------
Reaction time is 150-250 ms.  §15 puts the roll mode's e-folding time at
0.218 s.  **Sensory correction is not fast enough to catch a roll
excursion** -- only to prevent one by keeping the crew together in the
first place.  That is a structural conclusion, not a tuning result, and it
is why this module governs *timing* rather than being another balance
loop.

References
----------
[L-PUB] Leonard, N. E., Princeton University, Mechanical and Aerospace
        Engineering / Applied and Computational Mathematics.
        Publications: https://naomi.princeton.edu/publications/
[KUR]   Kuramoto, Y. Chemical Oscillations, Waves, and Turbulence (1984) --
        the canonical phase-oscillator model.
[K-VAR] Kleshnev, V. "Rowing Science: New Analysis of Variability of
        Rower's Technique", parts 1-3, row2k.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CoupledCrew", "stroke_chain_topology",
           "all_to_all_topology", "mixed_topology"]


def stroke_chain_topology(n_seats: int, include_stroke: bool = False):
    """Directed coupling: each rower follows the one toward the stern.

    Seat 0 is bow and seat ``n-1`` is stroke in this catalogue's ordering,
    and rowers face the stern, so rower *i* sees rower *i+1*.  The stroke
    seat sees nobody and sets the reference -- which is exactly the role
    the seat has in practice.

    Returns an ``(n, n)`` matrix ``A`` with ``A[i, j]`` the weight rower
    *i* puts on rower *j*.
    """
    matrix = np.zeros((n_seats, n_seats))
    for i in range(n_seats - 1):
        matrix[i, i + 1] = 1.0
    if include_stroke and n_seats > 1:
        matrix[n_seats - 1, n_seats - 2] = 1.0
    return matrix


def all_to_all_topology(n_seats: int):
    """Mean-field coupling: everyone weights everyone equally.

    The null hypothesis against :func:`stroke_chain_topology`.  The two
    predict different phase-lag patterns down the boat -- a chain
    accumulates lag toward bow, mean-field does not -- which telemetry can
    distinguish.
    """
    matrix = np.ones((n_seats, n_seats)) - np.eye(n_seats)
    return matrix / max(n_seats - 1, 1)


@dataclass
class CoupledCrew:
    """Kuramoto-type phase dynamics for a rowing crew.

    State is ``(phi, psi)``, both ``(n_seats,)`` in radians of stroke
    phase.  Integrate with :meth:`step`; read the result as stroke
    fractions with :meth:`phase_offsets`.
    """

    n_seats: int
    #: Natural frequency spread, rad/s.  Rowers left alone do not have
    #: identical intrinsic rhythms; this is what the coupling must overcome.
    detuning: np.ndarray = None
    #: Sensory coupling strength, 1/s.
    sensory_gain: float = 4.0
    #: Mechanical coupling to hull motion, 1/s per m/s^2.
    hull_gain: float = 0.0
    #: Reaction delay, seconds.  Human sensorimotor delay is 150-250 ms.
    delay: float = 0.20
    #: Coupling graph; defaults to the directed stroke chain.
    topology: np.ndarray = None
    #: Timing noise, rad/sqrt(s).
    noise: float = 0.0
    #: How tightly the vertical phase follows the horizontal one.  A large
    #: value is a rower whose blade work is locked to their body; a small
    #: one washes out early or rows it in.
    vertical_coupling: float = 6.0
    seed: int = 0

    phi: np.ndarray = field(default=None, repr=False)
    psi: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.n_seats < 1:
            raise ValueError("need at least one seat")
        if self.delay < 0.0:
            raise ValueError("delay must be non-negative")
        if self.topology is None:
            self.topology = stroke_chain_topology(self.n_seats)
        if self.detuning is None:
            self.detuning = np.zeros(self.n_seats)
        self.detuning = np.asarray(self.detuning, dtype=float)
        if self.detuning.shape != (self.n_seats,):
            raise ValueError("detuning must have one entry per seat")
        self._rng = np.random.default_rng(self.seed)
        if self.phi is None:
            self.phi = np.zeros(self.n_seats)
        if self.psi is None:
            self.psi = np.zeros(self.n_seats)
        self._history = []

    # -- dynamics ---------------------------------------------------------
    def _delayed_phi(self, t):
        """Phase as seen ``delay`` seconds ago.

        Before enough history exists, the present is used -- the crew have
        not been going long enough to be looking at the past.
        """
        if self.delay <= 0.0 or not self._history:
            return self.phi
        target = t - self.delay
        for time, phases in reversed(self._history):
            if time <= target:
                return phases
        return self._history[0][1]

    def step(self, t: float, dt: float, omega: float,
             hull_roll_rate: float = 0.0):
        """Advance one step.

        ``omega`` is the nominal stroke frequency in rad/s.
        ``hull_roll_rate`` drives the mechanical channel: a boat that is
        rolling is telling every rower something simultaneously, with no
        delay and no choice about listening.
        """
        self._history.append((t, self.phi.copy()))
        if len(self._history) > 4096:
            del self._history[:2048]

        # What a rower saw, plus what they know has happened since.
        #
        # Naive delayed coupling -- comparing your phase *now* to someone
        # else's phase as it was 200 ms ago -- makes the whole crew row
        # slow, because everyone is perpetually chasing a stale target.
        # The collective frequency drops by an amount set by the delay,
        # independent of gain: measured here as a persistent 1.47 rad/s
        # against a nominal 3.35.  A crew that did that would row 44% below
        # the rate they were called.
        #
        # Real rowers anticipate.  Sensorimotor synchronisation is
        # predictive, not reactive -- people tap *with* a metronome, not
        # behind it -- so the seen phase is advanced by the delay before
        # being compared.  A synchronised crew then feels no coupling force
        # at all, which is the correct behaviour.
        seen = self._delayed_phi(t) + omega * self.delay
        # sin of the phase difference: the canonical Kuramoto coupling,
        # attracting each rower toward whoever they are watching
        difference = seen[None, :] - self.phi[:, None]
        sensory = self.sensory_gain * (self.topology
                                       * np.sin(difference)).sum(axis=1)

        mechanical = self.hull_gain * float(hull_roll_rate) \
            * np.ones(self.n_seats)

        drift = omega + self.detuning + sensory + mechanical

        # Both phases advance from the *same* starting state.  Updating phi
        # first and then feeding the new value into psi would make psi chase
        # a target that has already moved, leaving a spurious steady lag of
        # order omega*dt that looks exactly like a rower washing out.
        previous_phi = self.phi
        self.phi = previous_phi + dt * drift
        if self.noise > 0.0:
            self.phi = self.phi + self._rng.normal(
                0.0, self.noise * np.sqrt(dt), self.n_seats)

        # the vertical phase chases the horizontal one
        self.psi = self.psi + dt * (
            omega + self.vertical_coupling * np.sin(previous_phi - self.psi))
        return self.phi, self.psi

    # -- readout ----------------------------------------------------------
    @property
    def reference(self) -> float:
        """The stroke seat's phase, which is what the crew is following."""
        return float(self.phi[-1])

    def phase_offsets(self) -> np.ndarray:
        """Horizontal offsets relative to the stroke seat, in stroke
        fractions -- directly assignable to ``Boat.phase_offsets``."""
        return (self.phi - self.reference) / (2.0 * np.pi)

    def vertical_offsets(self) -> np.ndarray:
        """Vertical phase relative to each rower's own horizontal phase.

        Positive means the blade work *leads* the body: washing out early.
        """
        return (self.psi - self.phi) / (2.0 * np.pi)

    @property
    def order_parameter(self) -> complex:
        """Kuramoto order parameter ``r e^{i psi}``.

        ``|r| = 1`` is perfect synchrony, ``0`` is complete incoherence.
        The natural single-number measure of how together a crew is, and
        it is measurable from telemetry.
        """
        return complex(np.mean(np.exp(1j * self.phi)))

    @property
    def coherence(self) -> float:
        return float(abs(self.order_parameter))

    @property
    def spread_seconds(self) -> float:
        """Peak-to-peak timing spread, given the crew's own frequency."""
        return float(self.phi.max() - self.phi.min())


def mixed_topology(n_seats: int, chain_fraction: float = 0.25):
    """Blend the directed sensory chain with the mean-field hull channel.

    ``chain_fraction`` is how much of the coupling is the delayed,
    directional, visual/auditory chain; the remainder is the immediate,
    everyone-to-everyone mechanical coupling through the shell.

    This is the parameter the question "which channel dominates?" reduces
    to.  With realistic detuning (sd 0.10 rad/s, 3% of stroke rate) and
    ``sensory_gain = 4``, the steady-state timing spread is:

    ======================  ==========  ==========
    chain / mean-field      coherence   spread
    ======================  ==========  ==========
    0% / 100%               0.9999      12.0 ms
    25% / 75%               0.9999      13.2 ms
    50% / 50%               0.9998      14.7 ms
    75% / 25%               0.9995      24.7 ms
    100% / 0%               0.9962      75.1 ms
    ======================  ==========  ==========

    Section 21 shows about 65 ms of port/starboard split spends an eight's
    entire recovery balance authority.  So a crew coupled purely through
    the delayed, directional visual chain sits **right at the edge of the
    balance budget**, while one coupled through the shared hull has six
    times the margin.  Both channels work; the hull one is far more
    effective per unit of coupling, because it is immediate and reaches
    everyone at once.

    A chain-coupled crew can still get inside the budget, but it takes
    strong coupling -- 187 ms of spread at gain 1, 75 ms at 4, 48 ms at 16.
    That is a plausible reading of what drilling together actually trains.

    The two also leave different signatures: a chain accumulates lag
    monotonically toward bow, mean-field does not, which telemetry can tell
    apart.

    """

    chain_fraction = float(np.clip(chain_fraction, 0.0, 1.0))
    return (chain_fraction * stroke_chain_topology(n_seats)
            + (1.0 - chain_fraction) * all_to_all_topology(n_seats))
