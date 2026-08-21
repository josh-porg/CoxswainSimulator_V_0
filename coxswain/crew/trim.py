"""Stroke-to-stroke learned trim: how a crew actually uses hand heights.

There is an apparent paradox in the balance literature.  [D96] shows that
adjusting hand height to hold blade clearance is **positive feedback** --
if the boat is down on your side and you lift to keep your blade off the
water, you unweight your rigger and it goes down further.  Yet crews
plainly do use hand heights to set a boat, and it plainly works.

Both are true, because they are different control laws.

**What [D96] analyses** is a within-stroke reflex: an instantaneous
response to the blade's clearance right now.  That is destabilising, and
it is also hopeless on timing grounds -- §15 puts the roll mode's
e-folding time at 0.218 s against a human reaction of 150-250 ms, so a
rower who waits to feel the boat go over has lost the margin before the
correction starts.

**What rowers actually do** is different in kind:

    "if you notice the boat dips down to your side at the catch, on the
    next stroke you will carry your hands slightly higher near the catch,
    and vice versa.  Similarly if the boat is down to starboard at the
    finish, the next stroke starboards will raise their hands more at the
    finish and ports will lower theirs."

Every clause of that is a specification:

* the error is observed on **one stroke** and corrected on the **next**;
* the correction is applied at the **same phase** at which the error
  appeared -- a catch error is fixed at the catch, a finish error at the
  finish;
* it is **antisymmetric between the sides**, which is exactly what turns
  hand height into a roll moment through the rigger geometry;
* it is **anticipatory**, not reactive: it is already applied before the
  error would recur.

That is iterative learning control [BTA06].  The stroke is a repetitive
process; error at phase θ on cycle k updates the input at phase θ on cycle
k+1.  The one-stroke delay, which would be fatal in a feedback loop, is
not a delay at all here -- for a disturbance that repeats every stroke it
is memory.

Why this resolves the paradox
-----------------------------
The learned trim never tries to arrest a roll excursion in progress.  It
reduces the **initial condition**: the heel at the finish, from which the
recovery's ×180 amplification runs.  §15 puts the tolerance there at about
0.013°, which no reactive loop could hold.  Learned trim can, because it
has as many strokes as it needs and the disturbance it is cancelling is
the repeating one.

It also explains the training.  A crew that has not converged its trim
cannot sit the boat however hard they concentrate, and convergence takes
many strokes together -- which is what "hours of drilling" buys, and why
it does not transfer to a different crew or a different boat.

Where the authority comes from
------------------------------
The corrections in the description above are made **at the catch and at
the finish** -- both points where the blade is in or entering the water,
so the crew have the drive's authority, not the recovery's.  Nothing here
special-cases that: the command is saturated by the same
:class:`~coxswain.crew.balance.PhaseAuthority` window as the reactive
loop, and the learning naturally puts its corrections where the authority
to execute them exists.

References
----------
[BTA06] Bristow, D. A., Tharayil, M., & Alleyne, A. G. (2006). A survey of
        iterative learning control.  *IEEE Control Systems Magazine*
        **26**(3), 96-114.
[D96]   "Balance of Racing Rowing Boats", Furnivall Sculling Club, 1996.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["StrokeTrim"]


@dataclass
class StrokeTrim:
    """Phase-indexed trim learned from previous strokes.

    The memory holds a roll-moment command per phase bin.  It is updated
    once per stroke from the roll error recorded at each phase, and played
    back as a feedforward command on the next stroke.
    """

    n_bins: int = 24
    #: ILC learning gain.  Convergence of the standard update requires
    #: ``|1 - L G| < 1`` for the plant gain G; in practice this is tuned
    #: down for robustness to the non-repeating part of the disturbance.
    learning_gain: float = 0.6
    #: Robustness filter, the ``Q`` of [BTA06].  Below one it trades exact
    #: convergence for rejection of disturbances that do *not* repeat,
    #: which is what stops a crew learning the wash from a passing launch.
    forgetting: float = 0.92
    #: Proportionality between roll error and the corrective moment, in
    #: N m per radian.  Interpreted through the rigger geometry, this is
    #: how hard the crew push for a given observed heel.
    gain: float = 4000.0
    memory: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.n_bins < 2:
            raise ValueError("need at least two phase bins")
        if self.memory is None:
            self.memory = np.zeros(self.n_bins)

    # -- playback ---------------------------------------------------------
    def bin_of(self, t, timing) -> int:
        phase = float(np.mod(t, timing.period) / timing.period)
        return min(int(phase * self.n_bins), self.n_bins - 1)

    def command(self, t, timing) -> float:
        """Feedforward roll moment for this instant, learned last stroke."""
        return float(self.memory[self.bin_of(t, timing)])

    # -- learning ---------------------------------------------------------
    def observe(self, times, rolls, timing) -> np.ndarray:
        """Average roll error in each phase bin over one stroke."""
        error = np.zeros(self.n_bins)
        count = np.zeros(self.n_bins)
        for t, roll in zip(np.asarray(times), np.asarray(rolls)):
            index = self.bin_of(t, timing)
            error[index] += float(roll)
            count[index] += 1.0
        return np.where(count > 0, error / np.maximum(count, 1.0), 0.0)

    def update(self, times, rolls, timing) -> "StrokeTrim":
        """One ILC iteration: ``u <- Q u - L K e``.

        The sign is negative because a positive roll must be met by a
        negative moment.  Returns ``self`` so a caller can chain.
        """
        error = self.observe(times, rolls, timing)
        self.memory = (self.forgetting * self.memory
                       - self.learning_gain * self.gain * error)
        return self

    @property
    def effort(self) -> float:
        """RMS of the learned command, a proxy for how much trim the crew
        is carrying.  A well-matched crew in a well-rigged boat converges
        to a small number; a large one means they are fighting something
        structural -- a rigging error, or a rower who is genuinely heavier
        on one side."""
        return float(np.sqrt(np.mean(self.memory ** 2)))

    def reset(self) -> None:
        self.memory = np.zeros(self.n_bins)
