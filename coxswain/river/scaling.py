"""Non-dimensionalisation for the trajectory NLP.

This should have been here before the first solve, and its absence is why
the first solves crawled.

An interior-point method works on the Jacobian and Hessian of the problem
as posed.  If the variables differ by orders of magnitude, so do the rows
and columns of those matrices, and the linear solve at every iteration is
correspondingly ill-conditioned.  IPOPT's own barrier and step-acceptance
logic also compares quantities across variables -- a trust region or a
fraction-to-the-boundary rule that is sensible for a position in metres is
meaningless for a roll angle in radians.

The transcription as first written spanned **six orders of magnitude**:

===================  ==============
quantity             typical value
===================  ==============
position x, y        1000 m
anaerobic W'         2e4 J
velocity             5 m/s
yaw                  3 rad
omega                0.5 rad/s
rudder               0.4 rad
split                0.15
heave                0.02 m
roll, pitch          0.02 rad
===================  ==============

The fix is standard and cheap: solve in variables that are all O(1).  Each
state and control is divided by a characteristic scale, the dynamics are
rescaled to match, and the answer is multiplied back at the end.  Nothing
about the physics changes -- this is a change of units, not of model.

Choosing the scales
-------------------
Each scale is the magnitude the quantity actually reaches on this problem,
not a round number: the length of the leg for position, racing speed for
velocity, the surrogate's sampled range for attitude, the crew's capacity
for W'.  A scale that is wrong by a factor of ten is still enormously
better than none, so these do not need to be precise -- but taking them
from the problem rather than from habit costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ..hydro.appendages import MAX_RUDDER_DEFLECTION

__all__ = ["ProblemScaling"]


@dataclass
class ProblemScaling:
    """Characteristic magnitudes for every state and control.

    ``state`` is ``(n_states,)`` and ``control`` is ``(n_controls,)``.
    Solve in ``x / state``, then multiply back.
    """

    state: np.ndarray
    control: np.ndarray
    objective: float = 1.0

    @property
    def spread(self) -> float:
        """Ratio of the largest characteristic magnitude to the smallest.

        This measures the **unscaled** problem: it is how far apart the
        physical quantities are, and therefore how badly conditioned the
        NLP would be without this class.  It stays large by construction --
        that is the point, it is what is being divided out.

        For the six-degree-of-freedom problem it is about 1e6, dominated by
        the crew's anaerobic capacity in joules against a split as a
        dimensionless fraction.
        """
        values = np.concatenate([np.abs(self.state), np.abs(self.control)])
        return float(values.max() / values.min())

    def scaled_spread(self, state, control) -> float:
        """Ratio of largest to smallest **after** scaling a sample point.

        This is the number that should be near one, and it is the honest
        check that the scaling did its job.  Pass a representative state
        and control -- typically the initial guess.
        """
        scaled = np.concatenate([
            np.abs(self.to_scaled_state(state)),
            np.abs(self.to_scaled_control(control))])
        scaled = scaled[scaled > 1e-12]
        if scaled.size == 0:
            return 1.0
        return float(scaled.max() / scaled.min())

    @classmethod
    def unscaled(cls, n_states: int, n_controls: int) -> "ProblemScaling":
        """No scaling, for comparison against the scaled solve."""
        return cls(state=np.ones(n_states), control=np.ones(n_controls))

    @classmethod
    def for_six_dof(cls, model, leg_length: float = 500.0,
                    speed: float = 5.0, rudder_limit: float = MAX_RUDDER_DEFLECTION,
                    split_limit: float = 0.15) -> "ProblemScaling":
        """Scales taken from the boat and the leg being flown.

        ``leg_length`` is how far the boat travels over the horizon, which
        is what sets the position scale -- using the absolute coordinate
        instead would scale by where the origin happens to be, which is
        arbitrary and can be enormous.
        """
        surrogate = getattr(model, "surrogate", None)
        heave = 0.05
        attitude = np.radians(5.0)
        if surrogate is not None:
            heave = max(float(np.abs(surrogate.heave).max()), 1e-3)
            attitude = max(float(np.abs(surrogate.roll).max()), 1e-3)

        state = np.array([
            leg_length, leg_length, heave,          # x, y, z
            attitude, attitude, np.pi,              # roll, pitch, yaw
            speed, speed, speed,                    # velocities
            1.0, 1.0, 1.0,                          # angular rates
            max(float(getattr(model, "anaerobic_capacity", 1.0)), 1.0),
        ], dtype=float)
        control = np.array([rudder_limit, split_limit, 1.0], dtype=float)
        return cls(state=state, control=control)

    # -- transforms --------------------------------------------------------
    def to_scaled_state(self, values):
        return np.asarray(values, dtype=float) / self.state

    def from_scaled_state(self, values):
        return np.asarray(values, dtype=float) * self.state

    def to_scaled_control(self, values):
        return np.asarray(values, dtype=float) / self.control

    def from_scaled_control(self, values):
        return np.asarray(values, dtype=float) * self.control

    def scaled_dynamics(self, dynamics, ca):
        """Wrap a dynamics function to act on scaled variables.

        ``d(x/s)/dt = f(x, u) / s``, so the returned function takes scaled
        arguments, unscales them, evaluates the *unchanged* dynamics, and
        divides the derivative by the same scales.  The physics is
        untouched; only the coordinates the solver sees have changed.
        """
        state_scale = ca.DM(self.state.reshape(-1, 1))
        control_scale = ca.DM(self.control.reshape(-1, 1))

        def scaled(x, u, t):
            return dynamics(x * state_scale, u * control_scale, t) \
                / state_scale

        return scaled
