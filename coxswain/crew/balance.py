"""How a crew actually applies a balance moment, and what else it does.

The balance reflex used to be a pure couple about the hull ``x`` axis: a
saturated PD loop on roll, added straight to the moment vector.  That is a
reduction, and in a sweep boat it is not a harmless one.

A crew has no way to apply a pure roll couple.  What they can do is change
the height of their hands, which loads the oar as a lever about the
oarlock and puts a **vertical force at the rigger**.  The riggers are at
fixed points on the hull, so the moment the crew can produce is whatever
that set of point forces happens to make -- and in a sweep boat rigged the
normal alternating way, the port and starboard riggers are not at the same
longitudinal stations.

For the eight in this catalogue the port oarlocks average ``x = -0.34 m``
and the starboard ``x = +0.88 m``: a 1.22 m offset, exactly one seat
spacing.  A balance effort that pushes one side down and lifts the other
therefore applies four downward forces that sit 1.22 m from the four
upward ones, which is a couple in the *vertical-longitudinal* plane as
well as the intended one in the transverse plane.

**Balancing a sweep eight pitches it.**  The coupling is not small: for
this rig the pitch moment is about 0.72 of the roll moment.  Nothing about
that is exotic -- it is a direct consequence of alternating rigging, and
it is invisible to any model that adds balance as a pure ``x`` couple.

The net *vertical force* does cancel, so the old docstring's claim was
right as far as it went.  It was the pitch term it was silent about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BalanceRig"]


@dataclass(frozen=True)
class BalanceRig:
    """Geometry linking a crew's balance effort to the load on the hull.

    The crew's effort is parameterised by a single scalar ``f``: the
    vertical force one oarlock carries, positive up on the port side and
    negative on starboard (or the reverse, which just flips the sign of
    every output).  Everything the hull feels is linear in ``f``, so the
    whole mechanism reduces to three exact coefficients -- no smoothing,
    no fit, and CasADi differentiates it trivially.
    """

    roll_per_unit: float      # N m of roll moment per newton of rigger load
    pitch_per_unit: float     # N m of pitch moment, the coupling term
    heave_per_unit: float     # N of net vertical force; zero for even rigs

    @classmethod
    def from_boat(cls, boat) -> "BalanceRig":
        roll = pitch = heave = 0.0
        for seat in boat.rig.seats:
            for lock in seat.oarlocks:
                x, y, _ = np.asarray(lock.position, dtype=float)
                sign = float(lock.side)
                # a vertical force F_z at (x, y) gives moment (y F_z, -x F_z, 0)
                roll += y * sign
                pitch += -x * sign
                heave += sign
        return cls(float(roll), float(pitch), float(heave))

    @property
    def pitch_coupling(self) -> float:
        """Pitch moment produced per unit of roll moment demanded.

        Zero for a sculling boat and for any rig symmetric fore-and-aft;
        about 0.72 for a conventionally rigged eight.
        """
        if self.roll_per_unit == 0.0:
            return 0.0
        return self.pitch_per_unit / self.roll_per_unit

    def loads(self, roll_moment):
        """``(force, moment)`` in the hull frame for a demanded roll moment.

        Works with floats or CasADi expressions.  The demanded roll moment
        is delivered exactly; the pitch moment and any net heave come along
        with it because the rig's geometry says they must.
        """
        if self.roll_per_unit == 0.0:
            zero = roll_moment * 0.0
            return (zero, zero, zero), (zero, zero, zero)
        unit = roll_moment / self.roll_per_unit
        zero = roll_moment * 0.0
        force = (zero, zero, unit * self.heave_per_unit)
        moment = (roll_moment, unit * self.pitch_per_unit, zero)
        return force, moment
