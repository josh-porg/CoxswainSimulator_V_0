"""Named physics configurations, and the freeze that keeps the game still.

Why this exists
---------------
Two known defects are being fixed offline -- the blade carries no velocity
term, and the crew kinematics are fitted from a stationary ergometer -- and
neither fix may reach the shipped trainer until it has been validated.  That
is not a branch: the offline report and the game are built from the same
package, run from the same working tree, and must be able to disagree about
the physics *at the same commit*.

So nothing constructs physics implicitly any more.  A caller names a
profile, and the profile owns every switch that has to move together:

    >>> from coxswain import physics
    >>> profile = physics.resolve("shipped")
    >>> profile.blade_tier
    0

``shipped`` is **frozen, not current**.  Its calibrations, golden
trajectories and speed targets stay pinned to it for as long as the game is
meant to stay still, and no research change may move them.  The guard is a
test (``tests/test_physics_profiles.py``), not a convention: every path the
game can take to a boat is checked to resolve ``shipped`` and nothing else.

The tiers
---------
Blade, after ``docs/SOURCES.md`` sec. 7 and the plan of 2026-09-12:

==== ============================= ====================================
tier  model                         state of play
==== ============================= ====================================
0     prescribed force profile      what ships; force is a function of
                                    stroke phase and nothing else
1     [CR06] Model 1, slip          normal load from blade slip, with
      quadratic                     the oar angle as a dynamic state
2     [CR06] Model 2 family, lift   lift and drag resolved against
      and drag on angle of attack   angle of attack
==== ============================= ====================================

Rower:

============ ==============================================================
prescribed    joint angles from ergometer motion capture, phase-driven
forward       torque-driven multibody rower (Rongere's formalism)
learned       a trained policy outputting joint torques
============ ==============================================================

Adding a profile is deliberately cheap; *changing* ``shipped`` is
deliberately not.  :func:`resolve` refuses to hand back a mutable view of a
frozen profile's configuration, and :meth:`PhysicsProfile.apply` on a frozen
profile asserts that it is a no-op against a freshly built boat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = [
    "PhysicsProfile", "PROFILES", "SHIPPED", "resolve", "names",
    "FrozenProfileError",
]

#: The profile the game and the online report must always resolve.
SHIPPED = "shipped"


class FrozenProfileError(RuntimeError):
    """Raised when something tries to move a frozen profile's physics."""


@dataclass(frozen=True)
class PhysicsProfile:
    """One self-consistent physics configuration, resolved by name.

    A profile is not a bag of options.  The blade tier, the rower driver
    and the calibration constants that go with them only make sense
    together -- switching the blade model without re-calibrating cost 14%
    of boat speed the first time it was tried (SOURCES sec. 7) -- so they
    move as a unit or not at all.
    """

    name: str
    summary: str
    #: 0, 1 or 2; see the module docstring.
    blade_tier: int
    #: ``"prescribed"``, ``"forward"`` or ``"learned"``.
    rower: str
    #: Frozen profiles may not be altered, and the game may resolve only
    #: a frozen one.
    frozen: bool = False

    def __post_init__(self) -> None:
        if self.blade_tier not in (0, 1, 2):
            raise ValueError(
                f"blade_tier must be 0, 1 or 2; got {self.blade_tier!r}")
        if self.rower not in ("prescribed", "forward", "learned"):
            raise ValueError(
                f"unknown rower driver {self.rower!r}")

    # -- the blade -------------------------------------------------------
    def blade_model(self, boat=None):
        """The :class:`~coxswain.crew.oarlock.BladeModel` for this profile.

        ``None`` at tier 0, which is what ``Boat.blade_model`` already
        means: the simulator then uses the oar's fixed
        ``blade_efficiency`` rather than computing one from slip.

        Above tier 0 the coefficient depends on the rig -- [CR06] fit
        ``C2`` separately for sculling (58.7) and sweep (84.5) -- so the
        boat is consulted when one is given.  A boat whose rig cannot be
        read falls back to the sweep fit, which is the conservative
        choice here: it is the larger coefficient, so it grips harder and
        slips less, and an error shows up as too *little* loss rather
        than a plausible-looking too much.
        """
        if self.blade_tier == 0:
            return None
        from .crew.oarlock import BladeModel

        sculling = False
        rig = getattr(boat, "rig", None)
        if rig is not None:
            try:
                sculling = not bool(rig.is_sweep)
            except Exception:
                sculling = False
        outboard = _outboard_of(boat)
        if sculling:
            return (BladeModel.sculling(outboard=outboard)
                    if outboard else BladeModel.sculling())
        return (BladeModel.sweep(outboard=outboard) if outboard
                else BladeModel.sweep())

    # -- application -----------------------------------------------------
    def apply(self, boat):
        """Configure ``boat`` for this profile, and return it.

        Mutates the boat in place -- a ``Boat`` is expensive to build and
        callers already hold references to it -- and returns it so this
        reads as a pipeline at the call site.
        """
        boat.blade_model = self.blade_model(boat)
        boat.physics_profile = self.name
        return boat

    def is_shipped(self) -> bool:
        return self.name == SHIPPED

    def __str__(self) -> str:  # pragma: no cover - display only
        return "%s (blade tier %d, %s rower)" % (
            self.name, self.blade_tier, self.rower)


def _outboard_of(boat):
    """The rig's outboard length, or ``None`` if it cannot be read.

    [CR06]'s ``C2`` is fitted with their own outboard, so using the rig's
    actual value is the consistent choice; but a catalogue boat is not the
    only thing that reaches here and the default must survive a boat with
    no rig at all.
    """
    rig = getattr(boat, "rig", None)
    if rig is None:
        return None
    try:
        for seat in rig.seats:
            for lock in seat.oarlocks:
                return float(lock.oar.outboard)
    except Exception:
        return None
    return None


#: Every configuration this repository knows how to build.
#:
#: ``shipped`` is what the trainer and the online report run.  It is frozen
#: and is to stay that way until a change has been justified against the
#: validation scorecard.  The other two are the offline programme.
PROFILES: Dict[str, PhysicsProfile] = {
    SHIPPED: PhysicsProfile(
        name=SHIPPED,
        summary="What the released trainer runs: prescribed oar force, "
                "prescribed crew kinematics. Frozen.",
        blade_tier=0,
        rower="prescribed",
        frozen=True,
    ),
    # WARNING, and it is not a small one.  ``blade_tier=1`` is the TARGET.
    # What tier 1 currently resolves to is the efficiency-only wiring --
    # the transmitted force scaled by the instantaneous slip efficiency --
    # and that has **no viable operating point**: the eight collapses from
    # 3.92 m/s to 0.63 because the efficiency factor is the destabilising
    # half of the blade physics and the restoring half lives in the force
    # model.  See ``tests/test_blade_tier1.py`` and TRACKING.
    #
    # It is left resolving to tier 1 rather than quietly dropped to tier 0
    # because the collapse is the finding, and a profile that silently
    # behaves like ``shipped`` would hide it.  Do not quote a speed from
    # this profile until phase 2 lands the slip FORCE.
    "research": PhysicsProfile(
        name="research",
        summary="The offline programme: blade force from slip, oar angle "
                "a dynamic state, torque-driven rower. UNSTABLE until "
                "phase 2 -- the blade is wired as an efficiency only and "
                "the boat collapses. Not for speed numbers.",
        blade_tier=1,
        rower="prescribed",
    ),
    # Declared so the shape of the programme is visible in the code and not
    # only in the plan.  NOTHING behind it exists yet: there is no tier 2
    # blade and no policy.  Resolving it gets you a tier-1 blade model and a
    # prescribed rower, which is not what the name says -- so it is named
    # UNBUILT for the same reason ``research`` is named UNSTABLE.
    "learned": PhysicsProfile(
        name="learned",
        summary="Training and study runs only: tier 2 blade, policy-driven "
                "rower. UNBUILT -- neither the tier 2 blade nor the policy "
                "exists; this is a placeholder for the shape of the "
                "programme. Not for any number at all.",
        blade_tier=2,
        rower="learned",
    ),
}


def names():
    """Profile names, ``shipped`` first."""
    rest = sorted(n for n in PROFILES if n != SHIPPED)
    return [SHIPPED] + rest


def resolve(name=SHIPPED) -> PhysicsProfile:
    """The profile called ``name``.

    Defaults to :data:`SHIPPED`, so a caller that has not thought about it
    gets the frozen configuration rather than whatever is newest.  That
    default is the whole safety property: forgetting to pass a profile
    cannot silently promote research physics into the game.

    ``None`` resolves to :data:`SHIPPED` too, and deliberately.  A profile
    threads through call chains as an optional argument, and an unset
    optional argument arrives as ``None`` rather than as a missing one --
    so treating ``None`` as "unknown profile" would make the common way of
    forgetting raise instead of falling back to the safe configuration.
    """
    if name is None:
        name = SHIPPED
    if isinstance(name, PhysicsProfile):
        return name
    try:
        return PROFILES[str(name)]
    except KeyError:
        raise KeyError(
            "unknown physics profile %r; available: %s"
            % (name, ", ".join(names()))
        ) from None
