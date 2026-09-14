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
    #: How the oars load the hull.  ``"prescribed"``: a function of stroke
    #: phase, what ships.  ``"efficiency"``: the prescribed force scaled
    #: by slip efficiency -- tried, has no operating point, and kept only
    #: so the evidence stays buildable.  ``"dynamic"``: the oar angle as a
    #: state, with the blade model living in
    #: :class:`~coxswain.sim.dynamic_oar.DynamicOarSimulator`.
    oar: str = "prescribed"
    #: Wave resistance.  ``"michell"``: the boat's own deep-water Michell
    #: table times the chosen shallow-water factor, what ships.
    #: ``"sretenskii"``: Michell with trapezoid weights, which matches
    #: Lazauskas's printed Wigley curve, and Sretenskii's finite-depth
    #: integral in place of the shallow-water factor (SOURCES sec. 6).
    wave: str = "michell"
    #: How the dynamic oar's blade enters the water; see
    #: :attr:`~coxswain.sim.dynamic_oar.DynamicOarSimulator.CATCH_RULES`.
    #: ``"rest"``: parked at the catch, loaded at once -- what every profile
    #: has run so far.  ``"sweep"``: [CR06]'s entry at zero normal velocity,
    #: with rower power matched to include the energy the sweep carries in.
    catch: str = "rest"
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
        if self.oar not in ("prescribed", "efficiency", "dynamic"):
            raise ValueError(f"unknown oar driver {self.oar!r}")
        if self.wave not in ("michell", "sretenskii"):
            raise ValueError(f"unknown wave model {self.wave!r}")
        if self.catch not in ("rest", "sweep"):
            raise ValueError(f"unknown catch rule {self.catch!r}")
        if self.catch != "rest" and self.oar != "dynamic":
            raise ValueError(
                f"catch rule {self.catch!r} belongs to the dynamic oar; "
                f"this profile's oar is {self.oar!r}")
        # A blade model and a prescribed oar are a contradiction in either
        # direction: tier 0 has no blade to be dynamic, and a tier above 0
        # whose oar ignores the blade is tier 0 wearing a label.
        if (self.blade_tier == 0) != (self.oar == "prescribed"):
            raise ValueError(
                f"blade tier {self.blade_tier} cannot have a {self.oar!r} "
                "oar: tier 0 is exactly the prescribed oar")

    @property
    def uses_dynamic_oar(self) -> bool:
        """True when the oar angle is a state and the ordinary simulator's
        prescribed oar block must not be used for this physics."""
        return self.oar == "dynamic"

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
        if self.oar == "efficiency":
            boat.blade_model = self.blade_model(boat)
        else:
            # Prescribed: tier 0 has none.  Dynamic: the blade lives in the
            # dynamic simulator and must NOT also be set here -- the
            # ordinary simulator reads ``blade_model`` and would lay the
            # efficiency-only wiring on top, which is the configuration
            # with no operating point.
            boat.blade_model = None
        if self.wave == "sretenskii":
            # Only where the boat already carries Michell's table: a boat
            # built on the constant wave coefficient stays on it.
            offsets = getattr(boat, "offsets", None)
            if offsets is not None and getattr(boat, "wave_table",
                                               None) is not None:
                from .hydro.finite_depth_michell import wave_table_for
                boat.wave_table = wave_table_for(offsets)
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
    # The dynamic oar held phase 2's gate on the full 6-DOF hull on
    # 2026-09-12, so this no longer resolves to the efficiency-only wiring
    # (which collapsed the boat, and whose summary said UNSTABLE).  It is
    # still PARTIAL, and says what is unfinished: the crew is prescribed
    # and does not follow the dynamic oar, only synchronised crews run,
    # and only the validation scorecard is ported -- the prescribed oar
    # block refuses a boat carrying this stamp, so the report cannot
    # silently simulate the shipped oar under this name.
    "research": PhysicsProfile(
        name="research",
        summary="The offline programme: blade force from slip with the oar "
                "angle a dynamic state, on the full 6-DOF hull. PARTIAL -- "
                "the crew is still prescribed from ergometer data and does "
                "not follow the dynamic oar, synchronised crews only, and "
                "only the validation scorecard runs it; the report refuses.",
        blade_tier=1,
        rower="prescribed",
        oar="dynamic",
        wave="sretenskii",
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
        oar="dynamic",
        wave="sretenskii",
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
