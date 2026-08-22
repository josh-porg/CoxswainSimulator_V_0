"""Blades touching the water on the recovery.

The model has assumed the blades are clear throughout the recovery. That
is the intent of good rowing, and elite crews achieve it on nearly every
stroke of the Charles. It is not what always happens, and the difference
is not a detail: it is the mechanism that gives roll error a cost in
seconds.

Why it matters more than it looks
---------------------------------
The blade tip sits **3.41 m** from the centreline on this eight -- an
0.85 m rigger plus 2.56 m of outboard. Roll rotates that lever, so a very
small heel puts a blade down:

==============  ================
carried clearance   roll to touch
==============  ================
5 cm            0.84 deg
10 cm           1.68 deg
15 cm           2.52 deg
20 cm           3.36 deg
==============  ================

Section 15-16 put an eight's roll swing at about 1 degree and its recovery
balance authority at 2.3 degrees of heel. **A crew carrying a normal
5-10 cm of clearance is one degree of roll away from dragging a blade**,
which is the same order as the roll the boat does anyway. This is not an
edge case; it is the operating point.

Feathering is what makes it survivable
--------------------------------------
The cost depends entirely on whether the blade is **feathered** or
**squared**, and by more than an order of magnitude. A feathered blade
lies flat and presents its *edge*: the frontal area is thickness by span,
and it **saturates** once the blade is submerged past its own thickness
rather than growing with depth. A squared blade presents its full face,
and the area keeps growing.

At 4.6 m/s, all eight blades:

=========  ===========  ==============  ==============  =======
roll       immersion    feathered       squared         ratio
=========  ===========  ==============  ==============  =======
1.5 deg    9 mm         -20.6 N         -117.6 N        5.7x
2.0 deg    39 mm        **-22.2 N**     -495 N          22x
3.0 deg    98 mm        -22.2 N         -1250 N         56x
=========  ===========  ==============  ==============  =======

against a hull resistance of about 330 N. So a properly feathered skim
costs about **7% of hull drag** and stops getting worse; a squared blade
at the same heel costs four times the entire hull resistance. That is the
difference between a boat that is untidy and one that has stopped, and it
is why feathering is not a stylistic matter.

The squared case is a crab, a different regime, and is kept behind a flag
so the two cannot be confused.

The other side of it
--------------------
The same contact is a *stabiliser*. [D96] is explicit that a blade on the
water unweights its rigger and that a sculler can use it deliberately:

    "if you drag one blade along the water lightly some of the weight is
    taken by the spoon and it will reduce the weight acting down on the
    rigger... By exact hand control you can scull a boat dead flat this
    way."

And crucially, **feathering does not reduce the lift**. A feathered blade
is exactly the right shape to plane on its flat underside, so it keeps
its vertical force while shedding almost all of its drag. At two degrees
of heel the contact produces about **2100 N m** of roll moment against the
crew's own recovery authority of 84 N m -- twenty-five times as much --
for 22 N of drag.

That is why skimming works as a technique rather than merely being a
mistake, and it reframes the trade: a feathered skim is close to free
stabilisation. The real cost of an unset boat is not the skim. It is the
squared blade catching, and the truncated drive that follows -- once a
blade is in at the catch the rower must go with it, so the stroke starts
early and short.

[D96] also notes crew boats must clear the puddles of the crew ahead, so
this is a *fault* in an eight rather than a technique -- unlike a single,
where it is a legitimate way to sit the boat.

Smoothing
---------
Contact is a switch, and IPOPT needs a derivative. The transition is a
logistic in blade-to-surface clearance whose width is set by something
physical -- surface roughness and small waves, of order a centimetre --
rather than chosen for numerical convenience.
:meth:`BladeContact.switch_error` reports the departure from the hard
switch.

References
----------
[D96] "Balance of Racing Rowing Boats", Furnivall Sculling Club, 1996.
      https://eodg.atm.ox.ac.uk/user/dudhia/rowing/physics/Balance_of_Racing_Rowing_Boats_v3.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BladeContact"]


@dataclass
class BladeContact:
    """Loads from a blade skimming or dragging the surface.

    Applies on the recovery only: during the drive the blade is meant to be
    in the water and the oar model already handles it.
    """

    #: Distance from the centreline to the blade tip, metres.
    reach: float = 3.41
    #: Blade width, metres -- sets the wetted area per unit immersion.
    width: float = 0.25
    #: Blade span along the shaft, metres -- the length of the leading
    #: edge a feathered blade presents to the oncoming water.
    span: float = 0.35
    #: Blade thickness, metres.  This is what sets the drag of a
    #: *feathered* blade, and it is why feathering matters so much.
    thickness: float = 0.010
    #: Is the blade feathered?  True through the recovery, which is the
    #: only phase this model applies to.  A squared blade catching is a
    #: crab, a different regime, kept separate so they cannot be confused.
    feathered: bool = True
    #: Drag coefficient of a **squared** blade, face-on.
    squared_drag_coefficient: float = 1.2
    #: Clearance the crew carry above the water on the recovery, metres.
    #: Crews in bigger boats carry more, because they must clear puddles.
    clearance: float = 0.08
    #: Drag coefficient of a **feathered** blade, edge-on to the flow.
    #: A thin plate aligned with the stream, not a bluff body.
    drag_coefficient: float = 0.15
    #: Vertical force per unit immersion, N/m.  A skimming blade planes and
    #: carries load, which is what unweights the rigger.
    lift_per_metre: float = 4.0e3
    #: Largest vertical force a blade can carry before it simply dives.
    max_lift: float = 300.0
    #: Logistic width for the contact switch, metres.  Set by surface
    #: roughness and small waves rather than by numerical taste.
    softness: float = 0.01
    density: float = 1000.0

    @classmethod
    def from_boat(cls, boat, **kwargs):
        lock = boat.rig.seats[0].oarlocks[0]
        reach = abs(float(lock.position[1])) + float(lock.oar.outboard)
        width = float(getattr(lock.oar, "blade_width", 0.25))
        return cls(reach=reach, width=width, **kwargs)

    # -- geometry ---------------------------------------------------------
    def roll_to_touch(self, clearance=None) -> float:
        """Roll angle at which the low blade first reaches the water."""
        clearance = self.clearance if clearance is None else clearance
        return float(np.arcsin(np.clip(clearance / self.reach, -1.0, 1.0)))

    def immersion(self, roll, side, ca=None):
        """Depth of the blade below the surface, metres; zero if clear.

        ``side`` is ``+1`` for port and ``-1`` for starboard.  A positive
        roll lowers one side and raises the other, so the two sides see
        opposite signs of the same lever.
        """
        drop = self.reach * (np.sin(roll) if ca is None else ca.sin(roll))
        depth = float(side) * drop - self.clearance
        if ca is None:
            return float(np.maximum(depth, 0.0))
        # softplus: a smooth max(depth, 0) whose width is the surface
        # roughness, so the smoothing corresponds to something real
        return self.softness * ca.log1p(ca.exp(depth / self.softness))

    # -- loads ------------------------------------------------------------
    def loads(self, roll, surge_speed, ca=None):
        """``(drag, roll_moment)`` in the hull frame, for the whole crew.

        ``drag`` is negative -- it opposes motion.  ``roll_moment`` is the
        restoring couple from the two sides' vertical forces, which is the
        stabilising half of the trade.
        """
        drag = 0.0
        moment = 0.0
        speed = surge_speed if ca is not None else float(surge_speed)
        for side in (1.0, -1.0):
            depth = self.immersion(roll, side, ca)
            if self.feathered:
                # Feathered: the blade lies flat and presents its *edge*.
                # Frontal area is the leading edge -- thickness by span --
                # and it **saturates** once the blade is submerged past
                # its own thickness.  It does not keep growing with depth
                # the way a face-on plate would.  Treating a feathered
                # skim as if it were squared overstates the drag by more
                # than an order of magnitude: 0.15 against 1.2, on an area
                # that stops growing instead of one that does not.
                if ca is None:
                    fraction = min(float(depth) / self.thickness, 1.0)
                else:
                    fraction = ca.tanh(depth / self.thickness)
                area = self.thickness * self.span * fraction
                coefficient = self.drag_coefficient
            else:
                area = self.width * depth
                coefficient = self.squared_drag_coefficient
            per_blade = (0.5 * self.density * coefficient * area
                         * speed * speed)
            lift = self.lift_per_metre * depth
            if ca is not None:
                lift = self.max_lift * ca.tanh(lift / self.max_lift)
            else:
                lift = float(np.minimum(lift, self.max_lift))
            # four blades a side on an eight; the moment opposes the heel
            drag = drag - 4.0 * per_blade
            moment = moment - 4.0 * lift * side * self.reach
        return drag, moment

    # -- how good is the smoothing ----------------------------------------
    def switch_error(self, samples: int = 400) -> float:
        """Worst departure of the softplus from a hard ``max(depth, 0)``.

        Reported in metres of immersion.  The softplus is above the hard
        switch everywhere, worst at the contact point where it reads
        ``softness * ln 2``.
        """
        import casadi as ca

        rolls = np.linspace(-0.1, 0.1, samples)
        worst = 0.0
        for roll in rolls:
            hard = self.immersion(float(roll), 1.0)
            soft = float(ca.DM(self.immersion(ca.DM(float(roll)), 1.0, ca)))
            worst = max(worst, abs(soft - hard))
        return worst

    # -- the truncated drive ----------------------------------------------
    #: Vertical closing speed of the blade onto the water as the hands come
    #: up into the catch, m/s.  Sets how much earlier a heeled blade
    #: touches down than a level one.
    approach_speed: float = 0.40

    def lost_sweep(self, roll, oar_rate, side=None):
        """Oar angle given up because the blade went in early, radians.

        This is the cost the coxswain described: *"when the rower needs to
        square up for the catch, once the blade is in the water they have
        to go with it and drive, so an unset boat will lead to earlier
        drives and not full extension."*

        A blade that is already ``d`` metres under when the catch arrives
        went in ``d / approach_speed`` seconds early, and in that time the
        oar swept ``oar_rate`` radians it will never get back.  The drive
        therefore starts short.

        Unlike the skim drag -- which feathering makes almost free -- this
        cost does **not** go away with good blade work.  It is set by the
        heel, and it is the reason an unset boat is slow even when nobody
        catches a crab.
        """
        sides = (1.0, -1.0) if side is None else (float(side),)
        worst = 0.0
        for value in sides:
            depth = self.immersion(roll, value)
            worst = max(worst, float(depth))
        early = worst / max(self.approach_speed, 1e-6)
        return float(abs(oar_rate) * early)

    def length_fraction(self, roll, oar_rate, total_sweep):
        """Fraction of the intended sweep the crew actually gets.

        One when the blades are clear; below one once a blade is in early.
        Multiply the stroke's impulse by this.
        """
        lost = self.lost_sweep(roll, oar_rate)
        return float(np.clip(1.0 - lost / max(total_sweep, 1e-9), 0.0, 1.0))
