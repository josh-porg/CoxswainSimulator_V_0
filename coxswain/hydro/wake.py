"""Rowing in another crew's water.

There is no peer-reviewed model of one rowing shell following another, so
this builds one from the pieces that *are* established, and is honest
about the one constant it cannot pin down without a measurement.

The exact constraint nobody can argue with
------------------------------------------
A boat at steady speed feels no net force, so by Newton's third law it
exerts no net force on the water.  **The total momentum in the wake of a
self-propelled body is zero** -- the standard momentumless-wake result
[N65]_, [SP00]_, and the reason a submarine's wake is so much harder to
detect than a towed body's.

That is not a claim that the wake is harmless.  It is a claim that it has
two parts of opposite sign, and they sit in different places:

**The hull wake** carries momentum *forward*, at ``+D`` per second: water
the hull has dragged along with it.  A boat sitting in it sees a reduced
relative velocity and less drag.  It **helps**, and it lies on the
centreline, a hull-beam wide, spreading slowly.

**The puddles** carry momentum *aft*, at ``-D`` per second: the water the
blades pushed backwards to make the thrust.  A blade placed in one is
trying to get a purchase on water already running away from it.  It
**hurts**, and it lies in two lines about three metres either side of the
centreline, at the blade's mid-drive station.

So "how much does following cost" has a geometric answer before it has a
numerical one.  A shell is 0.57 m wide at the waterline and its blades
reach out past 3 m.  Directly astern, **the follower's hull is in the
helpful half of the wake and its blades are in the harmful half** --
exactly the arrangement a coxswain is taught to avoid, now with a reason
rather than a tradition.

Puddle decay, with one free constant
------------------------------------
A puddle is a turbulent vortex ring carrying a fixed hydrodynamic
impulse.  Conservation of that impulse, plus the ring translating at its
own induced velocity, fixes the decay laws with no empiricism at all::

    I ~ rho R^3 U = const,   U = dR/dt
    =>  R ~ t^(1/4),   U ~ t^(-3/4)

This is the classical turbulent-vortex-ring result [M74]_, [GC90]_, and it
is the reason CFD is not needed for a first pass: the *shape* of the
decay is analytic.  What is not analytic is the virtual origin -- how big
the puddle is when it is born, :attr:`initial_radius` -- and everything
downstream is cubically sensitive to it, because ``U0 ~ I / (rho R0^3)``.

That single constant is the entire uncertainty here, and it is measurable
in an afternoon with two boats and two speed coaches, which is a better
use of a week than a free-surface CFD run whose aeration model would be
guessed anyway.

The impulse is not guessed
--------------------------
It follows from the momentum budget.  Over one stroke period the crew
replaces exactly the momentum drag removed, ``D * T``, shared among the
blades that took a stroke::

    I = D * T / n_blades

so a slower crew leaves weaker puddles than a fast one, in the right
proportion, with nothing fitted.

What this deliberately does not model
-------------------------------------
The surface wave field, and what it does to a crew.  Rough water costs
time mostly by making eight people row worse -- checked catches, missed
timing, a boat that will not sit level -- and that is a coupling to the
crew, not to the hull.  It is also the part CFD would answer least well
and a stopwatch would answer best.

References
----------
.. [N65] Naudascher, E. (1965) *Flow in the wake of self-propelled bodies
   and related sources of turbulence*, J. Fluid Mech. 22(4), 625-656.
.. [SP00] Sirviente, A. I. and Patel, V. C. (2000) *Wake of a
   self-propelled body*, AIAA Journal 38(4), 613-627.
.. [M74] Maxworthy, T. (1974) *Turbulent vortex rings*, J. Fluid Mech.
   64(2), 227-239.
.. [GC90] Glezer, A. and Coles, D. (1990) *An experimental study of a
   turbulent vortex ring*, J. Fluid Mech. 211, 243-283.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PuddleWake", "blade_track"]

WATER_DENSITY = 1000.0


def blade_track(boat) -> float:
    """Lateral offset of the blade's mid-drive centre of pressure, m.

    The oarlock sits ``span`` off the centreline and the blade hangs
    ``outboard`` beyond it, so at the perpendicular the centre of
    pressure is roughly ``span + 0.9 * outboard`` out, the 0.9 because
    the centre of pressure sits inboard of the blade tip.
    """
    oarlock = boat.rig.seats[0].oarlocks[0]
    span = abs(float(oarlock.position[1]))
    oar = oarlock.oar
    return span + 0.9 * (oar.length - oar.inboard)


@dataclass
class PuddleWake:
    """The aft-going half of a shell's wake, as a train of vortex rings.

    ``drag``, ``speed`` and ``period`` describe the boat being *followed*,
    so exposure scales with how hard the crew in front is working, which
    is physically right and costs nothing to get.
    """

    drag: float                     # N, the leader's total resistance
    speed: float                    # m/s
    period: float                   # s, the leader's stroke period
    n_blades: int = 8
    #: The one constant this model cannot derive.  A sweep blade is
    #: 0.11 m^2, about 0.19 m equivalent radius; the puddle it leaves is
    #: visibly larger than the blade.  Everything goes as its cube.
    initial_radius: float = 0.30
    #: What fraction of the puddle's peak velocity the blade actually
    #: works against.  A vortex ring is not a uniform slug of moving
    #: water -- it recirculates, so a blade spanning it sees an average
    #: well below the centre value, and it is only inside for part of the
    #: drive.  The second uncertain constant, and the reason the answer
    #: below is a band rather than a number.
    sampling: float = 0.5
    #: Blade slip at mid-drive, m/s.  ``None`` uses :attr:`initial_speed`,
    #: which is self-consistent -- the same momentum seen from the other
    #: side -- but a real eight shows a little more than that.
    slip: float = None
    density: float = WATER_DENSITY

    @property
    def impulse(self) -> float:
        """Momentum in one puddle, N s, from the momentum budget."""
        return self.drag * self.period / self.n_blades

    @property
    def initial_speed(self) -> float:
        """Aft-going water speed in a fresh puddle, m/s.

        A check on the whole construction rather than an output: this
        should land near the blade slip velocity a crew actually shows,
        0.5 to 1.0 m/s, because it is the same water and the same
        momentum seen from the two sides.
        """
        volume = (4.0 / 3.0) * np.pi * self.initial_radius ** 3
        return self.impulse / (self.density * volume)

    @property
    def time_scale(self) -> float:
        """Virtual origin ``R0 / U0``, s."""
        return self.initial_radius / self.initial_speed

    def velocity(self, age):
        """Aft-going water speed at puddle age ``age``, m/s."""
        return self.initial_speed * (1.0 + np.asarray(age, float)
                                     / self.time_scale) ** -0.75

    def radius(self, age):
        """Puddle radius at age ``age``, m."""
        return self.initial_radius * (1.0 + np.asarray(age, float)
                                      / self.time_scale) ** 0.25

    # -- the geometry of being behind ------------------------------------
    def spacing(self) -> float:
        """Distance along the track between consecutive puddles, m."""
        return self.speed * self.period

    def duty_cycle(self, age, blade_track_length: float = 0.5):
        """Fraction of strokes whose blades land in a puddle.

        Two crews at the same rate and speed put their blades in at the
        same *ground* positions every stroke, so being in the puddles is
        not a matter of degree: it is fixed by the gap and it repeats
        every :meth:`spacing` metres.  At a gap nobody chose deliberately
        the honest figure is the random-phase one returned here -- the
        chance that the blade's short water-track overlaps a puddle of
        the current radius.

        Half a beat of rate difference walks that phase through a whole
        cycle in a couple of minutes, which is why puddles are felt to
        come and go rather than to sit on a crew.
        """
        width = 2.0 * self.radius(age) + blade_track_length
        return np.clip(width / self.spacing(), 0.0, 1.0)

    def force_loss(self, age, slip: float = None):
        """Fractional blade-force loss for a blade that is in a puddle.

        Blade force goes as the square of the blade's velocity relative
        to the water.  Water already running aft at ``u`` takes that
        straight off the relative velocity, so the loss is first order in
        ``u / slip`` and bites hard.  This is "no grip", not "slightly
        heavier".

        :attr:`sampling` is what keeps this honest: the blade does not
        sit in a uniform slug of water moving at the puddle's centre
        velocity, so the peak value overstates the loss by roughly a
        factor of two.
        """
        if slip is None:
            slip = self.initial_speed if self.slip is None else self.slip
        effective = self.sampling * self.velocity(age)
        return np.clip(1.0 - (1.0 - effective / float(slip)) ** 2, 0.0, 1.0)

    def blades_engaged(self, spacing_of_seats: float = 1.22,
                       n_seats: int = 8) -> float:
        """Why the gap does not have to be chosen carefully in an eight.

        An eight's blades are spread over ``1.22 * 7 = 8.5`` m of boat,
        and consecutive puddles are :meth:`spacing` apart -- about 9.1 m
        at masters speed and rate 28.  The two numbers are within 7% of
        each other, so **the eight blades sample very nearly one whole
        phase cycle at once**: at any gap, one or two of the eight are in
        a puddle and the rest are clear, and shifting the gap moves which
        ones rather than how many.

        A four spans 3.7 m of a 9.1 m cycle and a single spans nothing,
        so for small boats the gap really does matter and this model
        would need the phase carried explicitly.
        """
        return float(spacing_of_seats * (n_seats - 1)) / self.spacing()

    def power_penalty(self, gap, blade_track_length: float = 0.5):
        """Mean fraction of propulsive power lost at a steady gap.

        ``gap`` is metres behind the leader's blades; the puddle age
        follows from the speed.  Random phase assumed, see
        :meth:`duty_cycle`.
        """
        age = np.asarray(gap, float) / self.speed
        return self.force_loss(age) * self.duty_cycle(age, blade_track_length)

    def momentum_area(self) -> float:
        """``D / (rho U^2)``, the wake's momentum thickness area, m^2."""
        return self.drag / (self.density * self.speed ** 2)

    def hull_benefit(self, gap, constant: float = 1.2,
                     overlap: float = 0.25):
        """Fractional drag reduction from sitting in the hull's own wake.

        The other half of the momentumless wake, on the centreline where
        the follower's hull actually is.  Self-similar axisymmetric
        turbulent wake: the centreline deficit falls as ``x^(-2/3)``
        scaled on the momentum area, which is the textbook result and
        needs no rowing-specific input at all::

            u_d / U = C (x / sqrt(theta))^(-2/3),   theta = D / (rho U^2)

        ``overlap`` is the fraction of the follower's wetted area that
        actually sits inside the deficit -- the wake is under a metre
        across at a length astern, so this is not close to one, and the
        boat is pitching and yawing through it besides.  The
        self-similar form is a *far*-wake result and should not be
        believed inside about one boat length, where it diverges.

        The first version of this used a fixed 0.3 m deep, 0.57 m wide
        wake at every distance, which does not spread and therefore does
        not decay: it returned an 8% drag reduction at a length astern,
        or half a minute over a race, which would make sitting on a
        stern the best tactic in rowing.  Wakes spread.
        """
        gap = np.maximum(np.asarray(gap, float), 1.0)
        scale = np.sqrt(self.momentum_area())
        deficit = constant * (gap / scale) ** (-2.0 / 3.0)
        return np.clip(2.0 * overlap * deficit, 0.0, 0.5)


# -- the mechanism that is not the puddles ------------------------------
#: Hancock and Bradshaw's slope: the fractional rise in skin-friction
#: coefficient per unit of their parameter ``b``.  Their data span
#: ``b`` up to about 6 and reach roughly 30% at the top of that range.
HANCOCK_SLOPE = 0.05


def boundary_layer_thickness(length: float, speed: float,
                             viscosity: float = 1.0e-6) -> float:
    """Turbulent boundary-layer thickness at the stern, m.

    ``0.37 L / Re^(1/5)``, the flat-plate power-law result.  For a 17.3 m
    shell at masters speed this is about 0.18 m, which matters because it
    is the length scale the free-stream turbulence has to be compared
    against.
    """
    reynolds = max(speed * length / viscosity, 1.0)
    return 0.37 * length / reynolds ** 0.2


class TurbulentWater:
    r"""What disturbed water does to the hull, as distinct from the blades.

    The question "does the wake go laminar or turbulent" has an
    uninteresting answer for a rowing shell: at ``Re = 6e7`` the boundary
    layer trips within the first hundred millimetres of the bow and the
    hull is turbulent over more than 99% of its length whatever the water
    is doing.  **There is no laminar run to lose.**

    The interesting question is the one the turbomachinery literature has
    already answered: a turbulent boundary layer developing under a
    *turbulent free stream* has a higher skin-friction coefficient than
    the same layer under a quiet one.  Hancock and Bradshaw [HB83]_
    collapse the effect onto

    .. math::

        b = rac{Tu\,(\%)}{L_u / \delta + 2}

    where ``L_u`` is the free-stream eddy scale and ``delta`` the boundary
    layer thickness, with ``Cf`` rising roughly linearly in ``b``.  The
    length-scale term is not decoration: eddies much smaller than the
    boundary layer do little, and Hancock and Bradshaw found they can
    even reduce the turbulence inside it.

    For a shell following another crew the numbers are not marginal.  The
    boundary layer is 0.18 m thick at the stern and the puddle eddies are
    half a metre across, so ``L_u / delta`` is around 3 and the
    denominator is small.  A couple of percent of turbulence intensity is
    then enough to move ``Cf`` by a few percent -- and viscous drag is
    over 70% of a shell's total resistance, so it goes almost straight
    through to boat speed.

    Two things follow that the puddle model alone gets wrong.  This
    penalty is **continuous**, not one stroke in six.  And it applies to
    the hull wherever the hull is in disturbed water, **including on the
    centreline**, so it eats into the hull-wake benefit rather than
    adding to it.

    References
    ----------
    .. [HB83] Hancock, P. E. and Bradshaw, P. (1983) *The effect of
       free-stream turbulence on turbulent boundary layers*, J. Fluids
       Engineering 105(3), 284-289.
    """

    def __init__(self, wake: "PuddleWake", length: float = 17.3,
                 viscous_fraction: float = 0.71):
        self.wake = wake
        self.length = float(length)
        #: Share of total resistance that is skin friction.  Measured off
        #: this hull at masters speed: 186 N viscous of 261 N total.
        self.viscous_fraction = float(viscous_fraction)
        self.delta = boundary_layer_thickness(length, wake.speed)

    def intensity(self, gap):
        """Free-stream turbulence intensity ``u' / U`` at a gap, as a
        fraction.  Taken as the puddle's own velocity scale, which is
        what the momentum budget already fixed."""
        age = np.asarray(gap, float) / self.wake.speed
        return self.wake.velocity(age) / self.wake.speed

    def parameter(self, gap):
        """Hancock and Bradshaw's ``b``."""
        age = np.asarray(gap, float) / self.wake.speed
        scale = 2.0 * self.wake.radius(age)          # eddy size ~ puddle
        return 100.0 * self.intensity(gap) / (scale / self.delta + 2.0)

    def drag_penalty(self, gap):
        """Fractional rise in total resistance from dirty water."""
        rise = HANCOCK_SLOPE * self.parameter(gap)
        return np.clip(self.viscous_fraction * rise, 0.0, 0.5)
