r"""Linear damping in all six degrees of freedom, from potential flow.

Every damping term in this model used to be **quadratic** -- cross-flow
drag of the form :math:`-\tfrac12\rho C_d A\,v|v|` -- and three of the six
degrees of freedom had no hull damping at all, because the forces were
applied as lumped resultants at the origin and a force at the origin
exerts no moment about it.

Quadratic damping alone is not a modelling simplification, it is a
qualitatively different system.  :math:`v|v|` vanishes **faster than the
energy going in** as the amplitude falls, so small motions are
effectively undamped: anything that feeds the mode grows until the
quadratic term finally catches it.  That is exactly what a coxed four did
at rate 30 -- pitch growing from 0.7 to 25 degrees, at 0.867 Hz, which is
no harmonic of the 0.5 Hz stroke and therefore the boat's own mode rather
than the forcing.

Real damping at these amplitudes is **linear**, and it comes from two
places, neither of them cross-flow drag.

Radiation damping (potential flow)
----------------------------------
A hull oscillating at a free surface radiates waves, and the energy
carried away is a damping force linear in velocity.  This is a
potential-flow result -- no viscosity, no separation -- and for a slender
hull it is obtained by strip theory: each transverse section is treated
as a two-dimensional body oscillating in an infinite free surface, and
the sectional coefficients are integrated along the length [S70]_.

For a section of beam :math:`B` oscillating at frequency :math:`\omega`,
the damping per unit length follows from the energy radiated to infinity
[N77]_:

.. math::

    b(\omega) = \frac{\rho g^{2} \bar{A}^{2}}{\omega^{3}}

with :math:`\bar{A}` the amplitude of the radiated wave per unit motion
amplitude.  Strip theory then gives the coupled matrix directly, and the
coupling is the point: a hull that heaves also pitches, because the
sectional forces act at a lever arm.

.. math::

    B_{33} = \int b_{33}\,\mathrm{d}x, \qquad
    B_{55} = \int x^{2} b_{33}\,\mathrm{d}x, \qquad
    B_{35} = B_{53} = -\int x\, b_{33}\,\mathrm{d}x

and likewise :math:`B_{22}`, :math:`B_{66}`, :math:`B_{26}` from the
sway sectional damping.

**Where this is weak.** :math:`\bar{A}` depends on section shape and
frequency, and this project has no measurement of it for a racing shell.
It is a *parameter, not a result*: see :data:`HEAVE_AMPLITUDE_RATIO`, and
:func:`damping_report` for what any choice implies in fractions of
critical damping, which is the number to argue about.

Roll is different, and potential flow will not do it
----------------------------------------------------
A rowing shell is 0.15 m deep and 0.5 m wide.  A section that shallow
rolling about its own waterline displaces almost no fluid vertically, so
it radiates almost no waves: **potential-flow roll damping is
negligible**, and computing only that would leave roll essentially
undamped -- which is wrong, and would be wrong in the same way the pitch
bug was wrong.

Roll damping is viscous and lift-driven, and the standard treatment is
Ikeda's component method, which is what the ITTC recommends [I78]_,
[HK12]_.  It splits the damping into friction, eddy-making, lift, wave
and bilge-keel parts.  For this hull:

* **Lift** dominates.  A hull with forward speed rolling about a
  longitudinal axis behaves like a low-aspect-ratio lifting surface, and
  the damping is linear in :math:`U`.  At 4-5 m/s that is the largest
  term by far.  There are no bilge keels on a shell.
* **Friction** is small but not zero, and is the only term that survives
  at zero speed.
* **Eddy-making** falls away rapidly with forward speed [I78]_ and a
  round-bilge shell sheds little; it is not modelled, which makes this
  estimate a **floor** on roll damping rather than a best estimate.

What this module does *not* replace
-----------------------------------
The quadratic cross-flow terms in :mod:`coxswain.hydro.crossflow` and
:mod:`coxswain.hydro.heaveflow` stay.  They are viscous, they dominate at
large amplitude, and they are what holds a hull straight in a turn.  This
adds the linear part that dominates at small amplitude.  Both are real
and they act at different scales.

References
----------
.. [S70] Salvesen, N., Tuck, E. O. & Faltinsen, O. (1970) *Ship motions
   and sea loads*, Trans. SNAME 78, 250-287 -- the strip-theory
   assembly of the coupled damping matrix used here.
.. [N77] Newman, J. N. (1977) *Marine Hydrodynamics*, MIT Press, ch. 6 --
   the radiated-energy relation between damping and wave amplitude.
.. [F90] Faltinsen, O. M. (1990) *Sea Loads on Ships and Offshore
   Structures*, Cambridge, ch. 3 -- sectional added mass and damping,
   and the reduced-frequency dependence of the amplitude ratio.
.. [I78] Ikeda, Y., Himeno, Y. & Tanaka, N. (1978) *A prediction method
   for ship roll damping*, Report 00405, Dept. Naval Architecture,
   University of Osaka Prefecture -- the component method.
.. [HK12] Himeno, Y. (1981) *Prediction of ship roll damping: state of
   the art*, Report 239, Dept. Naval Architecture, University of
   Michigan; and Falzarano, J. et al. (2015) *An overview of the
   prediction methods for roll damping of ships*, Ocean Systems
   Engineering 5(2), 55-76 -- review and the ITTC recommendation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["StripDamping", "HEAVE_AMPLITUDE_RATIO", "SWAY_AMPLITUDE_RATIO",
           "damping_report", "natural_frequencies"]

GRAVITY = 9.80665

#: Radiated wave amplitude per unit heave amplitude, non-dimensional.
#:
#: The weak number in this module.  For a wall-sided section at the
#: reduced frequency :math:`\omega^{2}B/2g \approx 0.4` where a shell
#: sits, published curves run about 0.4-0.7 [F90]_.  0.55 is the middle.
HEAVE_AMPLITUDE_RATIO = 0.55

#: The same for sway.  Much smaller: a shallow section moving sideways
#: pushes very little water *up*, and it is vertical displacement of the
#: free surface that radiates.  Scaled by draft/beam below rather than
#: given a separate constant, because that ratio is what controls it.
SWAY_AMPLITUDE_RATIO = 0.15


class StripDamping:
    """The linear 6x6 damping matrix of a hull, in the hull frame.

    Ordered ``(surge, sway, heave, roll, pitch, yaw)`` to match
    :class:`coxswain.hydro.addedmass.AddedMass`.

    Surge is left at zero: a slender hull surging radiates almost
    nothing, and the longitudinal resistance is already modelled properly
    by Michell's integral plus the ITTC friction line in
    :mod:`coxswain.hydro.resistance`.  Putting a radiation term there too
    would double-count the wave-making that is the whole point of the
    Michell calculation.
    """

    def __init__(self, offsets, n_strips: int = 41):
        x = np.asarray(offsets.station, dtype=float)
        self.station = np.linspace(float(x[0]), float(x[-1]), int(n_strips))
        self.beam = np.interp(self.station, x,
                              np.asarray(offsets.beam, dtype=float))
        self.draft = np.interp(self.station, x,
                               np.asarray(offsets.depth, dtype=float))
        self.length = float(x[-1] - x[0])
        self.max_beam = float(self.beam.max())
        self.max_draft = float(self.draft.max())
        self.plan_area = float(np.trapezoid(self.beam, self.station))
        self.lateral_area = float(np.trapezoid(self.draft, self.station))
        # Wetted surface, approximated sectionally as girth times length.
        girth = self.beam + 2.0 * self.draft
        self.wetted_area = float(np.trapezoid(girth, self.station))

    # -- sectional potential-flow coefficients ---------------------------
    def sectional_heave(self, frequency: float, rho: float,
                        amplitude_ratio: float = HEAVE_AMPLITUDE_RATIO):
        """``b_33`` per unit length, N s/m^2.

        Scaled by local beam over maximum beam: a fine section radiates
        less than the midship one, because it displaces less surface.
        """
        if frequency <= 0.0:
            return np.zeros_like(self.beam)
        peak = max(self.max_beam, 1e-9)
        return (rho * GRAVITY ** 2 * amplitude_ratio ** 2 / frequency ** 3
                * (self.beam / peak) ** 2)

    def sectional_sway(self, frequency: float, rho: float,
                       amplitude_ratio: float = SWAY_AMPLITUDE_RATIO):
        """``b_22`` per unit length, N s/m^2.

        Weighted by the section's draft-to-beam ratio: swaying radiates
        only through the surface elevation it pushes up, and a shallow
        wide section pushes up very little.
        """
        if frequency <= 0.0:
            return np.zeros_like(self.beam)
        slenderness = self.draft / np.maximum(self.beam, 1e-9)
        return (rho * GRAVITY ** 2 * amplitude_ratio ** 2 / frequency ** 3
                * slenderness ** 2)

    # -- Ikeda's roll components -----------------------------------------
    def roll_lift(self, speed: float, rho: float,
                  vertical_centre: float = 0.0) -> float:
        """Ikeda's lift component of roll damping, N m s/rad.

        A hull with forward speed rolling about a longitudinal axis acts
        as a very low aspect-ratio lifting surface.  Linear in ``U`` and
        independent of roll frequency [I78]_.

        ``vertical_centre`` is ``OG``: the distance from the waterline
        down to the centre of gravity, positive downwards.
        """
        speed = abs(float(speed))
        if speed <= 0.0 or self.max_draft <= 0.0:
            return 0.0
        draft, length, beam = self.max_draft, self.length, self.max_beam
        # k_N, the lift-slope term.  kappa is 0 for the fine midship
        # sections of a shell (Ikeda ties it to the midship coefficient,
        # and it only becomes non-zero above C_m = 0.92).
        k_n = 2.0 * np.pi * draft / length + 0.0
        lever_0 = 0.3 * draft
        lever_r = 0.5 * draft
        og = float(vertical_centre)
        shape = (lever_0 * lever_r
                 + 1.4 * og * lever_0
                 + 0.7 * og ** 2 * lever_0 / max(lever_r, 1e-9))
        return 0.5 * rho * speed * length * draft * k_n * shape

    def roll_friction(self, frequency: float, amplitude: float, rho: float,
                      viscosity: float = 1.0e-6) -> float:
        """Kato's friction component of roll damping, N m s/rad.

        Small, and the only component that survives at zero speed.
        ``amplitude`` is the roll amplitude in radians, because skin
        friction is quadratic and this is its equivalent linearisation --
        so the coefficient depends on how hard the boat is rolling.
        """
        if frequency <= 0.0 or amplitude <= 0.0:
            return 0.0
        # Effective bilge radius: the distance from the roll axis to the
        # representative wetted point.
        radius = 0.5 * np.hypot(self.max_beam, 2.0 * self.max_draft)
        reynolds = (radius ** 2 * frequency * amplitude) / viscosity
        if reynolds <= 0.0:
            return 0.0
        friction = 1.328 * np.sqrt(2.0 * np.pi / max(reynolds, 1e-9))
        return ((4.0 / (3.0 * np.pi)) * rho * self.wetted_area
                * radius ** 3 * frequency * amplitude * friction)

    # -- assembly ---------------------------------------------------------
    def matrix(self, frequency: float, rho: float, speed: float = 0.0,
               roll_amplitude: float = np.radians(2.0),
               vertical_centre: float = 0.0,
               heave_ratio: float = HEAVE_AMPLITUDE_RATIO,
               sway_ratio: float = SWAY_AMPLITUDE_RATIO,
               immersion: float = 1.0) -> np.ndarray:
        """The 6x6 linear damping matrix, N s/m and N m s/rad.

        ``frequency`` is the oscillation frequency in rad/s.  A
        time-domain model cannot carry a frequency-dependent coefficient
        without a convolution, so it is evaluated once at the mode's own
        natural frequency -- the standard constant-coefficient
        simplification, and the right frequency because that is the mode
        the damping has to control.
        """
        matrix = np.zeros((6, 6))
        if frequency <= 0.0:
            return matrix

        heave = self.sectional_heave(frequency, rho, heave_ratio) * immersion
        sway = self.sectional_sway(frequency, rho, sway_ratio) * immersion
        x = self.station

        # heave / pitch, coupled through the lever arm
        matrix[2, 2] = float(np.trapezoid(heave, x))
        matrix[4, 4] = float(np.trapezoid(x ** 2 * heave, x))
        matrix[2, 4] = matrix[4, 2] = -float(np.trapezoid(x * heave, x))

        # sway / yaw, likewise
        matrix[1, 1] = float(np.trapezoid(sway, x))
        matrix[5, 5] = float(np.trapezoid(x ** 2 * sway, x))
        matrix[1, 5] = matrix[5, 1] = float(np.trapezoid(x * sway, x))

        # roll: viscous and lift, because potential flow gives it nothing
        matrix[3, 3] = (self.roll_lift(speed, rho, vertical_centre)
                        + self.roll_friction(frequency, roll_amplitude, rho))
        return matrix


def natural_frequencies(offsets, mass: float, inertia, rho: float,
                        metacentric_height: float = 0.0) -> dict:
    r"""Undamped natural frequencies in heave, pitch and roll, rad/s.

    Restoring stiffness from the waterplane, inertia from the boat plus
    the strip-theory added mass in :mod:`coxswain.hydro.addedmass`.  Both
    halves come from the model's own geometry, so nothing here is a
    tuned number:

    .. math::

        C_{33} = \rho g A_{wp}, \qquad
        C_{55} = \rho g \!\int\! x^{2} b(x)\,\mathrm{d}x, \qquad
        C_{44} = \rho g \nabla \, \overline{GM}

    Roll is returned as ``None`` unless a metacentric height is given.
    A racing shell's ``GM`` is set by the crew, not the hull -- four
    people with their mass a foot above the waterline -- so it is not a
    property this table can derive, and guessing it would be worse than
    saying so.
    """
    from .addedmass import AddedMass

    strip = StripDamping(offsets)
    added = AddedMass.from_offsets(offsets, rho=rho).matrix
    x, beam = strip.station, strip.beam
    inertia = np.asarray(inertia, dtype=float)

    waterplane = strip.plan_area
    second_moment = float(np.trapezoid(x ** 2 * beam, x))
    volume = float(np.trapezoid(0.25 * np.pi * beam * strip.draft, x))

    def frequency(stiffness, generalised):
        if stiffness <= 0.0 or generalised <= 0.0:
            return 0.0
        return float(np.sqrt(stiffness / generalised))

    out = {
        "heave": frequency(rho * GRAVITY * waterplane, mass + added[2, 2]),
        "pitch": frequency(rho * GRAVITY * second_moment,
                           inertia[1, 1] + added[4, 4]),
        "roll": None,
    }
    if metacentric_height > 0.0:
        out["roll"] = frequency(
            rho * GRAVITY * volume * float(metacentric_height),
            inertia[0, 0] + added[3, 3])
    return out


def damping_report(offsets, mass, inertia, rho: float, speed: float = 0.0,
                   frequency: float = None, **kwargs) -> dict:
    """Damping as a fraction of critical, per degree of freedom.

    The number to quote and to argue about.  A coefficient in N s/m means
    nothing on its own; ``zeta`` says whether the boat is realistically
    damped, and published values for slender ships put heave and pitch
    around 0.1-0.4 and roll around 0.02-0.10.

    Each mode is evaluated at **its own** natural frequency and against
    **its own** generalised inertia, including added mass.  Doing it with
    one frequency and the bare hull inertia reported pitch as 1.37 of
    critical -- over-damped, which it very much is not -- because the
    hull of a four is 51 kg and the thing that actually resists pitching
    is four rowers and the water they have to move.
    """
    strip = StripDamping(offsets)
    inertia = np.asarray(inertia, dtype=float)
    modes = natural_frequencies(offsets, mass, inertia, rho,
                                metacentric_height=kwargs.pop(
                                    "metacentric_height", 0.0))

    from .addedmass import AddedMass
    added = AddedMass.from_offsets(offsets, rho=rho).matrix
    generalised = np.array([mass + added[0, 0], mass + added[1, 1],
                            mass + added[2, 2],
                            inertia[0, 0] + added[3, 3],
                            inertia[1, 1] + added[4, 4],
                            inertia[2, 2] + added[5, 5]])

    # Sway and yaw have no restoring force, so no natural frequency of
    # their own; they are reported at the heave frequency purely so the
    # radiation term has somewhere to be evaluated, and the number should
    # be read as indicative only.
    reference = modes["heave"] or 1.0
    per_mode = {"surge": reference, "sway": reference,
                "heave": modes["heave"] or reference,
                "roll": modes["roll"] or reference,
                "pitch": modes["pitch"] or reference,
                "yaw": reference}

    out = {"frequencies": modes}
    for index, name in enumerate(("surge", "sway", "heave",
                                  "roll", "pitch", "yaw")):
        omega = float(frequency or per_mode[name])
        matrix = strip.matrix(omega, rho, speed=speed, **kwargs)
        critical = 2.0 * generalised[index] * omega
        out[name] = (float(matrix[index, index] / critical)
                     if critical > 0.0 else 0.0)
    return out
