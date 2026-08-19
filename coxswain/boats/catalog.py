"""Ready-made boats.

Each entry is a factory so that stroke rate, crew weights and water can be
varied without mutating shared state.  Hull dimensions and rigging come
from World Rowing minimum-weight rules and standard rigging practice;
inertias are estimated from the hull mass and a slender-body radius of
gyration, and are documented per boat.

Adding a boat means adding a factory here.  Nothing in
:mod:`coxswain.sim` needs to change.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..crew.anthropometry import PORT, RowerAnthropometry
from ..crew.oarlock import OarForceProfile
from ..crew.stroke import StrokeTiming
from ..hydro.appendages import LiftingSurface
from ..hydro.hull import parametric_offsets
from ..hydro.resistance import FRESH_WATER, WaterProperties
from .boat import Boat
from .rig import SCULLING_OAR, SWEEP_OAR, build_sculling_rig, build_sweep_rig

__all__ = ["eight", "coxed_four", "single_scull", "CATALOG", "build"]


def _slender_inertia(mass: float, length: float, beam: float,
                     depth: float) -> np.ndarray:
    """Hull inertia about ``G_h``, treating the shell as a thin rod.

    Roll inertia uses the section half-breadth as the radius of gyration;
    pitch and yaw use ``L / sqrt(12)``, exact for a uniform rod.  The
    shell is not uniform -- mass concentrates amidships -- so a 0.8
    correction is applied, which is the usual naval rule of thumb for a
    fine-ended hull.
    """
    roll = mass * (0.5 * beam) ** 2 / 2.0
    longitudinal = 0.8 * mass * length ** 2 / 12.0
    yaw = longitudinal
    pitch = longitudinal + mass * (0.5 * depth) ** 2
    return np.diag([roll, pitch, yaw])


def _skeg(x: float, span: float = 0.129, chord: float = 0.202,
          depth: float = -0.18) -> LiftingSurface:
    return LiftingSurface(span=span, chord=chord,
                          position=np.array([x, 0.0, depth]),
                          sweep=np.radians(5.0))


def _rudder(x: float, span: float = 0.090, chord: float = 0.120,
            depth: float = -0.16) -> LiftingSurface:
    return LiftingSurface(span=span, chord=chord,
                          position=np.array([x, 0.0, depth]),
                          controllable=True)


#: Peak longitudinal oarlock force per oar, by boat class.
#:
#: Formaggia et al. quote 1200 N as typical, measured on a *sculler's*
#: oarlock (two oars per athlete).  A sweep athlete drives one oar, and
#: what matters downstream is that the resulting mean thrust balances the
#: modelled resistance at the boat's known cruising speed.  These values
#: are calibrated to do that: an eight at rate 32 settles at about
#: 5.2 m/s, which is where an eight actually cruises at that rate.
PEAK_OARLOCK_FORCE = {"8+": 950.0, "4+": 1050.0, "1x": 700.0}


def eight(rate: float = 32.0, rower_mass: float = 88.0,
          rower_stature: float = 1.90, coxswain_mass: float = 55.0,
          water: WaterProperties = FRESH_WATER,
          crew_phase_offsets: Sequence[float] = None, **kwargs) -> Boat:
    """Coxed eight (8+).

    17.3 m, 0.57 m waterline beam.  World Rowing minimum hull mass is
    96 kg.  Sweep rigged, alternating from a port-side stroke, 1.22 m
    between seats, 0.85 m span.
    """
    length, beam, draft = 17.30, 0.57, 0.165
    hull_mass = 96.0
    offsets = parametric_offsets(length, beam, draft, fullness=2.6,
                                 freeboard=0.26)
    rig = build_sweep_rig(
        n_seats=8, spacing=1.22, stern_station=-4.30, span=0.85,
        oarlock_height=0.38, oar=SWEEP_OAR, stroke_side=PORT,
        coxswain_position=np.array([-6.10, 0.0, 0.10]),
        coxswain_mass=coxswain_mass,
    )
    return Boat(
        name="eight (8+)",
        offsets=offsets,
        rig=rig,
        hull_mass=hull_mass,
        hull_inertia=_slender_inertia(hull_mass, length, beam, draft),
        timing=StrokeTiming(rate),
        appendages=(_skeg(-6.0), _rudder(-6.6)),
        water=water,
        crew_phase_offsets=crew_phase_offsets,
        default_anthropometry=RowerAnthropometry(mass=rower_mass,
                                                 stature=rower_stature),
        **{"force_profile": OarForceProfile(
            max_x=PEAK_OARLOCK_FORCE["8+"]), **kwargs},
    )


def coxed_four(rate: float = 32.0, rower_mass: float = 88.0,
               rower_stature: float = 1.90, coxswain_mass: float = 55.0,
               water: WaterProperties = FRESH_WATER,
               crew_phase_offsets: Sequence[float] = None, **kwargs) -> Boat:
    """Coxed four (4+).

    13.4 m, 0.50 m waterline beam, 51 kg minimum hull mass.  Sweep
    rigged, 1.22 m between seats, 0.83 m span.  A shorter, slightly
    beamier hull than the eight for its displacement, so it sits deeper
    and is relatively draggier -- which the model reproduces rather than
    being told.
    """
    length, beam, draft = 13.40, 0.50, 0.155
    hull_mass = 51.0
    offsets = parametric_offsets(length, beam, draft, fullness=2.6,
                                 freeboard=0.25)
    rig = build_sweep_rig(
        n_seats=4, spacing=1.22, stern_station=-2.20, span=0.83,
        oarlock_height=0.38, oar=SWEEP_OAR, stroke_side=PORT,
        coxswain_position=np.array([-4.00, 0.0, 0.10]),
        coxswain_mass=coxswain_mass,
    )
    return Boat(
        name="coxed four (4+)",
        offsets=offsets,
        rig=rig,
        hull_mass=hull_mass,
        hull_inertia=_slender_inertia(hull_mass, length, beam, draft),
        timing=StrokeTiming(rate),
        appendages=(_skeg(-4.7), _rudder(-5.2)),
        water=water,
        crew_phase_offsets=crew_phase_offsets,
        default_anthropometry=RowerAnthropometry(mass=rower_mass,
                                                 stature=rower_stature),
        **{"force_profile": OarForceProfile(
            max_x=PEAK_OARLOCK_FORCE["4+"]), **kwargs},
    )


def single_scull(rate: float = 30.0, rower_mass: float = 85.0,
                 rower_stature: float = 1.86,
                 water: WaterProperties = FRESH_WATER, **kwargs) -> Boat:
    """Single scull (1x).

    8.2 m and 15 kg, matching the hull Formaggia et al. section 7 use for
    their validation case, so the regression suite can compare against
    their published velocity traces.
    """
    length, beam, draft = 8.20, 0.285, 0.125
    hull_mass = 15.0
    offsets = parametric_offsets(length, beam, draft, fullness=2.4,
                                 freeboard=0.20)
    rig = build_sculling_rig(
        n_seats=1, spacing=1.22, stern_station=-0.35, span=0.80,
        oarlock_height=0.32, oar=SCULLING_OAR,
    )
    # the paper quotes 66 kg m^2 about the pitching axis for this hull
    inertia = np.diag([hull_mass * (0.5 * beam) ** 2 / 2.0, 66.0, 66.0])
    return Boat(
        name="single scull (1x)",
        offsets=offsets,
        rig=rig,
        hull_mass=hull_mass,
        hull_inertia=inertia,
        timing=StrokeTiming(rate),
        appendages=(_skeg(-3.0, span=0.10, chord=0.16, depth=-0.14),),
        water=water,
        default_anthropometry=RowerAnthropometry(mass=rower_mass,
                                                 stature=rower_stature),
        **{"force_profile": OarForceProfile(
            max_x=PEAK_OARLOCK_FORCE["1x"]), **kwargs},
    )


#: Name -> factory, for configuration-driven use.
CATALOG = {
    "8+": eight,
    "eight": eight,
    "4+": coxed_four,
    "coxed_four": coxed_four,
    "1x": single_scull,
    "single": single_scull,
}


def build(name: str, **kwargs) -> Boat:
    """Build a boat by catalogue name."""
    try:
        factory = CATALOG[name]
    except KeyError:
        raise KeyError(
            f"unknown boat {name!r}; available: {sorted(set(CATALOG))}"
        ) from None
    return factory(**kwargs)
