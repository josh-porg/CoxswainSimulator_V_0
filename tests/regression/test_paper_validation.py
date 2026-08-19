"""Validation against the source papers and published rowing measurements.

Sources
-------
[F09]  L. Formaggia, E. Miglio, A. Mola, A. Montano, "A model for the
       dynamics of rowing boats", Int. J. Numer. Meth. Fluids 61 (2009)
       119-143.
[dL96] P. de Leva, "Adjustments to Zatsiorsky-Seluyanov's segment inertia
       parameters", J. Biomechanics 29 (1996) 1223-1230.

Anything checked here is a number that appears in a paper or in published
towing/on-water data, not a number this code produced.  Where the model
knowingly departs from [F09] the test says so and pins the departure.

See ``docs/validation.md`` for the full correspondence table.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.anthropometry import DE_LEVA_MALE, RowerAnthropometry
from coxswain.crew.stroke import StrokeTiming
from coxswain.hydro.hull import HullMesh, parametric_offsets
from coxswain.hydro.resistance import (
    FRESH_WATER,
    friction_coefficient,
    hull_resistance,
)
from coxswain.sim import RowingSimulator

pytestmark = pytest.mark.slow


# ==========================================================================
# [F09] section 5 -- stroke timing
# ==========================================================================
def test_f09_drive_duration_formula():
    """tau_a = 0.00015625 (r-24)^2 - 0.008125 (r-24) + 0.8"""
    for rate in (20.0, 24.0, 30.0, 36.0, 40.0):
        offset = rate - 24.0
        expected = 0.00015625 * offset ** 2 - 0.008125 * offset + 0.8
        assert StrokeTiming(rate).drive_duration == pytest.approx(expected,
                                                                  abs=1e-9)


def test_f09_recovery_is_period_minus_drive():
    """tau_r = 60/r - tau_a"""
    timing = StrokeTiming(34.0)
    assert timing.recovery_duration == pytest.approx(
        60.0 / 34.0 - timing.drive_duration, abs=1e-12)


# ==========================================================================
# [F09] section 4.2 / [dL96] -- the 12-segment rower
# ==========================================================================
def test_f09_uses_twelve_body_segments():
    """[F09] 4.2: "subdividing the body into p = 12 parts"."""
    assert len(RowerAnthropometry().segments) == 12


@pytest.mark.parametrize("segment,fraction", [
    ("head", 0.0694),
    ("upper_trunk", 0.1596),
    ("mid_trunk", 0.1633),
    ("lower_trunk", 0.1117),
    ("upper_arm", 0.0271),
    ("forearm", 0.0162),
    ("hand", 0.0061),
    ("thigh", 0.1416),
    ("shank", 0.0433),
    ("foot", 0.0137),
])
def test_deleva_table_4_male_mass_fractions(segment, fraction):
    assert DE_LEVA_MALE[segment].mass_fraction == pytest.approx(fraction,
                                                                abs=1e-6)


def test_deleva_mass_fractions_close():
    """head + trunk + 2 x (six limb segments) must be exactly the body."""
    total = sum(spec.mass_fraction * (2 if spec.paired else 1)
                for spec in DE_LEVA_MALE.values())
    assert total == pytest.approx(1.0, abs=2e-4)


# ==========================================================================
# [F09] section 4.3 -- the oar as an ideal lever
# ==========================================================================
def test_f09_eq12_lever_relation():
    """F_h = -(L - r_h)/L F_o, so the hull sees (r_h/L) F_o."""
    from coxswain.boats.rig import SWEEP_OAR
    handle_fraction = (SWEEP_OAR.length - SWEEP_OAR.inboard) / SWEEP_OAR.length
    assert SWEEP_OAR.gearing == pytest.approx(1.0 - handle_fraction, rel=1e-12)


def test_f09_quoted_peak_oarlock_forces():
    """[F09] 5.1: "typical values ... F_max_x = 1200 N and F_max_z = 200 N"."""
    from coxswain.crew.oarlock import OarForceProfile
    profile = OarForceProfile()
    assert profile.max_x == pytest.approx(1200.0)
    assert profile.max_z == pytest.approx(200.0)


# ==========================================================================
# [F09] section 6.1 -- resistance
# ==========================================================================
def test_f09_ittc_1957_friction_line():
    """C_f = C_f0 / (log10(Re) - 2)^2 with C_f0 = 0.075."""
    for reynolds in (1e6, 1e7, 1e8):
        expected = 0.075 / (np.log10(reynolds) - 2.0) ** 2
        assert friction_coefficient(reynolds) == pytest.approx(expected,
                                                               rel=1e-12)


def test_f09_shape_coefficient_default():
    """[F09] 6.1: "A typical value of C_dX is 0.01"."""
    from coxswain.hydro.resistance import ResistanceCoefficients
    assert ResistanceCoefficients().shape == pytest.approx(0.01)


def test_f09_literal_wave_model_is_reproducible_but_not_physical():
    """[F09] 6.1 quotes C_dw = 0.02 on |Gamma_Z|.

    Taken literally that predicts more wave drag on the paper's own
    single scull than the scull's entire measured resistance, so it is
    available as PAPER_LITERAL but is not the default.  Documented in
    docs/validation.md.
    """
    from coxswain.hydro.resistance import PAPER_LITERAL
    offsets = parametric_offsets(8.2, 0.285, 0.125, fullness=2.4,
                                 freeboard=0.20)
    mesh = HullMesh(offsets)
    heave = mesh.equilibrium_heave(100.0, rho=FRESH_WATER.density)
    props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3),
                           rho=FRESH_WATER.density)

    _, literal = hull_resistance(np.array([4.5, 0.0, 0.0]), props, 8.2,
                                 coefficients=PAPER_LITERAL)
    assert literal["wave"] > 150.0, (
        "if this drops, re-check the claim in docs/validation.md"
    )


# ==========================================================================
# [F09] section 7 -- secondary motion amplitudes
# ==========================================================================
@pytest.mark.parametrize("name,rate", [("8+", 32.0), ("4+", 32.0),
                                       ("1x", 30.0)])
def test_f09_fig13_pitch_amplitude(name, rate, simulate):
    """[F09] Fig. 13: pitch stays inside +-0.02 rad (1.15 deg).

    The figure is for a single scull; the bound is checked for every class
    because a shell that pitches more than a degree or two is wrong.
    """
    result = simulate(name, rate=rate, duration=16.0, surge_speed=4.5,
                      dt=0.006)
    assert result.pitch_amplitude() < 0.045, (
        f"{name} pitch amplitude {np.degrees(result.pitch_amplitude()):.2f} deg"
    )


@pytest.mark.parametrize("name,rate", [("8+", 32.0), ("4+", 32.0),
                                       ("1x", 30.0)])
def test_f09_fig13_heave_amplitude(name, rate, simulate):
    """[F09] Fig. 13: heave stays inside about 0.08 m peak to peak."""
    result = simulate(name, rate=rate, duration=16.0, surge_speed=4.5,
                      dt=0.006)
    assert result.heave_amplitude() < 0.15, (
        f"{name} heave amplitude {result.heave_amplitude():.3f} m"
    )


def test_f09_single_scull_hull_properties():
    """[F09] section 7: 8.2 m hull, 15 kg, 66 kg m^2 about the pitch axis."""
    boat = catalog.single_scull()
    assert boat.length == pytest.approx(8.2, abs=0.05)
    assert boat.hull_mass == pytest.approx(15.0)
    assert boat.hull_inertia[1, 1] == pytest.approx(66.0)


# ==========================================================================
# Published on-water performance
# ==========================================================================
@pytest.mark.parametrize("name,rate,low,high", [
    ("8+", 24.0, 4.3, 5.1),
    ("8+", 32.0, 5.0, 5.6),
    ("8+", 38.0, 5.4, 6.2),
    ("4+", 32.0, 4.5, 5.1),
    ("1x", 20.0, 3.3, 4.0),
    ("1x", 30.0, 4.1, 4.7),
])
def test_steady_speed_matches_published_race_pace(name, rate, low, high,
                                                  simulate):
    result = simulate(name, rate=rate, duration=22.0, surge_speed=4.2,
                      dt=0.006)
    speed = result.mean_speed()
    assert low <= speed <= high, (
        f"{name} at rate {rate} settled at {speed:.3f} m/s, outside the "
        f"published band [{low}, {high}]"
    )


def test_higher_rate_gives_higher_speed(simulate):
    speeds = [
        simulate("8+", rate=r, duration=20.0, surge_speed=4.5,
                 dt=0.006).mean_speed()
        for r in (24.0, 32.0, 38.0)
    ]
    assert speeds[0] < speeds[1] < speeds[2]


def test_an_eight_is_faster_than_a_four_at_the_same_rate(simulate):
    """More crew per unit drag: the classic ordering of the boat classes."""
    eight = simulate("8+", rate=32.0, duration=20.0, surge_speed=4.5,
                     dt=0.006).mean_speed()
    four = simulate("4+", rate=32.0, duration=20.0, surge_speed=4.5,
                    dt=0.006).mean_speed()
    assert eight > four


def test_hull_resistance_of_an_eight_matches_towing_data():
    """400-550 N at 5.5 m/s, i.e. 2.2-3.0 kW delivered to the hull."""
    boat = catalog.eight()
    heave = boat.mesh.equilibrium_heave(boat.total_mass,
                                        rho=boat.water.density)
    props = boat.mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3),
                                rho=boat.water.density)
    _, detail = hull_resistance(np.array([5.5, 0.0, 0.0]), props, boat.length,
                                boat.water, boat.resistance)
    assert 350.0 < detail["total_longitudinal"] < 620.0


def test_the_boat_runs_fastest_on_the_recovery(simulate):
    """The signature of rowing: the crew moves bow-ward on the drive, so the
    hull is checked then and surges while they come back."""
    boat = catalog.eight(rate=32.0)
    result = simulate("8+", rate=32.0, duration=20.0, surge_speed=5.0,
                      dt=0.005)

    window = result.last_cycles(2)
    phase = boat.timing.phase(result.time[window])
    speed = result.surge_speed[window]

    on_drive = phase < boat.timing.drive_fraction
    assert speed[~on_drive].mean() > speed[on_drive].mean()


# ==========================================================================
# Known departures from the paper, pinned so they cannot drift silently
# ==========================================================================
def test_speed_fluctuation_is_larger_than_measured(simulate):
    """A known, documented gap.

    The model's crew kinematics are synthesised from joint-angle profiles
    rather than fitted to motion capture, and they put more momentum swing
    into the stroke than a real crew: an eight comes out at roughly 40% of
    mean speed peak-to-peak against a measured 20-30%.  The bound here is
    deliberately generous; it exists to catch the number moving, not to
    claim it is right.  Fitting real mocap through
    ``FourierProfile.fit_samples`` is the fix.  See docs/validation.md.
    """
    result = simulate("8+", rate=32.0, duration=20.0, surge_speed=5.0,
                      dt=0.005)
    ratio = result.speed_fluctuation_ratio()
    assert 0.25 < ratio < 0.55, (
        f"speed fluctuation ratio {ratio:.3f}; if this has improved towards "
        "0.25 the crew kinematics have been recalibrated -- update "
        "docs/validation.md"
    )
