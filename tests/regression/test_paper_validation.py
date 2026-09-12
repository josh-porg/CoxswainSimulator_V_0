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
def _f09_drive_duration(rate):
    """[F09] section 5: ``tau_a = 0.00015625 (r-24)^2 - 0.008125 (r-24) + 0.8``"""
    offset = rate - 24.0
    return 0.00015625 * offset ** 2 - 0.008125 * offset + 0.8


def test_the_model_no_longer_uses_f09s_drive_duration():
    """Deliberate, and recorded here because it cost an out-of-sample check.

    ``StrokeTiming.drive_fraction`` was changed to ``0.63067 - 5.20991/r``,
    fitted to Telfer et al. (2023) -- who measured **ergometer** rowers --
    because [F09]'s quadratic gave a catch fraction of 0.300 against
    Telfer's measured 0.394 at 22 spm.

    The trade was not free, and the tests below record what it cost. This
    one just pins that the change is real, so that nobody re-reads [F09]
    and assumes the formula in the paper is the formula in the code.
    """
    for rate in (20.0, 24.0, 30.0, 36.0, 40.0):
        timing = StrokeTiming(rate)
        assert timing.drive_fraction == pytest.approx(
            0.63067 - 5.20991 / rate, abs=1e-9)
        assert timing.drive_duration != pytest.approx(
            _f09_drive_duration(rate), rel=0.05), rate


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
@pytest.mark.slow
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
@pytest.mark.slow
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
@pytest.mark.xfail(strict=True, reason=(
    "THE TEST IS MIS-SPECIFIED, not the model. It runs each boat at the "
    "catalogue default power_scales of 1.0, which is not a race power and "
    "is not even a consistent one: 540 W per rower for an eight at 24 spm, "
    "720 at 32, 854 at 38, 795 for the four, and 1124 for a single at 30 -- "
    "against roughly 330 W that the two-parameter model says a crew can "
    "hold for a six-minute race. So every boat is driven at two to three "
    "times race power and every one of them beats published race pace, "
    "which is the only thing that could have happened. Comparing against a "
    "published pace means first putting the crew at a published POWER, and "
    "choosing that number needs a source this project does not yet have. "
    "See docs/PHYSICS_PROGRAMME.md, 'Blocked, and on what'."))
@pytest.mark.parametrize("name,rate,low,high", [
    ("8+", 24.0, 4.3, 5.1),
    ("8+", 32.0, 5.0, 5.6),
    ("8+", 38.0, 5.4, 6.2),
    ("4+", 32.0, 4.5, 5.1),
    ("1x", 20.0, 3.3, 4.0),
    ("1x", 30.0, 4.1, 4.7),
])
@pytest.mark.slow
def test_steady_speed_matches_published_race_pace(name, rate, low, high,
                                                  simulate):
    result = simulate(name, rate=rate, duration=22.0, surge_speed=4.2,
                      dt=0.006)
    speed = result.mean_speed()
    assert low <= speed <= high, (
        f"{name} at rate {rate} settled at {speed:.3f} m/s, outside the "
        f"published band [{low}, {high}]"
    )


@pytest.mark.slow
def test_scale_one_is_not_a_race_power_on_any_boat():
    """The diagnosis above, asserted so it cannot quietly stop being true.

    ``power_scales = 1.0`` is a force scale, not a wattage, and what it
    means in watts depends on the boat and the rate. Until a boat is put
    at a stated power, no comparison with a published pace means anything.
    """
    from coxswain.crew.exertion import mean_handle_power, optimal_pace

    sustainable = optimal_pace(360.0)          # a six-minute race
    watts = {}
    for name, rate in (("8+", 24.0), ("8+", 32.0), ("8+", 38.0),
                       ("4+", 32.0), ("1x", 20.0), ("1x", 30.0)):
        boat = catalog.build(name, rate=rate)
        watts[(name, rate)] = mean_handle_power(boat, samples=180)

    assert min(watts.values()) > 1.5 * sustainable, watts
    # and it is not even consistent between boats, which is the sharper point
    assert max(watts.values()) > 2.0 * min(watts.values()), watts


@pytest.mark.slow
def test_higher_rate_gives_higher_speed(simulate):
    speeds = [
        simulate("8+", rate=r, duration=20.0, surge_speed=4.5,
                 dt=0.006).mean_speed()
        for r in (24.0, 32.0, 38.0)
    ]
    assert speeds[0] < speeds[1] < speeds[2]


@pytest.mark.slow
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


@pytest.mark.slow
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
@pytest.mark.slow
def test_speed_fluctuation_is_larger_than_measured(simulate):
    """A known, documented gap -- now with a properly sourced target.

    Measured peak-to-peak surge as a fraction of mean speed:

    * **37.5%** for elite male single scullers over a 2000 m race at
      33.65 spm -- "Intracycle Velocity Variation During a Single-Sculling
      2000 m Rowing Competition", PMC12349136.  The paper's IVV is
      5.78 km/h on a mean of 15.40 km/h; **that** is the 37.5%.
      Its quoted max 21.39 and min 11.15 km/h are **whole-race**
      extremes, not intracycle, and putting them next to an intracycle
      percentage is what sent an earlier investigation down the wrong
      path -- they give 66%, which is a different quantity;
    * **41.2%** for elite females in the same study;
    * "almost 50%" for a men's pair at rate 35, from Kleshnev (2002)
      acceleration data as reported by Day et al. (2011).

    The model gives 55-65% depending on class.  The bound below is
    deliberately wide because the two published figures disagree with each
    other in the direction boat-class theory does not predict -- a pair
    should fluctuate *less* than a single, not more -- so the honest
    target is a band, not a number.

    The fluctuation is set almost entirely by the crew centre-of-mass
    velocity: momentum conservation alone predicts 50.8% of the 53.9%
    simulated.  Section 23 of docs/SOURCES.md closes the diagnosis.

    It is **not** the slide travel -- that is 0.633 m against a measured
    0.60-0.70; the earlier note blaming 0.77 m had misread the head for
    the seat.  It is not the dataset (ergometer versus on-water kinematics
    is worth 5 points of 22) and not crew synchronisation (tested: 49.4%
    at zero timing spread, 46.0% at an unrealistic 80 ms).

    It is the **shape** of the centre-of-mass velocity profile.  Real
    rowers traverse at 13% above the constant-rate floor; a cubic spline
    through four keyframes, truncated to three harmonics, lands at 48%
    above it -- essentially a sinusoid.  That is a data-resolution limit:
    four instants per stroke do not determine the shape between them.
    Closing it needs a densely sampled seat-position trace.

    See docs/SOURCES.md section 4.
    """
    result = simulate("1x", rate=33.65, duration=24.0, surge_speed=4.3,
                      dt=0.006)
    ratio = result.speed_fluctuation_ratio()
    assert 0.45 < ratio < 0.80, (
        f"single-scull speed fluctuation ratio {ratio:.3f}; measured is "
        "0.375 (PMC12349136). If this has moved towards 0.4 the crew "
        "kinematics have been recalibrated -- update docs/SOURCES.md."
    )


@pytest.mark.slow
def test_the_boat_class_ordering_of_fluctuation_is_right(simulate):
    """Longer boats run smoother.

    Day et al. and the scoping-review literature both report lower
    intracycle velocity variation in longer boats; whatever the absolute
    level, the ordering must come out right.
    """
    single = simulate("1x", rate=32.0, duration=20.0, surge_speed=4.3,
                      dt=0.006).speed_fluctuation_ratio()
    eight = simulate("8+", rate=32.0, duration=20.0, surge_speed=4.3,
                     dt=0.006).speed_fluctuation_ratio()
    assert eight < single


# ==========================================================================
# [HF09] Hill & Fahrig (2009) -- independent check on stroke timing
#
# THE MODEL NO LONGER PASSES THIS, AND THAT IS THE POINT OF KEEPING IT.
#
# These were a genuine out-of-sample validation: [F09]'s drive-duration
# quadratic, fitted to entirely different data, reproduced Hill & Fahrig's
# on-water measurements to better than 1% at racing rates.  Replacing it
# with a fit to Telfer et al.'s ERGOMETER kinematics lost that:
#
#   rate   measured   model     error    [F09] for comparison
#   20.6    862 ms    1100 ms   +27.6%    829 ms  (-3.8%)
#   24.2    810 ms    1030 ms   +27.1%    798 ms  (-1.4%)
#   27.7    779 ms     959 ms   +23.1%    772 ms  (-0.9%)
#   31.5    752 ms     886 ms   +17.9%    748 ms  (-0.6%)
#
# On the water the drive is 29.6-39.5% of the cycle; the ergometer fit
# says 37.8-46.5%.  The gap is ~0.08 of the cycle at every rate, and it
# has an obvious cause: on the water a crew has to let the boat run, and
# on a stationary ergometer there is no boat to run.  Both datasets are
# right about their own conditions -- and this model is of a boat.
#
# Marked strict-xfail rather than deleted or loosened, so it announces
# itself when the drive fraction is put right.  See
# docs/PHYSICS_PROGRAMME.md and docs/TRACKING.md.
# ==========================================================================
HF09_DRIVE_MS = {20.6: 862, 24.2: 810, 27.7: 779, 31.5: 752}


@pytest.mark.xfail(strict=True, reason=(
    "drive_fraction is fitted to Telfer et al.'s ergometer kinematics and "
    "is 18-28% longer than Hill & Fahrig measured on the water. [F09]'s "
    "quadratic, which this replaced, matched them to 0.6-3.8%."))
@pytest.mark.parametrize("rate,measured_drive_ms,tolerance_ms", [
    (20.6, 862, 40),
    (24.2, 810, 20),
    (27.7, 779, 12),
    (31.5, 752, 10),
])
def test_drive_duration_matches_measured_pairs(rate, measured_drive_ms,
                                               tolerance_ms):
    """[HF09] Table 1: eight elite coxless pairs, stepped stroke rates."""
    predicted_ms = StrokeTiming(rate).drive_duration * 1000.0
    assert predicted_ms == pytest.approx(measured_drive_ms,
                                         abs=tolerance_ms), (
        f"at {rate} spm the formula gives {predicted_ms:.0f} ms against a "
        f"measured {measured_drive_ms} ms"
    )


@pytest.mark.xfail(strict=True, reason=(
    "the ergometer-fitted drive fraction is worst at low rates and still "
    "18% out at 31.5 spm; it was better than 1% before the refit."))
def test_drive_duration_accuracy_improves_towards_racing_rates():
    """The fit is at its best where racing happens."""
    errors = {rate: abs(StrokeTiming(rate).drive_duration * 1000.0 - ms) / ms
              for rate, ms in HF09_DRIVE_MS.items()}
    assert errors[31.5] < errors[20.6]
    assert errors[31.5] < 0.01, "better than 1% at racing rate"


def test_f09s_formula_is_the_one_that_matched_the_water():
    """The evidence for the item above, asserted rather than asserted-in-prose.

    If this ever stops holding, the comparison in the comment block above
    is wrong and the conclusion drawn from it has to be revisited.
    """
    for rate, measured_ms in HF09_DRIVE_MS.items():
        f09_ms = _f09_drive_duration(rate) * 1000.0
        model_ms = StrokeTiming(rate).drive_duration * 1000.0
        assert abs(f09_ms / measured_ms - 1.0) < 0.04, (rate, f09_ms)
        assert model_ms / measured_ms - 1.0 > 0.17, (rate, model_ms)
