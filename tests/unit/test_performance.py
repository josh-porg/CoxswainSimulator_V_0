"""Young's scaling laws, and where this model departs from them.

Young derives every one of his rankings from ``R = C v^2 S`` with ``C``
constant.  This project no longer has that ``C``: Michell's integral
gives wave resistance its own speed dependence and ITTC-57 gives friction
another.  These tests pin the departures, because each one is a claim
about the boat that Young's algebra cannot make.

The rate law gets the most attention.  Young's eq. (33) predicts +0.87 to
+1.02% per spm over the range Holt's crews actually raced at, which lands
inside Holt's measured +0.6 to +1.1% -- meaning the rate effect this
model was accused of missing may be the power channel rather than a
missing rower.  That coincidence is worth a test of its own so it cannot
be quietly forgotten.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.hydro.michell import MichellWave, elliptical_offsets
from coxswain.sim.performance import (
    HOLT_RATE_RANGE,
    HOLT_RATE_SLOPE,
    YOUNG_AREA_FROM_WEIGHT,
    YOUNG_POWER_EXPONENT,
    YOUNG_WEIGHT_EXPONENT,
    SpeedResponse,
    young_rate_slope,
)


@pytest.fixture(scope="module")
def boat():
    return catalog.eight(rate=32.0, rower_mass=74.8, rower_stature=1.86,
                         coxswain_mass=56.7)


@pytest.fixture(scope="module")
def michell_table(boat):
    x, z, half = elliptical_offsets(boat.offsets, stations=641, levels=81)
    return MichellWave(station=x, level=z, half_beam=half).tabulate()


# --------------------------------------------------------------------------
# Young's rate law against Holt's measurement
# --------------------------------------------------------------------------
def test_young_rate_law_lands_inside_holt_measured_band():
    """The whole reason the +0.00% per spm result may not be a defect.

    Young holds work per stroke constant, so power rises with rate.  If
    that channel alone reproduces what Holt measured after adjusting for
    power, their adjustment may not have removed it.
    """
    low_rate, high_rate = HOLT_RATE_RANGE
    predicted = young_rate_slope(np.array([low_rate, high_rate]))
    holt_low, holt_high = HOLT_RATE_SLOPE
    assert np.all(predicted >= holt_low)
    assert np.all(predicted <= holt_high)


def test_young_rate_law_falls_with_rate():
    """``1/(3 SR)`` -- each extra stroke is worth less than the last."""
    rates = np.array([24.0, 32.0, 40.0])
    slopes = young_rate_slope(rates)
    assert np.all(np.diff(slopes) < 0.0)


def test_young_rate_law_is_the_cube_root_law_differentiated():
    """Consistency with ``v ~ SR^(1/3)`` rather than an independent fit."""
    rate, step = 32.0, 1e-4
    finite = (np.log((rate + step) ** (1.0 / 3.0))
              - np.log((rate - step) ** (1.0 / 3.0))) / (2.0 * step)
    assert finite == pytest.approx(young_rate_slope(rate), rel=1e-6)


# --------------------------------------------------------------------------
# the measured drag exponent
# --------------------------------------------------------------------------
def test_drag_exponent_is_near_two_but_not_two(boat):
    """ITTC friction falls with Reynolds number, so ``n < 2`` even with
    a constant wave coefficient.  Young's algebra cannot see this."""
    response = SpeedResponse(boat, wave_table=None)
    n = response.drag_exponent(5.5)
    assert 1.8 < n < 2.0
    assert n != pytest.approx(2.0, abs=0.02)


def test_constant_coefficient_gives_a_speed_independent_exponent(boat):
    """A constant wave coefficient makes wave drag a fixed fraction of a
    ``v^2`` law, so the exponent cannot move with speed.  This is the
    defect Michell was brought in to fix, pinned from the other side."""
    response = SpeedResponse(boat, wave_table=None)
    exponents = [response.drag_exponent(v) for v in (4.23, 5.5, 6.0)]
    assert max(exponents) - min(exponents) < 0.02


def test_michell_makes_the_exponent_move_with_speed(boat, michell_table):
    """Real wave resistance has humps and hollows, so ``n`` must vary.

    At masters racing speed the hull sits in the hollow Pulman predicts,
    wave drag is falling relative to ``v^2``, and the total rises more
    slowly -- a materially smaller exponent than at higher speed.
    """
    response = SpeedResponse(boat, wave_table=michell_table)
    racing = response.drag_exponent(4.23)
    faster = response.drag_exponent(5.5)
    assert racing < faster - 0.1


def test_power_is_worth_more_than_young_says(boat, michell_table):
    """``d ln v / d ln P = 1/(1+n)`` and ``n < 2``, so the return on power
    exceeds Young's 1/3 -- by most at racing speed, where the hollow is."""
    response = SpeedResponse(boat, wave_table=michell_table)
    exponent = response.power_exponent(4.23)
    assert exponent > YOUNG_POWER_EXPONENT
    assert exponent < 0.45  # not a licence for anything


# --------------------------------------------------------------------------
# weight, where Young assumes a shape this hull does not have
# --------------------------------------------------------------------------
def test_wetted_area_grows_faster_with_weight_than_similarity_says(boat):
    """Young's ``S ~ W^(1/3)`` is geometric similarity between *different*
    hulls.  Loading one hull deeper is a different question and this hull
    answers it differently."""
    response = SpeedResponse(boat)
    assert response.area_from_weight() > YOUNG_AREA_FROM_WEIGHT


def test_weight_costs_more_than_youngs_ninth_root(boat):
    """Composed from two measured halves rather than two assumed ones."""
    response = SpeedResponse(boat)
    exponent = response.weight_exponent(4.23)
    assert exponent < 0.0
    assert abs(exponent) > abs(YOUNG_WEIGHT_EXPONENT)


def test_weight_hurts_through_transverse_area_as_well_as_wetted(boat):
    """Young's route from weight to speed runs only through wetted area.

    It is not the bigger half.  Sinking a shell deeper grows the midship
    *transverse* area -- which feeds the shape term -- proportionally far
    faster than it grows wetted area, because wetted area is dominated by
    a length that does not change.  Composing Young's two steps therefore
    accounts for well under the measured weight penalty, and the shortfall
    is a channel his eq. (24) has no term for.
    """
    response = SpeedResponse(boat)
    measured = response.weight_exponent(4.23)
    wetted_only = response.area_exponent(4.23) * response.area_from_weight()
    assert measured < wetted_only < 0.0
    missing = 1.0 - wetted_only / measured
    assert 0.25 < missing < 0.60


# --------------------------------------------------------------------------
# the seconds the time budget quotes
# --------------------------------------------------------------------------
def test_seconds_per_percent_is_the_exponent_times_the_race(boat,
                                                            michell_table):
    response = SpeedResponse(boat, wave_table=michell_table)
    seconds = response.seconds_per_percent(4.23, race_time=1140.0)
    assert seconds == pytest.approx(1140.0 * response.power_exponent(4.23)
                                    * 0.01)
    # A quasi-static estimate, and knowingly below the 5.6 s the full
    # unsteady simulator measures for the same lever.  The difference is
    # the surge oscillation, which this calculation does not carry.
    assert 3.5 < seconds < 5.0


def test_area_exponent_is_negative_and_bounded_by_youngs(boat):
    """Wetted area is a lever on the viscous term only, so its exponent
    cannot exceed Young's -1/3 in magnitude."""
    response = SpeedResponse(boat, wave_table=None)
    exponent = response.area_exponent(5.5)
    assert exponent < 0.0
    assert abs(exponent) < 1.0 / 3.0
