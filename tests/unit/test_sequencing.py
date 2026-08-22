"""The §29 diagnosis, as executable checks.

Two of these validate the *target* -- that the measured 37.3% of
intracycle velocity variation is a sound number to fit to.  The third
pins the *defect*: the model's segments move in phase when real rowing
sequences them, and that is what inflates the hull surge.
"""
import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.kinematics import SEGMENT_ORDER
from coxswain.data.telemetry import StrokeTrace

MEASURED_IVV = 0.373


def _segment_x(rower, period, n=801):
    t = np.linspace(0.0, period, n)
    return t, np.array([np.asarray(rower.segment_state(x)[0])[:, 0]
                        for x in t])


def test_ten_hertz_sampling_does_not_corrupt_the_ivv_target():
    """IVV is peak-to-peak, so sampling should bias it *low*.  It barely does.

    The DGPS trace is 10 Hz on a 2.5 s cycle.  Sampling a waveform with
    the measured peakiness at that rate, over every phase offset, must
    recover the true peak-to-peak to within a couple of percent -- else
    the target itself would be an artefact.
    """
    period, mean_speed, true_ivv = 2.5, 4.5, MEASURED_IVV

    def wave(t, p=0.45):
        base = -np.cos(2 * np.pi * t / period)
        return np.sign(base) * np.abs(base) ** p

    fine = wave(np.linspace(0.0, period, 20000, endpoint=False))
    amp = true_ivv * mean_speed / (fine.max() - fine.min())

    got = []
    for phase in np.linspace(0.0, period, 41, endpoint=False):
        t = np.arange(phase, phase + 8 * period, 0.1)
        v = mean_speed + amp * wave(t % period)
        got.append(StrokeTrace(time=t, velocity=v)
                   .intracycle_variation(period) / mean_speed)
    assert abs(np.mean(got) - true_ivv) / true_ivv < 0.02


@pytest.mark.parametrize("drive_fraction, expected", [(0.33, 0.386),
                                                      (0.40, 0.356)])
def test_momentum_balance_alone_reproduces_the_measured_ivv(drive_fraction,
                                                            expected):
    """No hydrodynamics, no blade model: conservation of momentum only.

    If this did *not* bracket the measured value, the target would be
    suspect.  It does, from two independent directions, which is what
    licenses treating 37.3% as ground truth in §29.
    """
    boat = catalog.single_scull(rate=24.0)
    crew = sum(float(np.sum(c.rower.segment_masses)) for c in boat.crew)
    frac = crew / (crew + boat.hull_mass)
    period, travel, mean_speed = boat.timing.period, 0.72, 4.5

    swing = (np.pi / 2) * travel * (1.0 / (drive_fraction * period)
                                    + 1.0 / ((1 - drive_fraction) * period))
    assert frac * swing / mean_speed == pytest.approx(expected, abs=0.01)


def test_segment_masses_match_de_leva():
    """Guards the §29 finding that the mass vector is *not* the bug."""
    rower = catalog.single_scull(rate=24.0).crew[0].rower
    m = np.asarray(rower.segment_masses, float)
    share = dict(zip(SEGMENT_ORDER, m / m.sum()))
    trunk = sum(v for k, v in share.items() if "trunk" in k)
    assert share["head"] == pytest.approx(0.069, abs=0.015)
    assert trunk == pytest.approx(0.435, abs=0.03)
    assert share["thigh_port"] == pytest.approx(0.142, abs=0.02)
    assert share["shank_foot_port"] == pytest.approx(0.057, abs=0.015)


def test_crew_com_travel_is_the_quantity_that_sets_the_surge():
    """Seat travel is right; COM travel is not, and COM travel is what counts."""
    rower = catalog.single_scull(rate=24.0).crew[0].rower
    period = rower.timing.period
    m = np.asarray(rower.segment_masses, float)
    _, X = _segment_x(rower, period)
    com = (m[None, :] * X).sum(axis=1) / m.sum()

    assert 0.60 <= rower.slide_travel() <= 0.70      # matches literature
    # and yet the centre of mass travels far further than the 0.635 m
    # that the measured IVV implies
    assert (com.max() - com.min()) > 0.75


@pytest.mark.xfail(reason="§29: segments move in phase; the fix is §30's "
                          "rower model. Expected to pass once sequencing "
                          "is represented.",
                   strict=True)
def test_segments_are_sequenced_not_synchronous():
    """Real rowing sequences legs, then trunk, then arms.

    Measured as the cancellation between the in-phase upper bound on
    crew-COM travel and what the kinematics actually realise.  The model
    currently achieves 0.3%; reaching the measured IVV needs ~22%.
    """
    rower = catalog.single_scull(rate=24.0).crew[0].rower
    period = rower.timing.period
    m = np.asarray(rower.segment_masses, float)
    _, X = _segment_x(rower, period)

    travel = X.max(axis=0) - X.min(axis=0)
    in_phase = float((m * travel).sum() / m.sum())
    com = (m[None, :] * X).sum(axis=1) / m.sum()
    cancellation = (in_phase - (com.max() - com.min())) / in_phase

    assert cancellation > 0.15
