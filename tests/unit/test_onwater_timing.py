"""On-water single-scull timing (sprint 1 #3): Kleshnev (2005)'s two rates, and the two
athletes it was not fitted to (SOURCES sec. 160)."""
import pytest

from coxswain.crew.stroke import OnWaterTiming, StrokeTiming


@pytest.mark.parametrize("rate, rhythm, drive_s", [(20.1, 0.420, 1.26), (32.3, 0.540, 1.00)])
def test_reproduces_kleshnev_2005_on_water_single(rate, rhythm, drive_s):
    timing = OnWaterTiming(rate)
    assert timing.drive_fraction == pytest.approx(rhythm, abs=0.001)
    assert timing.drive_duration == pytest.approx(drive_s, abs=0.01)


@pytest.mark.parametrize("rate, measured", [(30.9, 0.525), (32.4093, 0.539)])
def test_predicts_athletes_it_was_not_fitted_to(rate, measured):
    """[CR06]'s single and [BR24], catch to finish by oar angle."""
    assert OnWaterTiming(rate).drive_fraction == pytest.approx(measured, abs=0.007)


def test_on_water_drive_is_longer_than_the_ergometer_fit():
    for rate in (20, 26, 32, 36):
        assert OnWaterTiming(rate).drive_fraction > StrokeTiming(rate).drive_fraction


def test_the_default_timing_is_untouched():
    assert StrokeTiming(32.0).drive_fraction == pytest.approx(0.63067 - 5.20991 / 32.0)
