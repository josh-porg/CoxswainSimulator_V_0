"""The population drive law: [K05]'s on-water handle speed as the hands' drive time law."""
import numpy as np
import pytest

from coxswain.crew.drive_law import PopulationDriveSweep, k05_handle_speed, progress_in_time
from coxswain.crew.stroke import OnWaterTiming


@pytest.fixture(scope="module")
def sweep():
    return PopulationDriveSweep(catch_angle=np.radians(67.2), finish_angle=np.radians(-36.4))


def test_the_drive_runs_catch_to_finish_monotonically(sweep):
    timing = OnWaterTiming(32.4)
    t = np.linspace(0.0, 0.999 * timing.drive_duration, 400)
    phi = np.array([float(sweep(x, timing)) for x in t])
    assert phi[0] == pytest.approx(np.radians(67.2), abs=1e-9)
    assert np.all(np.diff(phi) <= 1e-12)
    end = float(sweep(timing.drive_duration, timing))
    assert end == pytest.approx(np.radians(-36.4), abs=np.radians(0.5))


def test_the_speed_along_the_path_is_the_population_profile(sweep):
    """dt = ds / v: the handle speed recovered from the law is the digitised profile's shape."""
    s, v = k05_handle_speed()
    tf, pf = progress_in_time(s, v)
    speed = np.gradient(pf, tf)
    mid = (s > 0.1) & (s < 0.9)
    ratio = speed[mid] / v[mid]
    assert np.std(ratio) / np.mean(ratio) < 0.03
    assert s[np.argmax(v)] == pytest.approx(0.65, abs=0.07)       # [K05]: max at 59-65% of the drive


def test_the_recovery_is_the_parents(sweep):
    from coxswain.crew.oarlock import OarAngleSweep
    timing = OnWaterTiming(32.4)
    base = OarAngleSweep(catch_angle=np.radians(67.2), finish_angle=np.radians(-36.4))
    t = timing.drive_duration + 0.3
    assert float(sweep(t, timing)) == pytest.approx(float(base(t, timing)), abs=1e-12)


def test_the_law_is_smooth_to_its_acceleration(sweep):
    """No spikes at the digitised table's nodes: the rate is the angle's derivative and the
    acceleration, differenced on a fine grid, changes by a small step between neighbours."""
    timing = OnWaterTiming(32.4)
    t = np.linspace(0.01, 0.99 * timing.drive_duration, 4001)
    rate = np.asarray(sweep.rate(t, timing), dtype=float)
    angle = np.array([float(sweep(x, timing)) for x in t])
    assert np.max(np.abs(np.gradient(angle, t) - rate)) < 0.01 * np.max(np.abs(rate))
    accel = np.gradient(rate, t)
    assert np.max(np.abs(np.diff(accel))) < 0.01 * np.max(np.abs(accel))


def test_the_smoothing_leaves_the_digitisation_error():
    """The spline sits within the digitisation's +-0.03 m/s of the table, no closer (s = m)."""
    from coxswain.crew.drive_law import smooth_progress
    s, v = k05_handle_speed()
    tf, pf = progress_in_time(s, v)
    progress, speed, _accel = smooth_progress(s, v)
    assert np.max(np.abs(progress(tf) - pf)) < 0.005
    assert progress(0.0) == pytest.approx(0.0, abs=1e-12)
    assert progress(1.0) == pytest.approx(1.0, abs=1e-12)
    assert np.min(speed(np.linspace(0.0, 1.0, 501))) > -1e-3


def test_turning_points_put_the_unresolved_time_at_the_ends():
    """Constant acceleration into and out of rest from the 3% points: [K05]'s drive takes 0.95 s
    against the table's 1.00 s, and the recovery 0.91 s against 0.86 s (rhythm 54%, 32.3 spm)."""
    import csv
    from coxswain.crew.drive_law import K05_FIG1, turning_samples
    s, v = k05_handle_speed()
    t, _p, _v = turning_samples(s, v)
    assert t[-1] * 1.59 == pytest.approx(1.00, rel=0.06)
    path = K05_FIG1.replace("k05_fig1_onwater", "k05_fig1_recovery")
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    sr = (1.0 - np.array([float(r["length_pct"]) for r in rows]) / 100.0)[::-1]
    vr = -np.array([float(r["handle_speed"]) for r in rows])[::-1]
    tr, _p, _v = turning_samples(sr, vr)
    assert tr[-1] * 1.59 == pytest.approx(60.0 / 32.3 * 0.46, rel=0.07)


def test_the_turning_law_keeps_the_measured_mid_drive_speeds():
    """Normalised to the table's 1.00 s drive, the hands run within 7% of [K05]'s measured speed
    through the middle of the drive (the turning ends account for 0.95 of the 1.00 s), and closer
    than the pinned law at every point, which stretched every speed by the 12% of unresolved time."""
    from coxswain.crew.drive_law import smooth_progress, turning_progress
    s, v = k05_handle_speed()
    x = np.linspace(0.0, 1.0, 4001)
    errors = {}
    for name, law in (("turning", turning_progress(s, v)), ("pinned", smooth_progress(s, v))):
        p, hand = law[0](x), law[1](x) * 1.59 / 1.00
        errors[name] = np.array([np.interp(q, p, hand) / np.interp(q, s, v) - 1.0
                                 for q in (0.3, 0.5, 0.65, 0.8)])
    assert np.all(np.abs(errors["turning"]) < 0.07)
    assert np.all(np.abs(errors["turning"]) < np.abs(errors["pinned"]))
