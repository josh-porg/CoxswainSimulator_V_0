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
    # and yet the centre of mass travels further than the ~0.65 m that
    # the measured IVV implies.  Section 30 removed the interpolant
    # overshoot, taking this from 0.811 to 0.751; the rest is section 31.
    assert 0.70 < (com.max() - com.min()) < 0.78


def test_reconstruction_does_not_exceed_the_data_it_interpolates():
    """Section 30: the periodic cubic spline overshot its own keyframes.

    It inflated the trunk link swing from 54.7 to 62.4 degrees and the
    shank from 76.9 to 85.2 -- 11 to 14% on every joint excursion, which
    propagated into crew centre-of-mass travel and hence the hull surge.
    A shape-preserving interpolant cannot do this, which is the whole
    point of using one.
    """
    rower = catalog.single_scull(rate=24.0).crew[0].rower
    data = rower.dataset
    t = np.linspace(0.0, rower.timing.period, 1501)

    for profile, keyframes in ((rower.joint_angles.shank, data.shank),
                               (rower.joint_angles.trunk, data.trunk_link)):
        measured = np.asarray(keyframes, dtype=float)
        rebuilt = np.degrees([profile(x).value for x in t])
        # truncating to a few harmonics may undershoot slightly; it must
        # never overshoot the data it is interpolating
        assert (rebuilt.max() - rebuilt.min()) <= (
            measured.max() - measured.min()) * 1.02


def test_trunk_lagging_the_legs_reduces_the_fluctuation():
    """Sequencing is a real lever -- but only once section 30's fix is in.

    Measured through the overshooting interpolant, every warp made the
    fluctuation worse and sequencing looked like a dead end.  With the
    shape-preserving interpolant the anatomically correct direction --
    legs at nominal, trunk lagging -- reduces the crew centre-of-mass
    velocity swing, which is what the drive sequence is supposed to do.
    """
    from coxswain.crew.kinematics import JointDrivenRower, SegmentSequencing

    r = catalog.single_scull(rate=24.0).crew[0].rower

    def swing(seq):
        built = JointDrivenRower(
            r.anthropometry, r.station, r.timing, dataset=r.dataset,
            thigh_mode=r.thigh_mode, hand_targets=r.hand_targets,
            n_harmonics=r.n_harmonics,
            recovery_arrival=r.recovery_arrival, sequencing=seq)
        t, X = _segment_x(built, built.timing.period)
        m = np.asarray(built.segment_masses, float)
        com = (m[None, :] * X).sum(axis=1) / m.sum()
        return float(np.ptp(np.gradient(com, t)))

    reference = swing(None)
    lagged = swing(SegmentSequencing(shank=0.0, thigh=0.0, trunk=-0.15))
    assert lagged < reference

    # and the ordering constraint is enforced: the trunk may not lead
    with pytest.raises(ValueError, match="legs must lead"):
        SegmentSequencing(shank=-0.10, thigh=-0.10, trunk=0.10).validate()


def test_double_scull_exists_for_like_for_like_validation():
    """Section 32: the measured fluctuation is a 2x, so the model needs one.

    Comparing a 1x model against 2x telemetry cost 8.6 points of apparent
    disagreement, because IVV is normalised by mean speed and a 2x is
    faster at the same rate.
    """
    boat = catalog.double_scull(rate=24.0)
    assert boat.n_seats == 2
    assert boat.hull_mass == pytest.approx(27.0)      # World Rowing minimum
    assert 10.0 < boat.length < 11.0
    crew = sum(float(np.sum(c.rower.segment_masses)) for c in boat.crew)
    # crew mass fraction must be close to the 1x, or the momentum coupling
    # that carries the whole diagnosis does not transfer between classes
    single = catalog.single_scull(rate=24.0)
    one = sum(float(np.sum(c.rower.segment_masses)) for c in single.crew)
    assert (crew / (crew + boat.hull_mass)) == pytest.approx(
        one / (one + single.hull_mass), abs=0.02)


def test_hull_velocity_waveform_matches_the_dgps_shape():
    """SOURCES sec. 57: the model's within-stroke velocity waveform is
    validated against the DGPS profile (37 catch-aligned cycles) at
    r = 0.92.  The waveform is textbook: minimum after the catch, rising
    through the drive, maximum near the finish, drag-rate decay through
    the recovery.  Guard the qualitative structure so it cannot silently
    regress; the full quantitative check lives in scripts/validate.py.
    """
    from coxswain.sim.simulator import RowingSimulator

    boat = catalog.double_scull(rate=24.0)
    period = boat.timing.period
    res = RowingSimulator(boat).run(duration=9 * period, dt=0.005,
                                    surge_speed=3.8)
    t = res.time
    v = np.asarray(res.speed, float)
    keep = t >= 7 * period
    phase = ((t[keep] - t[keep][0]) % period) / period
    bins = np.linspace(0.0, 1.0, 49)
    prof = np.array([v[keep][(phase >= bins[i]) & (phase < bins[i + 1])].mean()
                     for i in range(48)])
    prof = np.roll(prof - prof.mean(), -int(np.argmin(prof)))
    u = (np.arange(48) + 0.5) / 48
    drive = boat.timing.drive_fraction

    # minimum at the catch (by construction of the alignment), and the
    # maximum must fall in the last third of the drive or early recovery
    # -- NOT deep in the recovery, which is what the corrupted
    # accelerometer profile claimed
    peak = u[np.argmax(prof)]
    assert 0.25 <= peak <= drive + 0.15
    # velocity must RISE through the mid-drive
    mid = (u > 0.15) & (u < drive - 0.05)
    assert np.all(np.diff(prof[mid]) > -0.02)


def test_charles_landmarks_sit_on_the_channel_and_in_the_right_order():
    """Bridge coordinates, checked two ways.

    A landmark can be *in* the channel and still be in the wrong place
    along it -- Weeks used to sit 6 m from the centreline but 370 m too
    far upstream, almost on top of Anderson, leaving 259 m between two
    bridges that are 426 m apart.  Offset alone did not catch it.  The
    spacing and sinuosity do.  See coxswain.river.charles.
    """
    from coxswain.river import charles

    raster = charles.charles_channel()
    seen = []
    for name, latlon in charles.BRIDGES:
        station, offset = charles.landmark_station(latlon, raster)
        assert offset < 40.0, (name, offset)
        east, north = charles.local_tangent_plane(
            latlon[0], latlon[1], charles.CHARLES_ORIGIN)
        seen.append((name, station, np.array([float(east), float(north)])))

    # upstream order is decreasing station, and no two bridges collide
    for (n0, s0, p0), (n1, s1, p1) in zip(seen[:-1], seen[1:]):
        gap = s0 - s1
        straight = float(np.linalg.norm(p1 - p0))
        assert gap > 300.0, (n0, n1, gap)
        assert 0.95 <= gap / straight <= 1.8, (n0, n1, gap / straight)


def test_hocr_course_is_three_miles_from_the_boathouse():
    """The race starts off DeWolfe Boathouse, not at the BU Bridge."""
    from coxswain.river import charles

    raster = charles.charles_channel()
    start_xy, finish_xy, line, (start, finish) = charles.hocr_course(raster)
    assert start - finish == pytest.approx(4828.0, abs=1.0)

    bu, _ = charles.landmark_station(charles.BU_BRIDGE, raster)
    assert 100.0 < start - bu < 260.0          # start is below the bridge

    eliot, _ = charles.landmark_station(charles.ELIOT_BRIDGE, raster)
    assert 600.0 < eliot - finish < 1100.0     # finish is above Eliot

    # every bridge falls inside the race
    for name, latlon in charles.BRIDGES:
        station, _ = charles.landmark_station(latlon, raster)
        assert finish <= station <= start, name
