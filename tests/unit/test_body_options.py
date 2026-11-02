"""Body options and the hand chain: pluggable pieces that change nothing until asked for (SOURCES §210–211)."""
from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.crew.body_options import BodyOptions, SegmentClock, body_options
from coxswain.crew.hand_chain import HandChain, HandContext, StretchTerm, hand_chain
from coxswain.crew.kinematics import MAX_SHOULDER_PROTRACTION, JointDrivenRower, RowerStation
from coxswain.crew.stroke import OnWaterTiming

ANTHRO = RowerAnthropometry(mass=85.0, stature=1.90)
TIMING = OnWaterTiming(32.0)


def _rower(body=None):
    return JointDrivenRower(ANTHRO, RowerStation(x_ankle=0.0), TIMING, body=body)


def _course(theta, f, n):
    c = (theta - theta.min()) / np.ptp(theta)
    return c if c[int(f * n)] > c[0] else 1.0 - c


def test_the_default_body_changes_nothing():
    plain, explicit = _rower(), _rower(BodyOptions())
    assert plain.kinematics_signature() == explicit.kinematics_signature()
    for t in (0.0, 0.4, 1.1):
        a, b = plain.joint_positions(t), explicit.joint_positions(t)
        assert all(np.allclose(a[k], b[k]) for k in a)


def test_a_trunk_clock_retimes_the_trunk_and_keeps_its_range():
    plain, clocked = _rower(), _rower(body_options(["pitto2024_trunk"]))
    assert plain.kinematics_signature() != clocked.kinematics_signature()   # every cache sees it
    n, P, f = 2000, float(TIMING.period), float(TIMING.drive_fraction)
    u = np.arange(n) / n
    th0 = np.asarray(plain.joint_angles.trunk(u * P).value)
    th1 = np.asarray(clocked.joint_angles.trunk(u * P).value)
    assert abs(np.ptp(th1) - np.ptp(th0)) < np.radians(0.5)                # only *when* it moves changes
    g = np.where(u < f, 0.5 * u / f, 0.5 + 0.5 * (u - f) / (1.0 - f))
    target = SegmentClock.pitto2024_trunk().target(g)
    assert np.sqrt(np.mean((_course(th1, f, n) - target) ** 2)) < 0.01
    assert np.sqrt(np.mean((_course(th0, f, n) - target) ** 2)) > 0.05       # the keyframes do not
    np.testing.assert_allclose(np.asarray(clocked.joint_angles.shank(u * P).value),
                               np.asarray(plain.joint_angles.shank(u * P).value))   # other joints untouched


def test_body_options_are_validated():
    clock = SegmentClock.pitto2024_trunk()
    with pytest.raises(ValueError, match="one clock per joint"):
        BodyOptions(clocks=(clock, clock))
    with pytest.raises(ValueError, match="drives one of"):
        SegmentClock("elbow", clock.clock, clock.course)
    with pytest.raises(ValueError, match="distance"):
        BodyOptions(reach_allowance=-0.01)


def test_the_reach_allowance_is_what_the_elbow_solve_accepts():
    upper, fore = _rower().upper_arm_length, _rower().forearm_length
    shoulders = np.zeros((4, 3))
    hands = np.zeros((4, 3))
    hands[:, 0] = upper + fore + 0.10                   # 10 cm beyond a straight arm
    with pytest.raises(ValueError, match="cannot reach"):
        _rower()._solve_elbows(shoulders, hands, side=1)
    _rower(BodyOptions(reach_allowance=0.12))._solve_elbows(shoulders, hands, side=1)


def test_the_hand_chain_adds_its_terms_and_sets_its_allowance():
    empty = hand_chain()
    ctx = HandContext("drive", np.linspace(0.0, 1.0, 5), np.array([0.0, 0.5, 1.0, 0.5, 0.0]))
    np.testing.assert_allclose(empty.extra(ctx), 0.0)
    assert empty.reach_allowance() is None
    chain = hand_chain(["stretch"])
    assert chain == HandChain((StretchTerm(),))
    np.testing.assert_allclose(chain.extra(ctx), StretchTerm().peak * ctx.force_fraction)
    assert abs(chain.reach_allowance() - (MAX_SHOULDER_PROTRACTION + StretchTerm().peak)) < 1e-12
    two = HandChain((StretchTerm(0.05), StretchTerm(0.02)))
    np.testing.assert_allclose(two.extra(ctx), 0.07 * ctx.force_fraction)
    with pytest.raises(KeyError):
        hand_chain(["no such term"])


def test_a_population_travel_scales_the_keyframes_to_the_stature_scaled_targets():
    from coxswain.crew.body_options import SegmentTravel
    from coxswain.crew.kinematics import segment_travels
    travel = SegmentTravel.kleshnev2005_racing()
    rower = _rower(body_options(travel="k05_racing"))
    legs, trunk = segment_travels(rower)
    want_legs, want_trunk = travel.targets(ANTHRO.stature)
    assert abs(legs - want_legs) < 1e-3 and abs(trunk - want_trunk) < 1e-3
    assert segment_travels(_rower())[0] > want_legs + 0.03          # the keyframes travel further
    assert rower.kinematics_signature() != _rower().kinematics_signature()
    with pytest.raises(ValueError):
        SegmentTravel(0.5, -0.1, 1.8)


def test_a_seat_clock_is_carried_to_the_shank_through_the_linkage():
    n, P, f = 2000, float(TIMING.period), float(TIMING.drive_fraction)
    u = np.arange(n) / n
    g = np.where(u < f, 0.5 * u / f, 0.5 + 0.5 * (u - f) / (1.0 - f))
    target = SegmentClock.kleshnev2005_legs().target(g)

    def seat(rower):
        x = np.asarray(rower._chain(u * P)["hip"][0].value)
        return _course(x, f, n), float(np.ptp(x))
    plain, clocked = _rower(), _rower(body_options(["k05_legs"]))
    (c0, travel0), (c1, travel1) = seat(plain), seat(clocked)
    assert np.sqrt(np.mean((c1 - target) ** 2)) < 0.005 < np.sqrt(np.mean((c0 - target) ** 2))
    assert abs(travel1 - travel0) < 1e-3                         # the seat travels as far, only later or earlier
    with pytest.raises(ValueError, match="channel"):
        SegmentClock("trunk", target[:50:5], target[:50:5] / max(target[:50:5].max(), 1e-9), channel="seat")


def test_the_research_profile_builds_and_applies_the_trunk_clock_and_shipped_does_not():
    from coxswain import physics
    from coxswain.boats import catalog
    assert physics.resolve("shipped").body() is None
    built = physics.resolve("research").body()
    assert [c.joint for c in built.clocks] == ["trunk"]
    boat = catalog.single_scull(rate=32.0, rower_mass=85.0, rower_stature=1.90)
    physics.resolve("research").apply(boat)
    assert [c.joint for c in boat.crew[0].rower.body.clocks] == ["trunk"]
    physics.resolve("research").apply(boat)                    # idempotent
    assert len(boat.crew[0].rower.body.clocks) == 1


def test_a_recovery_only_clock_leaves_the_drive_alone():
    n, P, f = 2000, float(TIMING.period), float(TIMING.drive_fraction)
    u = np.arange(n) / n
    plain = _rower()
    rec = _rower(body_options(["k05_legs_recovery"]))
    full = _rower(body_options(["k05_legs"]))
    x0, xr, xf = (np.asarray(r._chain(u * P)["hip"][0].value) for r in (plain, rec, full))
    drive = (u > 0.05 * f) & (u < 0.95 * f)                     # away from the ends, where the time map is smoothed
    rest = (u > f + 0.1 * (1 - f)) & (u < 1 - 0.1 * (1 - f))
    assert np.max(np.abs(xr[drive] - x0[drive])) < 0.004 < np.max(np.abs(xf[drive] - x0[drive]))
    assert np.max(np.abs(xr[rest] - x0[rest])) > 0.01           # the recovery is retimed
