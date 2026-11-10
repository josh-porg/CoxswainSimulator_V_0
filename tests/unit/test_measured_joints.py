"""A measured channel outranks a population clock for its joint: BodyOptions.measured (SOURCES §228)."""
import pytest

from coxswain import physics
from coxswain.crew.body_options import BodyOptions, body_options


def test_the_profiles_clock_skips_a_measured_joint():
    own = body_options(measured=("shank", "thigh", "trunk"))
    merged = physics.merged_body(own, ("pitto2024_trunk",))
    assert merged is own                                         # nothing added: the trunk is the athlete's
    assert physics.merged_body(None, ("pitto2024_trunk",)).clock_for("trunk") is not None


def test_a_joint_is_clocked_or_measured_not_both():
    with pytest.raises(ValueError, match="not both"):
        body_options(clocks=("pitto2024_trunk",), measured=("trunk",))
    with pytest.raises(ValueError, match="measured joints"):
        BodyOptions(measured=("elbow",))


def test_measured_joints_are_in_the_body_signature():
    assert body_options(measured=("trunk",)).signature() != BodyOptions().signature()
    assert not body_options(measured=("trunk",)).is_default
