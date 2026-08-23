"""Per-rower stroke phase, and what timing error does to the boat.

Until now every rower shared one stroke phase.  That is not a small
idealisation: sections 15-16 show roll is an unstable mode held through
the recovery by about 5% of the drive's authority, and port/starboard
timing asymmetry is one of the main things that disturbs it.  Setting all
phases equal set that disturbance to exactly zero.

See ``docs/PLAN_SYNCHRONISATION_AND_BLADES.md`` for the coupled-oscillator
programme this is the first step of.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.core.state import State
from coxswain.crew.balance import PhaseAuthority
from coxswain.sim.simulator import RowingSimulator


@pytest.fixture
def eight():
    return catalog.eight(rate=32.0)


def _sides(boat):
    return np.array([seat.oarlocks[0].side for seat in boat.rig.seats],
                    dtype=float)


def _split(boat, lag_seconds):
    """Port side lagging starboard by ``lag_seconds``."""
    return np.where(_sides(boat) > 0,
                    lag_seconds / boat.timing.period, 0.0)


def _oar_loads_over_a_stroke(boat, samples=48):
    simulator = RowingSimulator(boat)
    state = State.from_vector(simulator.initial_state(surge_speed=4.6))
    times = np.linspace(0.0, boat.timing.period, samples, endpoint=False)
    roll, yaw, surge = [], [], []
    for t in times:
        breakdown = simulator.breakdown(float(t), state)
        moment = np.asarray(breakdown.oar_moment)
        roll.append(moment[0])
        yaw.append(moment[2])
        surge.append(np.asarray(breakdown.oar_force)[0])
    return (np.array(roll), np.array(yaw), np.array(surge))


# --------------------------------------------------------------------------
# the plumbing
# --------------------------------------------------------------------------
def test_a_crew_is_synchronised_by_default(eight):
    """Every prior result depends on this, so it is asserted rather than
    assumed."""
    np.testing.assert_allclose(eight.phase_offsets, np.zeros(eight.n_seats))


def test_zero_offsets_change_nothing(eight):
    before = eight.crew_field(0.3)
    eight.phase_offsets = np.zeros(eight.n_seats)
    after = eight.crew_field(0.3)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_offsets_must_have_one_entry_per_seat(eight):
    with pytest.raises(ValueError, match="one entry per seat"):
        eight.phase_offsets = np.zeros(3)


def test_offsetting_a_rower_moves_them_in_time(eight):
    """A rower offset by a whole stroke is back where they started."""
    reference = eight.crew_field(0.3)[1]
    eight.phase_offsets = np.full(eight.n_seats, 1.0)
    whole_cycle = eight.crew_field(0.3)[1]
    np.testing.assert_allclose(whole_cycle, reference, atol=1e-9)


def test_a_desynchronised_crew_costs_more_to_evaluate(eight):
    """Honest bookkeeping: individuals cannot share a kinematic chain."""
    assert len(eight._crew_groups()) == 1
    eight.phase_offsets = np.linspace(0.0, 0.05, eight.n_seats)
    assert len(eight._crew_groups()) == eight.n_seats


def test_mass_is_conserved_under_desynchronisation(eight):
    total = eight.crew_field(0.3)[0].sum()
    eight.phase_offsets = _split(eight, 0.03)
    assert eight.crew_field(0.3)[0].sum() == pytest.approx(total)


def test_the_oars_follow_the_rower_not_the_boat(eight):
    """A late rower is late in their oar as well as their body.

    If the oar were evaluated at the boat's time while the body ran on its
    own, the hands would come off the handle -- the constraint the whole
    crew model rests on.
    """
    simulator = RowingSimulator(eight)
    state = State.from_vector(simulator.initial_state(surge_speed=4.6))
    plain = np.asarray(simulator.breakdown(0.2, state).oar_force)

    eight.phase_offsets = np.full(eight.n_seats, 0.25)
    shifted = np.asarray(
        RowingSimulator(eight).breakdown(0.2, state).oar_force)
    assert not np.allclose(plain, shifted)


# --------------------------------------------------------------------------
# what timing error actually does
# --------------------------------------------------------------------------
def test_a_timing_split_injects_a_roll_moment(eight):
    """The link between synchronisation and the balance work.

    Sections 15-16 leave an eight about 84 N m of roll authority through
    the recovery.  Timing error spends it.
    """
    eight.phase_offsets = np.zeros(eight.n_seats)
    synchronised = np.sqrt(np.mean(_oar_loads_over_a_stroke(eight)[0] ** 2))

    eight.phase_offsets = _split(eight, 0.040)
    split = np.sqrt(np.mean(_oar_loads_over_a_stroke(eight)[0] ** 2))

    assert split > 20.0 * max(synchronised, 1e-3), (synchronised, split)


def test_the_roll_disturbance_grows_with_the_split(eight):
    magnitudes = []
    for lag in (0.010, 0.020, 0.040):
        eight.phase_offsets = _split(eight, lag)
        magnitudes.append(
            np.sqrt(np.mean(_oar_loads_over_a_stroke(eight)[0] ** 2)))
    assert magnitudes == sorted(magnitudes)


def test_a_modest_split_exhausts_the_recovery_balance_authority(eight):
    """The headline number, and it is uncomfortably small.

    A tenth of a second of port/starboard split -- about 5% of a
    stroke at rate 32 -- produces a roll disturbance equal to
    everything the crew have left once the blades leave the water.

    The split needed was 0.080 s until the drive fraction was
    corrected (SOURCES sec. 50).  A longer drive leaves a shorter
    recovery, and the recovery balance authority rose from 84.1 to
    93.1 N m with it, so slightly more mistiming is now needed to
    exhaust it.  The conclusion is unchanged and the margin is still
    uncomfortable.
    """
    authority = PhaseAuthority.from_boat(eight)
    eight.phase_offsets = _split(eight, 0.100)
    disturbance = np.sqrt(np.mean(_oar_loads_over_a_stroke(eight)[0] ** 2))
    assert disturbance > authority.recovery, (disturbance, authority.recovery)


def test_a_small_split_can_cancel_the_sweep_rigs_own_yaw_bias(eight):
    """An unexpected result worth keeping.

    A sweep eight yaws even with a perfectly synchronised crew -- the rig
    is not port-starboard symmetric.  A small timing offset between the
    sides pushes the other way, and around 20 ms it very nearly cancels.

    That makes crew timing a steering trim as well as a disturbance, which
    is a different statement from "timing error is bad".
    """
    eight.phase_offsets = np.zeros(eight.n_seats)
    synchronised = np.sqrt(np.mean(_oar_loads_over_a_stroke(eight)[1] ** 2))

    eight.phase_offsets = _split(eight, 0.020)
    trimmed = np.sqrt(np.mean(_oar_loads_over_a_stroke(eight)[1] ** 2))

    assert trimmed < 0.5 * synchronised, (synchronised, trimmed)


def test_the_yaw_disturbance_is_not_monotonic_in_the_split(eight):
    """Because it first cancels the rig's own bias and then overtakes it."""
    magnitudes = []
    for lag in (0.0, 0.020, 0.080):
        eight.phase_offsets = _split(eight, lag)
        magnitudes.append(
            np.sqrt(np.mean(_oar_loads_over_a_stroke(eight)[1] ** 2)))
    assert magnitudes[1] < magnitudes[0]
    assert magnitudes[2] > magnitudes[0]


def test_small_splits_barely_touch_propulsion(eight):
    """Where the cost is, and where it is not.

    Stroke-averaged surge is almost unchanged at these splits.  The price
    of poor timing at this scale is paid in roll and yaw, not in thrust --
    which is worth stating because the folk explanation is that being out
    of time "wastes power".
    """
    eight.phase_offsets = np.zeros(eight.n_seats)
    synchronised = _oar_loads_over_a_stroke(eight)[2].mean()
    eight.phase_offsets = _split(eight, 0.040)
    split = _oar_loads_over_a_stroke(eight)[2].mean()
    assert split == pytest.approx(synchronised, rel=0.02)
