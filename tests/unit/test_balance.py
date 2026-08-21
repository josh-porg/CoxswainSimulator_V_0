"""How the crew's balance effort reaches the hull, and what it drags along.

The balance reflex used to be added as a pure couple about the hull ``x``
axis.  A crew cannot apply one.  They change handle height, which loads
each oar as a lever about its oarlock and puts a vertical force at that
rigger -- and the riggers are wherever the rig puts them.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.balance import BalanceRig


@pytest.fixture(scope="module")
def eight():
    return catalog.eight(rate=32.0)


# --------------------------------------------------------------------------
# the geometry
# --------------------------------------------------------------------------
def test_sweep_rig_is_longitudinally_asymmetric(eight):
    """The fact the whole coupling rests on.

    A conventionally rigged eight alternates sides down the boat, so the
    four port oarlocks and the four starboard ones have different mean
    longitudinal stations -- by exactly one seat spacing.
    """
    port = [np.asarray(lock.position)[0]
            for seat in eight.rig.seats for lock in seat.oarlocks
            if lock.side > 0]
    starboard = [np.asarray(lock.position)[0]
                 for seat in eight.rig.seats for lock in seat.oarlocks
                 if lock.side < 0]
    assert len(port) == len(starboard) == 4
    offset = abs(np.mean(port) - np.mean(starboard))
    spacing = abs(np.asarray(eight.rig.seats[1].oarlocks[0].position)[0]
                  - np.asarray(eight.rig.seats[0].oarlocks[0].position)[0])
    assert offset == pytest.approx(spacing, rel=1e-9), \
        "the offset is one seat spacing, not a coincidence"


def test_balancing_a_sweep_eight_also_pitches_it(eight):
    """The coupling a pure ``x`` couple cannot represent."""
    rig = BalanceRig.from_boat(eight)
    assert rig.pitch_coupling == pytest.approx(0.718, abs=0.01)


def test_a_sculler_has_no_pitch_coupling():
    """The control that proves it is rigging and not a bug.

    A sculler's two oarlocks share a longitudinal station and sit at
    opposite ``y``, so the balance forces make a pure roll couple -- which
    is exactly what the old model assumed for every boat.
    """
    rig = BalanceRig.from_boat(catalog.single_scull(rate=32.0))
    assert rig.pitch_per_unit == pytest.approx(0.0, abs=1e-12)
    assert rig.pitch_coupling == pytest.approx(0.0, abs=1e-12)


def test_a_coxed_four_couples_too():
    rig = BalanceRig.from_boat(catalog.coxed_four(rate=32.0))
    assert rig.pitch_coupling > 0.5


def test_balance_adds_no_net_vertical_force(eight):
    """The one thing the old docstring claimed, and it was right.

    Handle-height trim is equal and opposite across the boat, so the
    vertical forces cancel.  It was the pitch term it was silent about.
    """
    rig = BalanceRig.from_boat(eight)
    assert rig.heave_per_unit == pytest.approx(0.0, abs=1e-12)
    force, _ = rig.loads(3000.0)
    assert float(force[2]) == pytest.approx(0.0, abs=1e-9)


def test_the_demanded_roll_moment_is_delivered_exactly(eight):
    """The coupling is a byproduct, not a tax: roll is still what was asked
    for.  Only the extra pitch comes along."""
    rig = BalanceRig.from_boat(eight)
    for demand in (-4000.0, -250.0, 0.0, 1500.0, 4000.0):
        _, moment = rig.loads(demand)
        assert float(moment[0]) == pytest.approx(demand, rel=1e-12, abs=1e-12)


def test_loads_are_linear_in_the_demand(eight):
    """Exact geometry, no fit and no smoothing -- so it must be linear."""
    rig = BalanceRig.from_boat(eight)
    _, one = rig.loads(1000.0)
    _, two = rig.loads(2000.0)
    assert float(two[1]) == pytest.approx(2.0 * float(one[1]), rel=1e-12)


def test_pitch_coupling_matches_a_hand_computation(eight):
    """Derived independently of the implementation.

    Each rigger carries ``+f`` on one side and ``-f`` on the other, so the
    roll moment is ``sum y_i s_i`` and the pitch moment ``-sum x_i s_i``.
    """
    roll = pitch = 0.0
    for seat in eight.rig.seats:
        for lock in seat.oarlocks:
            x, y, _ = np.asarray(lock.position, dtype=float)
            roll += y * lock.side
            pitch += -x * lock.side
    rig = BalanceRig.from_boat(eight)
    assert rig.roll_per_unit == pytest.approx(roll, rel=1e-12)
    assert rig.pitch_per_unit == pytest.approx(pitch, rel=1e-12)


# --------------------------------------------------------------------------
# and it has to reach the dynamics, in both paths
# --------------------------------------------------------------------------
def test_the_numpy_simulator_pitches_when_it_corrects_roll(eight):
    from coxswain.sim.simulator import RowingSimulator

    simulator = RowingSimulator(eight)
    state = simulator.initial_state(surge_speed=5.0)
    state[3] = np.radians(4.0)

    with_coupling = simulator.derivative(0.2, state)
    simulator.balance_rig = type(simulator.balance_rig)(
        roll_per_unit=simulator.balance_rig.roll_per_unit,
        pitch_per_unit=0.0, heave_per_unit=0.0)
    without = simulator.derivative(0.2, state)

    assert abs(with_coupling[10] - without[10]) > 1e-4, \
        "the pitch acceleration must actually change"
    # Roll is *nearly* untouched, but not exactly: the mass matrix
    # couples the two, so the extra pitch moment feeds a little back
    # into roll.  That feedback is itself part of what a pure x couple
    # threw away.
    assert with_coupling[9] == pytest.approx(without[9], rel=1e-3)
    assert with_coupling[9] != without[9], "the coupling is not one-way"


def test_the_casadi_model_pitches_when_it_corrects_roll(eight):
    pytest.importorskip("casadi")
    from coxswain.river.hullsurrogate import HullSurrogate
    from coxswain.river.sixdof import SixDofModel

    surrogate = HullSurrogate.from_boat(eight, n_heave=9, n_pitch=5, n_roll=5)
    model = SixDofModel(eight, surrogate=surrogate)
    assert model.balance_rig.pitch_coupling == pytest.approx(0.718, abs=0.01)

    state = np.zeros(13)
    state[3] = np.radians(4.0)
    state[6] = 5.2
    state[12] = model.anaerobic_capacity
    value = np.array(model.function()(state, [0., 0., 1.], 0.2)).ravel()

    assert np.all(np.isfinite(value))
    assert value[9] < 0.0, "a roll to port must be corrected back"
    assert abs(value[10]) > 1e-3, "and it must pitch while doing it"


def test_both_paths_agree_on_the_balance_geometry(eight):
    pytest.importorskip("casadi")
    from coxswain.river.hullsurrogate import HullSurrogate
    from coxswain.river.sixdof import SixDofModel
    from coxswain.sim.simulator import RowingSimulator

    surrogate = HullSurrogate.from_boat(eight, n_heave=9, n_pitch=5, n_roll=5)
    symbolic = SixDofModel(eight, surrogate=surrogate).balance_rig
    numeric = RowingSimulator(eight).balance_rig
    assert symbolic == numeric


# --------------------------------------------------------------------------
# the rowers have to be holding the oars laterally too
# --------------------------------------------------------------------------
def test_sweep_hands_follow_the_handle_sideways(eight):
    """The rower's joint chain is sagittal, so on its own it puts the hand
    on the centreline.  A sweep handle is nowhere near the centreline: it
    swings a wide lateral arc, and the hands are on it."""
    lateral = [np.asarray(eight.hand_positions(f * eight.timing.period))[0, 1]
               for f in (0.02, 0.25, 0.50)]
    assert max(lateral) - min(lateral) > 0.3, lateral


def test_port_and_starboard_hands_mirror(eight):
    """Two rowers on opposite sides reach opposite ways.  This was the
    other half of the bug: the batching copied the group leader's lateral
    position to every seat, so all eight reached the same way."""
    for f in (0.02, 0.25, 0.50, 0.75):
        hands = np.asarray(eight.hand_positions(f * eight.timing.period))
        sides = [seat.oarlocks[0].side for seat in eight.rig.seats]
        port = [h for h, s in zip(hands[:, 1], sides) if s > 0]
        starboard = [h for h, s in zip(hands[:, 1], sides) if s < 0]
        assert np.mean(port) == pytest.approx(-np.mean(starboard), abs=1e-9)
        assert abs(np.mean(port)) > 1e-3, "and not both zero"


def test_a_scullers_hands_stay_on_the_centreline():
    """The control.  A sculler holds two handles that mirror each other, so
    their mean is on the centreline -- which is what the old code gave
    every boat, and is only right for this one."""
    scull = catalog.single_scull(rate=32.0)
    for f in (0.02, 0.25, 0.50):
        hands = np.asarray(scull.hand_positions(f * scull.timing.period))
        assert hands[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_hands_agree_with_the_rig_geometry(eight):
    """The hand is the handle, so it must be exactly where the oar puts it."""
    from coxswain.crew.oarlock import handle_position

    for f in (0.05, 0.30, 0.60):
        t = f * eight.timing.period
        hands = np.asarray(eight.hand_positions(t))
        for index, seat in enumerate(eight.rig.seats):
            expected = np.mean([
                float(handle_position(t, eight.timing, lock,
                                      eight.oar_sweep)[1])
                for lock in seat.oarlocks])
            assert hands[index, 1] == pytest.approx(expected, abs=1e-9)


def test_the_lateral_moment_arm_is_not_constant(eight):
    """Why it matters: the oar's lateral moment arm was a fixed 0.85 m.

    The true arm swings from about 0.71 m at the catch to 1.13 m through
    mid-drive -- and the error changes sign, so the oar yaw moment history
    was distorted in shape, not merely scaled.  That is the quantity the
    whole steering model rests on.
    """
    from coxswain.crew.oarlock import handle_position

    lock = eight.rig.seats[0].oarlocks[0]
    arms = []
    for f in (0.05, 0.25):
        handle = handle_position(f * eight.timing.period, eight.timing, lock,
                                 eight.oar_sweep)
        arms.append(abs(float(handle[1]) - float(lock.position[1])))
    assert min(arms) < 0.85 < max(arms), arms


# --------------------------------------------------------------------------
# Phase-dependent balance authority
#
# [D96] "Balance of Racing Rowing Boats", Furnivall Sculling Club 1996,
#       2013 PDF revision, https://eodg.atm.ox.ac.uk/user/dudhia/rowing/
#       physics/Balance_of_Racing_Rowing_Boats_v3.pdf
# --------------------------------------------------------------------------
def test_a_loaded_racing_shell_is_statically_unstable_in_roll(eight):
    """[D96]'s central claim, reproduced independently.

    "No racing rowing boat is statically stable with the crew rigid in the
    boat and the oars off the water."  A shell carries its centre of
    gravity above its metacentre, so buoyancy does not right it -- every
    newton-metre of flatness comes from the crew.

    Positive stiffness here means destabilising.
    """
    from coxswain.crew.balance import static_roll_stiffness

    assert static_roll_stiffness(eight) > 0.0


def test_every_boat_class_is_unstable_not_just_the_eight():
    from coxswain.crew.balance import static_roll_stiffness

    for factory in (catalog.eight, catalog.coxed_four, catalog.single_scull):
        boat = factory(rate=32.0)
        assert static_roll_stiffness(boat) > 0.0, boat


def test_centre_of_gravity_height_matches_the_published_table(eight):
    """[D96] Table 1 gives 28 cm above the waterline for an eight.

    This model is built from de Leva segment inertias and a hull mesh with
    no reference to that table, so agreement is a real check on both.
    """
    mass, position, _, _ = eight.crew_field(0.0)
    centre = float((mass * position[:, 2]).sum() / eight.total_mass)
    height = 100.0 * (centre - eight.equilibrium_heave())
    assert 22.0 < height < 32.0, height


def test_recovery_authority_is_a_small_fraction_of_the_drive(eight):
    """The whole point.

    On the drive the blade is buried and the crew push against the water.
    On the recovery the blade is in the air and the only reaction available
    is the oar's own inertia -- [D96] notes hand-height changes "can only
    produce transient forces", and that skimming the blades, the other
    mechanism, is unavailable to a crew boat because the spoons must clear
    the puddles of the crew ahead.
    """
    from coxswain.crew.balance import PhaseAuthority

    authority = PhaseAuthority.from_boat(eight)
    assert authority.recovery < 0.10 * authority.drive
    assert authority.recovery > 0.0, "not zero -- the oar still has mass"


def test_every_class_loses_most_of_its_authority_on_the_recovery():
    """The ratio is small everywhere, but not identical.

    The oar contribution scales with the rig and is nearly class
    independent.  The trunk-lean contribution does not: it scales with how
    much crew mass there is per oar, so a sculler -- one body, two oars --
    gets proportionally less from it than an eight does.  [D96] observes
    the consequence from the other end, that small boats depend on
    skimming the blades, a mechanism crew boats cannot use.
    """
    from coxswain.crew.balance import PhaseAuthority

    ratios = {name: PhaseAuthority.from_boat(factory(rate=32.0)).ratio
              for name, factory in (("8+", catalog.eight),
                                    ("4+", catalog.coxed_four),
                                    ("1x", catalog.single_scull))}
    assert all(r < 0.10 for r in ratios.values()), ratios
    assert ratios["1x"] < ratios["8+"], ratios


def test_the_crew_runs_out_of_recovery_authority_at_a_couple_of_degrees(
        eight):
    """The number that answers the rowers' complaint.

    Recovery authority divided by the destabilising stiffness is the
    largest heel the crew can still arrest with the blades out of the
    water.  About 2.3 degrees for an eight -- and the roll oscillation the
    model actually produces sits just under it, which is not a
    coincidence: the boat runs up against this limit every recovery.
    """
    from coxswain.crew.balance import PhaseAuthority, static_roll_stiffness

    authority = PhaseAuthority.from_boat(eight)
    holdable = np.degrees(authority.recovery / static_roll_stiffness(eight))
    assert 1.0 < holdable < 4.0, holdable


def test_trunk_lean_beats_hand_heights_on_the_recovery(eight):
    """Which actuator actually holds the boat.

    With the blades in the air the oars can only offer their own inertia.
    Leaning the trunk moves 700 kg of crew on a 0.2 m lever, and two
    degrees of it is worth more than the oars are.  This is [D96]'s "body
    inertia", and it matches what coaches actually say -- sit up and use
    your body, not lift and drop your hands, which [D96] shows is positive
    feedback.
    """
    from coxswain.crew.balance import PhaseAuthority, trunk_lean_authority

    lean = trunk_lean_authority(eight)
    total = PhaseAuthority.from_boat(eight).recovery
    oars = total - lean
    assert lean > oars, (lean, oars)


def test_trunk_lean_is_what_makes_an_eight_sittable(eight):
    """Remove it and the boat is materially harder to hold."""
    from coxswain.crew.balance import PhaseAuthority
    from coxswain.sim.control import BalanceController, Coxswain
    from coxswain.sim.simulator import RowingSimulator

    def swing(authority):
        controller = BalanceController(authority=authority,
                                       timing=eight.timing)
        result = RowingSimulator(
            eight, coxswain=Coxswain(balance=controller)).run(
            duration=12.0, dt=0.006, surge_speed=4.6)
        roll = np.degrees(np.asarray(result.roll))
        half = len(roll) // 2
        return roll[half:].max() - roll[half:].min()

    full = PhaseAuthority.from_boat(eight)
    oars_only = PhaseAuthority.from_boat(eight, lean_angle=0.0)
    assert swing(oars_only) > 1.4 * swing(full)


def test_the_recovery_lasts_several_instability_time_constants(eight):
    """Why it is hard rather than merely annoying.

    The uncontrolled roll mode grows exponentially with an e-folding time
    of about 0.2 s, and the recovery is over a second long.  The crew are
    not damping a stable mode; they are catching a diverging one.
    """
    from coxswain.crew.balance import roll_divergence_time

    tau = roll_divergence_time(eight)
    assert 0.1 < tau < 0.4, tau
    assert eight.timing.recovery_duration / tau > 3.0


def test_the_authority_window_is_high_on_the_drive_and_low_after(eight):
    from coxswain.crew.balance import PhaseAuthority

    authority = PhaseAuthority.from_boat(eight)
    period = eight.timing.period
    mid_drive = authority.window(0.2 * period, eight.timing)
    mid_recovery = authority.window(0.8 * period, eight.timing)
    assert mid_drive > 10.0 * mid_recovery


def test_the_smoothed_window_tracks_the_square_wave(eight):
    """A bounded smoothing has to have its bound measured.

    The transition is smoothed because the mesh puts nodes exactly at the
    catch and the finish and a square wave has no derivative there.  Away
    from the edges the smoothed limit must still be the real one.
    """
    from coxswain.crew.balance import PhaseAuthority

    authority = PhaseAuthority.from_boat(eight)
    assert authority.window_error(eight.timing) < 0.05


def test_phase_dependent_balance_makes_the_boat_harder_to_hold(eight):
    """It must cost something, or it is not modelling anything."""
    from coxswain.crew.balance import PhaseAuthority
    from coxswain.sim.control import BalanceController, Coxswain
    from coxswain.sim.simulator import RowingSimulator

    def swing(authority):
        controller = BalanceController(
            authority=authority,
            timing=eight.timing if authority is not None else None)
        result = RowingSimulator(
            eight, coxswain=Coxswain(balance=controller)).run(
            duration=14.0, dt=0.005, surge_speed=4.6)
        roll = np.degrees(np.asarray(result.roll))
        half = len(roll) // 2
        return roll[half:].max() - roll[half:].min()

    constant = swing(None)
    phased = swing(PhaseAuthority.from_boat(eight))
    assert phased > 1.05 * constant, (constant, phased)


# --------------------------------------------------------------------------
# the buoyancy frame bug that the honest authority exposed
# --------------------------------------------------------------------------
def test_hull_surrogate_stores_the_centre_of_buoyancy_in_the_hull_frame(eight):
    """The bug a too-generous balance limit had been hiding.

    ``HullMesh.submerged`` returns the centre of buoyancy in the
    **absolute** frame -- its own ``buoyancy_moment`` is ``cross(centre,
    force)`` against an unrotated vertical force.  The surrogate used to
    tabulate that value directly and the dynamics then rotated it again,
    applying roll and pitch twice.

    At one degree of heel it turned an 18.2 N m righting moment into
    2.4 N m: 87% of the hull's roll stiffness, gone.  It was invisible at
    zero roll, which is where every earlier check had looked, and it stayed
    invisible while the crew were credited with 4000 N m of balance
    authority they do not have.
    """
    pytest.importorskip("casadi")
    from coxswain.core.frames import hull_to_abs
    from coxswain.river.hullsurrogate import HullSurrogate

    surrogate = HullSurrogate.from_boat(eight, n_heave=17, n_pitch=9,
                                        n_roll=13,
                                        roll_range=np.radians(4.0))
    heave = eight.equilibrium_heave()
    roll = np.radians(1.0)

    values = surrogate(heave, 0.0, roll)
    buoyancy = eight.water.density * 9.80665 * values["volume"]
    centre_hull = np.array([values["buoyancy_x"], values["buoyancy_y"],
                            values["buoyancy_z"]])
    rotation = hull_to_abs(np.array([roll, 0.0, 0.0]))
    moment = np.cross(rotation @ centre_hull, np.array([0.0, 0.0, buoyancy]))

    exact = eight.mesh.submerged(
        np.array([0.0, 0.0, heave]), np.array([roll, 0.0, 0.0]),
        rho=eight.water.density, gravity=9.80665, water_level=0.0)
    expected = float(np.asarray(exact.buoyancy_moment)[0])
    assert moment[0] == pytest.approx(expected, rel=0.02), (moment[0],
                                                            expected)


def test_the_two_paths_agree_on_roll_acceleration_at_heel(eight):
    """The check that would have caught the double rotation.

    Comparing at zero roll proves nothing about a frame error in roll.
    """
    pytest.importorskip("casadi")
    from coxswain.crew.balance import PhaseAuthority
    from coxswain.river.hullsurrogate import HullSurrogate
    from coxswain.river.sixdof import SixDofModel
    from coxswain.sim.control import BalanceController, Coxswain
    from coxswain.sim.simulator import RowingSimulator

    authority = PhaseAuthority.from_boat(eight)
    surrogate = HullSurrogate.from_boat(eight, n_heave=21, n_pitch=9,
                                        n_roll=15,
                                        roll_range=np.radians(4.0))
    model = SixDofModel(eight, surrogate=surrogate)
    function = model.function()
    simulator = RowingSimulator(eight, coxswain=Coxswain(
        balance=BalanceController(authority=authority, timing=eight.timing)))

    time = 0.1 * eight.timing.period
    state = simulator.initial_state(surge_speed=4.6)
    state[3] = np.radians(1.0)
    numeric = simulator.derivative(time, state)

    symbolic_state = np.zeros(13)
    symbolic_state[3] = np.radians(1.0)
    symbolic_state[6] = 4.6
    symbolic_state[2] = eight.equilibrium_heave()
    symbolic_state[12] = model.anaerobic_capacity
    symbolic = np.array(
        function(symbolic_state, [0.0, 0.0, 1.0], time)).ravel()

    assert symbolic[9] == pytest.approx(numeric[9], rel=0.05), \
        (numeric[9], symbolic[9])


# --------------------------------------------------------------------------
# the criterion is bounded growth over the recovery, not stability
# --------------------------------------------------------------------------
def test_the_boat_need_not_be_stable_only_bounded_over_the_recovery(eight):
    """Framing matters here, because it sets the control objective.

    Closed loop, nothing has to converge.  The blades come back every
    stroke and the drive re-establishes roughly eighteen times the
    recovery's authority.  The requirement is only that roll does not grow
    too much between the finish and the next catch -- a finite-horizon
    boundedness condition, not an asymptotic stability one.
    """
    from coxswain.crew.balance import PhaseAuthority

    authority = PhaseAuthority.from_boat(eight)
    assert authority.drive > 10.0 * authority.recovery, \
        "the drive is where the authority is"


def test_the_tolerance_on_heel_at_the_finish_is_tiny(eight):
    """The operative number, and the reason the skill is taught as
    *set the boat at the finish* rather than *correct it on the recovery*.

    By the time there is something to correct, the authority to correct it
    is gone.
    """
    from coxswain.crew.balance import (PhaseAuthority, roll_divergence_time,
                                       static_roll_stiffness)

    authority = PhaseAuthority.from_boat(eight)
    holdable = np.degrees(authority.recovery / static_roll_stiffness(eight))
    growth = np.exp(eight.timing.recovery_duration
                    / roll_divergence_time(eight))
    tolerance = holdable / growth
    assert tolerance < 0.1, tolerance
    assert growth > 20.0, growth


def test_a_boat_is_far_more_forgiving_at_high_rating():
    """A prediction that was not put in.

    Growth over the recovery is exponential in ``recovery_duration / tau``,
    and tau is fixed by hydrostatics and inertia, so shortening the
    recovery helps disproportionately.  The model therefore reproduces one
    of the most universally reported facts in the sport -- boats fall over
    at low rate and feel solid at racing rate -- without it having been
    encoded anywhere.
    """
    from coxswain.crew.balance import (PhaseAuthority, roll_divergence_time,
                                       static_roll_stiffness)

    def tolerance(rate):
        boat = catalog.eight(rate=rate)
        authority = PhaseAuthority.from_boat(boat)
        holdable = authority.recovery / static_roll_stiffness(boat)
        growth = np.exp(boat.timing.recovery_duration
                        / roll_divergence_time(boat))
        return holdable / growth

    low, high = tolerance(24.0), tolerance(40.0)
    assert high > 20.0 * low, (low, high)


def test_the_divergence_timescale_is_set_by_the_boat_not_the_rating():
    """Which is what makes the rating scaling exponential rather than
    something the crew can trade against."""
    from coxswain.crew.balance import roll_divergence_time

    taus = [roll_divergence_time(catalog.eight(rate=rate))
            for rate in (24.0, 32.0, 40.0)]
    assert max(taus) - min(taus) < 0.01 * np.mean(taus), taus


def test_reaction_time_is_comparable_to_the_divergence_time(eight):
    """Why correction during the recovery cannot work.

    Human sensorimotor reaction is 150-250 ms; the roll mode e-folds in
    about 218 ms.  A rower who waits to feel the boat go over has already
    lost most of the margin before their correction begins.  This is the
    quantitative basis for the coaching instruction to *anticipate* rather
    than react.
    """
    from coxswain.crew.balance import roll_divergence_time

    tau = roll_divergence_time(eight)
    assert 0.15 < tau < 0.30, tau
