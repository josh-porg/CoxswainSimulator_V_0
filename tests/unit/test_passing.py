"""The HOCR passing rules, as a state machine, with no physics involved.

These tests are the point of the module: the rules are discrete, stateful
and adversarial, and the only way to trust them is to assert each clause
of the rulebook separately.
"""

import numpy as np
import pytest

from coxswain.river.passing import (EIGHT_LENGTH, Entry, HeadRace, Phase,
                                    PassingRules, build_field)


def race(n=12, interval=15.0, spread=0.06, seed=3, compliance=1.0,
         length=4800.0, **kwargs):
    field = build_field(n, interval=interval, spread=spread, seed=seed)
    return HeadRace(field, length=length, compliance=compliance,
                    seed=seed, **kwargs)


# --------------------------------------------------------------------------
# geometry: open water, not bow-to-bow
# --------------------------------------------------------------------------
def test_open_water_is_clear_water_not_bow_to_bow():
    """Two bows one length apart have ZERO open water, not one length.

    Reading the rulebook's distances as bow-to-bow fires every threshold a
    full boat length early, which is the easiest way to get head racing
    wrong.
    """
    ahead = Entry(bow=1, start=0.0, speed=EIGHT_LENGTH + 4.0)
    behind = Entry(bow=2, start=0.0, speed=4.0)
    event = HeadRace([ahead, behind])
    assert event.open_water(ahead, behind, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_overlap_is_negative_open_water():
    ahead = Entry(bow=1, start=0.0, speed=5.0)
    behind = Entry(bow=2, start=0.0, speed=4.0)
    event = HeadRace([ahead, behind])
    assert event.open_water(ahead, behind, 1.0) < 0.0


# --------------------------------------------------------------------------
# the clauses
# --------------------------------------------------------------------------
def test_a_complying_field_is_never_penalised():
    """The headline invariant.

    If every crew yields when asked, the rulebook must never fire.  An
    earlier version penalised ten crews in exactly this field, because a
    role reversal begins with the boats overlapped and the yield threshold
    was breached before anyone could physically move.
    """
    event = race(compliance=1.0)
    log = event.run()
    assert log.of_kind("declare"), "field never produced a pass"
    assert log.of_kind("penalty") == []
    assert log.of_kind("disqualified") == []
    assert all(row["penalty"] == 0.0 for row in event.results())


def test_every_yield_follows_its_own_declaration():
    """No crew yields to a call that was never made."""
    log = race().run()
    declared = {}
    for entry in log.events:
        key = (entry.get("passer"), entry.get("passee"))
        if entry["event"] == "declare":
            declared[key] = entry["time"]
        elif entry["event"] == "yield":
            assert key in declared, "yield with no declaration"
            assert entry["time"] >= declared[key]


def test_declaration_happens_within_one_length_of_open_water():
    """The Passer declares at one length; never earlier."""
    event = race()
    log = event.run()
    for entry in log.of_kind("declare"):
        assert entry["open_water"] <= event.rules.declare_at + 1e-6


def test_a_complying_crew_moves_over_as_fast_as_it_physically_can():
    """The invariant that actually holds, and why it is not the obvious one.

    The tempting assertion is "a complying crew has yielded by half a
    length of open water".  It is false, and the counterexample is
    physical rather than a modelling slip: the yield takes
    ``yield_width / yield_rate`` = 10 s, and a crew closing at more than
    about 0.87 m/s covers the 8.65 m from declaration to threshold in less
    than that.  Such a pass was observed completing its yield at 6.72 m.

    The rulebook already anticipates this -- the obligation is conditioned
    on "adequate room and time" -- so the crew is not at fault and is not
    penalised.  What must hold is that they started moving immediately and
    took no longer than the manoeuvre physically requires.
    """
    event = race(compliance=1.0)
    log = event.run()
    declares = {(e["passer"], e["passee"]): e for e in log.of_kind("declare")}
    late = 0
    for entry in log.of_kind("yield"):
        opened = declares[(entry["passer"], entry["passee"])]
        assert (entry["time"] - opened["time"]
                <= event.rules.adequate_time + 1e-6)
        if entry["open_water"] < event.rules.yield_by:
            late += 1
    # Some are late; none of them are offences, which is the whole point.
    assert log.of_kind("penalty") == []


def test_a_crew_given_room_and_time_does_finish_by_half_a_length():
    """The rulebook's clause, in the conditions it was written for.

    A modest speed difference leaves plenty of time, and then the yield
    really is complete by the half-length threshold.
    """
    ahead = Entry(bow=1, start=0.0, speed=4.20)
    behind = Entry(bow=2, start=12.0, speed=4.32)     # closes at 0.12 m/s
    event = HeadRace([ahead, behind], length=4800.0)
    log = event.run(dt=0.5)
    yields = log.of_kind("yield")
    assert yields, "the pass never developed"
    assert yields[0]["open_water"] >= event.rules.yield_by - 1e-6
    assert log.of_kind("penalty") == []


def test_the_passer_chooses_the_side():
    """And the passee yields to that side, not one of its own choosing."""
    log = race().run()
    for entry in log.of_kind("yield"):
        match = [d for d in log.of_kind("declare")
                 if d["passer"] == entry["passer"]
                 and d["passee"] == entry["passee"]
                 and d["time"] <= entry["time"]]
        assert match and match[-1]["side"] == entry["side"]


def test_ignoring_the_call_is_penalised():
    log = race(compliance=0.0).run()
    assert log.of_kind("penalty"), "nobody penalised for never yielding"


def test_penalties_escalate_sixty_then_onetwenty_then_disqualification():
    rules = PassingRules()
    assert rules.penalty_for(1) == 60.0
    assert rules.penalty_for(2) == 120.0
    assert rules.penalty_for(3) is None      # disqualification


def test_a_third_offence_disqualifies():
    event = race(n=14, compliance=0.0, seed=5)
    event.run()
    offenders = [r for r in event.results() if r["offences"] >= 3]
    assert offenders, "no crew reached a third offence"
    assert all(row["disqualified"] for row in offenders)


def test_penalty_is_added_to_the_official_time():
    event = race(compliance=0.5, seed=7)
    event.run()
    for row in event.results():
        if np.isfinite(row["raw"]):
            assert row["official"] == pytest.approx(row["raw"]
                                                    + row["penalty"])


def test_nobody_is_penalised_without_adequate_time():
    """The rulebook's own condition, and the bug it guards.

    A penalty may never land sooner after the declaration than the yield
    physically takes.
    """
    event = race(compliance=0.0)
    log = event.run()
    declares = {}
    for entry in log.events:
        if entry["event"] == "declare":
            declares[(entry["passer"], entry["passee"])] = entry["time"]
        elif entry["event"] == "penalty":
            key = (entry["passer"], entry["crew"])
            assert key in declares
            assert (entry["time"] - declares[key]
                    >= event.rules.adequate_time - 1e-6)


# --------------------------------------------------------------------------
# the part that is not in the rulebook
# --------------------------------------------------------------------------
def test_a_yield_persists_while_the_boats_are_overlapped():
    """They stay on your port until you clear them.

    A Passer who draws level but cannot finish does not get to make the
    Passee move again, and the Passee does not drift back on their own.
    """
    ahead = Entry(bow=1, start=0.0, speed=4.20)
    behind = Entry(bow=2, start=10.0, speed=4.24)      # closes very slowly
    event = HeadRace([ahead, behind], length=4000.0)
    event.run(dt=0.5)
    pair = event.encounters[(2, 1)]
    assert pair.phase in (Phase.YIELDED, Phase.OVERLAPPED, Phase.COMPLETE)
    if pair.phase is not Phase.COMPLETE:
        assert abs(ahead.lateral) > 0.5 * event.rules.yield_width


def test_dropping_back_resets_the_encounter():
    """A stalled approach must be re-declared, not owed forever."""
    ahead = Entry(bow=1, start=0.0, speed=4.30)
    behind = Entry(bow=2, start=8.0, speed=4.10)
    event = HeadRace([ahead, behind], length=4000.0)
    log = event.run(dt=0.5)
    if log.of_kind("declare"):
        assert log.of_kind("reset") or log.of_kind("complete")


def test_identical_speeds_produce_no_passes_at_all():
    """The control: with no speed difference nobody ever closes."""
    field = [Entry(bow=i + 1, start=i * 15.0, speed=4.23) for i in range(8)]
    log = HeadRace(field, length=4800.0).run()
    assert log.of_kind("declare") == []
    assert log.of_kind("penalty") == []


def test_a_wider_speed_spread_produces_more_passes():
    tight = race(spread=0.02, seed=11).run().of_kind("declare")
    loose = race(spread=0.10, seed=11).run().of_kind("declare")
    assert len(loose) > len(tight)


def test_a_longer_start_interval_produces_fewer_passes():
    close = race(interval=10.0, seed=13).run().of_kind("declare")
    apart = race(interval=40.0, seed=13).run().of_kind("declare")
    assert len(apart) < len(close)


def test_everybody_finishes_or_is_disqualified():
    event = race(compliance=0.7, seed=17)
    event.run()
    for row in event.results():
        assert row["disqualified"] or np.isfinite(row["raw"])
