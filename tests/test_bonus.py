r"""The bonus run: laid along the course, collected at the bow, paid for
on the call.  And the secret that unlocks it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from coxswain.viz.bonus import (BOOST_CALL, BOOST_SECONDS, COIN_EVERY,
                                SECRET, BonusRun, SecretTyper, pickup_solids)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _straight(length=3000.0):
    return np.column_stack([np.linspace(0.0, length, 61), np.zeros(61)])


def test_pickups_are_laid_along_the_course_left_on_and_right():
    run = BonusRun.along(_straight())
    coins = [p for p in run.pickups if p.kind == "coin"]
    boosts = [p for p in run.pickups if p.kind == "boost"]
    assert len(coins) == int(3000.0 / COIN_EVERY) - 1
    assert len(boosts) >= 4
    # alternating offsets: a third on the line, a third each side
    offs = np.array([p.north for p in coins])
    assert (np.abs(offs) < 1e-9).sum() >= len(coins) // 3 - 1
    assert (offs > 1.0).sum() >= len(coins) // 3 - 1
    assert (offs < -1.0).sum() >= len(coins) // 3 - 1
    # spaced along the line, none at the start
    xs = sorted(p.east for p in coins)
    assert xs[0] >= COIN_EVERY - 1e-9
    assert np.allclose(np.diff(xs), COIN_EVERY)
    # boosts sit on the line
    assert all(abs(p.north) < 1e-9 for p in boosts)


def test_a_boat_on_the_line_collects_a_third_and_a_steered_one_collects_all():
    run = BonusRun.along(_straight())
    total = run.total_coins
    for x in np.arange(0.0, 3000.0, 1.0):
        run.collect(x, 0.0, now=0.0)
    on_line = run.coins
    assert total // 3 - 1 <= on_line <= total // 3 + 2
    run = BonusRun.along(_straight())
    for p in list(run.pickups):
        run.collect(p.east, p.north, now=0.0)
    assert run.coins == total
    assert all(p.taken for p in run.pickups)


def test_a_coin_is_taken_once_and_the_reach_is_a_blades_length():
    run = BonusRun.along(_straight())
    coin = next(p for p in run.pickups if p.kind == "coin" and p.north == 0.0)
    assert run.collect(coin.east + 3.4, 0.0, now=0.0) == [coin]
    assert run.collect(coin.east, 0.0, now=0.0) == []           # once
    far = next(p for p in run.pickups if p.kind == "coin" and p.north > 1.0)
    assert run.collect(far.east, 0.0, now=0.0) == []            # 6 m off


def test_a_boost_adds_to_the_call_for_eight_seconds_and_extends_from_now():
    run = BonusRun.along(_straight())
    boost = next(p for p in run.pickups if p.kind == "boost")
    assert run.call_bonus(now=10.0) == 0.0
    run.collect(boost.east, boost.north, now=10.0)
    assert run.call_bonus(now=10.0) == BOOST_CALL
    assert run.call_bonus(now=10.0 + BOOST_SECONDS - 0.01) == BOOST_CALL
    assert run.call_bonus(now=10.0 + BOOST_SECONDS + 0.01) == 0.0
    # a second boost during the first runs from NOW, not stacked on the end
    second = [p for p in run.pickups if p.kind == "boost"][1]
    run.collect(second.east, second.north, now=14.0)
    assert run.boost_left(now=14.0) == pytest.approx(BOOST_SECONDS)
    assert run.boosts == 2 and run.score() == run.coins + 10


def test_the_hud_line_says_what_matters():
    run = BonusRun.along(_straight())
    line = run.hud_line(now=0.0)
    assert line.startswith("BONUS") and "coins 0/" in line and "BOOST" not in line
    boost = next(p for p in run.pickups if p.kind == "boost")
    run.collect(boost.east, boost.north, now=0.0)
    assert "BOOST 8 s" in run.hud_line(now=0.0)


def test_visible_is_only_what_is_near_and_not_taken():
    run = BonusRun.along(_straight())
    near = run.visible(600.0, 0.0, within=200.0)
    assert near and all(abs(p.east - 600.0) <= 200.0 for p in near)
    for p in near:
        run.collect(p.east, p.north, now=0.0)
    assert run.visible(600.0, 0.0, within=200.0) == []


def test_pickup_solids_are_small_closed_octahedra_in_two_colours():
    run = BonusRun.along(_straight())
    shown = run.visible(600.0, 0.0, within=300.0)
    vertices, colours = pickup_solids(shown, 0.0)
    assert len(vertices) == 24 * len(shown)          # 8 faces x 3
    assert vertices.dtype == np.float32
    gold = np.array([1.0, 0.82, 0.2]); green = np.array([0.3, 1.0, 0.45])
    kinds = {p.kind for p in shown}
    if "coin" in kinds:
        assert np.any(np.all(np.isclose(colours, gold), axis=1))
    if "boost" in kinds:
        assert np.any(np.all(np.isclose(colours, green), axis=1))
    assert pickup_solids([], 0.0) is None


def test_the_secret_is_the_word_typed_and_nothing_else_is_kept():
    typer = SecretTyper()
    assert SECRET == "boost"
    hits = [typer.feed(c) for c in "xxbooxboost"]
    assert hits.count(True) == 1 and hits[-1]
    assert typer.buffer == ""                        # cleared on the hit
    assert not typer.feed("\n") and not typer.feed("")
    assert len(typer.buffer) <= len(SECRET)
    # case does not matter; a second typing unlocks again (harmless)
    assert [typer.feed(c) for c in "BOOST"][-1]


def test_the_trainer_wires_it_secret_row_pickups_boost_hud_and_best():
    text = source("scripts", "fpv.py")
    assert "secret = SecretTyper()" in text
    assert 'if secret.feed(getattr(event, "unicode", "") or ""):' in text
    assert '_settings.update(bonus_unlocked="on")' in text
    assert "def add_bonus_row(menu" in text
    assert 'Choice("bonus", "Bonus run"' in text
    assert "bonus = (BonusRun.along(course)" in text
    assert "pickup_vao.render(vertices=len(p_vertices))" in text
    # the boost rides the call on BOTH places the call is spent
    assert text.count("nominal_power * call_live") == 2
    assert "nominal_power * call," not in text and "nominal_power * call\n" not in text
    assert "bonus.collect(float(_bow[0]), float(_bow[1]))" in text
    assert "lines.append(bonus.hud_line())" in text
    assert "_settings_mod.update(bonus_best=score)" in text
    assert '"--bonus"' in text
