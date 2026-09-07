"""The menu model, which has no pygame in it and therefore no display."""

import pytest

from coxswain.viz.menu import (BOATS, COURSES, blurb_for, build_boat,
                               pause_menu, setup_menu)


def test_every_boat_the_menu_offers_can_actually_be_built():
    """A menu offering a shell the catalog cannot make is worse than a
    short menu, so this is the test that keeps the two in step."""
    for key, _label, _seats, _coxed in BOATS:
        boat, made = build_boat(key, 30.0)
        assert made == key, "%s fell back to %s" % (key, made)
        assert boat.length > 0.0


def test_the_boat_menu_matches_the_seats_it_claims():
    for key, _label, seats, _coxed in BOATS:
        boat, _made = build_boat(key, 30.0)
        assert boat.n_seats == seats


def test_the_rate_reaches_the_boat():
    slow, _ = build_boat("4+", 20.0)
    quick, _ = build_boat("4+", 36.0)
    assert quick.timing.period < slow.timing.period


def test_settings_come_back_as_keys_not_labels():
    """The caller wants "totl", not "Tail of the Lake"."""
    menu = setup_menu()
    got = menu.settings()
    assert got["race"] in [key for key, _l, _b in COURSES]
    assert got["boat"] in [key for key, _l, _s, _c in BOATS]


def test_moving_and_adjusting_the_cursor():
    menu = setup_menu(boat="4+", course="charles")
    assert menu.settings()["boat"] == "4+"
    menu.adjust(1)                       # cursor starts on Boat
    assert menu.settings()["boat"] != "4+"
    menu.adjust(-1)
    assert menu.settings()["boat"] == "4+"
    menu.move(1)                         # Course
    menu.adjust(1)
    assert menu.settings()["race"] != "charles"


def test_numeric_rows_clamp_at_their_limits():
    menu = setup_menu(rate=30.0)
    row = menu.by_key("rate")
    for _ in range(100):
        row.adjust(1)
    assert row.value == row.numeric[1]
    for _ in range(200):
        row.adjust(-1)
    assert row.value == row.numeric[0]


def test_action_rows_report_themselves_and_settings_ignore_them():
    menu = setup_menu()
    while menu.rows[menu.cursor].key != "start":
        menu.move(1)
    assert menu.enter() == "start"
    assert menu.chosen == "start"
    assert "start" not in menu.settings()


def test_the_pause_menu_offers_the_things_that_can_change_live():
    """Rate and wind can change without rebuilding the world; boat and
    course cannot, so those send you back to setup rather than pretending."""
    menu = pause_menu(rate=28.0, wind=7.0)
    keys = [row.key for row in menu.rows]
    # Rate is on the pause menu itself; wind moved in with the weather,
    # because that is what it is -- it sets the chop and the ripple.
    assert "rate" in keys
    # Wind is weather, and weather is its own menu -- not a graphics
    # preference and not something buried under one.
    assert "weather" in keys
    from coxswain.viz.menu import weather_menu
    assert "wind" in [row.key for row in weather_menu().rows]
    assert "boat" not in keys and "race" not in keys
    assert "setup" in keys
    assert menu.by_key("rate").value == 28.0


def test_the_help_line_follows_the_cursor():
    menu = setup_menu()
    first = blurb_for(menu)
    menu.move(1)
    assert blurb_for(menu) != first
    assert blurb_for(menu)


def test_an_empty_menu_does_not_crash():
    from coxswain.viz.menu import Menu

    menu = Menu("nothing", [])
    menu.move(1)
    menu.adjust(1)
    assert menu.enter() is None
    assert menu.settings() == {}
