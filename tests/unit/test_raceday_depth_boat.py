"""card.depth_boat: the race-day evaluator takes the research depth physics only when asked.

Unstamped and shipped boats must keep the default factor path, so race cards and the report do not
move; a boat stamped with a research profile is handed to the evaluator (Sretenskii wave term only).
"""
from types import SimpleNamespace

from coxswain.raceday import card


def test_unstamped_boat_keeps_default_path():
    assert card.depth_boat(SimpleNamespace()) is None
    assert card.depth_boat(None) is None


def test_shipped_boat_keeps_default_path():
    assert card.depth_boat(SimpleNamespace(physics_profile="shipped")) is None


def test_research_boat_is_passed_through():
    boat = SimpleNamespace(physics_profile="research")
    assert card.depth_boat(boat) is boat
