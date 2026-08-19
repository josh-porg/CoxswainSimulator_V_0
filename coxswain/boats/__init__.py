"""Boats: hull offsets, rigging, crew layout, and a catalogue of presets."""

from . import catalog, rig
from .boat import Boat
from .catalog import build, coxed_four, eight, single_scull

__all__ = ["Boat", "build", "catalog", "coxed_four", "eight", "rig",
           "single_scull"]
