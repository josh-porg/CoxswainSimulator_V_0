"""The oar's measured centre of mass (SOURCES §260): default unchanged, inertia follows it, the cache key sees it."""
import dataclasses

import pytest

from coxswain.boats import catalog
from coxswain.boats.rig import SCULLING_OAR, Oar
from coxswain.sim.dynamic_oar import _match_key


def test_default_is_the_uniform_rod():
    oar = Oar(length=2.885, inboard=0.875, blade_area=0.083, blade_length=0.43, mass=1.2)
    assert oar.centre_of_mass_offset == pytest.approx(2.885 / 2 - 0.875)
    assert oar.inertia_about_lock == pytest.approx(1.2 * (2.885 ** 2 / 12 + (2.885 / 2 - 0.875) ** 2))


def test_a_measured_centre_moves_the_inertia():
    rod = Oar(length=2.885, inboard=0.875, blade_area=0.074, blade_length=0.43, mass=1.375)
    measured = dataclasses.replace(rod, centre_of_mass=0.665)
    assert measured.centre_of_mass_offset == 0.665
    assert measured.inertia_about_lock == pytest.approx(1.375 * (2.885 ** 2 / 12 + 0.665 ** 2))
    assert measured.inertia_about_lock > rod.inertia_about_lock


def test_the_centre_must_lie_on_the_outboard():
    with pytest.raises(ValueError):
        Oar(length=2.885, inboard=0.875, centre_of_mass=2.5)
    with pytest.raises(ValueError):
        Oar(length=2.885, inboard=0.875, centre_of_mass=-0.1)


def _with(boat, **changes):
    rig = boat.rig
    boat.rig = dataclasses.replace(rig, seats=tuple(
        dataclasses.replace(seat, oarlocks=tuple(
            dataclasses.replace(lock, oar=dataclasses.replace(lock.oar, **changes)) for lock in seat.oarlocks))
        for seat in rig.seats))
    return boat


def test_the_match_key_sees_blade_area_and_centre_of_mass():
    base = _match_key(catalog.single_scull(), 300.0, "sweep", "slip")
    assert _match_key(_with(catalog.single_scull(), blade_area=0.074), 300.0, "sweep", "slip") != base
    assert _match_key(_with(catalog.single_scull(), centre_of_mass=0.665), 300.0, "sweep", "slip") != base
    assert SCULLING_OAR.centre_of_mass is None
