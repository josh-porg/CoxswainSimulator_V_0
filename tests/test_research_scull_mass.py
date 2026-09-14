"""Research profiles row [CR06]'s measured scull; the shipped one does not.

The catalogue scull sets no mass and so inherits ``Oar``'s 2.7 kg, which is
documented as a composite sweep oar.  [CR06] Table 1 measured a scull at
1.2 kg.  The shipped game reads oar mass for recovery roll authority, so the
correction is a profile field, applied to sculling rigs only.
"""

import dataclasses

import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.boats.rig import SCULLING_OAR, SWEEP_OAR


def oar_masses(boat):
    return {lock.oar.mass for seat in boat.rig.seats for lock in seat.oarlocks}


def test_shipped_keeps_the_rig_and_research_profiles_carry_cr06s_scull():
    assert physics.resolve(physics.SHIPPED).scull_mass is None
    assert physics.resolve("research").scull_mass == 1.2
    assert physics.resolve("learned").scull_mass == 1.2


def test_the_catalogue_scull_still_inherits_the_sweep_default():
    """What the correction is for; if this changes, the profile field's
    reason has gone and it should be revisited."""
    assert SCULLING_OAR.mass == SWEEP_OAR.mass == 2.7


def test_shipped_apply_leaves_a_single_s_oars_alone():
    boat = physics.resolve(physics.SHIPPED).apply(catalog.build("1x", rate=30.0))
    assert oar_masses(boat) == {2.7}


def test_research_apply_lightens_sculls_and_only_their_mass():
    before = catalog.build("1x", rate=30.0)
    boat = physics.resolve("research").apply(catalog.build("1x", rate=30.0))
    assert oar_masses(boat) == {1.2}
    for old, new in zip(before.rig.seats, boat.rig.seats):
        for a, b in zip(old.oarlocks, new.oarlocks):
            assert dataclasses.replace(a.oar, mass=1.2) == b.oar
            assert (a.position == b.position).all() and a.side == b.side


def test_research_apply_leaves_sweep_oars_alone():
    boat = physics.resolve("research").apply(catalog.build("8+", rate=32.0))
    assert oar_masses(boat) == {2.7}


def test_a_non_positive_scull_mass_is_refused():
    with pytest.raises(ValueError, match="scull_mass"):
        physics.PhysicsProfile(name="x", summary="x", blade_tier=0,
                               rower="prescribed", scull_mass=0.0)
