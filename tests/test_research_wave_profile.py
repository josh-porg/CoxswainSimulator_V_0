"""Research profiles get the corrected wave physics; the shipped one does not.

``research`` and ``learned`` carry ``wave="sretenskii"``: Michell's integral
with trapezoid weights, and Sretenskii's finite-depth integral in place of
the chosen shallow-water factor.  ``shipped`` stays on ``"michell"`` -- the
game is frozen -- and its resistance path must be exactly what it was.
"""

import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.hydro.finite_depth_michell import FiniteDepthWaveTable
from coxswain.hydro.michell import MichellWave, elliptical_offsets
from coxswain.hydro.resistance import hull_resistance
from coxswain.hydro.shallow import ShallowWaterModel


@pytest.fixture(scope="module")
def eight():
    return catalog.build("8+", rate=32.0)


@pytest.fixture(scope="module")
def coarse(eight):
    """A cheap depth-aware table and the plain deep one it wraps."""
    x, z, half = elliptical_offsets(eight.offsets, stations=161, levels=21)
    table = FiniteDepthWaveTable(station=x, level=z, half_beam=half)
    # Sampled as the research table samples its deep curve (301 speeds).
    plain = MichellWave(station=x, level=z, half_beam=half,
                        quadrature="trapezoid").tabulate(points=301)
    return table, plain


def submerged(boat):
    return boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)


def resistance(boat, speed, shallow, table):
    return hull_resistance(np.array([speed, 0.0, 0.0]), submerged(boat),
                           boat.length, boat.water, boat.resistance,
                           shallow, wave_table=table)[1]


def test_shipped_keeps_michell_and_research_profiles_move():
    assert physics.resolve(physics.SHIPPED).wave == "michell"
    assert physics.resolve("research").wave == "sretenskii"
    assert physics.resolve("learned").wave == "sretenskii"


def test_unknown_wave_model_is_refused():
    with pytest.raises(ValueError, match="wave model"):
        physics.PhysicsProfile(name="x", summary="x", blade_tier=0,
                               rower="prescribed", wave="constant")


def test_shipped_apply_leaves_the_wave_table_alone():
    boat = catalog.build("4+", rate=30.0)
    before = boat.wave_table
    physics.resolve(physics.SHIPPED).apply(boat)
    assert boat.wave_table is before
    assert not hasattr(boat.wave_table, "at_depth")


def test_research_apply_hands_the_boat_a_depth_aware_table():
    boat = physics.resolve("research").apply(catalog.build("4+", rate=30.0))
    assert isinstance(boat.wave_table, FiniteDepthWaveTable)


def test_deep_water_uses_the_trapezoid_table(eight, coarse):
    table, plain = coarse
    detail = resistance(eight, 5.0, None, table)
    assert detail["wave"] == pytest.approx(float(plain(5.0)), rel=1e-12)
    assert detail["depth_factor"] == 1.0


def test_shallow_water_answers_from_the_integral_not_the_factor(eight, coarse):
    table, _plain = coarse
    shallow = ShallowWaterModel(depth=2.6)
    speed = 5.5                                   # Fr_h 1.09
    detail = resistance(eight, speed, shallow, table)
    assert detail["wave"] == pytest.approx(table.at_depth(speed, 2.6),
                                           rel=1e-12)
    assert detail["depth_factor"] == pytest.approx(
        detail["wave"] / float(table(speed)), rel=1e-12)
    # and it is not the chosen factor, which reads near 2.9 here
    assert detail["depth_factor"] < 0.7 * float(shallow.factor(speed))


def test_a_plain_table_keeps_the_shipped_product_exactly(eight, coarse):
    _table, plain = coarse
    shallow = ShallowWaterModel(depth=2.6)
    detail = resistance(eight, 5.5, shallow, plain)
    assert detail["wave"] == float(np.abs(plain(5.5))) * float(
        shallow.factor(5.5))
