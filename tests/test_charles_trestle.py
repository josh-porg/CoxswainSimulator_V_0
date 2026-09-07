r"""The Grand Junction trestle, which is a trestle and not a plank.

The Charles' two non-arch bridges were drawn as a flat slab on two piers
at the thirds.  For the Eliot that is merely plain; for the Grand
Junction it is wrong in the one way that matters, because a 149 m steel
railway trestle standing on a long row of legs is the most recognisable
thing on the opening 500 m, and its pier positions have been surveyed
since before the renderer existed -- ``MEASURED_PIERS`` was being read
by the navigation code and ignored by the picture.
"""

from __future__ import annotations

import numpy as np


def _gate(name="Grand Junction RR"):
    from coxswain.river import charles
    from coxswain.river.charts import CourseGeometry

    geometry = CourseGeometry(channel=charles.charles_channel())
    for gate, _distance in geometry.gates_on_course():
        if gate.name == name:
            return gate
    raise AssertionError("%s is not on the course" % name)


def test_the_surveyed_piers_are_where_the_survey_puts_them():
    """Five piers, at the measured stations, not seven evenly spaced."""
    from coxswain.river.bridges import MEASURED_PIERS
    from coxswain.river.charles import CHARLES_ORIGIN
    from coxswain.river.course import local_tangent_plane

    gate = _gate()
    start = np.asarray(gate.start, dtype=float)
    end = np.asarray(gate.end, dtype=float)
    length = float(np.hypot(*(end - start)))
    along = (end - start) / length

    measured = MEASURED_PIERS["Grand Junction RR"]
    lats = np.array([point[0] for point in measured])
    lons = np.array([point[1] for point in measured])
    east, north = local_tangent_plane(lats, lons, CHARLES_ORIGIN)
    stations = sorted(float((np.array([e, n]) - start) @ along)
                      for e, n in zip(east, north))

    assert len(stations) == 5
    # Every pier lands on the bridge, and they are not evenly spaced --
    # which is what the old construction assumed.
    assert all(0.0 < value < length for value in stations)
    gaps = np.diff(stations)
    assert gaps.std() > 1.0, gaps


def test_the_trestle_is_open_steel_not_a_slab():
    """It must carry far more geometry than a slab on two boxes.

    A slab plus two piers is 12 + 2 x 12 triangles.  Open steelwork on
    five piers cannot come out anywhere near that, so this catches a
    silent fall back to the old construction without pinning an exact
    count that any change to the lattice would break.
    """
    from coxswain.viz.worldmesh import bridge_solids

    built = bridge_solids("charles", None)
    assert built is not None
    assert built.vertices.shape[0] // 3 > 500, built.vertices.shape


def test_arch_bridges_are_drawn_at_their_inventory_length():
    """Piers must sit near the NBI centre span, not at deck-line thirds.

    OpenStreetMap's deck way runs the length of the *roadway*, approach
    embankments included: Western Avenue's is 152 m against a bridge of
    85.3 m.  Dividing the whole way into three equal arches put the
    piers 51 m apart where the centre span is 26.8 m -- about twice the
    real spacing, on a bridge a crew lines an arch up on from several
    hundred metres out.

    Trimming to the inventory length lands them within about a metre of
    laying the centre span symmetrically about the middle, which is what
    the navigation side (``derive_piers``) has always done.
    """
    from coxswain.river.bridges import BRIDGE_STRUCTURE, derive_piers

    from coxswain.river import charles
    from coxswain.river.charts import CourseGeometry

    geometry = CourseGeometry(channel=charles.charles_channel())
    checked = 0
    for gate, _distance in geometry.gates_on_course():
        structure = BRIDGE_STRUCTURE.get(gate.name)
        if structure is None or not getattr(structure, "structure_length",
                                            None):
            continue
        if not getattr(structure, "max_span", None) or structure.main_spans != 3:
            continue
        start = np.asarray(gate.start, dtype=float)
        end = np.asarray(gate.end, dtype=float)
        full = float(np.hypot(*(end - start)))
        length = float(structure.structure_length)
        if not 0.0 < length < full:
            continue

        piers = derive_piers(gate, geometry.channel)
        if len(piers) < 2:
            continue
        # Centred on the channel, as derive_piers lays it, then divided
        # into three.  Centring on the deck line instead put Western
        # Avenue's arches 27 m out, because the river does not run under
        # the middle of the road.
        middle = 0.5 * (float(piers[0].centre) + float(piers[-1].centre))
        drawn = [middle - length / 2.0 + length * k / 3.0 for k in (1, 2)]
        wanted = [float(piers[0].centre), float(piers[-1].centre)]
        for got, want in zip(drawn, wanted):
            assert abs(got - want) < 2.5, (gate.name, got, want)
        checked += 1
    assert checked >= 3, checked
