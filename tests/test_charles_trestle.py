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


def test_every_arch_bridge_reaches_dry_land():
    """A bridge may not begin in the middle of the river.

    Laid out on the inventory length alone, River Street did exactly
    that: the raster water across its gate is 77.9 m wide and its NBI
    ``structure_length`` is 64.0, so the arches stopped about 7 m short
    of each bank and the approach embankments stood in open water.

    ``waterway`` will not reveal this, because it clamps the wet opening
    *to* the structure length -- right for navigation, where a bridge
    cannot open wider than it is long, and circular if you use it to
    check how long to draw the bridge.  So this measures the wet run raw
    and requires the drawn structure to contain it.
    """
    from coxswain.river.bridges import BRIDGE_STRUCTURE
    from coxswain.viz.worldmesh import ABUTMENT, _raw_waterway

    from coxswain.river import charles
    from coxswain.river.charts import CourseGeometry

    geometry = CourseGeometry(channel=charles.charles_channel())
    checked = 0
    for gate, _distance in geometry.gates_on_course():
        structure = BRIDGE_STRUCTURE.get(gate.name)
        if structure is None or not getattr(structure, "structure_length",
                                            None):
            continue
        wet = _raw_waterway(gate, geometry.channel)
        if wet is None:
            continue
        start = np.asarray(gate.start, dtype=float)
        end = np.asarray(gate.end, dtype=float)
        full = float(np.hypot(*(end - start)))
        length = float(structure.structure_length)

        low = max(min(wet[0] - ABUTMENT, 0.5 * (full - length)), 0.0)
        high = min(max(wet[1] + ABUTMENT, 0.5 * (full + length)), full)
        assert low <= wet[0] + 1e-6, (gate.name, low, wet)
        assert high >= wet[1] - 1e-6, (gate.name, high, wet)
        checked += 1
    assert checked >= 4, checked


def test_arch_piers_stand_under_the_arch_springings():
    """The solid piers must be where the openings are cut, in the MESH.

    ``arch_bridge`` cut its openings between the measured pier stations
    and then drew its solid piers at even fractions of the length -- two
    loops with two ideas of where a pier is, agreeing only when the bays
    happen to be equal.  Western Avenue's are not: its navigable arch is
    set by the channel and the side arches take up the rest, and the
    solid pier stood 17.3 m from the springing it was meant to carry.
    River Street was 6.0 m out, Larz Anderson 5.4 m.  Weeks, with even
    bays, never showed it.

    Every earlier test here checked *stations* on the navigation side.
    This one reads the drawn geometry back, because that is where the
    bug lived.
    """
    from coxswain.viz.worldmesh import arch_bridge

    start = np.array([0.0, 0.0])
    end = np.array([113.6, 0.0])
    pier_colour = (0.58, 0.57, 0.54)
    stations = [55.1, 81.9]                    # Western Avenue, measured
    mesh = arch_bridge(start, end, 18.0, 7.5, 1.4, 3, piers=stations,
                       pier_colour=pier_colour)
    assert mesh is not None

    # Pier faces are the ones in the pier colour; take their x-centres
    # and cluster them, which gives one station per solid.
    is_pier = np.all(np.abs(mesh.colours - np.asarray(pier_colour)) < 1e-6,
                     axis=1)
    xs = mesh.vertices[is_pier, 0]
    assert len(xs), "no pier faces found -- did the colour change?"
    found = []
    # A pier is 0.13 of a bay thick -- 3.5 m on the 26.8 m centre span
    # -- so its two faces must land in ONE cluster: split only on a gap
    # wider than any pier could be.
    for x in sorted(set(np.round(xs, 3))):
        if not found or x - found[-1][-1] > 8.0:
            found.append([x])
        else:
            found[-1].append(x)
    centres = sorted(float(np.mean(group)) for group in found)
    wanted = [0.0, 55.1, 81.9, 113.6]
    assert len(centres) == len(wanted), centres
    for got, want in zip(centres, wanted):
        # An abutment is trimmed to the end of the bridge, so it sits
        # half a pier inboard; the inner piers must be dead on.
        tolerance = 0.5 * 0.13 * 26.8 + 0.05 if want in (0.0, 113.6) else 0.05
        assert abs(got - want) < tolerance, (got, want, centres)

    # And the even layout it replaced would have FAILED this: 37.9 and
    # 75.8 are nowhere near 55.1 and 81.9.
    for wrong in (37.9, 75.8):
        assert min(abs(wrong - c) for c in centres) > 5.0


def test_a_pier_is_sized_for_its_narrower_neighbour():
    """A pier between a wide arch and a narrow one must not eat the narrow one."""
    from coxswain.viz.worldmesh import PIER_FRACTION, arch_bridge

    pier_colour = (0.58, 0.57, 0.54)
    # One 60 m bay and one 10 m bay.
    mesh = arch_bridge(np.array([0.0, 0.0]), np.array([70.0, 0.0]), 12.0,
                       6.0, 1.0, 2, piers=[60.0], pier_colour=pier_colour)
    is_pier = np.all(np.abs(mesh.colours - np.asarray(pier_colour)) < 1e-6,
                     axis=1)
    xs = mesh.vertices[is_pier, 0]
    inner = xs[(xs > 50.0) & (xs < 68.0)]
    width = float(inner.max() - inner.min())
    # Sized on the 10 m bay, not on the 60 m one.
    assert width <= PIER_FRACTION * 10.0 + 1e-3, width      # float32
    assert width > 0.5


def _arch_meshes_on_the_course():
    """Every arch bridge the real assembly path builds, keyed by gate.

    ``bridge_solids`` hands back ONE merged part, so each bridge is
    recovered by projecting the vertices onto its own gate line and
    keeping those within the gate's run and 15 m of its axis.
    """
    from coxswain.river import charles
    from coxswain.river.bridges import deck_geometry
    from coxswain.river.charts import CourseGeometry
    from coxswain.viz.worldmesh import bridge_solids

    geometry = CourseGeometry(channel=charles.charles_channel())
    merged = bridge_solids("charles", None)
    assert merged is not None
    out = []
    for gate, _d in geometry.gates_on_course():
        if (deck_geometry(gate.name) or ("",))[0] != "arch":
            continue
        start = np.asarray(gate.start, dtype=float)
        end = np.asarray(gate.end, dtype=float)
        full = float(np.hypot(*(end - start)))
        along = (end - start) / full
        across = np.array([-along[1], along[0]])
        rel = merged.vertices[:, :2] - start
        near = ((np.abs(rel @ across) < 15.0) & (rel @ along > -1.0)
                & (rel @ along < full + 1.0))
        assert near.any(), gate.name
        out.append((gate, merged.vertices[near], merged.colours[near]))
    return geometry, out


def test_every_arch_bridge_has_its_piers_where_navigation_has_them():
    """Read the mesh back and check it against ``derive_piers``, per bridge.

    Not a check on stations -- earlier tests do that -- but on the drawn
    solids, through the same assembly path the renderer uses.  It found
    two separate bugs: the solid piers drawn at even fractions while
    the openings followed the measured stations (Western Avenue 17.3 m
    out), and Weeks skipped by a gate on an inventory length a footbridge
    does not have (13.4 m out, and arches on dry land).
    """
    from coxswain.river.bridges import derive_piers

    geometry, bridges = _arch_meshes_on_the_course()
    assert len(bridges) >= 4, [g.name for g, _v, _c in bridges]
    pier_colour = np.asarray((0.58, 0.57, 0.54))
    checked = 0
    for gate, vertices, colours in bridges:
        start = np.asarray(gate.start, dtype=float)
        end = np.asarray(gate.end, dtype=float)
        along = (end - start) / float(np.hypot(*(end - start)))
        piers = derive_piers(gate, geometry.channel)
        if len(piers) < 1:
            continue
        is_pier = np.all(np.abs(colours - pier_colour) < 1e-6, axis=1)
        assert is_pier.any(), gate.name
        stations = (vertices[is_pier, :2] - start) @ along
        for pier in piers:
            want = float(pier.centre)
            nearest = float(np.min(np.abs(stations - want)))
            # Inside the solid, or at worst against its face.
            assert nearest < 2.0, (gate.name, want, nearest)
        checked += 1
    assert checked >= 4, checked


def test_no_arch_bridge_stands_on_dry_land_past_its_abutments():
    """Arches end within an abutment of the water, both ends."""
    from coxswain.viz.worldmesh import ABUTMENT, _raw_waterway

    geometry, bridges = _arch_meshes_on_the_course()
    arch_colour = np.asarray((0.72, 0.71, 0.67))
    for gate, vertices, colours in bridges:
        wet = _raw_waterway(gate, geometry.channel)
        if wet is None:
            continue
        start = np.asarray(gate.start, dtype=float)
        end = np.asarray(gate.end, dtype=float)
        along = (end - start) / float(np.hypot(*(end - start)))
        # The ARCH faces only: the approach slabs legitimately run on
        # to the deck-way ends over dry land.
        is_arch = np.all(np.abs(colours - arch_colour) < 1e-6, axis=1)
        assert is_arch.any(), gate.name
        stations = (vertices[is_arch, :2] - start) @ along
        assert stations.min() >= wet[0] - ABUTMENT - 6.0, (
            gate.name, float(stations.min()), wet)
        assert stations.max() <= wet[1] + ABUTMENT + 6.0, (
            gate.name, float(stations.max()), wet)
