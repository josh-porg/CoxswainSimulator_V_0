"""The bend as a dynamic state: the blade's mass and added mass on the shaft's hinge (BendDynamics; SOURCES §224)."""
import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.crew.oar_bending import BendDynamics, OarBending
from coxswain.sim.control import Coxswain
from coxswain.sim.dynamic_oar import DynamicOarSimulator


def _sim(bending, dynamics, mass="patton"):
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    boat.power_scales = np.ones(boat.n_seats)
    return DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle", blade_law="liftdrag",
                               blade_added_mass=mass, oar_bending=bending, bend_dynamics=dynamics,
                               coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)


def _scull_law():
    boat = catalog.single_scull(rate=30.0)
    oar = boat.rig.seats[0].oarlocks[0].oar
    return OarBending.laschowski2014_scull(float(oar.blade_centre_outboard), float(oar.blade_length))


def test_bend_dynamics_need_the_hands_on_the_handle_the_added_mass_and_a_shaft():
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    kw = dict(peak_torque=0.0, catch="sweep", blade_law="liftdrag", bend_dynamics=BendDynamics())
    with pytest.raises(ValueError, match="added mass"):
        DynamicOarSimulator(boat, crew="handle", oar_bending=_scull_law(), blade_added_mass="none", **kw)
    with pytest.raises(ValueError, match="bending shaft"):
        DynamicOarSimulator(boat, crew="handle", oar_bending=False, blade_added_mass="patton", **kw)
    sim = DynamicOarSimulator(boat, crew="handle", oar_bending=_scull_law(), blade_added_mass="patton", **kw)
    assert sim._per_oar == 4                                   # angle, rate, bend, bend rate


@pytest.mark.slow
def test_a_stiff_dynamic_shaft_converges_to_the_rigid_oar_with_the_added_mass(monkeypatch):
    """The stiff limit: as K grows the run approaches the rigid oar's coupled added-mass run, the gap falling ~1/K
    (SOURCES §224: +2.35 / 0.76 / 0.24 / 0.07 % speed at K = 3e4 / 1e5 / 3e5 / 1e6). The dry blade's in-air mode at
    K = 3e5 needs a short step for RK4."""
    import coxswain.core.integrators as I
    monkeypatch.setattr(I, "estimate_step", lambda *a, **k: 0.0004)
    rigid = _sim(False, None).run_strokes(2, surge_speed=4.2)
    hinge = _scull_law().hinge_inboard_of_root
    gaps = []
    for K in (1.0e5, 3.0e5):
        run = _sim(OarBending(K, hinge), BendDynamics(damping_ratio=0.5)).run_strokes(2, surge_speed=4.2)
        gaps.append(run.settled_speed() / rigid.settled_speed() - 1.0)
    assert abs(gaps[1]) < 5e-3
    assert abs(gaps[1]) < 0.5 * abs(gaps[0])                   # converging, roughly as 1/K


def test_the_measured_scull_shaft_bends_dynamically():
    sim = _sim(_scull_law(), BendDynamics())
    run = sim.run_strokes(3, surge_speed=4.2)
    bend = np.degrees(sim._bend_history[0])
    assert np.isfinite(run.settled_speed())
    assert 1.0 < bend.max() < 20.0                             # [L14]'s shafts bend 4-5 degrees at their test load


def test_the_blade_track_is_the_gate_plus_the_bend():
    """One place forms the bent blade's angle and rate (SOURCES §227); a rigid oar's blade is the gate."""
    rigid = _sim(False, None)
    assert rigid._blade_track(0, 0.3, -2.0) == (0.3, -2.0)
    sim = _sim(_scull_law(), BendDynamics())
    sim._bend[:], sim._bend_rate[:] = 0.1, 0.5
    angle, rate = sim._blade_track(0, 0.3, -2.0)
    assert angle == pytest.approx(0.4)
    assert rate == pytest.approx(-2.0 + float(sim._bend_lever[0]) * 0.5)


def test_a_bent_blades_load_points_along_the_bent_blades_normal():
    """The loads are computed on the bent blade, so they act along its normal and shaft (SOURCES §227: they were
    applied along the straight oar's, while the added mass already used the bent normal)."""
    from coxswain.core.state import STATE_SIZE, State
    sim = _sim(_scull_law(), BendDynamics())
    state = State.from_vector(sim.augmented_initial_state(4.2)[:STATE_SIZE])
    theta, phi = 0.3, 0.12
    sim._bend[:], sim._bend_rate[:] = phi, 0.0
    sim._in_air = np.zeros(sim.n_oar_states, dtype=bool)
    sim._oar_state = (np.array([theta]), np.array([-2.0]))
    force, _moment = sim._oar_loads(0.0, state)
    expect = np.zeros(3)
    b = theta + phi
    for lock in sim.boat.rig.seats[sim._seats[0]].oarlocks:
        f_n, f_t, _arm = sim._blade_loads_arm(0, theta, -2.0, state, lock)
        side = int(lock.side)
        expect += f_n * np.array([np.cos(b), -side * np.sin(b), 0.0]) + f_t * np.array([np.sin(b), side * np.cos(b), 0.0])
    assert abs(force[0]) > 1.0
    assert force == pytest.approx(expect, rel=1e-12, abs=1e-9)


@pytest.mark.slow
def test_release_after_leaves_a_run_without_an_early_release_unchanged():
    """``release_after`` only stops the stopped-driving release acting near the catch (SOURCES §226); a rigid run whose
    blades release at the finish rows exactly as before."""
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    boat.power_scales = np.ones(boat.n_seats)
    kw = dict(peak_torque=0.0, catch="sweep", crew="handle", blade_law="liftdrag", blade_added_mass="patton",
              oar_bending=False, coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    plain = DynamicOarSimulator(boat, **kw).run_strokes(2, surge_speed=4.2)
    held = DynamicOarSimulator(boat, release_after=0.5, **kw).run_strokes(2, surge_speed=4.2)
    assert held.settled_speed() == pytest.approx(plain.settled_speed(), rel=1e-9)
