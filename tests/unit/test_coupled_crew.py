"""Crew synchronisation as coupled oscillators.

Section 21 gave rowers individual phases as *inputs*.  This tests them as
a *dynamical* outcome: a crew converges on a common rhythm, imperfectly,
through two physically different coupling channels.

[L-PUB] Leonard, N. E. (Princeton MAE / PACM) -- phase models of coupled
        oscillators and collective motion.
        https://naomi.princeton.edu/publications/
[KUR]   Kuramoto, Y. Chemical Oscillations, Waves, and Turbulence (1984).
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.synchronisation import (CoupledCrew, all_to_all_topology,
                                           mixed_topology,
                                           stroke_chain_topology)


@pytest.fixture(scope="module")
def omega():
    return 2.0 * np.pi / catalog.eight(rate=32.0).timing.period


def _settle(crew, omega, seconds=18.0, dt=0.004):
    for i in range(int(seconds / dt)):
        crew.step(i * dt, dt, omega)
    return crew


# --------------------------------------------------------------------------
# topologies
# --------------------------------------------------------------------------
def test_the_chain_is_directed_towards_the_stern():
    """Rowers face the stern, so they can only see the crew ahead of them.

    The stroke seat sees nobody and sets the reference, which is exactly
    the role the seat has in a real boat.
    """
    matrix = stroke_chain_topology(8)
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 0.0
    assert matrix[7].sum() == 0.0, "stroke follows nobody"


def test_mean_field_is_symmetric_and_normalised():
    matrix = all_to_all_topology(8)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), np.zeros(8))
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(8))


def test_mixed_topology_interpolates():
    pure_chain = mixed_topology(8, 1.0)
    pure_field = mixed_topology(8, 0.0)
    np.testing.assert_allclose(pure_chain, stroke_chain_topology(8))
    np.testing.assert_allclose(pure_field, all_to_all_topology(8))


# --------------------------------------------------------------------------
# the dynamics
# --------------------------------------------------------------------------
def test_an_identical_crew_stays_together(omega):
    crew = CoupledCrew(n_seats=8, topology=all_to_all_topology(8))
    _settle(crew, omega, seconds=6.0)
    assert crew.coherence == pytest.approx(1.0, abs=1e-6)


def test_an_uncoupled_detuned_crew_falls_apart(omega):
    """The control: without coupling there is nothing holding them."""
    rng = np.random.default_rng(0)
    crew = CoupledCrew(n_seats=8, detuning=rng.normal(0.0, 0.10, 8),
                       sensory_gain=0.0)
    _settle(crew, omega, seconds=30.0)
    assert crew.coherence < 0.9


def test_coupling_pulls_a_detuned_crew_together(omega):
    rng = np.random.default_rng(0)
    detuning = rng.normal(0.0, 0.10, 8)
    loose = _settle(CoupledCrew(n_seats=8, detuning=detuning,
                                sensory_gain=0.0), omega, 30.0).coherence
    tight = _settle(CoupledCrew(n_seats=8, detuning=detuning,
                                sensory_gain=6.0,
                                topology=all_to_all_topology(8)),
                    omega, 30.0).coherence
    assert tight > loose
    assert tight > 0.99


def test_the_order_parameter_measures_togetherness(omega):
    crew = CoupledCrew(n_seats=8, topology=all_to_all_topology(8))
    crew.phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    assert crew.coherence == pytest.approx(0.0, abs=1e-9)
    crew.phi = np.zeros(8)
    assert crew.coherence == pytest.approx(1.0, abs=1e-12)


def test_a_crew_rejects_a_bad_detuning_shape():
    with pytest.raises(ValueError, match="one entry per seat"):
        CoupledCrew(n_seats=8, detuning=np.zeros(3))


def test_a_negative_delay_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        CoupledCrew(n_seats=8, delay=-0.1)


# --------------------------------------------------------------------------
# the result that answers the question
# --------------------------------------------------------------------------
def _spread_ms(chain_fraction, omega, seed=0):
    rng = np.random.default_rng(seed)
    crew = CoupledCrew(n_seats=8, detuning=rng.normal(0.0, 0.10, 8),
                       sensory_gain=4.0,
                       topology=mixed_topology(8, chain_fraction),
                       delay=0.20)
    _settle(crew, omega, seconds=20.0)
    period = catalog.eight(rate=32.0).timing.period
    offsets = crew.phase_offsets() * period * 1000.0
    return float(offsets.max() - offsets.min())


def test_the_hull_channel_synchronises_far_better_than_the_visual_one(
        omega):
    """The claim, stated at the strength the evidence supports.

    Mean-field coupling through the shared hull holds a crew about six
    times tighter than the delayed, directional visual chain at the same
    gain.  Section 21 puts the balance budget at about 65 ms of
    port/starboard split, so a crew relying on vision alone sits right at
    the edge of it while one coupled through the boat has room to spare.

    An earlier version of this test claimed something much stronger -- a
    sharp transition, with visually-coupled crews unable to synchronise at
    all.  That was an artefact of naive delayed coupling (see
    ``CoupledCrew.step``), not physics, and it did not survive making the
    coupling anticipatory.
    """
    field = _spread_ms(0.0, omega)
    chain = _spread_ms(1.0, omega)
    assert chain > 4.0 * field, (field, chain)
    assert field < 30.0


def test_a_visually_coupled_crew_sits_near_the_balance_budget(omega):
    """Section 21: about 65 ms of split spends the recovery authority."""
    assert _spread_ms(1.0, omega) > 50.0


def test_stronger_coupling_tightens_the_crew(omega):
    """And a chain-coupled crew needs a lot of it to get inside budget."""
    rng = np.random.default_rng(0)
    detuning = rng.normal(0.0, 0.10, 8)
    period = catalog.eight(rate=32.0).timing.period

    def spread(gain):
        crew = CoupledCrew(n_seats=8, detuning=detuning, sensory_gain=gain,
                           topology=stroke_chain_topology(8), delay=0.20)
        _settle(crew, omega, seconds=20.0)
        offsets = crew.phase_offsets() * period * 1000.0
        return float(offsets.max() - offsets.min())

    weak, strong = spread(1.0), spread(16.0)
    assert weak > strong
    assert strong < 65.0, "strong coupling must get inside the budget"


def test_anticipation_is_what_keeps_the_rate(omega):
    """Why the coupling advances the seen phase by the delay.

    Comparing your phase now against someone else's phase 200 ms ago means
    perpetually chasing a stale target, and it slows the whole crew --
    independently of gain, which is the signature of a structural error
    rather than a tuning one.  Human sensorimotor synchronisation is
    predictive; people tap *with* a metronome, not behind it.
    """
    crew = CoupledCrew(n_seats=8, sensory_gain=6.0, delay=0.25,
                       topology=all_to_all_topology(8))
    start = crew.phi.copy()
    seconds = 12.0
    _settle(crew, omega, seconds=seconds)
    measured = float((crew.phi - start).mean() / seconds)
    assert measured == pytest.approx(omega, rel=0.02), (measured, omega)


def test_the_two_topologies_leave_different_signatures(omega):
    """How telemetry could tell them apart.

    A chain accumulates lag monotonically toward bow; mean-field does not.
    """
    rng = np.random.default_rng(0)
    detuning = rng.normal(0.0, 0.05, 8)

    chain = _settle(CoupledCrew(n_seats=8, detuning=detuning,
                                sensory_gain=8.0,
                                topology=stroke_chain_topology(8)),
                    omega, 25.0).phase_offsets()
    field = _settle(CoupledCrew(n_seats=8, detuning=detuning,
                                sensory_gain=8.0,
                                topology=all_to_all_topology(8)),
                    omega, 25.0).phase_offsets()

    # chain: monotone from bow to stroke.  mean-field: no such ordering.
    assert np.all(np.diff(chain) > 0) or np.all(np.diff(chain) < 0)
    assert not (np.all(np.diff(field) > 0) or np.all(np.diff(field) < 0))


# --------------------------------------------------------------------------
# the vertical phase
# --------------------------------------------------------------------------
def test_the_vertical_phase_follows_the_horizontal_one(omega):
    """Blade work locked to the body is the well-rowed case."""
    crew = CoupledCrew(n_seats=8, vertical_coupling=10.0,
                       topology=all_to_all_topology(8))
    crew.psi = crew.phi + 0.5
    _settle(crew, omega, seconds=8.0)
    np.testing.assert_allclose(crew.vertical_offsets(), np.zeros(8),
                               atol=0.02)


def test_weak_vertical_coupling_leaves_the_blade_out_of_step(omega):
    """Washing out early: psi leads phi.

    Invisible to a single-phase model, and a roll disturbance arriving
    exactly when sections 15-16 say roll authority is lowest.
    """
    loose = CoupledCrew(n_seats=8, vertical_coupling=0.5,
                        topology=all_to_all_topology(8))
    loose.psi = loose.phi + 0.5
    _settle(loose, omega, seconds=4.0)

    tight = CoupledCrew(n_seats=8, vertical_coupling=10.0,
                        topology=all_to_all_topology(8))
    tight.psi = tight.phi + 0.5
    _settle(tight, omega, seconds=4.0)

    assert np.abs(loose.vertical_offsets()).max() > \
        np.abs(tight.vertical_offsets()).max()


def test_offsets_are_expressed_relative_to_the_stroke_seat(omega):
    crew = CoupledCrew(n_seats=8, topology=all_to_all_topology(8))
    crew.phi = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert crew.phase_offsets()[-1] == pytest.approx(0.0)


def test_offsets_are_in_stroke_fractions_ready_for_the_boat(omega):
    """They are assigned straight to ``Boat.phase_offsets``, which is a
    fraction of a stroke -- not radians and not seconds."""
    crew = CoupledCrew(n_seats=8, topology=all_to_all_topology(8))
    crew.phi = np.zeros(8)
    crew.phi[0] = 2.0 * np.pi
    assert crew.phase_offsets()[0] == pytest.approx(1.0)
