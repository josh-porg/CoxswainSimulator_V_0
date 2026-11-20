"""Excitation modules from on-water EMG and the stroke's phase oscillator (coxswain/crew/excitation.py; SOURCES §253)."""
import numpy as np
import pytest

from coxswain.crew.excitation import Modules, PhaseOscillator, PITTO2024_FINISH


def test_the_modules_reproduce_the_sources_own_timing():
    """Pitto et al. (2024) state the arm pullers peak together at ~55 % of the drive, triceps at 34 %, and the trunk
    extensors load early: the digitised modules must say the same (the reading's check, not a fit)."""
    m = Modules.pitto2024()
    drive = lambda name: m.peak_phase(name) / PITTO2024_FINISH  # noqa: E731
    assert drive("arm_pullers") == pytest.approx(0.55, abs=0.03)
    assert drive("arm_extensors") == pytest.approx(0.34, abs=0.03)
    assert drive("trunk_extensors") < drive("arm_pullers")          # trunk before arms (the rowing sequence)
    assert m.peak_phase("trunk_flexors") > PITTO2024_FINISH - 0.1   # the flexors at the finish and after
    for name in m.names:
        w = m(name, np.linspace(0, 1, 200))
        assert w.max() == pytest.approx(1.0, abs=0.01) and w.min() >= -0.05


def test_the_leg_extensors_are_not_invented():
    with pytest.raises(KeyError, match="leg extensors were not recorded"):
        Modules.pitto2024()("leg_extensors", 0.1)


def test_the_oscillator_entrains_to_the_boats_stroke():
    """Free-running at 1.7 s against events every 1.85 s, phase resetting with period adaptation locks on: the period
    converges to the events' interval and the phase error at each event shrinks (Kuo 2002's predictor behaviour)."""
    osc = PhaseOscillator(period=1.70, reset_gain=0.5, period_gain=0.3)
    dt, T, t = 0.001, 1.85, 0.0
    errs = []
    for k in range(1, 31):
        while t < k * T - 1e-9:
            osc.step(dt)
            t += dt
        errs.append(abs(osc.event(t)))
    assert osc.period == pytest.approx(T, rel=0.01)
    assert errs[-1] < 0.01 and errs[-1] < errs[0]


def test_a_free_running_oscillator_ignores_events():
    osc = PhaseOscillator(period=2.0, reset_gain=0.0, period_gain=0.0)
    osc.step(0.5)
    osc.event(0.5)
    assert osc.phase == pytest.approx(0.25) and osc.period == 2.0
    with pytest.raises(ValueError):
        PhaseOscillator(period=-1.0)
