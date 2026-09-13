r"""The blade-path figure: traces from the prescribed sweep and the dynamic oar.

The figure was asked for with four labelled events -- the catch, where the
blade starts to slip, where it re-anchors, and the finish.  Drawing it from
the dynamic oar showed that one of those four is an artefact of the schedule:

* The prescribed sweep forces the oar's rate to exactly zero at the finish, so
  near the finish the boat carries the blade and slip turns positive again --
  a second flow reversal, "the blade re-anchors".
* With the oar angle a state, nothing forces that.  The oar reaches the finish
  angle still swinging, still over-driven, and the second reversal does not
  happen.  Measured on a single at 380 W: one reversal, not two.

Pinned both ways, so a change to the pull shape or the blade model that brings
the re-anchor back announces itself here rather than silently in a picture.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.viz.bladepath import (BladeTrace, dynamic_trace, plot,
                                    prescribed_trace)


@pytest.fixture(scope="module")
def single():
    return catalog.build("1x", rate=30.0)


@pytest.fixture(scope="module")
def dynamic_single(single):
    from coxswain.sim.dynamic_oar import DynamicOarSimulator

    sim = DynamicOarSimulator(
        single, peak_torque=DynamicOarSimulator.peak_torque_for_power(single,
                                                                      380.0))
    return sim, sim.run_strokes(6, surge_speed=3.9)


# ---------------------------------------------------------------------------
# the prescribed sweep
# ---------------------------------------------------------------------------
def test_the_prescribed_sweep_reverses_twice(single):
    """Zero sweep rate at both ends forces two reversals."""
    trace = prescribed_trace(single, 4.0, "prescribed")
    assert len(trace.crossings) == 2, trace.crossings
    assert trace.slip[0] > 0.0 and trace.slip[-1] > 0.0
    assert trace.x[0] == 0.0 and trace.y[0] == 0.0


def test_prescribed_run_back_shrinks_as_the_boat_speeds_up(single):
    backs = [prescribed_trace(single, v, "p").run_back
             for v in (2.0, 3.0, 4.0)]
    assert backs[0] > backs[1] >= backs[2], backs


# ---------------------------------------------------------------------------
# the dynamic oar
# ---------------------------------------------------------------------------
def test_the_dynamic_trace_is_the_drive_from_the_catch(dynamic_single):
    sim, run = dynamic_single
    trace = dynamic_trace(sim, run, "dynamic")
    assert trace.x.size > 20
    assert trace.x[0] == 0.0 and trace.y[0] == 0.0
    for array in (trace.x, trace.y, trace.load, trace.slip):
        assert np.all(np.isfinite(array))
    assert np.allclose(np.linalg.norm(trace.direction, axis=1), 1.0)
    # The boat carries the blade: net travel along the course is forward.
    assert trace.x[-1] > 0.0
    assert trace.speed == pytest.approx(float(np.mean(run.last_speed)))


def test_the_dynamic_oar_does_not_re_anchor_at_the_finish(dynamic_single):
    """The finding: the second reversal belongs to the schedule, not rowing.

    The blade starts anchored -- at the catch the oar is still -- becomes
    over-driven once, and is still over-driven when it reaches the finish.
    """
    sim, run = dynamic_single
    trace = dynamic_trace(sim, run, "dynamic")
    assert trace.slip[0] > 0.0, "anchored at the catch"
    assert len(trace.crossings) == 1, trace.crossings
    assert trace.slip[-1] < 0.0, "still over-driven at the finish"


def test_a_run_without_its_states_is_refused(dynamic_single):
    import dataclasses

    sim, run = dynamic_single
    bare = dataclasses.replace(run, last_states=None)
    with pytest.raises(ValueError, match="states"):
        dynamic_trace(sim, bare, "dynamic")


# ---------------------------------------------------------------------------
# the picture
# ---------------------------------------------------------------------------
def test_plot_writes_an_image(tmp_path, single, dynamic_single):
    sim, run = dynamic_single
    traces = [dynamic_trace(sim, run, "dynamic"),
              prescribed_trace(single, 4.0, "prescribed")]
    path = plot(traces, str(tmp_path / "blade.png"), title="test")
    assert os.path.getsize(path) > 10_000
    with open(path, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_a_trace_with_no_crossings_still_plots(tmp_path):
    flat = BladeTrace(label="flat", x=np.linspace(0, 1, 10),
                      y=np.zeros(10), load=np.ones(10),
                      direction=np.tile([1.0, 0.0], (10, 1)),
                      slip=np.ones(10), speed=3.0)
    path = plot([flat], str(tmp_path / "flat.png"))
    assert os.path.getsize(path) > 5_000


def test_the_prescribed_trace_is_drawn_on_the_oars_own_side():
    """A starboard oar is the port oar's mirror image, not a copy of it.

    It used to ignore the side, so beside a starboard seat's dynamic trace
    the prescribed "comparison" panel was mirrored -- a difference a reader
    would take for physics.
    """
    four = catalog.build("4+", rate=32.0)
    port = prescribed_trace(four, 4.0, "p", side=+1)
    starboard = prescribed_trace(four, 4.0, "s", side=-1)
    assert np.allclose(port.x, starboard.x)
    assert np.allclose(port.y, -starboard.y)
    assert np.allclose(port.direction[:, 0], starboard.direction[:, 0])
    assert np.allclose(port.direction[:, 1], -starboard.direction[:, 1])
    assert np.allclose(port.load, starboard.load)

    # And by default it is the boat's own first lock -- the one the dynamic
    # trace draws.
    lock_side = int(four.rig.seats[0].oarlocks[0].side)
    default = prescribed_trace(four, 4.0, "d")
    assert np.allclose(default.y, (port if lock_side > 0 else starboard).y)


def test_a_mixed_tier_figure_plots_and_says_which_is_which(tmp_path, single):
    """Tier 1 and tier 2 panels side by side: an image, and a footnote that
    names the provisional coefficients and what the plain panels cannot show."""
    from coxswain.sim.dynamic_oar import DynamicOarSimulator
    from coxswain.viz import bladepath

    torque = DynamicOarSimulator.peak_torque_for_power(single, 380.0)
    traces = []
    for law in ("slip", "liftdrag"):
        sim = DynamicOarSimulator(single, peak_torque=torque, blade_law=law)
        traces.append(dynamic_trace(sim, sim.run_strokes(2, surge_speed=4.0),
                                    law))
    path = plot(traces, str(tmp_path / "mixed.png"))
    assert os.path.getsize(path) > 10_000

    footnote = bladepath._footnote(traces)
    assert "tangential" in footnote and "provisional" in footnote
    assert "tier 1" in footnote
    assert "provisional" not in bladepath._footnote(traces[:1])
