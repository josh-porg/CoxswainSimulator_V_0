r"""The machine probe and the diagnostics file, without a GPU.

Both exist because "it was sluggish on my laptop" cannot be acted on.
The probe has to classify a renderer string correctly and notice a
dedicated GPU that is sitting idle; the log has to be cheap per frame,
summarise rather than spam, and record a stall with its context.
"""

from __future__ import annotations

import io
import os

import pytest

from coxswain.viz.hardware import Probe
from coxswain.viz.telemetry import STALL_MS, Telemetry, build_version


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------
def test_an_integrated_renderer_is_recognised():
    probe = Probe(renderer="Intel(R) UHD Graphics 620", cpu_count=4)
    assert probe.integrated
    assert not probe.software
    assert probe.recommended_tier() == "ultra"


def test_a_dedicated_renderer_is_not_integrated():
    probe = Probe(renderer="NVIDIA GeForce RTX 3060/PCIe/SSE2", cpu_count=8)
    assert not probe.integrated
    assert probe.recommended_tier() == "high"
    older = Probe(renderer="NVIDIA GeForce GTX 1050/PCIe/SSE2", cpu_count=4)
    assert older.recommended_tier() == "standard"


def test_software_rendering_is_the_lowest_tier_and_says_so():
    probe = Probe(renderer="llvmpipe (LLVM 15.0.7, 256 bits)", cpu_count=16)
    assert probe.software and probe.integrated
    assert probe.recommended_tier() == "ultra"
    assert any("software" in line for line in probe.lines())


def test_a_dedicated_gpu_sitting_idle_is_noticed_and_explained():
    """The hybrid-laptop case: the iGPU is drawing, the dGPU is not."""
    probe = Probe(renderer="Intel(R) Iris(R) Xe Graphics",
                  adapters=["Intel(R) Iris(R) Xe Graphics",
                            "NVIDIA GeForce RTX 3050 Laptop GPU"],
                  cpu_count=8)
    assert probe.dedicated_idle == "NVIDIA GeForce RTX 3050 Laptop GPU"
    lines = probe.lines()
    assert any("NOTE" in line and "integrated" in line for line in lines)
    assert any("Graphics" in line or "PRIME" in line or "switching" in line
               for line in lines), "must tell the user how to fix it"


def test_no_false_alarm_when_the_dedicated_gpu_is_drawing():
    probe = Probe(renderer="NVIDIA GeForce RTX 3050 Laptop GPU/PCIe/SSE2",
                  adapters=["Intel(R) Iris(R) Xe Graphics",
                            "NVIDIA GeForce RTX 3050 Laptop GPU"])
    assert probe.dedicated_idle is None
    assert not any("NOTE" in line for line in probe.lines())


def test_apple_silicon_counts_as_integrated_but_capable():
    probe = Probe(renderer="Apple M2", cpu_count=8)
    assert probe.integrated
    assert probe.recommended_tier() == "minimal"


# ---------------------------------------------------------------------------
# the log
# ---------------------------------------------------------------------------
@pytest.fixture()
def log(tmp_path):
    now = [0.0]

    def clock():
        return now[0]

    telemetry = Telemetry(directory=str(tmp_path), clock=clock)
    telemetry._now = now
    yield telemetry
    telemetry.close()


def _text(telemetry):
    with io.open(telemetry.path, encoding="utf-8") as handle:
        return handle.read()


def test_the_log_opens_in_the_given_folder_with_the_build_header(log):
    assert log.enabled and os.path.exists(log.path)
    text = _text(log)
    assert "[" in text and "build" in text
    assert "version: %s" % build_version() in text


def test_frames_are_summarised_not_listed(log):
    for _ in range(600):
        log._now[0] += 0.016
        log.frame(16.0, physics_ms=5.0, draw_ms=3.0, present_ms=8.0)
    text = _text(log)
    assert text.count("frames ") <= 2, "a summary every ten seconds, not 600 lines"
    assert "STALL" not in text
    log.close()
    text = _text(log)
    assert "fps" in text and "closed after 600 frames" in text


def test_a_stall_is_logged_with_its_context(log):
    log.frame(16.0)
    log.frame(STALL_MS + 250.0, physics_ms=300.0, draw_ms=40.0,
              context=lambda: "t=12.3 speed=4.4 quality=standard")
    text = _text(log)
    assert "STALL 350 ms" in text
    assert "physics 300" in text
    assert "quality=standard" in text


def test_a_broken_context_does_not_take_the_game_down(log):
    def bad():
        raise RuntimeError("no")

    log.frame(STALL_MS + 1.0, context=bad)
    assert "context failed" in _text(log)


def test_exceptions_are_written_with_a_traceback(log):
    try:
        raise ValueError("the thing that happened")
    except ValueError as error:
        log.exception(error)
    text = _text(log)
    assert "EXCEPTION ValueError: the thing that happened" in text
    assert "Traceback" in text


def test_disabled_telemetry_costs_nothing_and_writes_nothing(tmp_path):
    off = Telemetry(directory=str(tmp_path), enabled=False)
    for _ in range(100):
        off.frame(16.0)
    off.note("x")
    off.close()
    assert off.path is None
    assert os.listdir(str(tmp_path)) == []


def test_old_logs_are_pruned(tmp_path):
    for k in range(15):
        with io.open(os.path.join(str(tmp_path), "coxswain-2000010%02d-000000.log" % k),
                     "w") as handle:
            handle.write("old\n")
    fresh = Telemetry(directory=str(tmp_path))
    fresh.close()
    logs = [n for n in os.listdir(str(tmp_path)) if n.startswith("coxswain-")]
    assert len(logs) <= 11


def test_a_tier_can_be_chosen_before_there_is_a_context():
    """The tier gates the world build, which precedes any GL context."""
    assert Probe(adapters=["NVIDIA GeForce RTX 3060"]).recommended_tier() == "standard"
    assert Probe(adapters=["Intel(R) UHD Graphics 620"], cpu_count=4).recommended_tier() == "ultra"
    assert Probe(adapters=["Intel(R) Iris(R) Xe Graphics"], cpu_count=8).recommended_tier() == "minimal"
    assert Probe().recommended_tier() == "standard", "no information: the middle"
