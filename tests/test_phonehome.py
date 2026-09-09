r"""Performance reports: what leaves the machine, and when.

Three promises.  Nothing identifying is in a report -- checked on a
report built from the real objects, not a fixture.  Nothing is sent
unless the person switched it on.  And when it is sent, a plain HTTP
endpoint receives exactly the JSON the program built.
"""

from __future__ import annotations

import http.server
import json
import os
import threading

import pytest

from coxswain.viz import phonehome


class _FakeTelemetry:
    def __init__(self):
        self.notes = []

    def snapshot(self):
        return {"frames": 1200, "p50_ms": 16.4, "p95_ms": 22.0,
                "worst_ms": 141.0, "stalls": 2, "dropped_s": 0.05,
                "physics_ms": 3.1, "draw_ms": 12.0}

    def note(self, text):
        self.notes.append(text)


class _Probe:
    renderer = "Intel(R) UHD Graphics"
    adapters = ["Intel(R) UHD Graphics"]
    cpu_count = 12
    ram_gb = 15.7
    integrated = True
    software = False


def _report(**extra):
    return phonehome.build_report(_FakeTelemetry(), _Probe(),
                                  settings={"quality": "minimal",
                                            "physics": 60.0},
                                  tier="minimal", extra=extra)


def test_a_report_is_numbers_and_product_names_only():
    report = _report()
    phonehome.scrub(report)                    # must not raise
    text = json.dumps(report)
    assert os.path.expanduser("~") not in text
    assert "\\" not in text and "/Users/" not in text
    assert report["hardware"]["renderer"] == "Intel(R) UHD Graphics"
    assert report["frames"]["p50_ms"] == 16.4
    assert report["tier"] == "minimal"


@pytest.mark.parametrize("bad", [
    {"note": "C:\\Users\\somebody\\Desktop"},
    {"note": "/home/somebody/coxswain"},
    {"contact": "somebody@example.com"},
])
def test_anything_that_could_name_the_person_is_refused(bad):
    with pytest.raises(ValueError):
        phonehome.scrub(_report(**bad))


def test_the_user_name_itself_is_refused():
    import getpass
    name = getpass.getuser()
    if len(name) < 3:
        pytest.skip("user name too short to test")
    with pytest.raises(ValueError):
        phonehome.scrub(_report(note="run by %s" % name))


def test_nothing_is_sent_when_reports_are_off():
    log = []
    line = phonehome.send(_report(), "http://127.0.0.1:9/never", enabled=False,
                          log=log.append)
    assert line.startswith("not sent")
    assert log and "reports are off" in log[0]


def test_with_no_url_the_report_is_written_beside_the_logs(tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(phonehome, "log_directory", lambda: str(tmp_path))
    line = phonehome.send(_report(), None, enabled=True)
    assert line.startswith("no report URL; written to")
    files = list(tmp_path.glob("report-*.json"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as handle:
        assert json.load(handle)["tier"] == "minimal"


def test_a_plain_endpoint_receives_the_json():
    received = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            received["body"] = json.loads(self.rfile.read(length))
            received["type"] = self.headers.get("Content-Type")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:%d/report" % server.server_port
        line = phonehome.send(_report(), url, enabled=True)
    finally:
        server.shutdown()
    assert line == "sent (200)", line
    assert received["type"] == "application/json"
    assert received["body"]["hardware"]["cpu_count"] == 12
    assert received["body"]["frames"]["stalls"] == 2


def test_a_dead_endpoint_fails_quietly_and_quickly():
    import time
    t0 = time.perf_counter()
    line = phonehome.send(_report(), "http://127.0.0.1:9/dead", enabled=True)
    assert line.startswith("failed") or line.startswith("refused"), line
    assert time.perf_counter() - t0 < phonehome.TIMEOUT + 2.0


def test_the_url_comes_from_flag_then_environment_then_file(tmp_path,
                                                            monkeypatch):
    monkeypatch.delenv(phonehome.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert phonehome.report_url(None) is None
    (tmp_path / phonehome.URL_FILE).write_text("https://file.example/x\n")
    assert phonehome.report_url(None) == "https://file.example/x"
    monkeypatch.setenv(phonehome.ENV_VAR, "https://env.example/y")
    assert phonehome.report_url(None) == "https://env.example/y"
    assert phonehome.report_url("https://flag.example/z") == "https://flag.example/z"
