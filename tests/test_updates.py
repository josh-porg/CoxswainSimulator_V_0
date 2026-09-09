r"""The update check: right about versions, silent about everything else.

No test here touches the network.  ``fetch`` is injected, and the one
thing the real fetch is held to is that it cannot raise.
"""

from __future__ import annotations

import time

from coxswain.viz.updates import (PAGE, Update, UpdateCheck, check, newer,
                                  parse_version)


def test_tags_compare_as_numbers_not_strings():
    """The whole reason this exists: v0.10 is newer than v0.9."""
    assert parse_version("v0.10") == (0, 10, 0)
    assert parse_version("v0.9") == (0, 9, 0)
    assert newer("v0.9", "v0.10")
    assert not newer("v0.10", "v0.9")
    assert not newer("v0.10", "v0.10")
    assert newer("v0.10", "v1.0")
    assert newer("0.10", "v0.10.1")


def test_a_source_checkout_is_never_told_to_update():
    assert parse_version("source") is None
    assert parse_version("local") is None
    assert not newer("source", "v9.9")
    assert check("source", fetch=lambda: "v9.9") is None
    assert check("local", fetch=lambda: "v9.9") is None


def test_check_reports_the_tag_and_the_releases_page():
    got = check("v0.10", fetch=lambda: "v0.11")
    assert got == Update(tag="v0.11", url=PAGE)
    assert "v0.11" in got.line() and PAGE in got.line()
    assert check("v0.10", fetch=lambda: "v0.10") is None
    assert check("v0.10", fetch=lambda: "v0.9") is None


def test_every_failure_is_none_not_an_exception():
    def boom():
        raise RuntimeError("no network")

    assert check("v0.10", fetch=boom) is None
    assert check("v0.10", fetch=lambda: None) is None
    assert check("v0.10", fetch=lambda: "not-a-tag") is None
    assert check("v0.10", fetch=lambda: "") is None


def test_the_background_check_never_blocks_the_start():
    """A fetch that hangs must not hold up the caller."""
    def slow():
        time.sleep(5.0)
        return "v9.9"

    started = time.perf_counter()
    chk = UpdateCheck("v0.10", enabled=True, fetch=slow).start()
    assert time.perf_counter() - started < 0.5
    assert chk.result is None            # not answered yet: draw nothing


def test_the_background_check_answers_when_it_can():
    chk = UpdateCheck("v0.10", enabled=True, fetch=lambda: "v0.12").start()
    got = chk.wait(2.0)
    assert got is not None and got.tag == "v0.12"


def test_switched_off_means_no_fetch_at_all():
    calls = []

    def spy():
        calls.append(1)
        return "v9.9"

    chk = UpdateCheck("v0.10", enabled=False, fetch=spy).start()
    assert chk.wait(1.0) is None
    assert calls == []


def test_the_real_fetch_cannot_raise_even_with_no_network():
    """Pointed at a port nothing listens on, with a short timeout."""
    from coxswain.viz import updates

    saved = updates.API
    try:
        updates.API = "https://127.0.0.1:9/nothing"
        assert updates.fetch_latest_tag(timeout=0.5) is None
    finally:
        updates.API = saved
