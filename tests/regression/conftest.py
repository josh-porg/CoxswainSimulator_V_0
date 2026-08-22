"""Everything under ``tests/regression`` is slow by definition.

Regression tests here mean *validation against published measurements*:
they integrate the boat for tens of seconds, often several boats, and
compare against numbers from the literature.  That is inherently
expensive, and it is not the layer you want in an edit-run loop.

Marking by directory rather than by decorator keeps the two ideas from
drifting apart -- a new regression test is slow whether or not its author
remembered to say so.
"""

def pytest_collection_modifyitems(config, items):
    """Mark only the items that live under *this* directory.

    ``pytest_collection_modifyitems`` is handed the whole collected list,
    not just this package's, so an unfiltered version marks the entire
    suite slow -- which it did, deselecting all 1077 tests.
    """
    import pathlib

    import pytest

    here = pathlib.Path(__file__).parent.resolve()
    for item in items:
        try:
            path = pathlib.Path(str(item.fspath)).resolve()
        except Exception:
            continue
        if here in path.parents:
            item.add_marker(pytest.mark.slow)
