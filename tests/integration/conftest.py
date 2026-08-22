"""Everything under ``tests/integration`` is slow by definition.

Integration here means the assembled 6-DOF model actually integrated
forward, as opposed to the unit layer which exercises one piece at a time
and runs in milliseconds.  See ``tests/regression/conftest.py``.
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
