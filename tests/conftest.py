"""Shared fixtures and helpers for the Coxswain test suite.

Layout
------
``tests/unit``         one module per source module; pure, fast, no integration
``tests/integration``  the assembled dynamics: conservation, coupling, control
``tests/regression``   values pinned against the source papers and against a
                       stored golden trajectory

Boats are module-scoped fixtures because building one panels a hull mesh
and constructs 8 kinematic chains, which is wasted work to repeat.
"""

import numpy as np
import pytest

from coxswain.boats import catalog


@pytest.fixture(scope="session")
def eight():
    return catalog.eight(rate=32.0)


@pytest.fixture(scope="session")
def coxed_four():
    return catalog.coxed_four(rate=32.0)


@pytest.fixture(scope="session")
def single_scull():
    return catalog.single_scull(rate=30.0)


def assert_within(value, low, high, what=""):
    """Assert a value falls in a published range, with a readable message."""
    assert low <= value <= high, (
        f"{what or 'value'} = {value:.4g} is outside the expected "
        f"range [{low:.4g}, {high:.4g}]"
    )
