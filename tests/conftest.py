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


@pytest.fixture(scope="session")
def simulate():
    """Memoised full-model run, keyed on its parameters.

    Integrating 20 s of a coxed eight is a few seconds of work, and the
    validation suite checks many independent properties of the *same*
    trajectory -- heave amplitude, pitch amplitude, mean speed, where in
    the stroke the boat runs fastest.  Without memoisation each assertion
    pays for its own integration and the suite takes over an hour.

    Usage::

        result = simulate("8+", rate=32.0, duration=16.0, dt=0.006)

    Results are shared, so treat them as read-only.
    """
    from coxswain.sim.simulator import RowingSimulator

    cache = {}

    def run(name, rate, duration, surge_speed=4.5, dt=0.006, **kwargs):
        key = (name, rate, duration, surge_speed, dt,
               tuple(sorted(kwargs.items())))
        if key not in cache:
            boat = catalog.build(name, rate=rate)
            cache[key] = RowingSimulator(boat).run(
                duration=duration, surge_speed=surge_speed, dt=dt, **kwargs)
        return cache[key]

    return run


def assert_within(value, low, high, what=""):
    """Assert a value falls in a published range, with a readable message."""
    assert low <= value <= high, (
        f"{what or 'value'} = {value:.4g} is outside the expected "
        f"range [{low:.4g}, {high:.4g}]"
    )
