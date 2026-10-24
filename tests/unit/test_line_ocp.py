"""The smooth reference frame used by the Radau line optimiser and the line displays."""
import numpy as np
import pytest

from coxswain.raceday import card as C
from coxswain.raceday import line_ocp as O


@pytest.fixture(scope="module")
def frame():
    return O.SmoothFrame(C.CourseModel.build().course)


def test_round_trip(frame):
    s = np.linspace(50.0, frame.length - 50.0, 200)
    n = np.linspace(-40.0, 40.0, 200)
    p = frame.to_plane(s, n)
    s2, n2 = frame.from_plane(p[:, 0], p[:, 1])
    assert np.max(np.abs(s2 - s)) < 1.5
    assert np.max(np.abs(n2 - n)) < 0.5


def test_close_to_the_centreline_and_smoother(frame):
    course = frame.course
    st = np.linspace(0.0, float(course.length), 600)
    _, n = frame.from_course(st, np.zeros_like(st))
    assert np.sqrt(np.mean(n ** 2)) < 4.0          # follows the river, not a chord across its bends
    p = course.offset_position(np.arange(0.0, float(course.length), 2.0), np.zeros(int(course.length // 2) + 1))
    th = np.unwrap(np.arctan2(*np.gradient(p, axis=0)[:, ::-1].T))
    raw_curvature = np.gradient(th, 2.0)
    assert np.std(frame.curvature) < 0.5 * np.std(raw_curvature)   # the bank wiggles are gone


def test_sign_matches_the_course(frame):
    q = frame.course.offset_position(np.array([2000.0]), np.array([20.0]))[0]
    assert frame.from_plane(q[0], q[1])[1][0] > 0
