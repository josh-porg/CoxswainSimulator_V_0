"""Kleshnev's population reference curves (RBN 2023/12 Fig. 1), as extracted by extract_rbn2023.py (SOURCES §237).

Per 10 % of the drive's time, >50 000 samples: handle force and handle velocity as % of their peaks, seat and
upper-body (trunk + arms) velocity as % of the peak handle velocity, the vertical oar angle, and the drive length (the
handle velocity's integral). The numbers live with the newsletter in data/local (shared by its author).
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(ROOT, "data", "local", "literature", "biorow_newsletters", "rbn2023_fig1.csv")
KEYS = ("time", "force", "vertical", "handle_v", "seat_v", "upper_v", "length")
#: the figure is vector, so the reading error is the plot's own rounding: about half a point of peak
SPEED_ERROR = 0.005


def reference():
    """The curves as a dict of arrays (all in %), keys ``KEYS``."""
    d = np.loadtxt(CSV, delimiter=",", comments="#", skiprows=6)
    return {k: d[:, i] for i, k in enumerate(KEYS)}


def drive_profile():
    """``(length_fraction, handle_speed)`` for coxswain.crew.drive_law.PopulationDriveSweep(profile=...): the
    population's handle velocity (fraction of its peak, zero at both turning points) against drive length."""
    ref = reference()
    v = np.clip(ref["handle_v"] / 100.0, 0.0, None)
    v[0] = v[-1] = 0.0
    return ref["length"] / 100.0, v


def drive_sweep(catch_angle, finish_angle):
    """The oar sweep whose drive follows the population's handle velocity (the recovery is the parent's)."""
    from coxswain.crew.drive_law import PopulationDriveSweep
    s, v = drive_profile()
    # the first and last samples sit inside PopulationDriveSweep's default turning edge; here the turning spans are
    # the first and last 10 % of the drive's time, which the data resolve, so the edge is their path (3.2 %)
    return PopulationDriveSweep(catch_angle=catch_angle, finish_angle=finish_angle, profile=(s, v),
                                speed_error=SPEED_ERROR, turning_edge=float(s[1]) - 1e-9)
