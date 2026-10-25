"""[CR06]'s woman, one measured stroke: the shared inputs of step 4.3's rungs.

Every step-4.3 script (stretcher, hip from above, trunk forward dynamics, knee and ankle) reads her stroke
through this module, so the series, the smoothing and the body are defined once.

Source: Cabrera, Ruina & Kleshnev (2006), Human Movement Science 25(2):192-220, Fig. 3, one women's single,
T = 1.94 s; digitised into data/literature/cr06_fig3_measured.csv by research/cr06/extract_fig3.py.
Channels: boat velocity, handle force (both handles together, see stretcher_force), oar angle, leg
displacement x_B/F (seat relative to the feet) and back displacement x_S/B (shoulder relative to the seat).

Body: de Leva (1996) women's table (coxswain.crew.anthropometry.DE_LEVA_FEMALE) at 75 kg and 1.758 m, the
stature at which the package rower's leg travel equals her measured 0.581 m (female_segments_cr06.py;
TRACKING 2026-09-14).
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.interpolate import make_smoothing_spline

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CODE, "..", ".."))
OUT = os.path.join(ROOT, "data", "local", "cr06_study")       # outputs (gitignored)
sys.path.insert(0, ROOT)
from coxswain.crew.anthropometry import DE_LEVA_FEMALE         # noqa: E402

T = 1.94                    # her stroke period, s
M = 75.0                    # rower mass, kg
STATURE = 1.758             # m (see the docstring)
G = 9.81
MEAS = os.path.join(ROOT, "data", "literature", "cr06_fig3_measured.csv")
# smoothing weights of the periodic splines: rung 1 used 3e-6 to 3e-5; the middle is the default and the
# band is reported wherever a result sits on a second derivative (the catch)
LAM_DEFAULT = 1e-5
LAM_BAND = (3e-6, 1e-5, 3e-5)
N_GRID = 777                # samples per stroke on the common grid


def series(name):
    """(t, value) of one measured channel, t folded into [0, T), the duplicate sample at the seam dropped."""
    rows = [r for r in csv.DictReader(l for l in open(MEAS) if not l.startswith("#")) if r["series"] == name]
    d = np.array(sorted((float(r["t_s"]) % T, float(r["value"])) for r in rows))
    keep = np.concatenate([[True], np.diff(d[:, 0]) > 1e-9])
    return d[keep, 0], d[keep, 1]


def periodic_spline(t, y, lam):
    """A smoothing spline on three copies of the stroke, so it is periodic at the seam."""
    tt = np.concatenate([t - T, t, t + T]); yy = np.concatenate([y, y, y])
    return make_smoothing_spline(tt, yy, lam=lam)


class Stroke:
    """Her stroke on a common grid, with one choice of smoothing."""

    def __init__(self, lam_v=LAM_DEFAULT, lam_x=LAM_DEFAULT, n=N_GRID):
        self.lam_v, self.lam_x = lam_v, lam_x
        self.t = np.linspace(0.0, T, n, endpoint=False)
        self.dt = self.t[1] - self.t[0]
        tv, v = series("boat_velocity_mps"); self.tF, self.F_raw = series("handle_force_N"); ta, ang = series("oar_angle_deg")
        tl, leg = series("leg_disp_m"); tb, back = series("back_disp_m")
        vS = periodic_spline(tv, v, lam_v)
        self.v_boat = vS(self.t)
        self.a_boat = vS.derivative(1)(self.t)
        self.legS, self.backS = periodic_spline(tl, leg, lam_x), periodic_spline(tb, back, lam_x)
        self.leg = self.legS(self.t); self.back = self.backS(self.t)
        self.F_handle = np.interp(self.t, self.tF, self.F_raw, period=T)        # both handles, N
        self.oar_deg = np.interp(self.t, ta, ang, period=T)
        self.k_landmark = int(np.argmin(self.a_boat))                            # [LE26]: minimum boat acceleration
        drive = self.F_handle > 30.0
        self.i_mhf = int(np.argmax(np.where(drive, self.F_handle, -1)))          # maximum handle force

    def acc(self, y):
        """Second derivative of a periodic series on the grid."""
        return (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / self.dt ** 2

    def vel(self, y):
        return (np.roll(y, -1) - np.roll(y, 1)) / (2 * self.dt)

    def stretcher_force(self, r_back=0.4):
        """Rung 1: the stretcher's horizontal force on the feet from the body's momentum balance along the boat,
        frictionless seat: F_h = m (a_boat + a_rel) + the handle pull along the boat, with [CR06] eq. 5's centre
        of mass (all 75 kg on the seat plus r_back x the back channel). The handle channel is both handles
        together: that convention reproduces rung 1's recorded 675-726 N peak (twice it does not)."""
        a_rel = self.legS.derivative(2)(self.t) + r_back * self.backS.derivative(2)(self.t)
        return M * (self.a_boat + a_rel) + self.F_handle * np.cos(np.radians(self.oar_deg))


def her_rower():
    """The package rower on the women's table at her stature (as female_segments_cr06.py builds her)."""
    from coxswain.boats import catalog
    from coxswain.boats.boat import Boat
    base = catalog.single_scull(rate=60.0 / T, rower_mass=M, rower_stature=STATURE)
    b = Boat(name="CR06 women's single", offsets=base.offsets, rig=base.rig, hull_mass=19.7, hull_inertia=base.hull_inertia,
             timing=base.timing, appendages=base.appendages, water=base.water,
             default_anthropometry=catalog.RowerAnthropometry(mass=M, stature=STATURE, sex="female"),
             force_profile=base.force_profile, oar_sweep=base.oar_sweep)
    return b.crew[0].rower, b


def segment(name):
    """(mass of BOTH sides for paired segments, length, centre-of-mass fraction from the proximal end)."""
    s = DE_LEVA_FEMALE[name]
    return (2 if s.paired else 1) * M * s.mass_fraction, s.length_fraction * STATURE, s.com_fraction
