"""Multi-clock body, step 1 (review §12): can sourced population muscles row the athletes' strokes?

    .venv/Scripts/python.exe research/crew/muscle_capacity.py [--strokes 8] [--athletes cr06 br24]

For each athlete, the model's reference stroke is run on the research blade (rung 2's boat: hands on the handle,
the clock = time, the blade placed by burial). The script then reads off, along the last stroke:

* the **hip extensor** moment the upper body needs (rung2_trunk's top-down free body);
* the **knee extensor** moment, from below. The stretcher's horizontal force comes from the whole body's momentum,
  with a frictionless seat. The vertical force uses [BU15]'s women's direction, RF 0.81, held through the drive
  (knee_ankle_cr06's assumption). The force acts at [VI17]'s measured centre of pressure, 0.73 of the foot;
* the **elbow flexor** moment, from the handle force at the hands (horizontal, as in rung 2), with the forearms'
  weight and inertia;
* the **shoulder** moment, reported only, since shoulder strength is unsourced.

Each moment is divided by the sourced maximum voluntary torque at the joint's angle and speed. The hip and knee use
Anderson et al. (2007) for adults aged 18-25 of the athlete's sex, scaled by weight x height; the hip's passive term
is dropped (coxswain/crew/joint_muscles.py). The elbow uses Frey-Law et al. (2012). The result is the activation each
group needs.

The gate (review §12, step 1): every group's required activation stays at or below 1 through the drive, with no
fit. Reported with it:

* how far outside the strength data's tested angles the stroke goes (the catch is extrapolated);
* the order in which the groups load, against rowing's legs -> trunk -> arms.

Outputs: data/local/crew/muscle_capacity.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import rung2_clock as C2                                         # noqa: E402
import rung2_population as RP                                    # noqa: E402
import rung2_trunk as RT                                         # noqa: E402

from coxswain.crew import joint_muscles as J                     # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402

OUT = C2.OUT
G = 9.81
RF = 0.81                    # [BU15] horizontal / resultant foot force at maximum handle force
PHI = np.radians(42.0)       # [BU15] footplate incline
SOLE = 0.07                  # ankle above the sole, m (knee_ankle_cr06)
COP_FOOT = 0.73              # [VI17] centre of pressure as a fraction of foot length
HEEL_TO_ANKLE = 0.05
GROUP = {"cr06": "F18-25", "br24": "M18-25"}
SEX = {"cr06": "F", "br24": "M"}
TESTED = {"hip": (-0.5, 1.3), "knee": (0.0, 1.8), "elbow": (np.radians(15), np.radians(110))}


class CapSim(RT.TrunkClockSim):
    """The stroke under test, logging the handle force, the hull's acceleration, the clock and the hip moment at
    accepted times. Reference mode: rung 1 (the clock is time, the hands imposed). Free mode (``--rung2``): rung 2,
    the clock solved from the population muscles scaled to the athlete's power, with the trunk's internal model
    perfect so the trunk stays on its path (§206: that reproduces rung 2 exactly); the hands are an output (§216)."""

    def derivative(self, t, y):
        out = super().derivative(t, y)
        log = self.__dict__.setdefault("caplog", [])
        if not log or t > log[-1][0]:
            c = t - self._stroke_start
            if self.mode != "free":
                clock, trunk, base = (c, 1.0, 0.0), (0.0, 0.0, 0.0), y
            else:
                yy = np.asarray(y, float)
                base, dl, dld, tau, td = yy[:-4], float(yy[-4]), float(yy[-3]), float(yy[-2]), float(yy[-1])
                if self._rec is not None:
                    tau, td, _ = self._recovery_clock(c)
                clock, trunk = (tau, td, float(out[-1])), (dl, dld, float(out[-3]))
            self._clock, self._trunk = clock, trunk
            try:
                d = DynamicOarSimulator.derivative(self, t, base)
                F = self._handle_force(t, base)
                if self.mode == "free":
                    self.mlog.append((t, clock[0], self._equations(t, base, d, *clock)[1]))
            finally:
                self._clock, self._trunk = None, None
            log.append((t, clock[0], F, d[6], d[8], clock[1], clock[2]))
        return out


def make(boat):
    boat.power_scales = np.ones(boat.n_seats)
    from coxswain.sim.control import Coxswain
    sim = CapSim(boat, peak_torque=0.0, catch="sweep", crew="handle",
                 coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True, **RP.blade_options(boat))
    sim.setup_trunk("reference")
    return sim


def _angle(u, v):
    return np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))


def _cross(r, f):
    return r[0] * f[1] - r[1] * f[0]


UPPER_ARM_COM = 0.5772           # de Leva: upper-arm centre of mass from the shoulder, fraction of its length


def _vec(track, c):
    return np.array([float(getattr(q, "value", q)) for q in np.atleast_1d(track(float(c)))])


def arm3d(rower, c, side=1):
    """One arm in 3-D at clock ``c``: shoulder, elbow, hand, and the rows of its upper arm and forearm. A sculler's
    hands sit ~0.4 m out from the centreline at the catch, so the side view's elbow angle is wrong (it read the arms
    as bent ~100 deg at a straight-arm catch)."""
    sh, h = _vec(rower._shoulder_tracks[side], c), _vec(rower._hand_tracks[side], c)
    pos = np.asarray(rower.segment_state(float(c))[0], float)
    # the arm on the shoulder's side: a sculler's hands cross the centreline mid-drive, the shoulders never do
    # (keying on the hand's side picked the other arm there; found 2026-11-02, SOURCES §215)
    ua = 4 if np.sign(pos[4, 1]) == np.sign(sh[1]) else 5
    fa = 6 if np.sign(pos[6, 1]) == np.sign(sh[1]) else 7
    el = sh + (pos[ua] - sh) / UPPER_ARM_COM
    return sh, el, h, ua, fa


def elbow_flexion(rower, c):
    sh, el, h, _ua, _fa = arm3d(rower, c)
    return np.pi - _angle(sh - el, h - el)


def geometry(rower, P, n=1200):
    """Joint angles (flexion, rad) and their clock-rates on a periodic grid."""
    grid = np.linspace(0.0, P, n, endpoint=False)
    th = {"hip": [], "knee": [], "elbow": []}
    for c in grid:
        j = {k: v[[0, 2]] for k, v in rower.joint_positions(float(c)).items()}
        th["hip"].append(np.pi - _angle(j["knee"] - j["hip"], j["shoulder"] - j["hip"]))
        th["knee"].append(np.pi - _angle(j["hip"] - j["knee"], j["ankle"] - j["knee"]))
        th["elbow"].append(elbow_flexion(rower, c))
    dt = grid[1] - grid[0]
    rates = {k: (np.roll(np.unwrap(v), -1) - np.roll(np.unwrap(v), 1)) / (2 * dt) for k, v in th.items()}
    return grid, {k: np.array(v) for k, v in th.items()}, rates


RUNG2 = False
_TABLES = {}


def rung2_sim(name, boat, strokes):
    """Rung 2 on ``boat`` (§216): the population muscles, scaled to the athlete's measured power, drive the clock
    from a settled rung-1 run; the trunk's internal model is perfect, so the trunk stays on its path."""
    if "pop" not in _TABLES:
        _TABLES["pop"] = RP.population_tables(strokes)
    s_p, (s_pop, q_drive, ph, q) = _TABLES["pop"]
    m = C2.R.V.ATHLETES[name][0]()
    ref = make(boat)
    run_r = ref.run_strokes(strokes, surge_speed=4.3)
    P = float(boat.timing.period)
    D = float(boat.timing.drive_fraction) * P
    k = m["power"] / s_p["power"]

    def Q(tau):
        if tau <= D:
            return k * float(np.interp(float(RP.progress_of(boat, tau)[0]), s_pop, q_drive))
        return k * float(np.interp(np.mod(tau / P, 1.0), ph, q, period=1.0))
    boat.power_scales = np.ones(boat.n_seats)
    from coxswain.sim.control import Coxswain
    sim = CapSim(boat, peak_torque=0.0, catch="sweep", crew="handle",
                 coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True, **RP.blade_options(boat))
    sim.setup_trunk("free", Q=Q, tau_ff="internal", ff_delay=0.0, delay=0.0, spin=True, efference=True)
    sim.start_state = np.concatenate([run_r.last_states[:, -1], [0.0, 0.0, P, 1.0]])
    return sim, k


def athlete(name, strokes):
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    rower = boat.crew[0].rower
    mass, stature = float(rower.total_mass), float(rower.anthropometry.stature)
    if RUNG2:                                         # rung 2's power-matched stroke, the hands an output
        sim, k = rung2_sim(name, boat, strokes)
        print("%s: rung 2, population muscles x %.3f" % (name, k), flush=True)
    else:
        sim = make(boat)
    run = sim.run_strokes(strokes, surge_speed=4.3)
    P = float(boat.timing.period)
    start = float(run.last_time[0])
    log = [r for r in sim.caplog if r[0] >= start]
    hipM = {round(t, 9): M for t, c, M in sim.mlog if t >= start}
    grid, TH, W = geometry(rower, P)
    masses = np.asarray(rower.segment_masses, float)
    foot = 0.152 * stature if SEX[name] == "M" else 0.1484 * stature   # de Leva foot length (men / women)
    s_cop = COP_FOOT * foot - HEEL_TO_ANKLE
    ux, uz, nx, nz = -np.cos(PHI), np.sin(PHI), np.sin(PHI), np.cos(PHI)

    HE = J.anderson("HE", GROUP[name], mass, stature, passive=False)
    KE = J.anderson("KE", GROUP[name], mass, stature)
    EF = J.elbow("flexion", SEX[name])
    rows = []
    for rec in log:
        t, c, F, ax, az = rec[:5]
        td, tdd = (rec[5], rec[6]) if len(rec) > 5 else (1.0, 0.0)   # the clock's rate (1 on time)
        cc = float(c) % P
        pos, vel, acc = rower.segment_state(cc)[:3]
        acc = np.asarray(acc, float) * td * td + np.asarray(vel, float) * tdd   # the chain rule through the clock
        pos, acc = np.asarray(pos, float)[:, [0, 2]], acc[:, [0, 2]]
        a_abs = acc + np.array([ax, az])
        j = {k: v[[0, 2]] for k, v in rower.joint_positions(cc).items()}
        # stretcher: the whole body's horizontal momentum, the handle's reaction (-F) on the hands
        Fh = float(np.sum(masses * a_abs[:, 0])) + F
        Fv = Fh * np.tan(np.arccos(RF))
        p = j["ankle"] + np.array([s_cop * ux - SOLE * nx, s_cop * uz - SOLE * nz])
        knee = _cross(p - j["knee"], (Fh, Fv))
        for i in (10, 11):                                       # shank and foot: weight minus inertia
            r = pos[i] - j["knee"]
            knee += _cross(r, (0.0, -masses[i] * G)) - _cross(r, masses[i] * a_abs[i])
        # elbow flexors, in 3-D, one arm (the two are symmetric): the handle's reaction on the hand (half the total,
        # opposing the grip's motion along its path) and the forearm's weight and inertia
        sh3, el3, h3, ua, fa = arm3d(rower, cc)
        dc = 1e-3
        tang = _vec(rower._hand_tracks[1], (cc + dc) % P) - _vec(rower._hand_tracks[1], (cc - dc) % P)
        tang = tang / max(np.linalg.norm(tang), 1e-12)
        F_hand = -0.5 * F * tang
        pos3 = np.asarray(rower.segment_state(cc)[0], float)
        st3 = rower.segment_state(cc)
        acc3 = (np.asarray(st3[2], float) * td * td + np.asarray(st3[1], float) * tdd) + np.array([ax, 0.0, az])
        u3, v3 = sh3 - el3, h3 - el3
        axis = np.cross(v3, u3)
        axis = axis / max(np.linalg.norm(axis), 1e-12)          # rotating the forearm toward the upper arm
        Mext = np.cross(v3, F_hand) + np.cross(pos3[fa] - el3, masses[fa] * (np.array([0, 0, -G]) - acc3[fa]))
        elbow_flex = 2.0 * float(-np.dot(Mext, axis))             # both arms
        # the shoulder (reported only, strength unsourced): the side-view moment of both hands' reaction and the arms
        sh_ext = _cross(j["hand"] - j["shoulder"], (-F, 0.0))
        for i in (4, 5, 6, 7):
            r = pos[i] - j["shoulder"]
            sh_ext += _cross(r, (0.0, -masses[i] * G)) - _cross(r, masses[i] * a_abs[i])
        th = {k: float(np.interp(cc, grid, TH[k], period=P)) for k in TH}
        w = {k: td * float(np.interp(cc, grid, W[k], period=P)) for k in W}
        rows.append(dict(t=t - start, c=cc, F=F, Fh=Fh, hip=hipM.get(round(t, 9), np.nan), knee=knee, elbow=elbow_flex,
                         shoulder=sh_ext, th_hip=th["hip"], th_knee=th["knee"], th_elbow=th["elbow"],
                         w_hip=-w["hip"], w_knee=-w["knee"], w_elbow=w["elbow"]))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    # the hip moment's extensor sign: the drive's dominant sign (rung2_trunk's M is the moment the trunk needs)
    drive = R["F"] > 0.05 * R["F"].max()
    sgn = np.sign(np.nanmedian(R["hip"][drive]))
    R["hip_ext"] = sgn * R["hip"]
    act = dict(hip=HE.activation_needed(R["hip_ext"], R["th_hip"], R["w_hip"]),
               knee=KE.activation_needed(R["knee"], R["th_knee"], R["w_knee"]),
               elbow=EF.activation_needed(np.maximum(R["elbow"], 0.0), R["th_elbow"], R["w_elbow"]))
    out = dict(mass=mass, stature=stature, strength_group=GROUP[name], hip_moment_sign=float(sgn),
               measured=dict(speed=m["speed"], ivv=m["ivv"], power=m["power"]), model=C2.summary(run))
    td = R["t"][drive]
    for k, a in act.items():
        ad = a[drive]
        mom = {"hip": R["hip_ext"], "knee": R["knee"], "elbow": R["elbow"]}[k][drive]
        th = R["th_" + k][drive]
        lo, hi = TESTED[k]
        pk = int(np.nanargmax(ad))
        onset = td[np.argmax(ad >= 0.3 * np.nanmax(ad))]
        out[k] = dict(peak_activation=float(ad[pk]), t_peak=float(td[pk]), onset_30pct=float(onset),
                      mean_activation=float(np.nanmean(np.clip(ad, 0, None))), infeasible_fraction=float(np.mean(ad > 1.0)),
                      peak_moment=float(np.nanmax(mom)), angle_range_deg=[float(np.degrees(th.min())), float(np.degrees(th.max()))],
                      outside_tested_fraction=float(np.mean((th < lo) | (th > hi))),
                      peak_speed=float(np.nanmax(R["w_" + k][drive])))
        print("%s %-5s: peak activation %.2f at %.2f s (onset %.2f s), mean %.2f, infeasible %.0f%% of the drive; peak moment"
              " %.0f N.m; angle %.0f-%.0f deg (%.0f%% outside the tested range); peak concentric speed %.1f rad/s" % (
                  name, k, out[k]["peak_activation"], out[k]["t_peak"], out[k]["onset_30pct"], out[k]["mean_activation"],
                  100 * out[k]["infeasible_fraction"], out[k]["peak_moment"], *out[k]["angle_range_deg"],
                  100 * out[k]["outside_tested_fraction"], out[k]["peak_speed"]), flush=True)
    out["shoulder"] = dict(peak_moment=float(np.nanmax(np.abs(R["shoulder"][drive]))))
    out["stretcher_peak"] = float(R["Fh"].max())
    out["series"] = {k: R[k][::4].round(4).tolist() for k in ("t", "F", "hip_ext", "knee", "elbow", "th_hip", "th_knee",
                                                              "th_elbow", "w_hip", "w_knee", "w_elbow")}
    out["series"].update({"a_" + k: np.round(v[::4], 4).tolist() for k, v in act.items()})
    print("%s: model %.3f m/s at %.0f W (measured %.0f W); stretcher peak %.0f N; shoulder moment peak %.0f N.m (unsourced"
          " strength)" % (name, out["model"]["speed"], out["model"]["power"], m["power"], out["stretcher_peak"],
                          out["shoulder"]["peak_moment"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=8)
    ap.add_argument("--athletes", nargs="+", default=["cr06", "br24"])
    ap.add_argument("--catch-posture", default="as-built", choices=["as-built", "straight-arms"])
    ap.add_argument("--clocks", nargs="*", default=[], help="extra segment clocks (coxswain.crew.body_options.CLOCKS)")
    ap.add_argument("--hand-terms", nargs="*", default=[], help="hand-chain reach terms, e.g. stretch")
    ap.add_argument("--rung2", action="store_true",
                    help="check rung 2's power-matched stroke (hands an output) instead of rung 1's (§216)")
    a = ap.parse_args()
    global RUNG2
    RUNG2 = bool(a.rung2)
    C2.R.CATCH_POSTURE = a.catch_posture
    C2.R.BODY_CLOCKS = tuple(a.clocks)
    C2.R.HAND_TERMS = tuple(a.hand_terms)
    RP.BLADE = "crossflow_normal"
    C2.TAKEOVER_DEG = C2.BIOROW_CATCH_TO_BURIED_DEG
    out = {n: athlete(n, a.strokes) for n in a.athletes}
    json.dump(out, open(os.path.join(OUT, "muscle_capacity%s.json" % (
        ("" if a.catch_posture == "as-built" else "_straightarms") + "".join("_" + c for c in a.clocks)
        + "".join("_" + t for t in a.hand_terms) + ("_rung2" if a.rung2 else ""))), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
