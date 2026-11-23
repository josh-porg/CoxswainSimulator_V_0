"""Multi-clock body, step 4 on the trunk: the trunk's feedforward from its muscles' EMG modules (§262).

Computed muscle control (Thelen & Anderson 2006) in one joint: a forward simulation whose muscle drive is a
feedforward plus a feedback that tracks the intended kinematics. Rung 2 with the trunk free (rung2_trunk.py) already
has the structure; its feedforward was the inverse-dynamics hip moment of the reference run (the trunk asked to hold
the posture it was given, rung 5). Here that moment is replaced by the trunk's own muscles, excited on the measured
on-water EMG (Pitto et al. 2024), the antagonist pair included:

    T_ff(t) = sigma [ a_E m_E(phi(t-d)) T_HE(theta_h, -w_h tau') - a_F m_F(phi(t-d)) T_HF(theta_h, w_h tau') ]

* m_E, m_F: the trunk-extensor and trunk-flexor modules (coxswain.crew.excitation); d = 0.075 s (placeholder);
* T_HE, T_HF: Anderson et al. 2007's hip extensors and flexors at the planned hip angle theta_h(tau) and its speed;
* a_E, a_F: **EMG-assisted calibration** (Lloyd & Besier 2003; Sartori et al. 2012): non-negative least squares of
  the module torques on the reference run's own hip moment, so the EMG sets each muscle's timing and the fit only
  their levels (the antagonists' included); sigma the hip moment's sign convention, chosen by the better fit.

The feedback is rung 2's trunk PD (rung 5's design on the coupled plant), at no delay and at the trunk's 100 ms. The
clock's own force is the population's (as rung2_trunk's ``athletes`` check), so the trunk is the only change.
Gate (MULTI_CLOCK_BODY_PLAN step 4, trunk): wander < 5 deg; IVV on the shared clock.

    python research/crew/rung2_trunk_modules.py [--strokes 10] [--delays 0 0.1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "biorow"), os.path.join(ROOT, "research", "blade")]
import rung2_trunk as RT                                         # noqa: E402
import rung2_modules as RM                                       # noqa: E402

from coxswain.crew import joint_muscles as J                     # noqa: E402
from coxswain.crew.excitation import Modules                     # noqa: E402

C2, RP = RT.C2, RT.RP
DELAY = 0.075


def population_Q(strokes, name, boat, m):
    """The population's muscle force on the athlete's clock, as rung2_trunk.athletes builds it."""
    from coxswain.crew.stroke import OnWaterTiming
    import k05_body
    import k05_population_test as KP
    V = C2.R.V
    pop_boat = V.cr06_boat(dict(leg_travel=None), timing=OnWaterTiming(KP.K05_RATE), stature=KP.K05_STATURE,
                           drive_law=True, mass=KP.K05_MASS)
    body, _k = k05_body.body_field(pop_boat, leg_travel=0.51)
    pop = RP.make(pop_boat, "reference", body=body)
    run_p = pop.run_strokes(strokes, surge_speed=4.3)
    s_p = C2.summary(run_p)
    ph, q = RP.q_table(pop, run_p)
    P_pop = float(pop_boat.timing.period)
    on = ph * P_pop <= float(pop_boat.timing.drive_fraction) * P_pop
    s_pop = np.maximum.accumulate(RP.progress_of(pop_boat, ph[on] * P_pop))
    q_drive = q[on]
    keep = np.concatenate([[True], np.diff(s_pop) > 1e-9])
    s_pop, q_drive = s_pop[keep], q_drive[keep]
    P = float(boat.timing.period)
    D = float(boat.timing.drive_fraction) * P
    k = m["power"] / s_p["power"]

    def Q(tau):
        if tau <= D:
            return k * float(np.interp(float(RP.progress_of(boat, tau)[0]), s_pop, q_drive))
        return k * float(np.interp(np.mod(tau / P, 1.0), ph, q, period=1.0))
    return Q


#: Silder, Whittington, Heiderscheit & Thelen (2007), J Biomech 40:2628, Table 4: mean passive-moment parameters of 20
#: young adults (gains per rad, offsets in degrees); their eq. A.5 gives the hip's passive moment, flexor-positive:
#:   M_h = e^{-b_RFh (h - (b_RFk/b_RFh) k - a_RF)} + e^{-b_HF (h - a_HF)} - e^{b_HAMh (h - (b_HAMk/b_HAMh) k - a_HAM)}
#:         - e^{b_HE (h - a_HE)}            (h hip flexion, k knee flexion, rad; N m, one leg)
#: Measured to 100 deg of hip flexion (85 deg with the knee flexed to 135); beyond, an extrapolation of their fit.
SILDER2007 = dict(RF=(3.1, 1.9, 24.4), HAM=(5.1, 3.9, 30.8), HF=(5.1, 19.5), HE=(2.0, 27.3))
#: include the passive hip moment in the trunk's muscle model (§267); False: active modules only (§262-266)
PASSIVE = False


def silder_passive_hip_extension(h, k, sides=2):
    """The passive hip moment, EXTENSOR-positive, both legs (Silder et al. 2007 eq. A.5 with Table 4's means)."""
    bRFh, bRFk, aRF = SILDER2007["RF"]
    bHAMh, bHAMk, aHAM = SILDER2007["HAM"]
    bHF, aHF = SILDER2007["HF"]
    bHE, aHE = SILDER2007["HE"]
    r = np.radians
    flexor = (np.exp(-bRFh * (h - (bRFk / bRFh) * k - r(aRF))) + np.exp(-bHF * (h - r(aHF)))
              - np.exp(bHAMh * (h - (bHAMk / bHAMh) * k - r(aHAM))) - np.exp(bHE * (h - r(aHE))))
    return -sides * flexor


class TrunkMuscles:
    """The hip moment from the trunk extensor and flexor modules, calibrated on a reference hip moment; with
    ``PASSIVE`` the passive hip moment (Silder et al. 2007) acts too and the modules are calibrated on the rest."""

    def __init__(self, name, boat, M_ref, delay=DELAY):
        self.P, grid, th, w = RM.tables(boat)
        self.grid, self.th, self.w = grid, th["hip"], w["hip"]
        self.th_knee = th["knee"]
        mass = float(np.sum(np.asarray(boat.crew_field(0.0)[0], float)))
        stature = float(boat.crew[0].rower.anthropometry.stature)
        g = RM.MC.GROUP[name]
        self.HE = J.anderson("HE", g, mass, stature, passive=False)
        self.HF = J.anderson("HF", g, mass, stature, passive=False)
        self.mods = Modules.pitto2024()
        self.delay = delay
        taus = np.linspace(0.0, self.P, 600, endpoint=False)
        y = np.array([M_ref(x) for x in taus])
        A = np.column_stack([self._cols(x, x, 1.0) for x in taus]).T
        pas = np.array([self.passive(x) for x in taus])
        best = None
        for sigma in (+1.0, -1.0):
            x, _ = nnls(sigma * A, y - sigma * pas)
            pred = sigma * (A @ x + pas)
            r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
            if best is None or r2 > best[0]:
                best = (r2, sigma, x)
        self.r2, self.sigma, (self.aE, self.aF) = float(best[0]), best[1], (float(best[2][0]), float(best[2][1]))
        self.y_ref, self.taus = y, taus

    def _cols(self, tau, c, td):
        x = float(tau) % self.P
        theta = float(np.clip(np.interp(x, self.grid, self.th, period=self.P), *RM.TESTED["hip"]))
        w = float(np.interp(x, self.grid, self.w, period=self.P)) * td        # d(flexion)/dt
        phase = ((c - self.delay) / self.P) % 1.0
        tE = float(self.mods("trunk_extensors", phase)) * float(np.ravel(self.HE.max_active(theta, -w))[0])
        tF = float(self.mods("trunk_flexors", phase)) * float(np.ravel(self.HF.max_active(theta, w))[0])
        return np.array([tE, -tF])

    def passive(self, tau):
        """The passive hip moment along the planned posture (extensor-positive, both legs), zero unless PASSIVE."""
        if not PASSIVE:
            return 0.0
        x = float(tau) % self.P
        h = float(np.interp(x, self.grid, self.th, period=self.P))
        k = float(np.interp(x, self.grid, self.th_knee, period=self.P))
        return float(silder_passive_hip_extension(h, k))

    def __call__(self, tau, c, td):
        cE, cF = self._cols(tau, c, td)
        return self.sigma * (self.aE * cE + self.aF * cF + self.passive(tau))


def measured_trunk_calibration(name, strokes):
    """The literature's check before tracking: does a *measured* trunk motion fit its EMG-timed muscles? The reference
    run on the athlete's own leg and back channels (rung2_population.own_body, a diagnostic input); its hip moment
    calibrated as above. Feasible means r^2 near the model body's or better and both levels <= 1."""
    RP.BLADE = "crossflow_normal"
    C2.TAKEOVER_DEG = C2.BIOROW_CATCH_TO_BURIED_DEG
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True, own_joints=RP.OWN_BODY_JOINTS)
    ref = RT.make(boat, "reference", body=RP.own_body(name, boat))
    run_r = ref.run_strokes(strokes, surge_speed=4.3)
    P = float(boat.timing.period)
    start = float(run_r.last_time[0])
    M = RT.table([(t, c, v) for t, c, v in ref.mlog if t >= start], P)
    out = {}
    for d in DELAYS_CHECKED:
        tm = TrunkMuscles(name, boat, M, delay=d)
        print("%s, OWN trunk motion, delay %.0f ms: extensors %.3f, flexors %.3f of maximal strength at the module peak "
              "(sign %+d); r^2 %.2f" % (name, 1000 * d, tm.aE, tm.aF, tm.sigma, tm.r2), flush=True)
        out["delay%.0fms" % (1000 * d)] = dict(extensors=tm.aE, flexors=tm.aF, sign=tm.sigma, r2=tm.r2)
    return out


#: excitation-to-force delays the feasibility check spans (the 50-100 ms window and either side)
DELAYS_CHECKED = (0.0, 0.075, 0.15)


def run(name, strokes, delays):
    RP.BLADE = "crossflow_normal"
    C2.TAKEOVER_DEG = C2.BIOROW_CATCH_TO_BURIED_DEG
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    ref, run_r, _Qr, M = RT.reference(boat, strokes)
    Q = population_Q(strokes, name, boat, m)
    tm = TrunkMuscles(name, boat, M)
    print("%s: calibrated trunk muscles: extensors %.3f, flexors %.3f of maximal strength at the module peak (sign %+d);"
          " fit to the reference hip moment r^2 %.2f" % (name, tm.aE, tm.aF, tm.sigma, tm.r2), flush=True)
    out = dict(calibration=dict(extensors=tm.aE, flexors=tm.aF, sign=tm.sigma, r2=tm.r2))
    for ff_kind in ("reference", "modules"):
        for dly in delays:
            boat2 = C2.R.boat_for(name, m, "on-water", drive_law=True)
            P = float(boat2.timing.period)
            holder = {}

            def ff(tau, holder=holder):
                v = float(M(tau)) if ff_kind == "reference" else tm(tau, holder.get("c", 0.0), holder.get("td", 1.0))
                sim = holder.get("sim")
                if sim is not None:
                    sim._ff_now = v                    # the layers' effort reads the feedforward (rung2_trunk_layers)
                return v
            free = RT.make(boat2, "free", Q=Q, tau_ff=ff, delay=dly)
            holder["sim"] = free
            orig = free.derivative

            def derivative(t, y, free=free, orig=orig, holder=holder):
                holder["c"], holder["td"] = t - free._stroke_start, float(y[-1])
                return orig(t, y)
            free.derivative = derivative
            free.start_state = np.concatenate([run_r.last_states[:, -1], [0.0, 0.0, P, 1.0]])
            key = "%s_fb%.0fms" % (ff_kind, 1000 * dly)
            try:
                run_f = free.run_strokes(strokes, surge_speed=4.3)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                tr = np.array(free.trlog) if free.trlog else np.zeros((1, 7))
                out[key] = dict(diverged=True, delta_max_deg=float(np.degrees(np.nanmax(np.abs(tr[:, 3])))),
                                error=str(exc))
                print("  %-22s DIVERGED (trunk error reached %.1f deg)" % (key, out[key]["delta_max_deg"]), flush=True)
                continue
            s = RT.summary(free, run_f)
            s.update(measured=dict(speed=m["speed"], ivv=m["ivv"], power=m["power"]))
            out[key] = s
            print("  %-22s %.3f m/s, IVV %.1f%% (measured %.1f%%), %.0f W; trunk error max %.2f deg, feedback max %.0f N m"
                  % (key, s["speed"], 100 * s["ivv"], 100 * m["ivv"], s["power"], s.get("delta_max_deg", np.nan),
                     s.get("feedback_max", np.nan)), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--delays", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--athletes", nargs="+", default=["cr06", "br24"])
    ap.add_argument("--own-body", action="store_true",
                    help="only the calibration on each athlete's own measured trunk motion (the feasibility check)")
    ap.add_argument("--layers", nargs="*", default=None, choices=["intrinsic", "reflex", "ll"],
                    help="the trunk's measured feedback layers (rung2_trunk_layers: activation-scaled intrinsic "
                         "stiffness, 40 ms reflex, long-latency PD) in place of the plain PD")
    ap.add_argument("--predict", action="store_true", help="the long-latency layer on the predicted state")
    ap.add_argument("--observer", type=float, default=0.0, help="with --predict, rung 5's disturbance observer gain")
    ap.add_argument("--passive", action="store_true",
                    help="the passive hip moment (Silder et al. 2007 eq. A.5, mean parameters) in the trunk's muscles")
    a = ap.parse_args()
    global PASSIVE
    PASSIVE = bool(a.passive)
    tag = "_passive" if a.passive else ""
    if a.layers is not None:
        import rung2_trunk_layers as TL
        TL.LayersSim.LAYERS = tuple(a.layers)
        TL.LayersSim.PREDICT = bool(a.predict or a.observer)
        TL.LayersSim.OBSERVER = float(a.observer)
        RT.TrunkClockSim = TL.LayersSim
        tag += "_layers_" + "_".join(a.layers) + ("_pred" if a.predict else "") + ("_obs%g" % a.observer if a.observer else "")
    if a.own_body:
        res = {n: measured_trunk_calibration(n, a.strokes) for n in a.athletes}
        path = os.path.join(RT.OUT, "rung2_trunk_modules_ownbody%s.json" % tag)
        json.dump(res, open(path, "w"), indent=1, default=float)
        print("wrote", path)
        return
    res = {n: run(n, a.strokes, a.delays) for n in a.athletes}
    path = os.path.join(RT.OUT, "rung2_trunk_modules%s.json" % tag)
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
