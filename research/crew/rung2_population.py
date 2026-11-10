"""Phase 4.3, rung 2's test (sprint 2, S2.5): the population's muscles on two athletes. Are the hands right?

    .venv/Scripts/python.exe research/crew/rung2_population.py [--strokes 10] [--athletes cr06 br24]

Rung 2a imposes [K05]'s on-water population drive law on the hands. Rung 2 (research/crew/rung2_clock.py) makes the
hands' time law an output: the stroke clock tau is solved from the body-and-oar virtual-work balance, driven by a
generalised muscle force Q, with a prescribed handover to the recovery just before the finish's turning point
(``rung2_clock.HANDOVER``) and the recovery retimed to keep the period.

The test, with nothing fitted:
  1. The population. [K05]'s crew (1.80 m, 72.2 kg, 32.3 spm, on-water rhythm) on [CR06]'s rig, with the population's
     body (research/crew/k05_body.py, legs at [K05]'s 0.51 m) and its drive law, run as rung 1 (tau = clock). Its
     inverse dynamics give Q_pop against drive progress s (the oar's fraction of its arc, catch 0 -> finish 1):
     the population's muscles, on the population's motion.
  2. Each athlete, on his or her own rig and rate with the model's body at their stature. Rung 1 with the population's
     drive law imposed is the baseline. Rung 2 drives the athlete with Q_pop at the same drive progress, scaled by the
     athlete's measured power over the population run's (equal work per stroke at their power). Posture, not cycle
     phase, carries the force across: by phase the population's late push landed after her blade had let go.
  3. The score: the athlete's measured oar angle as a fraction of the arc against time from the catch, rms over the
     drive (and in degrees for [CR06], whose rig the model shares); speed and IVV against theirs.

S2.5's gate: the athletes' IVV within about 5 points of measured, or stop and record.

``--blade crossflow_h10``: the same test on the review's cross-flow blade with its tangential load at [H10]'s on-water
level (x 0.25 of the quasi-steady law; SOURCES §200), strips across the blade, for the population run and the athletes
alike. ``--blade crossflow_normal``: the same normal load with no tangential load ([CG07]'s measured points resolve to a
force normal to the blade within ~2 %, SOURCES §201). Default ``slip``: the research profile's slip law (the run of SOURCES §181).

Outputs: data/local/crew/rung2_population.json (``_crossflow_h10`` suffix for that blade).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "biorow")]
import rung2_clock as C2                                         # noqa: E402
import k05_population_test as KP                                # noqa: E402
import k05_body                                                  # noqa: E402

from coxswain.crew.stroke import OnWaterTiming                  # noqa: E402
from coxswain.sim.control import Coxswain                        # noqa: E402

V = C2.R.V
OUT = C2.OUT


BLADE = "slip"
H10_TANGENTIAL = 0.25                                            # SOURCES §200: [H10]'s share needs x 0.22-0.29


#: the oar shaft bends as a quasi-static hinge ([L14]'s measured scull shaft, SOURCES sec. 217, coxswain.crew.oar_bending; SOURCES §193, §216)
BENDING = False
#: force the rigid oar even where a physics profile bends it (SOURCES §217)
RIGID = False
#: the crew's handle force-velocity on rung 2's muscles (coxswain.crew.handle_fv.HandleForceVelocity; SOURCES §218)
FV = None
#: the crew's hold on its planned speed, beta (rung2_clock.ClockSim.setup_clock; SOURCES §220): studied, not sourced
HOLD = 0.0
#: hold the rower's own pace (the previous drive's mean tau') instead of the plan's
HOLD_OWN = False
#: each athlete's last rung-2 simulator and run, ``(free sim, run)``, for traces (research/blade/bending_ring_trace.py)
LAST_RUN = {}
#: the blade's added mass ("none", "patton", "labbe"; coxswain.crew.blade_added_mass) and, with bending, the bend as a
#: dynamic state (coxswain.crew.oar_bending.BendDynamics; SOURCES §224)
ADDED_MASS = "none"
BEND_DYNAMICS = None
#: how the population's muscle force is handed to an athlete through the drive (SOURCES §225): "progress" -- at the same
#: fraction of the oar's arc (as built); "time" -- at the same fraction of the drive's duration (a muscle's activation
#: is timed, not placed); "split" -- the athlete's own body part (its measured timing is an input) plus the population's
#: handle part, scaled. The recovery is by cycle phase either way. Default "split" since §227 (user decision 2026-11-08).
Q_MAP = "split"
#: the arc fraction past the catch before the "stopped driving" release may act (DynamicOarSimulator release_after).
#: Default 0.5 since §227 (user decision 2026-11-08); 0 restores the release anywhere in the drive (§226's bug).
RELEASE_AFTER = 0.5
#: the blade's water at the surface as a fraction of Patton's unbounded value (EntryImmersion.surface_factor), with the
#: blade wetting over BioRow's 4 deg; None: no immersion on the buried catch, the whole added mass (as before §228)
SURFACE_FACTOR = None
#: the cross-flow law's lift slope over Helmbold's (LiftDragBlade.crossflow lift_scale): 1 as built; the three tank loads
#: agree on 1.2-1.4 (SOURCES §231, research/blade/lift_slope_profile.py)
LIFT_SCALE = 1.0
#: how the athletes' measured body channels are read before differentiation (own body): "gcv" -- Woltring's
#: generalised cross-validation spline on the raw points (default since §232: kinematic data smoothed before they are
#: differentiated, the amount set by the data); "as-built" -- an interpolating spline ([BR24]) and a chosen lam ([CR06])
CHANNEL_SMOOTHING = "gcv"
#: the population's drive fraction, set by population_tables
POP = {}


def blade_options(boat):
    """Simulator options for the blade under test (``BLADE``), and the shaft's bending if ``BENDING``."""
    opts = _blade_law_options(boat)
    if RIGID:
        opts["oar_bending"] = False
    elif BENDING:
        from coxswain.crew.oar_bending import OarBending
        oar = boat.rig.seats[0].oarlocks[0].oar
        opts["oar_bending"] = OarBending.laschowski2014_scull(float(oar.blade_centre_outboard), float(oar.blade_length))
        if BEND_DYNAMICS is not None:                            # the bend as a dynamic state (SOURCES §224)
            opts["bend_dynamics"] = BEND_DYNAMICS
    if RELEASE_AFTER:                                            # 0 / None: the release anywhere (the old rule)
        opts["release_after"] = RELEASE_AFTER
    if ADDED_MASS != "none":
        opts["blade_added_mass"] = ADDED_MASS
    return opts


def _blade_law_options(boat):
    if BLADE == "slip":
        return {}
    sys.path.insert(0, os.path.join(ROOT, "research", "blade"))
    from tangential_level_sweep import ScaledTangential
    from coxswain.crew.liftdrag import LiftDragBlade
    oar = boat.rig.seats[0].oarlocks[0].oar
    area, rho, r = float(oar.blade_area), float(boat.water.density), float(oar.blade_centre_outboard)
    cf = LiftDragBlade.crossflow(outboard=r, area=area, length=float(oar.blade_length), face_on=2.2, density=rho,
                                 lift_scale=LIFT_SCALE)
    blade = ScaledTangential(lift_amplitude=cf.lift_amplitude, drag_amplitude=cf.drag_amplitude, area=area, outboard=r,
                             density=rho, normal_from=cf, tangential_from=cf,
                             tangential_scale=H10_TANGENTIAL if BLADE == "crossflow_h10" else 0.0)
    opts = dict(blade_law="liftdrag", liftdrag_blade=blade, blade_span=float(oar.blade_length))
    if C2.TAKEOVER_DEG == "entry" or SURFACE_FACTOR is not None:
        # the load ramps in over the placement and the blade's water follows the wetted blade, kappa x Patton
        # (SOURCES sec. 173; kappa < 0.1 from the population force curves, and [RLBQV19]'s bend, §228)
        from coxswain.crew.blade_immersion import EntryImmersion
        opts["blade_immersion"] = EntryImmersion(surface_factor=1.0 if SURFACE_FACTOR is None else SURFACE_FACTOR)
    return opts


BODY = "model"


#: the joints an athlete's own leg and back channels time (coxswain.crew.body_options.CLOCKED_JOINTS)
OWN_BODY_JOINTS = ("shank", "thigh", "trunk")


def athlete_boat(name, m):
    """The athlete's boat; with their own body (``BODY == "own"``) built with no population clock on the joints their
    channels time (``own_joints``; SOURCES §228), as their boat was before the research profile's trunk clock."""
    return C2.R.boat_for(name, m, "on-water", drive_law=True, own_joints=OWN_BODY_JOINTS if BODY == "own" else ())


def own_body(name, boat):
    """The athlete's own measured leg and back time-laws on the model's postures (a diagnostic input, not a fit):
    [BR24] research/biorow/measured_body.py (seat Ls, trunk Lt, his back travel); [CR06]
    research/cr06/self_consistent_body_cr06.py (leg and back channels, her back travel 0.398 / 0.513).

    The channels time the legs and the trunk, so ``boat`` must carry no population clock on those joints (build it
    with ``own_joints=OWN_BODY_JOINTS``): the research profile's trunk clock (§213) had stayed in the posture base the
    channels are warped onto -- two trunk time laws in one body, +1.7 / +0.4 IVV points (SOURCES §228)."""
    clocked = {c.joint for mbr in boat.crew for c in (mbr.rower.body.clocks if mbr.rower.body is not None else ())}
    if clocked & set(OWN_BODY_JOINTS):
        raise ValueError("own body on a boat whose %s carry a population clock: build it with own_joints "
                         "(athlete_boat; SOURCES §228)" % sorted(clocked & set(OWN_BODY_JOINTS)))
    if name == "br24":
        import measured_body as MBd
        MBd.CHANNEL_SMOOTHING = "gcv" if CHANNEL_SMOOTHING == "gcv" else "interp"
        return MBd.body_field(boat, True)[0]
    path = os.path.join(ROOT, "research", "cr06", "self_consistent_body_cr06.py")
    src = open(path, encoding="utf-8").read().split("grid = np.linspace")[0]
    ns = {"__file__": path, "__name__": "cr06_body"}
    exec(compile(src, path, "exec"), ns)
    ns["CHANNEL_SMOOTHING"] = "gcv" if CHANNEL_SMOOTHING == "gcv" else "as-built"
    return ns["body_field"](boat, True, ns["K_BACK"])


def make(boat, mode, Q=None, body=None):
    boat.power_scales = np.ones(boat.n_seats)
    sim = C2.ClockSim(boat, peak_torque=0.0, catch="sweep", crew="handle",
                      coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True, **blade_options(boat))
    sim.setup_clock(mode, Q, body=body, fv=FV if mode == "free" else None, hold=HOLD if mode == "free" else 0.0,
                    hold_own=HOLD_OWN)
    return sim


def parts_table(sim, run):
    """The last stroke's residual split into the body's part and the handle's, against cycle phase (§225)."""
    P = float(sim.boat.timing.period)
    start = float(run.last_time[0])
    lg = np.array([(c, b, h) for t, c, b, h in sim.qparts if t >= start])
    o = np.argsort(lg[:, 0])
    lg = lg[o]
    keep = np.concatenate([[True], np.diff(lg[:, 0]) > 1e-9])
    lg = lg[keep]
    return lg[:, 0] / P, lg[:, 1], lg[:, 2]


def q_table(sim, run):
    P = float(sim.boat.timing.period)
    start = float(run.last_time[0])
    lg = np.array([(c, q) for t, c, q in sim.qlog if t >= start])
    o = np.argsort(lg[:, 0]); c, q = lg[o, 0], lg[o, 1]
    keep = np.concatenate([[True], np.diff(c) > 1e-9])
    return c[keep] / P, q[keep]


def progress_of(boat, tau):
    sw, tm = boat.oar_sweep, boat.timing
    return (sw.catch_angle - np.array([float(sw(x, tm)) for x in np.atleast_1d(tau)])) / (sw.catch_angle - sw.finish_angle)


def model_track(sim, run, free):
    """The model's drive progress and oar angle (deg) against time from the catch, over the last stroke's drive."""
    t = run.last_time
    if free:
        tl = np.array(sim.tlog)
        tl = tl[tl[:, 0] >= t[0] - 1e-9]
        tt, tau = tl[:, 0] - t[0], tl[:, 1]
    else:
        tt = t - t[0]; tau = tt
        keep = tau <= float(sim.boat.timing.drive_fraction) * float(sim.boat.timing.period)
        tt, tau = tt[keep], tau[keep]
    sw, tm = sim.boat.oar_sweep, sim.boat.timing
    return tt, progress_of(sim.boat, tau), np.degrees([float(sw(x, tm)) for x in tau])


def measured_track(name):
    """The athlete's measured oar angle: (time from the catch, drive progress, angle deg) over the drive."""
    if name == "cr06":
        f = V._cr06_series("oar_angle_deg")
        T = V.CR06_T
        g = np.linspace(0.0, T, 2000, endpoint=False)
        a = np.array([float(f(x)) for x in g])
        i0 = int(np.argmax(a))                                   # her catch: the angle's maximum
    else:
        import like_for_like as L
        D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
        T = 60.0 / L.RATE
        t0 = np.arange(len(D)) * T / len(D)
        raw = 0.5 * (D["A1"] + D["A2"])
        g = np.linspace(0.0, T, 2000, endpoint=False)
        a = -np.interp(g, t0, raw, period=T)                     # his catch is his minimum: flip to the model's sense
        i0 = int(np.argmax(a))
    a2 = np.roll(a, -i0)
    tt = np.arange(a2.size) * T / a2.size
    j = int(np.argmin(a2))                                       # the finish: the angle's minimum
    a2, tt = a2[:j + 1], tt[:j + 1]
    s = (a2[0] - a2) / (a2[0] - a2[-1])
    return tt, s, a2


def compare(tt, x, ht, hx):
    end = min(tt[-1], ht[-1])
    grid = np.linspace(0.0, end, 200)
    m = np.interp(grid, tt, x); h = np.interp(grid, ht, hx)
    return float(np.sqrt(np.mean((m - h) ** 2))), float(np.max(np.abs(m - h)))


def athlete(name, strokes, s_pop, q_drive, ph, q, p_pop):
    m = V.ATHLETES[name][0]()
    ht, hs, ha = measured_track(name)
    out = dict(measured=dict(speed=m["speed"], ivv=m["ivv"], power=m["power"]))
    base_boat = athlete_boat(name, m)
    body = own_body(name, base_boat) if BODY == "own" else None
    base = make(base_boat, "reference", body=body)
    run_b = base.run_strokes(strokes, surge_speed=4.3)
    s_b = C2.summary(run_b)
    tt, ss, aa = model_track(base, run_b, False)
    prog_b = compare(tt, ss, ht, hs)
    ang_b = compare(tt, aa, ht, ha) if name == "cr06" else (float("nan"), float("nan"))
    out["rung1"] = dict(s_b, progress_rms=prog_b[0], progress_max=prog_b[1], angle_rms=ang_b[0], angle_max=ang_b[1])

    P = float(base_boat.timing.period)
    D = float(base_boat.timing.drive_fraction) * P
    k = m["power"] / p_pop

    if Q_MAP == "split":
        own_ph, body_part, _own_handle = parts_table(base, run_b)

    def Q(tau):
        if tau <= D:
            if Q_MAP == "time":                                  # the population's force at the same fraction of its drive
                return k * float(np.interp(float(tau) / D * POP["drive_fraction"], ph, q, period=1.0))
            if Q_MAP == "split":
                # the athlete's own body (its measured timing is an input) plus the population's handle force, scaled
                s_now = float(progress_of(base_boat, tau)[0])
                return (float(np.interp(float(tau) / P, own_ph, body_part, period=1.0))
                        + k * float(np.interp(s_now, *POP["handle_drive"])))
            return k * float(np.interp(float(progress_of(base_boat, tau)[0]), s_pop, q_drive))
        return k * float(np.interp(np.mod(tau / P, 1.0), ph, q, period=1.0))

    free_boat = athlete_boat(name, m)
    free = make(free_boat, "free", Q=Q, body=own_body(name, free_boat) if BODY == "own" else None)
    free.start_state = np.concatenate([run_b.last_states[:, -1], [P, 1.0]])
    run_f = free.run_strokes(strokes, surge_speed=4.3)
    LAST_RUN[name] = (free, run_f)
    s_f = C2.summary(run_f)
    tt, ss, aa = model_track(free, run_f, True)
    prog_f = compare(tt, ss, ht, hs)
    ang_f = compare(tt, aa, ht, ha) if name == "cr06" else (float("nan"), float("nan"))
    tl = np.array(free.tlog); last = tl[tl[:, 0] >= float(run_f.last_time[0])]
    over = sum(1 for r in free.rlog if r[4] != "ok")
    import handle_force_shape as HFS                             # the drive's force shape (§227; the validation ladder)
    out["rung2"] = dict(s_f, progress_rms=prog_f[0], progress_max=prog_f[1], angle_rms=ang_f[0], angle_max=ang_f[1],
                        scale=k, overruns=over, tau_dot_last=[float(last[:, 2].min()), float(last[:, 2].max())],
                        force_shape=HFS.shape(*HFS.model_stroke(free, run_f)),
                        speed_trace=dict(t=(run_f.last_time - run_f.last_time[0]).tolist(),
                                         v=run_f.last_states[6].tolist(),
                                         air=np.asarray(free._air_mask[0], dtype=bool).tolist()),
                        trace=dict(t=(last[:, 0] - float(run_f.last_time[0])).tolist(), tau_dot=last[:, 2].tolist()))
    for lab, s, pr, an in (("rung 1, population law imposed", s_b, prog_b, ang_b),
                           ("rung 2, population muscles x %.3f" % k, s_f, prog_f, ang_f)):
        print("%s, %s: %.3f m/s, IVV %.1f%%, %.0f W; oar progress vs measured: rms %.3f (max %.3f)%s" % (
            name, lab, s["speed"], 100 * s["ivv"], s["power"], pr[0], pr[1],
            "; angle rms %.1f deg (max %.1f)" % an if name == "cr06" else ""), flush=True)
    print("%s, measured: %.3f m/s, IVV %.1f%%, %.0f W; rung 2 drive tau' %.3f to %.3f, overruns %d" % (
        name, m["speed"], 100 * m["ivv"], m["power"], last[:, 2].min(), last[:, 2].max(), over), flush=True)
    return out


def population_tables(strokes):
    """[K05]'s population at rung 1 with its own body and drive law: ``(summary, (s_pop, q_drive, phase, q))``, the
    muscles' generalised force against drive progress and against cycle phase, as rung 2 carries them to an athlete."""
    pop_boat = V.cr06_boat(dict(leg_travel=None), timing=OnWaterTiming(KP.K05_RATE), stature=KP.K05_STATURE,
                           drive_law=True, mass=KP.K05_MASS)
    body, _k = k05_body.body_field(pop_boat, leg_travel=0.51)
    pop = make(pop_boat, "reference", body=body)
    run_p = pop.run_strokes(strokes, surge_speed=4.3)
    s_p = C2.summary(run_p)
    ph, q = q_table(pop, run_p)
    P_pop = float(pop_boat.timing.period)
    on = ph * P_pop <= float(pop_boat.timing.drive_fraction) * P_pop
    s_pop = np.maximum.accumulate(progress_of(pop_boat, ph[on] * P_pop))
    q_drive = q[on]
    keep = np.concatenate([[True], np.diff(s_pop) > 1e-9])
    POP["drive_fraction"] = float(pop_boat.timing.drive_fraction)
    # the handle's part alone against drive progress, for Q_MAP "split"
    pph, _pbody, phandle = parts_table(pop, run_p)
    pon = pph * P_pop <= float(pop_boat.timing.drive_fraction) * P_pop
    ps = np.maximum.accumulate(progress_of(pop_boat, pph[pon] * P_pop))
    pkeep = np.concatenate([[True], np.diff(ps) > 1e-9])
    POP["handle_drive"] = (ps[pkeep], phandle[pon][pkeep])
    return s_p, (s_pop[keep], q_drive[keep], ph, q)


def solved_clock(name, strokes, tables, p_pop):
    """Rung 2 for one athlete (population muscles scaled to the athlete's measured power, hands an output): its
    last stroke's clock, ``(P, c, tau, tau', tau'')`` against time since the stroke's start, and its summary."""
    s_pop, q_drive, ph, q = tables
    m = V.ATHLETES[name][0]()
    base_boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    base = make(base_boat, "reference")
    run_b = base.run_strokes(strokes, surge_speed=4.3)
    P = float(base_boat.timing.period)
    D = float(base_boat.timing.drive_fraction) * P
    k = m["power"] / p_pop

    def Q(tau):
        if tau <= D:
            return k * float(np.interp(float(progress_of(base_boat, tau)[0]), s_pop, q_drive))
        return k * float(np.interp(np.mod(tau / P, 1.0), ph, q, period=1.0))
    free_boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    free = make(free_boat, "free", Q=Q)
    free.start_state = np.concatenate([run_b.last_states[:, -1], [P, 1.0]])
    run_f = free.run_strokes(strokes, surge_speed=4.3)
    t0 = float(run_f.last_time[0])
    tl = np.array([r for r in free.tlog if r[0] >= t0])
    o = np.argsort(tl[:, 0])
    tl = tl[o]
    keep = np.concatenate([[True], np.diff(tl[:, 0]) > 1e-9])
    tl = tl[keep]
    tau = np.unwrap(np.mod(tl[:, 1], P) * 2 * np.pi / P) * P / (2 * np.pi)
    return (P, tl[:, 0] - t0, tau - tau[0], tl[:, 2], tl[:, 3]), C2.summary(run_f), k


def main():
    global FV, HOLD, HOLD_OWN, ADDED_MASS, BEND_DYNAMICS, Q_MAP, RELEASE_AFTER, SURFACE_FACTOR, LIFT_SCALE, CHANNEL_SMOOTHING
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--athletes", nargs="+", default=["cr06", "br24"])
    ap.add_argument("--blade", default="slip", choices=["slip", "crossflow_h10", "crossflow_normal"])
    ap.add_argument("--body", default="model", choices=["model", "own"],
                    help="the posture path: the model's (ergometer keyframes on the clock) or, as a diagnostic, each "
                         "athlete's own measured leg and back time-laws")
    ap.add_argument("--catch-posture", default="as-built", choices=["as-built", "straight-arms"],
                    help="the athletes' rig: as built (arms bent at the catch) or the footboard for a straight-arm "
                         "catch with the body-coupled drive (handle_rung1.CATCH_POSTURE; SOURCES sec. 208)")
    ap.add_argument("--catch", default="dynamic", choices=["dynamic", "buried", "entry"],
                    help="when the drive's clock turns dynamic: at the catch (as first built) or once the blade is "
                         "buried (BioRow's 6 deg catch slip; rung2_clock.TAKEOVER_DEG)")
    ap.add_argument("--clocks", nargs="*", default=[],
                    help="the body's segment clocks by name (coxswain.crew.body_options.CLOCKS), e.g. pitto2024_trunk "
                         "(SOURCES sec. 211)")
    ap.add_argument("--hand-terms", nargs="*", default=[],
                    help="with --catch-posture straight-arms: the hand chain's reach terms by name "
                         "(coxswain.crew.hand_chain.TERMS), e.g. stretch (SOURCES sec. 210; athlete-measured)")
    ap.add_argument("--travel", default=None,
                    help="the legs' and trunk's travel from a population by name (coxswain.crew.body_options.TRAVELS), "
                         "e.g. k05_racing (SOURCES sec. 212)")
    ap.add_argument("--bending", action="store_true",
                    help="the oar shaft bends ([L14]'s measured scull shaft as a quasi-static hinge; SOURCES sec. 216-217)")
    ap.add_argument("--fv", type=float, default=None,
                    help="the muscles' handle force-velocity with this V0, m/s (SOURCES sec. 218)")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="the crew's hold on its planned speed, beta (SOURCES sec. 220)")
    ap.add_argument("--hold-own", action="store_true", help="the hold's reference is the rower's own pace")
    ap.add_argument("--added-mass", default="none", choices=["none", "patton", "labbe"],
                    help="the blade's added mass (coxswain.crew.blade_added_mass)")
    ap.add_argument("--bend-dynamics", action="store_true",
                    help="with --bending: the bend as a dynamic state with the blade's mass and added mass (§224)")
    ap.add_argument("--structural-damping", type=float, default=None,
                    help="the shaft's structural damping ratio for --bend-dynamics (default BendDynamics')")
    ap.add_argument("--dt", type=float, default=None,
                    help="fixed integration step, s (default: integrators.estimate_step): a convergence test")
    ap.add_argument("--release-after", type=float, default=RELEASE_AFTER,
                    help="the arc fraction past the catch before the stopped-driving release may act (§226; default "
                         "0.5 since §227; 0 = anywhere, the old rule)")
    ap.add_argument("--channel-smoothing", default=CHANNEL_SMOOTHING, choices=["gcv", "as-built"],
                    help="the athletes' body channels before differentiation (§232; default gcv)")
    ap.add_argument("--lift-scale", type=float, default=1.0,
                    help="the cross-flow law's lift slope over Helmbold's (1 as built; tank profile 1.2-1.4, §231)")
    ap.add_argument("--surface-factor", type=float, default=None,
                    help="the blade wets over 4 deg and carries kappa x Patton of water (§173; §228)")
    ap.add_argument("--q-map", default=Q_MAP, choices=["progress", "time", "split"],
                    help="hand the population's muscle force over by drive progress (as built), drive time, or the "
                         "athlete's own body part plus the population's handle part (§225; default split since §227)")
    a = ap.parse_args()
    if a.dt:
        import coxswain.core.integrators as I
        I.estimate_step = lambda *args, **kw: float(a.dt)
    Q_MAP = a.q_map
    RELEASE_AFTER = a.release_after
    SURFACE_FACTOR = a.surface_factor
    LIFT_SCALE = a.lift_scale
    CHANNEL_SMOOTHING = a.channel_smoothing
    HOLD = float(a.hold)
    HOLD_OWN = bool(a.hold_own)
    ADDED_MASS = a.added_mass
    if a.bend_dynamics:
        from coxswain.crew.oar_bending import BendDynamics
        BEND_DYNAMICS = BendDynamics() if a.structural_damping is None else BendDynamics(
            damping_ratio=a.structural_damping)
    if a.fv is not None:
        from coxswain.crew.handle_fv import HandleForceVelocity
        FV = HandleForceVelocity(a.fv, "--fv")
    global BLADE, BODY, BENDING
    BENDING = bool(a.bending)
    BLADE = a.blade
    BODY = a.body
    C2.TAKEOVER_DEG = {"buried": C2.BIOROW_CATCH_TO_BURIED_DEG, "entry": "entry"}.get(a.catch)
    C2.R.CATCH_POSTURE = a.catch_posture
    C2.R.BODY_CLOCKS = tuple(a.clocks)
    C2.R.HAND_TERMS = tuple(a.hand_terms)
    C2.R.BODY_TRAVEL = a.travel
    print("blade: %s; catch: %s" % (BLADE, a.catch), flush=True)
    out = dict(handover=C2.HANDOVER)

    # 1. the population, rung 1 with its own body and drive law
    s_p, (s_pop, q_drive, ph, q) = population_tables(a.strokes)
    print("population (rung 1, [K05] body and law): %.3f m/s, IVV %.1f%%, %.0f W; Q_pop %.0f to %.0f"
          % (s_p["speed"], 100 * s_p["ivv"], s_p["power"], q.min(), q.max()), flush=True)
    out["population"] = s_p

    # 2. each athlete
    for name in a.athletes:
        out[name] = athlete(name, a.strokes, s_pop, q_drive, ph, q, s_p["power"])
    json.dump(out, open(os.path.join(OUT, "rung2_population%s%s.json" % ("" if BLADE == "slip" else "_" + BLADE,
                                                            ("" if a.catch == "dynamic" else "_" + a.catch) + ("" if a.body == "model" else "_ownbody")
                                                            + ("" if a.catch_posture == "as-built" else "_straightarms")
                                                            + "".join("_" + c for c in a.clocks)
                                                            + "".join("_" + t for t in a.hand_terms)
                                                            + ("_" + a.travel if a.travel else "")
                                                            + ("_bending" if a.bending else "")
                                                            + ("_fv%g" % a.fv if a.fv is not None else "")
                                                            + ("_hold%g" % a.hold if a.hold else "") + ("own" if a.hold_own else "")
                                                            + ("" if a.added_mass == "none" else "_am" + a.added_mass)
                                                            + ("_benddyn%g" % BEND_DYNAMICS.damping_ratio if BEND_DYNAMICS is not None else "")
                                                            + ("_dt%g" % a.dt if a.dt else "")
                                                            # always named: the defaults moved in §227 (S4)
                                                            + "_qmap" + a.q_map + "_release%g" % a.release_after
                                                            + ("_kappa%g" % a.surface_factor if a.surface_factor is not None else "")
                                                            + ("" if a.lift_scale == 1.0 else "_lift%g" % a.lift_scale)
                                                            + ("_chan" + a.channel_smoothing)
                                                            + ("" if a.athletes == ["cr06", "br24"] else "_" + "_".join(a.athletes)))), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
