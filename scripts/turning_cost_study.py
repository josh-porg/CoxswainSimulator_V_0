"""Turning cost: the research-profile 6DOF boat driven through measured turns.

SOURCES sec. 177.  The squad's Tail of the Lake 2026 CoxBox GPS gives the speed a turn costs,
``dv/v = k omega^2`` (``research/raceday/rudder_totl.py``).  Here the research boat -- dynamic
oar, [CR06] sweep catch, [WF58] fins at reflection 2, Munk 0.50 (SOURCES sec. 169) -- is steered
through each measured turn's heading-rate trace and its speed lost per stroke is compared with the
crew's.  Nothing is fitted to the measurement.

The coxswain is the ordinary PD loop (``HeadingController``'s gains) on the measured heading, with
the measured turn rate as the rate reference so the boat turns *with* the trace instead of lagging
it.  The same boat steered straight on the same heading loop is the baseline, stroke for stroke.

Each run also books where the turn's extra power goes, from the states the simulator produced:
the hull's cross-flow drag, its linear (radiation) damping, the fins' induced drag, the work of
the fins' side force against the local cross-flow, and the rudder's share of the fin loads (the
loads at the run's helm less the loads at zero helm, same state).  The Munk/added-mass Coriolis
load does no net work (``-C_A(nu) nu`` is power-free for any scale) and is not booked.

Operating point.  The research model at the crew's own power runs faster than the crew did (the
four at its Concept2 5 k power, 131 W per rower, settles at ~4.1 m/s against 3.5 m/s raced; see
SOURCES sec. 177), and the turning cost goes as ``omega^2 / V^2`` for drift drag at fixed omega,
so each boat is rowed at the handle power that gives the crew's raced straight-line speed at its
raced rate.  That sets the operating point only; no turning coefficient is touched.  The run at
the four's erg power is kept as a check (``--watts erg``).

    python scripts/turning_cost_study.py --crew four               # every case, one process
    python scripts/turning_cost_study.py --crew eight
    python scripts/turning_cost_study.py --crew four --watts erg   # at the crew's erg power
    python scripts/turning_cost_study.py --combine

Output: ``data/local/turning_cost/<crew>[_<tag>].json``.
"""
import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data" / "local" / "raceday"))

from coxswain import physics                                      # noqa: E402
from coxswain.boats import catalog                                # noqa: E402
from coxswain.core.state import State                             # noqa: E402
from coxswain.hydro.appendages import surface_load                # noqa: E402
from coxswain.sim.control import Coxswain, HeadingController      # noqa: E402
from coxswain.sim.dynamic_oar import simulator_for                # noqa: E402

OUT = ROOT / "data" / "local" / "turning_cost"

#: Raced rate (median over the race) and straight-line speed, from the CoxBox files.
CREWS = {
    "four": {"file": "four (women's masters 4+)", "rate": 30.0, "speed": 3.51,
             "turns": [(488.5, 508.5), (608.5, 636.3), (526.3, 545.9)]},
    "eight": {"file": "eight (mixed masters 8+)", "rate": 30.5, "speed": 4.08,
              "turns": [(472.4, 498.3), (544.0, 575.8), (92.8, 106.8)]},
}
#: Straight lead-in before each window, and the margin of measured trace kept either side of
#: the turn (the GPS turn rate is real heading there too, not zero).
LEAD, MARGIN, TAPER = 30.0, 8.0, 2.0
STEADY_RATES = (1.0, 2.0, 3.0)        # deg/s
STEADY_HOLD = 30.0
GAIN, RATE_GAIN = HeadingController.gain, HeadingController.rate_gain
SPLITS = (0.0, 0.15, 0.30)


def build(crew, watts):
    """The crew's boat on the research profile at ``watts`` handle power per rower."""
    profile = physics.resolve("research")
    rate = CREWS[crew]["rate"]
    if crew == "four":
        import hocr_crew
        boat, powers = hocr_crew.build(rate)
        boat = profile.apply(boat)
        scales = powers / powers.mean()
        erg = float(powers.mean())
    else:
        boat = profile.apply(catalog.build("8+", rate=rate))
        scales = np.ones(boat.n_seats)
        erg = None
    boat.handle_watts = float(erg if watts == "erg" else watts)
    boat.power_scales = scales
    return boat, erg


class Steer:
    """The coxswain's PD loop on a heading trace ``psi(t)`` with rate reference ``r(t)``."""

    def __init__(self, t, rate_deg, split=0.0):
        self.t = np.asarray(t, float)
        self.r = np.radians(np.asarray(rate_deg, float))
        self.psi = np.concatenate([[0.0], np.cumsum(0.5 * (self.r[1:] + self.r[:-1])
                                                     * np.diff(self.t))])
        peak = float(np.max(np.abs(self.r))) or 1.0
        self.split_amp = float(split)
        self._peak = peak

    def target(self, t):
        return float(np.interp(t, self.t, self.psi)), float(np.interp(t, self.t, self.r))

    def __call__(self, t, state):
        psi, r = self.target(t)
        demand = GAIN * (state.yaw - psi) + RATE_GAIN * (float(state.omega_hull[2]) - r)
        return float(np.clip(demand, -np.radians(45.0), np.radians(45.0)))

    def split(self, t, state):
        # positive split yaws the bow to starboard (negative yaw): a port turn takes a negative one
        _, r = self.target(t)
        return -self.split_amp * r / self._peak


def measured_window(crew, t0, t1):
    """Measured strokes over the window and its sim-time heading-rate trace (sim yaw + = port)."""
    import rudder_totl as rt
    path = os.path.join(rt.COXBOX, rt.FILES[CREWS[crew]["file"]])
    s, om, _u, _turning, _expected, loss, _ok = rt.per_stroke(path)
    a, b = t0 - MARGIN, t1 + MARGIN
    m = (s["t"] >= a) & (s["t"] <= b)
    tm, omm = s["t"][m], -om[m]                       # GPS + = starboard; sim yaw + = port
    tt = np.arange(0.0, LEAD + (b - a) + 15.0, 0.1)
    rel = tt - LEAD + a                               # race time at sim time tt
    r = np.interp(rel, tm, omm, left=0.0, right=0.0)
    ramp = np.clip((rel - a) / TAPER, 0, 1) * np.clip((b - rel) / TAPER, 0, 1)
    return dict(sim_t=tt, rate=r * ramp, race_t=tm, om=-omm, loss=loss[m], v=s["v"][m],
                rate_spm=s["rate"][m], a=a, b=b)


def steady_trace(omega):
    tt = np.arange(0.0, LEAD + STEADY_HOLD + 5.0, 0.1)
    r = omega * np.clip((tt - LEAD) / TAPER, 0, 1)
    return tt, r


def run(boat, steer, duration, speed0):
    cox = Coxswain(rudder_override=steer, pressure_split=steer.split)
    sim = simulator_for(boat, coxswain=cox, fast=True)
    res = sim.run(duration=duration, surge_speed=speed0)
    return sim, res


def strokes(boat, res):
    """Stroke-mean speed, |yaw rate| (deg/s) and the stroke's mid time."""
    period = float(boat.timing.period)
    tt = np.asarray(res.time)
    v = np.asarray(res.speed, float)
    yaw_rate = np.degrees(np.asarray(res.omega)[2])
    k = np.floor(tt / period + 1e-9).astype(int)
    out = []
    for i in range(k.max()):
        m = k == i
        if m.sum() > 3:
            out.append((period * (i + 0.5), float(v[m].mean()), float(np.abs(yaw_rate[m]).mean())))
    return np.array(out)


def book(sim, steer, res, every=0.1):
    """Mean dissipated power by mechanism, per stroke-sized sample block (W)."""
    boat = sim.boat
    rho = boat.water.density
    tt = np.asarray(res.time)
    idx = np.unique(np.searchsorted(tt, np.arange(tt[0], tt[-1], every)))
    rows = []
    for i in idx:
        st = State.from_vector(np.asarray(res.states)[:, i])
        vh, wh = st.velocity_hull, st.omega_hull
        u, v, r = float(vh[0]), float(vh[1]), float(wh[2])
        delta = steer(float(tt[i]), st)
        y, n = sim._cross_flow.load(v, r, rho, boat.resistance.cross_flow_lateral, 1.0)
        crossflow = -(y * v + n * r)
        rates = np.concatenate([vh, wh])
        lin = sim.damping_matrix.copy()
        lin[0, 0] = 0.0
        linear = float(rates @ lin @ rates)
        induced = side = rud_ind = rud_side = 0.0
        for surf in boat.appendages:
            f, _ = surface_load(surf, vh, r, delta, boat.water)
            local = v + r * float(surf.position[0])
            induced += -f[0] * u
            side += -f[1] * local
            if surf.controllable:
                f0, _ = surface_load(surf, vh, r, 0.0, boat.water)
                rud_ind += -(f[0] - f0[0]) * u
                rud_side += -(f[1] - f0[1]) * local
        beta = float(np.degrees(np.arctan2(v, u)))
        rows.append((tt[i], crossflow, linear, induced, side, rud_ind, rud_side,
                     abs(np.degrees(delta)), beta, u))
    return np.array(rows)


def block_means(rows, t0, t1):
    m = (rows[:, 0] >= t0) & (rows[:, 0] < t1)
    keys = ("crossflow", "linear", "fin_induced", "fin_side", "rudder_induced", "rudder_side",
            "helm_deg", "drift_deg", "u")
    return {k: float(np.mean(rows[m, j + 1])) if k != "drift_deg"
            else float(np.mean(np.abs(rows[m, j + 1]))) for j, k in enumerate(keys)}


def compare(boat, base, turn, t0, t1):
    """Mean stroke loss (fraction) over strokes whose mid time lies in [t0, t1]."""
    m = (turn[:, 0] >= t0) & (turn[:, 0] <= t1)
    n = min(len(base), len(turn))
    loss = 1.0 - turn[:n, 1] / base[:n, 1]
    return float(np.mean(loss[m[:n]])), float(np.mean(turn[:n][m[:n], 2])), loss


def study(crew, watts):
    started = time.time()
    tag = crew + ("" if watts is None else "_" + str(watts))

    def log(msg):
        print("[%s %6.0fs] %s" % (tag, time.time() - started, msg), flush=True)

    cfg = CREWS[crew]
    if watts is None:
        watts = operating_watts(crew, log)
    boat, erg = build(crew, watts)
    out = {"crew": crew, "rate": cfg["rate"], "raced_speed": cfg["speed"],
           "handle_watts": boat.handle_watts, "erg_watts": erg,
           "fins": [{"area": a.area, "aspect_ratio": a.aspect_ratio, "x": float(a.position[0]),
                     "controllable": bool(a.controllable),
                     "lift_model": getattr(a, "lift_model", None),
                     "reflection": getattr(a, "reflection", None)} for a in boat.appendages],
           "turns": [], "steady": []}
    speed0 = cfg["speed"]

    # straight baseline, long enough for every case
    longest = max(LEAD + (t1 - t0) + 2 * MARGIN + 15.0 for t0, t1 in cfg["turns"])
    longest = max(longest, LEAD + STEADY_HOLD + 5.0)
    straight = Steer([0.0, longest + 1.0], [0.0, 0.0])
    sim, res = run(boat, straight, longest, speed0)
    base = strokes(boat, res)
    base_book = book(sim, straight, res)
    settled = base[base[:, 0] > LEAD]
    out["straight_speed"] = float(settled[:, 1].mean())
    log("straight %.3f m/s at %.1f W" % (out["straight_speed"], boat.handle_watts))

    for omega in STEADY_RATES:
        tt, r = steady_trace(omega)
        steer = Steer(tt, r)
        sim, res = run(boat, steer, tt[-1], speed0)
        turn = strokes(boat, res)
        hold = (LEAD + STEADY_HOLD - 15.0, LEAD + STEADY_HOLD)
        loss, rate_got, _ = compare(boat, base, turn, *hold)
        rows = book(sim, steer, res)
        out["steady"].append({"omega": omega, "omega_achieved": rate_got, "loss": loss,
                              "k": loss / omega ** 2, "book": block_means(rows, *hold),
                              "book_straight": block_means(base_book, *hold)})
        log("steady %.1f deg/s: achieved %.2f, loss %.2f%%, helm %.1f deg, drift %.2f deg"
            % (omega, rate_got, 100 * loss, out["steady"][-1]["book"]["helm_deg"],
               out["steady"][-1]["book"]["drift_deg"]))

    for t0, t1 in cfg["turns"]:
        w = measured_window(crew, t0, t1)
        sim_t0, sim_t1 = LEAD + (t0 - w["a"]), LEAD + (t1 - w["a"])
        meas_turn = (w["race_t"] >= t0) & (w["race_t"] <= t1)
        entry = {"t0": t0, "t1": t1, "degrees": float(np.trapezoid(w["om"][meas_turn],
                                                                  w["race_t"][meas_turn])),
                 "measured_loss": float(np.nanmean(w["loss"][meas_turn])),
                 "measured_rate_spm": float(np.mean(w["rate_spm"][meas_turn])),
                 "measured_strokes": [(float(a), float(b), float(c)) for a, b, c in
                                      zip(w["race_t"], w["om"], w["loss"])],
                 "splits": []}
        for split in SPLITS:
            steer = Steer(w["sim_t"], w["rate"], split)
            sim, res = run(boat, steer, w["sim_t"][-1], speed0)
            turn = strokes(boat, res)
            loss, rate_got, per = compare(boat, base, turn, sim_t0, sim_t1)
            rows = book(sim, steer, res)
            yaw = np.degrees(np.unwrap(np.asarray(res.attitude)[2]))
            tt = np.asarray(res.time)
            turned = float(np.interp(sim_t1, tt, yaw) - np.interp(sim_t0, tt, yaw))
            entry["splits"].append({
                "split": split, "loss": loss, "mean_abs_rate": rate_got,
                "sim_degrees": turned,
                "target_degrees": float(np.degrees(steer.target(sim_t1)[0]
                                                   - steer.target(sim_t0)[0])),
                "book": block_means(rows, sim_t0, sim_t1),
                "book_straight": block_means(base_book, sim_t0, sim_t1),
                "per_stroke": [(float(a - LEAD + w["a"]), float(b), float(c))
                               for a, b, c in zip(turn[:len(per), 0], per,
                                                  turn[:len(per), 2])]})
            log("turn %.0f-%.0f split %.2f: measured %.2f%%, sim %.2f%%, turned %.1f of %.1f deg,"
                " helm %.1f deg" % (t0, t1, split, 100 * entry["measured_loss"], 100 * loss,
                                    turned, entry["splits"][-1]["target_degrees"],
                                    entry["splits"][-1]["book"]["helm_deg"]))
        out["turns"].append(entry)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (tag + ".json")).write_text(json.dumps(out, indent=1))
    log("done")


def operating_watts(crew, log):
    """Handle power per rower for the raced straight speed: two secant steps on v ~ P^(1/3)."""
    cfg = CREWS[crew]
    watts = 131.0 if crew == "four" else 160.0
    for _ in range(3):
        boat, _ = build(crew, watts)
        straight = Steer([0.0, 61.0], [0.0, 0.0])
        _, res = run(boat, straight, 60.0, cfg["speed"])
        v = strokes(boat, res)
        speed = float(v[v[:, 0] > 35.0, 1].mean())
        log("  %.1f W -> %.3f m/s (target %.2f)" % (watts, speed, cfg["speed"]))
        if abs(speed / cfg["speed"] - 1.0) < 0.004:
            break
        watts *= (cfg["speed"] / speed) ** 3
    return round(watts, 1)


def combine():
    for p in sorted(OUT.glob("*.json")):
        r = json.loads(p.read_text())
        print("\n%s: %.1f W/rower, straight %.2f m/s (raced %.2f)"
              % (p.stem, r["handle_watts"], r["straight_speed"], r["raced_speed"]))
        for s in r["steady"]:
            b, b0 = s["book"], s["book_straight"]
            print("  steady %.1f deg/s (got %.2f): loss %.2f%%  k %.5f  helm %.1f  drift %.2f"
                  % (s["omega"], s["omega_achieved"], 100 * s["loss"], s["k"], b["helm_deg"],
                     b["drift_deg"]))
            print("     extra W: crossflow %.1f linear %.1f fin-induced %.1f fin-side %.1f"
                  " (rudder share: induced %.1f side %.1f)"
                  % tuple(b[k] - b0[k] for k in ("crossflow", "linear", "fin_induced",
                                                  "fin_side", "rudder_induced", "rudder_side")))
        for t in r["turns"]:
            print("  turn %.0f-%.0f (%.0f deg): measured %.2f%%"
                  % (t["t0"], t["t1"], t["degrees"], 100 * t["measured_loss"]))
            for s in t["splits"]:
                print("     split %.2f: sim %.2f%%, turned %.1f/%.1f, helm %.1f, drift %.2f"
                      % (s["split"], 100 * s["loss"], s["sim_degrees"], s["target_degrees"],
                         s["book"]["helm_deg"], s["book"]["drift_deg"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crew", choices=tuple(CREWS))
    parser.add_argument("--watts", default=None,
                        help="handle W per rower, or 'erg' (the four); default: the raced speed")
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.combine:
        combine()
        sys.exit(0)
    w = args.watts
    if w not in (None, "erg"):
        w = float(w)
    study(args.crew, w)
