"""Where does a crew find sixty seconds?

    python scripts/time_budget.py --race-time 1140 --target 60
    python scripts/time_budget.py --race-time 1140 --no-line

Every other script in this project asks "what does X cost?" and answers
in seconds off a reference crew rowing at 5.06 m/s.  That reference is a
collegiate heavyweight eight, and it is the wrong boat for the question a
masters crew actually asks, which is not "what is the biggest effect" but
"where do I find a specific number of seconds by next October".

Two things change when the crew is slower, and they pull in the same
direction, which is the useful part:

**Fractional levers pay more seconds.**  Speed responds to power as
``v ~ P^n`` with ``n`` measured here rather than assumed -- the textbook
cube root gives 0.33, and this hull comes out higher because its
resistance is sub-quadratic in the speed range that matters.  Whatever
``n`` is, a 1% power gain is a fixed *fraction* of race time, so a longer
race converts it into more seconds.

**Distance levers pay more seconds too.**  Rowing an extra 40 m costs
``40 / v`` seconds, and a slower boat spends longer covering the excess.

So the same steering error that costs a fast crew ten seconds costs a
slower one twelve, and every number in the standing report is a *lower*
bound for the crew reading it.  That is worth knowing before deciding the
tactical levers are too small to bother with.

What this script does not model
-------------------------------
Rate, ratio, catch timing as a skill rather than as scatter, and the
crew's ability to hold a power for sixteen minutes rather than produce it
for twenty strokes.  The last one is the important omission: this budget
prices *having* the power, and the critical-power model says holding it
is a separate question with a separate answer.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                        # noqa: E402
from coxswain.boats.rig import RIG_PATTERNS               # noqa: E402
from coxswain.crew.variability import CLUB, ELITE, JUNIOR  # noqa: E402
from coxswain.progress import progress                    # noqa: E402
from coxswain.sim.control import Coxswain                 # noqa: E402
from coxswain.sim.simulator import RowingSimulator        # noqa: E402

RACE_LENGTH = 4822.0

#: A women's masters eight, not the collegiate heavyweight the catalog
#: defaults to.  Mass and stature move the boat's displacement, wetted
#: area and roll inertia, so they are not cosmetic: an 88 kg default crew
#: sits 130 kg deeper than this one.
MASTERS = dict(rower_mass=72.0, rower_stature=1.72, coxswain_mass=55.0)


def build(power=1.0, rate=28.0, rig=None, mass=None, stature=None,
          weak=None, deficit=0.0):
    kwargs = dict(MASTERS)
    if mass is not None:
        kwargs["rower_mass"] = mass
    if stature is not None:
        kwargs["rower_stature"] = stature
    boat = catalog.eight(rate=rate, rig_pattern=rig, **kwargs)
    scales = np.full(boat.n_seats, float(power))
    if weak is not None:
        scales[weak] *= 1.0 - deficit
    boat.power_scales = scales
    return boat


def steady_speed(boat, duration=20.0, dt=0.02, variability=None, seed=0,
                 guess=4.4):
    """Mean speed over a whole number of settled stroke cycles.

    Averaging over an arbitrary window biases the mean, because speed
    swings by a few percent within every stroke and a partial cycle
    weights part of that swing twice.  The bias is common-mode and would
    cancel out of the differences this script reports, but every lever
    here is worth a fraction of a percent and it costs nothing to remove
    it: take the last whole number of periods instead of the last half.
    """
    if variability is not None:
        model = dataclasses.replace(variability, seed=seed)
        model.reset()
        # On top of the crew's calibrated power, not instead of it.
        model.apply(boat, base=boat.power_scales)
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=dt, surge_speed=guess)
    t = np.asarray(result.time)
    period = boat.timing.period
    cycles = int((t[-1] - 0.5 * t[-1]) // period)       # settled, whole
    if cycles < 1:
        raise SystemExit("duration %.1f s is under two stroke cycles at "
                         "rate %.0f" % (duration, boat.timing.rate))
    keep = t >= t[-1] - cycles * period
    velocity = np.asarray(result.velocity)[:2].T[keep]
    return float(np.hypot(*velocity.T).mean())


def calibrate(race_time, rate, tol=0.4, limit=18):
    """Uniform power scale that makes the model row the stated race time.

    The crew is anchored to an observed result rather than to a power
    number nobody has measured.  A women's grand masters eight is not a
    collegiate crew at 302.7 W and pretending otherwise would put every
    lever below at the wrong operating point.
    """
    target = RACE_LENGTH / float(race_time)
    low, high = 0.20, 1.60
    bar = progress(total=limit, desc="calibrating", unit="run")
    speed = None
    for _ in range(limit):
        mid = 0.5 * (low + high)
        speed = steady_speed(build(power=mid, rate=rate), guess=target)
        if abs(RACE_LENGTH / speed - race_time) < tol:
            bar.close()
            return mid, speed
        if speed < target:
            low = mid
        else:
            high = mid
        bar.update(1)
    bar.close()
    if low <= 0.21 or high >= 1.59:
        raise SystemExit("could not bracket that race time -- the bisection "
                         "ran to its limit, which means %.0f s is outside "
                         "what this hull does at rate %.0f" % (race_time, rate))
    return mid, speed


def seconds(speed, reference):
    """Race-time change from a speed change, positive meaning slower."""
    return RACE_LENGTH / speed - RACE_LENGTH / reference


def pacing_cost(exponent, race_time, critical_power=200.0, capacity=9000.0,
                overshoot=0.08, collapse=0.04):
    """What going out too hard costs, in seconds.

    Masters values, not the collegiate ones in :mod:`coxswain.crew.exertion`
    -- CP falls with age and this crew is not a group of 21-year-olds.
    The numbers barely matter for the conclusion, which is a structural
    one about how little of a head race sits above CP.

    Two versions of the same mistake:

    **Disciplined.**  Out hard, reserve empties, crew settles to exactly
    CP and holds it.  Work is conserved and the only loss is the
    concavity of ``v ~ P^n`` -- second order in the overshoot, and small.

    **Real.**  The reserve empties, the rate falls away, and the crew
    finishes *below* CP because a blown crew does not calmly find its
    aerobic ceiling.  This loss is first order and it is the one worth
    coaching against.
    """
    race_power = critical_power + capacity / race_time
    reference_speed = RACE_LENGTH / race_time

    def speed_at(power):
        return reference_speed * (power / race_power) ** exponent

    hard = race_power * (1.0 + overshoot)
    burn = capacity / max(hard - critical_power, 1.0)   # s to empty W'
    covered = speed_at(hard) * burn
    out = {}
    for name, after in (("disciplined", critical_power),
                        ("real", critical_power * (1.0 - collapse))):
        remaining = max(RACE_LENGTH - covered, 0.0)
        out[name] = burn + remaining / speed_at(after) - race_time
    out["race power over CP"] = race_power / critical_power - 1.0
    out["seconds above CP"] = burn
    return out


def measure(args):
    rate = args.rate
    print("women's masters eight: %.0f kg rowers, %.2f m, rate %.0f"
          % (MASTERS["rower_mass"], MASTERS["rower_stature"], rate))
    scale, base = calibrate(args.race_time, rate)
    print("calibrated to %.0f s: power scale %.3f, %.3f m/s (%.0f s)\n"
          % (args.race_time, scale, base, RACE_LENGTH / base))

    rows = []          # (lever, seconds, kind, note)

    # -- how speed answers power -----------------------------------------
    up = steady_speed(build(power=scale * 1.05, rate=rate), guess=base)
    exponent = float(np.log(up / base) / np.log(1.05))
    per_percent = -seconds(steady_speed(build(power=scale * 1.01, rate=rate),
                                        guess=base), base)
    print("speed responds to power as P^%.3f (cube root would be 0.333),"
          % exponent)
    print("so 1%% of crew power is %.1f s here, and %.1f%% is the whole %.0f s."
          % (per_percent, args.target / per_percent, args.target))
    print()

    # -- crew mass --------------------------------------------------------
    plan_gain = 0.0
    print("crew mass")
    for drop in args.mass_drops:
        v = steady_speed(build(power=scale, rate=rate,
                               mass=MASTERS["rower_mass"] - drop), guess=base)
        gain = -seconds(v, base)
        print("  %.0f kg per rower (%.0f kg off the boat): %+6.1f s"
              % (drop, 8 * drop, gain))
        if abs(drop - args.mass_plan) < 1e-6:
            plan_gain = gain
            rows.append(("lose %.0f kg per rower" % drop, gain, "training",
                         "%.0f kg less displacement; wetted area falls with "
                         "it. Worth it only if less than %.1f%% of power "
                         "goes with the mass"
                         % (8 * drop, gain / per_percent)))
    # The model prices displacement and nothing else, so it will always
    # say a lighter crew is faster.  A rower who sheds mass and power
    # together can easily come out behind, and this is the break-even:
    # above it the power loss costs more than the draft saves.
    print("  the model only sees displacement.  Losing %.0f kg per rower is"
          % args.mass_plan)
    print("  worth %+.1f s, so it pays only if the crew gives up less than"
          % plan_gain)
    print("  %.1f%% of its power getting there -- which is the whole question"
          % (plan_gain / per_percent))
    print("  and not one this model can answer.")
    print()

    # -- consistency ------------------------------------------------------
    print("crew consistency (%d draws each)" % args.draws)
    levels = {"junior": JUNIOR, "club": CLUB, "elite": ELITE}
    means = {}
    bar = progress(total=3 * args.draws, desc="scatter", unit="draw")
    for name, model in levels.items():
        speeds = []
        for seed in range(args.draws):
            speeds.append(steady_speed(build(power=scale, rate=rate),
                                       variability=model, seed=seed,
                                       guess=base))
            bar.update(1)
        means[name] = float(np.mean(speeds))
        print("  %-7s sigma %.3f: %.4f m/s, %+6.1f s vs a perfect crew"
              % (name, model.power_sigma, means[name],
                 seconds(means[name], base)))
    bar.close()
    tighten = seconds(means["club"], base) - seconds(means["elite"], base)
    rows.append(("club to elite consistency", tighten, "training",
                 "same average power, 3.5% stroke-to-stroke scatter "
                 "down to 2.3%"))
    print("  club -> elite is worth %+.1f s at identical average power" % tighten)
    print()

    # -- rig --------------------------------------------------------------
    print("rigging")
    best_rig, best_gain = "standard", 0.0
    for name in RIG_PATTERNS:
        v = steady_speed(build(power=scale, rate=rate, rig=name), guess=base)
        gain = -seconds(v, base)
        flag = ""
        if gain > best_gain + 1e-6 and name != "standard":
            best_rig, best_gain, flag = name, gain, "  <-"
        print("  %-17s %.4f m/s  %+6.1f s%s" % (name, v, gain, flag))
    rows.append(("re-rig standard -> %s" % best_rig, best_gain, "free",
                 "one afternoon with a spanner; permanent"))
    print()

    # -- one weaker rower, which side -------------------------------------
    print("a rower %.0f%% down, port seat vs starboard seat"
          % (100 * args.deficit))
    reference = build(power=scale, rate=rate)
    sides = [s.rigged_side for s in reference.rig.seats]
    port = next(i for i, s in enumerate(sides) if s > 0)
    stbd = next(i for i, s in enumerate(sides) if s < 0)
    costs = {}
    for label, index in (("port", port), ("starboard", stbd)):
        v = steady_speed(build(power=scale, rate=rate, weak=index,
                               deficit=args.deficit), guess=base)
        costs[label] = seconds(v, base)
        print("  %-10s %+6.1f s" % (label, costs[label]))
    side_gain = costs["starboard"] - costs["port"]
    rows.append(("weakest rower to port, not starboard", side_gain, "free",
                 "cancels the rig's standing bias instead of adding to it"))
    print("  putting them on port rather than starboard: %+.1f s" % side_gain)
    print()

    # -- pacing -----------------------------------------------------------
    print("pacing, %.0f%% out over race power" % (100 * args.overshoot))
    pace = pacing_cost(exponent, args.race_time, overshoot=args.overshoot)
    print("  a %.0f s race is rowed %.1f%% above CP, and the whole reserve "
          "lasts %.0f s" % (args.race_time, 100 * pace["race power over CP"],
                            pace["seconds above CP"]))
    print("  out hard then settling to exactly CP:   %+6.1f s"
          % pace["disciplined"])
    print("  out hard then blowing up 4%% below CP:   %+6.1f s" % pace["real"])
    rows.append(("pace it evenly rather than blowing up", pace["real"],
                 "free", "the loss is in falling below CP afterwards, not "
                 "in the reserve itself"))
    print()
    return rows, base, per_percent, exponent


def line_gain(reference_speed, iterations=45):
    """Seconds between an optimised line and a competently steered one.

    The honest comparison is not against the centreline -- nobody steers
    the centreline -- but against a crew that takes the right arches and
    holds a reasonable line through them.  ``arch_route`` is that crew.
    Scored at the masters reference speed, because extra metres cost
    ``ds / v`` and a slower boat pays more for them.
    """
    from coxswain.river import lines                      # noqa: E402
    from coxswain.river.route import Route, optimise_route  # noqa: E402
    from coxswain.river.trajectory import ReducedModel    # noqa: E402
    import racing_line as RL                              # noqa: E402

    raster, course, flow, gates = RL.build(month=10)
    ev = RL.evaluator(course, flow, raster, gates, ReducedModel())
    ev.reference_speed = reference_speed
    honest = lines.arch_route(course, raster, gates, margin=4.0)
    honest = lines.legalise(honest, course, raster, gates, margin=4.0)
    best = optimise_route(ev, n_control=13, iterations=iterations, seed=0,
                          initial=honest)
    route = Route(best.route.stations, best.route.offsets, name="optimised")
    a, b = ev.evaluate(honest), ev.evaluate(route)
    return (a.elapsed_clean + 60.0 * a.illegal_arches
            - b.elapsed_clean - 60.0 * b.illegal_arches,
            a.path_length, b.path_length)


def waterfall(rows, race_time, target, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, RULE = "#16211f", "#dce2e0"
    COLOUR = {"free": "#2a7f62", "training": "#1f5673", "power": "#c8901a"}
    rows = sorted(rows, key=lambda r: -r[1])
    figure, ax = plt.subplots(figsize=(9.0, 0.52 * len(rows) + 2.4))
    running, ticks, labels = race_time, [], []
    for i, (name, gain, kind, _note) in enumerate(rows):
        ax.barh(i, -gain, left=running, height=0.62,
                color=COLOUR.get(kind, "#5c6968"), edgecolor="none")
        ax.text(running - gain - 0.6, i, "%.0f s" % gain, va="center",
                ha="right", fontsize=8.5, color="white", weight="bold")
        running -= gain
        ticks.append(i)
        labels.append(name)
    ax.axvline(race_time, color=INK, lw=1.2)
    ax.axvline(race_time - target, color="#a2382a", lw=1.2, ls="--")
    ax.text(race_time - target, -1.1, "  %.0f s faster" % target,
            color="#a2382a", fontsize=9, va="top")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("race time, seconds")
    ax.set_title("Finding %.0f seconds, largest lever first" % target,
                 fontsize=11, color=INK, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (COLOUR["free"], COLOUR["training"], COLOUR["power"])]
    ax.legend(handles, ["free -- decide it once", "trainable by October",
                        "raw power"], frameon=False, fontsize=8.5,
              loc="lower left", ncol=3, bbox_to_anchor=(0.0, -0.30))
    figure.tight_layout()
    if not os.path.isdir(out):
        os.makedirs(out)
    path = os.path.join(out, "time_budget.png")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0,
                        help="last year's time, seconds (default 1140)")
    parser.add_argument("--target", type=float, default=60.0,
                        help="seconds needed (default 60)")
    parser.add_argument("--rate", type=float, default=28.0)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--deficit", type=float, default=0.15)
    parser.add_argument("--overshoot", type=float, default=0.08,
                        help="how far over race power the first minutes go")
    parser.add_argument("--mass-drops", type=float, nargs="+",
                        default=[1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--mass-plan", type=float, default=3.0,
                        help="which mass drop to carry into the budget")
    parser.add_argument("--no-line", action="store_true",
                        help="skip the route optimisation, which is the "
                             "slow half of this script")
    parser.add_argument("--out", default="out/budget")
    args = parser.parse_args(argv)

    rows, base, per_percent, exponent = measure(args)

    if not args.no_line:
        print("steering: optimised line vs a competently steered one ...")
        gain, honest_m, best_m = line_gain(base)
        print("  %.0f m -> %.0f m, %+.1f s\n" % (honest_m, best_m, gain))
        rows.append(("steer the optimised line", gain, "free",
                     "%.0f m less river, at the same speed"
                     % (honest_m - best_m)))

    found = sum(gain for _n, gain, _k, _note in rows)
    shortfall = args.target - found
    power_needed = max(shortfall, 0.0) / per_percent
    if shortfall > 0:
        rows.append(("crew power, +%.1f%%" % power_needed, shortfall, "power",
                     "%.1f s per 1%%; this is the balance after everything "
                     "else" % per_percent))

    print("=" * 68)
    print("THE BUDGET: %.0f s off %.0f s" % (args.target, args.race_time))
    print("=" * 68)
    print("  %-38s %8s  %s" % ("lever", "seconds", "kind"))
    for name, gain, kind, note in sorted(rows, key=lambda r: -r[1]):
        print("  %-38s %+8.1f  %s" % (name, gain, kind))
        print("  %-38s           %s" % ("", note))
    print()
    print("  everything but raw power:            %+8.1f s" % found)
    print("  raw power still needed:              %+8.1f s (%.1f%% fitter)"
          % (max(shortfall, 0.0), power_needed))
    print("  total:                               %+8.1f s -> %.0f s"
          % (max(found + max(shortfall, 0.0), found),
             args.race_time - max(found, args.target)))
    if shortfall <= 0:
        print("\n  the tactical levers alone cover the target.")
    path = waterfall(rows, args.race_time, args.target, args.out)
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
