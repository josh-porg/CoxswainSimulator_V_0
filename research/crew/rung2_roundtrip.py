"""Rung 2's round trip on an athlete's own stroke: is a stall the dynamics or the muscle table? (SOURCES §225)

rung2_population drives each athlete's clock with the POPULATION's muscle force (its rung-1 inverse dynamics, scaled to
the athlete's power and mapped by drive progress). Here the athlete's own rung-1 run gives Q_ref(tau) by inverse
dynamics on its last stroke (rung2_clock's first check), and the free clock is driven by exactly that. A consistent
model returns tau' = 1 and rung 1's stroke; a run that still stalls has dynamics that cannot hold even the stroke they
were built from, while one that returns has a table that does not fit the athlete.

    python research/crew/rung2_roundtrip.py --athlete br24 [--bending --bend-dynamics --added-mass patton]
        [--catch buried|entry] [--fv 4.53] [--hold-own] [--strokes 10] [--dt 0.00625]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]
import rung2_population as RP                                    # noqa: E402

C2, V = RP.C2, RP.V
#: the last round trip's own muscle table, (cycle phase, Q)
ROUND = {}


def roundtrip(name, strokes):
    m = V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    body = RP.own_body(name, boat) if RP.BODY == "own" else None
    ref = RP.make(boat, "reference", body=body)
    run_r = ref.run_strokes(strokes, surge_speed=4.3)
    s_r = C2.summary(run_r)
    ph, q = RP.q_table(ref, run_r)
    ROUND["table"] = (ph, q)
    P = float(boat.timing.period)

    def Q(tau):
        return float(np.interp(np.mod(tau / P, 1.0), ph, q, period=1.0))
    free_boat = C2.R.boat_for(name, m, "on-water", drive_law=True)
    free = RP.make(free_boat, "free", Q=Q, body=RP.own_body(name, free_boat) if RP.BODY == "own" else None)
    free.start_state = np.concatenate([run_r.last_states[:, -1], [P, 1.0]])
    run_f = free.run_strokes(strokes, surge_speed=4.3)
    s_f = C2.summary(run_f)
    tl = np.array([r for r in free.tlog if r[0] >= float(run_f.last_time[0])])
    tl = tl[np.argsort(tl[:, 0])]
    return s_r, s_f, (q.min(), q.max()), tl, float(run_f.last_time[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athlete", default="br24", choices=["br24", "cr06"])
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--catch", default="buried", choices=["buried", "entry"])
    ap.add_argument("--fv", type=float, default=4.53)
    ap.add_argument("--hold-own", action="store_true")
    ap.add_argument("--bending", action="store_true")
    ap.add_argument("--bend-dynamics", action="store_true")
    ap.add_argument("--added-mass", default="none", choices=["none", "patton", "labbe"])
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--compare-population", action="store_true",
                    help="also print the population's muscle force, as rung 2 hands it over, against the own one")
    a = ap.parse_args()
    if a.dt:
        import coxswain.core.integrators as I
        I.estimate_step = lambda *args, **kw: float(a.dt)
    from coxswain.crew.handle_fv import HandleForceVelocity
    RP.FV = HandleForceVelocity(a.fv, "--fv") if a.fv else None
    RP.HOLD_OWN = bool(a.hold_own)
    RP.BLADE, RP.BODY, RP.BENDING, RP.ADDED_MASS = "crossflow_normal", "own", bool(a.bending), a.added_mass
    if a.bend_dynamics:
        from coxswain.crew.oar_bending import BendDynamics
        RP.BEND_DYNAMICS = BendDynamics()
    C2.TAKEOVER_DEG = {"buried": C2.BIOROW_CATCH_TO_BURIED_DEG, "entry": "entry"}[a.catch]
    print("%s round trip: %s, %s catch, added mass %s, F-V %s%s" % (
        a.athlete, "dynamic bend" if a.bend_dynamics else ("quasi-static bend" if a.bending else "rigid"),
        a.catch, a.added_mass, a.fv, ", own pace" if a.hold_own else ""), flush=True)
    s_r, s_f, qr, tl, t0 = roundtrip(a.athlete, a.strokes)
    print("rung 1 (own stroke imposed): %.3f m/s, IVV %.1f%%, %.0f W; Q_ref %.0f to %.0f"
          % (s_r["speed"], 100 * s_r["ivv"], s_r["power"], qr[0], qr[1]))
    print("rung 2 on its own Q_ref:     %.3f m/s, IVV %.1f%%, %.0f W; tau' %.3f to %.3f"
          % (s_f["speed"], 100 * s_f["ivv"], s_f["power"], tl[:, 2].min(), tl[:, 2].max()))
    print("  last stroke, tau' every 0.04 s:", " ".join("%.2f" % float(np.interp(x, tl[:, 0] - t0, tl[:, 2]))
                                                      for x in np.arange(0.0, 1.0, 0.04)))
    if a.compare_population:
        # the population's muscle force as rung2_population.athlete hands it to this athlete, against the athlete's own
        m = V.ATHLETES[a.athlete][0]()
        boat = C2.R.boat_for(a.athlete, m, "on-water", drive_law=True)
        P = float(boat.timing.period)
        D = float(boat.timing.drive_fraction) * P
        s_p, (s_pop, q_drive, _ph, _q) = RP.population_tables(a.strokes)
        k = m["power"] / s_p["power"]
        own_ph, own_q = ROUND["table"]
        print("\ndrive progress | own Q_ref | population x %.3f | ratio" % k)
        for tau in np.linspace(0.0, D, 21):
            s = float(RP.progress_of(boat, tau)[0])
            own = float(np.interp(tau / P, own_ph, own_q, period=1.0))
            pop = k * float(np.interp(s, s_pop, q_drive))
            print("  t %.2f s  s %.2f | %8.0f | %8.0f | %6.2f" % (tau, s, own, pop, pop / own if abs(own) > 1 else np.nan))


if __name__ == "__main__":
    main()
