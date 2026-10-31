"""Hearing, seeing and a calling coxswain: three ways a crew can hold a common rhythm (SOURCES section 194).

    .venv/Scripts/python.exe research/crew/hearing_and_pacer.py

Section 22 gave the crew two channels: the shared hull (immediate, everyone) and a *sensory* channel modelled
as a delayed visual chain, each rower watching the one toward the stern. Sound was folded into that chain.
The rhythm literature says it should not be:

- beat-based timing is far stronger by ear than by eye; a heard rhythm even primes a seen one, not the
  reverse (Grahn, Henry & McAuley 2011, NeuroImage 54:1231); tapping to flashes is much less stable than
  to tones, and a heard sequence captures visuomotor timing (Repp & Penel 2002, JEP:HPP);
- sound is a *broadcast*: every seat hears the stroke's catch and the coxswain at once, whichever way it
  faces, whereas sight runs down the boat.

So `CoupledCrew` now has an auditory channel (broadcast, its own gain and delay) and an external pacer --
the coxswain calling the rhythm -- whose adaptivity runs from a metronome (holds the called rate) to a
follower (tracks the crew). This script asks three questions with the section 22 crew (detuning sd
0.10 rad/s, 3 % of rate; 18 s to settle; 24 random crews per cell):

  E1  how much does hearing the stroke add to, or replace, watching the seat ahead?
  E2  with a crew whose preferred rate is 3 % below the call, what does the pacer's adaptivity buy and cost?
  E3  the same with stroke-to-stroke timing noise, where joint-tapping work finds responsive partners help.

Spreads are peak-to-peak seat timing in ms at rate 32 (section 21: about 65 ms of port/starboard split
spends an eight's recovery balance authority).
"""
from __future__ import annotations

import json
import os

import numpy as np

from coxswain.boats import catalog
from coxswain.crew.synchronisation import (CoupledCrew, all_to_all_topology, broadcast_topology,
                                           stroke_chain_topology)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OMEGA = 2.0 * np.pi / catalog.eight(rate=32.0).timing.period
N_CREWS, SETTLE, MEASURE, DT = 24, 18.0, 6.0, 0.004


def run(seed, detuning_mean=0.0, noise=0.0, **kw):
    rng = np.random.default_rng(seed)
    crew = CoupledCrew(n_seats=8, detuning=detuning_mean + rng.normal(0.0, 0.10, 8), noise=noise, seed=seed,
                       **kw)
    t, spreads, asyn = 0.0, [], []
    phi0, th0 = None, None
    for i in range(int((SETTLE + MEASURE) / DT)):
        crew.step(t, DT, OMEGA)
        t += DT
        if t >= SETTLE:
            if phi0 is None:
                phi0, th0, t0 = crew.phi.copy(), crew.theta, t
            spreads.append(crew.phi.max() - crew.phi.min())
            if kw.get("pacer_gain", 0.0) > 0.0:
                mean = np.angle(np.mean(np.exp(1j * crew.phi)))
                asyn.append(np.angle(np.exp(1j * (mean - crew.theta))))
    rate = float((crew.phi - phi0).mean() / (t - t0))
    to_ms = 1000.0 / OMEGA
    out = dict(spread_ms=float(np.mean(spreads)) * to_ms, rate_error_pct=100.0 * (rate / OMEGA - 1.0))
    if asyn:
        a = np.array(asyn)
        out.update(pacer_lag_ms=float(np.mean(a)) * to_ms, pacer_sd_ms=float(np.std(a)) * to_ms)
    return out


def cell(**kw):
    rows = [run(s, **kw) for s in range(N_CREWS)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def practice():
    """The coxswain as practice describes it (user, 2026-10-29): rhythm speech follows the boat, the crew are told to
    listen to the sound of the strokes, and long drawn-out calls nudge now and then. A synced race recording agrees:
    rhythm words land just before the catch and counts just after the finish, both timed off the boat's own sound,
    and the calls do not predict the next stroke (research/coxing/rhythm_calls.py, SOURCES section 195).

    E4  "listen to the boat": the crew's coupling to its own catches (all-to-all hearing) raised from 0 to 4 on top
        of the visual chain, the following coxswain contributing nothing to timing (section 194, E2: a follower
        leaves the spread unchanged).
    E5  a nudge: for 3 strokes the coxswain's call holds a rate 4 % below the crew's (a metronome, adaptivity 0,
        gain 2), then returns to following. Crew rate before, during and after."""
    chain, field = stroke_chain_topology(8), all_to_all_topology(8)
    res = {"E4": {}, "E5": {}}
    print("\nE4  listen to the boat: attention to the crew's own catches (all-to-all hearing), timing noise 0.15")
    for g in (0.0, 0.5, 1.0, 2.0, 4.0):
        r = cell(sensory_gain=4.0, topology=chain, auditory_gain=g, auditory_topology=field, noise=0.15)
        res["E4"]["hearing %g" % g] = r
        print("  hearing gain %-4g spread %6.1f ms" % (g, r["spread_ms"]))
    print("\nE5  a 3-stroke nudge 4% below the crew's rate, then back to following")
    period = 2.0 * np.pi / OMEGA
    traces = []
    for seed in range(N_CREWS):
        rng = np.random.default_rng(seed)
        crew = CoupledCrew(n_seats=8, detuning=rng.normal(0.0, 0.10, 8), sensory_gain=4.0, topology=chain,
                           pacer_gain=2.0, pacer_adaptivity=16.0, seed=seed)
        t, rows, last = 0.0, [], None
        for i in range(int((SETTLE + 12 * period) / DT)):
            nudge = SETTLE <= t < SETTLE + 3 * period
            crew.pacer_adaptivity = 0.0 if nudge else 16.0
            crew.pacer_rate = 0.96 * OMEGA if nudge else None
            crew.step(t, DT, OMEGA)
            t += DT
            if t >= SETTLE - 3 * period and int((t - (SETTLE - 3 * period)) / period) != last:
                last = int((t - (SETTLE - 3 * period)) / period)
                rows.append(float(np.mean(crew.phi)))
        rate = np.diff(rows) / period / OMEGA - 1.0
        traces.append(100.0 * rate)
    tr = np.mean(traces, axis=0)
    res["E5"] = dict(rate_change_pct_by_stroke=[float(x) for x in tr], strokes_before=3, nudge_strokes=3)
    print("  crew rate vs called, per stroke (3 before, 3 nudged, then released): " +
          " ".join("%+.1f" % x for x in tr))
    return res


def main():
    res = {}
    chain, bcast, field = stroke_chain_topology(8), broadcast_topology(8), all_to_all_topology(8)
    print("E1  channels (sensory = visual chain; auditory = everyone hears the stroke's catch)")
    e1 = {}
    for name, kw in (("visual chain, gain 4", dict(sensory_gain=4.0, topology=chain)),
                     ("hearing the stroke only, gain 4", dict(sensory_gain=0.0, topology=chain, auditory_gain=4.0)),
                     ("visual 4 + hearing 1", dict(sensory_gain=4.0, topology=chain, auditory_gain=1.0)),
                     ("visual 4 + hearing 2", dict(sensory_gain=4.0, topology=chain, auditory_gain=2.0)),
                     ("visual 4 + hearing 4", dict(sensory_gain=4.0, topology=chain, auditory_gain=4.0)),
                     ("hearing all catches, gain 4", dict(sensory_gain=0.0, topology=chain, auditory_gain=4.0,
                                                          auditory_topology=field))):
        e1[name] = cell(**kw)
        print("  %-34s spread %6.1f ms" % (name, e1[name]["spread_ms"]))
    res["E1"] = e1

    for label, noise in (("E2", 0.0), ("E3", 0.15)):
        print("\n%s  pacer gain 2 over the visual chain; crew prefers 3%% below the call; timing noise %.2f rad/s^0.5"
              % (label, noise))
        rows = {}
        base = cell(sensory_gain=4.0, topology=chain, detuning_mean=-0.03 * OMEGA, noise=noise)
        rows["no calls"] = base
        print("  %-22s spread %6.1f ms   rate %+5.2f%%" % ("no calls", base["spread_ms"], base["rate_error_pct"]))
        for a in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            r = cell(sensory_gain=4.0, topology=chain, detuning_mean=-0.03 * OMEGA, noise=noise, pacer_gain=2.0,
                     pacer_adaptivity=a)
            rows["adaptivity %g" % a] = r
            print("  %-22s spread %6.1f ms   rate %+5.2f%%   crew-to-call lag %+6.1f ms (sd %5.1f)"
                  % ("adaptivity %g /s" % a, r["spread_ms"], r["rate_error_pct"], r["pacer_lag_ms"],
                     r["pacer_sd_ms"]))
        res[label] = rows
    res.update(practice())
    out = os.path.join(ROOT, "data", "local", "crew", "hearing_and_pacer.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("\nwritten", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
