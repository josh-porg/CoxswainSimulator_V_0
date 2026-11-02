"""The legs' timing scored against published on-water landmarks (SOURCES §214).

    .venv/Scripts/python.exe research/crew/leg_landmarks.py

No published on-water seat trace against time was found (§214). What is published are landmarks of the seat's timing:

* **Rowing Style Factor** (BioRow RBN 2026/03, elite single scullers on water, n = 30 rower-rates): the seat's
  displacement over the handle's during the first 20 % of the drive length, 80-82 % on water at every rate.
* **Catch Factor** (same): the time between the seat's and the handle's change of direction at the catch,
  +6 ms on water on average (the seat turns slightly after the handle; target zone for sculling -5 to -25 ms).
* **Legs down** (McBride & Elliott, ISBS; 16 male sweep rowers in pairs at 28 spm): after the finish the seat is
  still for 28.6 +- 5.6 % of the cycle.
* **Peak seat velocity in the drive** (Kleshnev 2007, ISBS; 294 crews, sculling): inside micro-phase D3, which spans
  20.2-38.2 % of drive time on average (D1 10.4, D2 9.8, D3 18.0 %).

Scored here: the ergometer keyframes' legs, [K05]'s leg clock (`--clocks k05_legs`), and both athletes' measured
seat channels. "Still" is read as moving less than 2 % of the seat's travel.

Outputs: data/local/crew/leg_landmarks.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]
import rung2_clock as C2                                         # noqa: E402

OUT = C2.OUT
STILL = 0.02            # fraction of the seat's travel


def landmarks(t, seat, handle, T, drive_time):
    """``t`` from the catch over one cycle; ``seat`` (m, + toward the bow) and ``handle`` path (m, + toward the bow)."""
    seat = seat - seat[0]
    travel = float(np.ptp(seat))
    hp = handle - handle[0]
    on = t <= drive_time
    length = float(hp[on].max())
    k20 = int(np.argmax(hp >= 0.2 * length))
    rsf = float(seat[k20] / hp[k20])
    # catch factor: direction change of the seat and of the handle around the catch (minima of position)
    window = (t > T - 0.25) | (t < 0.25)
    tt = np.where(t > T / 2, t - T, t)
    k_seat = int(np.argmin(np.where(window, seat, np.inf)))
    k_hand = int(np.argmin(np.where(window, hp, np.inf)))
    cf = float(1000.0 * (tt[k_seat] - tt[k_hand]))
    # legs down: from the finish (the handle's furthest) until the seat has returned STILL of its travel
    k_fin = int(np.argmax(hp))
    peak = float(seat[k_fin:].max())
    after = np.flatnonzero(seat[k_fin:] < peak - STILL * travel)
    legs_down = float((t[k_fin + after[0]] - t[k_fin]) / T) if after.size else float("nan")
    v = np.gradient(seat, t)
    k_pk = int(np.argmax(np.where(on, v, -np.inf)))
    return dict(rsf=100.0 * rsf, catch_factor_ms=cf, legs_down_pct=100.0 * legs_down,
                peak_seat_velocity_pct_drive=100.0 * float(t[k_pk] / drive_time), travel=travel)


def athlete(name):
    import measured_reach as MR
    t, phi, seat, _tp, _f, _d = MR.channels(name)            # catch to finish only
    # the full cycle from the catch: re-read the channels without the finish cut
    import like_for_like as L
    import validate_blade as V
    if name == "br24":
        D = np.genfromtxt(L.DATA, delimiter=",", names=True)[:-1]
        T = 60.0 / L.RATE
        t0 = np.arange(len(D)) * T / len(D)
        g = np.linspace(0.0, T, 2000, endpoint=False)
        ang = -np.interp(g, t0, 0.5 * (D["A1"] + D["A2"]), period=T)
        s = np.interp(g, t0, D["Ls"], period=T)
        r = 0.875
    else:
        T = V.CR06_T
        g = np.linspace(0.0, T, 2000, endpoint=False)
        ang = V._cr06_series("oar_angle_deg")(g)
        s = V._cr06_series("leg_disp_m")(g)
        r = V.CR06_FORCE_POINT
    i0 = int(np.argmax(ang))
    ang, s = np.roll(ang, -i0), np.roll(s, -i0)
    j = int(np.argmin(ang))
    s = s - s[0]
    if s[j] < 0:
        s = -s
    handle = r * (np.radians(ang[0]) - np.radians(ang))          # path along the arc, + toward the bow
    return landmarks(g, s, handle, T, g[j])


def model(clocks):
    m = C2.R.V.ATHLETES["br24"][0]()
    boat = C2.R.boat_for("br24", m, "on-water", drive_law=True, catch_posture="as-built", clocks=clocks)
    tm, sw = boat.timing, boat.oar_sweep
    T, D = float(tm.period), float(tm.drive_fraction) * float(tm.period)
    t = np.linspace(0.0, T, 2000, endpoint=False)
    rower = boat.crew[0].rower
    hip = np.asarray(rower._chain(t)["hip"][0].value, dtype=float)
    phi = np.asarray(sw(t, tm), dtype=float)
    r = float(boat.rig.seats[0].oarlocks[0].oar.inboard)
    return landmarks(t, hip - hip[0], r * (phi[0] - phi), T, D)


PUBLISHED = dict(rsf=(80.0, 82.0), catch_factor_ms=(6.0, 6.0), legs_down_pct=(28.6 - 5.6, 28.6 + 5.6),
                 peak_seat_velocity_pct_drive=(20.2, 38.2))


def main():
    rows = {"keyframes (research trunk)": model(()), "[K05] leg clock": model(("k05_legs",)),
            "[BR24] measured": athlete("br24"), "[CR06] measured": athlete("cr06")}
    print("%-28s %10s %14s %14s %18s" % ("", "RSF %", "catch factor", "legs down %", "peak Vseat % drive"))
    print("%-28s %10s %14s %14s %18s" % ("published", "80-82", "+6 ms", "23.0-34.2", "20.2-38.2"))
    for k, r in rows.items():
        print("%-28s %10.1f %11.0f ms %14.1f %18.1f" % (k, r["rsf"], r["catch_factor_ms"], r["legs_down_pct"],
                                                         r["peak_seat_velocity_pct_drive"]))
    json.dump(dict(rows=rows, published=PUBLISHED), open(os.path.join(OUT, "leg_landmarks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
