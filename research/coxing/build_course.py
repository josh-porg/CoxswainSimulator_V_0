"""Per-stroke course covariates for every session with a CoxBox GPS track: sessions/<id>/course.csv.

Columns: t, heading (deg true, from the positions one stroke either side), turn (|heading change|
in deg/s, 3-stroke median). Used by call_properties.prepare(course=True): the boat's own forecast
then also carries cos/sin of heading (a steady wind or stream, whatever its direction, without
weather data) and the turn rate (bends, arches, buoy lines). race_01 (video only) has no track.
"""
import glob, os, sys
import numpy as np
ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
import coxbox_data as C  # noqa: E402

SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
CB = {"practice_01": "20260925 0555AM", "practice_02": "20260925 0620AM", "control_01": "20261002 0525AM",
      "race_04": "20261004 0803AM", "race_05": "20261004 1237PM", "control_02": "20261007 0850AM",
      "practice_03": "20261007 0903AM", "practice_04": "20261007 0920AM", "practice_05": "20261007 0936AM",
      "practice_06": "20261007 0948AM", "practice_07": "20261007 1003AM", "practice_08": "20261007 1014AM",
      "race_06": "20261017 0737AM",
      # another coxswain's boat and CoxBox unit (a full file name): the second crew in the study
      "race_07": "Sharing file  CoxBox 011007363 20261017 0713AM.csv"}

for sid, cb in CB.items():
    name = cb if cb.endswith(".csv") else "Sharing file  CoxBox 011009704 %s.csv" % cb
    s = C.load_csv(os.path.join(ROOT, "data", "local", "coxbox", name))
    boat = np.genfromtxt(os.path.join(SDIR, sid, "boat.csv"), delimiter=",", names=True)
    assert len(boat["t"]) == len(s.t) and np.allclose(boat["t"], s.t), sid
    lat0 = np.radians(np.nanmean(s.lat))
    x = np.radians(s.lon - np.nanmean(s.lon)) * 6371000.0 * np.cos(lat0)
    y = np.radians(s.lat - np.nanmean(s.lat)) * 6371000.0
    n = len(x)
    i0, i1 = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)
    hd = np.degrees(np.arctan2(x[i1] - x[i0], y[i1] - y[i0])) % 360
    un = np.degrees(np.unwrap(np.radians(hd)))
    dt = np.maximum(s.t[i1] - s.t[i0], 0.5)
    turn = np.abs(un[i1] - un[i0]) / dt
    turn = np.array([np.median(turn[max(k - 1, 0):k + 2]) for k in range(n)])
    bad = ~np.isfinite(hd)
    with open(os.path.join(SDIR, sid, "course.csv"), "w") as f:
        f.write("t,heading,turn\n")
        for t, h, r in zip(s.t, hd, turn):
            f.write("%.2f,%.1f,%.3f\n" % (t, h, r))
    print("%-12s %4d strokes, heading spread %3.0f deg, turn median %.2f p95 %.2f deg/s%s" % (
        sid, n, np.ptp(un), np.median(turn), np.percentile(turn, 95), " (%d bad fixes)" % bad.sum() if bad.any() else ""))
