"""NK LiNK export import (sprint 1 #8), on a synthetic snippet in the export's layout."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest import read_nk_export, write_boat_rows   # noqa: E402

SNIPPET = """Session Information:,,,,Device Information:,,,,Oarlock Information:
Name:,Test,,,Name:,CBGPS 0,,,Firmware:,---
Start Time:,09/20/2026 08:06:40,,,Model:,CBGPS,,,,

Per-Stroke Data:

Interval,Distance (GPS),Distance (IMP),Elapsed Time,Split (GPS),Speed (GPS),Split (IMP),Speed (IMP),Stroke Rate,Total Strokes,Distance/Stroke (GPS),Distance/Stroke (IMP),Heart Rate,Power,Catch,Slip,Finish
(Interval),(Meters),(Meters),(HH:MM:SS.tenths),(/500),(M/S),(/500),(M/S),(SPM),(Strokes),(Meters),(Meters),(BPM),(Watts),(Degrees),(Degrees),(Degrees)
1,7.0,0.0,00:00:02.2,00:02:21.2,3.54,00:00:00.0,0.00,26.5,1,7.0,0.0,---,---,---,---,---
1,14.1,0.0,00:00:04.4,00:02:20.0,,00:00:00.0,0.00,27.0,2,7.1,0.0,---,250,-58.0,9.0,35.0
"""


def test_nk_export_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "export.csv")
        open(path, "w", encoding="utf-8").write(SNIPPET)
        rows, meta = read_nk_export(path, offset=0.2)
        assert meta["Model"] == "CBGPS"
        assert [round(r["t"], 3) for r in rows] == [2.0, 4.2]
        assert rows[0]["speed"] == 3.54
        assert abs(rows[1]["speed"] - 500.0 / 140.0) < 1e-12      # from the split when speed is blank
        assert "power" not in rows[0] and rows[1]["power"] == 250.0   # '---' means absent
        assert rows[1]["catch_deg"] == -58.0
        write_boat_rows(tmp, "s1", rows)
        header = open(os.path.join(tmp, "sessions", "s1", "boat.csv"), encoding="utf-8").readline().strip()
        assert header.startswith("t,speed,rate") and "power" in header
