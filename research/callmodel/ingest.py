"""Build session folders from raw sources.

    python ingest.py captions --raw <dir of .vtt> --index channel_index.txt --root <root>
        one call-only session per transcript; ids are generic (transcript_01, ...) in date
        order, and the video id stays in meta.json for provenance

    python ingest.py session --root <root> --id race_04 --vtt x.vtt --boat log.csv \
        [--offset 1.0] [--t0 100 --t1 900] [--kind race] [--boat-class 4+]
        one synchronised session; boat.csv needs columns t and speed and/or rate

`--offset` is the number of seconds to add to a caption time to reach the boat log's
clock. Everything is written on the caption clock, so the boat times are shifted by
minus the offset. Sessions with a transcript and a boat log need nothing else to enter
the model: run.py picks up every folder under <root>/sessions.
"""
import argparse
import csv
import json
import os
import re

from captions import parse

_CLASS = re.compile(r"\b(8\+|8|4\+|4x|4-|2x|2-|1x)(?=\s|$|\b)", re.I)


def write_session(root, sid, meta, words=None, boat=None, labels=None):
    """words: [(t, w)]; boat: [(t, speed, rate)] with None for missing values."""
    d = os.path.join(root, "sessions", sid)
    os.makedirs(d, exist_ok=True)
    meta = dict(meta, id=sid)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), indent=1)
    if words is not None:
        with open(os.path.join(d, "words.jsonl"), "w", encoding="utf-8") as f:
            for t, w in words:
                f.write(json.dumps({"t": round(float(t), 3), "w": w}) + "\n")
    if boat is not None:
        with open(os.path.join(d, "boat.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "speed", "rate"])
            for t, v, r in boat:
                wr.writerow([round(float(t), 3),
                             "" if v is None or v != v else round(float(v), 4),
                             "" if r is None or r != r else round(float(r), 3)])
    if labels is not None:
        json.dump(labels, open(os.path.join(d, "labels.json"), "w", encoding="utf-8"), indent=1)
    return d


def read_index(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("|", 3)
        if len(parts) == 4:
            vid, dur, date, title = parts
            rows.append(dict(video=vid, duration=float(dur), date=date, title=title))
    return rows


def _guess(title):
    m = _CLASS.search(title.replace("'", ""))
    kind = "practice" if re.search(r"practi[cs]e|training|steady", title, re.I) else "race"
    return kind, (m.group(1) if m else "")


def from_captions(raw, index, root, exclude=(), prefix="transcript"):
    rows = sorted((r for r in read_index(index) if r["video"] not in set(exclude)),
                  key=lambda r: (r["date"], r["video"]))
    made = []
    for i, r in enumerate(rows, 1):
        p = os.path.join(raw, r["video"] + ".en.vtt")
        if not os.path.exists(p):
            continue
        words = parse(p)
        if len(words) < 20:
            continue
        kind, cls = _guess(r["title"])
        sid = "%s_%02d" % (prefix, i)
        write_session(root, sid, dict(title=r["title"], kind=kind, boat_class=cls,
                                      date=r["date"], source="youtube:" + r["video"]),
                      words=words)
        made.append((sid, len(words)))
    return made


def _clock(text):
    """'HH:MM:SS.t' or 'MM:SS.t' -> seconds."""
    parts = [float(x) for x in text.strip().split(":")]
    sec = 0.0
    for x in parts:
        sec = sec * 60.0 + x
    return sec


#: NK LiNK per-stroke columns kept when they carry values (Empower oarlock fields are '---'
#: without one). Keys are the export's headers; values the session's column names.
NK_EXTRAS = {"Distance (GPS)": "distance", "Distance/Stroke (GPS)": "dps", "Heart Rate": "heart_rate",
             "Power": "power", "Catch": "catch_deg", "Slip": "slip_deg", "Finish": "finish_deg",
             "Wash": "wash_deg", "Force Avg": "force_avg", "Work": "work", "Force Max": "force_max",
             "Max Force Angle": "max_force_angle", "GPS Lat.": "lat", "GPS Lon.": "lon"}


def read_nk_export(path, offset=0.0):
    """Per-stroke rows from an NK LiNK export (CoxBox Core, SpeedCoach, CBGPS).

    Returns ``(rows, meta)``: rows are dicts with ``t`` (s, minus ``offset``), ``speed`` (m/s,
    from the speed column or, failing that, the split), ``rate`` (spm), and whichever of
    :data:`NK_EXTRAS` carry numbers. ``meta`` holds the session header fields.
    """
    lines = open(path, encoding="utf-8-sig").read().splitlines()
    meta = {}
    for line in lines[:12]:
        cells = [c.strip() for c in line.split(",")]
        for k, v in zip(cells[0::4], cells[1::4]):
            if k.endswith(":") and v:
                meta[k[:-1]] = v
    try:
        i = next(n for n, l in enumerate(lines) if l.startswith("Per-Stroke Data:"))
    except StopIteration:
        raise ValueError("%s has no 'Per-Stroke Data:' section; not an NK LiNK export" % path)
    header = next(n for n in range(i + 1, len(lines)) if lines[n].startswith("Interval"))
    rows = []
    for r in csv.DictReader(lines[header:]):
        if not r.get("Elapsed Time") or r["Elapsed Time"].startswith("("):
            continue
        num = lambda k: (float(r[k]) if r.get(k) not in (None, "", "---") else None)
        speed = num("Speed (GPS)") or num("Speed (IMP)")
        if not speed:
            split = r.get("Split (GPS)") or r.get("Split (IMP)")
            speed = 500.0 / _clock(split) if split and _clock(split) > 0 else None
        row = dict(t=_clock(r["Elapsed Time"]) - offset, speed=speed, rate=num("Stroke Rate"))
        for key, name in NK_EXTRAS.items():
            value = num(key) if key in r else None
            if value is not None:
                row[name] = value
        rows.append(row)
    return rows, meta


def write_boat_rows(root, sid, rows):
    """Write per-stroke rows as the session's boat.csv, t/speed/rate first, extras after."""
    d = os.path.join(root, "sessions", sid)
    os.makedirs(d, exist_ok=True)
    extras = [k for k in NK_EXTRAS.values() if any(k in r for r in rows)]
    with open(os.path.join(d, "boat.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "speed", "rate"] + extras)
        for r in rows:
            w.writerow([round(r["t"], 3), "" if r["speed"] is None else round(r["speed"], 4),
                        "" if r["rate"] is None else r["rate"]] + [r.get(k, "") for k in extras])


def read_boat_csv(path, offset=0.0):
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        g = lambda k: float(r[k]) if r.get(k) not in (None, "", "nan") else None
        out.append((float(r["t"]) - offset, g("speed"), g("rate")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("captions")
    a.add_argument("--raw", required=True)
    a.add_argument("--index", required=True)
    a.add_argument("--root", required=True)
    a.add_argument("--exclude", nargs="*", default=[], help="video ids to skip")
    b = sub.add_parser("session")
    b.add_argument("--root", required=True)
    b.add_argument("--id", required=True)
    b.add_argument("--vtt")
    b.add_argument("--boat", help="a t,speed,rate CSV")
    b.add_argument("--export", help="an NK LiNK export (CoxBox Core, SpeedCoach); keeps oarlock fields")
    b.add_argument("--offset", type=float, default=0.0)
    b.add_argument("--t0", type=float)
    b.add_argument("--t1", type=float)
    b.add_argument("--kind", default="race")
    b.add_argument("--boat-class", default="")
    b.add_argument("--source", default="")
    args = ap.parse_args()

    if args.cmd == "captions":
        for sid, n in from_captions(args.raw, args.index, args.root, args.exclude):
            print("%-16s %6d words" % (sid, n))
    else:
        meta = dict(kind=args.kind, boat_class=args.boat_class, source=args.source)
        if args.t0 is not None:
            meta["t0"] = args.t0
        if args.t1 is not None:
            meta["t1"] = args.t1
        words = parse(args.vtt) if args.vtt else None
        boat = read_boat_csv(args.boat, args.offset) if args.boat else None
        print(write_session(args.root, args.id, meta, words, boat))
        if args.export:
            rows, info = read_nk_export(args.export, args.offset)
            write_boat_rows(args.root, args.id, rows)
            print("%d strokes from %s (%s)" % (len(rows), os.path.basename(args.export),
                                               info.get("Model", "NK LiNK")))


if __name__ == "__main__":
    main()
