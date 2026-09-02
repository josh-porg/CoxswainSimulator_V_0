r"""Turn GPS tracks of the Charles into lines this project can study.

    python tools/ingest_gps.py --gpx "tracks/*.gpx" --out data/gps_tracks.json
    python tools/ingest_gps.py --osm --out data/gps_tracks.json

A GPX file is a list of latitudes. What a line study needs is
**(station, offset)**: how far down the course a point is, and how far off
the centreline. That projection is the whole job, and it needs the
surveyed channel this package already carries, which is why this lives
here rather than in a spreadsheet.

Once projected, a track is directly comparable with anything
:mod:`coxswain.river.route` produces -- the optimised line, the
centreline, another crew's track -- because they are all offsets against
the same station axis.

What it keeps
-------------
Everything that could later separate one kind of line from another:
year, month, elapsed time, mean and median speed, direction of travel,
point count, sample interval, how much of the track was actually on the
water, and any name or description the source supplied. Boat type and crew
are **not** inferable from GPS and are left as fields for the caller to
fill; guessing them from speed would manufacture data.

Direction matters more than it looks
------------------------------------
The Charles is rowed both ways every day and raced only one. A track that
runs downstream is a paddle home, not a race piece, and mixing the two
would average a racing line with a warm-up. Direction is therefore
computed and stored, and the default filter keeps only upstream tracks --
the direction the Head of the Charles runs.

Sources
-------
``--osm`` pulls public traces from the OpenStreetMap trackpoints API,
which is the only bulk source that is both free and unambiguously
redistributable. It is heavily rate limited and returns 429 from some
networks; the loop backs off and reports honestly rather than retrying
forever. Strava and Garmin hold far more rowing data but neither permits
bulk access to other people's activities, so the realistic path for those
is the user's own exports, which ``--gpx`` reads.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The rowing reach, as a lon/lat bounding box.
CHARLES_BBOX = (-71.135, 42.352, -71.105, 42.373)
OSM_TRACKPOINTS = "https://api.openstreetmap.org/api/0.6/trackpoints"
GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1",
          "gpx0": "http://www.topografix.com/GPX/1/0"}


# -- reading ------------------------------------------------------------
def parse_gpx(text: str):
    """Yield ``(name, points)`` per track segment; points are dicts."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return
    tag = root.tag
    namespace = tag[tag.find("{"):tag.find("}") + 1] if "{" in tag else ""

    def find(node, path):
        return node.findall(path.replace("gpx:", namespace))

    for track in find(root, ".//gpx:trk") or [root]:
        label = ""
        for name in find(track, "gpx:name"):
            label = (name.text or "").strip()
        for index, segment in enumerate(find(track, ".//gpx:trkseg")):
            points = []
            for node in find(segment, "gpx:trkpt"):
                try:
                    lat = float(node.get("lat"))
                    lon = float(node.get("lon"))
                except (TypeError, ValueError):
                    continue
                stamp = None
                for element in find(node, "gpx:time"):
                    stamp = (element.text or "").strip()
                points.append({"lat": lat, "lon": lon, "time": stamp})
            if len(points) >= 20:
                yield ("%s#%d" % (label, index) if label else "seg%d" % index,
                       points)


def read_files(patterns):
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            for name, points in parse_gpx(text):
                yield (os.path.basename(path), name, points)


def fetch_osm(bbox, pages: int, pause: float = 6.0):
    """Public traces in a bounding box, one page at a time.

    The API is strict.  Anything other than a 200 stops the loop and says
    so; hammering a 429 helps nobody and is rude to a volunteer-funded
    service.
    """
    import requests

    headers = {"User-Agent": "CoxswainSimulator/0.1 (research, "
                             "github.com/satur/CoxswainSimulator)"}
    for page in range(pages):
        url = "%s?bbox=%s&page=%d" % (
            OSM_TRACKPOINTS, ",".join("%.6f" % v for v in bbox), page)
        response = requests.get(url, timeout=90, headers=headers)
        if response.status_code != 200:
            print("  page %d: HTTP %d -- stopping"
                  % (page, response.status_code))
            if response.status_code == 429:
                print("  (rate limited; this API refuses some networks "
                      "outright)")
            return
        count = response.text.count("<trkpt")
        print("  page %d: %d trackpoints" % (page, count))
        if not count:
            return
        for name, points in parse_gpx(response.text):
            yield ("osm-page%d" % page, name, points)
        time.sleep(pause)


# -- projecting ---------------------------------------------------------
class Projector:
    """Lat/lon to ``(station, offset)`` on the surveyed Charles."""

    def __init__(self, course=None, channel=None):
        from coxswain.river import charles as charles_module

        self.charles = charles_module
        self.course = course or charles_module.charles_course()
        self.channel = channel or charles_module.charles_channel()
        self.origin = charles_module.CHARLES_ORIGIN
        station = np.linspace(0.0, self.course.length, 1200)
        self.station = station
        self.line = np.array([self.course.position_at(s) for s in station])

    def to_local(self, lat, lon):
        from coxswain.river.course import local_tangent_plane
        east, north = local_tangent_plane(np.asarray(lat), np.asarray(lon),
                                          self.origin)
        return np.asarray(east, float), np.asarray(north, float)

    def project(self, lat, lon):
        """Nearest station and signed offset for each point.

        Offset sign follows the course convention: **positive to port**,
        which is the same sign :class:`~coxswain.river.route.Route` uses,
        so a track and an optimised line can be plotted on one axis
        without a conversion nobody remembers to apply.
        """
        east, north = self.to_local(lat, lon)
        points = np.column_stack([east, north])
        # nearest sample on the centreline
        deltas = points[:, None, :] - self.line[None, :, :]
        distance = np.hypot(deltas[:, :, 0], deltas[:, :, 1])
        index = np.argmin(distance, axis=1)
        nearest = self.line[index]
        ahead = self.line[np.minimum(index + 1, len(self.line) - 1)]
        behind = self.line[np.maximum(index - 1, 0)]
        tangent = ahead - behind
        length = np.hypot(tangent[:, 0], tangent[:, 1])
        tangent = tangent / np.maximum(length, 1e-9)[:, None]
        gap = points - nearest
        # z of the cross product gives the side
        offset = tangent[:, 0] * gap[:, 1] - tangent[:, 1] * gap[:, 0]
        return self.station[index], offset, np.min(distance, axis=1)

    def on_water(self, lat, lon):
        east, north = self.to_local(lat, lon)
        keep = np.zeros(len(east), dtype=bool)
        for i, (x, y) in enumerate(zip(east, north)):
            try:
                row, column = self.channel.index_of(float(x), float(y))
                keep[i] = bool(self.channel.water[row, column])
            except (IndexError, ValueError):
                keep[i] = False
        return keep


# -- summarising --------------------------------------------------------
def parse_time(stamp):
    if not stamp:
        return None
    for form in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                 "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(stamp, form)
        except ValueError:
            continue
    return None


def summarise(source, name, points, projector, min_on_water=0.5,
              min_span=600.0):
    lat = np.array([p["lat"] for p in points])
    lon = np.array([p["lon"] for p in points])
    water = projector.on_water(lat, lon)
    fraction = float(water.mean())
    if fraction < min_on_water:
        return None
    lat, lon = lat[water], lon[water]
    if len(lat) < 20:
        return None

    station, offset, miss = projector.project(lat, lon)
    span = float(station.max() - station.min())
    if span < min_span:
        return None

    times = [parse_time(p["time"]) for p, keep in zip(points, water) if keep]
    stamps = [t for t in times if t is not None]
    elapsed = ((stamps[-1] - stamps[0]).total_seconds()
               if len(stamps) >= 2 else None)

    # Direction: does station rise with time?  The Charles is rowed both
    # ways daily and raced one way, and averaging the two would blend a
    # racing line with a paddle home.
    order = np.arange(len(station))
    slope = float(np.polyfit(order, station, 1)[0])
    upstream = slope > 0

    speed = None
    if elapsed and elapsed > 0:
        east, north = projector.to_local(lat, lon)
        step = np.hypot(np.diff(east), np.diff(north))
        speed = float(step.sum() / elapsed)

    return {
        "source": source,
        "name": name,
        "year": stamps[0].year if stamps else None,
        "month": stamps[0].month if stamps else None,
        "date": stamps[0].strftime("%Y-%m-%d") if stamps else None,
        "points": int(len(lat)),
        "fraction_on_water": round(fraction, 3),
        "station_span_m": round(span, 1),
        "station_min_m": round(float(station.min()), 1),
        "station_max_m": round(float(station.max()), 1),
        "elapsed_s": round(elapsed, 1) if elapsed else None,
        "mean_speed_ms": round(speed, 3) if speed else None,
        "sample_interval_s": (round(elapsed / max(len(lat) - 1, 1), 2)
                              if elapsed else None),
        "direction": "upstream" if upstream else "downstream",
        "median_abs_offset_m": round(float(np.median(np.abs(offset))), 2),
        "max_abs_offset_m": round(float(np.abs(offset).max()), 2),
        "median_centreline_miss_m": round(float(np.median(miss)), 2),
        # Not inferable from GPS.  Left for the caller rather than guessed.
        "boat_type": None, "crew": None, "event": None,
        "station": [round(float(s), 1) for s in station],
        "offset": [round(float(o), 2) for o in offset],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpx", nargs="*", default=[],
                        help="glob(s) of local GPX files")
    parser.add_argument("--osm", action="store_true",
                        help="also pull public OpenStreetMap traces")
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--out", default="data/gps_tracks.json")
    parser.add_argument("--keep-downstream", action="store_true",
                        help="keep paddles home as well as race-direction "
                             "pieces")
    args = parser.parse_args(argv)

    print("building the course projection")
    projector = Projector()

    incoming = []
    if args.gpx:
        incoming.append(read_files(args.gpx))
    if args.osm:
        print("fetching OpenStreetMap public traces")
        incoming.append(fetch_osm(CHARLES_BBOX, args.pages))
    if not incoming:
        parser.error("give --gpx and/or --osm")

    tracks, seen, rejected = [], 0, 0
    for stream in incoming:
        for source, name, points in stream:
            seen += 1
            record = summarise(source, name, points, projector)
            if record is None:
                rejected += 1
                continue
            if record["direction"] == "downstream" and not args.keep_downstream:
                rejected += 1
                continue
            tracks.append(record)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"tracks": tracks}, handle, separators=(",", ":"))

    print()
    print("saw %d segments, kept %d, rejected %d" % (seen, len(tracks),
                                                     rejected))
    if tracks:
        years = [t["year"] for t in tracks if t["year"]]
        spans = [t["station_span_m"] for t in tracks]
        print("  years %s" % (("%d-%d" % (min(years), max(years)))
                              if years else "unknown"))
        print("  median span %.0f m, longest %.0f m"
              % (float(np.median(spans)), max(spans)))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
