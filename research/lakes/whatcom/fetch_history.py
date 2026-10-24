"""History for calibrating the Lake Whatcom model: the lakeside CWOP stations and the archived forecast.

    .venv/Scripts/python.exe research/lakes/whatcom/fetch_history.py

* Stations: findu.com's weather table (CWOP via APRS), about two months back, every ~15 min:
  time (UTC), wind direction, speed and gust (mph). Saved as history/<call>.csv (merged with what is
  there, so re-running extends the record).
* Forecast: Open-Meteo's historical forecast archive (the forecast as issued, gfs_seamless: HRRR then GFS,
  what the live page uses) at each station, hourly 10 m wind, saved as history/forecast_<call>.csv.
Public data of other people's stations: kept in data/local; the page names them by place only.
"""
import csv, json, os, re, time, urllib.request
from datetime import datetime, timezone

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "whatcom"))   # its data and outputs (gitignored)
OUT = os.path.join(HERE, "history")
UA = {"User-Agent": "CoxswainSimulator lake study (research)"}
STATIONS = {"DW0884": (48.76116, -122.36255), "FW1160": (48.75083, -122.3545),
            "FW6124": (48.745, -122.3955), "EW7158": (48.7035, -122.32317)}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode("utf-8", "replace")


def findu(call, hours=2400):
    html = get("http://www.findu.com/cgi-bin/wx.cgi?call=%s&last=%d&units=english" % (call, hours))
    rows = {}
    for line in re.sub(r"<[^>]+>", " ", html).splitlines():
        f = line.split()
        if len(f) >= 5 and re.fullmatch(r"\d{14}", f[0]):
            try:
                rows[f[0]] = dict(t=f[0], temp_f=f[1], dir=f[2], mph=f[3], gust_mph=f[4])
            except IndexError:
                pass
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    for call, (lat, lon) in STATIONS.items():
        path = os.path.join(OUT, "%s.csv" % call)
        old = {r["t"]: r for r in csv.DictReader(open(path))} if os.path.exists(path) else {}
        new = findu(call)
        old.update(new)
        rows = [old[k] for k in sorted(old)]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["t", "temp_f", "dir", "mph", "gust_mph"]); w.writeheader(); w.writerows(rows)
        print("%s: %d readings (%d new from findu), %s to %s" % (call, len(rows), len(new), rows[0]["t"], rows[-1]["t"]))
        time.sleep(2)
        start = datetime.strptime(rows[0]["t"], "%Y%m%d%H%M%S").date().isoformat()
        end = datetime.now(timezone.utc).date().isoformat()
        url = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=%.5f&longitude=%.5f&start_date=%s&end_date=%s"
               "&hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m&models=gfs_seamless&wind_speed_unit=ms&timezone=GMT"
               % (lat, lon, start, end))
        d = json.loads(get(url))["hourly"]
        with open(os.path.join(OUT, "forecast_%s.csv" % call), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t", "speed_ms", "dir", "gust_ms"])
            for t, s, dd, g in zip(d["time"], d["wind_speed_10m"], d["wind_direction_10m"], d["wind_gusts_10m"]):
                w.writerow([t, s, dd, g])
        print("   forecast archive: %d hours" % len(d["time"]))


if __name__ == "__main__":
    main()
