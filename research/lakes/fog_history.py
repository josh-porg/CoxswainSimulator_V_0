"""Morning fog history for a lake: observed visibility at the nearest airport and the forecast as issued.

    .venv/Scripts/python.exe research/lakes/fog_history.py data/local/<lake>/lake.json

* Observed: the airport named in lake.json "fog" (Renton for Sammamish, Bellingham for Whatcom), every routine
  report 2023 to now from the Iowa Environmental Mesonet ASOS archive: visibility (miles) and present weather
  (FG fog, BR mist).
* Forecast: Open-Meteo's historical forecast archive (gfs_seamless: HRRR then GFS, as the page uses) at the airport
  and at the lake's middle forecast point, hourly: visibility, temperature, dew point, humidity, low cloud, wind,
  weather code.
Saved as fog/obs_<airport>.csv and fog/forecast_<where>.json in the lake's folder.
"""
import json, os, sys, urllib.request
from datetime import date, timedelta

CFG_PATH = os.path.abspath(sys.argv[1]); CFG = json.load(open(CFG_PATH))
OUT = os.path.join(os.path.dirname(CFG_PATH), "fog")
UA = {"User-Agent": "CoxswainSimulator lake study (research)"}
VARS = ("visibility,temperature_2m,dew_point_2m,relative_humidity_2m,cloud_cover_low,cloud_cover,wind_speed_10m,weather_code,"
        "temperature_925hPa,temperature_850hPa,boundary_layer_height")       # the valley inversion that traps a cold pool


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180).read().decode("utf-8", "replace")


def main():
    os.makedirs(OUT, exist_ok=True)
    fog = CFG["fog"]
    today = date.today()
    end = today + timedelta(days=1)                          # the archive's end date is exclusive
    # the fog airport, plus any others: their cloud layers tell fog on the lake from a low overcast over everyone
    for ap in [fog["airport"]] + fog.get("cloud_airports", []):
        asos = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=%s&data=vsby&data=wxcodes&data=tmpf&data=dwpf&data=sknt"
                "&data=skyc1&data=skyl1&data=skyc2&data=skyl2"
                "&year1=2023&month1=1&day1=1&year2=%d&month2=%d&day2=%d&tz=America%%2FLos_Angeles&format=onlycomma&latlon=no"
                "&missing=M&trace=T&direct=no&report_type=3" % (ap[1:], end.year, end.month, end.day))
        txt = get(asos)
        open(os.path.join(OUT, "obs_%s.csv" % ap), "w").write(txt)
        print("%s: %d reports" % (ap, txt.count("\n") - 1))
    for where, (la, lo) in (("airport", fog["airport_latlon"]), ("lake", CFG["forecast_points"][1])):
        url = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f&start_date=2023-01-01&end_date=%s"
               "&hourly=%s&models=gfs_seamless&timezone=America%%2FLos_Angeles" % (la, lo, today.isoformat(), VARS))
        d = json.loads(get(url))
        json.dump(d["hourly"], open(os.path.join(OUT, "forecast_%s.json" % where), "w"))
        print("forecast at the %s: %d hours, visibility present %d" % (where, len(d["hourly"]["time"]),
              sum(v is not None for v in d["hourly"]["visibility"])))


if __name__ == "__main__":
    main()
