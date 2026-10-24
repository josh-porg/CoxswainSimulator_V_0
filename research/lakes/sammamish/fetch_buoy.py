"""King County's Lake Sammamish buoy: hourly meteorological history (wind on open water) for calibration.

    .venv/Scripts/python.exe research/lakes/sammamish/fetch_buoy.py [YEARS...]
    BUOY_TYPE=Hourly .venv/Scripts/python.exe research/lakes/sammamish/fetch_buoy.py [YEARS...]   # water, near the surface

The buoy (King County lake buoy programme, https://green2.kingcounty.gov/lake-buoy/) sits in the open water of the
southern half of the lake (47.5817 N, 122.0900 W) and reports hourly wind speed and direction, air temperature,
humidity and solar radiation. Its download page is an ASP.NET form: choose the type (Meteorological), which posts
back and fills in the years, then the buoy, year and months, and Download. This does those steps, one year at a
time, politely, and saves buoy/met_<year>.csv as served. Provisional data for the current year.
"""
import html, os, re, sys, time, urllib.parse, urllib.request
from http.cookiejar import CookieJar

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "sammamish"))   # its data and outputs (gitignored)
OUT = os.path.join(HERE, "buoy")
URL = "https://green2.kingcounty.gov/lake-buoy/Data.aspx"
P = "ctl00$kcMasterPagePlaceHolder$"
UA = {"User-Agent": "CoxswainSimulator lake study (research)"}
TYPE = os.environ.get("BUOY_TYPE", "Met")
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))


def hidden(page):
    return {m.group(1): html.unescape(m.group(2)) for m in re.finditer(r'<input type="hidden" name="(__[A-Z]+)" id="[^"]*" value="([^"]*)"', page)}


def post(fields):
    data = urllib.parse.urlencode(fields).encode()
    r = opener.open(urllib.request.Request(URL, data=data, headers=dict(UA, **{"Content-Type": "application/x-www-form-urlencoded"})), timeout=120)
    return r.read(), r.headers.get("Content-Type", ""), r.headers.get("Content-Disposition", "")


def main(years):
    os.makedirs(OUT, exist_ok=True)
    page = opener.open(urllib.request.Request(URL, headers=UA), timeout=60).read().decode("utf-8", "replace")
    f = hidden(page)
    f.update({"__EVENTTARGET": P + "c_TypeDropDownList", "__EVENTARGUMENT": "", P + "c_TypeDropDownList": TYPE, P + "c_BuoyDropDownList": "Sammamish"})
    body, _, _ = post(f)
    page = body.decode("utf-8", "replace")
    m = re.search(r'name="ctl00\$kcMasterPagePlaceHolder\$c_YearDropDownList".*?</select>', page, re.S)
    available = re.findall(r'value="(\d{4})"', m.group(0)) if m else []
    print("years offered:", " ".join(available))
    for y in (years or available):
        if y not in available:
            print(y, "not offered"); continue
        f = hidden(page)
        f.update({"__EVENTTARGET": "", "__EVENTARGUMENT": "", P + "c_TypeDropDownList": TYPE, P + "c_BuoyDropDownList": "Sammamish",
                  P + "c_YearDropDownList": y, P + "c_MonthStartDropDownList": "1", P + "c_EndMonthDropDownList": "12",
                  P + "c_DownloadButton": "Download"})
        body, ctype, disp = post(f)
        path = os.path.join(OUT, "%s_%s.csv" % (TYPE.lower(), y))
        if b"<html" in body[:500].lower():
            print(y, "returned a page, not a file (%s); skipped" % ctype); continue
        open(path, "wb").write(body)
        lines = body.decode("utf-8", "replace").splitlines()
        print(y, "%d lines, %s .. %s" % (len(lines), lines[1][:60] if len(lines) > 1 else "", lines[-1][:60] if lines else ""))
        time.sleep(3)


if __name__ == "__main__":
    main(sys.argv[1:])
