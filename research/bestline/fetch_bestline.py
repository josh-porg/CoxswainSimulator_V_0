"""Save the HOCR 2026 tracks published on bestline.row3r.com (a row3r.com project) for our line studies.

    .venv/Scripts/python.exe research/bestline/fetch_bestline.py

The site ranks crews by how far they rowed beyond its shortest legal line ("% over best line"); it does
not publish the uploaded .fit files. What each entry's public map page carries, and what is kept here:
the leaderboard row (crew label as typed, boat class, % over, distance, extra metres, mean distance off
the line, course time) and the rowed track as (lat, lon, signed metres off the site's best line), plus the
site's best line itself. No timestamps per point, so no speeds along the course: course time only.

Other people's data: kept in data/local (never committed); any published use is aggregate and unnamed.
One page every two seconds; the index is re-read each run and only new entries are fetched.
"""
import json, os, re, time, urllib.request

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "bestline"))   # its data and outputs (gitignored)
BASE = "https://bestline.row3r.com"
UA = {"User-Agent": "CoxswainSimulator line study (research; contact via row3r.com)"}


def get(path):
    return urllib.request.urlopen(urllib.request.Request(BASE + path, headers=UA), timeout=60).read().decode("utf-8")


def rows(html):
    out = []
    for m in re.finditer(r'<tr class="row[^"]*">(.*?)</tr>', html, re.S):
        cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", m.group(1), re.S)]
        link = re.search(r'href="(/map/[0-9a-f]+)"', m.group(1))
        cells = [re.sub(r"\s+", " ", c) for c in cells]
        out.append(dict(cells=cells, map=link.group(1) if link else None))
    return out


def arr(html, name):
    m = re.search(r"const %s = (\[.*?\]);" % name, html, re.S)
    return json.loads(m.group(1)) if m else []


def main():
    index = get("/")
    best_m = re.search(r"Best line for this course:\s*<[^>]*>?\s*([\d,]+)\s*m", index) or re.search(r"Best line for this course:\s*([\d,]+)", index)
    path = os.path.join(HERE, "bestline_2026.json")
    data = json.load(open(path)) if os.path.exists(path) else dict(entries={})
    data.update(source=BASE, read_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                best_line_m=float(best_m.group(1).replace(",", "")) if best_m else None)
    for r in rows(index):
        if not r["map"]:
            continue
        key = r["map"].rsplit("/", 1)[1]
        entry = data["entries"].get(key, {})
        entry["row"] = r["cells"]
        if "rowed" not in entry:
            page = get(r["map"])
            entry["rowed"] = arr(page, "rowed")
            if "best" not in data:
                data["best"] = arr(page, "best")
            time.sleep(2.0)
        data["entries"][key] = entry
        print(key, r["cells"][1:8], len(entry["rowed"]), "points")
    json.dump(data, open(path, "w"))
    print("saved %d entries; best line %s m, %d points" % (len(data["entries"]), data.get("best_line_m"), len(data.get("best", []))))


if __name__ == "__main__":
    main()
