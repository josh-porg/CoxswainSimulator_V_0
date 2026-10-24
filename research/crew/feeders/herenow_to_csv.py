"""HereNow rows (event|place|bow|label|club|raw|+/-|adjusted, as extracted in the browser) -> the
regatta CSV format of data/local/20XX_<regatta>.csv. Crews are 'Club' + '(Cox, X.)'; scullers and
small boats are 'Name' + '(Club)': the club is whichever field is not a person's name in brackets."""
import csv
import re
import sys


def secs(t):
    t = t.strip()
    if not t or not re.match(r"^\d+:\d+(\.\d+)?$", t):
        return ""
    m, s = t.split(":")
    return "%.1f" % (int(m) * 60 + float(s))


def convert(src, dst):
    out = []
    for line in open(src, encoding="utf-8"):
        f = line.rstrip("\n").split("|")
        if len(f) < 8:
            continue
        ev, place, bow, label, paren, raw, adj, adjusted = f[:8]
        paren = paren.strip().strip("()")
        crew = "," in paren and re.match(r"^[^,]+, [A-Z]\.?$", paren.strip())
        club = label.strip() if crew else paren
        age = re.search(r"Age: (\d+)", adj)
        corr = re.search(r"Age: \d+ (-?[\d:.]+)", adj)
        c = ""
        if corr:
            v = corr.group(1).lstrip("-")
            c = str(float(v.split(":")[0]) * 60 + float(v.split(":")[1])) if ":" in v else v
        out.append(dict(event=ev, place=place, bow=bow, club=club, age=age.group(1) if age else "",
                        raw=raw, raw_seconds=secs(raw), correction_seconds=c,
                        adjusted=adjusted, adjusted_seconds=secs(adjusted)))
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("wrote", dst, len(out), "rows")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
