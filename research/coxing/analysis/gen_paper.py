import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import numpy as np

z = np.load("irf.npz")
L = z["lags"].astype(int)
CI = json.load(open("irf_ci.json"))
SER = [("split", "Split stated", "tactical"),
       ("in-two / take-N", "In two / take a N", "tactical"),
       ("steering", "Steering", "tactical"),
       ("ratio/length", "Ratio / length", "technical"),
       ("catch/press", "Catch / press", "technical"),
       ("body/together", "Body / together", "technical"),
       ("motivational", "Motivational", "motivational")]
YMAX = 0.008
W, H, PL, PB = 232, 118, 34, 24


def panel(key, label, theme):
    b = z[key]
    sx = lambda i: PL + (W - PL - 8) * i / (len(L) - 1)
    sy = lambda val: (H - PB) / 2 - (val / YMAX) * ((H - PB) / 2 - 8)
    pts = " ".join("%.1f,%.1f" % (sx(i), sy(v)) for i, v in enumerate(b))
    zero = sy(0)
    col = "var(--amber)" if theme == "technical" else "var(--teal)"
    ci = CI[key]
    star = "&#8225;" if (ci["alo"] > 0 or ci["ahi"] < 0) else ""
    return (
        '<figure class="sp">\n'
        '<svg viewBox="0 0 %d %d" role="img" aria-label="%s impulse response">\n' % (W, H, label) +
        '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="var(--rule)" stroke-width="1"/>\n' % (PL, zero, W - 8, zero) +
        '<polyline points="%s" fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>\n' % (pts, col) +
        '<text x="%d" y="%.1f" class="ax" text-anchor="end">+8</text>\n' % (PL - 6, sy(YMAX) + 4) +
        '<text x="%d" y="%.1f" class="ax" text-anchor="end">0</text>\n' % (PL - 6, zero + 4) +
        '<text x="%d" y="%.1f" class="ax" text-anchor="end">&#8722;8</text>\n' % (PL - 6, sy(-YMAX) + 4) +
        '<text x="%d" y="%d" class="ax">0</text>\n' % (PL, H - 8) +
        '<text x="%.1f" y="%d" class="ax" text-anchor="middle">10</text>\n' % (sx(10), H - 8) +
        '<text x="%d" y="%d" class="ax" text-anchor="end">25</text>\n' % (W - 8, H - 8) +
        '</svg>\n'
        '<figcaption><span class="sp-t">%s%s</span><span class="sp-n">peak %+.1f at lag %d</span></figcaption>\n'
        '</figure>' % (label, star, ci["peak"] * 1000, int(ci["lag"])))


panels = "\n".join(panel(k, lab, th) for k, lab, th in SER)

rows = []
for k, lab, th in SER:
    c = CI[k]
    cls = "pos" if c["peak"] > 0 else "neg"
    dag = "&#8225;" if (c["alo"] > 0 or c["ahi"] < 0) else ""
    rows.append(
        "<tr><td>%s</td><td class=\"%s\">%+.1f</td><td>%d</td><td>[%+.1f, %+.1f]</td>"
        "<td>%+.1f [%+.1f, %+.1f]%s</td></tr>"
        % (lab, cls, c["peak"] * 1000, int(c["lag"]), c["lo"] * 1000, c["hi"] * 1000,
           c["area"] * 1000, c["alo"] * 1000, c["ahi"] * 1000, dag))
irf_rows = "\n".join(rows)

HTML = open("paper_template.html", encoding="utf-8").read()
HTML = HTML.replace("{{PANELS}}", panels).replace("{{IRFROWS}}", irf_rows)
open("paper.html", "w", encoding="utf-8").write(HTML)
print("wrote paper.html  %d bytes" % len(HTML))
