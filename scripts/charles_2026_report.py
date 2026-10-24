r"""The Charles in 2026: this year's line and the real trades, as one page (out/report/charles_2026.html).

    python scripts/charles_2026_report.py

Built for the analysis session of 15 Oct 2026 from the corrected models and this year's data:
``data/local/raceday/analysis_2026.json`` (where each crew's time goes; this year's optimal line),
the race cards (targets, chances, decisions, traffic, map), ``river_conditions.json`` (flow and level)
and ``wind_line_check.json``. Self-contained: no outside files.

Gusts: a switch at the top shows every wind-dependent number with or without the gust allowance
(``coxswain.raceday.gusts``, the forecast's peak gust at each start); the page carries both.
"""
from __future__ import annotations

import html
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(ROOT, "data", "local", "raceday")
OUT = os.path.join(ROOT, "out", "report", "charles_2026.html")
sys.path.insert(0, ROOT)
from coxswain.raceday import gusts as GU          # noqa: E402


def dual(steady, gusty):
    """Both versions of a number; the page's gust switch shows one."""
    return '<span class="sv">%s</span><span class="gv">%s</span>' % (steady, gusty)


def gusty_rows(rows, table, wind, gust):
    """The stretch rows with the wind part including the gust allowance (the table's per-stretch change)."""
    if not wind or not gust:
        return rows, None
    st, gu = GU.table_at(table, wind[0], wind[1]), GU.with_gusts(table, wind[0], wind[1], gust)
    ref = table["reference_s"]
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["wind_s"] += ref * (gu["factor"] * gu["share"][i] - st["factor"] * st["share"][i])
        out.append(r)
    return out, GU.allowance(table, wind[0], wind[1], gust)


def fmt(s):
    m, x = divmod(round(float(s), 1), 60)
    return "%d:%04.1f" % (m, x)


def sgn(x, d=1):
    return ("+" if x >= 0 else "−") + ("%." + str(d) + "f") % abs(x)


def map_svg(card, analysis, width=560):
    mp = card["map"]
    W, H = mp["width"], mp["height"]
    pts = lambda a: " ".join("%d,%d" % (p[0], H - p[1]) for p in a)        # noqa: E731
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Map of the course with the planned line">' % (W, H)]
    for b in mp["banks"]:
        out.append('<polyline class="bank" points="%s"/>' % pts(b))
    for p in mp["pushes"]:
        out.append('<polyline class="pushl" points="%s"/>' % pts(p))
    out.append('<polyline class="line" points="%s"/>' % pts(mp["line"]))
    for b in mp["bridges"]:
        out.append('<line class="bridge" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (b["a"][0], H - b["a"][1], b["b"][0], H - b["b"][1]))
    for m in mp["marks"]:
        if m["name"] in ("Powerhouse", "Riverside BC", "Mile 1", "Mile 2", "Big turn", "Eliot Bridge", "Finish", "Weeks Footbridge"):
            out.append('<circle class="mark" cx="%d" cy="%d" r="20"/>' % (m["xy"][0], H - m["xy"][1]))
            out.append('<text class="ml" x="%d" y="%d">%s</text>' % (m["xy"][0] + 36, H - m["xy"][1] + 22,
                                                                     html.escape(m["name"].replace(" Footbridge", "").replace(" Bridge", ""))))
    out.append('<circle class="st" cx="%d" cy="%d" r="28"/>' % (mp["start"][0], H - mp["start"][1]))
    out.append("</svg>")
    return "".join(out)


def decomposition_svg(rows):
    """Stacked bars per stretch: depth, current, wind, turning seconds."""
    keys = [("depth_s", "Depth", "c1"), ("current_s", "Current", "c2"), ("wind_s", "Wind", "c3"), ("turning_s", "Turning", "c4")]
    n = len(rows)
    W, rowh, left, right = 760, 30, 230, 70
    H = n * rowh + 40
    top = max(sum(max(r[k], 0) for k, _, _ in keys) for r in rows)
    lo = min(sum(min(r[k], 0) for k, _, _ in keys) for r in rows)
    span = (top - lo) or 1.0
    sx = lambda v: left + (v - lo) / span * (W - left - right)        # noqa: E731
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Seconds lost on each stretch to depth, current, wind and turning">' % (W, H)]
    out.append('<line class="axis" x1="%.1f" y1="10" x2="%.1f" y2="%d"/>' % (sx(0), sx(0), H - 24))
    for i, r in enumerate(rows):
        y = 14 + i * rowh
        out.append('<text class="lab" x="%d" y="%d" text-anchor="end">%s → %s</text>'
                   % (left - 10, y + 15, html.escape(r["start"].replace(" Footbridge", "")), html.escape(r["end"].replace(" Footbridge", ""))))
        pos, neg = 0.0, 0.0
        for k, _, cls in keys:
            v = r[k]
            if v >= 0:
                x0, x1 = sx(pos), sx(pos + v); pos += v
            else:
                x0, x1 = sx(neg + v), sx(neg); neg += v
            if abs(x1 - x0) > 0.5:
                out.append('<rect class="%s" x="%.1f" y="%d" width="%.1f" height="%d" rx="2"/>' % (cls, x0, y + 3, x1 - x0, rowh - 10))
        tot = sum(r[k] for k, _, _ in keys)
        out.append('<text class="val" x="%.1f" y="%d">%s s</text>' % (sx(pos) + 6, y + 15, sgn(tot)))
    out.append('<text class="lab" x="%.1f" y="%d" text-anchor="middle">0 s</text>' % (sx(0), H - 6))
    out.append("</svg>")
    return "".join(out)


def arch_row():
    """River Street and Western Avenue: the two practised lines (Cambridge or centre arches, centre at Weeks),
    each optimised for the crew, timed on Saturday (data/local/raceday/fields_<crew>.json)."""
    d = {}
    for crew in ("four", "eight"):
        p = os.path.join(RD, "fields_%s.json" % crew)
        lines = json.load(open(p))["lines"] if os.path.exists(p) else []
        alt = next((L for L in lines if L["name"].startswith("Other practised line")), None)
        if alt:
            own_cam = "Cambridge" in lines[0]["name"]
            d[crew] = alt["delta_s"] if own_cam else -alt["delta_s"]          # centre minus Cambridge
    if len(d) < 2:
        return ("River Street and Western Avenue arches (Cambridge or centre)", "under 2 s", "Inside the model's precision: traffic decides.")
    return ("River Street and Western Avenue: centre arches against Cambridge (Weeks centre in both)",
            "%+.1f / %+.1f s" % (d["four"], d["eight"]),
            "Both lines practised. A few seconds at most, near the model's precision: take the one traffic leaves open. "
            "Negative = centre quicker.")


def trades(an, cards, gwind):
    """The ranked trades table, from this run's numbers (four / eight); ``gwind`` the wind totals with gusts."""
    f, e = an["four"]["totals"], an["eight"]["totals"]
    st4 = next(r for r in an["four"]["rows"] if r["start"] == "Start")
    weeks = {c: next(d for d in cards[c]["decisions"] if d["bridge"] == "Weeks Footbridge") for c in cards}
    rows = [
        ("Wind on Saturday (every crew pays it)", dual("%+.0f / %+.0f s" % (f["wind_s"], e["wind_s"]),
                                                       "%+.0f / %+.0f s" % (gwind["four"], gwind["eight"])),
         "%.0f s of the four's is the first kilometre, straight into the north-wester. Judge splits against the card." % st4["wind_s"]),
        ("Current after the rain (every crew pays it)", "%+.0f / %+.0f s" % (f["current_s"], e["current_s"]),
         "Against slack water; about 570 cfs forecast. In the targets; no line escapes it."),
        ("Shallow water", "%+.0f / %+.0f s" % (f["depth_s"], e["depth_s"]),
         "The faster boat pays twice as much: the eight sits nearer the depth drag rise. Push in the shallows."),
        ("Weeks: the Cambridge arch", "%+.0f / %+.0f s" % (weeks["four"]["price_s"][0], weeks["eight"]["price_s"][0]),
         "The one arch choice that matters: hold the centre, or ease."),
        ("Turning on the planned line", "%+.0f / %+.0f s" % (f["turning_s"], e["turning_s"]),
         "A late hard turn costs several times an even one (loss goes with the turn rate squared)."),
        ("Basin level across its normal range", "±3 / ±6 s", "Near normal this week: about 0."),
        ("Re-steering the line for Saturday", "%.1f / %.1f s" % (an["four"]["gain_s"], an["eight"]["gain_s"]),
         "The line moves a metre or so. Keep the planned line."),
        arch_row(),
    ]
    return "".join("<tr><td>%s</td><td class=n>%s</td><td>%s</td></tr>"
                   % (html.escape(a), b if b.startswith("<span") else html.escape(b), html.escape(c)) for a, b, c in rows)


def gust_text(gnote):
    parts = []
    for crew, name in (("four", "the four"), ("eight", "the eight")):
        gu = gnote.get(crew)
        if gu and gu["gust_ms"]:
            parts.append("%s: gusts to %.0f mph, %+.0f s" % (name, gu["gust_ms"] / 0.44704, gu["added_s"]))
    if not parts:
        return "No gusts in the forecast for the starts."
    return ("Every wind number below can include the forecast gusts or not. Counting them adds the extra drag of "
            "the strong moments (the air's drag grows with the square of the wind, so gusts cost more than lulls "
            "give back): " + "; ".join(parts) + " at our pace. Not in it: the steering corrections gusts "
            "call for. Every crew pays it, so the chances do not change.")


def main():
    an = json.load(open(os.path.join(RD, "analysis_2026.json")))
    cards = {c: json.load(open(os.path.join(RD, "card_%s.json" % c))) for c in ("four", "eight")}
    names = {"four": "Women's 60+ four (bow 21)", "eight": "Women's 50+ eight (bow 7)"}
    blocks, gwind, gnote = [], {}, {}
    for crew in ("four", "eight"):
        a, card = an[crew], cards[crew]
        t = a["totals"]
        table = json.load(open(os.path.join(RD, "wind_table_%s.json" % crew)))
        cond = card.get("conditions") or {}
        # the card's gust factor (gust / sustained) on the analysis's wind, so a newer forecast's gusts
        # are not set against an older forecast's mean
        g_ms = (cond["gust"] / cond["speed"] * a["wind"][0]) if a["wind"] and cond.get("gust") and cond.get("speed") else None
        grows, gu = gusty_rows(a["rows"], table, a["wind"], g_ms)
        if gu:
            gu["gust_ms"] = cond["gust"]                  # quote the forecast's own gust
        k = gu["ratio"] if gu else 1.0
        gwind[crew] = sum(r["wind_s"] for r in grows)
        gnote[crew] = gu
        both = lambda x: dual(fmt(x), fmt(x * k))         # noqa: E731
        goals = "".join("<tr><td>%s</td><td class=n>%s</td><td class=n>%s</td><td class=n>%s</td></tr>" % (
            html.escape(g["name"]), both(g["today_central"]), both(card["short_card"]["targets"][i]["today_central"]),
            ("%d%%" % round(100 * g["chance"])) if g.get("chance") is not None else "place %d" % round(g.get("median_place", 0)))
            for i, g in enumerate(card["targets"]))
        decs = "".join("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (html.escape(d["bridge"]), html.escape(d["verdict"]),
                                                                       html.escape(d["advice"])) for d in card["decisions"])
        tr = card["traffic"]
        blocks.append("""
<section class="crew">
<h2>%s</h2>
<div class="grid2">
<div class="panel"><h3>Where the time goes on Saturday</h3>
<p class="lead">Against still air, slack water and deep water, on the planned line at requalifying pace:</p>
<div class="stats">
<div><b class="n">%s s</b><span>current (%.0f cfs forecast)</span></div>
<div><b class="n">%s s</b><span>wind (%s)</span></div>
<div><b class="n">%s s</b><span>shallow water</span></div>
<div><b class="n">%s s</b><span>turning</span></div>
</div>
<p class="note">Re-optimising the line for Saturday's wind and current gains <b>%.1f s</b>; it moves %.1f m on average (most %s m at %.1f km). Keep the planned line.</p>
</div>
<div class="panel"><h3>The line</h3>%s<p class="note">Orange: planned line. Shaded: push stretches (the shallows).</p></div>
</div>
<div class="panel"><h3>Stretch by stretch</h3>%s
<p class="legend"><i class="c1"></i>Depth <i class="c2"></i>Current <i class="c3"></i>Wind <i class="c4"></i>Turning</p></div>
<div class="grid2">
<div class="panel"><h3>Targets today</h3><table><tr><th>Goal</th><th>Full</th><th>Short</th><th>Chance</th></tr>%s</table>
<p class="note">Full 4.82 km; short 3.73 km from Riverside (high-wind course). Today = forecast wind, river and the field's 2026 entries.</p></div>
<div class="panel"><h3>Traffic</h3><p>Expect to pass <b>%.1f</b> and be passed by <b>%.1f</b> crews; clear race %d%%.</p>
<p class="note">Start interval 15 s; bows from the published start order.</p></div>
</div>
<div class="panel"><h3>Arches</h3><table class="dec"><tr><th>Bridge</th><th>Verdict</th><th>Why</th></tr>%s</table></div>
</section>""" % (names[crew], sgn(t["current_s"], 0), card["river"].get("discharge_cfs", 0),
                 dual(sgn(t["wind_s"], 0), sgn(gwind[crew], 0)),
                 ("%.0f mph from %03.0f°" % (a["wind"][0] / 0.44704, a["wind"][1]) if a["wind"] else "calm")
                 + (", gusts %.0f mph" % (gu["gust_ms"] / 0.44704) if gu and gu["gust_ms"] else ""),
                 sgn(t["depth_s"], 0), sgn(t["turning_s"], 0), a["gain_s"], a["mean_shift_m"], sgn(a["max_shift_m"], 0),
                 a["max_shift_at_m"] / 1000, map_svg(card, a),
                 '<div class="sv">%s</div><div class="gv">%s</div>' % (decomposition_svg(a["rows"]), decomposition_svg(grows)), goals,
                 tr["expected_passes_made"], tr["expected_passed_by"], round(100 * tr["p_clear_race"]), decs))
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Charles in 2026</title>
<style>
:root{--paper:#f3f6f5;--card:#fff;--ink:#13222b;--muted:#5a6a73;--line:#d5dde0;--accent:#d9532b;--c1:#2f6f9f;--c2:#7a9a3a;--c3:#d9532b;--c4:#8a6bb0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#0d161b;--card:#15222a;--ink:#e5eef2;--muted:#93a5ae;--line:#26363f;--accent:#ff7b52;--c1:#6fa8d6;--c2:#a6c46a;--c3:#ff7b52;--c4:#b69ad8}}
:root[data-theme="dark"]{--paper:#0d161b;--card:#15222a;--ink:#e5eef2;--muted:#93a5ae;--line:#26363f;--accent:#ff7b52;--c1:#6fa8d6;--c2:#a6c46a;--c3:#ff7b52;--c4:#b69ad8}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px 60px;display:flex;flex-direction:column;gap:18px}
h1{font-size:34px;line-height:1.1;margin:0;text-wrap:balance}h2{font-size:24px;margin:8px 0 0}h3{font-size:14px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.lead{color:var(--muted);margin:0 0 10px}.note{font-size:14px;color:var(--muted);margin:8px 0 0}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;overflow-x:auto}
.crew{display:flex;flex-direction:column;gap:14px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.stats b{display:block;font-size:26px}.stats span{font-size:13px;color:var(--muted)}
.n{font-variant-numeric:tabular-nums}table{border-collapse:collapse;width:100%%;font-size:14px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}td.n{text-align:right}
svg{width:100%%;height:auto;display:block}.bank{fill:none;stroke:var(--muted);stroke-width:9}.line{fill:none;stroke:var(--accent);stroke-width:16}
.pushl{fill:none;stroke:var(--accent);stroke-opacity:.3;stroke-width:70;stroke-linecap:round}.bridge{stroke:var(--ink);stroke-width:20;opacity:.5}
.mark{fill:var(--card);stroke:var(--ink);stroke-width:8}.st{fill:#2b7a55}.ml{fill:var(--ink);font:600 64px system-ui,sans-serif}
.axis{stroke:var(--muted);stroke-width:1}.lab,.val{fill:var(--ink);font:12px system-ui,sans-serif}.val{fill:var(--muted)}
.c1{fill:var(--c1);background:var(--c1)}.c2{fill:var(--c2);background:var(--c2)}.c3{fill:var(--c3);background:var(--c3)}.c4{fill:var(--c4);background:var(--c4)}
.legend{font-size:13px;color:var(--muted)}.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 4px 0 12px;vertical-align:-1px}
.trades td:nth-child(2){white-space:nowrap}
body:not(.g) .gv,body.g .sv{display:none}
.seg{display:inline-grid;grid-template-columns:1fr 1fr;gap:4px;background:var(--line);border-radius:10px;padding:4px}
.seg button{font:inherit;font-weight:600;font-size:14px;border:0;border-radius:7px;padding:7px 12px;background:transparent;color:var(--muted);cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--card);color:var(--ink)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style></head><body><div class="wrap">
<header><p class="note">Head of the Charles · Saturday 17 October 2026 · analysis for the coxswains, %s</p>
<h1>The Charles in 2026: this year's line and the real trades</h1>
<p class="lead">Saturday's forecast: north-west wind around 14 mph at the starts; the river rising to about 570 cfs after rain (five times the October median); the basin at its normal level. Corrected models throughout (wind against each boat's own resistance, turning cost measured on our boats).</p></header>
<div class="panel"><h3>Gusts</h3>
<div class="seg" role="group" aria-label="Gusts"><button type="button" data-g="off" aria-pressed="true">Steady wind</button><button type="button" data-g="on" aria-pressed="false">Count gusts</button></div>
<p class="note">%s</p></div>
<div class="panel"><h3>The trades, ranked</h3>
<table class="trades"><tr><th>Decision or factor</th><th>Four / eight</th><th>What it means</th></tr>
%s</table></div>
%s
<p class="note">Sources: research/raceday/analysis_2026.py (this page's numbers), river_conditions.py, the race cards (scripts/race_card.py), docs/RACEDAY.md. The earlier reports in out/report (September) predate this week's corrections and should be read with them.</p>
</div>
<script>
(function(){
  var KEY = "charles-2026-report:gusts", on = false;
  try { on = localStorage.getItem(KEY) === "on"; } catch (e) {}
  function set(v){ on = v; document.body.classList.toggle("g", on);
    document.querySelectorAll(".seg button").forEach(function(b){ b.setAttribute("aria-pressed", (b.getAttribute("data-g") === "on") === on ? "true" : "false"); });
    try { localStorage.setItem(KEY, on ? "on" : "off"); } catch (e) {} }
  document.querySelectorAll(".seg button").forEach(function(b){ b.addEventListener("click", function(){ set(b.getAttribute("data-g") === "on"); }); });
  set(on);
})();
</script>
</body></html>""" % (cards["four"]["generated_at"][:10], gust_text(gnote), trades(an, cards, gwind), "".join(blocks))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(page)
    print("written", OUT, "%.0f kB" % (os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    main()
