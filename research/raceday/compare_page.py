"""The club lines comparison page (compare.html beside this file), anonymised by default.

    .venv/Scripts/python.exe research/raceday/line_compare.py     # the timings
    .venv/Scripts/python.exe research/raceday/compare_page.py     # this page
    .venv-tools/Scripts/python.exe research/raceday/compare_site.py   # encrypt into site/ (own password)

Blind by default so the lines can be judged without knowing whose is whose: every load deals the lines
fresh random letters (A, B, ...), and names, events, boats, places, official times and speeds stay hidden
until "Show names" is switched on (never remembered between loads). Times are shown as differences
only; the settings are named by the same letters. Styles from race_review_template.html.
"""
import json, os, re

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)


def main():
    L = json.load(open(os.path.join(HERE, "line_compare.json")))
    crews = {k: dict(cox=v["cox"], boat=v["boat"], event=v["event"], event_name=v["event_name"], bow=v["bow"], place=v["place"],
                     field=v["field"], official=round(v["official"], 1), speed=round(v["speed"], 2),
                     pins={g: a for g, a in v["pins"].items() if g in ("River Street", "Western Avenue", "Weeks Footbridge")})
             for k, v in L["crews"].items()}
    timed = {s: {ln: dict(total=round(v["total"], 2), distance=round(v["distance"], 2), depth=round(v["depth"], 2),
                          wind=round(v["wind"], 2), current=round(v["current"], 2), turning=round(v["turning"], 2),
                          path=round(v["path_m"]), sec=v["sections"]) for ln, v in res.items()} for s, res in L["timed"].items()}
    data = dict(crews=crews, timed=timed, sections=[dict(name="%s → %s" % (s["start"], s["end"]), m=round(s["to_m"] - s["from_m"]))
                                                     for s in L["sections"]],
                profile=L["profile"], off=L["offsets_by_stretch"])
    sp = os.path.join(HERE, "line_smoothness.json")
    if os.path.exists(sp):                                    # intended line vs steering (line_smoothness.py)
        M = json.load(open(sp))
        cut = str(M["main_cutoff"])
        steer = {}
        for k, c in M["crews"].items():
            if k not in crews:
                continue
            steer[k] = dict(wobble={str(x): c["wobble_deg"]["wavelet_%s" % x] for x in M["cutoffs"]},
                            lateral=c["lateral_rms_m"]["wavelet_" + cut], agree=c["methods_agree"]["position_rms_m"],
                            bands=[b["heading_rms_deg"] for b in c["bands"][:5]], glitches=len(c["removed"]),
                            cost={s_: round(M["timed"][s_][k]["raw"] - M["timed"][s_][k]["intended"]["wavelet_" + cut], 2) for s_ in M["timed"]},
                            intended={s_: round(M["timed"][s_][k]["intended"]["wavelet_" + cut], 2) for s_ in M["timed"]},
                            own=c.get("own"))
        data["steer"] = dict(crews=steer, cutoff=M["main_cutoff"], cutoffs=M["cutoffs"], corr=M["check"]["inter_crew_fast_heading_correlation"],
                             by_setting=M.get("by_setting"),
                             band_labels=["< %d m" % c_["to_m"] if not c_["from_m"] else "%d–%d m" % (c_["from_m"], c_["to_m"]) for c_ in M["crews"][next(iter(M["crews"]))]["bands"][:5]])
    wp, wc = os.path.join(HERE, "wind_perturbation.json"), os.path.join(HERE, "wind_cross_profiles.json")
    if os.path.exists(wp) and os.path.exists(wc):            # wind and gusts (wind_perturbation.py)
        Wd = json.load(open(wp)); Xc = json.load(open(wc))
        data["wind"] = dict(crews={k: v for k, v in Wd["crews"].items() if k in crews}, segments=Wd["pooled"]["segments_wobble"],
                            cross=Xc, offsets=list(range(-60, 61, 10)))
    sk = os.path.join(HERE, "skill_separation.json")
    if os.path.exists(sk):                                   # skill with externals removed (skill_separation.py)
        Sk = json.load(open(sk))
        data["skill"] = {kk: v for kk, v in Sk.items() if kk != "segments"}
    wr = os.path.join(HERE, "window_robustness.json")
    if os.path.exists(wr):                                   # sliding windows and leave-one-out (window_robustness.py)
        data["windows"] = json.load(open(wr))
        for v in data["windows"]["leave_one_out"].values():
            v["blocks_n"] = len(data["windows"]["leave_one_out"]["line_own"]["blocks"])
    ad = os.path.join(HERE, "adaptation.json")
    if os.path.exists(ad):                                   # Monte Carlo of nominal runs (adaptation.py)
        A = json.load(open(ad))
        for v in A["realised"].values():
            v.pop("segments", None); v.pop("stations", None)
        data["adapt"] = {kk: A[kk] for kk in ("lam", "n_runs", "n_boot", "ref_time_s", "r2_within", "conditions", "pooled", "slopes", "nominal",
                                              "hist_edges", "beats", "realised", "conditions_summary")}
    op = os.path.join(HERE, "operator.json")
    if os.path.exists(op):                                   # human-operator models (coxswain_operator.py)
        O = json.load(open(op))
        for v in O["crews"].values():
            v["A"].pop("profile", None)
            # what the correction costs, from the full-resolution step response: back within 1 m, overshoot, and the
            # seconds it loses (turning: speed lost = k x turn rate^2, the four's k; distance: the weave's extra path)
            s_ = np.array(v["A"]["step"]); t_, y_, psi_ = s_[:, 0], s_[:, 1], s_[:, 2]; dt_ = t_[1] - t_[0]
            om_ = np.gradient(psi_, t_)
            far = np.flatnonzero(np.abs(y_) > 1.0)
            settle = float(t_[far[-1] + 1]) if len(far) and far[-1] + 1 < len(t_) else (0.0 if not len(far) else None)
            v["A"]["step_metrics"] = dict(settle_s=settle, overshoot_m=round(float(max(0.0, -y_.min())), 2),
                                          cost_s=round(float(np.sum(0.0071 * om_ ** 2 * dt_) + np.sum(0.5 * np.radians(psi_) ** 2 * dt_)), 3),
                                          peak_turn=round(float(np.max(np.abs(om_))), 2))
            v["A"]["step"] = [r for i, r in enumerate(v["A"]["step"]) if i % 2 == 0]
        data["operator"] = O["crews"]
    rm = os.path.join(HERE, "river_map.json")
    if os.path.exists(rm):                                   # course geometry for the map (river_map.py)
        data["map"] = json.load(open(rm))
    data, names = anonymise(data, crews)
    json.dump(names, open(os.path.join(HERE, "compare_names.json"), "w"))
    tpl = open(os.path.join(HERE, "race_review_template.html"), encoding="utf-8").read()
    head = tpl.split("<style>")[0].replace("<title>HOCR 2026 Race Review</title>", "<title>Club lines compared</title>")
    style = re.search(r"<style>(.*?)</style>", tpl, re.S).group(1)
    page = head + "<style>" + style + EXTRA_CSS + "</style>\n" + MARKUP + "\n<script>\nconst D = " + json.dumps(data, separators=(",", ":")) + ";\n" + SCRIPT + "</script>\n"
    open(os.path.join(HERE, "compare.html"), "w", encoding="utf-8").write(page)
    print("written compare.html (%.0f kB, %d crews)" % (len(page) / 1024, len(crews)))


IDENTIFYING = ("cox", "boat", "event", "event_name", "bow", "place", "field", "official", "speed", "window_et", "rudder_k", "boat_length",
               "gps_s", "start_et")
NO_RENAME = ("coefficients", "covariates", "segments", "segments_wobble")


def ids_for(keys):
    """Opaque, stable ids for the crews (random once, kept in compare_ids.json; the mapping never leaves data/local)."""
    import secrets
    p = os.path.join(HERE, "compare_ids.json")
    ids = json.load(open(p)) if os.path.exists(p) else {}
    for k in keys:
        if k not in ids:
            while True:
                v = "x" + secrets.token_hex(3)
                if v not in ids.values():
                    break
            ids[k] = v
    json.dump(ids, open(p, "w"))
    return ids


def anonymise(data, crews):
    """Split the page data: analysis numbers keyed by opaque ids (opened with the comparison password), and the
    identity of each id - coxswain, boat, event, bow, place, time, speed, race window (opened with a second one)."""
    keys = list(crews)
    ids = ids_for(keys)
    names = dict(crews={ids[k]: {f: crews[k][f] for f in ("cox", "boat", "event_name", "bow", "place", "field", "official", "speed")} for k in keys})
    for k in keys:
        w = data.get("wind", {}).get("crews", {}).get(k)
        if w:
            names["crews"][ids[k]]["window_et"] = w.get("window_et")

    def walk(o):
        if isinstance(o, dict):
            out = {}
            for kk, v in o.items():
                if kk in IDENTIFYING:
                    continue
                nk = ids.get(kk, kk)
                if isinstance(kk, str) and kk.startswith("best:") and kk[5:] in ids:
                    nk = "best:" + ids[kk[5:]]
                out[nk] = v if kk in NO_RENAME else walk(v)
            return out
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str) and o in ids:
            return ids[o]
        return o
    return walk(data), names


EXTRA_CSS = """
[hidden]{display:none !important}
:root{--c1:#d1495b;--c2:#0e6b7d;--c3:#c58a00;--c4:#5b4fb3;--c5:#2e8b57;--c6:#8c564b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--c1:#ff6f7f;--c2:#4fc0d4;--c3:#f0b429;--c4:#9d92ff;--c5:#5fc08a;--c6:#c49a8c}}
:root[data-theme="dark"]{--c1:#ff6f7f;--c2:#4fc0d4;--c3:#f0b429;--c4:#9d92ff;--c5:#5fc08a;--c6:#c49a8c}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:12px}
.switch{display:flex;align-items:center;gap:10px;font-size:14px;color:var(--muted)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:baseline}
td.best{font-weight:600}
td.zero{color:var(--gain);font-weight:600}
.blind{background:var(--river);border-radius:10px;padding:10px 14px;font-size:14px}
.toggles{display:flex;flex-wrap:wrap;gap:14px;font-size:14px;color:var(--muted)}
.toggles label{display:flex;gap:6px;align-items:center;cursor:pointer}
td:first-child{white-space:nowrap}
table.heat td.r{text-align:center;font-variant-numeric:tabular-nums;min-width:2.1em;padding-left:4px;padding-right:4px}
table.heat td.bad{outline:2px solid var(--loss);outline-offset:-3px;font-weight:600}
table.heat th.w{text-align:center;font-variant-numeric:tabular-nums;padding-left:4px;padding-right:4px}
#rivermap{width:100%;height:auto;display:block}
#rivermap text{font-size:12px;fill:var(--ink)}
#rivermap .tick{fill:var(--muted);font-size:11px}
.focus{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;font-size:14px;margin-top:8px}
.focus ol{margin:4px 0 0;padding-left:20px;display:grid;gap:3px}
.scale{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted)}
.scale i{display:inline-block;width:120px;height:10px;border-radius:5px;background:linear-gradient(90deg,var(--gain),color-mix(in srgb,var(--gain) 50%,var(--loss)),var(--loss))}
ul.loo{margin:6px 0 0;padding-left:20px;font-size:14px;display:grid;gap:4px}
"""

MARKUP = """
<div class="wrap">
<header class="top">
  <div style="display:flex;flex-direction:column;gap:8px">
    <div class="eyebrow">Head of the Charles · Saturday 17 October 2026 · club coxswains</div>
    <h1>Club lines compared</h1>
    <p class="muted">Each coxswain's line from the CoxBox GPS, timed in every crew's conditions, so the only thing that changes is the line.</p>
  </div>
  <div class="switch"><span id="modeword">Names hidden</span><button id="names" type="button" aria-pressed="false">Show names</button></div>
  <form id="unlock" hidden autocomplete="off" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;width:100%;justify-content:flex-end">
    <input id="npass" type="password" aria-label="Names password" placeholder="names password" style="font:inherit;font-size:16px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)">
    <button type="submit" id="ngo">Show names</button><button type="button" id="ncancel">Cancel</button>
    <span id="nerr" class="muted" role="status" style="width:100%;text-align:right"></span>
  </form>
</header>

<p class="blind" id="blindnote">Blind mode: the lines carry letters dealt at random each time the page opens, and names, events, boats, places and race times are hidden. Judge the lines first, then show the names.</p>

<section>
  <h2>The lines, timed in every setting</h2>
  <p id="lead"></p>
  <div class="panel"><div class="tablewrap"><table id="matrix"></table></div>
  <p class="note">Seconds behind the fastest club line in that setting (0.0 = fastest). A setting is one crew's boat, pace and conditions on the day: its wind, the river flow and level, and its turning cost. The model reproduces each crew's own GPS race time on its own line. Negative values in the last two rows mean faster than every club line.</p></div>
</section>

<section>
  <h2>Where the lines run</h2>
  <div class="panel">
    <div class="toggles"><label><input type="checkbox" id="showbest"> best lines</label><label><input type="checkbox" id="showshort"> shortest legal line</label></div>
    <svg id="prof" role="img" aria-label="Offset from the course centreline along the course for each line"></svg>
    <div class="legend" id="legend"></div>
    <p class="note">Distance from a smoothed course centreline (the extracted centreline's 150&ndash;200 m bank wiggles removed, so no line shows kinks that are only the chart's), Boston side up, Cambridge side down; the shaded band is the water. Bridges are marked.</p>
  </div>
</section>

<section>
  <h2>Stretch by stretch</h2>
  <div class="panel">
    <div class="toggles"><label>Setting <select id="setsel" aria-label="Setting for the stretch table"></select></label></div>
    <div class="tablewrap"><table id="stretch"></table></div>
    <p class="note">Each cell: seconds behind the fastest club line on that stretch, and the line's median offset from the centreline (negative = Cambridge side). Course stretches are the race card's.</p>
  </div>
</section>

<section>
  <h2>Against the best line</h2>
  <p>The best line is the solution of an optimal-control problem, solved by Radau collocation: the boat's curvature is the control, the race time is minimised through each crew's own arches in that crew's setting, with the measured turning loss for its boat, so the line is as smooth as a boat can steer it (a point every 5 m). It is a guide, not a target: buoy lines are not in the model (it keeps 6 m inside the water's edge), and its shallow-water saving near the banks is an upper bound that the GPS has not yet confirmed.</p>
  <div class="panel"><div class="tablewrap"><table id="bestt"></table></div></div>
</section>

<section id="steersec" hidden>
  <h2>Intended line and steering</h2>
  <p>Each track split by length scale into the line the coxswain meant to steer (changes over more than about 160 m) and the steering around it (the weave over about 20 to 160 m). Two independent methods give the same split: wavelets along the track, and the track's own Laplace&ndash;Beltrami modes from a diffusion map.</p>
  <div class="panel"><div class="eyebrow">Each crew in its own boat</div><div class="tablewrap"><table id="steert"></table></div>
  <p class="note" id="steernote"></p></div>
  <div class="panel"><div class="eyebrow">Every line in every setting's boat</div><div class="tablewrap"><table id="steerset"></table></div>
  <p class="note">Each cell: steering cost / turning built into the plan, in seconds, for that line rowed in that setting's boat at that setting's speed. The eight's setting rows every line as an eight (its larger turning loss); the fours' settings row the eight's line as a four. Within a column only the line differs.</p></div>
  <div class="panel">
    <svg id="bands" role="img" aria-label="Steering heading wobble by length scale for each line"></svg>
    <div class="legend" id="bandlegend"></div>
    <p class="note">Heading wobble by length scale. The finest band is mostly each CoxBox unit's own GPS filtering and is left out of the steering scores; the longest bands are the course's bends, common to every line.</p>
  </div>
</section>

<section id="skillsec" hidden>
  <h2>Skill, with the externals removed</h2>
  <p id="skilllead"></p>
  <div class="panel"><div class="tablewrap"><table id="skillt"></table></div>
  <p class="note" id="skillnote"></p></div>
</section>

<section id="winsec" hidden>
  <h2>Do the rankings hold along the course?</h2>
  <p id="winlead"></p>
  <div class="panel">
    <div class="toggles"><label>Measure <select id="winsel" aria-label="Measure for the window table">
      <option value="line_own">Line choice, own conditions</option><option value="line_common">Line choice, every setting</option>
      <option value="steer_wobble">Steering: heading wobble</option><option value="steer_weave">Steering: weave per boat length</option></select></label></div>
    <div class="tablewrap"><table id="wint" class="heat"></table></div>
    <p class="note" id="winnote"></p>
  </div>
  <div class="panel"><div class="eyebrow">Where on the river</div>
    <div class="toggles"><label>Line <select id="mapsel" aria-label="Line for the river map"></select></label><span class="scale">best <i></i> worst</span></div>
    <svg id="rivermap" role="img" aria-label="The course on the river, each 250 m coloured by the chosen line's rank there"></svg>
    <div class="focus"><div><b>Work on</b><ol id="focusbad"></ol></div><div><b>Strengths</b><ol id="focusgood"></ol></div></div>
    <p class="note" id="mapnote"></p></div>
  <div class="panel"><div class="eyebrow">Leave one 500 m block out</div><p id="loolead" style="margin:0"></p><ul class="loo" id="loo"></ul></div>
</section>

<section id="adaptsec" hidden>
  <h2>Adapting to the conditions</h2>
  <p id="adaptlead"></p>
  <div class="panel"><div class="tablewrap"><table id="adaptt"></table></div><p class="note" id="adaptnote"></p></div>
  <div class="panel"><div class="eyebrow">Time lost to the conditions over the nominal runs</div>
    <svg id="adapth" role="img" aria-label="Distribution over the nominal runs of each line's time lost to the conditions"></svg>
    <div class="legend" id="adaptleg"></div>
    <p class="note">Share of the nominal runs (every line rows the same draws) by seconds lost to the conditions, at a common 19:00 pace. Further left: loses less; narrower: more consistent from race to race.</p></div>
</section>

<section id="opsec" hidden>
  <h2>Each coxswain as a control loop</h2>
  <p id="oplead"></p>
  <div class="panel"><p id="opguide" style="margin:0 0 10px"></p><div class="tablewrap"><table id="opt"></table></div><p class="note" id="opnote"></p></div>
  <div class="panel"><div class="eyebrow">Each fitted coxswain, starting 5 m off the line</div>
    <svg id="opstep" role="img" aria-label="Simulated lateral offset over time for each fitted coxswain model"></svg>
    <div class="legend" id="opleg"></div>
    <p class="note">The fitted models (A) steering a boat that starts 5 m to one side of the line it means to hold, heading along it, in calm water: how fast and how smoothly each brings it back. Overshoot below zero means it crossed the line. Read it with the table's first three columns: a curve that comes back gently and stays near zero is cheap; one that dives through zero and swings back turned hard twice.</p></div>
</section>

<section id="windsec" hidden>
  <h2>Wind and gusts: steering or perturbation?</h2>
  <p id="windlead"></p>
  <div class="panel"><div class="eyebrow">Each crew's wind and how much of its weave the wind explains</div><div class="tablewrap"><table id="windt"></table></div>
  <p class="note">Wind: one-minute records at Logan, Bedford and Norwood during each crew's race (mean speed, the minute's peak 5-second gust, the gust factor, direction). Felt crosswind: that wind times the shelter at the boat's position (fetch to the upwind bank, the bank's buildings and trees, at 1.5 m), across the boat's heading. Direction test: wind pushes a boat downwind, so wind-driven weave has the crosswind's sign; r is the correlation of the sideways weave with the felt crosswind (and with its gust part), p against circular shifts of the weave. Explained: the share of the weave the felt crosswind and gusts explain (lags 0-3 strokes) beyond what shifted wind explains by chance. The stations are 5-25 km away, so these are lower bounds: they see each minute's gust climate, not the gust that hit a given stroke.</p></div>
  <div class="panel"><div class="eyebrow">Felt wind across the river, by stretch</div>
    <div class="toggles"><label>Wind from <select id="windsel" aria-label="Wind direction"><option value="330">330° (morning)</option><option value="312">312° (noon)</option></select></label></div>
    <div class="tablewrap"><table id="crosst"></table></div>
    <p class="note">Felt wind at 1.5 m as a fraction of the open wind at 10 m, at each distance from the smoothed centreline (negative = Cambridge side), averaged over the stretch. Darker = more sheltered. A dash is outside the water.</p></div>
</section>

<section id="whos" hidden>
  <h2>Who's who</h2>
  <div class="panel"><div class="tablewrap"><table id="crews"></table></div></div>
</section>

<p class="note">Model: the course model with shallow water charged on the wave-making drag only (corrected 21 October 2026), the HOCR arch rules (corrected 22 October) and the line optimiser that frees bridge points within their arches. Lines are medians per 25 m of course from each CoxBox track; the tracks end 20 to 80 m before the finish line. On the start stretch the model's course edge stops 60 m from the centreline while crews row about 68 m out, so lines are barely separated there.</p>
</div>
"""

SCRIPT = r"""
(function(){
const NS = "http://www.w3.org/2000/svg";
const K = Object.keys(D.crews);
const LET = "ABCDEFGH";
// fresh random letters on every load
let deal = [];
const letter = {}, colour = {};
function shuffle(){                       // fresh letters and colours: on load, and every time names are hidden again
  deal = K.map(k => [k, Math.random()]).sort((a, b) => a[1] - b[1]).map(x => x[0]);
  deal.forEach((k, i) => { letter[k] = LET[i]; colour[k] = "var(--c" + (i + 1) + ")"; });
}
shuffle();
let named = false;
const order = () => named ? K.slice().sort((a, b) => D.crews[a].cox.localeCompare(D.crews[b].cox)) : deal;
const lab = k => named ? D.crews[k].cox : "Line " + letter[k];
const setLab = k => named ? D.crews[k].cox + "'s crew" : "Setting " + letter[k];
let unlocked = false;
async function openBlob(url, pass){
  const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const enc = await (await fetch(url, {cache: "no-cache"})).json();
  const km = await crypto.subtle.importKey("raw", new TextEncoder().encode(pass), "PBKDF2", false, ["deriveKey"]);
  const key = await crypto.subtle.deriveKey({name: "PBKDF2", salt: b64(enc.salt), iterations: enc.iterations, hash: "SHA-256"}, km, {name: "AES-GCM", length: 256}, false, ["decrypt"]);
  return JSON.parse(new TextDecoder().decode(await crypto.subtle.decrypt({name: "AES-GCM", iv: b64(enc.iv)}, key, b64(enc.data))));
}
const f1 = v => (v > 0.05 ? "+" : v < -0.05 ? "−" : "") + Math.abs(v).toFixed(1);
const mmss = s => { const m = Math.floor(s / 60); return m + ":" + (s - 60 * m).toFixed(1).padStart(4, "0"); };
const el = (tag, txt, cls) => { const e = document.createElement(tag); if (txt != null) e.textContent = txt; if (cls) e.className = cls; return e; };
const S = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); return e; };
const dot = k => { const d = el("span", null, "dot"); d.style.background = colour[k]; return d; };

function delta(set, line){ const T = D.timed[set]; const m = Math.min(...K.map(k => T[k].total)); return T[line].total - m; }

function matrix(){
  const t = document.getElementById("matrix"); t.textContent = "";
  const O = order(), hr = el("tr"); hr.appendChild(el("th", "Line"));
  O.forEach(s => { const th = el("th", setLab(s)); th.style.textAlign = "right";
    if (named) { th.appendChild(el("br")); th.appendChild(el("span", D.crews[s].boat + ", " + D.crews[s].speed.toFixed(2) + " m/s")); th.lastChild.style.textTransform = "none"; }
    hr.appendChild(th); });
  const thm = el("th", "Mean"); thm.style.textAlign = "right"; hr.appendChild(thm); t.appendChild(hr);
  const means = {};
  O.forEach(k => means[k] = O.reduce((a, s) => a + delta(s, k), 0) / O.length);
  O.forEach(k => { const tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(lab(k))); tr.appendChild(td);
    O.forEach(s => { const d = delta(s, k); tr.appendChild(el("td", f1(d), "n" + (d < 0.05 ? " zero" : ""))); });
    tr.appendChild(el("td", f1(means[k]), "n best")); t.appendChild(tr); });
  [["best", "Best line (Radau optimum, that setting's own arches)"], ["shortest", "Shortest legal line (bestline.row3r.com)"]].forEach(([ln, name]) => {
    const tr = el("tr"); tr.appendChild(el("td", name));
    O.forEach(s => tr.appendChild(el("td", f1(delta(s, ln)), "n")));
    tr.appendChild(el("td", f1(O.reduce((a, s) => a + delta(s, ln), 0) / O.length), "n")); t.appendChild(tr); });
  const ranked = O.slice().sort((a, b) => means[a] - means[b]);
  const spread = means[ranked[ranked.length - 1]] - means[ranked[0]];
  const firsts = {}; O.forEach(s => { const w = O.slice().sort((a, b) => delta(s, a) - delta(s, b))[0]; firsts[w] = (firsts[w] || 0) + 1; });
  const top = ranked[0];
  document.getElementById("lead").textContent = lab(top) + " is the fastest club line on average and the fastest in " + (firsts[top] || 0) + " of " + O.length +
    " settings. The club lines are " + spread.toFixed(1) + " s apart on average; the best line found gains " +
    Math.min(...O.map(s => -delta(s, "best"))).toFixed(0) + " to " + Math.max(...O.map(s => -delta(s, "best"))).toFixed(0) + " s on the fastest club line in each setting.";
}

function profile(){
  const svg = document.getElementById("prof"); svg.textContent = "";
  const P = D.profile, W = 760, H = 320, x0 = 46, x1 = W - 10, y0 = 18, y1 = H - 28;
  const smax = P.s[P.s.length - 1], omax = Math.max(...P.hi.map(Math.abs), ...P.lo.map(Math.abs)) + 5;
  const X = s => x0 + (s / smax) * (x1 - x0), Y = o => y0 + (1 - (o + omax) / (2 * omax)) * (y1 - y0);
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  let d = "M" + P.s.map((s, i) => X(s).toFixed(1) + "," + Y(P.hi[i]).toFixed(1)).join("L");
  d += "L" + P.s.slice().reverse().map((s, i) => X(s).toFixed(1) + "," + Y(P.lo[P.s.length - 1 - i]).toFixed(1)).join("L") + "Z";
  svg.appendChild(S("path", {d: d, fill: "var(--river)"}));
  svg.appendChild(S("line", {x1: x0, x2: x1, y1: Y(0), y2: Y(0), stroke: "var(--bank)", "stroke-width": 0.6, "stroke-dasharray": "3 4"}));
  let lastX = -1e9, row = 0;
  P.bridges.forEach(b => { svg.appendChild(S("line", {x1: X(b.s), x2: X(b.s), y1: y0, y2: y1, stroke: "var(--bank)", "stroke-width": 0.6, "stroke-dasharray": "2 3"}));
    row = X(b.s) - lastX < 70 ? row + 1 : 0; lastX = X(b.s);           // stack labels of bridges close together
    const t = S("text", {x: X(b.s) + 3, y: y0 + 10 + 11 * row, "font-size": 10, fill: "var(--muted)"}); t.textContent = b.name.replace(" Footbridge", "").replace(" Bridge", "").replace("Grand Junction RR", "Railway"); svg.appendChild(t); });
  [-40, -20, 0, 20, 40].forEach(o => { const t = S("text", {x: x0 - 6, y: Y(o) + 3, "font-size": 10, "text-anchor": "end", fill: "var(--muted)"}); t.textContent = (o > 0 ? "+" : "") + o; svg.appendChild(t); });
  const tb = S("text", {x: x0, y: H - 6, "font-size": 10.5, fill: "var(--muted)"}); tb.textContent = "Cambridge side ↓   Boston side ↑   (m from centreline; km along the course →)"; svg.appendChild(tb);
  const line = (vals, col, dash, w) => svg.appendChild(S("path", {d: "M" + P.s.map((s, i) => X(s).toFixed(1) + "," + Y(vals[i]).toFixed(1)).join("L"),
    fill: "none", stroke: col, "stroke-width": w, "stroke-dasharray": dash || "none", "stroke-linejoin": "round"}));
  if (document.getElementById("showshort").checked) line(P.lines.shortest, "var(--short)", "2 3", 1.4);
  if (document.getElementById("showbest").checked) order().forEach(k => line(P.lines["best:" + k], colour[k], "6 4", 1.3));
  order().forEach(k => line(P.lines[k], colour[k], null, 2));
  const lg = document.getElementById("legend"); lg.textContent = "";
  order().forEach(k => { const s = el("span"); s.appendChild(dot(k)); s.appendChild(document.createTextNode(lab(k))); lg.appendChild(s); });
}

function stretch(){
  const sel = document.getElementById("setsel"), cur = sel.value || "avg"; sel.textContent = "";
  const o = el("option", "Average over all settings"); o.value = "avg"; sel.appendChild(o);
  order().forEach(s => { const op = el("option", setLab(s)); op.value = s; sel.appendChild(op); });
  sel.value = [...sel.options].some(x => x.value === cur) ? cur : "avg";
  const sets = sel.value === "avg" ? K : [sel.value];
  const t = document.getElementById("stretch"); t.textContent = "";
  const O = order(), hr = el("tr"); hr.appendChild(el("th", "Stretch")); hr.appendChild(el("th", "m"));
  O.forEach(k => { const th = el("th"); th.appendChild(dot(k)); th.appendChild(document.createTextNode(lab(k))); hr.appendChild(th); });
  const thb = el("th", "Best line"); hr.appendChild(thb); t.appendChild(hr);
  D.sections.forEach((sec, i) => {
    const tr = el("tr"); tr.appendChild(el("td", sec.name)); tr.appendChild(el("td", String(sec.m), "n"));
    const dl = ln => sets.reduce((a, s) => { const T = D.timed[s]; const m = Math.min(...K.map(k => T[k].sec[i])); return a + T[ln].sec[i] - m; }, 0) / sets.length;
    O.forEach(k => { const d = dl(k), off = D.off[k][i]; tr.appendChild(el("td", f1(d) + " s · " + (off > 0 ? "+" : "") + Math.round(off) + " m", "n" + (d < 0.05 ? " zero" : ""))); });
    const db = sets.reduce((a, s) => { const T = D.timed[s]; const m = Math.min(...K.map(k => T[k].sec[i])); return a + T.best.sec[i] - m; }, 0) / sets.length;
    tr.appendChild(el("td", f1(db) + " s", "n")); t.appendChild(tr); });
}

function best(){
  const t = document.getElementById("bestt"); t.textContent = "";
  const hr = el("tr"); ["Setting", "Own line vs best", "Distance", "Shallow water", "Wind", "Path difference"].forEach((h, i) => { const th = el("th", h); if (i) th.style.textAlign = "right"; hr.appendChild(th); });
  t.appendChild(hr);
  order().forEach(s => { const T = D.timed[s], a = T[s], b = T.best, tr = el("tr");
    const td = el("td"); td.appendChild(dot(s)); td.appendChild(document.createTextNode(setLab(s) + " (" + lab(s) + ")")); tr.appendChild(td);
    [a.total - b.total, a.distance - b.distance, a.depth - b.depth, a.wind - b.wind].forEach(v => tr.appendChild(el("td", f1(v) + " s", "n")));
    tr.appendChild(el("td", f1(a.path - b.path) + " m", "n")); t.appendChild(tr); });
}

function crews(){
  const sec = document.getElementById("whos"), t = document.getElementById("crews");
  t.textContent = ""; sec.hidden = !named; if (!named) return;
  const hr = el("tr"); ["Cox", "Letter this time", "Event", "Bow", "Place", "Official time", "Arches: River St / Western Ave / Weeks"].forEach(h => hr.appendChild(el("th", h))); t.appendChild(hr);
  const A = p => ({"centre": "centre", "Cambridge shore": "Cambridge", "Boston shore": "Boston"}[p] || p || "–");
  order().forEach(k => { const c = D.crews[k], tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(c.cox)); tr.appendChild(td);
    [letter[k], c.event_name, String(c.bow), c.place + " of " + c.field, mmss(c.official),
     [c.pins["River Street"], c.pins["Western Avenue"], c.pins["Weeks Footbridge"]].map(A).join(" / ")].forEach(v => tr.appendChild(el("td", v)));
    t.appendChild(tr); });
}

function steer(){
  const S_ = D.steer; const sec = document.getElementById("steersec"); if (!S_) return; sec.hidden = false;
  const O = order(), sets = K, t = document.getElementById("steert"); t.textContent = "";
  const fastestInt = s => Math.min(...K.map(k => S_.crews[k].intended[s]));
  const hr = el("tr"); ["Line", "Intended line vs fastest intended (s)", "Turning built into the plan (s)", "Tightest planned turn (L/R)",
    "Steering cost (s)", "Weave (m, × boat length)", "Heading wobble (°)", "Corrections per km", "Smoothness (SPARC)"].forEach((h, i) => {
    const th = el("th", h); if (i) th.style.textAlign = "right"; hr.appendChild(th); }); t.appendChild(hr);
  O.forEach(k => { const c = S_.crews[k], o = c.own, tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(lab(k))); tr.appendChild(td);
    const intd = sets.reduce((a, s) => a + c.intended[s] - fastestInt(s), 0) / sets.length;
    [f1(intd), o.plan.turning_s.toFixed(1), o.plan.peak_rudder_index.toFixed(2), o.steer.seconds.toFixed(1),
     o.steer.weave_m.toFixed(2) + " (" + o.steer.weave_per_length.toFixed(3) + ")", o.steer.wobble_deg.toFixed(2),
     o.steer.corrections_per_km.toFixed(1), o.steer.sparc.toFixed(2)].forEach(v => tr.appendChild(el("td", v, "n"))); t.appendChild(tr); });
  const B = S_.by_setting, t2 = document.getElementById("steerset"); t2.textContent = "";
  if (B) {
    const h2 = el("tr"); h2.appendChild(el("th", "Line"));
    O.forEach(s => { const th = el("th", setLab(s)); th.style.textAlign = "right";
      if (named) { th.appendChild(el("br")); const sp = el("span", D.crews[s].boat + ", " + D.crews[s].speed.toFixed(2) + " m/s"); sp.style.textTransform = "none"; th.appendChild(sp); }
      h2.appendChild(th); });
    t2.appendChild(h2);
    O.forEach(k => { const tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(lab(k))); tr.appendChild(td);
      O.forEach(s => { const c = B[s].lines[k]; const m = Math.min(...K.map(j => B[s].lines[j].steer_s));
        tr.appendChild(el("td", c.steer_s.toFixed(1) + " / " + c.plan_turn_s.toFixed(1), "n" + (c.steer_s <= m + 0.01 ? " zero" : ""))); });
      t2.appendChild(tr); });
  }
  const agree = Math.max(...K.map(k => S_.crews[k].agree));
  document.getElementById("steernote").textContent = "Intended line: the slow part of the track (changes over more than " + S_.cutoff + " m), timed in every setting against the fastest intended line, averaged. " +
    "Everything else is judged in the crew's own boat at its own speed. Turning built into the plan: the seconds the intended line's turns cost, using the turning loss measured from our boats (speed lost = k × turn rate², k larger for an eight; turn rate = speed / radius, so the cost grows as speed² / radius²). " +
    "Tightest planned turn: boat length / radius, which sets the rudder angle a turn needs. Steering cost: the extra turning and distance of the actual track over its intended line, in that boat. " +
    "Weave: sideways RMS of the steering part, also as a fraction of the boat's length. Heading wobble: RMS of the fast heading. Corrections per km: reversals of the fast heading beyond 0.5°. " +
    "SPARC: spectral arc length of the turn rate per 500 m (Balasubramanian et al. 2015), closer to zero is smoother. " +
    "Because these are judged in each crew's own boat, the eight stands out in the plan columns (a longer boat needs more rudder for the same turn), even in blind mode. " +
    "The two methods' intended lines agree to within " + (agree * 100).toFixed(0) + " cm. The steering of different lines is " + S_.corr.toFixed(2) +
    " correlated, so some corrections are shared (everyone adjusts at the same bridges and bends); measured against the course centreline instead it was 0.68 to 0.97, the centreline's own kinks.";
  const svg = document.getElementById("bands"); svg.textContent = ""; const W = 760, H = 240, x0 = 50, x1 = W - 20, y0 = 14, y1 = H - 34;
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  const nb = S_.band_labels.length, ymax = Math.max(...K.map(k => Math.max(...S_.crews[k].bands.slice(0, 4)))) * 1.1;
  const X = i => x0 + i * (x1 - x0) / (nb - 1), Y = v => y1 - Math.min(v, ymax) / ymax * (y1 - y0);
  for (let g = 0; g <= ymax; g += 0.5) { svg.appendChild(S("line", {x1: x0, x2: x1, y1: Y(g), y2: Y(g), stroke: "var(--line)", "stroke-width": 1}));
    const tx = S("text", {x: x0 - 6, y: Y(g) + 3, "font-size": 10, "text-anchor": "end", fill: "var(--muted)"}); tx.textContent = g.toFixed(1) + "°"; svg.appendChild(tx); }
  S_.band_labels.forEach((b, i) => { const tx = S("text", {x: X(i), y: H - 14, "font-size": 10.5, "text-anchor": "middle", fill: "var(--muted)"}); tx.textContent = b + (i === 0 ? " (GPS)" : ""); svg.appendChild(tx); });
  svg.appendChild(S("rect", {x: x0 - 8, y: y0, width: (x1 - x0) / (nb - 1) / 2 + 8, height: y1 - y0, fill: "var(--line)", opacity: 0.35}));
  order().forEach(k => { const v = S_.crews[k].bands;
    svg.appendChild(S("path", {d: "M" + v.map((b, i) => X(i).toFixed(1) + "," + Y(b).toFixed(1)).join("L"), fill: "none", stroke: colour[k], "stroke-width": 2}));
    v.forEach((b, i) => svg.appendChild(S("circle", {cx: X(i), cy: Y(b), r: 3, fill: colour[k]}))); });
  const lg = document.getElementById("bandlegend"); lg.textContent = "";
  order().forEach(k => { const sp = el("span"); sp.appendChild(dot(k)); sp.appendChild(document.createTextNode(lab(k))); lg.appendChild(sp); });
}

function wind(){
  const Wd = D.wind; if (!Wd) return; document.getElementById("windsec").hidden = false;
  const O = order(), t = document.getElementById("windt"); t.textContent = "";
  const hr = el("tr"); ["Line", "Wind (m/s)", "Gust (m/s)", "Gust factor", "From", "Felt crosswind (m/s)", "Weave vs crosswind r (p)", "vs gusts r (p)", "Weave explained by wind"].forEach((h, i) => {
    const th = el("th", h); if (i) th.style.textAlign = "right"; hr.appendChild(th); }); t.appendChild(hr);
  O.forEach(k => { const c = Wd.crews[k], tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(lab(k) + (named && D.crews[k].window_et ? " (" + D.crews[k].window_et.join("–") + ")" : ""))); tr.appendChild(td);
    const sig = p => p < 0.05 ? " *" : "";
    [c.mean_wind.toFixed(1), c.mean_gust.toFixed(1), c.gust_factor.toFixed(2), Math.round(c.mean_dir) + "°", c.felt_cross_rms.toFixed(2),
     c.direction.r.toFixed(2) + " (" + c.direction.p.toFixed(3) + ")" + sig(c.direction.p), c.gust_direction.r.toFixed(2) + " (" + c.gust_direction.p.toFixed(3) + ")" + sig(c.gust_direction.p),
     Math.max(c.perturbation_share.excess * 100, 0).toFixed(1) + "%"].forEach(v => tr.appendChild(el("td", v, "n"))); t.appendChild(tr); });
  const sigk = O.filter(k => Wd.crews[k].direction.p < 0.05 || Wd.crews[k].gust_direction.p < 0.05);
  const sg = Wd.segments;
  document.getElementById("windlead").textContent = "Every crew raced in much the same gusty wind (gust factor " +
    Math.min(...O.map(k => Wd.crews[k].gust_factor)).toFixed(2) + "–" + Math.max(...O.map(k => Wd.crews[k].gust_factor)).toFixed(2) + "). " +
    (sigk.length ? "Only " + sigk.map(lab).join(" and ") + " shows weave with the wind's sign (marked *), so part of it was the wind pushing the boat; for the others the weave carries no detectable wind signature, which points to steering. " : "No line's weave carries a detectable wind signature. ") +
    "Across 100 m stretches, heading wobble did not rise with the felt crosswind (slope " + sg.felt_crosswind.slope.toFixed(2) + " ± " + sg.felt_crosswind.se.toFixed(2) + " deg per m/s) or with gusts, and stretches nearer a bank were not steered straighter. Shelter depends on which shore: see the table below.";
  const sel = document.getElementById("windsel"), X = Wd.cross[sel.value], ct = document.getElementById("crosst"); ct.textContent = "";
  const h2 = el("tr"); h2.appendChild(el("th", "Stretch")); Wd.offsets.forEach(o => { const th = el("th", (o > 0 ? "+" : "") + o); th.style.textAlign = "right"; h2.appendChild(th); }); ct.appendChild(h2);
  X.forEach(r => { const tr = el("tr"); tr.appendChild(el("td", r.stretch));
    r.profile.forEach(v => { const td = el("td", v == null ? "–" : v.toFixed(2), "n");
      if (v != null) { td.style.background = "color-mix(in srgb, var(--best) " + Math.round((1 - v) * 70) + "%, transparent)"; } tr.appendChild(td); }); ct.appendChild(tr); });
}

function skill(){
  const Q = D.skill; if (!Q) return; document.getElementById("skillsec").hidden = false;
  const O = order(), t = document.getElementById("skillt"); t.textContent = "";
  const hr = el("tr"); ["Line", "Line choice: own conditions (s behind best)", "Line choice: all settings (s)", "Planned turning, common boat (s)", "Track smoothness (SPARC)",
    "Steering: heading wobble, adjusted (°) [95%]", "Steering: weave per boat length, adjusted [95%]", "Passes"].forEach((h, i) => {
    const th = el("th", h); if (i) th.style.textAlign = "right"; hr.appendChild(th); }); t.appendChild(hr);
  const best = (obj, k, low) => { const vals = O.map(j => obj(j)); return obj(k) === Math.min(...vals) ? " zero" : ""; };
  O.forEach(k => { const tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(lab(k))); tr.appendChild(td);
    const w = Q.steering.wobble, v = Q.steering.weave_L;
    const cells = [[Q.line_choice[k].own_conditions_s.toFixed(1), best(j => Q.line_choice[j].own_conditions_s, k)],
                   [Q.line_choice[k].all_settings_s.toFixed(1), best(j => Q.line_choice[j].all_settings_s, k)],
                   [Q.plan[k].turning_common_boat_s.toFixed(2), best(j => Q.plan[j].turning_common_boat_s, k)],
                   [Q.plan[k].sparc.toFixed(2), best(j => -Q.plan[j].sparc, k)],
                   [w.adjusted[k].toFixed(2) + " [" + w.ci95[k][0].toFixed(2) + ", " + w.ci95[k][1].toFixed(2) + "]", best(j => w.adjusted[j], k)],
                   [v.adjusted[k].toFixed(3) + " [" + v.ci95[k][0].toFixed(3) + ", " + v.ci95[k][1].toFixed(3) + "]", best(j => v.adjusted[j], k)],
                   [String(Q.traffic[k].passes), ""]];
    cells.forEach(([txt, cls]) => tr.appendChild(el("td", txt, "n" + cls))); t.appendChild(tr); });
  const w = Q.steering.wobble, c = w.coefficients;
  document.getElementById("skilllead").textContent = "Line choice is where the coxswains differ: " +
    (Math.max(...O.map(k => Q.line_choice[k].own_conditions_s)) - Math.min(...O.map(k => Q.line_choice[k].own_conditions_s))).toFixed(0) +
    " s between the closest and furthest from the best line in their own conditions. Steering mostly is not: the course and the conditions explain " +
    Math.round(100 * w.r2_crews_and_externals) + "% of the heading wobble from stretch to stretch and who was steering about " + Math.round(100 * w.r2_crews_only) +
    "%; once the externals are held equal, the steering scores' 95% intervals overlap. The biggest external is the bridges: wobble is " + c.bridge.toFixed(1) +
    "° higher within 75 m of one, as coxswains line up for the arches.";
  const nz = Math.max(...O.map(k => Q.noise[k].max_noise_share));
  document.getElementById("skillnote").textContent = "Line choice: the intended line (the track's changes over more than 160 m) timed with the pace fixed and every line in the same boat, against the setting's best line; own conditions = that crew's boat, wind and river; all settings = averaged over every crew's. " +
    "Planned turning: the intended line's turning cost averaged over the settings (a common boat). Track smoothness: SPARC of the whole track's turn rate. Steering: per 100 m of course, regressed on the felt crosswind (scaled by 1/speed², how far wind pushes a boat), its gust part, the bend, the river's width, a bridge within 75 m, and crews of the same event within 25 m or passing (reconstructed from the start order, a " +
    Q.start_interval_s.toFixed(1) + " s start interval and every crew's official splits), with one term per crew; the crew term at common externals is the score, intervals from a block bootstrap (500 m blocks). " +
    "What is fixed per crew cannot be separated from the coxswain by regression, so physics handles it: weave per boat length, drift scaled by speed², turning in a common boat. GPS noise: at most " +
    Math.round(100 * nz) + "% of the steering band for any unit. The best value in each column is marked.";
}

function windows(){
  const W = D.windows; if (!W) return; document.getElementById("winsec").hidden = false;
  const m = document.getElementById("winsel").value || "line_own", M = W.measures[m], L = W.leave_one_out[m], O = order();
  const unit = {line_own: "s lost to the best line in the window", line_common: "s lost to the best line in the window, averaged over the settings",
                steer_wobble: "° residual heading wobble", steer_weave: "residual weave per boat length"}[m];
  const dp = m === "steer_weave" ? 3 : 2, nw = W.windows.length;
  const rank = j => { const v = K.map(k => [k, M.values[k][j]]).sort((a, b) => a[1] - b[1]); const r = {}; v.forEach(([k], i) => r[k] = i + 1); return r; };
  const R = W.windows.map((_, j) => rank(j));
  const t = document.getElementById("wint"); t.textContent = "";
  const hr = el("tr"); hr.appendChild(el("th", "Line"));
  W.windows.forEach(w => { const th = el("th", (w[0] / 1000).toFixed(2).replace(/\.?0+$/, ""), "w"); th.title = w[0] + "–" + w[1] + " m"; hr.appendChild(th); });
  ["Mean rank", "First", "Last"].forEach(h => { const th = el("th", h); th.style.textAlign = "right"; hr.appendChild(th); });
  t.appendChild(hr);
  const n = K.length;
  O.forEach(k => { const tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(lab(k))); tr.appendChild(td);
    const bad = new Set((M.bad_patches[k] || []).map(p => p.window[0]));
    W.windows.forEach((w, j) => { const r = R[j][k], c = el("td", String(r), "r" + (bad.has(w[0]) ? " bad" : ""));
      c.style.background = "color-mix(in srgb, " + colour[k] + " " + Math.round(70 * (n - r) / (n - 1)) + "%, transparent)";
      c.title = w[0] + "–" + w[1] + " m: " + M.values[k][j].toFixed(dp) + " " + unit + (bad.has(w[0]) ? " (bad patch)" : ""); tr.appendChild(c); });
    [M.mean_rank[k].toFixed(1), Math.round(M.rank_first[k] * nw) + " of " + nw, Math.round(M.rank_last[k] * nw) + " of " + nw].forEach(v => tr.appendChild(el("td", v, "n")));
    t.appendChild(tr); });
  const names = ord => ord.map(lab).join(", ");
  const lead = L.changes.every(c => c.order[0] === L.order[0]);
  document.getElementById("loolead").textContent = "Whole course, best first: " + names(L.order) + ". " +
    (L.changes.length ? "Dropping any one of " + L.blocks_n + " blocks changes this order for " + L.changes.length + "; " + lab(L.order[0]) + (lead ? " stays first in every case." : " loses first place in at least one.") : "No single block changes it.");
  const ul = document.getElementById("loo"); ul.textContent = "";
  L.changes.forEach(c => ul.appendChild(el("li", "Without " + c.block[0] + "–" + c.block[1] + " m: " + names(c.order))));
  const Wk = x => W.measures[x].kendall_w.toFixed(2);
  document.getElementById("winlead").textContent = "Each measure recomputed in " + nw + " windows of " + W.win_m + " m, sliding by " + W.step_m + " m, to see whether one stretch (a bad finish, a bridge) is carrying a ranking. " +
    "Agreement between the windows' rankings (Kendall's W, 1 = every window ranks the lines the same, 0 = no agreement): line choice " + Wk("line_own") + " in own conditions and " + Wk("line_common") + " in every setting; steering " + Wk("steer_wobble") + " for wobble and " + Wk("steer_weave") + " for weave. " +
    "So the line-choice ranking is partly consistent along the course and the steering ranking mostly is not: differences in steering live in particular stretches, not over the whole race.";
  document.getElementById("winnote").textContent = "Each cell: the line's rank in that window (1 = best), darker = better; hover for the value (" + unit + "). Columns: the window's start, in km from the start line. " +
    "Outlined: a bad patch, where the line's gap is more than 2 robust SD above its own median. Steering is the residual after the externals (wind, gusts, bends, width, bridges, traffic) at the whole-course coefficients, so every window is on one scale. " +
    "First / Last: windows in which the line ranks first / last.";
}

function rivermap(){
  const W = D.windows, G = D.map; if (!W || !G) return;
  const m = document.getElementById("winsel").value || "line_own", M = W.measures[m];
  const sel = document.getElementById("mapsel"), cur = K.includes(sel.value) ? sel.value : order()[0]; sel.textContent = "";
  order().forEach(k => { const o = el("option", lab(k)); o.value = k; sel.appendChild(o); }); sel.value = cur;
  const k = cur, n = K.length, P = 250;
  // each 250 m piece: the line's mean rank over the windows that cover it
  const rk = W.windows.map((_, j) => { const v = K.map(q => [q, M.values[q][j]]).sort((a, b) => a[1] - b[1]); const r = {}; v.forEach(([q], i) => r[q] = i + 1); return r; });
  const pieces = [];
  for (let a = 0; a < G.length - 1; a += P) {
    const b = Math.min(a + P, G.length), js = W.windows.map((w, j) => [w, j]).filter(([w]) => w[0] <= a + 1e-6 && w[1] >= b - 1e-6).map(([, j]) => j);
    let use = js.length ? js : W.windows.map((w, j) => [w, j]).filter(([w]) => w[0] < b && w[1] > a).map(([, j]) => j);
    if (!use.length) use = [W.windows.length - 1];          // past the last window's end
    pieces.push({a, b, r: use.reduce((s, j) => s + rk[j][k], 0) / use.length});
  }
  const svg = document.getElementById("rivermap"); svg.textContent = "";
  const xs = G.px.concat(G.sx), ys = G.py.concat(G.sy);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const Wd = 900, pad = 110, sc = (Wd - 2 * pad) / (x1 - x0), Hd = (y1 - y0) * sc + 2 * 60;
  const X = x => pad + (x - x0) * sc, Y = y => 60 + (y1 - y) * sc;
  svg.setAttribute("viewBox", "0 0 " + Wd + " " + Hd.toFixed(0));
  const pts = (ax, ay) => ax.map((x, i) => X(x).toFixed(1) + "," + Y(ay[i]).toFixed(1));
  svg.appendChild(S("polygon", {points: pts(G.px, G.py).concat(pts(G.sx, G.sy).reverse()).join(" "), fill: "var(--river)", stroke: "var(--bank)", "stroke-width": 1}));
  const at = mm => { let i = 0; while (i < G.st.length - 1 && G.st[i + 1] < mm) i++; const f = Math.min(1, Math.max(0, (mm - G.st[i]) / ((G.st[i + 1] - G.st[i]) || 1))); return [G.cx[i] + f * (G.cx[Math.min(i + 1, G.st.length - 1)] - G.cx[i]), G.cy[i] + f * (G.cy[Math.min(i + 1, G.st.length - 1)] - G.cy[i]), i]; };
  const col = r => { const t = Math.round(100 * (r - 1) / (n - 1)); return "color-mix(in srgb, var(--loss) " + t + "%, var(--gain))"; };
  // bad patches as a dashed band beside the course
  (M.bad_patches[k] || []).forEach(p => {
    const sel_ = G.st.map((s, i) => [s, i]).filter(([s]) => s >= p.window[0] && s <= p.window[1]).map(([, i]) => i);
    if (sel_.length < 2) return;
    svg.appendChild(S("polyline", {points: sel_.map(i => X(G.px[i]).toFixed(1) + "," + Y(G.py[i]).toFixed(1)).join(" "), fill: "none", stroke: "var(--loss)", "stroke-width": 3, "stroke-dasharray": "6 4"}));
  });
  pieces.forEach(p => {
    const sel_ = G.st.map((s, i) => [s, i]).filter(([s]) => s >= p.a && s <= p.b).map(([, i]) => i);
    const ends = [at(p.a), at(p.b)].map(([x, y]) => X(x).toFixed(1) + "," + Y(y).toFixed(1));
    const ln = S("polyline", {points: [ends[0]].concat(sel_.map(i => X(G.cx[i]).toFixed(1) + "," + Y(G.cy[i]).toFixed(1)), [ends[1]]).join(" "), fill: "none", "stroke-width": 9, "stroke-linecap": "butt", "stroke-linejoin": "round"});
    ln.style.stroke = col(p.r);
    const t = S("title", {}); t.textContent = p.a + "–" + p.b + " m: mean rank " + p.r.toFixed(1) + " of " + n; ln.appendChild(t); svg.appendChild(ln);
  });
  // km ticks on the course, landmarks off it, alternating sides
  for (let mm = 0; mm <= G.length; mm += 500) {
    const [x, y] = at(mm); svg.appendChild(S("circle", {cx: X(x), cy: Y(y), r: 2.5, fill: "var(--ink)"}));
    const tx = S("text", {x: X(x) + 5, y: Y(y) - 7, class: "tick"}); tx.textContent = (mm / 1000).toFixed(1) + " km"; svg.appendChild(tx);
  }
  G.marks.forEach((mk, i) => {
    const [, , j] = at(mk.m), j2 = Math.min(j + 1, G.st.length - 1), j1 = Math.max(j2 - 2, 0);
    const hx = G.cx[j2] - G.cx[j1], hy = G.cy[j2] - G.cy[j1], h = Math.hypot(hx, hy) || 1;
    const sgn = i % 2 ? 1 : -1, nx = -hy / h * sgn, ny = hx / h * sgn;
    const px = X(mk.x) + nx * 34, py = Y(mk.y) - ny * 34;
    svg.appendChild(S("line", {x1: X(mk.x), y1: Y(mk.y), x2: px, y2: py, stroke: "var(--muted)", "stroke-width": 1}));
    const tx = S("text", {x: px + (nx >= 0 ? 3 : -3), y: py + (ny > 0 ? -2 : 11), "text-anchor": nx >= 0 ? "start" : "end"}); tx.textContent = mk.name; svg.appendChild(tx);
  });
  const near = mm => { let b = G.marks[0]; G.marks.forEach(q => { if (Math.abs(q.m - mm) < Math.abs(b.m - mm)) b = q; }); return Math.abs(b.m - mm) < 60 ? "at " + b.name : (b.m > mm ? "before " : "after ") + b.name; };
  const desc = p => p.a + "–" + p.b + " m (" + near((p.a + p.b) / 2) + "), rank " + p.r.toFixed(1);
  const sorted = pieces.slice().sort((a, b) => b.r - a.r);
  const fb = document.getElementById("focusbad"), fg = document.getElementById("focusgood"); fb.textContent = ""; fg.textContent = "";
  sorted.slice(0, 4).forEach(p => fb.appendChild(el("li", desc(p))));
  sorted.slice(-3).reverse().forEach(p => fg.appendChild(el("li", desc(p))));
  document.getElementById("mapnote").textContent = "The course from the start (downstream, right) to the finish, north up. Each 250 m is coloured by " + lab(k) +
    "'s mean rank among the " + n + " lines over the 1000 m windows that cover it, in the measure chosen above (green best, red worst); hover a piece for its rank. " +
    "Dashed red beside the course: this line's bad patches. Work on / Strengths: its worst and best 250 m pieces, named by the nearest landmark.";
}

function adapt(){
  const A = D.adapt; if (!A) return; document.getElementById("adaptsec").hidden = false;
  const O = order(), N = A.nominal, t = document.getElementById("adaptt"); t.textContent = "";
  const lowest = f => Math.min(...O.map(f));
  const hr = el("tr"); ["Line", "Lost to the conditions, nominal runs (s)", "Against a typical response (s)", "Run-to-run spread (s)", "10th–90th percentile (s)",
    "Its own race: own vs typical response (s)", "Costliest condition in its race"].forEach((h, i) => { const th = el("th", h); if (i) th.style.textAlign = "right"; hr.appendChild(th); });
  t.appendChild(hr);
  const cname = {cross_drift: "crosswind", gust: "gusts", passing: "passing", passed: "being passed", near: "side by side"};
  O.forEach(k => { const n = N[k], r = A.realised[k], tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(lab(k))); tr.appendChild(td);
    const worst = Object.entries(r.by_condition_s).sort((a, b) => b[1] - a[1])[0];
    [[n.mean_s.toFixed(1) + " [" + n.mean_ci95[0].toFixed(0) + ", " + n.mean_ci95[1].toFixed(0) + "]", n.mean_s === lowest(j => N[j].mean_s)],
     [f1(n.rel_mean_s) + " [" + f1(n.rel_ci95[0]) + ", " + f1(n.rel_ci95[1]) + "]", n.rel_mean_s === lowest(j => N[j].rel_mean_s)],
     [n.sd_s.toFixed(1), n.sd_s === lowest(j => N[j].sd_s)],
     [n.p10_s.toFixed(1) + " – " + n.p90_s.toFixed(1), false],
     [r.own_response_s.toFixed(1) + " vs " + r.typical_response_s.toFixed(1), false],
     [worst[1] > 0.05 ? cname[worst[0]] + " (" + worst[1].toFixed(1) + " s)" : "none above 0.1 s", false]].forEach(([v, b]) => tr.appendChild(el("td", v, "n" + (b ? " zero" : ""))));
    t.appendChild(tr); });
  // distributions
  const svg = document.getElementById("adapth"); svg.textContent = "";
  const E = A.hist_edges, Wd = 760, Hd = 240, ml = 48, mr = 14, mt = 10, mb = 36;
  svg.setAttribute("viewBox", "0 0 " + Wd + " " + Hd);
  const tot = A.n_runs, ymax = Math.max(...O.map(k => Math.max(...N[k].hist))) / tot;
  // at least -2 to +2 s, so differences of a fraction of a second are not drawn as if they were large
  const x0 = Math.min(E[0], -2), x1 = Math.max(E[E.length - 1], 2);
  const X = v => ml + (v - x0) / (x1 - x0) * (Wd - ml - mr), Y = v => mt + (1 - v / ymax) * (Hd - mt - mb);
  const rg = x1 - x0, step = rg > 40 ? 10 : rg > 12 ? 5 : rg > 4 ? 1 : 0.5;
  for (let v = Math.ceil(x0 / step) * step; v <= x1 + 1e-9; v += step) {
    svg.appendChild(S("line", {x1: X(v), x2: X(v), y1: mt, y2: Hd - mb, stroke: "var(--line)", "stroke-width": 1}));
    const tx = S("text", {x: X(v), y: Hd - mb + 16, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12}); tx.textContent = +v.toFixed(2) + " s"; svg.appendChild(tx);
  }
  svg.appendChild(S("line", {x1: ml, x2: Wd - mr, y1: Hd - mb, y2: Hd - mb, stroke: "var(--muted)", "stroke-width": 1}));
  const yl = S("text", {x: 4, y: mt + 10, fill: "var(--muted)", "font-size": 12}); yl.textContent = "share"; svg.appendChild(yl);
  O.forEach(k => { const h = N[k].hist; let d = "M" + X(E[0]) + "," + Y(0);
    h.forEach((c, i) => { d += " L" + X(E[i]).toFixed(1) + "," + Y(c / tot).toFixed(1) + " L" + X(E[i + 1]).toFixed(1) + "," + Y(c / tot).toFixed(1); });
    d += " L" + X(E[E.length - 1]) + "," + Y(0);
    const p = S("path", {d: d, fill: "none", "stroke-width": 2}); p.style.stroke = colour[k]; svg.appendChild(p); });
  const lg = document.getElementById("adaptleg"); lg.textContent = ""; O.forEach(k => { const sp = el("span"); sp.appendChild(dot(k)); sp.appendChild(document.createTextNode(lab(k))); lg.appendChild(sp); });
  const C_ = A.conditions_summary, best_ = O.slice().sort((a, b) => N[a].rel_mean_s - N[b].rel_mean_s);
  const overlap = O.every(k => N[k].rel_ci95[0] < 0 && N[k].rel_ci95[1] > 0);
  const span = Math.max(...O.map(k => Math.abs(N[k].mean_s))), pooledAll = A.lam >= 1000;
  document.getElementById("adaptlead").textContent = "How much time the conditions cost each crew through the steering they forced, and how much that varies, measured on " + A.n_runs.toLocaleString() +
    " nominal runs: synthetic races built Monte Carlo style from the day's weather and real HOCR fields, every line rowing the same draws. " +
    "The answer is: very little, and the same for everyone. Crosswind, gusts and traffic changed the time lost to steering by less than " + (Math.ceil(span * 2 + 0.01) / 2).toFixed(1) + " s a race for every coxswain (slightly negative: steering a little less in those conditions), varying by about " +
    Math.max(...O.map(k => N[k].sd_s)).toFixed(1) + " s from one set of conditions to another" +
    (pooledAll ? "; cross-validation preferred one shared response for all " + K.length + " coxswains over any coxswain-specific one, so these races do not show anyone adapting better or worse than the others. " +
     "How to read the chart: further left loses less time, narrower is more consistent. With one shared response, the small shifts between the curves come only from differences in boat speed (the model scales the crosswind by how fast each boat goes), not from steering, and they are a few tenths of a second: no curve here is better or worse in a way that matters." :
     "; " + lab(best_[0]) + " loses least against a typical response (" + f1(N[best_[0]].rel_mean_s) + " s)" + (overlap ? ", but every interval includes zero." : "."));
  document.getElementById("adaptnote").textContent = "Method: the conditions cost a crew time through the extra steering they force. Per 100 m, the mean squared turn rate of the steering part of the track (the weave about the intended line) is regressed on the conditions a coxswain manages: " +
    "felt crosswind (scaled by 1/speed², how far it pushes the boat), gusts, passing, being passed and a crew within 25 m, each crew with its own slopes shrunk towards the pooled ones (partial pooling, strength by cross-validation), with the stroke rate, headwind, bridges, bends and the river's width as controls. " +
    "Seconds from the measured turning loss (speed lost = k × turn rate², the four's k for every crew, so the seconds compare coxswains, not boats). Speed itself was tried as the measure and set aside: a crew passes because it is faster and rows harder into the wind, so speed cannot separate the response from its causes. " +
    "Nominal runs: weather from a random start between 08:00 and 17:00 (one-minute ASOS, felt on the course with the banks' shelter; crosswind " + C_.cross_ms[0].toFixed(1) + "–" + C_.cross_ms[1].toFixed(1) +
    " m/s and gusts " + C_.gust_ms[0].toFixed(1) + "–" + C_.gust_ms[1].toFixed(1) + " m/s above the mean, 10th–90th percentile of the run averages), traffic from one of four real HOCR 2026 fields with the crew at a random bow and pace within it (" +
    C_.passes_per_race[0] + "–" + C_.passes_per_race[1] + " passes and " + C_.passed_per_race[0] + "–" + C_.passed_per_race[1] + " times passed per race). " +
    "Seconds at a common 19:00 pace. Typical response: the pooled slopes on the same runs. Intervals: 500 m block bootstrap of the fit. Own race: the conditions it met, its own response against the typical one. The best in each column is marked.";
}

function operator(){
  const P = D.operator; if (!P) return; document.getElementById("opsec").hidden = false;
  const O = order(), t = document.getElementById("opt"); t.textContent = "";
  const hr = el("tr"); ["Line", "Cost of a 5 m correction (s)", "Back within 1 m (s)", "Overshoot (m)", "Heading anticipation vs optimal", "Phase margin (°)",
    "Gain K (1/s)", "Aim point D (m)", "Delay τ (s): whole / halves", "Lead / lag (s)", "Crossover ωc (rad/s)", "Effective delay (s)",
    "Halves: ωc", "ARX peak lag (s)", "Correction threshold (°)", "Between corrections (s)"].forEach((h, i) => { const th = el("th", h); if (i) th.style.textAlign = "right"; hr.appendChild(th); });
  t.appendChild(hr);
  const fx = (v, d) => v == null ? "–" : Number(v).toFixed(d);
  const M_ = k => P[k].A.step_metrics || {}, minOf = f => Math.min(...O.map(j => f(j) == null ? Infinity : f(j)));
  const bestCost = minOf(k => M_(k).cost_s), bestOver = minOf(k => M_(k).overshoot_m), bestAnt = minOf(k => Math.abs(1 - P[k].C.anticipation)), bestPM = -minOf(k => -P[k].A.pm_deg);
  O.forEach(k => { const c = P[k], a = c.A, p = a.params, m_ = M_(k), tr = el("tr"); const td = el("td"); td.appendChild(dot(k)); td.appendChild(document.createTextNode(lab(k))); tr.appendChild(td);
    [[fx(m_.cost_s, 2), m_.cost_s === bestCost], [m_.settle_s == null ? "over 90" : fx(m_.settle_s, 0), false], [fx(m_.overshoot_m, 1), m_.overshoot_m === bestOver],
     [fx(c.C.anticipation, 2), Math.abs(1 - c.C.anticipation) === bestAnt], [fx(a.pm_deg, 0), a.pm_deg === bestPM],
     [fx(p.K, 2)], [fx(p.D, 0)], [fx(p.tau, 1) + " / " + fx(a.halves.first.tau, 1) + ", " + fx(a.halves.second.tau, 1)], [fx(p.T_L, 1) + " / " + fx(p.T_I, 1)],
     [fx(a.wc, 2)], [fx(a.tau_e, 1)], [fx(a.halves.first.wc, 2) + " / " + fx(a.halves.second.wc, 2)],
     [fx(c.B.peak_lag_s, 1)], [fx(c.D.threshold_deg, 1)], [fx(c.D.interval_s, 0)]].forEach(([v, b]) => tr.appendChild(el("td", v, "n" + (b ? " zero" : ""))));
    t.appendChild(tr); });
  document.getElementById("opguide").textContent = "How to read it: what matters in a race is the time a correction costs, and that comes from turning, not from being off the line. " +
    "Being a few metres to one side of the line you meant costs almost nothing in open water; every hard turn costs speed (speed lost grows with the square of the turn rate), and an overshoot means turning twice. " +
    "So a slow, smooth correction with little overshoot is usually the cheaper one, and a fast correction that overshoots is the most expensive. " +
    "Fast still matters where position matters: lining up for an arch, holding a buoy line, getting out of another crew's way; a correction that takes more than about a minute leaves the boat off line for a whole bridge approach. " +
    "Best values are marked: the cheapest correction, the least overshoot, the heading anticipation closest to optimal (1.00), and the largest phase margin (the least oscillatory loop).";
  const svg = document.getElementById("opstep"); svg.textContent = "";
  const Wd = 760, Hd = 240, ml = 44, mr = 14, mt = 10, mb = 34;
  svg.setAttribute("viewBox", "0 0 " + Wd + " " + Hd);
  const all = O.flatMap(k => P[k].A.step.map(r => r[1])), lo = Math.min(-1, ...all), hi = Math.max(5.5, ...all), T = P[O[0]].A.step[P[O[0]].A.step.length - 1][0];
  const X = v => ml + v / T * (Wd - ml - mr), Y = v => mt + (hi - v) / (hi - lo) * (Hd - mt - mb);
  for (let v = 0; v <= T; v += 15) { svg.appendChild(S("line", {x1: X(v), x2: X(v), y1: mt, y2: Hd - mb, stroke: "var(--line)"}));
    const tx = S("text", {x: X(v), y: Hd - mb + 16, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12}); tx.textContent = v + " s"; svg.appendChild(tx); }
  for (let v = Math.ceil(lo); v <= hi; v += 1) { const tx = S("text", {x: ml - 6, y: Y(v) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 12}); tx.textContent = v + " m"; svg.appendChild(tx); }
  svg.appendChild(S("line", {x1: ml, x2: Wd - mr, y1: Y(0), y2: Y(0), stroke: "var(--muted)", "stroke-dasharray": "4 3"}));
  O.forEach(k => { const p = S("polyline", {points: P[k].A.step.map(r => X(r[0]).toFixed(1) + "," + Y(r[1]).toFixed(1)).join(" "), fill: "none", "stroke-width": 2}); p.style.stroke = colour[k]; svg.appendChild(p); });
  const lg = document.getElementById("opleg"); lg.textContent = ""; O.forEach(k => { const sp = el("span"); sp.appendChild(dot(k)); sp.appendChild(document.createTextNode(lab(k))); lg.appendChild(sp); });
  const wc = O.map(k => P[k].A.wc).filter(v => v != null), te = O.map(k => P[k].A.tau_e).filter(v => v != null);
  document.getElementById("oplead").textContent = "Each coxswain fitted as a human operator in McRuer's crossover model: they aim at a point ahead, react after a delay (entered through a Padé approximant), and add lead or lag. " +
    "Crossover frequencies " + Math.min(...wc).toFixed(2) + "–" + Math.max(...wc).toFixed(2) + " rad/s and effective delays " + Math.min(...te).toFixed(1) + "–" + Math.max(...te).toFixed(1) +
    " s: slow loops with small phase margins, the signature of a weave. " +
    "Every coxswain also uses less heading anticipation than an optimal controller would pair with their lateral correction (" + Math.min(...O.map(k => P[k].C.anticipation)).toFixed(2) + "–" + Math.max(...O.map(k => P[k].C.anticipation)).toFixed(2) +
    " of optimal): each corrects the offset without enough regard for the heading already taking the boat back, so it overshoots. That is the most coachable finding here. The reaction delay itself is not resolvable from per-stroke GPS: its fits move between the race's halves.";
  document.getElementById("opnote").textContent = "A, crossover model (McRuer & Jex 1967), fitted by maximum likelihood with a Kalman filter on the per-stroke track: error e = heading off the intended line + lateral offset / D (an aim point D m ahead); " +
    "command = −K e^(−τs)(T_L s + 1)/(T_I s + 1) e, the delay by the first-order Padé approximant (1 − τs/2)/(1 + τs/2); the boat turns with a first-order lag; wind, wash and stream as a random lateral drift; remnant as white noise. " +
    "Delay: the whole race's fit and each half's; where they disagree the delay is not identified (one GPS fix per stroke, about 2 s apart). Crossover frequency, phase margin and effective delay: from the open loop with the exact delay; a larger crossover means tighter tracking, a larger phase margin smoother, less oscillatory corrections. Halves: the crossover refitted on each half of the race (stability of the fit). " +
    "B, ARX kernel: the heading change per stroke regressed on the error at lags of 0–6 strokes; the lag of the strongest response. " +
    "C, optimal-control check (after Kleinman, Baron & Levison 1970): holding the coxswain's gain on lateral offset, the heading gain an optimal (LQR) controller would use for the same boat; anticipation = the coxswain's heading gain over that. Below 1: too little heading anticipation (overshoot); 1: optimal. " +
    "D, intermittent control (Gawthrop et al. 2011): steering as discrete corrections; the error at which a correction becomes likely (logistic fit), and the median time between them. " +
    "Per-stroke GPS (about one fix every 2 s) cannot resolve reaction times under a second, so delays are those of the whole loop (seeing the drift, the rudder, the boat's yaw), not reflexes.";
}

function render(){ matrix(); profile(); stretch(); best(); steer(); skill(); windows(); rivermap(); adapt(); operator(); wind(); crews();
  document.getElementById("blindnote").hidden = named;
  document.getElementById("modeword").textContent = named ? "Names shown" : "Names hidden";
  const b = document.getElementById("names"); b.setAttribute("aria-pressed", String(named)); b.textContent = named ? "Hide names" : "Show names"; }
const form = document.getElementById("unlock");
document.getElementById("names").addEventListener("click", () => {
  if (named) { named = false; shuffle(); render(); return; }
  if (unlocked) { named = true; render(); return; }
  form.hidden = false; document.getElementById("npass").focus();
});
document.getElementById("ncancel").addEventListener("click", () => { form.hidden = true; document.getElementById("nerr").textContent = ""; });
form.addEventListener("submit", async e => {
  e.preventDefault();
  const err = document.getElementById("nerr"), go = document.getElementById("ngo");
  go.disabled = true; err.textContent = "Opening…";
  try {
    const N = await openBlob("compare-names.enc", document.getElementById("npass").value.trim());
    Object.entries(N.crews).forEach(([id, c]) => Object.assign(D.crews[id], c));
    unlocked = true; named = true; form.hidden = true; err.textContent = ""; document.getElementById("npass").value = ""; render();
  } catch (x) { err.textContent = "That password did not open the names."; }
  go.disabled = false;
});
["showbest", "showshort"].forEach(id => document.getElementById(id).addEventListener("change", profile));
document.getElementById("setsel").addEventListener("change", stretch);
document.getElementById("windsel").addEventListener("change", wind);
document.getElementById("winsel").addEventListener("change", () => { windows(); rivermap(); });
document.getElementById("mapsel").addEventListener("change", rivermap);
render();
})();
"""

if __name__ == "__main__":
    main()
