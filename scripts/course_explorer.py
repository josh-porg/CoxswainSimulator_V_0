r"""An interactive map of the course, its layers, and how the line responds.

    python scripts/course_explorer.py --out out/report/course_explorer.html

Every other figure in this project answers one question with one picture.
This one is for the questions that are really about a *derivative*: what
happens to the fast line when the river drops, or when the wind backs into
the west? Those cannot be read off a static plot, because the interesting
content is the difference between two of them.

So this precomputes the optimised line over a grid of conditions and emits
a self-contained page with three linked views:

**Plan view.** The reach in real coordinates, with a switchable base layer
-- depth, current along the course, or wind at chest height -- in the
manner of a GIS. The line is drawn over whichever layer is showing, so the
same geometry can be read against the thing that shaped it.

**Straightened view.** The same line with the river's curvature taken out:
along-course distance on one axis, offset from the centreline on the
other. A racing line is a small lateral excursion over a very long
distance, and in true proportions it is invisible. This is the view that
shows what the coxswain actually does.

**Profile.** Depth, speed and the chosen offset against distance, so the
line can be read against the water under it.

Why precompute
--------------
Optimising a line takes a few seconds and the browser cannot do it. The
grid is therefore solved here, once, and the page interpolates nothing --
each slider position shows a line that was genuinely optimised for those
conditions, not a blend of two that were.

That is also the honest constraint to state: the page can only show
conditions that were solved for. The grid is coarse on purpose, because a
fine one would take hours and the shape of the answer is what matters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.river import charles as charles_module        # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,    # noqa: E402
                                  optimise_route)

TEMPLATE_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Charles course explorer</title>
<style>
:root {
  --ink: #16202a; --muted: #63727f; --line: #d5dde4; --paper: #fbfcfd;
  --panel: #ffffff; --accent: #0b6f8f; --hot: #b4451f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
       font: 14px/1.5 "Segoe UI", system-ui, sans-serif; }
header { padding: 18px 22px 12px; border-bottom: 1px solid var(--line);
         background: var(--panel); }
h1 { margin: 0 0 3px; font-size: 17px; letter-spacing: -0.01em; }
header p { margin: 0; color: var(--muted); font-size: 12.5px; }
.wrap { display: grid; grid-template-columns: 232px 1fr; gap: 0;
        height: calc(100vh - 62px); }
.side { border-right: 1px solid var(--line); background: var(--panel);
        padding: 16px; overflow-y: auto; }
.side h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .09em;
           color: var(--muted); margin: 0 0 8px; font-weight: 600; }
.side section { margin-bottom: 22px; }
.layer { display: block; width: 100%; text-align: left; padding: 7px 10px;
         margin-bottom: 4px; border: 1px solid var(--line); background: #fff;
         border-radius: 5px; cursor: pointer; font-size: 13px; color: var(--ink); }
.layer[aria-pressed="true"] { background: var(--accent); border-color: var(--accent);
                              color: #fff; }
label.slider { display: block; margin-bottom: 14px; font-size: 12.5px; }
label.slider span { display: flex; justify-content: space-between;
                    color: var(--muted); margin-bottom: 3px; }
label.slider b { color: var(--ink); font-variant-numeric: tabular-nums; }
input[type=range] { width: 100%; accent-color: var(--accent); }
.views { display: grid; grid-template-rows: 1.35fr 0.75fr 0.9fr; gap: 0;
         min-width: 0; }
.view { position: relative; border-bottom: 1px solid var(--line);
        overflow: hidden; }
.view:last-child { border-bottom: 0; }
.view h3 { position: absolute; top: 8px; left: 12px; margin: 0; z-index: 2;
           font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
           color: var(--muted); font-weight: 600;
           background: rgba(251,252,253,.86); padding: 2px 7px; border-radius: 3px; }
canvas { display: block; width: 100%; height: 100%; }
.stats { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12.5px; }
.stat b { display: block; font-size: 17px; font-variant-numeric: tabular-nums;
          letter-spacing: -0.01em; }
.stat span { color: var(--muted); font-size: 11px; }
.legend { position: absolute; right: 12px; top: 8px; z-index: 2;
          background: rgba(251,252,253,.9); border: 1px solid var(--line);
          border-radius: 4px; padding: 6px 8px; font-size: 11px;
          color: var(--muted); }
.legend i { display: inline-block; width: 42px; height: 8px; vertical-align: -1px;
            border-radius: 2px; margin: 0 5px; }
.note { color: var(--muted); font-size: 11.5px; line-height: 1.45; }
</style>
<header>
  <h1>Charles course explorer</h1>
  <p>Optimised line against the layer that shaped it. Each slider position is a
     separately solved optimisation, not an interpolation.</p>
</header>
<div class="wrap">
  <div class="side">
    <section>
      <h2>Base layer</h2>
      <button class="layer" data-layer="depth" aria-pressed="true">Depth</button>
      <button class="layer" data-layer="current" aria-pressed="false">Current along course</button>
      <button class="layer" data-layer="wind" aria-pressed="false">Wind at chest height</button>
      <button class="layer" data-layer="none" aria-pressed="false">Plain</button>
    </section>
    <section id="controls"><h2>Conditions</h2></section>
    <section>
      <h2>This line</h2>
      <div class="stats" id="stats"></div>
    </section>
    <section>
      <p class="note" id="note"></p>
    </section>
  </div>
  <div class="views">
    <div class="view"><h3>Plan</h3><div class="legend" id="legend"></div>
      <canvas id="plan"></canvas></div>
    <div class="view"><h3>Straightened &mdash; offset from centreline</h3>
      <canvas id="straight"></canvas></div>
    <div class="view"><h3>Profile</h3><canvas id="profile"></canvas></div>
  </div>
</div>
<script>
const DATA = """

TEMPLATE_TAIL = r""";

const $ = s => document.querySelector(s);
let layer = "depth";
const axes = DATA.axes, cases = DATA.cases;
const pick = {};
axes.forEach(a => pick[a.key] = Math.floor(a.values.length / 2));

// -- controls ------------------------------------------------------------
const controls = $("#controls");
axes.forEach(a => {
  const label = document.createElement("label");
  label.className = "slider";
  label.innerHTML = `<span>${a.label}<b data-out="${a.key}"></b></span>`;
  const input = document.createElement("input");
  input.type = "range"; input.min = 0; input.max = a.values.length - 1;
  input.step = 1; input.value = pick[a.key];
  input.addEventListener("input", () => {
    pick[a.key] = +input.value; draw();
  });
  label.appendChild(input);
  controls.appendChild(label);
});

document.querySelectorAll(".layer").forEach(b => {
  b.addEventListener("click", () => {
    layer = b.dataset.layer;
    document.querySelectorAll(".layer").forEach(o =>
      o.setAttribute("aria-pressed", String(o === b)));
    draw();
  });
});

function current() {
  // Index, not value: JS renders 0.0 as "0" and Python as "0.0",
  // so a value-keyed lookup misses on every whole number.
  const key = axes.map(a => pick[a.key]).join("|");
  return cases[key];
}

// -- colour ramps --------------------------------------------------------
function ramp(t, stops) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n));
  const u = t * n - i, a = stops[i], b = stops[i + 1];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*u)},${Math.round(a[1]+(b[1]-a[1])*u)},${Math.round(a[2]+(b[2]-a[2])*u)})`;
}
const RAMPS = {
  depth:   [[232,242,247],[142,199,222],[46,124,168],[13,58,94]],
  current: [[247,244,236],[224,196,140],[190,132,58],[120,66,20]],
  wind:    [[244,246,240],[186,214,178],[110,166,120],[36,92,74]],
};
const LABELS = {
  depth: ["shallow", "deep", "m"],
  current: ["weak", "strong against", "m/s"],
  wind: ["sheltered", "exposed", "m/s"],
};

function fit(canvas) {
  const r = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * r; canvas.height = h * r;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(r, 0, 0, r, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

// -- plan view -----------------------------------------------------------
function drawPlan(c) {
  const [ctx, w, h] = fit($("#plan"));
  const pad = 26;
  const gx = DATA.grid.x, gy = DATA.grid.y;
  const x0 = Math.min(...gx), x1 = Math.max(...gx);
  const y0 = Math.min(...gy), y1 = Math.max(...gy);
  const s = Math.min((w - 2*pad) / (x1 - x0), (h - 2*pad) / (y1 - y0));
  const ox = (w - (x1 - x0) * s) / 2, oy = (h - (y1 - y0) * s) / 2;
  const X = v => ox + (v - x0) * s, Y = v => h - oy - (v - y0) * s;

  if (layer !== "none") {
    const field = layer === "wind" ? c.wind : DATA.grid[layer];
    const lo = DATA.grid.range[layer][0], hi = DATA.grid.range[layer][1];
    const cw = Math.abs(X(gx[1]) - X(gx[0])) + 1;
    const ch = Math.abs(Y(gy[1]) - Y(gy[0])) + 1;
    for (let j = 0; j < gy.length; j++) {
      for (let i = 0; i < gx.length; i++) {
        const v = field[j * gx.length + i];
        if (v === null) continue;
        ctx.fillStyle = ramp((v - lo) / (hi - lo), RAMPS[layer]);
        ctx.fillRect(X(gx[i]) - cw/2, Y(gy[j]) - ch/2, cw, ch);
      }
    }
    const [a, b, unit] = LABELS[layer];
    // Adaptive precision: the along-course current on this river spans
    // 0.001-0.03 m/s, which one decimal renders as "0.0-0.0".
    const dp = (hi - lo) < 0.05 ? 3 : (hi - lo) < 0.5 ? 2 : 1;
    $("#legend").innerHTML = `${a}<i style="background:linear-gradient(90deg,${
      ramp(0,RAMPS[layer])},${ramp(1,RAMPS[layer])})"></i>${b}
      &nbsp;<b>${lo.toFixed(dp)}&ndash;${hi.toFixed(dp)} ${unit}</b>`;
  } else {
    $("#legend").innerHTML = "";
  }

  ctx.strokeStyle = "rgba(99,114,127,.55)"; ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]); ctx.beginPath();
  DATA.centreline.forEach((p, k) => k ? ctx.lineTo(X(p[0]), Y(p[1]))
                                      : ctx.moveTo(X(p[0]), Y(p[1])));
  ctx.stroke(); ctx.setLineDash([]);

  ctx.strokeStyle = "#b4451f"; ctx.lineWidth = 2.4; ctx.beginPath();
  c.path.forEach((p, k) => k ? ctx.lineTo(X(p[0]), Y(p[1]))
                             : ctx.moveTo(X(p[0]), Y(p[1])));
  ctx.stroke();

  // Bridges cluster within a few hundred metres around Weeks, so a fixed
  // label offset overprints three of them.  Alternate the side and step
  // the vertical offset instead.
  ctx.font = "11px system-ui";
  DATA.landmarks.forEach((m, k) => {
    ctx.fillStyle = "#16202a";
    ctx.beginPath(); ctx.arc(X(m.x), Y(m.y), 2.8, 0, 7); ctx.fill();
    const up = k % 2 === 0;
    const dy = up ? -8 - (k % 4) * 5 : 13 + (k % 4) * 5;
    ctx.strokeStyle = "rgba(99,114,127,.45)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(m.x), Y(m.y));
    ctx.lineTo(X(m.x), Y(m.y) + dy + (up ? 3 : -3)); ctx.stroke();
    ctx.fillStyle = "#63727f"; ctx.textAlign = "center";
    ctx.fillText(m.name, X(m.x), Y(m.y) + dy);
    ctx.textAlign = "left";
  });
}

// -- straightened --------------------------------------------------------
function drawStraight(c) {
  const [ctx, w, h] = fit($("#straight"));
  const padL = 44, padR = 14, padT = 26, padB = 22;
  const n = c.station.length, L = c.station[n-1];
  const lim = DATA.halfWidthMax;
  const X = s => padL + (s / L) * (w - padL - padR);
  const Y = o => padT + (0.5 - o / (2*lim)) * (h - padT - padB);

  ctx.strokeStyle = "#eef2f5"; ctx.lineWidth = 1;
  for (let o = -Math.floor(lim/10)*10; o <= lim; o += 10) {
    ctx.beginPath(); ctx.moveTo(padL, Y(o)); ctx.lineTo(w - padR, Y(o)); ctx.stroke();
  }
  ctx.fillStyle = "rgba(11,111,143,.10)"; ctx.beginPath();
  for (let i = 0; i < n; i++) ctx.lineTo(X(c.station[i]), Y(DATA.halfWidth[i]));
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(X(c.station[i]), Y(-DATA.halfWidth[i]));
  ctx.closePath(); ctx.fill();

  ctx.strokeStyle = "rgba(99,114,127,.6)"; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0));
  ctx.stroke(); ctx.setLineDash([]);

  ctx.strokeStyle = "#b4451f"; ctx.lineWidth = 2.2; ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = X(c.station[i]), q = Y(c.offset[i]);
    i ? ctx.lineTo(p, q) : ctx.moveTo(p, q);
  }
  ctx.stroke();

  ctx.fillStyle = "#63727f"; ctx.font = "10.5px system-ui";
  ctx.fillText("port +" + lim.toFixed(0) + " m", 6, Y(lim) + 10);
  ctx.fillText("stbd", 6, Y(-lim) - 2);
  // Rotated, because the bridges are minutes apart on a 12 km axis and
  // horizontal labels overprint into an unreadable smear.
  DATA.landmarks.forEach(m => {
    if (m.station == null) return;
    ctx.strokeStyle = "rgba(99,114,127,.22)"; ctx.beginPath();
    ctx.moveTo(X(m.station), padT); ctx.lineTo(X(m.station), h - padB);
    ctx.stroke();
    ctx.save(); ctx.translate(X(m.station) + 3, h - padB - 3);
    ctx.rotate(-Math.PI / 2); ctx.fillStyle = "#63727f";
    ctx.fillText(m.name, 0, 0); ctx.restore();
  });
}

// -- profile -------------------------------------------------------------
function drawProfile(c) {
  const [ctx, w, h] = fit($("#profile"));
  const padL = 44, padR = 14, padT = 24, padB = 24;
  const n = c.station.length, L = c.station[n-1];
  const X = s => padL + (s / L) * (w - padL - padR);
  const series = [
    { v: c.depth, color: "#0b6f8f", label: "depth m" },
    { v: c.speed, color: "#b4451f", label: "ground speed m/s" },
  ];
  series.forEach((ser, k) => {
    const lo = Math.min(...ser.v), hi = Math.max(...ser.v);
    const Y = v => padT + (1 - (v - lo) / ((hi - lo) || 1)) * (h - padT - padB);
    ctx.strokeStyle = ser.color; ctx.lineWidth = 1.8; ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const p = X(c.station[i]), q = Y(ser.v[i]);
      i ? ctx.lineTo(p, q) : ctx.moveTo(p, q);
    }
    ctx.stroke();
    ctx.fillStyle = ser.color; ctx.font = "10.5px system-ui";
    ctx.textAlign = "right";
    ctx.fillText(`${ser.label}  ${lo.toFixed(2)}–${hi.toFixed(2)}`,
                 w - padR, padT - 12 + k * 11);
    ctx.textAlign = "left";
  });
  ctx.fillStyle = "#63727f"; ctx.font = "10.5px system-ui";
  ctx.fillText("0 m", padL, h - 8);
  ctx.fillText((L/1000).toFixed(1) + " km", w - padR - 34, h - 8);
}

function draw() {
  axes.forEach(a => {
    const out = document.querySelector(`[data-out="${a.key}"]`);
    if (out) out.textContent = a.format.replace("%", a.values[pick[a.key]]);
  });
  const c = current();
  if (!c) { $("#stats").innerHTML = "<span class='note'>not solved</span>"; return; }
  drawPlan(c); drawStraight(c); drawProfile(c);
  $("#stats").innerHTML = `
    <div class="stat"><b>${(c.elapsed/60).toFixed(2)}</b><span>minutes</span></div>
    <div class="stat"><b>${c.length.toFixed(0)}</b><span>metres rowed</span></div>
    <div class="stat"><b>${c.minDepth.toFixed(2)}</b><span>min depth m</span></div>
    <div class="stat"><b>${c.maxOffset.toFixed(0)}</b><span>max offset m</span></div>`;
  $("#note").textContent = c.note || "";
}

window.addEventListener("resize", draw);
draw();
</script>
"""


#: The bridges a HOCR entry passes, downstream to upstream.
BRIDGES = (
    ("BU", "BU_BRIDGE"),
    ("River St", "RIVER_ST_BRIDGE"),
    ("Western Ave", "WESTERN_AVE_BRIDGE"),
    ("Weeks", "WEEKS_FOOTBRIDGE"),
    ("Anderson", "LARZ_ANDERSON_BRIDGE"),
    ("Eliot", "ELIOT_BRIDGE"),
)


def landmark_stations(course, raster):
    """Bridges in local coordinates, with their along-course station.

    ``landmark_station`` already solves the projection and the nearest
    point on the channel, and it reports the OFFSET as well -- a large one
    means the published coordinate and the extracted channel disagree, so
    it is worth carrying rather than silently dropping.
    """
    from coxswain.river.charles import landmark_station, local_tangent_plane

    out = []
    for label, attribute in BRIDGES:
        latlon = getattr(charles_module, attribute, None)
        if latlon is None:
            continue
        east, north = local_tangent_plane(latlon[0], latlon[1],
                                          charles_module.CHARLES_ORIGIN)
        station, offset = landmark_station(latlon, raster)
        out.append({"name": label, "x": round(float(east), 1),
                    "y": round(float(north), 1),
                    "station": round(float(station), 1)
                    if station <= course.length else None,
                    "offset": round(float(offset), 1)})
    return out


def sample_grid(course, raster, step=45.0):
    """Depth and along-course current on a regular grid over the water."""
    line = np.array([course.position_at(s)
                     for s in np.linspace(0.0, course.length, 500)])
    low = line.min(axis=0) - 130.0
    high = line.max(axis=0) + 130.0
    gx = np.arange(low[0], high[0], step)
    gy = np.arange(low[1], high[1], step)

    depth, current = [], []
    for y in gy:
        for x in gx:
            row, column = raster.index_of(x, y)
            if not bool(raster.water[row, column]):
                depth.append(None)
                current.append(None)
                continue
            depth.append(round(float(course.depth_at(x, y)), 3))
            flow = np.asarray(course.current_at(x, y))[:2]
            current.append(round(float(np.hypot(*flow)), 4))
    return gx, gy, depth, current


def wind_layer(gx, gy, raster, structures, speed, bearing):
    """Chest-height wind speed over the water, one value per grid cell."""
    if speed <= 0.0:
        return [None] * (len(gx) * len(gy))
    from coxswain.hydro.canopy import ShelteredWind
    field = ShelteredWind(structures, raster, speed, bearing, height=1.5)
    out = []
    for y in gy:
        for x in gx:
            row, column = raster.index_of(x, y)
            out.append(None if not bool(raster.water[row, column])
                       else round(float(field.speed_at(x, y)), 3))
    return out


def solve_case(course, boat, level, wind_speed, wind_from, structures,
               raster, n_control, iterations):
    """Optimise the line for one set of conditions."""
    from coxswain.hydro.canopy import ShelteredWind

    shifted = charles_module.charles_course(level_offset=level)
    evaluator = RouteEvaluator(shifted, boat=boat)
    if wind_speed > 0.0:
        evaluator = evaluator.with_wind(
            ShelteredWind(structures, raster, wind_speed, wind_from,
                          height=1.5), boat=boat)
    best = optimise_route(evaluator, n_control=n_control,
                          iterations=iterations)
    route, station = best.route, best.station
    path = route.path(shifted, n=260)
    offsets = route.offset_at(station)
    keep = np.linspace(0, len(station) - 1, 160).astype(int)
    return {
        "path": [[round(float(a), 1), round(float(b), 1)]
                 for a, b in path],
        "station": [round(float(s), 1) for s in station[keep]],
        "offset": [round(float(o), 2) for o in offsets[keep]],
        "depth": [round(float(d), 3) for d in best.depth[keep]],
        "speed": [round(float(v), 4) for v in best.speed_ground[keep]],
        "elapsed": round(float(best.elapsed_clean), 2),
        "length": round(float(best.path_length), 1),
        "minDepth": round(float(best.depth.min()), 3),
        "maxOffset": round(float(np.abs(offsets).max()), 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/report/course_explorer.html")
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[-0.4, -0.2, 0.0, 0.2])
    parser.add_argument("--winds", type=float, nargs="+",
                        default=[0.0, 4.0, 8.0])
    parser.add_argument("--wind-from", type=float, default=250.0)
    parser.add_argument("--n-control", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--grid-step", type=float, default=45.0)
    args = parser.parse_args(argv)

    from coxswain.river.structures import charles_structures

    boat = catalog.eight(rate=32.0, rower_mass=68.0, rower_stature=1.70,
                         coxswain_mass=68.0)
    course = charles_module.charles_course()
    raster = charles_module.charles_channel()
    structures = charles_structures()

    print("sampling the base layers")
    gx, gy, depth, current = sample_grid(course, raster, args.grid_step)

    station = np.linspace(0.0, course.length, 160)
    half_width = [round(float(course.half_width_at(s)), 2) for s in station]
    centre = [[round(float(p[0]), 1), round(float(p[1]), 1)]
              for p in (course.position_at(s)
                        for s in np.linspace(0.0, course.length, 300))]

    winds = {}
    for speed in args.winds:
        winds[speed] = wind_layer(gx, gy, raster, structures, speed,
                                  args.wind_from)

    cases = {}
    total = len(args.levels) * len(args.winds)
    done = 0
    for level_index, level in enumerate(args.levels):
        for wind_index, speed in enumerate(args.winds):
            done += 1
            print("  solving %d/%d  level %+.2f m  wind %.1f m/s"
                  % (done, total, level, speed))
            case = solve_case(course, boat, level, speed, args.wind_from,
                              structures, raster, args.n_control,
                              args.iterations)
            case["wind"] = winds[speed]
            case["note"] = ("Water %+.2f m against the October median; "
                            "wind %.0f m/s from %.0f deg."
                            % (level, speed, args.wind_from))
            cases["%d|%d" % (level_index, wind_index)] = case

    finite = [d for d in depth if d is not None]
    flows = [c for c in current if c is not None]
    wind_all = [v for layer in winds.values() for v in layer
                if v is not None] or [0.0, 1.0]

    payload = {
        "axes": [
            {"key": "level", "label": "Water level",
             "format": "% m", "values": list(args.levels)},
            {"key": "wind", "label": "Wind speed",
             "format": "% m/s", "values": list(args.winds)},
        ],
        "grid": {
            "x": [round(float(v), 1) for v in gx],
            "y": [round(float(v), 1) for v in gy],
            "depth": depth, "current": current,
            "range": {
                "depth": [round(min(finite), 2), round(max(finite), 2)],
                "current": [round(min(flows), 3), round(max(flows), 3)],
                "wind": [round(min(wind_all), 2), round(max(wind_all), 2)],
            },
        },
        "centreline": centre,
        "halfWidth": half_width,
        "halfWidthMax": round(float(max(half_width)), 1),
        "landmarks": landmark_stations(course, raster),
        "cases": cases,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(TEMPLATE_HEAD)
        json.dump(payload, handle, separators=(",", ":"))
        handle.write(TEMPLATE_TAIL)
    print("wrote %s  (%.1f MB)"
          % (args.out, os.path.getsize(args.out) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
