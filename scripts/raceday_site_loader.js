  // ---- self-updating site: unlock, decrypt the bundle, fetch weather and flow, rebuild each card ----
  var COMPASS = {N:0,NNE:22.5,NE:45,ENE:67.5,E:90,ESE:112.5,SE:135,SSE:157.5,S:180,SSW:202.5,SW:225,WSW:247.5,W:270,WNW:292.5,NW:315,NNW:337.5};
  var MPH = 0.44704, POINT = "42.37,-71.122", USGS = "https://waterservices.usgs.gov/nwis/iv/?sites=01104500&parameterCd=00060&format=json&period=P1D";
  var bundle = null, live = {};

  function b64(s){ return Uint8Array.from(atob(s), function(c){ return c.charCodeAt(0); }); }
  async function decrypt(pass){
    var enc = await (await fetch("data.enc", {cache: "no-cache"})).json();
    var km = await crypto.subtle.importKey("raw", new TextEncoder().encode(pass), "PBKDF2", false, ["deriveKey"]);
    var key = await crypto.subtle.deriveKey({name: "PBKDF2", salt: b64(enc.salt), iterations: enc.iterations, hash: "SHA-256"},
                                            km, {name: "AES-GCM", length: 256}, false, ["decrypt"]);
    var pt = await crypto.subtle.decrypt({name: "AES-GCM", iv: b64(enc.iv)}, key, b64(enc.data));
    return JSON.parse(new TextDecoder().decode(pt));
  }
  function numbers(t){ return ((t || "").match(/\d+(\.\d+)?/g) || []).map(Number); }
  function upperMph(t){ var n = numbers(t); return n.length ? Math.max.apply(null, n) * MPH : NaN; }
  function midMph(t){ var n = numbers(t); return n.length ? 0.5 * (Math.min.apply(null, n) + Math.max.apply(null, n)) * MPH : NaN; }
  function celsius(p){ return p.temperature == null ? null : (p.temperatureUnit === "F" ? (p.temperature - 32) * 5 / 9 : p.temperature); }
  async function getJSON(u){ var r = await fetch(u, {headers: {Accept: "application/geo+json"}}); if (!r.ok) throw new Error(r.status); return r.json(); }

  function isoSpan(v){                          // "2026-10-17T14:00:00+00:00/PT3H" -> [start, end] in ms
    var p = v.split("/"), t0 = Date.parse(p[0]), m = /P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?/.exec(p[1] || "") || [];
    return [t0, t0 + (((+m[1] || 0) * 24 + (+m[2] || 0)) * 60 + (+m[3] || 0)) * 60e3];
  }
  async function gustAt(t){                     // the hourly forecast leaves gusts empty; the grid has them (km/h)
    try {
      var pts = await getJSON("https://api.weather.gov/points/" + POINT);
      var vals = (await getJSON(pts.properties.forecastGridData)).properties.windGust.values;
      for (var i = 0; i < vals.length; i++){ var se = isoSpan(vals[i].validTime); if (se[0] <= t && t < se[1] && vals[i].value != null) return vals[i].value / 3.6; }
    } catch (e) {}
    return null;
  }
  async function weatherAt(startIso){
    var wx = await weatherOnly(startIso);
    if (/^forecast/.test(wx.kind || "") && !(wx.gust > 0)) wx.gust = await gustAt(Date.parse(startIso));
    return wx;
  }
  async function weatherOnly(startIso){
    var t = Date.parse(startIso), pts = await getJSON("https://api.weather.gov/points/" + POINT);
    try {
      var hr = (await getJSON(pts.properties.forecastHourly)).properties.periods;
      for (var i = 0; i < hr.length; i++){ var p = hr[i];
        if (Date.parse(p.startTime) <= t && t < Date.parse(p.endTime))
          return {kind: "forecast for the start", speed: upperMph(p.windSpeed), direction: COMPASS[p.windDirection],
                  temperature_c: celsius(p), rain_pct: (p.probabilityOfPrecipitation || {}).value || 0, sky: p.shortForecast, at: p.startTime};
      }
    } catch (e) {}
    try {
      var dy = (await getJSON(pts.properties.forecast)).properties.periods;
      for (var j = 0; j < dy.length; j++){ var q = dy[j];
        if (Date.parse(q.startTime) <= t && t < Date.parse(q.endTime))
          return {kind: "forecast for race day (12-hour period; hourly not yet in range)", speed: midMph(q.windSpeed),
                  direction: COMPASS[q.windDirection], temperature_c: celsius(q), rain_pct: (q.probabilityOfPrecipitation || {}).value || 0,
                  sky: q.name + ": " + q.shortForecast + " (" + q.windSpeed + ")", at: q.startTime};
      }
    } catch (e) {}
    var o = (await getJSON("https://api.weather.gov/stations/KBOS/observations/latest")).properties;
    return {kind: "observed now at Logan (race-time forecast not yet in range)", speed: (o.windSpeed.value || 0) / 3.6,
            direction: o.windDirection.value, temperature_c: o.temperature.value, at: o.timestamp,
            gust: o.windGust && o.windGust.value != null ? o.windGust.value / 3.6 : null};
  }
  async function flowNow(){
    var d = await (await fetch(USGS)).json(), v = d.value.timeSeries[0].values[0].value;
    var last = v[v.length - 1];
    return {cfs: Number(last.value), m3s: Number(last.value) * 0.0283168466, at: last.dateTime};
  }

  var NWM = "https://api.water.noaa.gov/nwps/v1/reaches/5868225/streamflow?series=";
  var LEVEL = "https://waterservices.usgs.gov/nwis/iv/?sites=01104715&parameterCd=00065&format=json&period=P1D", LEVEL_MEDIAN_FT = 107.32;
  async function riverAt(startIso){             // forecast flow at the start (NWM) and the basin level now
    var t = Date.parse(startIso), out = {};
    var runs = [["short_range", "shortRange"], ["medium_range", "mediumRange"]];
    for (var i = 0; i < runs.length && out.cfs == null; i++){
      try {
        var blk = (await (await fetch(NWM + runs[i][0])).json())[runs[i][1]] || {};
        var mem = blk.series || blk.mean || blk[Object.keys(blk)[0]] || {}, best = null;
        (mem.data || []).forEach(function(x){ var d = Math.abs(Date.parse(x.validTime) - t); if (!best || d < best[0]) best = [d, x.flow]; });
        if (best && best[0] <= 3 * 3600e3){ out.cfs = best[1]; out.source = "National Water Model, " + runs[i][0].replace("_", " "); }
        else if (i === 0 && mem.data && mem.data.length){             // the start beyond the short range: its last hour, if within 18 h
          var lastS = mem.data[mem.data.length - 1], gap = t - Date.parse(lastS.validTime);
          if (gap >= 0 && gap <= 18 * 3600e3) out.fallback = {cfs: lastS.flow, source: "National Water Model, short range's last hour (the start not yet in range)"};
        }
      } catch (e) {}
    }
    if (out.cfs == null && out.fallback){ out.cfs = out.fallback.cfs; if (out.fallback.source) out.source = out.fallback.source; }
    delete out.fallback;
    try { var v = (await (await fetch(LEVEL)).json()).value.timeSeries[0].values[0].value; var last = v[v.length - 1];
          out.level_ft = Number(last.value) - LEVEL_MEDIAN_FT; out.level_at = last.dateTime; } catch (e) {}
    return out;
  }
  function shortOf(c, pre){                      // the shortened course: same as card.short_course in Python
    var i0 = c.sections.findIndex(function(x){ return x.start === pre.course.start; });
    if (i0 < 0) return null;
    var s = JSON.parse(JSON.stringify(c)); delete s.short_card;
    var secs = s.sections.slice(i0), remain = secs.reduce(function(a, x){ return a + x.share; }, 0);
    secs.forEach(function(x){ x.share = x.share / remain; });
    s.sections = secs;
    var path = secs.reduce(function(a, x){ return a + x.path_m; }, 0);
    s.targets.forEach(function(t){
      ["central", "fast", "slow"].forEach(function(k){
        var before = i0 > 0 ? t["at_landmark_" + k][i0 - 1] : 0;
        t["at_landmark_" + k] = t["at_landmark_" + k].slice(i0).map(function(v){ return v - before; });
        t["splits_" + k] = t["splits_" + k].slice(i0); t["rudder_" + k] = t["rudder_" + k].slice(i0);
        t["today_" + k] = t["at_landmark_" + k][t["at_landmark_" + k].length - 1];
        t["split_overall_" + k] = t["today_" + k] / path * 500; t[k] = t[k] * remain;
      });
      delete t.lake_split;
    });
    s.course = {name: "short", start: pre.course.start, start_m: secs[0].from_m, length_m: path, share: remain};
    s.traffic = pre.traffic;
    return s;
  }
  function lerp(a, b, w){ return a + (b - a) * w; }
  function mix(rows){                           // rows: [[row, weight], ...] -> weighted row
    var out = {factor: 0, share: null, felt: null, rudder: null};
    rows.forEach(function(rw){ var r = rw[0], w = rw[1];
      out.factor += w * r.factor;
      ["share", "felt", "rudder"].forEach(function(k){
        if (!out[k]) out[k] = r[k].map(function(){ return 0; });
        r[k].forEach(function(x, i){ out[k][i] += w * x; });
      });
    });
    var s = out.share.reduce(function(a, b){ return a + b; }, 0);
    out.share = out.share.map(function(x){ return x / s; });
    return out;
  }
  var PEAK_FACTOR = 3, GZ = [];                 // gusts: as coxswain/raceday/gusts.py
  for (var gz = -3; gz <= 3.0001; gz += 0.25) GZ.push([gz, Math.exp(-0.5 * gz * gz)]);
  function gustIntensity(speed, gust){ return (gust > speed && speed > 0.3) ? Math.min((gust / speed - 1) / PEAK_FACTOR, 0.5) : 0; }
  function interpGusts(T, speed, dir, gust){
    var sig = gustIntensity(speed, gust) * speed;
    if (!(sig > 0)) return interp(T, speed, dir);
    var tot = GZ.reduce(function(a, z){ return a + z[1]; }, 0);
    var out = mix(GZ.map(function(z){ return [interp(T, Math.max(0, speed + sig * z[0]), dir), z[1] / tot]; }));
    var S = T.speeds, h = S[1] - S[0], u = Math.min(Math.max(speed, S[0] + h), S[S.length - 1] - h);
    var curv = (interp(T, u + h, dir).factor - 2 * interp(T, u, dir).factor + interp(T, u - h, dir).factor) / (h * h);
    out.factor = interp(T, speed, dir).factor + 0.5 * curv * sig * sig;   // exact for a quadratic cost; see gusts.py
    return out;
  }
  function interp(T, speed, dir){
    var S = T.speeds, D = T.directions, n = D.length, v = Math.max(0, Math.min(speed || 0, S[S.length - 1]));
    var i = 0; while (i < S.length - 2 && v > S[i + 1]) i++;
    var ws = (v - S[i]) / (S[i + 1] - S[i]);
    var x = (((isFinite(dir) ? dir : 0) % 360) + 360) % 360 / (360 / n), j = Math.floor(x) % n, wd = x - Math.floor(x), j1 = (j + 1) % n;
    function row(a, b){ return T.rows[a * n + b]; }
    return mix([[row(i, j), (1 - ws) * (1 - wd)], [row(i, j1), (1 - ws) * wd], [row(i + 1, j), ws * (1 - wd)], [row(i + 1, j1), ws * wd]]);
  }
  function labelOf(rel){
    if (!isFinite(rel)) return "calm";
    var r = ((rel % 360) + 360) % 360;
    if (r <= 45 || r >= 315) return "headwind";
    if (r >= 135 && r <= 225) return "tailwind";
    return r < 180 ? "cross from starboard" : "cross from port";
  }
  function interp1(xs, ys, x){
    if (x <= xs[0]) return ys[0];
    for (var i = 0; i < xs.length - 1; i++) if (x <= xs[i + 1]) return lerp(ys[i], ys[i + 1], (x - xs[i]) / (xs[i + 1] - xs[i]));
    return ys[ys.length - 1];
  }
  var BASE_2025 = 1.02;
  function dayFactor(){
    var rows = []; try { rows = JSON.parse(load("day") || "[]"); } catch (e) {}
    var by = {};
    rows.forEach(function(r){ if (r.today > 0 && r.last > 0) (by[r.event] = by[r.event] || []).push(r.today / r.last); });
    function med(a){ a = a.slice().sort(function(x, y){ return x - y; }); var m = a.length >> 1; return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]); }
    var per = Object.keys(by).map(function(k){ return med(by[k]); });
    if (!per.length) return null;
    var lo = Math.min.apply(null, per), hi = Math.max.apply(null, per);
    // against the still-air goals: last year's Saturday ran BASE_2025 slower than them (scripts/day_from_results.py)
    return {factor: med(per) * BASE_2025, low: lo * BASE_2025, high: hi * BASE_2025, events: per.length, vs_last: med(per)};
  }

  function buildCard(cd, wx, fl, rv){
    var c = JSON.parse(JSON.stringify(cd.card)), T = cd.wind;
    var calm = !(wx.speed > 0.3) || !isFinite(wx.direction);
    var w = calm ? interp(T, 0, 0) : interp(T, wx.speed, wx.direction);
    c.conditions = wx;
    c.wind = calm ? null : {speed: wx.speed, direction: wx.direction};
    c.wind_factor = w.factor; c.wind_cost_pct = 100 * (w.factor - 1);
    c.line_s = T.still_air_s ? T.still_air_s * w.factor : null;     // this line today at our pace (river below)
    c.gust = null;
    if (!calm && wx.gust > wx.speed){
      var wg = interpGusts(T, wx.speed, wx.direction, wx.gust), gI = gustIntensity(wx.speed, wx.gust);
      c.gust = {gust_ms: wx.gust, intensity: gI, sigma_ms: gI * wx.speed, ratio: wg.factor / w.factor,
                wind_ratio: Math.abs(w.factor - 1) > 1e-6 ? (wg.factor - 1) / (w.factor - 1) : 1, added_s: (wg.factor - w.factor) * T.reference_s};
    }
    c.wind_cost_pct_range = calm ? null : [100 * (interp(T, 0.7 * wx.speed, wx.direction).factor - 1), 100 * (interp(T, 1.3 * wx.speed, wx.direction).factor - 1)];
    // the day factor: earlier races typed in on this phone, else the one the race-morning job read from the
    // published results (scripts/day_from_results.py -> the card's ``day``)
    var factor = w.factor, day = dayFactor();
    if (!day && cd.card.day && cd.card.day.factor > 0)
      day = {factor: cd.card.day.factor, low: cd.card.day.low, high: cd.card.day.high, events: cd.card.day.events, auto: true,
             vs_last: cd.card.day.factor / BASE_2025};
    c.day = day; c.factor_source = null;
    if (day){ c.model_wind_factor = factor; factor = day.factor; c.wind_factor = factor; c.factor_source = day.events + " earlier events today against last year" + (day.auto ? " (from the published results)" : ""); }
    c.sections.forEach(function(s, i){
      s.share = w.share[i]; s.wind_felt = calm ? null : w.felt[i];
      s.relative = calm ? NaN : (((wx.direction - T.headings[i]) % 360) + 360) % 360; s.wind_label = labelOf(s.relative);
    });
    var path = c.sections.reduce(function(a, s){ return a + s.path_m; }, 0);
    c.targets.forEach(function(t){
      ["central", "fast", "slow"].forEach(function(k){
        var today = t[k] * factor, cum = 0;
        t["today_" + k] = today;
        t["split_overall_" + k] = today / path * 500;
        t["splits_" + k] = c.sections.map(function(s, i){ return today * w.share[i] / s.path_m * 500; });
        t["at_landmark_" + k] = c.sections.map(function(s, i){ cum += today * w.share[i]; return cum; });
        t["rudder_" + k] = c.sections.map(function(s, i){ return today * w.rudder[i]; });
      });
    });
    if (fl) c.flow = {ok: true, discharge_cfs: fl.cfs, discharge_m3s: fl.m3s, at: fl.at, october_median_m3s: cd.flow.october_median_m3s,
                      effect_s: interp1(cd.flow.discharge_m3s, cd.flow.effect_fraction, fl.m3s) * cd.wind.reference_s};
    c.river = null;
    if (rv && (rv.cfs != null || rv.level_ft != null)){
      var ref = cd.wind.reference_s, q = rv.cfs != null ? rv.cfs * 0.0283168466 : null;
      var flow_s = q != null ? interp1(cd.flow.discharge_m3s, cd.flow.effect_fraction, q) * ref : 0;
      var level_s = (rv.level_ft != null && cd.level) ? interp1(cd.level.ft, cd.level.s, rv.level_ft) : 0;
      c.river = {discharge_cfs: rv.cfs, source: rv.source, level_ft: rv.level_ft, level_at: rv.level_at,
                 flow_s: flow_s, level_s: level_s, effect_s: flow_s + level_s, factor: 1 + (flow_s + level_s) / ref};
      if (!day) c.targets.forEach(function(t){
        ["central", "fast", "slow"].forEach(function(k){
          var f = c.river.factor;
          t["today_" + k] *= f; t["split_overall_" + k] *= f;
          ["splits_", "at_landmark_", "rudder_"].forEach(function(n){ t[n + k] = t[n + k].map(function(x){ return x * f; }); });
        });
      });
    }
    if (c.line_s) c.line_s *= (day ? day.factor / w.factor : 1) * (c.river && !day ? c.river.factor : 1);
    c.course = {name: "full", start: "Start", start_m: 0, length_m: path, share: 1};
    if (cd.card.short_card) c.short_card = shortOf(c, cd.card.short_card);
    var C = (c.plan && c.plan.variants || []).filter(function(v){ return v.letter === "C"; })[0];
    if (C){
      var runs = [], cur = null;
      c.sections.forEach(function(s){ if (s.wind_label === "headwind"){ cur = cur ? [cur[0], s.end] : [s.start, s.end]; } else if (cur){ runs.push(cur); cur = null; } });
      if (cur) runs.push(cur);
      C.do[1] = "Splits slow where the wind is ahead (" + (runs.map(function(r){ return r[0] + " to " + r[1]; }).join(", ") || "none today") + "); judge them against the card, not last week.";
    }
    c.generated_at = new Date().toISOString();
    return c;
  }

  async function refresh(){
    var flow = null; try { flow = await flowNow(); } catch (e) {}
    var rivers = {};
    for (var i = 0; i < CREWS.length; i++){
      var k = CREWS[i], cd = bundle.crews[k];
      if (!cd) continue;
      try {
        var wx = live[cd.card.race_start] || (live[cd.card.race_start] = await weatherAt(cd.card.race_start));
        var rv = rivers[cd.card.race_start] || (rivers[cd.card.race_start] = await riverAt(cd.card.race_start));
        cards[k] = buildCard(cd, wx, flow, rv); sources[k] = "live";
        save(k, JSON.stringify(cards[k]));
        (cd.alts || []).forEach(function(a){
          var ak = k + "|" + a.key;
          cards[ak] = buildCard({card: a.card, wind: a.wind, flow: a.flow, level: cd.level}, wx, flow, rv); sources[ak] = "live";
          save(ak, JSON.stringify(cards[ak]));
        });
      } catch (e) {
        if (!cards[k]) { cards[k] = buildCard(cd, {kind: "no weather (offline): still air", speed: 0, direction: NaN}, flow); sources[k] = "saved"; }
      }
    }
    render(); dayPanel();
  }

  function lockScreen(msg){
    app.textContent = "";
    var box = el("section", "card lock");
    box.appendChild(el("h2", null, "Race card · locked"));
    box.appendChild(el("p", "hint", msg || "Enter the coxswains' passphrase. It stays on this phone."));
    var inp = el("input"); inp.type = "password"; inp.autocomplete = "current-password"; inp.placeholder = "passphrase";
    inp.setAttribute("aria-label", "Passphrase");
    var go = el("button", "unlock", "Unlock"); go.type = "button";
    async function tryIt(){
      go.disabled = true; go.textContent = "Unlocking…";
      try { bundle = await decrypt(inp.value.trim()); save("pass", inp.value.trim()); refresh(); }
      catch (e) { lockScreen("That passphrase did not open the card. Check it and try again."); }
    }
    go.addEventListener("click", tryIt);
    inp.addEventListener("keydown", function(e){ if (e.key === "Enter") tryIt(); });
    box.appendChild(inp); box.appendChild(go);
    app.appendChild(box);
  }

  function dayPanel(){
    var host = document.getElementById("daypanel"); if (!host) return;
    host.textContent = "";
    var d = el("details", "card");
    d.appendChild(el("summary", null, "Earlier races today (race morning)"));
    d.appendChild(el("p", "hint", "One line per finished event: name, its time today, its time last year (m:ss or seconds), e.g. Men's GM 8+, 17:02, 16:40. The targets then use the measured river instead of the wind model. Kept on this phone only."));
    var ta = el("textarea"); ta.rows = 5; ta.setAttribute("aria-label", "Earlier events");
    var rows = []; try { rows = JSON.parse(load("day") || "[]"); } catch (e) {}
    ta.value = rows.map(function(r){ return r.event + ", " + fmt(r.today) + ", " + fmt(r.last); }).join("\n");
    function secs(x){ x = (x || "").trim(); if (x.indexOf(":") < 0) return Number(x); var p = x.split(":").map(Number); return p[0] * 60 + p[1]; }
    var ap = el("button", "unlock", "Apply"); ap.type = "button";
    ap.addEventListener("click", function(){
      var out = ta.value.split("\n").map(function(l){ var p = l.split(","); return p.length >= 3 ? {event: p[0].trim(), today: secs(p[1]), last: secs(p[2])} : null; })
                     .filter(function(r){ return r && r.today > 0 && r.last > 0; });
      save("day", JSON.stringify(out)); refresh();
    });
    var cl = el("button", "unlock ghost", "Clear"); cl.type = "button";
    cl.addEventListener("click", function(){ save("day", "[]"); refresh(); });
    d.appendChild(ta); var bar = el("div", "row2"); bar.appendChild(ap); bar.appendChild(cl); d.appendChild(bar);
    host.appendChild(d);
  }

  document.querySelectorAll(".crews button[data-crew]").forEach(function(b){
    b.addEventListener("click", function(){ crew = b.getAttribute("data-crew"); save("crew", crew); render(); });
  });
  document.querySelectorAll(".courses button").forEach(function(b){
    b.addEventListener("click", function(){ course = b.getAttribute("data-course"); save("course", course); render(); });
  });
  document.querySelectorAll(".gusts button").forEach(function(b){
    b.addEventListener("click", function(){ gusts = b.getAttribute("data-gust") === "on"; save("gusts", gusts ? "on" : "off"); render(); });
  });
  CREWS.forEach(function(k){
    [k, k + "|cambridge", k + "|centre"].forEach(function(kk){
      try { var s = JSON.parse(load(kk) || "null"); if (s && s.targets) { cards[kk] = s; sources[kk] = "saved"; } } catch (e) {}
    });
  });
  (async function(){
    var pass = load("pass");
    if (!pass) { lockScreen(); return; }
    if (cards[crew]) render();
    try { bundle = await decrypt(pass); }
    catch (e) { if (!cards[crew]) lockScreen("The saved passphrase no longer opens the card (it may have changed). Enter the new one."); return; }
    refresh();
    setInterval(function(){ live = {}; refresh(); }, 30 * 60 * 1000);
  })();
})();
</script>
</body>
