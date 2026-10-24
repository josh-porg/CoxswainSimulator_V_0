// One event's results summary from the RegattaCentral results page that is open in the browser (no requests
// of its own: the page already carries the results as JSON). Paste the whole expression into the browser's
// JavaScript tool after navigating to the event's page; it returns one CSV line for
// data/local/raceday/results/hocr_event_summaries.csv (year,event,code,status,start_utc,n,winner,q25,median,
// q75,sp1,sp2,sp3), or the page title if the page holds no results (e.g. a challenge page: then stop).
(function () {
  var s = [...document.scripts].find(function (x) { return !x.src && /"event_code"/.test(x.textContent); });
  if (!s) return "NO RESULTS: " + document.title;
  var d = JSON.parse(s.textContent);
  return (d.races || []).map(function (ra) {
    var res = (ra.results || []).filter(function (x) { return (x.rawTime || x.sp4RawTime) > 0; });
    var t = res.map(function (x) { return (x.rawTime || x.sp4RawTime) / 1000; }).sort(function (a, b) { return a - b; });
    var q = function (p) { return t.length ? t[Math.min(t.length - 1, Math.floor(p * (t.length - 1) + 0.5))].toFixed(1) : ""; };
    var sm = function (k) {
      var v = res.map(function (x) { return x["sp" + k + "RawTime"] / 1000; }).filter(function (v) { return v > 0; }).sort(function (a, b) { return a - b; });
      return v.length ? v[v.length >> 1].toFixed(1) : "";
    };
    var st = new Date(ra.actualStart || ra.scheduledStart);
    return [st.getUTCFullYear(), d.event_label, d.event_code, ra.status, st.toISOString().slice(0, 16), t.length,
            q(0), q(.25), q(.5), q(.75), sm(1), sm(2), sm(3)].join(",");
  }).join("\n");
})()
