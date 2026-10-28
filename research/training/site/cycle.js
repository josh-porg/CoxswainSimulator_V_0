// Optional cycle tracker. Everything here stays in this browser (localStorage, key rowing-plan/cycle-v1):
// the site is static, has no server, and nothing typed here is uploaded or written into the plan files.
//
// What it does, and the evidence for each part (details in the textbook, chapter 8):
// - no phase-based reshuffling of the plan: phase effects on performance are trivial and the evidence low
//   quality (McNulty et al. 2020), and phase does not change strength or training adaptations (Colenso-Semple
//   et al. 2023); symptoms, though, are common and do cost training (Bruinvels et al. 2021), so a symptom check
//   on the day changes that day's own session;
// - cycles longer than 35 days, or none for 90 days, are flagged: menstrual disruption is a warning sign of low
//   energy availability (REDs; Mountjoy et al. 2023);
// - estimated luteal days carry a heat note: core temperature runs ~0.3-0.7 C higher (Baker et al. 2020).
var Cycle = (function(){
  "use strict";
  var KEY = "cycle-v1";
  function load(){
    try { var s = JSON.parse(RP.get(KEY) || "null"); if (s && s.v === 1) return s; } catch (e) {}
    return {v: 1, on: false, hc: false, starts: [], len: 28, period: 5, log: {}};
  }
  var S = load();
  function save(){ RP.set(KEY, JSON.stringify(S)); }
  function wipe(){ RP.set(KEY, null); S = load(); }

  function sortedStarts(){ return S.starts.slice().sort(); }
  function intervals(){
    var st = sortedStarts(), out = [];
    for (var i = 1; i < st.length; i++) out.push(RP.days(RP.d(st[i - 1]), RP.d(st[i])));
    return out;
  }
  function avgLen(){
    var iv = intervals().filter(function(x){ return x >= 15 && x <= 60; }).slice(-6);
    if (!iv.length) return S.len || 28;
    return Math.round(iv.reduce(function(a, b){ return a + b; }, 0) / iv.length);
  }
  // Phase on a date, projected from the latest logged start; null when off, on hormonal contraception, or unknown.
  function phase(date){
    if (!S.on || S.hc || !S.starts.length) return null;
    var st = sortedStarts(), L = avgLen(), last = null;
    for (var i = st.length - 1; i >= 0; i--) { if (RP.d(st[i]) <= date) { last = RP.d(st[i]); break; } }
    if (!last) return null;
    var k = RP.days(last, date), predicted = k >= L;
    var day = k % L, ov = L - 14;
    var name = day < (S.period || 5) ? "period" : day < ov - 1 ? "follicular" : day <= ov + 1 ? "ovulation" : "luteal";
    return {name: name, day: day + 1, predicted: predicted || name === "ovulation" || name === "luteal"};
  }
  function alerts(now){
    var out = [], st = sortedStarts();
    if (!S.on || !st.length) return out;
    var since = RP.days(RP.d(st[st.length - 1]), now);
    if (!S.hc && since > 90) out.push(["warn", "No period logged for " + since + " days. Three months without a period should be checked by a doctor; in athletes it is a common sign of low energy availability (REDs)."]);
    else if (!S.hc && since > 35) out.push(["note", "This cycle is past 35 days. One long cycle happens; if long cycles repeat, mention it to a doctor and look at fuelling (Chapter 7)."]);
    var odd = intervals().slice(-6).filter(function(x){ return x > 35 || x < 21; });
    if (!S.hc && odd.length) out.push(["note", odd.length + " of your recent cycles fell outside 21–35 days. Irregular cycles are worth raising with a doctor, especially with fatigue or falling results."]);
    return out;
  }
  function symptom(dateIso){ return S.log[dateIso] || null; }
  function setSymptom(dateIso, v){ if (v) S.log[dateIso] = v; else delete S.log[dateIso]; save(); }

  // Advice for a day's own sessions, from the symptom check (today only) and the estimated phase.
  function advice(date, sessions){
    var notes = [], s = symptom(RP.iso(date)), p = phase(date);
    var own = sessions.some(function(x){ return x.kind !== "team"; });
    if (s === "bad") notes.push(own ? "Symptom day: swap your own session for 20–30 min easy (Z1) or rest. Team session as you feel." : "Symptom day: do the team session as you feel; easing off is fine.");
    else if (s === "mild") notes.push("Mild symptoms: keep it easy and stop if it gets worse. Lifting can stay; drop a set if you feel flat.");
    if (p && p.name === "luteal") notes.push("Estimated luteal phase: core temperature runs a little higher. On warm days drink more and go by watts, not heart rate.");
    return notes;
  }

  function panel(container, onChange){
    function render(){
      var now = RP.today(), st = sortedStarts(), p = phase(now), al = alerts(now), sym = symptom(RP.iso(now));
      var h = '<div class="cyc-grid">';
      h += '<div class="panel" style="display:grid;gap:10px">' +
           '<label class="remember"><input type="checkbox" id="cyc-on"' + (S.on ? " checked" : "") + '> Use cycle tracking on this device</label>';
      if (S.on) {
        h += '<label class="remember"><input type="checkbox" id="cyc-hc"' + (S.hc ? " checked" : "") + '> I use hormonal contraception (no natural cycle; track bleeds and symptoms only)</label>' +
             '<div class="cyc-row"><label for="cyc-date">Period started</label><input type="date" id="cyc-date" value="' + RP.iso(now) + '" max="' + RP.iso(now) + '"><button id="cyc-add">Log start</button></div>' +
             (st.length ? '<div class="quiet" style="font-size:13px">Logged starts: ' + st.slice(-6).map(function(x){ return '<span class="pill">' + RP.short(x) + ' <button class="x" data-del="' + x + '" aria-label="Remove ' + x + '">×</button></span>'; }).join(" ") + '</div>' : '') +
             (!S.hc ? '<div class="cyc-row"><label for="cyc-len">Typical cycle length</label><input type="number" id="cyc-len" min="18" max="60" value="' + (S.len || 28) + '"><span class="quiet" style="font-size:13px">days' + (intervals().length ? " (your logged average: " + avgLen() + ")" : "") + '</span></div>' : '') +
             '<div><div class="eyebrow" style="margin-bottom:6px">Symptoms today</div><div class="seg" role="group" aria-label="Symptoms today">' +
             [["", "None"], ["mild", "Mild"], ["bad", "Bad"]].map(function(o){ return '<button data-sym="' + o[0] + '" aria-pressed="' + ((sym || "") === o[0]) + '">' + o[1] + '</button>'; }).join("") + '</div></div>' +
             '<button id="cyc-wipe" class="linkish">Delete all cycle data from this device</button>';
      }
      h += '</div><div style="display:grid;gap:10px;align-content:start">';
      if (S.on && p) h += '<div class="stat"><div class="eyebrow">Today</div><div class="big" style="font-size:24px">' + RP.esc(p.name.charAt(0).toUpperCase() + p.name.slice(1)) + (p.predicted ? " (est.)" : "") + '</div><div class="sub">Day ' + p.day + ' of about ' + avgLen() + '</div></div>';
      al.forEach(function(a){ h += '<div class="alert ' + a[0] + '">' + RP.esc(a[1]) + '</div>'; });
      h += '<p class="quiet" style="font-size:13px">Your plan does not change by cycle phase: the research finds phase effects on performance trivial and on training gains absent. What changes your day is how you feel. A symptom check swaps your own session for an easy one. Logged cycles also act as a health check. <a href="textbook.html#ch8">Why, in the textbook</a>.</p>' +
           '<p class="quiet" style="font-size:13px"><strong>Private:</strong> everything here is stored only in this browser on this device. The site has no server and nothing is uploaded. It is gone if you clear site data or use another device.</p>';
      h += '</div></div>';
      container.innerHTML = h;
      var q = function(id){ return document.getElementById(id); };
      q("cyc-on").onchange = function(){ S.on = this.checked; save(); render(); onChange(); };
      if (!S.on) return;
      q("cyc-hc").onchange = function(){ S.hc = this.checked; save(); render(); onChange(); };
      q("cyc-add").onclick = function(){ var v = q("cyc-date").value; if (v && S.starts.indexOf(v) < 0) { S.starts.push(v); save(); render(); onChange(); } };
      if (q("cyc-len")) q("cyc-len").onchange = function(){ var v = Math.max(18, Math.min(60, +this.value || 28)); S.len = v; save(); render(); onChange(); };
      container.querySelectorAll("[data-del]").forEach(function(b){ b.onclick = function(){ S.starts = S.starts.filter(function(x){ return x !== b.dataset.del; }); save(); render(); onChange(); }; });
      container.querySelectorAll("[data-sym]").forEach(function(b){ b.onclick = function(){ setSymptom(RP.iso(RP.today()), b.dataset.sym); render(); onChange(); }; });
      q("cyc-wipe").onclick = function(){ if (confirm("Delete all cycle data stored on this device?")) { wipe(); render(); onChange(); } };
    }
    render();
  }
  return {phase: phase, advice: advice, panel: panel, enabled: function(){ return S.on; }};
})();
