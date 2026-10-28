// Training log and planning aids, stored only in this browser (localStorage key rowing-plan/log-v1).
//
// Each feature is a behaviour-change technique with evidence behind it; each is kept deliberately light,
// because heavy tracking can cost enjoyment and continued use (Etkin 2016, J Consumer Res) and goal-chasing
// trackers are linked to compulsive exercise:
//   - tick sessions done:  self-monitoring that is recorded raises goal attainment (Harkin et al. 2016,
//                          138 studies, d = 0.40; larger when recorded);
//   - when & where, if-then plans:  implementation intentions (Gollwitzer & Sheeran 2006, d = 0.65; stronger in
//                          if-then form, Sheeran et al. 2024); a stable time and place also builds habit
//                          (Lally et al. 2010: one missed day does not derail it);
//   - move your own sessions:  choice supports autonomous motivation, which predicts adherence (self-
//                          determination theory; Ntoumanis et al. 2021);
//   - weekly consistency, never daily streaks; rest and light weeks count as on-plan; no red "failed" states.
var Log = (function(){
  "use strict";
  var KEY = "log-v1";
  function load(){
    try { var s = JSON.parse(RP.get(KEY) || "null"); if (s && s.v === 1) return s; } catch (e) {}
    return {v: 1, done: {}, moved: {}, when: {}, ifthen: ["", ""]};
  }
  var S = load();
  function save(){ RP.set(KEY, JSON.stringify(S)); }
  var DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

  // A session's identity: the week, its planned day, its kind and its position that day.
  function sid(monday, day, s, i){ return monday + "|" + day + "|" + s.kind + "|" + i; }

  // The week's sessions by day after the athlete's own moves (team sessions never move).
  function arrange(week){
    var out = {}; DAYS.forEach(function(d){ out[d] = []; });
    DAYS.forEach(function(day){
      (week.sessions[day] || []).forEach(function(s, i){
        var id = sid(week.monday, day, s, i), to = s.kind !== "team" && S.moved[id] ? S.moved[id] : day;
        out[to].push({s: s, id: id, from: to !== day ? day : null});
      });
    });
    DAYS.forEach(function(d){ out[d].sort(function(a, b){ return (a.s.kind === "strength" ? 0 : a.s.kind === "team" ? 1 : 2) - (b.s.kind === "strength" ? 0 : b.s.kind === "team" ? 1 : 2); }); });
    return out;
  }
  function isDone(id){ return S.done[id] === 1; }
  function toggle(id){ if (S.done[id]) delete S.done[id]; else S.done[id] = 1; save(); }
  function move(id, to, from){ if (to === from) delete S.moved[id]; else S.moved[id] = to; save(); }

  // Warnings for a moved session, from the plan's own rules.
  function moveNote(arranged, day){
    var i = DAYS.indexOf(day), nextDay = DAYS[(i + 1) % 7], notes = [];
    var hasLift = arranged[day].some(function(x){ return x.s.kind === "strength"; });
    var hasRow = arranged[day].some(function(x){ return x.s.kind !== "strength"; });
    if (hasLift && (arranged[nextDay] || []).some(function(x){ return x.s.kind === "team" && /Z5|Z6/.test(x.s.zone); }))
      notes.push("Heavy legs the day before intervals can take the edge off the intervals. Fine if it's the only day that works.");
    if (hasLift && hasRow) notes.push("Lift first, row at least 6 hours later.");
    return notes;
  }

  // Own sessions done / planned, for the last n weeks up to and including week index wi.
  function consistency(B, wi, n){
    var rows = [];
    for (var k = Math.max(0, wi - n + 1); k <= wi; k++) {
      var w = B.weeks[k], planned = 0, done = 0;
      DAYS.forEach(function(day){ (w.sessions[day] || []).forEach(function(s, i){ if (s.kind !== "team") { planned++; if (isDone(sid(w.monday, day, s, i))) done++; } }); });
      rows.push({monday: w.monday, planned: planned, done: done, light: /deload|taper|transition/.test(w.phase)});
    }
    return rows;
  }

  function planner(container){
    function render(){
      var h = '<div class="panel" style="display:grid;gap:12px">' +
        '<p style="margin:0">People who decide <em>when</em> and <em>where</em> they will train, and what they’ll do if something gets in the way, follow through far more often. Set it once; it shows on your calendar.</p>' +
        '<div class="when-grid">' + DAYS.map(function(d){ return '<label class="when"><span>' + d + '</span><input data-when="' + d + '" value="' + RP.esc(S.when[d] || "") + '" placeholder="e.g. 7 am, gym" maxlength="40"></label>'; }).join("") + '</div>' +
        '<div style="display:grid;gap:6px"><div class="eyebrow">If–then backup plans</div>' +
        [0, 1].map(function(k){ return '<input class="ifthen" data-if="' + k + '" value="' + RP.esc(S.ifthen[k] || "") + '" placeholder="' + (k ? "If I miss the morning lift, then I lift after work on Sunday" : "If it’s raining, then I do the long row on the erg") + '" maxlength="120">'; }).join("") + '</div>' +
        '<p class="quiet" style="font-size:13px;margin:0">Saved only in this browser. <button class="linkish" id="log-export">Save a backup file</button> · <label class="linkish" style="cursor:pointer">Restore from a backup<input type="file" id="log-import" accept="application/json" hidden></label></p>' +
        '</div>';
      container.innerHTML = h;
      container.querySelectorAll("[data-when]").forEach(function(el){ el.onchange = function(){ var v = el.value.trim(); if (v) S.when[el.dataset.when] = v; else delete S.when[el.dataset.when]; save(); if (Log.onChange) Log.onChange(); }; });
      container.querySelectorAll("[data-if]").forEach(function(el){ el.onchange = function(){ S.ifthen[+el.dataset.if] = el.value.trim(); save(); }; });
      document.getElementById("log-export").onclick = function(){
        var data = {log: S, cycle: JSON.parse(RP.get("cycle-v1") || "null")};
        var a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], {type: "application/json"}));
        a.download = "training-log-backup-" + RP.iso(RP.today()) + ".json"; a.click();
      };
      document.getElementById("log-import").onchange = function(e){
        var f = e.target.files[0]; if (!f) return;
        f.text().then(function(t){ try { var d = JSON.parse(t); if (d.log && d.log.v === 1) { S = d.log; save(); } if (d.cycle && d.cycle.v === 1) RP.set("cycle-v1", JSON.stringify(d.cycle)); location.reload(); } catch (err) { alert("That file isn't a backup from this page."); } });
      };
    }
    render();
  }
  return {arrange: arrange, isDone: isDone, toggle: toggle, move: move, moveNote: moveNote, consistency: consistency,
          planner: planner, when: function(d){ return S.when[d] || ""; }, DAYS: DAYS, onChange: null};
})();
