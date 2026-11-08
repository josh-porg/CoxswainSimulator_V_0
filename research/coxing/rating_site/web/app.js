// Coxing-call rating site. Static: the phrases and clips are encrypted (AES-256-GCM); the key is the "k" part of the
// link's fragment (#k=...), which never leaves the browser. Ratings go to the study lead's Google Apps Script
// (config.json), which accepts them only with a token derived from the key and decides each rater's daily batch.
(async function () {
  const app = document.getElementById("app");
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const say = (title, body) => { app.innerHTML = `<div class="big">${title}</div><div>${body}</div>`; };
  const params = new URLSearchParams(location.hash.slice(1));
  const KEY = params.get("k"), ADMIN = params.get("admin");
  if (!KEY) return say("This page needs your full link", "Open the link exactly as you were sent it, including everything after the # sign.");
  const unb64 = s => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4)), c => c.charCodeAt(0));
  const b64u = buf => btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  let key, pack, cfg;
  try {
    key = await crypto.subtle.importKey("raw", unb64(KEY), "AES-GCM", false, ["decrypt"]);
    const open = async buf => { const b = new Uint8Array(buf); return crypto.subtle.decrypt({ name: "AES-GCM", iv: b.slice(0, 12) }, key, b.slice(12)); };
    window.__open = open;
    cfg = await (await fetch("config.json", { cache: "no-store" })).json();
    pack = JSON.parse(new TextDecoder().decode(await open(await (await fetch("pack.enc")).arrayBuffer())));
  } catch (e) {
    return say("This link doesn't open the study", "The link may be incomplete or out of date. Ask the person who sent it for a new one.");
  }
  const W = b64u(await crypto.subtle.digest("SHA-256", new TextEncoder().encode("write:" + KEY)));
  const backend = async payload => {
    let mock = window.MOCK_BACKEND;                // the test harness (test.html, never deployed) supplies a stand-in
    try { if (!mock && window.parent !== window) mock = window.parent.MOCK_BACKEND; } catch (e) { mock = null; }
    if (mock) return mock(payload);
    const r = await fetch(cfg.script, { method: "POST", body: JSON.stringify(payload) });   // text/plain: no CORS preflight
    return r.json();
  };
  const phraseOf = id => { const [pc, idx] = pack.items[id]; const ph = pack.pieces[pc]; return { text: ph[idx], before: ph.slice(Math.max(0, idx - 3), idx), after: ph.slice(idx + 1, idx + 3) }; };
  if (ADMIN) return admin();

  // ---------------------------------------------------------------- identity
  const STORE = "coxrate_v1";
  let me = null;
  try { me = JSON.parse(localStorage.getItem(STORE) || "null"); } catch (e) { me = null; }
  const save = () => { try { localStorage.setItem(STORE, JSON.stringify(me)); } catch (e) { } };
  if (!me || !me.username) signup(); else run();

  function signup(msg = "") {
    const device = (me && me.device) || (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2));
    app.innerHTML = `<div class="big">Before you start</div>
    <p>You're helping a study of what makes a coxswain's call work. Each day you'll see up to 15 short calls from real races and practices (names removed) and answer three or four quick questions about each, then listen to 10 short muffled clips and rate how fired-up the voice sounds. About five minutes. There are no right or wrong answers; your own reading is what's wanted.</p>
    <p>What's stored: your username, your answers, the time each took, which device you used, and the profile below. Your answers are never shown to other raters. Taking part is voluntary; you can stop at any time and ask the study lead to delete your answers.</p>
    <label class="f" for="u">Username (required; use the same one on every device)</label><input type="text" id="u" maxlength="40" autocomplete="nickname" placeholder="e.g. your first name and initial">
    <label class="f" for="role">You are mainly a</label><select id="role"><option value="">choose</option><option>coxswain</option><option>rower</option><option>both</option><option>coach</option><option>other</option></select>
    <label class="f" for="years">Years in rowing</label><select id="years"><option value="">choose</option><option>under 2</option><option>2-5</option><option>5-10</option><option>over 10</option></select>
    <label class="f" for="level">Level you mostly do it at</label><select id="level"><option value="">choose</option><option>learn to row</option><option>club or masters</option><option>school or junior</option><option>college</option><option>national or elite</option></select>
    <label class="f">Were you in any of these? (you won't be shown calls from them)</label>
    <div class="own">${pack.groups.map(([label, pcs], n) => `<label><input type="checkbox" data-g="${n}"> ${esc(label)}</label>`).join("")}</div>
    <label class="f"><input type="checkbox" id="agree"> I've read this and agree to take part</label>
    <button class="main" id="go">Start</button><div class="msg" id="m">${esc(msg)}</div>`;
    document.getElementById("go").onclick = async () => {
      const v = id => document.getElementById(id).value.trim();
      const m = document.getElementById("m");
      if (!v("u") || v("u").length < 2) return m.textContent = "Choose a username first (at least two characters).";
      if (!v("role") || !v("years") || !v("level") || !document.getElementById("agree").checked) return m.textContent = "Fill in all three questions and tick the box.";
      const own = [...app.querySelectorAll("[data-g]:checked")].flatMap(c => pack.groups[+c.dataset.g][1]);
      m.textContent = "Saving…";
      const r = await backend({ action: "register", w: W, user: v("u"), device, consent: true, own, profile: { role: v("role"), years: v("years"), level: v("level") } }).catch(() => ({ error: "No connection; try again." }));
      if (r.error) return m.textContent = r.error;
      me = { username: v("u"), device }; save(); run();
    };
  }

  // ---------------------------------------------------------------- rating
  let prog, item, cur, flags, t0, busy = false;
  async function run() {
    app.innerHTML = "Loading today's calls…";
    const r = await backend({ action: "today", w: W, user: me.username, device: me.device }).catch(() => ({ error: "No connection. Reload to try again." }));
    if (r.unknown) { me = { device: me.device }; save(); return signup("That username isn't registered yet; fill this in once."); }
    if (r.error) return say("Something went wrong", esc(r.error));
    prog = r.progress; next();
  }
  function head() {
    const p = prog, x = p.remaining[0];
    return `<div class="top"><span>${esc(me.username)} · <a href="#" id="switch">not you?</a></span><span>${p.done_today} of ${p.batch_size} done today</span></div>
      <div class="bar"><div style="width:${(100 * p.done_today / Math.max(1, p.batch_size)).toFixed(0)}%"></div></div>`;
  }
  function wireSwitch() { const s = document.getElementById("switch"); if (s) s.onclick = e => { e.preventDefault(); me = { device: me.device }; save(); signup(); }; }
  function next() {
    if (!prog.remaining.length) return finished();
    item = prog.remaining[0]; t0 = performance.now();
    item.kind === "text" ? showText() : showVoice();
  }
  function finished() {
    const total = Object.keys(pack.items).length, pct = Math.round(100 * prog.covered / total);
    app.innerHTML = `<div class="top"><span>${esc(me.username)} · <a href="#" id="switch">not you?</a></span></div>
      <div class="big">That's today done. Thank you.</div>
      <p>New calls tomorrow. Short daily sessions keep the ratings fresh, so there's a limit on purpose.</p>
      <div class="stat">You've rated ${prog.mine} items over ${prog.days} day${prog.days === 1 ? "" : "s"}. Together, ${prog.raters} rater${prog.raters === 1 ? " has" : "s have"} reached ${pct}% of the study's calls.</div>`;
    wireSwitch();
  }
  const TG = [["power", "effort, pressure, drive, legs or push as force"], ["rate", "stroke rate, rhythm, ratio, slide speed"], ["length", "reach, long, length"], ["catch", "entry, blade in, catch timing"], ["finish", "release, extraction, hands away"], ["body", "posture, sit up, core, breathing"], ["set", "balance, blade height"], ["together", "timing across the crew"], ["steer", "course, turns, side pressure"], ["info", "split, rate, distance or position read out"], ["morale", "encouragement with no rowing target"], ["tactic", "a move on another crew"], ["count", "a stroke count"], ["other", "anything else that isn't a call"]];
  const QS = [
    ["stage", "What does it do?", "", [["inf", "states a fact"], ["ass", "judges how it's going"], ["dec", "tells the crew what to do"]]],
    ["action", "Does it name a concrete action?", "a movement, force or rhythm change the crew can do", [["y", "yes"], ["n", "no"]]],
    ["arousal", "Does it raise or lower intensity?", "", [["up", "raises (now, go, attack)"], ["down", "lowers (relax, settle, breathe)"], ["-", "neither"]]],
  ];
  function showText() { cur = { targets: [] }; flags = new Set(); renderText(); }
  const skipped = () => cur.targets.length && cur.targets.every(t => t === "count" || t === "other");
  const complete = () => cur.targets.length && (skipped() || QS.every(q => cur[q[0]]));
  function renderText(msg = "") {
    const x = phraseOf(item.id);
    let h = head() + `<div class="ctx">${x.before.map(esc).join("<br>")}</div><div class="phrase">${esc(x.text)}</div><div class="ctx">${x.after.map(esc).join("<br>")}</div>`;
    h += `<div class="q"><div class="t">What is it about?</div><div class="d">Tap all that apply; the first you tap counts as the main one. The grey lines are context.</div><div class="chips">` +
      TG.map(([v, d]) => `<button class="chip ${cur.targets.includes(v) ? "on" : ""}" data-t="${v}" title="${d}">${v}${cur.targets.length > 1 && cur.targets[0] === v ? "<sup>main</sup>" : ""}</button>`).join("") + "</div></div>";
    if (!skipped()) QS.forEach(([k, t, d, opts]) => {
      h += `<div class="q"><div class="t">${t}</div>${d ? `<div class="d">${d}</div>` : ""}<div class="chips">` +
        opts.map(([v, l]) => `<button class="chip ${cur[k] === v ? "on" : ""}" data-k="${k}" data-v="${v}">${l}</button>`).join("") + "</div></div>";
    });
    h += `<div class="q"><div class="chips"><button class="chip flag ${flags.has("unsure") ? "on" : ""}" data-f="unsure">not sure</button><button class="chip flag ${flags.has("misheard") ? "on" : ""}" data-f="misheard">words look misheard</button></div></div>`;
    h += `<button class="main" id="save" ${complete() ? "" : "disabled"}>Next</button><div class="msg">${esc(msg)}</div>`;
    app.innerHTML = h; wireSwitch();
    app.querySelectorAll("[data-t]").forEach(b => b.onclick = () => { const v = b.dataset.t, i = cur.targets.indexOf(v); i >= 0 ? cur.targets.splice(i, 1) : cur.targets.push(v); renderText(); });
    app.querySelectorAll("[data-k]").forEach(b => b.onclick = () => { cur[b.dataset.k] = b.dataset.v; renderText(); });
    app.querySelectorAll("[data-f]").forEach(b => b.onclick = () => { const f = b.dataset.f; flags.has(f) ? flags.delete(f) : flags.add(f); renderText(); });
    document.getElementById("save").onclick = async () => {
      if (!complete() || busy) return; busy = true;
      const r = await backend({ action: "rate", w: W, user: me.username, device: me.device, kind: "text", id: item.id, codes: cur, flags: [...flags], ms: Math.round(performance.now() - t0) }).catch(() => ({ error: "No connection; try again." }));
      busy = false;
      if (r.error) return renderText(r.error);
      prog = r.progress; next();
    };
  }
  async function showVoice() {
    let plays = 0, full = false;
    app.innerHTML = head() + `<div class="big">How fired-up does this voice sound?</div>
      <div class="muted">The words are filtered out on purpose. Listen to the whole clip, then rate the voice, not what you think was said.</div>
      <button class="main" id="play" disabled>Loading…</button>
      <div class="scale">${[1, 2, 3, 4, 5, 6, 7].map(n => `<button data-r="${n}" disabled>${n}</button>`).join("")}</div>
      <div class="ends"><span>very calm</span><span>very fired-up</span></div><div class="msg" id="m"></div>`;
    wireSwitch();
    let audio;
    try {
      const buf = await window.__open(await (await fetch("clips/" + item.id + ".bin")).arrayBuffer());
      audio = new Audio(URL.createObjectURL(new Blob([buf], { type: "audio/wav" })));
    } catch (e) { document.getElementById("m").textContent = "This clip didn't load. Reload the page to try again."; return; }
    const play = document.getElementById("play");
    play.disabled = false; play.textContent = "Play";
    audio.onended = () => { full = true; app.querySelectorAll("[data-r]").forEach(b => b.disabled = false); play.textContent = "Play again"; };
    play.onclick = () => { plays++; audio.currentTime = 0; audio.play(); };
    app.querySelectorAll("[data-r]").forEach(b => b.onclick = async () => {
      if (busy) return; busy = true;
      const r = await backend({ action: "rate", w: W, user: me.username, device: me.device, kind: "voice", id: item.id, rating: +b.dataset.r, plays, played_full: full, ms: Math.round(performance.now() - t0) }).catch(() => ({ error: "No connection; try again." }));
      busy = false;
      if (r.error) { document.getElementById("m").textContent = r.error; return; }
      prog = r.progress; next();
    });
  }
  document.addEventListener("keydown", e => {
    if (!item || item.kind !== "voice") return;
    if (/^[1-7]$/.test(e.key)) { const b = app.querySelector(`[data-r="${e.key}"]`); if (b && !b.disabled) b.click(); }
    if (e.key === " ") { e.preventDefault(); document.getElementById("play")?.click(); }
  });

  // ---------------------------------------------------------------- admin
  function alpha(units, metric) {                    // Krippendorff's alpha; units: {unit: [values]}
    const U = Object.values(units).filter(v => v.length >= 2);
    if (!U.length) return null;
    const vals = U.flat(), n = vals.length;
    const d = metric === "interval" ? (a, b) => (a - b) ** 2 : (a, b) => (a === b ? 0 : 1);
    let Do = 0; U.forEach(v => { let s = 0; v.forEach((a, i) => v.forEach((b, j) => { if (i !== j) s += d(a, b); })); Do += s / (v.length - 1); }); Do /= n;
    let De = 0; vals.forEach((a, i) => vals.forEach((b, j) => { if (i !== j) De += d(a, b); })); De /= n * (n - 1);
    return De === 0 ? null : 1 - Do / De;
  }
  async function admin() {
    document.getElementById("main").classList.add("wide");
    app.innerHTML = "Loading the ratings…";
    const r = await backend({ action: "admin", admin: ADMIN }).catch(() => ({ error: "No connection." }));
    if (r.error) return say("Admin", esc(r.error));
    const man = await (await fetch("manifest.json")).json();
    const R = r.ratings, text = R.filter(x => x.kind === "text" && !Number(x.is_check) && !Number(x.is_repeat)), voice = R.filter(x => x.kind === "voice" && !Number(x.is_repeat));
    const units = (rows, f) => { const u = {}; rows.forEach(x => { const v = f(x); (u[x.item] = u[x.item] || []).push(v); }); return u; };
    const A = { target: alpha(units(text, x => String(x.targets).split("|")[0])), stage: alpha(units(text, x => x.stage)), action: alpha(units(text, x => x.action)),
                arousal: alpha(units(text, x => x.arousal)), "voice (interval)": alpha(units(voice, x => Number(x.rating)), "interval") };
    const hist = (rows, total) => { const c = {}; rows.forEach(x => c[x.item] = (c[x.item] || 0) + 1); const h = { 0: total - Object.keys(c).length }; Object.values(c).forEach(v => { const k = Math.min(v, 4); h[k] = (h[k] || 0) + 1; }); return h; };
    const ht = hist(text, man.items.length), hv = hist(voice, man.clips.length);
    const labelOf = pcs => pack.groups.filter(([l, g]) => g.some(p => String(pcs).split(",").includes(p))).map(([l]) => l).join("; ");
    const rows = r.raters.map(u => {
      const mine = R.filter(x => x.username === u.username), ch = mine.filter(x => Number(x.is_check));
      const ok = ch.filter(x => String(x.targets).split("|")[0] === pack.checks[x.item]).length;
      return { u: u.username, role: u.role, years: u.years, level: u.level, text: mine.filter(x => x.kind === "text").length, voice: mine.filter(x => x.kind === "voice").length,
               days: new Set(mine.map(x => x.day)).size, checks: ch.length ? `${ok}/${ch.length}` : "–", devices: String(u.devices || "").split(",").filter(Boolean).length,
               last: mine.length ? mine[mine.length - 1].logged : "–", own: labelOf(u.own_pieces) };
    });
    const k = ["0", "1", "2", "3", "4"];
    app.innerHTML = `<div class="big">Ratings admin</div><div class="muted">Day ${esc(r.today)} · ${R.length} ratings from ${r.raters.length} raters</div>
      <h2>Agreement across raters (Krippendorff's alpha)</h2><div class="grid">${Object.entries(A).map(([n, v]) => `<div class="tile"><div class="n">${v == null ? "–" : v.toFixed(2)}</div><div class="l">${n}</div></div>`).join("")}</div>
      <div class="muted">Items with two or more raters; check items and repeats left out; target uses each rater's main target. About 0.67 is a common lowest acceptable value for tentative conclusions.</div>
      <h2>Coverage (raters per item)</h2><div class="wrap"><table><tr><th></th>${k.map(x => `<th>${x === "4" ? "4+" : x}</th>`).join("")}</tr>
      <tr><td>calls</td>${k.map(x => `<td>${ht[x] || 0}</td>`).join("")}</tr><tr><td>clips</td>${k.map(x => `<td>${hv[x] || 0}</td>`).join("")}</tr></table></div>
      <h2>Raters</h2><div class="wrap"><table><tr><th>Username</th><th>Role</th><th>Years</th><th>Level</th><th>Calls</th><th>Clips</th><th>Days</th><th>Checks</th><th>Devices</th><th>Last</th><th>Own races</th></tr>
      ${rows.map(x => `<tr><td>${esc(x.u)}</td><td>${esc(x.role)}</td><td>${esc(x.years)}</td><td>${esc(x.level)}</td><td>${x.text}</td><td>${x.voice}</td><td>${x.days}</td><td>${x.checks}</td><td>${x.devices}</td><td>${esc(x.last)}</td><td>${esc(x.own)}</td></tr>`).join("")}</table></div>
      <p class="muted">Checks: phrases with an unambiguous answer (a bare count or a split readout). More than one device on a username can mean a second phone, or someone else using that name.</p>
      <button class="main" id="dl">Download all ratings (JSON)</button>`;
    document.getElementById("dl").onclick = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify({ raters: r.raters, ratings: R }, null, 1)], { type: "application/json" }));
      a.download = "ratings_" + r.today + ".json"; a.click();
    };
  }
})();
