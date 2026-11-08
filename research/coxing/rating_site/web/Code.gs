/**
 * Backend for the coxing-call rating site. Paste into the Apps Script editor of a Google Sheet you own
 * (Extensions > Apps Script), set the script properties, and deploy as a web app (Execute as: Me; Who has access:
 * Anyone). The script stores raters, today's batches and ratings in three sheets of that spreadsheet.
 *
 * Script properties (Project settings > Script properties):
 *   WRITE_TOKEN   printed by build_site.py; derived from the site key, so only people with the full link can rate
 *   ADMIN_KEY     printed by build_site.py; opens the admin view
 *   MANIFEST_URL  https://<user>.github.io/coxing-ratings/manifest.json (opaque item codes only)
 *   TEXT_PER_DAY, VOICE_PER_DAY  optional; calls and clips per set, default 15 and 10
 *   SETS_PER_DAY  optional; how many sets a rater may do in one day, default 3
 * Project settings > Time zone decides when a rater's day starts.
 *
 * Queue rules (study codebook): items with one or two ratings come before unrated items, the human coder's sample
 * first, in an order of its own for each rater; one check phrase and one repeated clip (from an earlier day) per set;
 * a set is fixed once drawn and the next set is drawn only when the rater asks for it after finishing one; an answer
 * is accepted only for an item in the rater's current set that they have not yet answered. To keep a rater away from
 * pieces they were in, type the piece codes (e.g. P03,P04) into their own_pieces cell in the raters sheet.
 */

const SHEETS = {
  raters: ["username", "created", "role", "years", "level", "own_pieces", "devices"],
  batches: ["day", "username", "batch", "set"],
  ratings: ["logged", "day", "username", "device", "kind", "item", "targets", "stage", "action", "arousal",
            "rating", "flags", "ms", "plays", "is_check", "is_repeat", "set"],
};
const TARGETS = ["power", "rate", "length", "catch", "finish", "body", "set", "together", "steer", "info", "morale",
                 "tactic", "count", "other"];

function doGet(e) { return reply_(handle_(e.parameter || {})); }
function doPost(e) { return reply_(handle_(JSON.parse((e.postData && e.postData.contents) || "{}"))); }
function reply_(o) { return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON); }

function prop_(k, d) { const v = PropertiesService.getScriptProperties().getProperty(k); return v === null || v === undefined ? d : v; }
function quota_() { return { text: Number(prop_("TEXT_PER_DAY", 15)), voice: Number(prop_("VOICE_PER_DAY", 10)), sets: Number(prop_("SETS_PER_DAY", 3)) }; }
function today_() { return prop_("TEST_DAY", null) || Utilities.formatDate(new Date(), Session.getScriptTimeZone(), "yyyy-MM-dd"); }

function handle_(p) {
  try {
    if (p.action === "ping") return { ok: true };
    if (p.action === "admin") {
      if (!p.admin || p.admin !== prop_("ADMIN_KEY", "")) return { error: "not admin" };
      return { raters: rows_("raters"), ratings: rows_("ratings"), today: today_() };
    }
    if (!p.w || p.w !== prop_("WRITE_TOKEN", "")) return { error: "invalid link" };
    const user = clean_(p.user);
    if (!user) return { error: "username needed" };
    const lock = LockService.getScriptLock();
    lock.waitLock(20000);
    try {
      if (p.action === "register") return register_(user, p);
      const rater = rows_("raters").find(r => r.username === user);
      if (!rater) return { error: "unknown username", unknown: true };
      if (p.action === "today") return todayAction_(rater, p);
      if (p.action === "more") return moreAction_(rater, p);
      if (p.action === "rate") return rate_(rater, p);
      return { error: "unknown action" };
    } finally { lock.releaseLock(); }
  } catch (err) {
    return { error: String(err && err.message || err) };
  }
}

function clean_(u) { return String(u || "").trim().replace(/\s+/g, " ").slice(0, 40); }

// ------------------------------------------------------------------ sheets
function sheet_(name) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(name);
  if (!sh) {                      // plain-text columns: Sheets would otherwise turn days and digit-only codes into dates and numbers
    sh = ss.insertSheet(name);
    sh.getRange("A:" + String.fromCharCode(64 + SHEETS[name].length)).setNumberFormat("@");
    sh.getRange(1, 1, 1, SHEETS[name].length).setValues([SHEETS[name]]);
  } else if (sh.getLastColumn() < SHEETS[name].length) {   // a sheet made by an earlier version: add the new columns
    sh.getRange("A:" + String.fromCharCode(64 + SHEETS[name].length)).setNumberFormat("@");
    sh.getRange(1, 1, 1, SHEETS[name].length).setValues([SHEETS[name]]);
  }
  return sh;
}
function rows_(name) {
  const v = sheet_(name).getDataRange().getValues();
  const head = v[0] || SHEETS[name];
  return v.slice(1).filter(r => r.join("") !== "").map(r => Object.fromEntries(head.map((h, i) => [h, r[i]])));
}
function append_(name, obj) {     // written as text (appendRow would turn "2 to 5" or a day into a date)
  const sh = sheet_(name), row = SHEETS[name].map(h => obj[h] === undefined ? "" : String(obj[h]));
  sh.getRange(sh.getLastRow() + 1, 1, 1, row.length).setNumberFormat("@").setValues([row]);
}

function manifest_() {
  const cache = CacheService.getScriptCache();
  const parts = cache.get("manifest_parts");
  if (parts) {
    const s = Array.from({ length: Number(parts) }, (_, i) => cache.get("manifest_" + i)).join("");
    if (s && !s.includes("null")) return JSON.parse(s);
  }
  const s = UrlFetchApp.fetch(prop_("MANIFEST_URL", "")).getContentText();
  const n = Math.ceil(s.length / 90000);
  for (let i = 0; i < n; i++) cache.put("manifest_" + i, s.slice(i * 90000, (i + 1) * 90000), 21600);
  cache.put("manifest_parts", String(n), 21600);
  return JSON.parse(s);
}

// ------------------------------------------------------------------ register
function register_(user, p) {
  const prof = p.profile || {};
  if (!prof.role || !prof.years || !prof.level || p.consent !== true) return { error: "consent and profile needed" };
  const sh = sheet_("raters");
  const all = rows_("raters");
  const device = String(p.device || "").slice(0, 40);
  const k = all.findIndex(r => r.username === user);
  if (k < 0) {
    append_("raters", { username: user, created: new Date().toISOString(), role: prof.role, years: prof.years,
                        level: prof.level, own_pieces: "", devices: device });
    return { ok: true, created: true };
  }
  const devs = String(all[k].devices || "").split(",").filter(Boolean);
  if (device && !devs.includes(device)) devs.push(device);
  const row = [user, all[k].created, prof.role, prof.years, prof.level, all[k].own_pieces, devs.join(",")].map(v => String(v == null ? "" : v));
  sh.getRange(k + 2, 1, 1, row.length).setNumberFormat("@").setValues([row]);   // own_pieces stays as the study lead set it
  return { ok: true, created: false };
}

// ------------------------------------------------------------------ today's batch
function hash_(s) {               // small deterministic string hash, for a per-rater order
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function drawBatch_(rater, day, set, ratings, m) {
  const q = quota_();
  const user = rater.username, seed = user + ":" + day + ":" + set;
  const own = new Set(String(rater.own_pieces || "").split(",").map(s => s.trim()).filter(Boolean));
  const mine = ratings.filter(r => r.username === user);
  const doneText = new Set(mine.filter(r => r.kind === "text").map(r => r.item));
  const doneClip = new Set(mine.filter(r => r.kind === "voice" && !Number(r.is_repeat)).map(r => r.item));
  const repeated = new Set(mine.filter(r => r.kind === "voice" && Number(r.is_repeat)).map(r => r.item));
  const count = {};
  ratings.filter(r => !Number(r.is_check) && !Number(r.is_repeat)).forEach(r => { const k = r.kind + ":" + r.item; count[k] = (count[k] || 0) + 1; });
  const tier = n => (n === 1 || n === 2) ? 0 : (n === 0 ? 1 : 2);   // finish items to three raters first
  const order = (kind, id, sampleFlag) => [tier(count[kind + ":" + id] || 0), sampleFlag ? 0 : 1, hash_(user + ":" + day + ":" + id)];
  const cmp = (a, b) => a.o[0] - b.o[0] || a.o[1] - b.o[1] || a.o[2] - b.o[2];
  const textPool = m.items.filter(([id, pc, f]) => !doneText.has(id) && !own.has(pc));
  const checks = textPool.filter(x => x[2].includes("k"));
  const regular = textPool.filter(x => !x[2].includes("k")).map(([id, pc, f]) => ({ id, o: order("text", id, f.includes("s")) })).sort(cmp);
  const text = regular.slice(0, Math.max(0, q.text - (checks.length ? 1 : 0))).map(x => ({ kind: "text", id: x.id, check: 0, repeat: 0 }));
  if (checks.length && text.length) {
    const c = checks[hash_(seed + "check") % checks.length];
    text.splice(hash_(seed + "pos") % (text.length + 1), 0, { kind: "text", id: c[0], check: 1, repeat: 0 });
  }
  const clipPool = m.clips.filter(([id, pc]) => !doneClip.has(id) && !own.has(pc)).map(([id]) => ({ id, o: order("voice", id, false) })).sort(cmp);
  const earlier = mine.filter(r => r.kind === "voice" && !Number(r.is_repeat) && String(r.rating) !== "" && r.day !== day && !repeated.has(r.item)).map(r => r.item);
  const voice = clipPool.slice(0, Math.max(0, q.voice - (earlier.length ? 1 : 0))).map(x => ({ kind: "voice", id: x.id, check: 0, repeat: 0 }));
  if (earlier.length && voice.length) {
    const r = earlier[hash_(seed + "rep") % earlier.length];
    voice.splice(hash_(seed + "rpos") % (voice.length + 1), 0, { kind: "voice", id: r, check: 0, repeat: 1 });
  }
  return text.concat(voice);
}

// A rater's sets for a day, in order; a set is stored only when it has items.
function sets_(rater, day) {
  return rows_("batches").filter(r => r.day === day && r.username === rater.username && r.batch && r.batch !== "[]")
    .map(r => ({ set: Number(r.set) || 1, batch: JSON.parse(r.batch) })).sort((a, b) => a.set - b.set);
}
function drawSet_(rater, day, set) {
  const batch = drawBatch_(rater, day, set, rows_("ratings"), manifest_());
  if (batch.length) append_("batches", { day: day, username: rater.username, batch: JSON.stringify(batch), set: set });
  return { set: set, batch: batch };
}
function currentSet_(rater, day, create) {
  const s = sets_(rater, day);
  if (s.length) return s[s.length - 1];
  return create ? drawSet_(rater, day, 1) : null;
}

function progress_(rater, day, cur, ratings) {
  const q = quota_();
  const mine = ratings.filter(r => r.username === rater.username);
  const inSet = mine.filter(r => r.day === day && (Number(r.set) || 1) === cur.set);
  const key = x => x.kind + ":" + x.id + ":" + x.repeat;
  const doneKeys = new Set(inSet.map(r => r.kind + ":" + r.item + ":" + Number(r.is_repeat)));
  const remaining = cur.batch.filter(x => !doneKeys.has(key(x)));
  const covered = new Set(ratings.filter(r => r.kind === "text" && !Number(r.is_check)).map(r => r.item)).size;
  return { remaining: remaining, done_in_set: cur.batch.length - remaining.length, batch_size: cur.batch.length,
           set: cur.set, sets_per_day: q.sets, more: !remaining.length && cur.batch.length > 0 && cur.set < q.sets,
           mine: mine.length, days: new Set(mine.map(r => r.day)).size, covered: covered, raters: rows_("raters").length };
}

function todayAction_(rater, p) {
  const day = today_();
  return { ok: true, day: day, profile: { role: rater.role, years: rater.years, level: rater.level },
           progress: progress_(rater, day, currentSet_(rater, day, true), rows_("ratings")) };
}

function moreAction_(rater, p) {
  const day = today_(), q = quota_();
  const ratings = rows_("ratings");
  const cur = currentSet_(rater, day, true);
  const prog = progress_(rater, day, cur, ratings);
  if (prog.remaining.length || !cur.batch.length) return { ok: true, day: day, progress: prog };   // finish this set first
  if (cur.set >= q.sets) return { error: "That's the most sets for one day. New calls tomorrow." };
  return { ok: true, day: day, progress: progress_(rater, day, drawSet_(rater, day, cur.set + 1), ratings) };
}

// ------------------------------------------------------------------ rate
function rate_(rater, p) {
  const day = today_();
  const cur = currentSet_(rater, day, false);
  if (!cur) return { error: "no set today" };
  const ratings = rows_("ratings");
  const done = new Set(ratings.filter(r => r.username === rater.username && r.day === day && (Number(r.set) || 1) === cur.set)
    .map(r => r.kind + ":" + r.item + ":" + Number(r.is_repeat)));
  const slot = cur.batch.find(x => x.kind === p.kind && x.id === p.id && !done.has(x.kind + ":" + x.id + ":" + x.repeat));
  if (!slot) return { error: "not in your current set" };
  const row = { logged: new Date().toISOString(), day: day, username: rater.username, device: String(p.device || "").slice(0, 40),
                kind: slot.kind, item: slot.id, ms: Number(p.ms) || 0, is_check: slot.check, is_repeat: slot.repeat, flags: "", set: cur.set };
  if (slot.kind === "text") {
    const c = p.codes || {};
    const tg = (Array.isArray(c.targets) ? c.targets : []).filter(t => TARGETS.includes(t));
    const skip = tg.length > 0 && tg.every(t => t === "count" || t === "other");
    if (!tg.length || new Set(tg).size !== tg.length) return { error: "choose at least one target" };
    if (!skip && !(["inf", "ass", "dec"].includes(c.stage) && ["y", "n"].includes(c.action) && ["up", "down", "-"].includes(c.arousal)))
      return { error: "answer every question" };
    Object.assign(row, { targets: tg.join("|"), stage: skip ? "-" : c.stage, action: skip ? "-" : c.action, arousal: skip ? "-" : c.arousal,
                         flags: (Array.isArray(p.flags) ? p.flags : []).filter(f => ["fragment", "unsure", "misheard"].includes(f)).join("|") });
  } else {
    if (p.played_full !== true) return { error: "play the clip to the end first" };
    if (p.novoice === true) Object.assign(row, { rating: "", flags: "novoice", plays: Number(p.plays) || 1 });   // no voice heard: no rating
    else {
      const r = Number(p.rating);
      if (!(r >= 1 && r <= 7 && r === Math.round(r))) return { error: "rate 1-7" };
      Object.assign(row, { rating: r, plays: Number(p.plays) || 1 });
    }
  }
  append_("ratings", row);
  ratings.push(row);
  return { ok: true, progress: progress_(rater, day, cur, ratings) };
}
