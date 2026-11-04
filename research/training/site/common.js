// Shared by the plan page and the textbook page: unlock, decryption, storage and formatting.
// The bundle (data.enc) is AES-256-GCM with a PBKDF2-SHA256 key from the passphrase; see site_build.py.
// Browser storage is namespaced ("rowing-plan/") because every github.io page of one account shares an origin.
var RP = (function(){
  "use strict";
  var NS = "rowing-plan/";
  var b64 = function(s){ return Uint8Array.from(atob(s), function(c){ return c.charCodeAt(0); }); };
  var esc = function(s){ return String(s).replace(/[&<>"']/g, function(c){ return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]; }); };
  function get(k){ try { return localStorage.getItem(NS + k); } catch (e) { return null; } }
  var asked = false, ok = true;
  function set(k, v){
    try { if (v === null) localStorage.removeItem(NS + k); else localStorage.setItem(NS + k, v); ok = true; }
    catch (e) { ok = false; }
    // On the first thing she saves, ask the browser to keep this site's data rather than clear it when space runs
    // low or the site goes unvisited (granted silently by Chrome and Safari for sites in use; Firefox may ask).
    if (!asked && v !== null && k !== "pass") {
      asked = true;
      try { if (navigator.storage && navigator.storage.persist) navigator.storage.persisted().then(function(p){ if (!p) navigator.storage.persist(); }); } catch (e) {}
    }
  }
  // Can this browser save at all? Private windows and blocked site data cannot (or forget on close).
  function canStore(){
    try { localStorage.setItem(NS + "probe", "1"); localStorage.removeItem(NS + "probe"); return ok; } catch (e) { return false; }
  }
  function storageWarning(){
    return canStore() ? "" : '<div class="alert warn">This browser isn’t saving what you enter here (a private window, or site data blocked). Open the page in a normal window to keep your entries.</div>';
  }
  async function decrypt(pass){
    var enc = await (await fetch("data.enc", {cache: "no-cache"})).json();
    var km = await crypto.subtle.importKey("raw", new TextEncoder().encode(pass), "PBKDF2", false, ["deriveKey"]);
    var key = await crypto.subtle.deriveKey({name: "PBKDF2", salt: b64(enc.salt), iterations: enc.iterations, hash: "SHA-256"},
                                            km, {name: "AES-GCM", length: 256}, false, ["decrypt"]);
    var pt = await crypto.subtle.decrypt({name: "AES-GCM", iv: b64(enc.iv)}, key, b64(enc.data));
    return JSON.parse(new TextDecoder().decode(pt));
  }
  // Show the lock form until the passphrase opens the bundle, then call ready(bundle).
  function unlock(ready){
    var lock = document.getElementById("lock"), err = document.getElementById("err");
    async function attempt(pass, remember){
      err.textContent = "Opening…";
      try {
        var B = await decrypt(pass);
        if (remember) set("pass", pass);
        err.textContent = "";
        lock.hidden = true;
        ready(B);
      } catch (e) {
        err.textContent = "That passphrase did not open the plan. Check it and try again.";
        set("pass", null);
        lock.hidden = false;
      }
    }
    document.getElementById("unlock").addEventListener("submit", function(e){
      e.preventDefault();
      attempt(document.getElementById("pass").value, document.getElementById("remember").checked);
    });
    var saved = get("pass");
    if (saved) { lock.hidden = true; attempt(saved, true); } else { lock.hidden = false; }
  }
  function mmss(t, tenths){ var k = tenths ? 10 : 1, r = Math.round(t * k) / k, m = Math.floor(r / 60), s = r - 60 * m; return m + ":" + (s < 10 ? "0" : "") + s.toFixed(tenths ? 1 : 0); }
  function split(w){ return 500 * Math.cbrt(2.80 / w); }
  function d(s){ var p = s.split("-"); return new Date(+p[0], +p[1] - 1, +p[2]); }
  function iso(x){ return x.getFullYear() + "-" + String(x.getMonth() + 1).padStart(2, "0") + "-" + String(x.getDate()).padStart(2, "0"); }
  var MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  function nice(s){ var x = d(s); return x.getDate() + " " + MON[x.getMonth()] + " " + x.getFullYear(); }
  function short(s){ var x = d(s); return x.getDate() + " " + MON[x.getMonth()]; }
  function today(){ var n = new Date(); return new Date(n.getFullYear(), n.getMonth(), n.getDate()); }
  function days(a, b){ return Math.round((b - a) / 864e5); }
  function forget(){ set("pass", null); location.reload(); }
  return {esc: esc, get: get, set: set, canStore: canStore, storageWarning: storageWarning, unlock: unlock, mmss: mmss, split: split, d: d, iso: iso, MON: MON,
          nice: nice, short: short, today: today, days: days, forget: forget};
})();
