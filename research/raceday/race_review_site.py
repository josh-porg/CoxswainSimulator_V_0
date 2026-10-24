r"""Publish the race review on the race-card site, encrypted: review.html (a locked shell) + review.enc.

    .venv/Scripts/python.exe research/raceday/race_review_page.py        # builds race_review_2026.html
    .venv-tools/Scripts/python.exe research/raceday/race_review_site.py  # this: encrypt into site/

Everything with our data in it (the text, the numbers, the GPS track, the replay) goes into review.enc:
AES-256-GCM under a key from PBKDF2-SHA256 (600 000 iterations, random salt) of the site passphrase
(site_passphrase.txt). The shell holds only the page's styles, a password box and the drawing code; it
asks search engines not to index it and is not linked from the race card. A static host cannot refuse a
script that fetches review.enc, but what it fetches is ciphertext; the iteration count makes each guess at
the passphrase cost a script about as long as it costs a phone to unlock the page.
"""
import base64, json, os, re, secrets

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
import sys
sys.path.insert(0, CODE)
from review_crews import CFG, WHO                                          # noqa: E402  (REVIEW_CREW; default ours)
SITE = os.path.join(HERE, "site")
ITERATIONS = 600_000

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow,noarchive,nosnippet,noimageindex">
<meta name="referrer" content="no-referrer">
<title>Race review</title>
%(fonts)s
<style>
%(style)s
.lock{max-width:420px;margin:12vh auto 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:12px}
.lock input{font:inherit;font-size:16px;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
.lock .err{color:var(--loss);font-size:14px;min-height:1.2em}
</style>
</head>
<body>
<div id="root"><div class="wrap">
  <form class="lock" id="lock" autocomplete="off">
    <div class="eyebrow">Private</div>
    <h2>Race review</h2>
    <p class="muted">Enter the coxswains' password to open it.</p>
    <input id="pass" type="password" aria-label="Password" placeholder="password" autocomplete="current-password">
    <button type="submit" id="go">Open</button>
    <div class="err" id="err" role="status"></div>
  </form>
</div></div>
<script>
async function openReview(pass){
  const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const enc = await (await fetch("%(enc)s", {cache: "no-cache"})).json();
  const km = await crypto.subtle.importKey("raw", new TextEncoder().encode(pass), "PBKDF2", false, ["deriveKey"]);
  const key = await crypto.subtle.deriveKey({name: "PBKDF2", salt: b64(enc.salt), iterations: enc.iterations, hash: "SHA-256"},
                                            km, {name: "AES-GCM", length: 256}, false, ["decrypt"]);
  const pt = await crypto.subtle.decrypt({name: "AES-GCM", iv: b64(enc.iv)}, key, b64(enc.data));
  return JSON.parse(new TextDecoder().decode(pt));
}
function start(D){
%(script)s
}
document.getElementById("lock").addEventListener("submit", async e => {
  e.preventDefault();
  const go = document.getElementById("go"), err = document.getElementById("err");
  go.disabled = true; go.textContent = "Opening…"; err.textContent = "";
  try {
    const p = await openReview(document.getElementById("pass").value.trim());
    const root = document.getElementById("root"); root.innerHTML = p.html; document.title = p.title;
    start(p.data);
  } catch (x) {
    go.disabled = false; go.textContent = "Open"; err.textContent = "That password did not open it. Check it and try again.";
  }
});
</script>
</body>
</html>
"""


def main():
    page = open(os.path.join(HERE, CFG["html"]), encoding="utf-8").read()
    title = re.search(r"<title>(.*?)</title>", page).group(1)
    fonts = "\n".join(re.findall(r"<link [^>]*>", page.split("<style>")[0]))
    style = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    markup = page.split("</style>", 1)[1].split("<script>", 1)[0].strip()
    script = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    first, rest = script.split("\n", 2)[1], script.split("\n", 2)[2]
    assert first.startswith("const D = ") and first.endswith(";"), first[:40]
    data = json.loads(first[len("const D = "):-1])
    payload = json.dumps(dict(title=title, html=markup, data=data), separators=(",", ":")).encode("utf-8")
    passphrase = open(os.path.join(HERE, "site_passphrase.txt")).read().strip()
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(passphrase.encode("utf-8"))
    ct = AESGCM(key).encrypt(iv, payload, None)
    b64 = lambda b: base64.b64encode(b).decode("ascii")      # noqa: E731
    json.dump(dict(v=1, kdf="PBKDF2-SHA256", iterations=ITERATIONS, salt=b64(salt), iv=b64(iv), data=b64(ct)),
              open(os.path.join(SITE, CFG["site"] + ".enc"), "w"))
    shell = SHELL % dict(fonts=fonts, style=style, script=rest, enc=CFG["site"] + ".enc")
    # the shell must carry none of the data
    off = int(CFG["official"])
    for probe in ("%d:%02d" % (off // 60, off % 60), "Sammamish", "%d / %d" % (CFG["place"], CFG["field"]), "%.1f" % data["track"]["t"][5]):
        assert probe not in shell, probe
    open(os.path.join(SITE, CFG["site"] + ".html"), "w", encoding="utf-8").write(shell)
    print("written site/%s.html (%.0f kB) and site/%s.enc (%.0f kB encrypted, %d PBKDF2 iterations)"
          % (CFG["site"], len(shell) / 1024, CFG["site"], os.path.getsize(os.path.join(SITE, CFG["site"] + ".enc")) / 1024, ITERATIONS))


if __name__ == "__main__":
    main()
