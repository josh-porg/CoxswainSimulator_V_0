r"""Publish the club lines comparison on the race-card site, encrypted under its own password.

    .venv-tools/Scripts/python.exe research/raceday/compare_site.py

compare.html (compare_page.py) is split like the race reviews: everything with data in it goes into
site/compare.enc (AES-256-GCM, PBKDF2-SHA256, 600 000 iterations); site/compare.html is the locked shell.
The password is its own, in compare_passphrase.txt (not the race card's), created on first run if absent.
"""
import base64, json, os, re, secrets, sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
sys.path.insert(0, CODE)
os.environ.setdefault("REVIEW_CREW", "ours")
import race_review_site as RS                                                # noqa: E402  (the shell)

SITE = os.path.join(HERE, "site")
PASS = os.path.join(HERE, "compare_passphrase.txt")
# words of the course that a coxswain's name can also be (then only the name's forms are checked)
COURSE_WORDS = {"bridge", "river", "street", "avenue", "beach", "landing"}


def main():
    page = open(os.path.join(HERE, "compare.html"), encoding="utf-8").read()
    title = re.search(r"<title>(.*?)</title>", page).group(1)
    fonts = "\n".join(re.findall(r"<link [^>]*>", page.split("<style>")[0]))
    style = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    markup = page.split("</style>", 1)[1].split("<script>", 1)[0].strip()
    script = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    first, rest = script.split("\n", 2)[1], script.split("\n", 2)[2]
    assert first.startswith("const D = ") and first.endswith(";"), first[:40]
    data = json.loads(first[len("const D = "):-1])
    payload = json.dumps(dict(title=title, html=markup, data=data), separators=(",", ":")).encode("utf-8")
    passphrase = open(PASS).read().strip()
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=RS.ITERATIONS).derive(passphrase.encode("utf-8"))
    ct = AESGCM(key).encrypt(iv, payload, None)
    b64 = lambda b: base64.b64encode(b).decode("ascii")      # noqa: E731
    json.dump(dict(v=1, kdf="PBKDF2-SHA256", iterations=RS.ITERATIONS, salt=b64(salt), iv=b64(iv), data=b64(ct)),
              open(os.path.join(SITE, "compare.enc"), "w"))
    shell = RS.SHELL % dict(fonts=fonts, style=style, script=rest, enc="compare.enc")
    shell = shell.replace("<title>Race review</title>", "<title>Club lines</title>").replace("<h2>Race review</h2>", "<h2>Club lines compared</h2>") \
                 .replace("Enter the coxswains' password to open it.", "Enter the line-comparison password to open it.")
    N = json.load(open(os.path.join(HERE, "compare_names.json")))
    for c in N["crews"].values():                             # the shell must carry none of the data
        # "Bridge" is also a course word (Eliot Bridge, the bridge labels in the code): check the name forms instead
        probes = ([c["cox"] + "'s", "N. " + c["cox"]] if c["cox"].lower() in COURSE_WORDS else [c["cox"]])
        assert not any(p in shell for p in probes) and c["event_name"] not in shell, c["cox"]
    assert "Sammamish" not in shell
    # the names: their own file and their own password; the comparison's payload must carry none of them
    text = json.dumps(dict(title=title, html=markup, data=data))
    for c in N["crews"].values():
        for v in (c["cox"], c["event_name"]):
            assert '"%s"' % v not in text and "%s's" % v not in text, "identifying value in the comparison payload"
    from review_crews import CREWS as _CREWS
    for k in _CREWS:
        assert '"%s"' % k not in text, "a crew key in the comparison payload"
    for f in ("cox", "event_name", "bow", "place", "official", "window_et", "start_et"):
        assert '"%s":' % f not in text, "an identifying field in the comparison payload"
    assert "Sammamish" not in text
    npass = open(os.path.join(HERE, "compare_names_passphrase.txt")).read().strip()
    assert npass != passphrase
    salt2, iv2 = secrets.token_bytes(16), secrets.token_bytes(12)
    key2 = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt2, iterations=RS.ITERATIONS).derive(npass.encode("utf-8"))
    ct2 = AESGCM(key2).encrypt(iv2, json.dumps(N, separators=(",", ":")).encode("utf-8"), None)
    json.dump(dict(v=1, kdf="PBKDF2-SHA256", iterations=RS.ITERATIONS, salt=b64(salt2), iv=b64(iv2), data=b64(ct2)),
              open(os.path.join(SITE, "compare-names.enc"), "w"))
    open(os.path.join(SITE, "compare.html"), "w", encoding="utf-8").write(shell)
    print("written site/compare.html (%.0f kB) and site/compare.enc (%.0f kB encrypted)"
          % (len(shell) / 1024, os.path.getsize(os.path.join(SITE, "compare.enc")) / 1024))


if __name__ == "__main__":
    main()
