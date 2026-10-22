r"""Encrypt the race card bundle for the public static site (run in .venv-tools, which has cryptography).

    .venv-tools/Scripts/python.exe scripts/raceday_encrypt.py

AES-256-GCM with a key from PBKDF2-SHA256 (250 000 iterations, random salt) over a passphrase
shared by the coxswains. The page decrypts it with the browser's Web Crypto. The passphrase lives
in ``data/local/raceday/site_passphrase.txt`` (never committed, never published); it is created with
a random value the first time. Output: ``data/local/raceday/site/data.enc`` (JSON: salt, iv,
ciphertext, all base64, and the iteration count) and a copy of the page, ``index.html``.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import shutil

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RACEDAY = os.path.join(ROOT, "data", "local", "raceday")
SITE = os.path.join(RACEDAY, "site")
ITERATIONS = 250_000
WORDS = ("powerhouse", "weeks", "eliot", "anderson", "magazine", "riverstreet", "western", "bigturn",
         "coxswain", "stroke", "catch", "finish", "rudder", "arch", "charles", "basin", "harbor", "boathouse")


def passphrase() -> str:
    path = os.path.join(RACEDAY, "site_passphrase.txt")
    if not os.path.exists(path):
        words = [secrets.choice(WORDS) for _ in range(4)]
        open(path, "w").write("-".join(words) + "-%02d\n" % secrets.randbelow(100))
    return open(path).read().strip()


def main():
    data = open(os.path.join(RACEDAY, "site_bundle.json"), "rb").read()
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(
        passphrase().encode("utf-8"))
    ct = AESGCM(key).encrypt(iv, data, None)
    os.makedirs(SITE, exist_ok=True)
    b64 = lambda b: base64.b64encode(b).decode("ascii")      # noqa: E731
    json.dump(dict(v=1, kdf="PBKDF2-SHA256", iterations=ITERATIONS, salt=b64(salt), iv=b64(iv), data=b64(ct)),
              open(os.path.join(SITE, "data.enc"), "w"))
    shutil.copy(os.path.join(ROOT, "scripts", "raceday_site_index.html"), os.path.join(SITE, "index.html"))
    brief = os.path.join(ROOT, "scripts", "raceday_briefing.html")
    if os.path.exists(brief):
        shutil.copy(brief, os.path.join(SITE, "briefing.html"))
    print("written %s (%.0f kB encrypted); passphrase in data/local/raceday/site_passphrase.txt"
          % (os.path.join(SITE, "data.enc"), os.path.getsize(os.path.join(SITE, "data.enc")) / 1024))


if __name__ == "__main__":
    main()
