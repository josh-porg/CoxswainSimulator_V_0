"""Build the athlete's training-plan website from the team erg sheet: the plan regenerates whenever a new test lands.

    python research/training/site_build.py --config <config.json> --out <dir> [--xlsx <file>]

Runs on a schedule in the site repository's GitHub Action (see research/training/README.md). Inputs from the
environment (GitHub secrets in CI, never committed anywhere):

    ERG_SHEET_ID     the team erg sheet's Google id; the sheet is link-viewable, so it downloads as xlsx
                     without credentials (not needed with --xlsx)
    ATHLETE_NAME     the athlete's row label on the "Erg Test POST HERE" tab
    SITE_PASSPHRASE  the passphrase the page asks for

The config holds only what is not on the sheet and identifies nobody: the team week, free days, own-volume
settings, key races, the plan's start and the target. Tests, sex, birth year and body mass come from the
athlete's row. The squad yardstick is recomputed from the whole sheet on every build.

Output: <out>/index.html (the page shell, research/training/site/index.html, which holds no data),
<out>/data.enc and <out>/.nojekyll. data.enc is AES-256-GCM with a key from PBKDF2-SHA256 (250 000
iterations, random salt) over the passphrase, the scheme of the race-card site (scripts/raceday_encrypt.py);
the browser decrypts it with Web Crypto. No name enters the bundle; figures that would reveal body mass are
left out (rules(p, grams=False)).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import secrets
import shutil
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan_generator as pg            # noqa: E402
import squad_progression as sq         # noqa: E402

ITERATIONS = 250_000
LB = 0.45359237


def download(sheet_id, folder):
    url = "https://docs.google.com/spreadsheets/d/%s/export?format=xlsx" % sheet_id
    path = os.path.join(folder, "sheet.xlsx")
    with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)
    return path


def athlete_row(rowers, name):
    key = name.strip().casefold()
    hits = [r for r in rowers if str(r["name"]).strip().casefold() == key]
    if len(hits) != 1:
        raise SystemExit("athlete row not found uniquely on the erg sheet (%d matches); check ATHLETE_NAME" % len(hits))
    return hits[0]


def profile_from(config, row):
    p = json.loads(json.dumps(config))
    p["tests"] = [dict(date=d.isoformat(), distance_m=5000, time_s=round(t, 1)) for d, t in row["tests"]]
    p["sex"] = row["sex"]
    if row.get("born"):
        p["birth_year"] = int(row["born"])
    if row.get("weight"):
        p["body_mass_kg"] = round(float(row["weight"]) * LB, 1)
    return p


def bundle(p, squad_path):
    z = pg.zones(p)
    proj = pg.projection(p, squad_path)
    n_weeks = (p["target"]["by"] - p["start"]).days // 7 + 1
    weeks = pg.plan(p, n_weeks)
    tests = pg.test_schedule(p)
    ms = pg.milestones(proj, tests)
    done = [dict(date=t["date"].isoformat(), time=t["time_s"], watts=pg.watts(t["time_s"] / 10.0))
            for t in p["tests"] if int(t["distance_m"]) == 5000]
    return dict(
        v=1,
        updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        tests=done,
        latest=dict(date=z["test"]["date"].isoformat(), time=z["test"]["time_s"], watts=z["watts_5k"]),
        ftp=z["ftp"], ftp_factor=pg.FTP_FROM_5K, zones=z["zones"],
        target=dict(time=p["target"]["time_s"], by=p["target"]["by"].isoformat(), watts=proj["target_watts"]),
        projection=dict(required_rate=proj["required_rate"], p_target=proj["p_target"], ladder=proj["ladder"],
                        path=proj["path"]),
        milestones=[dict(m, what=w) for m, (_, w) in zip(ms, tests)],
        macro=pg.macrocycle(p),
        weeks=weeks,
        strength=[dict(phase=k, focus=f, rx=rx, per_week=n) for k, (f, rx, n) in pg.STRENGTH.items()],
        exercises=list(pg.EXERCISES),
        rules=[dict(title=t, items=items) for t, items in pg.rules(p, grams=False)],
        evidence=pg.EVIDENCE,
        start=p["start"].isoformat(),
    )


def encrypt(data: bytes, passphrase: str):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(
        passphrase.encode("utf-8"))
    b64 = lambda b: base64.b64encode(b).decode("ascii")      # noqa: E731
    return dict(v=1, kdf="PBKDF2-SHA256", iterations=ITERATIONS, salt=b64(salt), iv=b64(iv),
                data=b64(AESGCM(key).encrypt(iv, data, None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlsx", help="a local copy of the sheet instead of downloading it")
    ap.add_argument("--plain", action="store_true", help="also write the unencrypted bundle (local checks only)")
    a = ap.parse_args()
    name, passphrase = os.environ.get("ATHLETE_NAME"), os.environ.get("SITE_PASSPHRASE")
    if not name or not passphrase:
        raise SystemExit("ATHLETE_NAME and SITE_PASSPHRASE must be set")
    config = json.load(open(a.config, encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        xlsx = a.xlsx or download(os.environ["ERG_SHEET_ID"], tmp)
        rowers, blocks = sq.read(xlsx)
        pairs = sq.yearly_changes(rowers)
        squad_path = os.path.join(tmp, "squad.json")
        json.dump(dict(pairs=pairs, women=sq.summarise(pairs, "F"), men=sq.summarise(pairs, "M")),
                  open(squad_path, "w"), default=float)
        prof_path = os.path.join(tmp, "profile.json")
        json.dump(profile_from(config, athlete_row(rowers, name)), open(prof_path, "w"))
        p = pg.load_profile(prof_path)
        b = bundle(p, squad_path)
    raw = json.dumps(b, default=str, separators=(",", ":")).encode("utf-8")
    os.makedirs(a.out, exist_ok=True)
    json.dump(encrypt(raw, passphrase), open(os.path.join(a.out, "data.enc"), "w"))
    shutil.copy(os.path.join(HERE, "site", "index.html"), os.path.join(a.out, "index.html"))
    open(os.path.join(a.out, ".nojekyll"), "w").close()
    if a.plain:
        open(os.path.join(a.out, "bundle.plain.json"), "wb").write(raw)
    print("built: %d tests, latest 5k %s, %d weeks planned, %d rowers on the sheet, bundle %.0f kB"
          % (len(b["tests"]), pg.mmss(b["latest"]["time"]), len(b["weeks"]), len(rowers), len(raw) / 1024))


if __name__ == "__main__":
    main()
