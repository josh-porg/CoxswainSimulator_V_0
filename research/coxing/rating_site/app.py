"""Self-hosted rating site for outside raters (CODEBOOK addendum of 7 Nov 2026). Standard library only.

    python app.py serve [--host 127.0.0.1] [--port 8080]
    python app.py invite "label" [--admin] [--exclude race_07,race_06] [--base-url https://ratings.example.org]
    python app.py raters

Environment: RATING_DATA (folder with pack.json, clips/ and the database; default data/local/coxing/rating_site),
RATING_TZ_HOURS (offset from UTC that defines a rater's day; default -8), RATING_SECURE=1 (send the cookie only over
HTTPS; set it when deployed behind HTTPS).

Each rater gets a personal invite link (no password). The link sets a cookie holding the rater's token; only its hash
is stored. A rater gives consent and a short profile, then codes up to 15 phrases a day on four fields and rates up
to 10 muffled clips, after which the day is closed. Items are served one at a time from the rater's queue (fewest
ratings first, random order per rater, the rater's own pieces excluded, one check phrase and one repeated clip per
day); the server only accepts an answer for the item it served. Ratings are append-only. The study lead's admin page
shows raters, coverage, check-item accuracy and Krippendorff's alpha.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import random
import secrets
import sqlite3
import sys
import threading
from http import cookies
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs
from wsgiref.simple_server import WSGIServer, make_server

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "local", "coxing", "rating_site"))
DATA = os.environ.get("RATING_DATA", DEFAULT_DATA)
TZ = dt.timezone(dt.timedelta(hours=float(os.environ.get("RATING_TZ_HOURS", "-8"))))
SECURE = os.environ.get("RATING_SECURE") == "1"
TEXT_PER_DAY, VOICE_PER_DAY = 15, 10
TARGETS = ["power", "rate", "length", "catch", "finish", "body", "set", "together", "steer", "info", "morale",
           "tactic", "count", "other"]
STAGES, ACTIONS, AROUSALS = ["inf", "ass", "dec"], ["y", "n"], ["up", "down", "-"]
LOCK = threading.Lock()
PACK = None


def pack():
    global PACK
    if PACK is None:
        PACK = json.load(open(os.path.join(DATA, "pack.json"), encoding="utf-8"))
        PACK["by_id"] = {x["id"]: x for x in PACK["items"]}
        PACK["clip_ids"] = {x["id"]: x for x in PACK["clips"]}
    return PACK


def db():
    con = sqlite3.connect(os.path.join(DATA, "ratings.db"), timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript("""
    CREATE TABLE IF NOT EXISTS raters (id INTEGER PRIMARY KEY, public_id TEXT UNIQUE, token_hash TEXT UNIQUE, label TEXT,
      is_admin INTEGER DEFAULT 0, exclude TEXT DEFAULT '', created TEXT, consent_at TEXT, role TEXT, years TEXT,
      level TEXT, seed INTEGER);
    CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY, rater_id INTEGER, kind TEXT, item_id TEXT, codes TEXT,
      rating INTEGER, flags TEXT, ms INTEGER, plays INTEGER, day TEXT, created TEXT, is_check INTEGER DEFAULT 0,
      is_repeat INTEGER DEFAULT 0, UNIQUE(rater_id, kind, item_id, is_repeat));
    CREATE TABLE IF NOT EXISTS pending (rater_id INTEGER PRIMARY KEY, kind TEXT, item_id TEXT, is_check INTEGER,
      is_repeat INTEGER, issued TEXT);
    """)
    return con


def today():
    return dt.datetime.now(TZ).strftime("%Y-%m-%d")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def h(token):
    return hashlib.sha256(token.encode()).hexdigest()


def order_key(seed, item_id):
    return hashlib.sha256(("%d:%s" % (seed, item_id)).encode()).hexdigest()


# ------------------------------------------------------------------ queue
def counts(con, kind):
    return {r["item_id"]: r["n"] for r in con.execute(
        "SELECT item_id, COUNT(*) n FROM ratings WHERE kind=? AND is_check=0 AND is_repeat=0 GROUP BY item_id", (kind,))}


def next_item(con, r):
    """The item this rater should do now, or None when today's quota (or the whole queue) is done."""
    p = pack()
    day, excl = today(), set(filter(None, (r["exclude"] or "").split(",")))
    pend = con.execute("SELECT * FROM pending WHERE rater_id=?", (r["id"],)).fetchone()
    if pend:
        return dict(pend)
    done = {(x["kind"], x["item_id"], x["is_repeat"]) for x in con.execute(
        "SELECT kind, item_id, is_repeat FROM ratings WHERE rater_id=?", (r["id"],))}
    seen_text = {i for k, i, _ in done if k == "text"}
    seen_clip = {i for k, i, rep in done if k == "voice" and not rep}
    n_text = con.execute("SELECT COUNT(*) FROM ratings WHERE rater_id=? AND kind='text' AND day=?", (r["id"], day)).fetchone()[0]
    n_voice = con.execute("SELECT COUNT(*) FROM ratings WHERE rater_id=? AND kind='voice' AND day=?", (r["id"], day)).fetchone()[0]
    rng = random.Random("%d:%s" % (r["seed"], day))
    pick = None
    if n_text < TEXT_PER_DAY:
        check_slot = rng.randrange(TEXT_PER_DAY)
        had_check = con.execute("SELECT COUNT(*) FROM ratings WHERE rater_id=? AND day=? AND is_check=1",
                                (r["id"], day)).fetchone()[0]
        pool = [x for x in p["items"] if x["id"] not in seen_text and x["session"] not in excl]
        checks = [x for x in pool if x["check"]]
        if n_text >= check_slot and not had_check and checks:
            pick = dict(kind="text", item_id=rng.choice(checks)["id"], is_check=1, is_repeat=0)
        else:
            c = counts(con, "text")
            regular = [x for x in pool if not x["check"]] or pool
            if regular:
                x = min(regular, key=lambda x: (c.get(x["id"], 0), order_key(r["seed"], x["id"])))
                pick = dict(kind="text", item_id=x["id"], is_check=0, is_repeat=0)
    if pick is None and n_voice < VOICE_PER_DAY:
        rep_slot = rng.randrange(VOICE_PER_DAY)
        had_rep = con.execute("SELECT COUNT(*) FROM ratings WHERE rater_id=? AND day=? AND is_repeat=1",
                              (r["id"], day)).fetchone()[0]
        repeated = {i for k, i, rep in done if k == "voice" and rep}
        earlier = [i for i in seen_clip if i not in repeated and con.execute(
            "SELECT day FROM ratings WHERE rater_id=? AND kind='voice' AND item_id=? AND is_repeat=0",
            (r["id"], i)).fetchone()[0] != day]
        if n_voice >= rep_slot and not had_rep and earlier:
            pick = dict(kind="voice", item_id=rng.choice(sorted(earlier)), is_check=0, is_repeat=1)
        else:
            c = counts(con, "voice")
            pool = [x for x in p["clips"] if x["id"] not in seen_clip and x["session"] not in excl]
            if pool:
                x = min(pool, key=lambda x: (c.get(x["id"], 0), order_key(r["seed"], x["id"])))
                pick = dict(kind="voice", item_id=x["id"], is_check=0, is_repeat=0)
    if pick:
        con.execute("INSERT OR REPLACE INTO pending VALUES (?,?,?,?,?,?)",
                    (r["id"], pick["kind"], pick["item_id"], pick["is_check"], pick["is_repeat"], now()))
        con.commit()
    return pick


def progress(con, r):
    day = today()
    q = lambda sql, *a: con.execute(sql, a).fetchone()[0]
    p = pack()
    covered = q("SELECT COUNT(DISTINCT item_id) FROM ratings WHERE kind='text' AND is_check=0")
    return dict(text_today=q("SELECT COUNT(*) FROM ratings WHERE rater_id=? AND kind='text' AND day=?", r["id"], day),
                voice_today=q("SELECT COUNT(*) FROM ratings WHERE rater_id=? AND kind='voice' AND day=?", r["id"], day),
                text_quota=TEXT_PER_DAY, voice_quota=VOICE_PER_DAY,
                mine=q("SELECT COUNT(*) FROM ratings WHERE rater_id=?", r["id"]),
                days=q("SELECT COUNT(DISTINCT day) FROM ratings WHERE rater_id=?", r["id"]),
                study_phrases=len(p["items"]), study_covered=covered,
                raters=q("SELECT COUNT(*) FROM raters WHERE consent_at IS NOT NULL"))


# ------------------------------------------------------------------ admin statistics
def alpha(units, metric="nominal"):
    """Krippendorff's alpha from {unit: [values]} (units with fewer than two values are ignored)."""
    units = [v for v in units.values() if len(v) >= 2]
    if not units:
        return None
    vals = [x for v in units for x in v]
    n = len(vals)
    d = (lambda a, b: 0.0 if a == b else 1.0) if metric == "nominal" else (lambda a, b: (a - b) ** 2)
    do = sum(sum(d(a, b) for i, a in enumerate(v) for j, b in enumerate(v) if i != j) / (len(v) - 1) for v in units) / n
    de = sum(d(a, b) for i, a in enumerate(vals) for j, b in enumerate(vals) if i != j) / (n * (n - 1))
    return None if de == 0 else round(1 - do / de, 3)


def admin_stats(con):
    p = pack()
    raters = []
    for r in con.execute("SELECT * FROM raters ORDER BY id"):
        rs = con.execute("SELECT * FROM ratings WHERE rater_id=?", (r["id"],)).fetchall()
        checks = [x for x in rs if x["is_check"]]
        ok = sum(1 for x in checks if json.loads(x["codes"])["targets"][0] == p["by_id"][x["item_id"]]["check"])
        raters.append(dict(public_id=r["public_id"], label=r["label"], admin=bool(r["is_admin"]), role=r["role"],
                           years=r["years"], level=r["level"], consent=r["consent_at"], exclude=r["exclude"],
                           text=sum(1 for x in rs if x["kind"] == "text"), voice=sum(1 for x in rs if x["kind"] == "voice"),
                           days=len({x["day"] for x in rs}), last=max((x["created"] for x in rs), default=None),
                           checks=len(checks), checks_ok=ok))
    text = con.execute("SELECT item_id, codes FROM ratings WHERE kind='text' AND is_check=0 AND is_repeat=0").fetchall()
    voice = con.execute("SELECT item_id, rating FROM ratings WHERE kind='voice' AND is_repeat=0").fetchall()
    cov = lambda rows, total: {str(k): v for k, v in sorted(_hist([x["item_id"] for x in rows], total).items())}
    units = lambda f: _units([(x["item_id"], json.loads(x["codes"])[f] if f != "target" else json.loads(x["codes"])["targets"][0]) for x in text])
    return dict(raters=raters, coverage_text=cov(text, len(p["items"])), coverage_voice=cov(voice, len(p["clips"])),
                alpha={f: alpha(units(f)) for f in ("target", "stage", "action", "arousal")} |
                {"voice (interval)": alpha(_units([(x["item_id"], x["rating"]) for x in voice]), "interval")},
                today=today())


def _units(pairs):
    u = {}
    for k, v in pairs:
        u.setdefault(k, []).append(v)
    return u


def _hist(ids, total):
    c = {}
    for i in ids:
        c[i] = c.get(i, 0) + 1
    out = {0: total - len(c)}
    for v in c.values():
        out[min(v, 4)] = out.get(min(v, 4), 0) + 1
    return out


# ------------------------------------------------------------------ WSGI
def respond(start, status, body, ctype="application/json", headers=()):
    b = body if isinstance(body, bytes) else json.dumps(body).encode()
    start(status, [("Content-Type", ctype), ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                   ("Referrer-Policy", "no-referrer")] + list(headers))
    return [b]


def app(env, start):
    path, method = env.get("PATH_INFO", "/"), env["REQUEST_METHOD"]
    page = lambda name: respond(start, "200 OK", open(os.path.join(HERE, name), "rb").read(), "text/html; charset=utf-8")
    if path.startswith("/i/") and method == "GET":                              # invite link: set the cookie
        token = path[3:]
        with LOCK:
            con = db()
            ok = con.execute("SELECT 1 FROM raters WHERE token_hash=?", (h(token),)).fetchone()
        if not ok:
            return respond(start, "404 Not Found", b"This invite link is not valid.", "text/plain")
        c = "rt=%s; Path=/; Max-Age=31536000; HttpOnly; SameSite=Lax%s" % (token, "; Secure" if SECURE else "")
        return respond(start, "303 See Other", b"", "text/plain", [("Set-Cookie", c), ("Location", "/")])
    if path == "/" and method == "GET":
        return page("page.html")
    if path == "/guide":
        return page("guide.html")
    jar = cookies.SimpleCookie(env.get("HTTP_COOKIE", ""))
    token = jar["rt"].value if "rt" in jar else ""
    with LOCK:
        con = db()
        r = con.execute("SELECT * FROM raters WHERE token_hash=?", (h(token),)).fetchone() if token else None
        if r is None:
            return respond(start, "401 Unauthorized", {"error": "invite only"})
        if path == "/admin" and r["is_admin"]:
            return page("admin.html")
        if path == "/api/admin" and r["is_admin"]:
            return respond(start, "200 OK", admin_stats(con))
        if path == "/api/admin/export" and r["is_admin"]:
            rows = [dict(x) for x in con.execute("SELECT r.public_id, r.role, r.years, r.level, g.* FROM ratings g "
                                                 "JOIN raters r ON r.id=g.rater_id ORDER BY g.id")]
            return respond(start, "200 OK", rows, headers=[("Content-Disposition", "attachment; filename=ratings.json")])
        if method == "POST" and env.get("CONTENT_TYPE", "").split(";")[0] != "application/json":
            return respond(start, "415 Unsupported Media Type", {"error": "json only"})
        body = {}
        if method == "POST":
            try:
                body = json.loads(env["wsgi.input"].read(min(int(env.get("CONTENT_LENGTH") or 0), 20000)) or b"{}")
            except ValueError:
                return respond(start, "400 Bad Request", {"error": "bad json"})
        if path == "/api/me":
            state = "consent" if not r["consent_at"] else "ready"
            return respond(start, "200 OK", dict(state=state, admin=bool(r["is_admin"]), progress=progress(con, r)))
        if path == "/api/consent" and method == "POST":
            if body.get("agree") is not True or not all(body.get(k) for k in ("role", "years", "level")):
                return respond(start, "400 Bad Request", {"error": "consent and profile needed"})
            con.execute("UPDATE raters SET consent_at=?, role=?, years=?, level=? WHERE id=?",
                        (now(), str(body["role"])[:40], str(body["years"])[:20], str(body["level"])[:40], r["id"]))
            con.commit()
            return respond(start, "200 OK", {"ok": True})
        if not r["consent_at"]:
            return respond(start, "403 Forbidden", {"error": "consent first"})
        if path == "/api/next":
            it = next_item(con, r)
            if it is None:
                return respond(start, "200 OK", dict(done=True, progress=progress(con, r)))
            out = dict(kind=it["kind"], id=it["item_id"], progress=progress(con, r))
            if it["kind"] == "text":
                x = pack()["by_id"][it["item_id"]]
                out.update(text=x["text"], before=x["before"], after=x["after"])
            return respond(start, "200 OK", out)
        if path.startswith("/clip/") and path.endswith(".wav"):
            cid = path[6:-4]
            pend = con.execute("SELECT * FROM pending WHERE rater_id=?", (r["id"],)).fetchone()
            if not pend or pend["kind"] != "voice" or pend["item_id"] != cid or cid not in pack()["clip_ids"]:
                return respond(start, "404 Not Found", b"", "text/plain")
            return respond(start, "200 OK", open(os.path.join(DATA, "clips", cid + ".wav"), "rb").read(), "audio/wav")
        if path in ("/api/text", "/api/voice") and method == "POST":
            pend = con.execute("SELECT * FROM pending WHERE rater_id=?", (r["id"],)).fetchone()
            kind = path[5:]
            if not pend or pend["kind"] != kind or pend["item_id"] != body.get("id"):
                return respond(start, "409 Conflict", {"error": "not the item served"})
            if kind == "text":
                c = body.get("codes") or {}
                tg = [t for t in c.get("targets", []) if t in TARGETS]
                only_skip = bool(tg) and all(t in ("count", "other") for t in tg)
                if not tg or len(tg) != len(set(tg)) or (not only_skip and not (
                        c.get("stage") in STAGES and c.get("action") in ACTIONS and c.get("arousal") in AROUSALS)):
                    return respond(start, "400 Bad Request", {"error": "incomplete"})
                codes = dict(targets=tg, stage="-" if only_skip else c["stage"], action="-" if only_skip else c["action"],
                             arousal="-" if only_skip else c["arousal"])
                vals = (json.dumps(codes), None, json.dumps([f for f in body.get("flags", []) if f in ("unsure", "misheard")]),
                        int(body.get("ms") or 0), None)
            else:
                rating = body.get("rating")
                if rating not in range(1, 8) or not body.get("played_full"):
                    return respond(start, "400 Bad Request", {"error": "play the clip to the end, then rate 1-7"})
                vals = (None, rating, "[]", int(body.get("ms") or 0), int(body.get("plays") or 1))
            try:
                con.execute("INSERT INTO ratings (rater_id, kind, item_id, codes, rating, flags, ms, plays, day, created, "
                            "is_check, is_repeat) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (r["id"], kind, pend["item_id"], *vals, today(), now(), pend["is_check"], pend["is_repeat"]))
            except sqlite3.IntegrityError:
                pass
            con.execute("DELETE FROM pending WHERE rater_id=?", (r["id"],))
            con.commit()
            return respond(start, "200 OK", {"ok": True})
    return respond(start, "404 Not Found", {"error": "not found"})


class Server(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    i = sub.add_parser("invite")
    i.add_argument("label")
    i.add_argument("--admin", action="store_true")
    i.add_argument("--exclude", default="", help="comma-separated session ids the rater coxed or rowed in")
    i.add_argument("--base-url", default="http://127.0.0.1:8080")
    sub.add_parser("raters")
    a = ap.parse_args()
    if a.cmd == "serve":
        p = pack()
        if not p.get("final"):
            print("WARNING: pack.json is not final (names not reviewed). Do not share links until it is.")
        print("rating site on http://%s:%d  (data: %s)" % (a.host, a.port, DATA))
        make_server(a.host, a.port, app, server_class=Server).serve_forever()
    elif a.cmd == "invite":
        con = db()
        token = secrets.token_urlsafe(24)
        bad = [x for x in a.exclude.split(",") if x and x not in {y["session"] for y in pack()["items"]}]
        if bad:
            sys.exit("unknown session(s): %s" % ", ".join(bad))
        con.execute("INSERT INTO raters (public_id, token_hash, label, is_admin, exclude, created, seed) VALUES (?,?,?,?,?,?,?)",
                    ("R" + secrets.token_hex(3), h(token), a.label, int(a.admin), a.exclude, now(), secrets.randbelow(2 ** 31)))
        con.commit()
        print("%s/i/%s" % (a.base_url.rstrip("/"), token))
    else:
        for r in db().execute("SELECT public_id, label, is_admin, exclude, consent_at, role FROM raters"):
            print(dict(r))


if __name__ == "__main__":
    main()
