"""Local server for the human coding page (CODEBOOK addendum of 6 Nov 2026). Private data never leave this machine.

    .venv/Scripts/python.exe research/coxing/human_coding/serve.py        then open http://127.0.0.1:8765

Serves index.html beside this file, the pack written by pack.py (data/local/coxing/human_coding/) and the voice
clips, and appends every submitted code to log_text.jsonl / log_voice.jsonl. Each log line carries the hash of the
line before it (a hash chain), so an edit to any earlier line is detectable; an item can be logged once only. The
voice task opens when every text item is logged. key.json (which piece and phrase each id is) is never served.
"""
import hashlib
import http.server
import json
import os
import re
import time

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
DATA = os.environ.get("CODING_DATA", os.path.join(ROOT, "data", "local", "coxing", "human_coding"))   # CODING_DATA: a test copy
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
PORT = 8765


def read_log(name):
    p = os.path.join(DATA, name)
    return [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []


def append(name, rec):
    log = read_log(name)
    if any(r["id"] == rec["id"] for r in log):
        return False
    rec["logged_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["prev"] = log[-1]["hash"] if log else "0" * 16
    rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
    with open(os.path.join(DATA, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


class H(http.server.BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            text = json.load(open(os.path.join(DATA, "pack_text.json"), encoding="utf-8"))
            voice = json.load(open(os.path.join(DATA, "pack_voice.json")))
            done_t = [r["id"] for r in read_log("log_text.jsonl")]
            done_v = [r["id"] for r in read_log("log_voice.jsonl")]
            unlocked = len(done_t) >= len(text["items"])
            return self.send(200, dict(text=text, done_text=done_t, voice=voice if unlocked else None,
                                       done_voice=done_v, voice_unlocked=unlocked))
        m = re.fullmatch(r"/clip/([0-9a-f]{8})\.wav", self.path)
        if m and os.path.exists(os.path.join(DATA, "voice", m.group(1) + ".wav")):
            return self.send(200, open(os.path.join(DATA, "voice", m.group(1) + ".wav"), "rb").read(), "audio/wav")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not re.fullmatch(r"[0-9a-f]{8}", str(body.get("id", ""))):
            return self.send(400, {"error": "bad id"})
        if self.path == "/api/text":
            rec = {k: body.get(k) for k in ("id", "codes", "flags", "ms")}
        elif self.path == "/api/voice":
            if not read_log("log_text.jsonl") or len(read_log("log_text.jsonl")) < len(
                    json.load(open(os.path.join(DATA, "pack_text.json"), encoding="utf-8"))["items"]):
                return self.send(403, {"error": "finish the text coding first"})
            rec = {k: body.get(k) for k in ("id", "rating", "plays", "ms")}
        else:
            return self.send(404, {"error": "not found"})
        ok = append("log_text.jsonl" if self.path == "/api/text" else "log_voice.jsonl", rec)
        self.send(200 if ok else 409, {"ok": ok})


if __name__ == "__main__":
    print("coding page: http://127.0.0.1:%d  (Ctrl+C to stop)" % PORT)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
