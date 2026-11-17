"""A second, short pack for the human coder: which calls demand power? (CODEBOOK addendum of 16 Nov 2026, part b)

    .venv/Scripts/python.exe research/coxing/human_coding/pack_power.py
    set CODING_DATA=data\\local\\coxing\\human_coding_power & set CODING_PORT=8766 & .venv\\Scripts\\python.exe research/coxing/human_coding/serve.py

The early power-call response depended on the primary coder's reading of which calls demand power. This pack holds
the lead phrase of every directive power episode of the primary coder, the lead phrase of every power episode of the
lexical coder that is not one of those, and half as many phrases again drawn at random from the other coded phrases of
the same pieces, in random order with the usual context and nothing else, plus 10% unannounced repeats. Two fields
only: target and stage. Writes data/local/coxing/human_coding_power/ (pack_text.json, an empty pack_voice.json,
key.json); the page reads the two fields from the pack. Refuses to redraw once codes are logged.
"""
import json
import os
import secrets
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "..", "experiment"))
import examiner_checks as X  # noqa: E402
import rulecoder as RC  # noqa: E402
CP = X.CP

OUT = os.path.abspath(os.path.join(X.EXP, "..", "human_coding_power"))
SEED, MIN_GAP = 20261116, 100


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "log_text.jsonl")):
        raise SystemExit("codes are already logged against the existing pack; not redrawing")
    rng = np.random.default_rng(SEED)
    D = X.Data()
    blank = dict(stage="-", level="-", action="n", function="-", focus="-", ref="n", discrep="n", appraisal="neu", form="-", arousal="-")
    chosen, pool = [], []                                   # (session, phrase index, source)
    for sid in X.SES:
        s = D.s[sid]
        coded = [p for p in s["phrases"] if p["i"] in D.codes[sid] and s["t"][0] <= p["t"] <= s["t"][-1]]
        prim = [e for e in D.episodes(sid) if e["cls"] == "power" and s["t"][0] <= e["t"] <= s["t"][-1]]
        pj = np.array([e["j"] for e in prim], int)
        rule = {p["i"]: dict(blank, target=RC.code(p["text"])["target"]) for p in coded}
        lex = [e for e in CP.episodes(s["phrases"], rule) if e["target"] == "power" and s["t"][0] <= e["t"] <= s["t"][-1]]
        lex = [e for e in lex if len(pj) == 0 or np.min(np.abs(pj - int(np.searchsorted(s["t"], e["t"])))) > 1]
        mine = {e["i"]: "primary" for e in prim}
        for e in lex:
            mine.setdefault(e["i"], "lexical")
        chosen += [(sid, i, src) for i, src in mine.items()]
        pool += [(sid, p["i"], "control") for p in coded if p["i"] not in mine]
    n_ctrl = len(chosen) // 2
    ctrl = [pool[k] for k in rng.choice(len(pool), n_ctrl, replace=False)]
    key, items = {}, []
    for sid, i, src in chosen + ctrl:
        allph = D.s[sid]["phrases"]
        k = next(q for q, p in enumerate(allph) if p["i"] == i)
        iid = secrets.token_hex(4)
        key[iid] = dict(kind="text", session=sid, i=int(i), source=src)
        items.append(dict(id=iid, text=allph[k]["text"], before=[x["text"] for x in allph[max(k - 3, 0):k]], after=[x["text"] for x in allph[k + 1:k + 3]]))
    order = list(rng.permutation(len(items)))
    seq = [items[k] for k in order]
    n_rep = round(0.1 * len(seq))
    for src in sorted(rng.choice(len(seq) - MIN_GAP, n_rep, replace=False))[::-1]:
        pos = int(rng.integers(src + MIN_GAP, len(seq) + 1))
        rid = secrets.token_hex(4)
        key[rid] = dict(key[seq[src]["id"]], repeat_of=seq[src]["id"])
        seq.insert(pos, dict(seq[src], id=rid))
    intro = ("A short second task: for each highlighted phrase choose what it is about (Target, several allowed, the first is the main one) and its Stage. "
             "Nothing else is asked. About an hour; it saves after every phrase, so you can stop and come back.")
    json.dump(dict(items=seq, block=40, fields=["target", "stage"], intro=intro), open(os.path.join(OUT, "pack_text.json"), "w", encoding="utf-8"),
              indent=0, ensure_ascii=False)
    json.dump(dict(items=[]), open(os.path.join(OUT, "pack_voice.json"), "w"))
    json.dump(dict(seed=SEED, items=key), open(os.path.join(OUT, "key.json"), "w"), indent=0)
    n = {src: sum(v["source"] == src and "repeat_of" not in v for v in key.values()) for src in ("primary", "lexical", "control")}
    print("power pack: %d items (%s; %d repeats) in %s" % (len(seq), n, n_rep, OUT))


if __name__ == "__main__":
    main()
