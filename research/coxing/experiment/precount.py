"""Pre-fill count phrases (bare numbers / count words) so hand coding covers the rest."""
import re, sys
NUM = set("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen twenty more and that's thats in".split())
def is_count(txt):
    w = re.findall(r"[a-z0-9']+", txt.lower())
    if not w:
        return True
    return all(x.isdigit() or x in NUM for x in w) and any(x.isdigit() or x in NUM - {"more", "and", "that's", "thats", "in"} for x in w)
for sid in sys.argv[1:]:
    rows = [l.rstrip("\n").split("|", 2) for l in open("%s/phrases.txt" % sid, encoding="utf-8")]
    n = 0
    with open("../experiment/todo_%s.txt" % sid, "w", encoding="utf-8") as f:
        for i, t, txt in rows:
            if is_count(txt):
                n += 1
                continue
            f.write("%s|%s|%s\n" % (i, t, txt))
    print(sid, len(rows), "phrases,", n, "counts pre-filled,", len(rows) - n, "to code")
