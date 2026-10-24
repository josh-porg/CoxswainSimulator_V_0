import re, sys, warnings
warnings.simplefilter("ignore")
from pypdf import PdfReader
path, pattern, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12
r = PdfReader(path)
txt = " ".join((p.extract_text() or "") for p in r.pages)
txt = re.sub(r"\s+", " ", txt)
seen = 0
for m in re.finditer(pattern, txt, re.I):
    a, b = max(0, m.start() - 180), min(len(txt), m.end() + 240)
    s = txt[a:b].encode("ascii", "replace").decode()
    print("...%s...\n" % s)
    seen += 1
    if seen >= n:
        break
print("[%d hits shown of %d total, %d pages]" % (seen, len(re.findall(pattern, txt, re.I)), len(r.pages)))
