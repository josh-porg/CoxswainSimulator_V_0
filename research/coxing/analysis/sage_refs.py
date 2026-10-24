"""sagej.cls sets secnumdepth to -2, so \ref to a section prints nothing. Name the target."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import io
import re

BS = chr(92)
p = "paper_sage.tex"
s = io.open(p, encoding="utf-8").read()

# repair the carriage return introduced by a non-raw "\ref" in an earlier edit
s = s.replace("Section~\nef{sec:coupled},", "Section~" + BS + "ref{sec:coupled},")

NAMES = {
    "sec:video": "Recovering boat data from video",
    "sec:rejected": "What validation rejected",
    "sec:survived": "What survived, and how far it goes",
    "sec:coupled": "Calls and boat as one coupled process",
    "app:cats": "Call category definitions",
    "app:results": "Full results",
    "app:sources": "Source recordings",
}
pat = re.compile(r"(Section|Appendix)~" + re.escape(BS) + r"ref\{([a-z]+:[a-z]+)\}")


def sub(m):
    kind, lab = m.group(1), m.group(2)
    word = "section" if kind == "Section" else "appendix"
    return "the %s " % word + BS + "emph{" + NAMES[lab] + "}"


s, n = pat.subn(sub, s)
# sentence-initial and parenthetical forms read better without the article
s = s.replace("(the section " + BS + "emph", "(section " + BS + "emph")
s = s.replace("(the appendix " + BS + "emph", "(appendix " + BS + "emph")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print(n, "references named")
