import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import subprocess
import sys

BS = chr(92)
for f in sys.argv[1:]:
    s = open(f, encoding="utf-8").read()
    a = s.index(BS + "section{Introduction}")
    b = s.index(BS + "appendix")
    body = s[a:b]
    open("_main.tex", "w", encoding="utf-8").write(
        BS + "begin{document}\n" + body + "\n" + BS + "end{document}\n")
    out = subprocess.run(["texcount", "-1", "-q", "_main.tex"], capture_output=True, text=True).stdout
    print(f, out.strip())
