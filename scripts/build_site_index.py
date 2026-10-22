"""Regenerate scripts/raceday_site_index.html from the Claude card page (data/local/raceday/race_card.html)
and scripts/raceday_site_loader.js, so the two pages share one renderer. Run after editing either:

    python scripts/build_site_index.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = '  document.querySelectorAll(".crews button[data-crew]").forEach(function(b){\n    b.addEventListener("click"'
CSS = """.lock{display:flex;flex-direction:column;gap:10px}
.lock input,.card textarea{font:inherit;font-size:16px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--paper);color:var(--ink);width:100%}
.unlock{font:inherit;font-weight:600;border:0;border-radius:10px;padding:10px 14px;background:var(--accent);color:var(--card);cursor:pointer}
.unlock.ghost{background:transparent;color:var(--muted);border:1px solid var(--line)}
.unlock:focus-visible,.lock input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.row2{display:flex;gap:8px;margin-top:8px}
details.card summary{cursor:pointer;font-weight:600}
"""


def main():
    src = open(os.path.join(ROOT, "data", "local", "raceday", "race_card.html"), encoding="utf-8").read()
    loader = open(os.path.join(ROOT, "scripts", "raceday_site_loader.js"), encoding="utf-8").read()
    s = src[:src.index(MARK)] + loader + "\n</html>\n"
    s = s.replace('</div>\n<script>', '<div id="daypanel"></div>\n</div>\n<script>', 1)
    s = s.replace('var PREFIX = "charles-race-card:";', 'var PREFIX = "charles-site:";')
    s = s.replace(".planv{display:flex;flex-direction:column;gap:8px}",
                  CSS + ".planv{display:flex;flex-direction:column;gap:8px}", 1)
    s = s.replace("Private to the coxswains. The reasoning behind every number is in the night-before brief.",
                  "Private to the coxswains. Weather and river flow update themselves each time the page opens; "
                  "the plan is from the night-before run.")
    open(os.path.join(ROOT, "scripts", "raceday_site_index.html"), "w", encoding="utf-8").write(s)
    print("written scripts/raceday_site_index.html")


if __name__ == "__main__":
    main()
