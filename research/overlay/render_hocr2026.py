"""Render the HOCR 2026 race video (Women's Veteran 4+) with its title card (cards_hocr2026.json), as render_titled.py."""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import app
MEDIA = json.load(open(os.path.join(ROOT, "data", "local", "coxing", "local_paths.json"), encoding="utf-8"))["media_root"]   # private, machine-specific
V = os.path.join(MEDIA, "HOCR 2026", "DJI_20261017105126_0149_D.MP4")
app.batch(["--batch", "--videos", V,
           "--coxbox", os.path.join(ROOT, "data", "local", "coxbox", "Sharing file  CoxBox 011009704 20261017 0737AM.csv"),
           "--out", os.path.join(ROOT, "data", "local", "overlay", "videos_titled", "hocr2026"),
           "--cards", os.path.join(ROOT, "data", "local", "overlay", "cards_hocr2026.json")])
print("ALL DONE", flush=True)
