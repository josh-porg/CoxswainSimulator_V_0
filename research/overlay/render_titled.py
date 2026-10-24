"""Render the eight videos with their title cards (cards.json), in date order, with the released pipeline."""
import glob, os, sys
ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import app
V = r"C:\Users\satur\Downloads\race videos\DJI_%s_D.MP4"
ids = ["20260925090905_0117", "20260925093420_0119", "20261002083010_0123", "20261002090450_0124",
       "20261002092830_0125", "20261002122646_0131", "20261004111335_0132", "20261004154606_0134"]
app.batch(["--batch", "--videos"] + [V % i for i in ids]
          + ["--coxbox"] + glob.glob(os.path.join(ROOT, "data", "local", "coxbox", "*.csv"))
          + ["--out", os.path.join(ROOT, "data", "local", "overlay", "videos_titled"),
             "--cards", os.path.join(ROOT, "data", "local", "overlay", "cards.json")])
print("ALL DONE", flush=True)
