r"""SRA CoxBox Overlay: put the CoxBox data over cox-camera video, synced, with no internet needed.

    "tools\overlay\CoxBox Overlay.bat"        (or: .venv-tools\Scripts\pythonw.exe tools\overlay\app.py)

1. Add the videos from the camera and the CoxBox CSV exports (or a whole folder of both).
2. The app pairs each video with the CoxBox session recorded at the same time (the camera's clock
   runs fast; the setting holds by how much, and is updated after each synced video). Change any
   pairing by double-clicking it.
3. Pick the layout and which HUD items to show; Preview shows one frame.
4. Generate: per video it reads the stroke rhythm from the picture, syncs it to the CoxBox (and,
   if ticked, checks it against the splits called in the audio), and writes the video with the
   overlay. A video with no CoxBox session gets only a stroke rate estimated from the camera, and
   the app asks before doing that.

Offline: ffmpeg is bundled, the speech model is loaded from the local cache only, and nothing is
fetched. Settings live in %APPDATA%\CoxBoxOverlay, working files in %LOCALAPPDATA%\CoxBoxOverlay.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import camera_motion as CM                                       # noqa: E402
import coxbox_data as C                                          # noqa: E402
import coxbox_overlay as O                                       # noqa: E402

APP = "SRA CoxBox Overlay"


def _dirs():
    """(settings dir, cache dir, default output dir) in each platform's usual places."""
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        return (os.path.join(os.environ.get("APPDATA", home), "CoxBoxOverlay"),
                os.path.join(os.environ.get("LOCALAPPDATA", home), "CoxBoxOverlay", "cache"),
                os.path.join(home, "Videos", "CoxBox overlays"))
    if sys.platform == "darwin":
        return (os.path.join(home, "Library", "Application Support", "CoxBoxOverlay"),
                os.path.join(home, "Library", "Caches", "CoxBoxOverlay"),
                os.path.join(home, "Movies", "CoxBox overlays"))
    return (os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.join(home, ".config")), "coxbox-overlay"),
            os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.join(home, ".cache")), "coxbox-overlay"),
            os.path.join(home, "Videos", "CoxBox overlays"))


_SETTINGS_DIR, CACHE, _OUTDIR = _dirs()
SETTINGS = os.path.join(_SETTINGS_DIR, "settings.json")


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def version():
    try:
        return open(O.resource("VERSION"), encoding="utf-8").read().strip()
    except OSError:
        return "source"
NO_SESSION = "No CoxBox data (camera-estimated rate)"
LAYOUTS = [("A", "A · Top strip (default)"), ("B", "B · Top corners"), ("D", "D · Side panel, picture uncovered")]
ZONES = [("Pacific, daylight (UTC−7)", -7), ("Pacific (UTC−8)", -8), ("Mountain, daylight (UTC−6)", -6),
         ("Mountain (UTC−7)", -7), ("Central, daylight (UTC−5)", -5), ("Central (UTC−6)", -6),
         ("Eastern, daylight (UTC−4)", -4), ("Eastern (UTC−5)", -5), ("UTC", 0)]
#: the Osmo's clock was measured 14.4 min fast on 2026-10-04 (two sessions, 4 s apart): calibrated
DEFAULTS = dict(layout="A", fields=list(O.FIELDS), calls=True, camera_fast=14.4, calibrated=True,
                zone=None, outdir=_OUTDIR)


class Cancelled(Exception):
    pass


# -- files ---------------------------------------------------------------------------------------
def probe_video(path):
    err = subprocess.run([O.ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True,
                         **O.NOWIN).stderr
    ct = re.search(r"creation_time\s*:\s*(\S+)", err)
    du = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not du:
        raise ValueError("not a video ffmpeg can read")
    dur = int(du.group(1)) * 3600 + int(du.group(2)) * 60 + float(du.group(3))
    start = dt.datetime.strptime(ct.group(1)[:19], "%Y-%m-%dT%H:%M:%S") if ct else None
    return start, dur


def cache_key(path):
    st = os.stat(path)
    return "%s_%d_%d" % (os.path.splitext(os.path.basename(path))[0], st.st_size, int(st.st_mtime))


class Video:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.cam_start_utc, self.duration = probe_video(path)
        self.session = None            # a Session, or None
        self.auto = True               # pairing chosen automatically
        self.status = "ready"

    def true_start_utc(self, camera_fast_min):
        if self.cam_start_utc is None:
            return None
        return self.cam_start_utc - dt.timedelta(minutes=camera_fast_min)


def session_label(s):
    return "%s  ·  %d:%02d  ·  %s m" % (s.start.strftime("%a %d %b %H:%M"), int(s.duration) // 60,
                                        int(s.duration) % 60, "{:,}".format(int(s.distance[-1])))


# -- the app -------------------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("%s  %s" % (APP, version()))
        self.geometry("1180x760")
        self.minsize(980, 640)
        self.settings = dict(DEFAULTS)
        try:
            self.settings.update(json.load(open(SETTINGS)))
        except Exception:
            pass
        if self.settings["zone"] is None:
            off = -time.altzone if time.daylight and time.localtime().tm_isdst else -time.timezone
            hours = round(off / 3600)
            self.settings["zone"] = next((n for n, h in ZONES if h == hours), "UTC")
        self.videos, self.sessions, self.extra_files = [], [], []
        self.q = queue.Queue()
        self.worker = None
        self.cancel = threading.Event()
        self._build()
        self.after(150, self._poll)

    # -- layout of the window --------------------------------------------------------------------
    def _build(self):
        style = ttk.Style(self)
        for theme in {"win32": ("vista",), "darwin": ("aqua",)}.get(sys.platform, ("clam",)):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                pass
        style.configure("Big.TButton", padding=(14, 8))
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(1, weight=1)

        # files
        left = ttk.Frame(root)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
        left.rowconfigure(1, weight=3)
        left.rowconfigure(4, weight=1)
        left.columnconfigure(0, weight=1)
        bar = ttk.Frame(left)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(bar, text="Add videos…", command=self.add_videos).pack(side="left")
        ttk.Button(bar, text="Add CoxBox files…", command=self.add_coxbox).pack(side="left", padx=6)
        ttk.Button(bar, text="Add folder…", command=self.add_folder).pack(side="left")
        ttk.Button(bar, text="Remove", command=self.remove_selected).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear all", command=self.clear_all).pack(side="left")

        cols = ("recorded", "length", "coxbox", "status")
        self.tree = ttk.Treeview(left, columns=cols, show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="Video")
        for c, w, t in (("recorded", 120, "Recorded"), ("length", 55, "Length"),
                        ("coxbox", 230, "CoxBox session (double-click)"), ("status", 85, "Status")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, minwidth=50, stretch=(c == "coxbox"))
        self.tree.column("#0", width=200, minwidth=120)
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", self._edit_pairing)

        ttk.Label(left, text="CoxBox sessions loaded", font=("Segoe UI", 10, "bold")).grid(
            row=3, column=0, sticky="w", pady=(10, 2))
        self.stree = ttk.Treeview(left, columns=("file",), show="tree headings", height=5)
        self.stree.heading("#0", text="Session")
        self.stree.heading("file", text="File")
        self.stree.column("#0", width=300)
        self.stree.grid(row=4, column=0, sticky="nsew")

        # options
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        lf = ttk.LabelFrame(right, text="Layout", padding=8)
        lf.pack(fill="x")
        self.layout = tk.StringVar(value=self.settings["layout"])
        for code, label in LAYOUTS:
            ttk.Radiobutton(lf, text=label, value=code, variable=self.layout, command=self._save).pack(anchor="w")
        hf = ttk.LabelFrame(right, text="Show on the video", padding=8)
        hf.pack(fill="x", pady=8)
        self.field_vars = {}
        for i, k in enumerate(O.FIELDS):
            v = tk.BooleanVar(value=k in self.settings["fields"])
            self.field_vars[k] = v
            ttk.Checkbutton(hf, text=O.FIELD_NAMES[k], variable=v, command=self._save).grid(
                row=i // 2, column=i % 2, sticky="w", padx=(0, 18))
        sf = ttk.LabelFrame(right, text="Sync", padding=8)
        sf.pack(fill="x")
        self.calls = tk.BooleanVar(value=self.settings["calls"])
        import transcribe_calls as TC
        self.calls_ok = TC.model_available()
        cb = ttk.Checkbutton(sf, text="Check the sync against the splits I call", variable=self.calls,
                             command=self._save)
        cb.pack(anchor="w")
        ttk.Label(sf, text="speech recognition, runs on this computer", foreground="#5f6b7a").pack(
            anchor="w", padx=(22, 0))
        if not self.calls_ok:
            self.calls.set(False)
            cb.state(["disabled"])
            ttk.Label(sf, text="Speech model not installed on this computer: sync uses the stroke rhythm only.",
                      foreground="#8a5a00").pack(anchor="w")
        row = ttk.Frame(sf)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Camera clock runs fast by").pack(side="left")
        self.fast = tk.StringVar(value="%.1f" % self.settings["camera_fast"])
        ttk.Spinbox(row, from_=-120, to=120, increment=0.1, width=6, textvariable=self.fast,
                    command=self._repair).pack(side="left", padx=4)
        ttk.Label(row, text="min").pack(side="left")
        self.calibrated = tk.BooleanVar(value=self.settings.get("calibrated", False))
        ttk.Checkbutton(sf, text="Camera clock is calibrated", variable=self.calibrated,
                        command=self._save).pack(anchor="w", pady=(4, 0))
        ttk.Label(sf, text="untick for a new camera, or after resetting its clock", foreground="#5f6b7a").pack(
            anchor="w", padx=(22, 0))
        row2 = ttk.Frame(sf)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="CoxBox time zone").pack(side="left")
        self.zone = tk.StringVar(value=self.settings["zone"])
        z = ttk.Combobox(row2, textvariable=self.zone, values=[n for n, _ in ZONES], state="readonly", width=28)
        z.pack(side="left", padx=4)
        z.bind("<<ComboboxSelected>>", lambda e: self._repair())
        of = ttk.LabelFrame(right, text="Save videos to", padding=8)
        of.pack(fill="x", pady=8)
        self.outdir = tk.StringVar(value=self.settings["outdir"])
        ttk.Entry(of, textvariable=self.outdir).pack(side="left", fill="x", expand=True)
        ttk.Button(of, text="Choose…", command=self.choose_outdir).pack(side="left", padx=(6, 0))

        act = ttk.Frame(right)
        act.pack(fill="x", pady=(4, 0))
        ttk.Button(act, text="Preview selected", command=self.preview).pack(side="left")
        self.go = ttk.Button(act, text="Generate videos", style="Big.TButton", command=self.generate)
        self.go.pack(side="right")
        self.stop = ttk.Button(act, text="Cancel", command=self.cancel_run, state="disabled")
        self.stop.pack(side="right", padx=6)

        # progress and log
        bottom = ttk.Frame(root)
        bottom.grid(row=1, column=1, sticky="nsew", pady=(8, 0))
        bottom.rowconfigure(2, weight=1)
        bottom.columnconfigure(0, weight=1)
        self.stage = tk.StringVar(value="Add videos and CoxBox files to begin.")
        ttk.Label(bottom, textvariable=self.stage).grid(row=0, column=0, sticky="w")
        self.bar = ttk.Progressbar(bottom, maximum=1.0)
        self.bar.grid(row=1, column=0, sticky="ew", pady=4)
        self.log = tk.Text(bottom, height=10, wrap="word", relief="solid", borderwidth=1,
                           background="#ffffff", foreground="#18222e")
        self.log.grid(row=2, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(bottom, orient="vertical", command=self.log.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    # -- settings --------------------------------------------------------------------------------
    def _fields(self):
        return [k for k in O.FIELDS if self.field_vars[k].get()]

    def _camera_fast(self):
        try:
            return float(self.fast.get())
        except ValueError:
            return self.settings["camera_fast"]

    def _zone_hours(self):
        return dict(ZONES).get(self.zone.get(), 0)

    def _save(self):
        self.settings.update(layout=self.layout.get(), fields=self._fields(), calls=bool(self.calls.get()),
                             camera_fast=self._camera_fast(), zone=self.zone.get(), outdir=self.outdir.get(),
                             calibrated=bool(self.calibrated.get()) if hasattr(self, "calibrated") else
                             self.settings.get("calibrated", False))
        try:
            os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
            json.dump(self.settings, open(SETTINGS, "w"), indent=1)
        except OSError:
            pass

    def say(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    # -- adding files ----------------------------------------------------------------------------
    def add_videos(self, paths=None):
        paths = paths or filedialog.askopenfilenames(title="Add videos", filetypes=[
            ("Videos", "*.mp4 *.MP4 *.mov *.MOV"), ("All files", "*.*")])
        for p in paths:
            if any(os.path.samefile(p, v.path) for v in self.videos):
                continue
            try:
                self.videos.append(Video(p))
            except Exception as e:
                messagebox.showwarning(APP, "Could not read %s:\n%s" % (os.path.basename(p), e))
        self._repair()

    def add_coxbox(self, paths=None):
        paths = paths or filedialog.askopenfilenames(title="Add CoxBox files", filetypes=[
            ("CoxBox exports", "*.csv *.fit *.gpx"), ("All files", "*.*")])
        for p in paths:
            if p.lower().endswith(".csv"):
                if any(os.path.samefile(p, s.path) for s in self.sessions):
                    continue
                try:
                    self.sessions.append(C.load_csv(p))
                except Exception as e:
                    messagebox.showwarning(APP, "%s is not a CoxBox CSV export I can read:\n%s"
                                           % (os.path.basename(p), e))
            else:
                self.extra_files.append(p)
        stems = {os.path.splitext(s.path)[0] for s in self.sessions}
        lone = [f for f in self.extra_files if os.path.splitext(f)[0] not in stems]
        if lone:
            messagebox.showinfo(APP, "The app reads the CoxBox CSV export, which has the per-stroke data.\n\n"
                                "These have no matching CSV:\n  " + "\n  ".join(os.path.basename(f) for f in lone)
                                + "\n\nExport the same session as CSV from NK LiNK Logbook and add it.")
            self.extra_files = [f for f in self.extra_files if f not in lone]
        self._repair()

    def add_folder(self):
        d = filedialog.askdirectory(title="Add a folder of videos and CoxBox files")
        if not d:
            return
        files = [os.path.join(d, f) for f in os.listdir(d)]
        self.add_videos([f for f in files if f.lower().endswith((".mp4", ".mov"))])
        cox = [f for f in files if f.lower().endswith((".csv", ".fit", ".gpx"))]
        if cox:
            self.add_coxbox(cox)

    def remove_selected(self):
        sel = set(self.tree.selection())
        self.videos = [v for i, v in enumerate(self.videos) if str(i) not in sel]
        ssel = set(self.stree.selection())
        if ssel:
            gone = [self.sessions[int(i)] for i in ssel]
            self.sessions = [s for s in self.sessions if s not in gone]
            for v in self.videos:
                if v.session in gone:
                    v.session, v.auto = None, True
        self._repair()

    def clear_all(self):
        if self.worker and self.worker.is_alive():
            return
        self.videos, self.sessions, self.extra_files = [], [], []
        self._repair()

    def choose_outdir(self):
        d = filedialog.askdirectory(title="Save videos to", initialdir=self.outdir.get())
        if d:
            self.outdir.set(d)
            self._save()

    # -- pairing ---------------------------------------------------------------------------------
    def _session_utc(self, s):
        start = s.start - dt.timedelta(hours=self._zone_hours())
        return start, start + dt.timedelta(seconds=s.duration)

    def _repair(self):
        """Re-pair every automatically paired video by time overlap, then redraw."""
        self._save()
        fast = self._camera_fast()
        for v in self.videos:
            if not v.auto:
                continue
            v.session = None
            a = v.true_start_utc(fast)
            if a is None:
                continue
            b = a + dt.timedelta(seconds=v.duration)
            best = 0.0
            for s in self.sessions:
                s0, s1 = self._session_utc(s)
                pad = dt.timedelta(minutes=3)
                ov = (min(b, s1 + pad) - max(a, s0 - pad)).total_seconds()
                taken = any(w is not v and w.session is s for w in self.videos)
                if ov > best and not taken:
                    best, v.session = ov, s
        self._redraw()

    def _redraw(self):
        self.tree.delete(*self.tree.get_children())
        zone = dt.timedelta(hours=self._zone_hours())
        for i, v in enumerate(self.videos):
            a = v.true_start_utc(self._camera_fast())
            rec = (a + zone).strftime("%a %d %b %H:%M") if a else "?"
            ses = session_label(v.session) if v.session else NO_SESSION
            if v.session and not v.auto:
                ses += "  (chosen)"
            self.tree.insert("", "end", iid=str(i), text=v.name, values=(
                rec, "%d:%02d" % (int(v.duration) // 60, int(v.duration) % 60), ses, v.status))
        self.stree.delete(*self.stree.get_children())
        for i, s in enumerate(self.sessions):
            self.stree.insert("", "end", iid=str(i), text=session_label(s), values=(os.path.basename(s.path),))
        n = len(self.videos)
        paired = sum(1 for v in self.videos if v.session)
        if n:
            self.stage.set("%d video%s, %d with CoxBox data." % (n, "" if n == 1 else "s", paired))

    def _edit_pairing(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        v = self.videos[int(iid)]
        win = tk.Toplevel(self)
        win.title("CoxBox session for %s" % v.name)
        win.transient(self)
        ttk.Label(win, text="Which CoxBox session belongs to %s?" % v.name, padding=10).pack(anchor="w")
        options = ["Automatic (by time)", NO_SESSION] + [session_label(s) for s in self.sessions]
        var = tk.StringVar(value=("Automatic (by time)" if v.auto else
                                  (session_label(v.session) if v.session else NO_SESSION)))
        cb = ttk.Combobox(win, textvariable=var, values=options, state="readonly", width=60)
        cb.pack(padx=10, fill="x")

        def ok():
            choice = var.get()
            if choice.startswith("Automatic"):
                v.auto = True
            elif choice == NO_SESSION:
                v.auto, v.session = False, None
            else:
                v.auto, v.session = False, self.sessions[options.index(choice) - 2]
            win.destroy()
            self._repair()
        ttk.Button(win, text="OK", command=ok).pack(pady=10)

    # -- preview ---------------------------------------------------------------------------------
    def preview(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo(APP, "Select a video in the list to preview.")
            return
        v = self.videos[int(sel[0])]
        layout, fields = self.layout.get(), self._fields()
        self.stage.set("Making a preview of %s…" % v.name)

        def work():
            try:
                if v.session:
                    s = v.session
                    s0, _ = self._session_utc(s)
                    off = (s0 - v.true_start_utc(self._camera_fast())).total_seconds()
                    tv = min(max(off + s.duration * 0.5, 1.0), v.duration - 1)
                    m = O.session_moments(s, off)(tv)
                    note = "Preview from the clock (approximate sync) at video %d:%02d." % (tv // 60, tv % 60)
                else:
                    tv = v.duration * 0.5
                    m = O.Moment(elapsed=None, distance=None, rate=30.0, split=None, per_stroke=None,
                                 rate_estimated=True, history=[], sample=True)
                    note = "No CoxBox data: the rate shown here is a placeholder."
                img = O.compose(O.grab_frame(v.path, tv), layout, m, fields)
                self.q.put(("preview", (img, v.name, note)))
            except Exception as e:
                self.q.put(("error", "Preview failed: %s" % e))
        threading.Thread(target=work, daemon=True).start()

    def _show_preview(self, img, name, note):
        from PIL import ImageTk
        win = tk.Toplevel(self)
        win.title("Preview · %s" % name)
        scale = min(1.0, (self.winfo_screenwidth() * 0.8) / img.width)
        shown = img.resize((int(img.width * scale), int(img.height * scale)))
        photo = ImageTk.PhotoImage(shown)
        lab = ttk.Label(win, image=photo)
        lab.image = photo
        lab.pack()
        ttk.Label(win, text=note, padding=6).pack(anchor="w")
        self.stage.set("Preview ready.")

    # -- generating ------------------------------------------------------------------------------
    def generate(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.videos:
            messagebox.showinfo(APP, "Add the videos first.")
            return
        if not self._fields():
            messagebox.showinfo(APP, "Tick at least one item to show on the video.")
            return
        unpaired = [v for v in self.videos if v.session is None]
        if not self.sessions:
            if not messagebox.askyesno(APP, "No CoxBox data has been added.\n\n"
                                       "If you have the CoxBox files (CSV) for these recordings, add them first: "
                                       "without them the video only shows a stroke rate estimated from the "
                                       "camera, with no split, distance, time or map.\n\n"
                                       "Generate without CoxBox data?", icon="warning", default="no"):
                return
        elif unpaired:
            names = "\n  ".join(v.name for v in unpaired)
            if not messagebox.askyesno(APP, "These videos have no CoxBox session:\n  %s\n\n"
                                       "If their CoxBox files exist, add them (or double-click a video to "
                                       "pair it by hand). Otherwise they get only a stroke rate estimated from "
                                       "the camera.\n\nGenerate anyway?" % names, icon="warning", default="no"):
                return
        outdir = self.outdir.get()
        os.makedirs(outdir, exist_ok=True)
        self._save()
        jobs = [(v, v.session) for v in self.videos]
        opts = dict(layout=self.layout.get(), fields=self._fields(), calls=bool(self.calls.get()) and self.calls_ok,
                    fast=self._camera_fast(), zone=self._zone_hours(), outdir=outdir,
                    prior=20.0 if self.calibrated.get() else None)
        self.cancel.clear()
        self.go.state(["disabled"])
        self.stop.state(["!disabled"])
        self.worker = threading.Thread(target=self._run, args=(jobs, opts), daemon=True)
        self.worker.start()

    def cancel_run(self):
        self.cancel.set()
        self.stage.set("Cancelling…")

    def _progress(self, i, n, label):
        def f(frac):
            if self.cancel.is_set():
                raise Cancelled()
            self.q.put(("progress", ((i + frac) / n, label, frac)))
        return f

    def _run(self, jobs, o):
        n = len(jobs)
        done = []
        for i, (v, s) in enumerate(jobs):
            try:
                self.q.put(("status", (v, "working")))
                os.makedirs(CACHE, exist_ok=True)
                key = cache_key(v.path)
                mpath = os.path.join(CACHE, key + "_motion.npz")
                if os.path.exists(mpath):
                    m = np.load(mpath)
                    mt, signals = m["t"], {k: m[k] for k in ("dy", "dx", "diff")}
                else:
                    self.q.put(("log", "%s: reading the stroke rhythm from the picture…" % v.name))
                    t, dy, dx, diff = CM.motion(v.path, progress=self._progress(i, n, "%s · reading motion" % v.name),
                                                total=v.duration)
                    np.savez(mpath, t=t, dy=dy, dx=dx, diff=diff)
                    mt, signals = t, dict(dy=dy, dx=dx, diff=diff)
                if s is not None:
                    calls = []
                    if o["calls"]:
                        cpath = os.path.join(CACHE, key + "_calls.json")
                        if os.path.exists(cpath):
                            segs = json.load(open(cpath, encoding="utf-8"))
                        else:
                            self.q.put(("log", "%s: listening to the calls…" % v.name))
                            import transcribe_calls as TC
                            segs = TC.transcribe(v.path, progress=self._progress(i, n, "%s · listening to calls"
                                                                                 % v.name))
                            json.dump(segs, open(cpath, "w", encoding="utf-8"))
                        calls = C.called_splits(segs)
                    s0 = s.start - dt.timedelta(hours=o["zone"])
                    guess = (s0 - v.true_start_utc(o["fast"])).total_seconds()
                    off, info = C.best_sync(s, mt, signals, guess, calls, prior_sd=o.get("prior"))
                    if off is None:
                        self.q.put(("log", "%s: could not sync (%s). Check the pairing or the camera clock "
                                           "setting; skipped." % (v.name, info.get("reason"))))
                        self.q.put(("status", (v, "not synced")))
                        continue
                    msg = "%s: synced, CoxBox start at video %d:%04.1f (stroke rate agrees to %.1f spm" % (
                        v.name, off // 60, off % 60, info["rate_err"])
                    if info.get("calls"):
                        msg += "; your called splits to %.1f s" % info["calls"][0]
                    self.q.put(("log", msg + ")."))
                    # the camera clock, measured: keep the setting current for the next videos
                    measured = (v.cam_start_utc - (s0 - dt.timedelta(seconds=off))).total_seconds() / 60.0
                    if abs(measured - o["fast"]) > 1.0:
                        self.q.put(("log", "  note: the camera clock measured %.1f min fast, not the %.1f set; "
                                           "updated." % (measured, o["fast"])))
                    self.q.put(("camera_fast", measured))
                    o["fast"] = measured
                    at = O.session_moments(s, off)
                else:
                    self.q.put(("log", "%s: no CoxBox data, showing the stroke rate estimated from the camera."
                                % v.name))
                    at = O.estimate_moments(mt, signals["dy"], 0.0, float(mt[-1]), min_conf=0.5)
                out = os.path.join(o["outdir"], "%s - overlay %s.mp4" % (os.path.splitext(v.name)[0], o["layout"]))
                k = 2
                while os.path.exists(out):
                    out = os.path.join(o["outdir"], "%s - overlay %s (%d).mp4"
                                       % (os.path.splitext(v.name)[0], o["layout"], k))
                    k += 1
                self.q.put(("log", "%s: writing the video…" % v.name))
                O.render_video(v.path, out, o["layout"], at, fields=o["fields"],
                               progress=self._progress(i, n, "%s · writing video" % v.name))
                done.append(out)
                self.q.put(("status", (v, "done")))
                self.q.put(("log", "%s: saved %s" % (v.name, out)))
            except Cancelled:
                self.q.put(("status", (v, "cancelled")))
                self.q.put(("log", "Cancelled."))
                break
            except Exception as e:
                self.q.put(("status", (v, "failed")))
                self.q.put(("log", "%s: failed: %s" % (v.name, e)))
        self.q.put(("finished", done))

    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "progress":
                    frac, label, part = data
                    self.bar["value"] = frac
                    self.stage.set("%s  %d%%" % (label, round(part * 100)))
                elif kind == "log":
                    self.say(data)
                elif kind == "status":
                    v, st = data
                    v.status = st
                    self._redraw()
                elif kind == "camera_fast":
                    self.fast.set("%.2f" % data)
                    self.calibrated.set(True)
                    self._save()
                elif kind == "preview":
                    self._show_preview(*data)
                elif kind == "error":
                    self.stage.set(data)
                    messagebox.showerror(APP, data)
                elif kind == "finished":
                    self.go.state(["!disabled"])
                    self.stop.state(["disabled"])
                    self.bar["value"] = 1.0 if data else 0.0
                    self.stage.set("Finished: %d video%s saved." % (len(data), "" if len(data) == 1 else "s"))
                    if data and messagebox.askyesno(APP, "%d video%s saved to\n%s\n\nOpen the folder?"
                                                    % (len(data), "" if len(data) == 1 else "s",
                                                       os.path.dirname(data[0]))):
                        open_folder(os.path.dirname(data[0]))
        except queue.Empty:
            pass
        self.after(150, self._poll)


def batch(argv):
    """Headless: the same pipeline as the window, for scripts and the release smoke test.

    app --batch --videos A.MP4 B.MP4 --coxbox S1.csv S2.csv --out DIR [--layout A] [--fields rate split ...]
            [--no-calls] [--camera-fast 14.4] [--zone -7] [--wide]
    """
    import argparse
    ap = argparse.ArgumentParser(prog="app --batch")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--coxbox", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", default="A", choices=["A", "B", "D"])
    ap.add_argument("--fields", nargs="*", default=list(O.FIELDS))
    ap.add_argument("--no-calls", action="store_true")
    ap.add_argument("--camera-fast", type=float, default=DEFAULTS["camera_fast"])
    ap.add_argument("--zone", type=float, default=None, help="CoxBox UTC offset, hours (default: this computer's)")
    ap.add_argument("--wide", action="store_true", help="the camera clock is not calibrated: search widely")
    a = ap.parse_args(argv)
    import transcribe_calls as TC
    zone = a.zone
    if zone is None:
        off = -time.altzone if time.daylight and time.localtime().tm_isdst else -time.timezone
        zone = round(off / 3600)
    videos = [Video(p) for p in a.videos]
    sessions = [C.load_csv(p) for p in a.coxbox if p.lower().endswith(".csv")]
    for v in videos:                       # pair by time overlap, as the window does
        st = v.true_start_utc(a.camera_fast)
        if st is None:
            continue
        en = st + dt.timedelta(seconds=v.duration)
        best = 0.0
        for s in sessions:
            s0 = s.start - dt.timedelta(hours=zone)
            s1 = s0 + dt.timedelta(seconds=s.duration)
            pad = dt.timedelta(minutes=3)
            ov = (min(en, s1 + pad) - max(st, s0 - pad)).total_seconds()
            if ov > best and not any(w is not v and w.session is s for w in videos):
                best, v.session = ov, s
        print("%s -> %s" % (v.name, session_label(v.session) if v.session else NO_SESSION), flush=True)
    os.makedirs(a.out, exist_ok=True)
    holder = types_simple()
    holder.q, holder.cancel = queue.Queue(), threading.Event()
    opts = dict(layout=a.layout, fields=a.fields, calls=(not a.no_calls) and TC.model_available(),
                fast=a.camera_fast, zone=zone, outdir=a.out, prior=None if a.wide else 20.0)

    def drain():
        while True:
            try:
                kind, data = holder.q.get_nowait()
            except queue.Empty:
                return
            if kind in ("log", "finished"):
                print(data if kind == "log" else "finished: %s" % data, flush=True)
    t = threading.Thread(target=App._run, args=(holder, [(v, v.session) for v in videos], opts), daemon=True)
    t.start()
    while t.is_alive():
        t.join(2.0)
        drain()
    drain()


class types_simple:
    """Just enough of App for ``App._run`` to run headless."""
    def _progress(self, i, n, label):
        def f(frac):
            if self.cancel.is_set():
                raise Cancelled()
        return f


def selftest(outdir):
    """Prove a build works on this machine, offline: ffmpeg, fonts, the speech model, the motion
    reader and all three layouts rendered over a synthetic clip. Writes ``selftest.log``; exit 0/1."""
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "selftest.log"), "w", encoding="utf-8")
    ok = True

    def say(msg):
        log.write(msg + "\n")
        log.flush()
        try:
            print(msg, flush=True)
        except Exception:
            pass

    def check(name, fn):
        nonlocal ok
        try:
            detail = fn()
            say("PASS  %s%s" % (name, (": %s" % detail) if detail else ""))
        except Exception as e:
            ok = False
            say("FAIL  %s: %s: %s" % (name, type(e).__name__, e))

    say("%s %s on %s (python %s)" % (APP, version(), sys.platform, sys.version.split()[0]))
    ff = O.ffmpeg_exe()
    check("ffmpeg runs", lambda: subprocess.run([ff, "-version"], capture_output=True, text=True, check=True,
                                                **O.NOWIN).stdout.splitlines()[0][:40])
    check("fonts", lambda: ", ".join(sorted(f for f in O.BARLOW.values() if os.path.exists(O.resource("fonts", f))))
          or (_ for _ in ()).throw(FileNotFoundError("no bundled fonts")))

    def model():
        import transcribe_calls as TC
        path = TC.bundled_model()
        if not TC.model_available():
            raise FileNotFoundError("speech model not available (calls check will be off)")
        from faster_whisper import WhisperModel
        WhisperModel(path or "small.en", device="cpu", compute_type="int8", local_files_only=True)
        return "loaded from %s" % ("the bundle" if path else "the cache")
    check("speech model", model)

    clip = os.path.join(outdir, "synthetic.mp4")
    check("make a synthetic clip", lambda: subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:d=8",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-metadata", "creation_time=2026-10-04T19:00:00Z", clip], check=True, **O.NOWIN) and None)
    check("read motion", lambda: "%d samples" % len(CM.motion(clip)[0]))
    t = np.arange(0.0, 6.0, 2.0)
    s = C.Session(start=dt.datetime(2026, 10, 4, 12, 0, 0), t=t, distance=t * 4.1, split=np.full(t.size, 122.0),
                  speed=np.full(t.size, 4.1), rate=np.full(t.size, 30.0), strokes=np.arange(1, t.size + 1.0),
                  per_stroke=np.full(t.size, 8.2), lat=47.64 + t * 1e-5, lon=-122.33 + t * 1e-5, path="synthetic")
    for lay in ("A", "B", "D"):
        out = os.path.join(outdir, "synthetic_%s.mp4" % lay)
        check("render layout %s" % lay, lambda lay=lay, out=out: "%.0f kB" % (
            os.path.getsize(O.render_video(clip, out, lay, O.session_moments(s, 1.0))) / 1024))
    say("RESULT: %s" % ("all checks passed" if ok else "FAILED"))
    log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        k = sys.argv.index("--selftest")
        sys.exit(selftest(sys.argv[k + 1] if len(sys.argv) > k + 1 else os.path.join(CACHE, "selftest")))
    if "--batch" in sys.argv:
        batch(sys.argv[1:])
    else:
        App().mainloop()
