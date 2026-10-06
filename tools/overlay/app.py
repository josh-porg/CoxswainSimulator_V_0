r"""SRA CoxBox Overlay: put the CoxBox data over cox-camera video, synced, with no internet needed.

    "tools\overlay\CoxBox Overlay.bat"        (or: .venv-tools\Scripts\pythonw.exe tools\overlay\app.py)

1. Add the videos from the camera and the CoxBox CSV exports (or a whole folder of both).
2. The app pairs each video with the CoxBox session recorded at the same time (the camera's clock
   runs fast; the setting holds by how much, and is updated after each synced video). Change any
   pairing by double-clicking it.
3. Pick the layout and which HUD items to show; Preview shows one frame.
4. Optionally, a title card: the race's name and the lineup for the first seconds, over a b-roll
   clip (retimed to fit) or over the start of the race video.
5. Generate: per video it reads the stroke rhythm from the picture, syncs it to the CoxBox (and,
   if ticked, checks it against the splits called in the audio), and writes the video with the
   overlay. A video with no CoxBox session gets only a stroke rate estimated from the camera, and
   the app asks before doing that.

Offline: ffmpeg is bundled, the speech model is loaded from the local cache only, and nothing is
fetched. Settings live in %APPDATA%\CoxBoxOverlay, working files in %LOCALAPPDATA%\CoxBoxOverlay.
"""
from __future__ import annotations

import datetime as dt
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
import title_card as T                                           # noqa: E402

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
#: the camera clock drifts (14.06 min fast on 2026-09-25, 14.40 on 2026-10-04: ~2.3 s a day), so each
#: measurement is kept with the time it was made, and a video uses the one nearest it in time; the
#: prior widens by CLOCK_DRIFT per day between them
CLOCK_DRIFT = 3.0                                                # s per day
CLOCK_SEED = [["2026-10-04T15:00:00", 14.4]]


def clock_for(history, when, fallback):
    """``(minutes fast, days away)``: the measurement nearest ``when`` (camera time), else ``fallback``."""
    if not history or when is None:
        return fallback, 0.0
    at, fast = min(history, key=lambda h: abs((dt.datetime.fromisoformat(h[0]) - when).total_seconds()))
    return fast, abs((dt.datetime.fromisoformat(at) - when).total_seconds()) / 86400.0


DEFAULTS = dict(layout="A", fields=list(O.FIELDS), calls=True, camera_fast=14.4, calibrated=True, clock=CLOCK_SEED,
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


def mmss(sec):
    """``-0:13.3`` for -13.3 s: the sign first, so a CoxBox started before the video reads right."""
    return "%s%d:%04.1f" % ("-" if sec < 0 else "", abs(sec) // 60, abs(sec) % 60)


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
        self.card = None               # a title_card.TitleCard, or None

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
        self.geometry("1320x780")
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

        cols = ("recorded", "length", "coxbox", "card", "status")
        self.tree = ttk.Treeview(left, columns=cols, show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="Video")
        for c, w, t in (("recorded", 120, "Recorded"), ("length", 55, "Length"),
                        ("coxbox", 230, "CoxBox session (double-click)"), ("card", 70, "Title card"),
                        ("status", 85, "Status")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, minwidth=50, stretch=(c == "coxbox"), anchor="center" if c == "card" else "w")
        self.tree.column("#0", width=200, minwidth=120)
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", self._edit_pairing)
        self.tree.bind("<Configure>", lambda e: self._fit_columns())

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
        ttk.Button(act, text="Title card…", command=self.edit_card).pack(side="left", padx=6)
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
            ses = v.session.start.strftime("%a %d %b %H:%M") if v.session else "none: camera rate"
            if v.session and not v.auto:
                ses += " (chosen)"
            card = "" if v.card is None else {"video": "yes", "still": "still", "broll": "b-roll"}[v.card.under]
            self.tree.insert("", "end", iid=str(i), text=v.name, values=(
                rec, "%d:%02d" % (int(v.duration) // 60, int(v.duration) % 60), ses, card, v.status))
        self._fit_columns()
        self.stree.delete(*self.stree.get_children())
        for i, s in enumerate(self.sessions):
            self.stree.insert("", "end", iid=str(i), text=session_label(s), values=(os.path.basename(s.path),))
        n = len(self.videos)
        paired = sum(1 for v in self.videos if v.session)
        if n:
            self.stage.set("%d video%s, %d with CoxBox data." % (n, "" if n == 1 else "s", paired))

    def _fit_columns(self):
        """Every column visible at any text size: the short ones as wide as their text, the video
        name up to its full length, and the CoxBox session takes what is left."""
        from tkinter import font as tkfont
        f = tkfont.nametofont("TkDefaultFont")
        pad = f.measure("00")
        need = {"recorded": f.measure("Sun 04 Oct 07:59") + pad, "length": f.measure("Length") + pad,
                "card": f.measure("Title card") + pad, "status": f.measure("not synced") + pad}
        avail = self.tree.winfo_width() - 4
        names = [v.name for v in self.videos] or ["DJI_00000000000000_0000_D.MP4"]
        video = min(max(f.measure(n) for n in names) + 2 * pad, int(avail * 0.34))
        cox = max(avail - video - sum(need.values()), f.measure("Sun 04 Oct 08:03 (chosen)") + pad)
        self.tree.column("#0", width=video)
        for c, w in need.items():
            self.tree.column(c, width=w)
        self.tree.column("coxbox", width=cox)

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

    # -- title card ------------------------------------------------------------------------------
    def edit_card(self):
        """The title card for the selected videos (all of them if none is selected)."""
        if not self.videos:
            messagebox.showinfo(APP, "Add the videos first.")
            return
        sel = [self.videos[int(i)] for i in self.tree.selection()] or list(self.videos)
        first = next((v.card for v in sel if v.card is not None), None)
        last = T.TitleCard.from_dict(self.settings["card_last"]) if self.settings.get("card_last") else T.TitleCard()
        card = first or last
        win = tk.Toplevel(self)
        win.title("Title card")
        win.transient(self)
        win.resizable(False, False)
        f = ttk.Frame(win, padding=12)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        who = sel[0].name if len(sel) == 1 else "%d videos" % len(sel)
        ttk.Label(f, text="Title card for %s" % who, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w")
        on = tk.BooleanVar(value=True)            # opening this dialog is asking for a card
        ttk.Checkbutton(f, text="Show a title card at the start", variable=on).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 8))
        title, sub = tk.StringVar(value=card.title), tk.StringVar(value=card.subtitle)
        ttk.Label(f, text="Race").grid(row=2, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(f, textvariable=title, width=52).grid(row=2, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Label(f, text="Second line").grid(row=3, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(f, textvariable=sub, width=52).grid(row=3, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Label(f, text="event, boat, date: optional", foreground="#5f6b7a").grid(row=4, column=1, sticky="w")
        ttk.Label(f, text="Lineup").grid(row=5, column=0, sticky="nw", pady=(6, 0))
        names = tk.Text(f, width=40, height=10, wrap="none", relief="solid", borderwidth=1,
                        background="#ffffff", foreground="#18222e", font="TkDefaultFont")
        names.grid(row=5, column=1, columnspan=2, sticky="ew", pady=(6, 2))
        names.insert("1.0", T.lineup_text(card.lineup))
        order = tk.StringVar(value=T.ORDERS[self.settings.get("card_order", "bow")])
        orow = ttk.Frame(f)
        orow.grid(row=6, column=1, columnspan=2, sticky="w")
        ttk.Label(orow, text="typed").pack(side="left")
        ttk.Combobox(orow, textvariable=order, values=list(T.ORDERS.values()), state="readonly",
                     width=28).pack(side="left", padx=(4, 10))
        ttk.Label(orow, text="shown").pack(side="left")
        seats = tk.StringVar(value=T.SEAT_ORDERS[card.seat_order])
        ttk.Combobox(orow, textvariable=seats, values=list(T.SEAT_ORDERS.values()), state="readonly",
                     width=46).pack(side="left", padx=(4, 0))
        ttk.Label(f, text="one name per line, or give the seat: Stroke: Sam", foreground="#5f6b7a").grid(
            row=7, column=1, columnspan=2, sticky="w")
        ttk.Label(f, text="Show for").grid(row=8, column=0, sticky="w", pady=(8, 0))
        rowf = ttk.Frame(f)
        rowf.grid(row=8, column=1, sticky="w", pady=(8, 0))
        secs = tk.StringVar(value="%g" % card.seconds)
        ttk.Spinbox(rowf, from_=2, to=60, increment=1, width=5, textvariable=secs).pack(side="left")
        ttk.Label(rowf, text="seconds").pack(side="left", padx=4)
        ttk.Label(f, text="Behind it").grid(row=9, column=0, sticky="w", pady=(8, 0))
        under = tk.StringVar(value=T.UNDER[card.under])
        ttk.Combobox(f, textvariable=under, values=list(T.UNDER.values()), state="readonly", width=34).grid(
            row=9, column=1, sticky="w", pady=(8, 0))
        ttk.Label(f, text="B-roll clip").grid(row=10, column=0, sticky="w", pady=(4, 0))
        broll = tk.StringVar(value=card.broll or "")
        ttk.Entry(f, textvariable=broll, width=40).grid(row=10, column=1, sticky="ew", pady=(4, 0))
        bf = ttk.Frame(f)
        bf.grid(row=10, column=2, sticky="w", pady=(4, 0), padx=(6, 0))

        def choose():
            p = filedialog.askopenfilename(parent=win, title="B-roll clip", filetypes=[
                ("Videos", "*.mp4 *.MP4 *.mov *.MOV"), ("All files", "*.*")])
            if p:
                broll.set(p)
                under.set(T.UNDER["broll"])
        ttk.Button(bf, text="Choose…", command=choose).pack(side="left")
        ttk.Button(bf, text="Clear", command=lambda: broll.set("")).pack(side="left", padx=(4, 0))
        ttk.Label(f, text="Playing: for a recording that starts before the piece.\n"
                          "Still: the first frame held, for one that starts mid-piece.\n"
                          "B-roll: another clip, sped up or slowed to fill the time.",
                  foreground="#5f6b7a", justify="left").grid(row=11, column=1, columnspan=2, sticky="w")

        def build():
            key = next(k for k, v in T.ORDERS.items() if v == order.get())
            try:
                n = float(secs.get())
            except ValueError:
                n = T.DEFAULT_SECONDS
            u = next(k for k, v in T.UNDER.items() if v == under.get())
            so = next(k for k, v in T.SEAT_ORDERS.items() if v == seats.get())
            return key, T.TitleCard(title=title.get().strip(), subtitle=sub.get().strip(),
                                    lineup=T.parse_lineup(names.get("1.0", "end"), key),
                                    seconds=min(max(n, 1.0), 60.0), broll=broll.get().strip() or None, under=u,
                                    seat_order=so)

        def apply(targets):
            key, c = build()
            if on.get() and c.empty:
                messagebox.showinfo(APP, "Give the race a name or a lineup, or untick the title card.", parent=win)
                return
            if on.get() and c.under == "broll" and not c.broll:
                messagebox.showinfo(APP, "Choose the b-roll clip, or pick something else to show behind the card.",
                                    parent=win)
                return
            if on.get() and c.under == "broll" and not os.path.exists(c.broll):
                messagebox.showwarning(APP, "Cannot find the b-roll clip:\n%s" % c.broll, parent=win)
                return
            for v in targets:
                v.card = c if on.get() else None
            if on.get():
                self.settings["card_last"] = c.to_dict()
                self.settings["card_order"] = key
                self._save()
            win.destroy()
            self._redraw()

        def show():
            _key, c = build()
            v = sel[0]
            self.stage.set("Making a preview of the title card…")

            def work():
                try:
                    img = T.card_frame(v.path, c)
                    self.q.put(("preview", (img, "title card", "The title card, %g s, over %s." % (
                        c.seconds, {"video": "the start of the race video", "still": "the video's first frame",
                                    "broll": "the b-roll"}[c.under]))))
                except Exception as e:
                    self.q.put(("error", "Preview failed: %s" % e))
            threading.Thread(target=work, daemon=True).start()

        def thumb():
            _key, c = build()
            if c.empty:
                messagebox.showinfo(APP, "Give the race a name or a lineup first.", parent=win)
                return
            v = sel[0]
            path = filedialog.asksaveasfilename(
                parent=win, title="Save the thumbnail", defaultextension=".jpg",
                initialdir=self.outdir.get() if os.path.isdir(self.outdir.get()) else None,
                initialfile="%s - thumbnail.jpg" % os.path.splitext(v.name)[0], filetypes=[("JPEG", "*.jpg")])
            if not path:
                return
            self.stage.set("Saving the thumbnail…")

            def work():
                try:
                    T.save_thumbnail(v.path, c, path)
                    self.q.put(("log", "Thumbnail saved: %s" % path))
                    self.q.put(("stage", "Thumbnail saved."))
                except Exception as e:
                    self.q.put(("error", "Could not save the thumbnail: %s" % e))
            threading.Thread(target=work, daemon=True).start()

        bar = ttk.Frame(f)
        bar.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        ttk.Button(bar, text="Preview", command=show).pack(side="left")
        ttk.Button(bar, text="Save thumbnail…", command=thumb).pack(side="left", padx=6)
        ttk.Button(bar, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(bar, text="Apply to all videos", command=lambda: apply(self.videos)).pack(side="right", padx=6)
        if len(sel) < len(self.videos):
            ttk.Button(bar, text="Apply to %s" % ("this video" if len(sel) == 1 else "these %d" % len(sel)),
                       command=lambda: apply(sel)).pack(side="right")

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
        lost = [v for v in self.videos if v.card and v.card.under == "broll"
                and not (v.card.broll and os.path.exists(v.card.broll))]
        if lost:
            messagebox.showwarning(APP, "The b-roll clip for these videos cannot be found:\n  %s\n\n"
                                   "Open Title card… to choose it again, or clear it to show the card over the "
                                   "race video." % "\n  ".join(v.name for v in lost))
            return
        outdir = self.outdir.get()
        os.makedirs(outdir, exist_ok=True)
        self._save()
        jobs = [(v, v.session) for v in self.videos]
        clock = [list(h) for h in self.settings.get("clock", [])]
        if not self.calibrated.get() or (clock and abs(clock[-1][1] - self._camera_fast()) > 0.05):
            clock = []                     # set by hand, or a new camera: the box is the only measurement
        opts = dict(layout=self.layout.get(), fields=self._fields(), calls=bool(self.calls.get()) and self.calls_ok,
                    fast=self._camera_fast(), zone=self._zone_hours(), outdir=outdir, clock=clock,
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
                    fast, days = clock_for(o.get("clock"), v.cam_start_utc, o["fast"])
                    guess = (s0 - v.true_start_utc(fast)).total_seconds()
                    prior = o.get("prior") and o["prior"] + CLOCK_DRIFT * days
                    off, info = C.best_sync(s, mt, signals, guess, calls, prior_sd=prior)
                    if off is None:
                        self.q.put(("log", "%s: could not sync (%s). Check the pairing or the camera clock "
                                           "setting; skipped." % (v.name, info.get("reason"))))
                        self.q.put(("status", (v, "not synced")))
                        continue
                    msg = "%s: synced, CoxBox start at video %s (stroke rate agrees to %.1f spm" % (
                        v.name, mmss(off), info["rate_err"])
                    if info.get("calls"):
                        msg += "; your called splits to %.1f s" % info["calls"][0]
                    self.q.put(("log", msg + ")."))
                    # the camera clock, measured: keep the setting current for the next videos
                    measured = (v.cam_start_utc - (s0 - dt.timedelta(seconds=off))).total_seconds() / 60.0
                    if abs(measured - fast) > 1.0:
                        self.q.put(("log", "  note: the camera clock measured %.1f min fast, not the %.1f expected; "
                                           "updated." % (measured, fast)))
                    if v.cam_start_utc is not None:
                        o.setdefault("clock", []).append([v.cam_start_utc.isoformat(timespec="seconds"), measured])
                    self.q.put(("camera_fast", (measured, o.get("clock", []))))
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
                O.render_video(v.path, out, o["layout"], at, fields=o["fields"], card=v.card,
                               progress=self._progress(i, n, "%s · writing video" % v.name))
                done.append(out)
                self.q.put(("status", (v, "done")))
                self.q.put(("log", "%s: saved %s" % (v.name, out)))
                if v.card is not None and not v.card.empty:
                    try:
                        jpg = T.save_thumbnail(v.path, v.card, os.path.splitext(out)[0] + " - thumbnail.jpg")
                        self.q.put(("log", "%s: thumbnail (title card, 1280 x 720) saved %s" % (v.name, jpg)))
                    except Exception as e:
                        self.q.put(("log", "%s: no thumbnail: %s" % (v.name, e)))
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
                    measured, clock = data
                    self.settings["clock"] = clock[-50:]
                    self.fast.set("%.2f" % measured)
                    self.calibrated.set(True)
                    self._save()
                elif kind == "stage":
                    self.stage.set(data)
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
            [--title "Head of the Lake" --subtitle "..." --lineup "Ana;Bea;..." --card-seconds 6 --broll B.MP4]
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
    ap.add_argument("--camera-fast", type=float, default=None,
                    help="minutes the camera clock runs fast (default: the measured history)")
    ap.add_argument("--zone", type=float, default=None, help="CoxBox UTC offset, hours (default: this computer's)")
    ap.add_argument("--wide", action="store_true", help="the camera clock is not calibrated: search widely")
    ap.add_argument("--title", default="", help="title card: the race")
    ap.add_argument("--subtitle", default="", help="title card: a second line")
    ap.add_argument("--lineup", default="", help="title card: names separated by ';' (or a text file, one per line)")
    ap.add_argument("--lineup-order", default="bow", choices=list(T.ORDERS))
    ap.add_argument("--card-seconds", type=float, default=T.DEFAULT_SECONDS)
    ap.add_argument("--broll", default=None, help="title card: a clip retimed to show under it")
    ap.add_argument("--card-under", default=None, choices=list(T.UNDER),
                    help="title card: what it is shown over (default: the b-roll if given, else the video)")
    ap.add_argument("--cards", default=None,
                    help="a JSON file of title cards per video: {video file name: {title, subtitle, lineup, "
                         "order, seconds, under, broll}}; lineup as text or [[seat, name], ...]")
    a = ap.parse_args(argv)
    clock = [] if (a.camera_fast is not None or a.wide) else [list(h) for h in DEFAULTS["clock"]]
    if a.camera_fast is None:
        a.camera_fast = DEFAULTS["camera_fast"]
    import transcribe_calls as TC
    zone = a.zone
    if zone is None:
        off = -time.altzone if time.daylight and time.localtime().tm_isdst else -time.timezone
        zone = round(off / 3600)
    videos = [Video(p) for p in a.videos]
    names = open(a.lineup, encoding="utf-8").read() if a.lineup and os.path.isfile(a.lineup) else a.lineup.replace(";", "\n")
    card = T.TitleCard(title=a.title, subtitle=a.subtitle, lineup=T.parse_lineup(names, a.lineup_order),
                       seconds=a.card_seconds, broll=a.broll, under=a.card_under or ("broll" if a.broll else "video"))
    per = json.load(open(a.cards, encoding="utf-8")) if a.cards else {}
    for v in videos:
        v.card = None if card.empty else card
        if v.name in per:
            d = dict(per[v.name])
            if isinstance(d.get("lineup"), str):
                d["lineup"] = T.parse_lineup(d["lineup"], d.pop("order", "bow"))
            d.pop("order", None)
            v.card = T.TitleCard.from_dict(d)
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
                fast=a.camera_fast, zone=zone, outdir=a.out, clock=clock, prior=None if a.wide else 20.0)

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
    broll = os.path.join(outdir, "synthetic_broll.mp4")

    def card(with_broll):
        if with_broll is True:
            subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                            "smptebars=size=640x480:rate=25:d=5", "-c:v", "libx264", "-pix_fmt", "yuv420p", broll],
                           check=True, **O.NOWIN)
        under = "still" if with_broll == "still" else ("broll" if with_broll else "video")
        c = T.TitleCard(title="Self-test", subtitle="title card", lineup=T.parse_lineup("A\nB\nC\nD\nE"),
                        seconds=3.0, broll=broll if under == "broll" else None, under=under)
        out = os.path.join(outdir, "synthetic_card_%s.mp4" % under)
        O.render_video(clip, out, "A", O.session_moments(s, 1.0), card=c)
        want = 8.0 + c.lead
        got = O.probe(out)[2]
        if abs(got - want) > 0.2:
            raise RuntimeError("%.2f s long, expected %.2f s" % (got, want))
        return "%.2f s" % got
    check("title card over the video", lambda: card(False))
    check("title card over a retimed b-roll", lambda: card(True))
    check("title card over a held first frame", lambda: card("still"))

    def thumb():
        c = T.TitleCard(title="Self-test", lineup=T.parse_lineup("A;B;C;D;E".replace(";", chr(10))), under="still")
        p = T.save_thumbnail(clip, c, os.path.join(outdir, "thumbnail.jpg"))
        from PIL import Image
        im = Image.open(p)
        if im.size != T.THUMB_SIZE or os.path.getsize(p) > 2 * 1024 * 1024:
            raise RuntimeError("%s, %d bytes" % (im.size, os.path.getsize(p)))
        return "%dx%d, %.0f kB" % (im.size + (os.path.getsize(p) / 1024,))
    check("thumbnail", thumb)
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
