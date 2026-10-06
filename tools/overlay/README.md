# SRA CoxBox Overlay

Puts CoxBox numbers over cox-camera video, in sync, offline. A side tool; nothing here touches the
trainer.

| file | what |
|---|---|
| `app.py` | the program (Tkinter). `--batch` runs headless, `--selftest DIR` checks an install. |
| `coxbox_data.py` | NK CSV reader, pairing, sync (stroke rhythm + called splits + clock prior) |
| `camera_motion.py` | head-camera motion signal (phase correlation, 15 fps) |
| `transcribe_calls.py` | the cox's calls, by a local faster-whisper model |
| `coxbox_overlay.py` | layouts A, B, D and the ffmpeg render |
| `make_overlay_videos.py` | the first batch, kept for reference |
| `build_app.py` | PyInstaller build, with the speech model bundled |
| `fonts/` | Barlow (SIL OFL), used where Bahnschrift is missing |

Run from source (Windows): `CoxBox Overlay.bat`, or

    .venv-tools/Scripts/python tools/overlay/app.py

with `requirements-app.txt` installed in `.venv-tools` (never the game's `.venv`).

Release: push a tag `overlay-vX.Y`; `.github/workflows/release-overlay.yml` builds all three
platforms, self-tests each, and publishes a release that is not marked latest.

The crew's footage and CoxBox files stay in `data/local/` and never enter the repository.
