r"""Build the SRA CoxBox Overlay app for the platform this runs on (PyInstaller cannot cross-build).

    python tools/overlay/build_app.py [--version overlay-v1.0] [--no-model]

Output: ``dist/overlay/CoxBoxOverlay/`` (Windows, Linux) or ``dist/overlay/CoxBoxOverlay.app`` (macOS).

The bundle carries everything the app needs with no internet: the ffmpeg build imageio-ffmpeg ships
for this platform, the Barlow fonts (SIL OFL), the speech engine's native libraries, and the
``small.en`` speech model, fetched here at build time from Hugging Face (Systran's CTranslate2
conversion of OpenAI's Whisper, both MIT). ``--no-model`` builds without it: the app then syncs by
the stroke rhythm and the camera clock only.

Run it in an environment with ``tools/overlay/requirements-app.txt`` and PyInstaller installed.
Release builds come from .github/workflows/release-overlay.yml on an ``overlay-v*`` tag.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NAME = "CoxBoxOverlay"
MODEL = "small.en"
MODEL_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]


def fetch_model(dest):
    if os.path.exists(os.path.join(dest, "model.bin")):
        return dest
    from huggingface_hub import snapshot_download
    snapshot_download("Systran/faster-whisper-%s" % MODEL, local_dir=dest, allow_patterns=MODEL_FILES)
    with open(os.path.join(dest, "NOTICE.txt"), "w", encoding="utf-8") as f:
        f.write("faster-whisper-%s: Systran's CTranslate2 conversion of OpenAI Whisper %s.\n"
                "Whisper model weights: MIT License, Copyright (c) 2022 OpenAI.\n"
                "Conversion: https://huggingface.co/Systran/faster-whisper-%s (MIT).\n" % (MODEL, MODEL, MODEL))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("OVERLAY_VERSION", "dev"))
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()
    build = os.path.join(ROOT, "build", "overlay")
    dist = os.path.join(ROOT, "dist", "overlay")
    os.makedirs(build, exist_ok=True)
    shutil.rmtree(dist, ignore_errors=True)
    version_file = os.path.join(build, "VERSION")
    with open(version_file, "w", encoding="utf-8") as f:
        f.write(a.version + "\n")
    sep = os.pathsep
    args = [os.path.join(HERE, "app.py"), "--name", NAME, "--noconfirm", "--clean", "--onedir", "--windowed",
            "--distpath", dist, "--workpath", os.path.join(build, "work"), "--specpath", build,
            "--paths", HERE,
            "--add-data", "%s%sfonts" % (os.path.join(HERE, "fonts"), sep),
            "--add-data", "%s%s." % (version_file, sep),
            "--collect-all", "faster_whisper", "--collect-all", "ctranslate2", "--collect-all", "onnxruntime",
            "--collect-all", "av", "--collect-data", "imageio_ffmpeg",
            "--hidden-import", "PIL.ImageTk"]
    if not a.no_model:
        model = fetch_model(os.path.join(build, "models", "faster-whisper-%s" % MODEL))
        args += ["--add-data", "%s%smodels/faster-whisper-%s" % (model, sep, MODEL)]
    if sys.platform == "darwin":
        args += ["--osx-bundle-identifier", "org.sammamishrowing.coxboxoverlay"]
    import PyInstaller.__main__
    PyInstaller.__main__.run(args)
    print("built", dist)


if __name__ == "__main__":
    main()
