r"""The head camera's frame-to-frame motion: a stroke-rhythm signal from the picture itself.

    python tools/overlay/camera_motion.py "C:\...\DJI_..._D.MP4" [--fps 15]

The camera rides on the coxswain's head, and the head pitches with the boat's surge once a stroke;
in an eight the stroke seat also swings through the frame. Per pair of consecutive frames
(downscaled to 160 x 90, grey): the global shift by phase correlation (dy: pitch, dx: yaw) and the
mean absolute difference. Tested on Sunday's races (2026-10-04): in the eight the pitch signal gives
the stroke rate at autocorrelation 0.6-0.8 through most windows; in a bow-loaded four, 0.1-0.4.

Output: ``data/local/overlay/motion/<video>.npz`` with t (s), dy, dx, diff.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "local", "overlay", "motion")


def motion(video, fps=15, W=160, H=90, t0=None, dur=None, progress=None, total=None):
    """``(t, dy, dx, diff)``; ``progress(fraction)`` against ``total`` seconds when given."""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if t0 is not None:
        cmd += ["-ss", str(t0)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-i", video, "-vf", "fps=%d,scale=%d:%d" % (fps, W, H), "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    prev, dy, dx, diff = None, [], [], []
    n = W * H
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W).astype(np.float32)
        if prev is not None:
            (sx, sy), _ = cv2.phaseCorrelate(prev, fr, win)
            dx.append(sx)
            dy.append(sy)
            diff.append(float(np.mean(np.abs(fr - prev))))
            if progress is not None and total and len(dy) % (fps * 10) == 0:
                progress(min(len(dy) / fps / total, 1.0))
        prev = fr
    proc.wait()
    t = (np.arange(len(dy)) + 1.0) / fps + (t0 or 0.0)
    return t, np.array(dy), np.array(dx), np.array(diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t, dy, dx, diff = motion(a.video, a.fps)
    out = os.path.join(OUT, os.path.splitext(os.path.basename(a.video))[0] + ".npz")
    np.savez(out, t=t, dy=dy, dx=dx, diff=diff, fps=a.fps)
    print("wrote", out, len(t), "samples")


if __name__ == "__main__":
    main()
