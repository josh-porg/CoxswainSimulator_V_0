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

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "local", "overlay", "motion")


def hanning(W, H):
    """OpenCV's ``createHanningWindow``: the square root of the outer product of two Hann windows."""
    wx = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(W) / (W - 1))
    wy = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(H) / (H - 1))
    return np.sqrt(np.outer(wy, wx))


def phase_correlate(a, b, win):
    """The shift ``(dx, dy)`` of ``b`` against ``a``, as OpenCV's ``phaseCorrelate`` finds it: the
    normalised cross-power spectrum, its inverse, and the weighted centroid of the 5 x 5 box around
    the peak. On race footage it agrees with OpenCV to 0.04 px (median) frame by frame; done in numpy
    so the app does not carry OpenCV, whose loader breaks inside a frozen macOS bundle."""
    Fa, Fb = np.fft.rfft2(a * win), np.fft.rfft2(b * win)
    P = Fa * np.conj(Fb)
    P /= np.abs(P) + np.finfo(np.float64).eps
    C = np.fft.fftshift(np.fft.irfft2(P, s=a.shape))
    py, px = np.unravel_index(np.argmax(C), C.shape)
    y0, y1 = max(py - 2, 0), min(py + 2, C.shape[0] - 1)
    x0, x1 = max(px - 2, 0), min(px + 2, C.shape[1] - 1)
    box = C[y0:y1 + 1, x0:x1 + 1]
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    s = box.sum()
    return C.shape[1] / 2 - (xs * box).sum() / s, C.shape[0] / 2 - (ys * box).sum() / s


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
    win = hanning(W, H)
    prev, dy, dx, diff = None, [], [], []
    n = W * H
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W).astype(np.float64)
        if prev is not None:
            sx, sy = phase_correlate(prev, fr, win)
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
