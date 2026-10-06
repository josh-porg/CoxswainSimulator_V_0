"""The CoxBox video overlay tool (tools/overlay): parsing, sync and drawing, on synthetic data only.

The crew's own footage and CoxBox files never enter the repository; everything here is made up.
"""
import datetime as dt
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
C = pytest.importorskip("coxbox_data")


def words(text, start=10.0, step=0.4):
    """A transcript segment with one word every ``step`` s."""
    toks = text.split()
    return [dict(words=[dict(word=" " + w, start=start + i * step, end=start + i * step + 0.3)
                        for i, w in enumerate(toks)])]


@pytest.mark.parametrize("said, split", [
    ("you're at a 221 bring it down", 141), ("we're at 2:05 now", 125), ("split 1.58 go", 118),
    ("you're at a two twenty one", 141), ("two oh two on the split", 122), ("one fifty eight yes", 118),
    ("two flat and holding", 120), ("we're at a two sixteen", 136),
])
def test_called_splits_in_digits_and_words(said, split):
    got = C.called_splits(words(said))
    assert [v for _t, v in got] == [split]


def test_counting_is_not_a_split():
    assert C.called_splits(words("one, two, three, four, five, six")) == []
    assert C.called_splits(words("in two we take ten, one, two")) == []


def test_called_split_time_is_the_word_start():
    got = C.called_splits(words("ok you're at a two oh four", start=100.0, step=0.5))
    assert got == [(100.0 + 4 * 0.5, 124)]


def synthetic_csv(tmp_path, n=240, rate=30.0):
    lines = ["Session Information:,,,,Device Information:", "", "Name:,JustGo,,,Name:,CBGPS",
             "Start Time:,10/04/2026 08:03:44,,,Model:,CBGPS", "", "Per-Stroke Data:", "",
             "Interval,Distance (GPS),Distance (IMP),Elapsed Time,Split (GPS),Speed (GPS),Split (IMP),Speed (IMP),"
             "Stroke Rate,Total Strokes,Distance/Stroke (GPS),Distance/Stroke (IMP),Heart Rate,Power,Catch,Slip,"
             "Finish,Wash,Force Avg,Work,Force Max,Max Force Angle,GPS Lat.,GPS Lon.",
             "(Interval),(Meters),(Meters),(HH:MM:SS.tenths),(/500),(M/S),(/500),(M/S),(SPM),(Strokes),(Meters),"
             "(Meters),(BPM),(Watts),(Degrees),(Degrees),(Degrees),(Degrees),(Newtons),(Joules),(Newtons),"
             "(Degrees),(Degrees),(Degrees)"]
    t = 0.0
    for k in range(n):
        t += 60.0 / rate
        d = 8.2 * (k + 1)
        lines.append("1,%.1f,0.0,00:%02d:%04.1f,00:02:02.0,4.10,00:00:00.0,0.00,%.1f,%d,8.2,0.0,---,---,---,---,"
                     "---,---,---,---,---,---,%.7f,%.7f" % (d, int(t // 60), t % 60, rate, k + 1,
                                                          47.6 + k * 1e-5, -122.3 + k * 1e-5))
    p = tmp_path / "Sharing file  CoxBox 0 20261004 0803AM.csv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_load_csv(tmp_path):
    s = C.load_csv(synthetic_csv(tmp_path))
    assert s.start == dt.datetime(2026, 10, 4, 8, 3, 44)
    assert len(s.t) == 240 and s.t[0] == pytest.approx(2.0)
    assert s.split[0] == pytest.approx(122.0) and s.rate[0] == 30.0
    xy = s.track_xy()
    assert xy[0] == pytest.approx([0.0, 0.0]) and np.all(np.isfinite(xy))


def test_sync_recovers_a_known_offset():
    """A stroke train with a rate profile (start, settle, push, sprint), seen as a head-motion pulse
    once a stroke with noise, ``true`` s into the video: the motion sync finds it, with and without
    called splits, starting 25 s off."""
    rng = np.random.default_rng(1)
    rates = np.concatenate([np.full(15, 36.0), np.full(160, 30.0), np.full(40, 32.0), np.full(30, 35.0)])
    t = np.cumsum(60.0 / rates)
    split = np.where(rates > 33, 115.0, np.where(rates > 31, 119.0, 122.0))
    s = C.Session(start=dt.datetime(2026, 10, 4, 8, 0, 0), t=t, distance=t * 4.1, split=split,
                  speed=np.full(t.size, 4.1), rate=rates, strokes=np.arange(1.0, t.size + 1), per_stroke=np.full(t.size, 8.2),
                  lat=np.full(t.size, 47.6), lon=np.full(t.size, -122.3))
    true = 137.3
    fps = 15.0
    mt = np.arange(0, true + t[-1] + 60, 1 / fps)
    sig = 0.3 * rng.standard_normal(mt.size)
    for st in t + true:
        sig += np.exp(-0.5 * ((mt - st - 0.3) / 0.15) ** 2)
    calls = [(true + t[i] + 4.0, int(split[i])) for i in range(20, t.size, 12)]
    for c in (None, calls):
        off, info = C.best_sync(s, mt, dict(dy=sig), true + 25.0, c, prior_sd=20.0)
        assert off == pytest.approx(true, abs=1.0), info


O = pytest.importorskip("coxbox_overlay")


def moment(estimated=False):
    hist = [(float(k), 30.0 + (k % 3), 122.0 - (k % 4)) for k in range(31)]
    track = np.column_stack([np.linspace(0, 3000, 50), np.sin(np.linspace(0, 3, 50)) * 400])
    if estimated:
        return O.Moment(elapsed=None, distance=None, rate=29.0, split=None, per_stroke=None, rate_estimated=True,
                        history=[(h[0], h[1], None) for h in hist])
    return O.Moment(elapsed=30.0, distance=1234.0, rate=31.0, split=121.4, per_stroke=8.4, history=hist,
                    track=track, position=(1500.0, 200.0))


@pytest.mark.parametrize("layout", ["A", "B", "D"])
@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (1440, 1080)])
@pytest.mark.parametrize("fields", [None, ["rate", "split"], ["trace", "map"], ["time"]])
def test_layers_draw_at_any_size_and_field_choice(layout, size, fields):
    for m in (moment(), moment(estimated=True), None):
        img = O.layer(layout, size, m, fields)
        assert img.size == size and img.mode == "RGBA"


def test_layout_a_leaves_the_picture_below_the_strip_untouched():
    img = np.asarray(O.layer("A", (1920, 1080), moment()))
    assert img[400:, :, 3].max() == 0          # nothing drawn over the blades and the crew
    assert img[:70, :, 3].min() > 0           # the strip itself is there


def test_layout_d_keeps_the_picture_window_clear():
    img = np.asarray(O.layer("D", (1920, 1080), moment()))
    vw, vh, vx, vy = O.d_geometry((1920, 1080))
    assert img[vy + 5:vy + vh - 5, vx + 5:vx + vw - 5, 3].max() == 0
    assert img[5:100, 5:100, 3].min() == 255
