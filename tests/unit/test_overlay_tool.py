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
    spiky = sig.copy()                  # near-black frames: shifts of hundreds of px, a few hundred of them
    bad = rng.choice(mt.size, 300, replace=False)
    spiky[bad] = rng.choice([-1.0, 1.0], bad.size) * rng.uniform(50, 900, bad.size)
    for c in (None, calls):
        for x in (sig, spiky):
            off, info = C.best_sync(s, mt, dict(dy=x), true + 25.0, c, prior_sd=20.0)
            assert off == pytest.approx(true, abs=1.0), info


def test_only_impossible_shifts_are_dropped():
    sig = dict(dy=np.array([0.5, -3.0, 44.0, 46.0, -900.0]), dx=np.array([79.0, 81.0, 0.0, 0.0, 0.0]),
               diff=np.array([100.0, 0, 0, 0, 0]))
    out = C.drop_impossible(sig)
    assert list(out["dy"]) == [0.5, -3.0, 44.0, 0.0, 0.0] and list(out["dx"]) == [79.0, 0.0, 0.0, 0.0, 0.0]
    assert out["diff"][0] == 100.0


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


def test_phase_correlation_recovers_a_known_shift():
    """The head-motion reader (numpy, in place of OpenCV) finds whole-pixel shifts of a textured
    frame, with OpenCV's sign: the shift of the second frame against the first. (A sub-pixel shift
    is only seen in direction: the 5 x 5 centroid OpenCV uses, copied here, pulls it toward zero.
    Against OpenCV 5.0 itself: these values to 0.001 px.)"""
    M = pytest.importorskip("camera_motion")
    rng = np.random.default_rng(3)
    f2 = np.add.outer(np.fft.fftfreq(180) ** 2, np.fft.rfftfreq(320) ** 2)
    big = np.fft.irfft2(np.fft.rfft2(rng.random((180, 320))) * np.exp(-2 * np.pi ** 2 * f2), s=(180, 320))
    win = M.hanning(160, 90)
    a = big[40:130, 60:220]
    for sx, sy in ((3, -2), (-1, 4), (0, 0)):
        dx, dy = M.phase_correlate(a, big[40 - sy:130 - sy, 60 - sx:220 - sx], win)
        assert (dx, dy) == (pytest.approx(sx, abs=0.06), pytest.approx(sy, abs=0.06))
    half = np.real(np.fft.ifft2(np.fft.fft2(big) * np.exp(-1j * np.pi * np.fft.fftfreq(320)[None, :])))
    dx, dy = M.phase_correlate(a, half[40:130, 60:220], win)
    assert 0.05 < dx <= 0.5 and abs(dy) < 0.05


T = pytest.importorskip("title_card")


@pytest.mark.parametrize("text, order, want", [
    ("A\nB\nC\nD\nE\nF\nG\nH\nI", "bow", ["Bow", "2", "3", "4", "5", "6", "7", "Stroke", "Cox"]),
    ("I\nH\nG\nF\nE", "stroke", ["Cox", "Stroke", "3", "2", "Bow"]),
    ("A\nB\nC\nD", "bow", ["Bow", "2", "3", "Stroke"]),
    ("A\nB", "bow", ["Bow", "Stroke"]),
    ("Solo", "bow", [""]),
    ("Stroke: Sam\n3 - Alex\n2. Jo\nBow: Kim\nCox: Pat", "bow", ["Stroke", "3", "2", "Bow", "Cox"]),
    ("Bow-Smith\nJones\nLee\nPark", "bow", ["Bow", "2", "3", "Stroke"]),
])
def test_lineup_seats(text, order, want):
    got = T.parse_lineup(text, order)
    assert [s for s, _n in got] == want
    assert got[0][1] == text.splitlines()[0].split(":")[-1].split(" - ")[-1].split(". ")[-1].strip()


def test_lineup_round_trips_through_its_text():
    lineup = T.parse_lineup("A\nB\nC\nD\nE")
    assert T.parse_lineup(T.lineup_text(lineup)) == lineup


def test_card_fades_in_and_out_within_its_time():
    assert T.fade(-0.1, 6) == 0 and T.fade(6.0, 6) == 0
    assert T.fade(0.0, 6) == 0 and T.fade(3.0, 6) == 1.0
    assert 0 < T.fade(0.2, 6) < 1 and 0 < T.fade(5.8, 6) < 1


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (1440, 1080)])
@pytest.mark.parametrize("card", [
    dict(title="Head of the Lake", subtitle="Youth Eight", lineup="A\nB\nC\nD\nE\nF\nG\nH\nI"),
    dict(title="A very long regatta name that will not fit on a single line of the title card at all"),
    dict(lineup="Ana\nBea"), dict(title="Only a title")])
def test_card_draws_at_any_size(size, card):
    c = T.TitleCard(title=card.get("title", ""), subtitle=card.get("subtitle", ""),
                    lineup=T.parse_lineup(card.get("lineup", "")))
    img = T.card_layer(size, c)
    assert img.size == size and img.mode == "RGBA"
    assert np.asarray(img)[..., 3].min() > 0           # the scrim covers the frame


def test_card_lead_and_old_settings():
    assert T.TitleCard(under="video").lead == 0.0
    assert T.TitleCard(seconds=5, under="still").lead == 5 and T.TitleCard(seconds=5, under="broll").lead == 5
    assert T.TitleCard.from_dict(dict(title="x", broll="b.mp4")).under == "broll"    # saved by 1.1 drafts
    assert T.TitleCard.from_dict(dict(title="x")).under == "video"
