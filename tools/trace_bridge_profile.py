r"""Trace a bridge's truss profile off its HAER elevation photograph.

    python tools/trace_bridge_profile.py --check out/bridge_trace

The renderer drew the Aurora Bridge as a Warren truss -- parallel chords
with a zig-zag web -- because that is the generic thing a "truss" tag
buys you.  It is not what the bridge looks like.  The Aurora Bridge is a
**steel cantilever**: a level deck over a bottom chord that sweeps down
to its greatest depth over each main pier and rises to meet the deck
again at midspan and at the ends.  That silhouette is the bridge.

So the silhouette is measured rather than invented, from the Historic
American Engineering Record: survey **HAER WA-107**, photograph 1,
"General view of Aurora Bridge looking west up the Washington Ship
Canal", which is very close to a true side elevation.  HAER is a US
Government work and public domain.

How the trace works
-------------------
The discriminator is **brightness against a local background**, not
darkness against the sky, and that took two attempts to get right.

The first version looked for the truss as a *dark* textured band under
the deck, on the reasoning that a bridge is dark against a bright sky.
That is true of the top of the bridge and false of the bottom: the lower
chord hangs against a wooded hillside, and against a hillside the sunlit
steel is the **bright** thing.  The dark-band trace produced noise --
604 columns of it, spiking to the bottom of the crop.

So: the deck is the first sharp darkening below the smooth sky, which is
reliable because the deck really is dark against sky.  The bottom chord
is the lowest pixel that stands out **above** the local median of
everything under the deck, which is the sunlit lattice.

What is measured and what is assumed
------------------------------------
Measured here: the *shape* -- where the truss is deep and where it is
shallow, as a fraction of the span and of the maximum depth.

Not measured here: the size.  The photograph has perspective in it and
no scale bar, so absolute dimensions come from the National Bridge
Inventory (``tools/fetch_nbi_bridges.py``) and the traced profile is
normalised to it.  A photograph is a good witness to form and a poor one
to length.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: HAER photographs to trace, by the crossing name NBI uses.
#: ``(url, crop as left, top, right, bottom)`` in the 1024 px derivative.
SOURCES = {
    "LAKE UNION": dict(
        survey="HAER WA-107",
        caption="1. General view of Aurora Bridge looking west up the "
                "Washington Ship Canal",
        url="https://tile.loc.gov/storage-services/service/pnp/habshaer/"
            "wa/wa0400/wa0440/photos/370416pv.jpg",
        crop=(196, 128, 800, 200),
    ),
}

#: How far above the local median a pixel must sit to be called steel.
#: In units of the local standard deviation.
BRIGHTNESS = 0.8

#: Smallest darkening, in grey levels, that counts as the deck edge.
DECK_STEP = 25.0


def column_bands(column):
    """``(deck row, chord row)`` for one column, or ``(None, None)``."""
    values = np.asarray(column, dtype=float)
    if len(values) < 12:
        return None, None

    # The deck: the first sharp darkening below the smooth sky.
    steps = np.nonzero(np.diff(values) < -DECK_STEP)[0]
    if not len(steps):
        return None, None
    deck = int(steps[0]) + 1
    below = values[deck:]
    if len(below) < 6:
        return None, None

    # The bottom chord: the lowest sunlit member under the deck.
    threshold = np.median(below) + BRIGHTNESS * below.std()
    bright = np.nonzero(below > threshold)[0]
    if not len(bright):
        return deck, None
    return deck, deck + int(bright.max())


def trace(image, crop):
    """``(station, deck, chord)`` in pixels across the cropped bridge."""
    left, top, right, bottom = crop
    band = np.asarray(image, dtype=float)[top:bottom, left:right]
    decks, chords = [], []
    for index in range(band.shape[1]):
        a, b = column_bands(band[:, index])
        decks.append(np.nan if a is None else a)
        chords.append(np.nan if b is None else b)
    return (np.arange(band.shape[1], dtype=float),
            np.array(decks, dtype=float), np.array(chords, dtype=float))


def smooth_profile(station, depth, window: int = 21):
    """Median filter, then normalise to (0-1, 0-1)."""
    good = np.isfinite(depth)
    if good.sum() < 20:
        raise SystemExit("too little of the bridge was traced")
    x = station[good]
    y = depth[good]
    half = window // 2
    filtered = np.array([np.median(y[max(0, i - half):i + half + 1])
                         for i in range(len(y))])
    # The deck meets the chord at the ends, so the baseline is the
    # minimum, not zero: perspective tilts the whole thing slightly.
    filtered = filtered - filtered.min()
    span = x.max() - x.min()
    return (x - x.min()) / span, filtered / max(filtered.max(), 1e-9)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", default="out/bridge_trace",
                        help="write a diagnostic plot here")
    parser.add_argument("--out", default="bridge_profiles.json")
    args = parser.parse_args(argv)

    import urllib.request

    from PIL import Image

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = os.path.join(root, "data", "raw", "haer")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(args.check, exist_ok=True)

    profiles = {}
    for crossing, source in SOURCES.items():
        name = source["url"].rsplit("/", 1)[-1]
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            print("  fetching %s ..." % source["survey"])
            request = urllib.request.Request(
                source["url"],
                headers={"User-Agent": "CoxswainSimulator/0.1 (research)"})
            with urllib.request.urlopen(request, timeout=180) as response:
                open(path, "wb").write(response.read())
        image = Image.open(path).convert("L")
        station, deck, chord = trace(image, source["crop"])
        depth = chord - deck
        fraction, shape = smooth_profile(station, depth)
        traced = int(np.isfinite(depth).sum())
        print("%s: traced %d of %d columns; deepest at %.0f%% of the span"
              % (crossing, traced, len(station),
                 100 * fraction[int(np.argmax(shape))]))

        # Resample to a compact table.
        grid = np.linspace(0.0, 1.0, 41)
        profiles[crossing] = {
            "survey": source["survey"],
            "caption": source["caption"],
            "url": source["url"],
            "station": grid.tolist(),
            "depth": np.interp(grid, fraction, shape).tolist(),
        }

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 1, figsize=(11, 6),
                                    gridspec_kw={"height_ratios": [2, 1]})
        left, top, right, bottom = source["crop"]
        axes[0].imshow(np.asarray(image)[top:bottom, left:right],
                       cmap="gray", aspect="auto")
        axes[0].plot(station, deck, color="#2f7fb5", linewidth=1.2,
                     label="deck, traced")
        axes[0].plot(station, chord, color="#ff9248", linewidth=1.2,
                     label="bottom chord, traced")
        axes[0].legend(fontsize=8)
        axes[0].set_title("%s -- %s" % (crossing, source["survey"]),
                          fontsize=10)
        axes[1].plot(grid, np.interp(grid, fraction, shape), "o-",
                     color="#1f7a4d")
        axes[1].set_xlabel("fraction of the traced length")
        axes[1].set_ylabel("depth / max")
        axes[1].grid(True, linewidth=0.4)
        figure.tight_layout()
        target = os.path.join(args.check, "%s.png"
                              % crossing.lower().replace(" ", "_"))
        figure.savefig(target, dpi=130)
        plt.close(figure)
        print("  wrote %s" % target)

    out = os.path.join(root, "coxswain", "data", args.out)
    with open(out, "w") as handle:
        json.dump(profiles, handle, indent=2)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
