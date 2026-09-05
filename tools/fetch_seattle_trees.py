r"""Seattle's trees, by species and measured height.

    python tools/fetch_seattle_trees.py

Why this replaces the OpenStreetMap trees
-----------------------------------------
The scene drew every tree as a green sphere of the same size, from 9,113
OpenStreetMap points that carry no species and almost no height.  On this
shore that is wrong in a way a coxswain would notice immediately: the
banks of Lake Union are Douglas fir, western red cedar and Sitka spruce
among bigleaf maple and London plane, and a conifer is a **cone**.  A row
of spheres is a different place.

Two city datasets, and they complement each other
-------------------------------------------------
**SPR Tree View** -- Parks Department, 3,292 trees in this box, with
species *and* ``HEIGHT_WL``, a measured height in feet, populated on 89%
of them.  Small, and measured.

**SDOT Trees** -- Transportation, 51,365 street trees in the same box,
with ``SCIENTIFIC_NAME`` but **no height at all**.  Large, and the ones
actually lining the shore.

So the heights come from Parks and are carried to Transportation *by
species*: the median measured height of every Western Red Cedar in the
park data becomes the height of a Western Red Cedar on a street.  That is
a real inference from real measurements rather than a table copied out of
a field guide, and the per-species sample size is stored alongside so
anything resting on it can be re-run against the well-sampled species
only.

Form is genus, and genus is in the name
---------------------------------------
Whether a tree is a cone or a ball is decided by its genus, which both
datasets give.  No cleverness: a lookup of the conifer genera, and
everything else is broadleaf.  ``form`` is stored, not a shape, because
what to draw is the renderer's business.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.terrain import SEATTLE_DEM_BOUNDS       # noqa: E402

ROOT = "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services"
SPR = ROOT + "/SPR_Tree_View/FeatureServer/0/query"
SDOT = ROOT + "/SDOT_Trees_CDL/FeatureServer/0/query"
AGENT = "CoxswainSimulator/0.1 (rowing research; Seattle open data)"
PAGE = 2000
FOOT = 0.3048

#: Growth forms, in the order the stored code indexes them.
FORMS = ("broadleaf", "conifer", "columnar", "palm")

#: Genera that are cones.  Everything not here is drawn as a broadleaf,
#: which is the safe default: Seattle's street stock is mostly maple,
#: plane, cherry and oak, and a conifer drawn as a ball reads as a tree
#: while a maple drawn as a cone reads as a mistake.
CONIFER = {
    "abies", "picea", "pinus", "pseudotsuga", "thuja", "tsuga",
    "chamaecyparis", "cupressus", "cupressocyparis", "juniperus",
    "cedrus", "sequoia", "sequoiadendron", "larix", "calocedrus",
    "cryptomeria", "metasequoia", "taxodium", "araucaria", "podocarpus",
    "x cupressocyparis", "xcupressocyparis",
}

#: Genera whose habit is a narrow column rather than a cone or a ball --
#: Lombardy poplar, Italian cypress, fastigiate hornbeam.  Worth its own
#: form because a line of them is a very recognisable silhouette.
COLUMNAR_HINT = ("fastigiat", "columnar", "pyramidal", "'stricta'",
                 "sentinel", "italian cypress", "lombardy")

PALM = {"trachycarpus", "chamaerops", "phoenix", "washingtonia"}

#: Fallback height where neither the species nor the park data says, m.
#: A street tree, not a forest one.
DEFAULT_HEIGHT = 9.0


def ask(url, params, tries: int = 4):
    for attempt in range(tries):
        request = urllib.request.Request(
            url + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except Exception as error:                       # noqa: BLE001
            if attempt == tries - 1:
                raise
            print("   %s, retrying" % type(error).__name__)
            time.sleep(4 * (attempt + 1))
    raise RuntimeError("unreachable")


def genus_of(name) -> str:
    text = str(name or "").strip().lower()
    return text.split()[0] if text else ""


def form_of(scientific, common="") -> int:
    """Index into :data:`FORMS`."""
    blob = ("%s %s" % (scientific or "", common or "")).lower()
    genus = genus_of(scientific)
    if genus in PALM:
        return FORMS.index("palm")
    if any(hint in blob for hint in COLUMNAR_HINT):
        return FORMS.index("columnar")
    if genus in CONIFER:
        return FORMS.index("conifer")
    return FORMS.index("broadleaf")


def page_through(url, bounds, fields, label):
    """Every feature in ``bounds``, paged."""
    south, west, north, east = bounds
    common = {
        "where": "1=1",
        "geometry": "%f,%f,%f,%f" % (west, south, east, north),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ",".join(fields), "returnGeometry": "true",
        "orderByFields": "OBJECTID", "f": "json",
    }
    total = ask(url, dict(common, returnCountOnly="true"))["count"]
    print("  %s: %d to fetch" % (label, total))
    out, offset = [], 0
    while offset < total:
        page = ask(url, dict(common, resultOffset=offset,
                             resultRecordCount=PAGE))
        features = page.get("features", [])
        if not features:
            break
        out.extend(features)
        offset += len(features)
        print("   %d / %d" % (min(offset, total), total), end="\r")
    print("   %d fetched               " % len(out))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="seattle_trees.npz")
    args = parser.parse_args(argv)

    bounds = SEATTLE_DEM_BOUNDS
    print("City of Seattle trees")

    park = page_through(SPR, bounds,
                        ["SPECIES", "COMMON", "HEIGHT_WL", "DBH_WL"],
                        "SPR Tree View (parks, measured heights)")

    # -- heights by species, from the measured set ---------------------
    measured = {}
    for feature in park:
        attributes = feature["attributes"]
        height = attributes.get("HEIGHT_WL")
        species = str(attributes.get("SPECIES") or
                      attributes.get("COMMON") or "").strip().lower()
        if not species or not height:
            continue
        value = float(height) * FOOT
        if 1.0 < value < 90.0:
            measured.setdefault(species, []).append(value)
    by_species = {name: (float(np.median(values)), len(values))
                  for name, values in measured.items()}
    print("  %d species with a measured height; median over all %.1f m"
          % (len(by_species),
             float(np.median([v for vs in measured.values() for v in vs]))))
    ranked = sorted(by_species.items(), key=lambda kv: -kv[1][1])[:6]
    for name, (height, count) in ranked:
        print("     %-28s %5.1f m  (n=%d)" % (name[:28], height, count))

    street = page_through(SDOT, bounds,
                          ["SCIENTIFIC_NAME", "BOTANICAL_NAME"],
                          "SDOT Trees (streets, species only)")

    points, heights, forms, species_out, sources = [], [], [], [], []

    def add(lat, lon, scientific, common, height, source):
        points.append((lat, lon))
        heights.append(height)
        forms.append(form_of(scientific, common))
        species_out.append(str(scientific or common or "")[:48])
        sources.append(source)

    for feature in park:
        geometry = feature.get("geometry") or {}
        if "x" not in geometry:
            continue
        attributes = feature["attributes"]
        raw = attributes.get("HEIGHT_WL")
        species = str(attributes.get("SPECIES") or "").strip().lower()
        if raw and 1.0 < float(raw) * FOOT < 90.0:
            height, source = float(raw) * FOOT, 0      # measured
        else:
            height, source = by_species.get(species,
                                            (DEFAULT_HEIGHT, 0))[0], 1
        add(geometry["y"], geometry["x"], attributes.get("SPECIES"),
            attributes.get("COMMON"), height, source)

    inferred = 0
    for feature in street:
        geometry = feature.get("geometry") or {}
        if "x" not in geometry:
            continue
        attributes = feature["attributes"]
        scientific = (attributes.get("SCIENTIFIC_NAME")
                      or attributes.get("BOTANICAL_NAME"))
        species = str(scientific or "").strip().lower()
        height, source = by_species.get(species, (None, 0))[0], 1
        if height is None:
            # Genus, then the global median, before a bare default.
            genus = genus_of(scientific)
            same = [v for name, (v, _n) in by_species.items()
                    if genus and name.startswith(genus)]
            height = float(np.median(same)) if same else DEFAULT_HEIGHT
            source = 2
        inferred += 1
        add(geometry["y"], geometry["x"], scientific, "", height, source)

    counts = np.bincount(np.array(forms, dtype=int), minlength=len(FORMS))
    print("  %d trees: %s" % (len(points),
                              ", ".join("%s %d" % (FORMS[i], counts[i])
                                        for i in range(len(FORMS))
                                        if counts[i])))
    provenance = np.bincount(np.array(sources, dtype=int), minlength=3)
    print("  heights: %d measured, %d from the species median, %d from "
          "the genus or a default" % tuple(provenance))

    target = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "coxswain", "data", args.out)
    np.savez_compressed(
        target,
        tree_xy=np.array(points, dtype=np.float64),
        tree_height=np.array(heights, dtype=np.float32),
        tree_form=np.array(forms, dtype=np.int8),
        tree_species=np.array(species_out, dtype="<U48"),
        tree_height_source=np.array(sources, dtype=np.int8),
        bounds=np.array(bounds, dtype=np.float64),
    )
    print("wrote %s (%.1f MB)" % (target, os.path.getsize(target) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
