# The scenery: what the 3-D renderer draws, and where every piece comes from

The 6-DOF boat is validated physics. The world it rows through is
**data**, from six public sources, and this page says which is which so
nothing in a render gets mistaken for a result.

Everything here is course-independent. `RiverScene` takes the elevation,
the footprints, the imagery and the docks as arguments; the Charles is
only the default. Pointing it at a second water was the test of that,
and it needed no new renderer.

---

## The sources

| Layer | Source | Licence | Fetched by |
|---|---|---|---|
| Ground shape | USGS 3DEP bare-earth DEM | Public domain | `tools/fetch_dem.py` |
| Ground colour | USGS/USDA NAIP + HRO orthoimagery | Public domain | `tools/fetch_imagery.py` |
| Building shape and height | City of Seattle *Building Outlines 2015* (lidar) | City of Seattle open data | `tools/fetch_seattle_buildings.py` |
| Building class and name | OpenStreetMap | ODbL | `tools/extract_structures.py` |
| Shoreline, docks, houseboats | OpenStreetMap | ODbL | `tools/extract_seattle_water.py` |
| Bridges, trees, canopy | OpenStreetMap | ODbL | `tools/extract_structures.py` |

To rebuild Seattle from nothing:

```bash
python tools/fetch_dem.py --bounds 47.590 -122.375 47.670 -122.300 --out seattle_dem.npz --metres 2.7
python tools/fetch_imagery.py --bounds 47.590 -122.375 47.670 -122.300 --out seattle_imagery.jpg
python tools/extract_structures.py --bounds 47.590,-122.375,47.670,-122.300 --out seattle_structures.npz
python tools/fetch_seattle_buildings.py
```

Order matters for the last one: it *merges over* the OSM extract rather
than replacing it, because lidar sees a roof and does not know it is a
boathouse.

Then:

```bash
python scripts/render_totl3d.py --stills            # four frames from the cox's seat
python scripts/render_totl3d.py --view cox          # the video
python scripts/render_totl3d.py --no-imagery        # flat colour, for comparison
```

---

## Heights: measured, not guessed

The OSM extract over this box has 40,386 footprints and knows the height
of **179** of them. 2,889 more come from `building:levels` times an
assumed storey height. The remaining 37,318 — 92% — were inferred from
the building type, which means every untagged building in Seattle was
exactly nine metres tall. A skyline built from that is a skyline of one
number.

Seattle's `Building Outlines 2015` carries `BP99_APEX`, the lidar-measured
roof apex elevation. Height is that apex minus the 3DEP ground under it.
Checked against published heights before being believed:

| Building | Published | Measured | Error |
|---|---|---|---|
| Columbia Center | 284 m | 280.4 m | −3.6 m |
| Amazon Doppler | 160 m | 156.3 m | −3.7 m |

53,710 buildings now carry a measured height; the median is 6.8 m, which
is a Seattle bungalow, with a proper tail out to 281 m.

**Units and datum are where this goes wrong.** `BP99_APEX` is in *feet*
and the DEM is in *metres*, both NAVD88, and nothing in the service says
so. It was settled by computing two known buildings both ways: feet gives
280 m for Columbia Center and metres gives 1013 m.

---

## Three traps, all of which failed silently

### 1. `exportImage` moves the bounding box

The ArcGIS `exportImage` endpoint honours its `size` argument exactly and
**changes the bounding box** to match that size's aspect ratio, announcing
it only in a JSON field. A size computed from metres — the only sensible
way to choose one — asks for an aspect the box in degrees does not have.

Requesting 47.590–47.670 N returned **47.571–47.689**: a 2.2 km shift that
georeferenced Lake Union onto a hillside while every summary statistic
stayed reasonable. 43% of the lake came back as ground above 10 m and
nothing raised an error.

Both fetchers now derive `size` from the box's own degree aspect *and*
read back the extent actually served, storing it in a `.json` sidecar
that the loader trusts over the request. `tests/test_terrain_registration.py`
locks it: OSM says where the shore is, 3DEP says where the low ground is,
and those have to be the same place. Neither dataset can check itself.

### 2. Lidar over water is not water

Thresholding the DEM half a metre above the pool to find the lake called
**48% of Lake Union dry land** — the specular return off water scatters
from the pool level up past 7 m. No threshold fixes it: loose enough to
recover the lake drowns 5.6% of the surrounding land.

So the elevation model is not asked. Water comes from OpenStreetMap
polygons and the DEM is left to do the thing it is good at, which is the
shape of the ground. Agreement with the racing shoreline: **98.8%**.

### 3. A photograph already has its light in it

PyVista's default lighting is a warm key with a cool fill. Applied on top
of an orthophoto — which was taken in real sunlight and has the shadows
baked in — it shifted Lake Union from (0.13, 0.22, 0.26) blue in the
photograph to (16, 45, 40) green on screen. Textured surfaces now render
at `ambient 0.92, diffuse 0.12`.

---

## What the renderer decides, and why

**Two tiers of building.** Near, out to the terrain window, everything is
drawn: at 200 m a boathouse is a landmark. Beyond that only what subtends
a real angle (`height / distance > 0.012`), tallest first, capped at 400.
Rowing south down Lake Union you are looking straight at downtown 4 km
away — the most recognisable thing in the frame and far too much geometry
to draw in full.

**Aerial perspective is not decoration.** Colour is mixed toward the
horizon as `1 − exp(−d / 5200 m)`. Without it the downtown towers render
at the same contrast as the boathouse 200 m away and the eye reads them
as small and near rather than large and far. Distance on open water is
judged almost entirely by haze.

**Roof colour comes from the photograph.** An orthophoto is a picture of
roofs seen from directly above, so this is not an approximation — it *is*
the roof colour, for all 53,710 buildings, from a source already fetched
for the ground. Sampled as a median over a grid across the footprint, not
at the centroid: one pixel is a skylight, a rooftop unit, or the shadow
of the next building along. The result is squeezed into a lightness band
of 0.30–0.78, because a roof spans white membrane to black asphalt and a
*wall* does not — read literally, one dark industrial roof put a solid
black block on the east shore.

**The docks are the coastline.** Lake Union has 499 piers, 141 houseboats,
7 marinas and 2 breakwaters, and they remove 40% of the lake. What a
coxswain steers off is not where the land legally starts; it is the end
of the dock, the outermost moored hull, the breakwater. They were already
in the optimiser's corridor and are now in the picture.

**Ground height has one definition.** `RiverScene.bank_height` is the only
place that answers "how high is the ground here", and the terrain mesh and
the building bases both go through it. When they were computed separately
they disagreed — the ground was clipped to 60 m while buildings stood at
true elevation, and more than half of Seattle's footprints hung in the air
over a flattened hillside. Buildings also extrude from the **lowest**
ground under the footprint, so one on a slope is buried into the hill
rather than left on its downhill corner; the gap runs to 1.8 m at the
ninth decile, which is a storey of daylight under a house.

---

## Known limits — read before trusting a picture

**These are massing models, not buildings.** A footprint extruded to a
measured height. The Space Needle is a round footprint pulled up 184 m,
so it renders as a cylinder; it has no saucer because nothing in either
dataset says it has one. The Gas Works machinery *is* present — 18
structures, 14–21 m tall and 3–9 m wide, which are the cracking towers —
because the lidar saw them, but they are boxes.

**The imagery is one day in the year NAIP flew.** Moored boats, wakes and
sun glint are baked into the water. Nothing in the render moves with the
weather.

**Depth is nominal, not surveyed.** `nominal_depth` is a shelf profile
invented so the course object has a depth field at all. At `Fr_h = 0.32`
the shallow-water correction is 1.00 regardless, so it changes no answer
currently drawn from it — but the course still declares `is_survey=False`
and should keep doing so until real soundings arrive.

**The traced course line crosses the dock survey.** 12% of the stations on
the line traced from the 2024 regatta map have under 8 m of clearance to a
mapped structure, and the line is 87.8% navigable rather than 100%. Either
the trace is offset or the dock polygons over-reach, and this code cannot
settle which. The corridor is therefore *pinned to the traced line* where
they conflict, rather than being given the 8 m of invented room it used to
get — that floor was letting the optimiser route 23% of the raced line
within 5 m of a pier. See §109 of [SOURCES.md](SOURCES.md).
