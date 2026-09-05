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
| Bridges, canopy, water | OpenStreetMap | ODbL | `tools/extract_structures.py` |
| Tree species and height | Seattle SPR Tree View + SDOT Trees | City of Seattle open data | `tools/fetch_seattle_trees.py` |

To rebuild Seattle from nothing:

```bash
python tools/fetch_dem.py --bounds 47.590 -122.375 47.670 -122.300 --out seattle_dem.npz --metres 2.7
python tools/fetch_imagery.py --bounds 47.590 -122.375 47.670 -122.300 --out seattle_imagery.jpg
python tools/extract_structures.py --bounds 47.590,-122.375,47.670,-122.300 --out seattle_structures.npz
python tools/fetch_seattle_buildings.py
python tools/fetch_seattle_trees.py
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
measured height, or for a tower, one prism per `building:part` between
its own `min_height` and `height`. That is enough for the Space Needle to
come out as three legs, a shaft, a saucer at 152–156 m and a deck at
158–161 m — a needle rather than the cylinder it was — but it is still
boxes and prisms, not architecture. The Gas Works machinery is present
(18 structures, 14–21 m tall, 3–9 m wide, the cracking towers) for the
same reason and with the same limit.

**Trees are three primitives.** A cone, a sphere, a cylinder, sized by
species and measured height. Good enough for a silhouette at 200 m,
which is the distance that matters; not a tree.

**The tree inventory is street and park trees only.** Nothing on private
land, which on the Eastlake and Westlake shores is most of it. The
`Seattle Tree Canopy 2021` polygons (49,876 in this box) would fill that
in and are fetched by nothing yet.

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

---

## Trees: species, form and measured height

The scene drew every tree as an identical green sphere from OpenStreetMap
points that carry no species and almost no height. On this shore that is
wrong in a way a coxswain notices: the banks are Douglas fir, western red
cedar and Sitka spruce among bigleaf maple and London plane, and a
conifer is a **cone**.

Two city datasets, fetched by `tools/fetch_seattle_trees.py`:

| | count in box | has species | has height |
|---|---|---|---|
| SPR Tree View (Parks) | 5,648 | yes | **yes, measured, 89% populated** |
| SDOT Trees (Transportation) | 87,008 | yes | no |

Heights come from Parks and are carried to Transportation **by species**:
the median measured height of every *Thuja plicata* in the park data
becomes the height of a *Thuja plicata* on a street. That is an
inference from measurements, not a field guide, and the per-tree
provenance is stored — 5,000 measured, 51,713 from a species median,
35,943 from the genus or a default.

92,656 trees: 86,240 broadleaf, 5,678 conifer, 610 columnar, 128 palm.
Form is decided by genus, which both datasets give. The tallest is a
Lombardy poplar at 39.6 m, correctly classified columnar.

**A fault in the city's data:** `SPECIES` carries the literal string
`trigger_error` on 142 records — their pipeline, not a plant. The heights
on those rows are fine and are kept; the name is blanked at load.

## Landmark bridges

`_LANDMARK_SPANS` records deck height and width for the Ship Canal Bridge
(I-5), the Aurora Bridge, the Portage Bay Viaduct and the three bascules,
drawn as a slab on piers — from the water a bridge is a horizontal line
held up by verticals, and the verticals are most of what makes it read.

**The deck height is published, not derived, and it has to be.** An OSM
bridge way runs out onto its approach embankment, and 3DEP is bare earth
so under the span it reads the water. Taking the height from the way's
own endpoints gave the Ship Canal Bridge a 37 m deck against a published
57, and the Aurora Bridge a 61 m one against a published 51 — one end of
it lands on the Queen Anne bluff, well above the roadway. This follows
the pattern `bridges.BRIDGE_STRUCTURE` already sets for the Charles.

## A fourth silent failure: the mirrored orthophoto

The imagery was flipped at load, on the reasoning that VTK's *v* axis
counts up from the south. PyVista flips the array itself when it builds
the texture, so doing it here as well **mirrored the photograph about the
middle of the tile**.

It hid for a long time, and the reason is worth recording: a point
mid-lake reflects to another point mid-lake, so the water still looked
like water. It only showed when the boat moved far enough north that its
reflection landed on Capitol Hill and the lake rendered pale grey. The
rendered water now matches the photograph to within one unit in 255.

The lesson is the same one for the fourth time — **rendering catches what
numbers hide** — with a corollary: a symmetric error hides longest,
because half the scene checks out.
