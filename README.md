# map-vibes

Poster-quality maps of **San Francisco** and **Manhattan**, rendered with
matplotlib from public data: OpenData neighborhoods, USGS 10 m DEMs, zoning
and land-use polygons, building footprints, bridge and tunnel geometry.

Each city lives in its own folder (`san_francisco/`, `new_york/`) and produces
a 14×18 to 18×18 inch print-ready TIFF plus a 200 DPI PNG preview. Shared
data-processing code (palettes, hillshading, contours, zoning) lives in
`common/`.

## Quick start

```bash
uv sync                                  # install dependencies
python san_francisco/download_data.py    # fetch SF open data + USGS DEM (~30 MB)
python san_francisco/make_map.py         # render preview to san_francisco/images/
```

Manhattan is the same:

```bash
python new_york/download_data.py
python new_york/make_map.py
```

By default each run produces a `*_preview.png` only. Pass `--print` to also
save the full-resolution TIFF (larger, slower, suitable for actual printing).

## Repository layout

```
common/             shared modules (imported by both cities)
  colors.py         background color + hex/rgb darkening helpers
  constants.py      DEM resolution + default extent
  draw_bridges.py   side-elevation bridge rendering
  process_*.py      neighborhoods, terrain, elevation, zoning, streets
san_francisco/
  download_data.py  fetches SF open data + DEM into san_francisco/data/
  make_map.py       full poster (neighborhoods + hillshade + contours + parks + bridges)
  make_topo_map.py  gray-hillshade topographic variant (no colors, no contours)
new_york/           Manhattan equivalents
```

## Key flags for `make_map.py`

Both cities share most flags. Unless noted, defaults produce a neighborhood-
colored poster; layers below are opt-in.

| Flag | What it does |
|---|---|
| `--palette NAME` | `earthy` (default), `nordic`, `coastal`, `dusk`, `clay`, `mineral`, `industrial`, or `transparent` (no neighborhood coloring) |
| `--show-zoning` / `--no-show-zoning` | Zoning-district overlay (residential, commercial, industrial, …). **On by default.** |
| `--show-land-use` | Overlay parcel-level land use (finer-grained than zoning; takes precedence over zoning) |
| `--show-buildings` | **NYC only.** Building footprints with isometric shadows |
| `--show-zoning-special` | **NYC only.** Special-purpose districts overlay |
| `--show-zoning-commercial` | **NYC only.** Commercial overlay districts |
| `--park-labels` | Label parks in addition to drawing their polygons |
| `--bridge-labels` | Label bridges (Golden Gate, Bay Bridge, Brooklyn, etc.) |
| `--no-neighborhood-labels` | Hide neighborhood names |
| `--no-minor-contours` / `--no-major-contours` | Hide elevation contour lines (**SF:** both on by default; **NYC:** both off by default — DEM is mostly flat) |
| `--width N --height N` | Poster size in inches |
| `--dpi N` | Resolution; `300` is ~4× faster than default `600` |
| `--print` | Also save full-resolution TIFF (default: PNG preview only) |

Colors for zoning categories, parks, and bridges are all hex-configurable
(`--park-color`, `--zoning-residential-color`, `--golden-gate-color`, …).
Run `python <city>/make_map.py --help` for the complete list.

### Flag combinations worth trying

```bash
# SF with the default zoning overlay plus park/bridge labels
python san_francisco/make_map.py --park-labels --bridge-labels

# Parcel-level land use instead of zoning districts
python san_francisco/make_map.py --show-land-use

# Neighborhood-only view (disable the default zoning overlay)
python san_francisco/make_map.py --no-show-zoning

# Manhattan with buildings + shadows, on top of the default zoning overlay
python new_york/make_map.py --show-buildings --bridge-labels

# Minimal: transparent palette, no labels, just terrain + contours
python san_francisco/make_map.py --palette transparent --no-show-zoning --no-neighborhood-labels

# Fast iteration at lower resolution
python new_york/make_map.py --dpi 300
```

## `make_topo_map.py`

An alternate style: gray hillshade over the entire DEM with neighborhood
borders + labels and bridges, but no neighborhood colors, no contour lines,
no parks. Useful as a muted topographic backdrop or as a base for hand
annotation. No CLI flags — edit constants at the top of the file to tweak.

```bash
python san_francisco/make_topo_map.py
python new_york/make_topo_map.py
```

## Data sources

- **USGS 3DEP**: 10 m Digital Elevation Models (hillshade + contours)
- **SF Open Data**: neighborhoods, streets, parks, zoning, land use, elevation isolines
- **NYC Open Data**: NTA neighborhoods, parks, buildings (with roof heights)
- **NYC DCP / ArcGIS**: zoning districts, special-purpose, commercial, MapPLUTO
- **OpenStreetMap (Overpass)**: tunnel centerlines (Holland, Lincoln, Queens-Midtown, Hugh L. Carey)

Downloaded files go under `<city>/data/` and are cached between runs.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
```
