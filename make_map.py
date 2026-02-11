#!/usr/bin/env python3
"""
Poster-quality map of San Francisco: neighborhoods + hillshade + contours.

Layers (bottom to top):
  1. Surrounding terrain — gray hillshade for Marin, San Mateo, islands
  2. Neighborhoods — earthy-colored polygons with graph coloring
  3. Colored hillshade from 10 m DEM
  4. Elevation contours (major every 50 m, minor in between)
  5. Parks — vivid green polygons with overlap-aware labels
  6. Bridges — side-elevation profiles (Golden Gate + Bay Bridge)
  7. Title, attribution, geographic labels

Output: 18x18 inch poster at 600 DPI (TIFF) + 200 DPI preview (PNG).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import json
import logging
import numpy as np

from process_neighborhoods import parse_neighborhoods, build_neighborhood_index
from process_elevation import load_elevation_from_file
from process_combined import add_combined_visualization_to_axis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BG_COLOR = "#faf8f4"  # Light paper background


# ---------------------------------------------------------------------------
# Parks
# ---------------------------------------------------------------------------

def add_parks(
    ax,
    parks_file: str = "data/sf_park.json",
    city_boundary=None,
    min_acres_for_label: float = 3.0,
):
    """Add park polygons clipped to city boundary with overlap-aware labels.

    Parks are drawn in vivid green and labeled largest-first.  Labels that
    would overlap existing neighborhood text are skipped.  Some parks are
    excluded from *auto*-labeling because they have manual rotated labels
    placed in make_poster() (e.g. Lower Great Highway, Park Presidio Blvd).
    """
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    from shapely.geometry import shape as shapely_shape, MultiPolygon, Polygon

    with open(parks_file, "r") as f:
        parks = json.load(f)

    # Parks whose polygons are skipped entirely (drawn as neighborhoods)
    skip_entirely = ("Golden Gate Park -", "Telegraph Hill -")
    # Parks whose polygons are kept but auto-labels are suppressed
    # (manually labeled with rotation in make_poster)
    skip_label_only = ("Lower Great Highway",)

    patches_list = []
    labeled_parks = []

    for park in parks:
        geom_data = park.get("shape")
        if not geom_data or geom_data.get("type") != "MultiPolygon":
            continue

        name = park.get("property_name", "")
        if any(name.startswith(p) for p in skip_entirely):
            continue

        try:
            park_geom = shapely_shape(geom_data)
            if not park_geom.is_valid:
                park_geom = park_geom.buffer(0)

            # Clip to city boundary
            if city_boundary is not None:
                park_geom = park_geom.intersection(city_boundary)

            if park_geom.is_empty:
                continue

            # Collect polygons from the clipped geometry
            polys = []
            if isinstance(park_geom, Polygon):
                polys = [park_geom]
            elif isinstance(park_geom, MultiPolygon):
                polys = list(park_geom.geoms)
            elif hasattr(park_geom, "geoms"):
                polys = [g for g in park_geom.geoms if isinstance(g, Polygon)]

            for poly in polys:
                coords = np.array(poly.exterior.coords)
                patches_list.append(mpatches.Polygon(coords, closed=True))

            # Track for auto-labeling (skip manually-labeled parks)
            acres = float(park.get("acres", 0) or 0)
            if (acres >= min_acres_for_label and name and not park_geom.is_empty
                    and not any(name.startswith(p) for p in skip_label_only)):
                centroid = park_geom.centroid
                labeled_parks.append((name, centroid.x, centroid.y, acres))

        except Exception:
            continue

    if patches_list:
        pc = PatchCollection(
            patches_list,
            facecolors="#1e7a2a",
            edgecolors="#145a1a",
            linewidths=0.6,
            alpha=0.75,
            zorder=3,
        )
        ax.add_collection(pc)

    # --- Overlap-aware labeling ---
    # Collect bounding boxes of existing text (neighborhood labels) so park
    # labels don't overlap them.
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_transform = ax.transData.inverted()

    existing_boxes = []
    for txt in ax.texts:
        try:
            bbox = txt.get_window_extent(renderer)
            data_bbox = inv_transform.transform_bbox(bbox)
            existing_boxes.append((data_bbox.x0, data_bbox.y0,
                                   data_bbox.x1, data_bbox.y1))
        except Exception:
            continue

    def overlaps_existing(cx, cy, name, fs):
        """Check if a label at (cx, cy) would overlap any existing text."""
        tmp = ax.text(cx, cy, name, fontsize=fs, ha="center", va="center",
                      fontweight="bold", fontfamily="sans-serif", fontstyle="italic")
        bbox = tmp.get_window_extent(renderer)
        tmp.remove()
        data_bbox = inv_transform.transform_bbox(bbox)
        x0, y0, x1, y1 = data_bbox.x0, data_bbox.y0, data_bbox.x1, data_bbox.y1
        for ex0, ey0, ex1, ey1 in existing_boxes:
            if not (x1 < ex0 or x0 > ex1 or y1 < ey0 or y0 > ey1):
                return True
        return False

    # Label parks — largest first, skip overlaps
    labeled_parks.sort(key=lambda p: -p[3])
    skipped = 0
    for name, cx, cy, acres in labeled_parks:
        if acres > 50:
            fs = 7.0
        elif acres > 15:
            fs = 5.5
        elif acres > 6:
            fs = 4.5
        else:
            fs = 3.8

        if overlaps_existing(cx, cy, name.upper(), fs):
            skipped += 1
            continue

        txt = ax.text(
            cx, cy, name.upper(),
            fontsize=fs, ha="center", va="center",
            color="#1a4a1a", fontweight="bold", fontfamily="sans-serif",
            fontstyle="italic", zorder=8,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )

        # Track this label's bbox so later labels avoid it too
        bbox = txt.get_window_extent(renderer)
        data_bbox = inv_transform.transform_bbox(bbox)
        existing_boxes.append((data_bbox.x0, data_bbox.y0,
                               data_bbox.x1, data_bbox.y1))

    logger.info(
        f"  Added {len(patches_list)} park polygons, "
        f"labeled {len(labeled_parks) - skipped} parks (skipped {skipped} overlaps)"
    )


# ---------------------------------------------------------------------------
# Surrounding terrain (Marin County, San Mateo County, islands)
# ---------------------------------------------------------------------------

def add_surrounding_terrain(ax, dem_path="data/sf_dem_10m_marin.tif", city_boundary=None):
    """Add gray hillshaded terrain for all land outside SF city limits.

    Ocean (elevation <= 1 m) is left transparent so the paper background
    shows through.  The SF city boundary is excluded since it is already
    rendered with colored neighborhood polygons.
    """
    import rasterio
    import shapely

    logger.info("  Adding surrounding terrain...")

    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        bounds = src.bounds
        rows, cols = elev.shape

    # Land mask: elevation > 1 m (reliably separates land from ocean)
    is_land = elev > 1.0

    # Exclude SF city boundary (already rendered with neighborhoods)
    if city_boundary is not None:
        lon = np.linspace(bounds.left, bounds.right, cols)
        lat = np.linspace(bounds.top, bounds.bottom, rows)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        pts = shapely.points(lon_grid.ravel(), lat_grid.ravel())
        in_sf = shapely.contains(city_boundary, pts).reshape(rows, cols)
        is_land = is_land & ~in_sf

    # Compute hillshade (sun from northwest at 45 deg altitude)
    dy, dx = np.gradient(elev * 2.0, 10.0)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad, alt_rad = np.radians(315), np.radians(45)
    hs = (np.sin(alt_rad) * np.cos(slope) +
          np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    hs = np.clip(hs, 0, 1)

    # Gray terrain with hillshade, washed toward paper white
    base_gray = 0.72
    shadow_floor = 0.55
    shade = shadow_floor + (1 - shadow_floor) * hs
    gray_val = base_gray * shade

    paper = np.array([0.96, 0.94, 0.91])
    wash = 0.3

    rgba = np.zeros((rows, cols, 4), dtype=float)
    for c in range(3):
        rgba[:, :, c] = gray_val * (1 - wash) + paper[c] * wash
    rgba[:, :, 3] = is_land.astype(float)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    ax.imshow(rgba, extent=extent, origin='upper', interpolation='bilinear',
              zorder=1)

    # --- Geographic labels for surrounding areas ---
    label_style = dict(
        ha="center", va="center", fontweight="bold", fontfamily="sans-serif",
        color="#6a6a6a",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=8,
    )
    # Counties
    ax.text(-122.513, 37.862, "MARIN COUNTY", fontsize=9,
            ha="left", **{k: v for k, v in label_style.items() if k != "ha"})
    ax.text(-122.447, 37.705, "SAN MATEO COUNTY", fontsize=9, **label_style)
    # Islands and peninsulas
    ax.text(-122.4204, 37.8284, "ALCATRAZ\nISLAND", fontsize=6,
            linespacing=1.1, **label_style)
    ax.text(-122.431, 37.862, "ANGEL\nISLAND", fontsize=7,
            linespacing=1.1, **label_style)
    ax.text(-122.463, 37.865, "TIBURON", fontsize=6, **label_style)

    logger.info("  Surrounding terrain added")


# ---------------------------------------------------------------------------
# Bridges (side-elevation profiles)
# ---------------------------------------------------------------------------

def add_bridges(ax):
    """Draw side-elevation suspension bridge profiles at geographic locations.

    Cable physics:
      - Main spans (tower-to-tower): parabola under uniform deck load,
        h(s) = peak_h * (2s - 1)^2, sagging to 0 at midspan.
      - Approach spans (anchorage-to-tower): catenary under cable self-weight,
        h(s) = h_tower * (cosh(a*s) - 1) / (cosh(a) - 1).

    Golden Gate Bridge: 2 towers at 17.5% and 82.5% (Wikipedia dimensions:
        side spans 345 m, main span 1280 m, total 1970 m).
    Bay Bridge west span: 3 towers at 16.7%, 50%, 83.3% (Wikipedia: side
        spans 1160 ft, main spans 2310 ft, center anchorage merged).

    Coordinates from OSM, trimmed to DEM coastline with slight land overlap.
    """

    def _cable_height(t, towers, peak_h):
        """Cable height above deck at fraction t along the bridge."""
        a = 2.0  # Catenary curvature parameter

        def _catenary_rise(s, h_target):
            """Catenary from 0 to h_target as s goes 0 -> 1."""
            return h_target * (np.cosh(a * s) - 1) / (np.cosh(a) - 1)

        pts = [(0.0, 0.0)] + [(f, h) for f, h in towers] + [(1.0, 0.0)]
        for i in range(len(pts) - 1):
            f0, h0 = pts[i]
            f1, h1 = pts[i + 1]
            if f0 <= t <= f1:
                span = f1 - f0
                if span == 0:
                    return h0
                s = (t - f0) / span  # 0..1 within this span

                if h0 < h1:
                    # Approach: anchorage (low) -> tower (high)
                    return _catenary_rise(s, h1)
                elif h0 > h1:
                    # Approach: tower (high) -> anchorage (low)
                    return _catenary_rise(1 - s, h0)
                else:
                    # Main span: tower to tower — parabolic sag to 0
                    return h0 * (2 * s - 1) ** 2
        return 0.0

    def _draw_bridge(ax, start, end, perp, tower_fracs, peak_h, n_rods,
                     color, cable_lw, rod_lw, rod_alpha, deck_lw, zorder):
        """Draw a complete bridge: deck line, cable curve, suspender rods."""
        vec = end - start
        towers = [(f, peak_h) for f in tower_fracs]

        # Deck line
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, lw=deck_lw, zorder=zorder, solid_capstyle="butt")

        # Cable curve (300-point resolution)
        t_arr = np.linspace(0, 1, 300)
        h_arr = np.array([_cable_height(t, towers, peak_h) for t in t_arr])
        cable_pts = (start[np.newaxis, :] +
                     np.outer(t_arr, vec) +
                     np.outer(h_arr, perp))
        ax.plot(cable_pts[:, 0], cable_pts[:, 1],
                color=color, lw=cable_lw, zorder=zorder + 1, alpha=0.85)

        # Evenly spaced suspender rods from deck up to cable
        for i in range(n_rods):
            ti = (i + 1) / (n_rods + 1)
            deck_pt = start + vec * ti
            h = _cable_height(ti, towers, peak_h)
            cable_pt = deck_pt + perp * h
            ax.plot([deck_pt[0], cable_pt[0]], [deck_pt[1], cable_pt[1]],
                    color=color, lw=rod_lw, zorder=zorder + 1, alpha=rod_alpha)

    zz = 9  # Base zorder for bridges

    # ==================== GOLDEN GATE BRIDGE ====================
    # OSM coordinates, trimmed to coastline with slight land extension
    gg_south = np.array([-122.47700, 37.81050])
    gg_north = np.array([-122.47970, 37.82630])
    gg_dir = (gg_north - gg_south) / np.linalg.norm(gg_north - gg_south)
    gg_perp = np.array([-gg_dir[1], gg_dir[0]])

    # Tower positions from Wikipedia: side spans 345m, main span 1280m
    # Total suspended = 1970m -> towers at 17.5% and 82.5%
    _draw_bridge(ax, gg_south, gg_north, gg_perp,
                 tower_fracs=[0.175, 0.825], peak_h=0.0045, n_rods=35,
                 color="#C0362C", cable_lw=0.8, rod_lw=0.3,
                 rod_alpha=0.55, deck_lw=1.8, zorder=zz)

    label_pos = (gg_south + gg_north) / 2 - gg_perp * 0.004
    ax.text(label_pos[0], label_pos[1], "GOLDEN GATE\nBRIDGE",
            ha="center", va="top", fontsize=5, color="#8B2500",
            fontweight="bold", fontfamily="sans-serif",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
            zorder=zz + 5, linespacing=1.1)

    # ==================== BAY BRIDGE (west span) ====================
    # OSM coordinates (SF to Yerba Buena Island), trimmed to coastline
    bb_west = np.array([-122.38800, 37.78870])
    bb_east = np.array([-122.36700, 37.80830])
    bb_dir = (bb_east - bb_west) / np.linalg.norm(bb_east - bb_west)
    bb_perp = np.array([-bb_dir[1], bb_dir[0]])

    # Wikipedia: two connected suspension spans with center anchorage
    # Side spans 1160 ft, main spans 2310 ft
    # Towers at ~16.7%, center anchorage at 50%, ~83.3%
    _draw_bridge(ax, bb_west, bb_east, bb_perp,
                 tower_fracs=[0.167, 0.50, 0.833],
                 peak_h=0.0045, n_rods=35,
                 color="#6a6a6a", cable_lw=0.7, rod_lw=0.25,
                 rod_alpha=0.5, deck_lw=1.6, zorder=zz)

    label_pos = (bb_west + bb_east) / 2 - bb_perp * 0.004
    ax.text(label_pos[0], label_pos[1], "BAY BRIDGE",
            ha="center", va="top", fontsize=5, color="#555555",
            fontweight="bold", fontfamily="sans-serif",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
            zorder=zz + 5)

    logger.info("  Added Golden Gate Bridge and Bay Bridge")


# ---------------------------------------------------------------------------
# Main poster composition
# ---------------------------------------------------------------------------

def make_poster(
    neighborhoods_file: str = "data/sf_neighborhoods.json",
    elevation_file: str = "data/sf_elevation.json",
    dem_path: str = "data/sf_dem_10m_marin.tif",
    save_path: str = "images/sf_combined_poster.tiff",
    figsize=(18, 18),
    dpi: int = 600,
    palette: str = "earthy",
    max_colors: int = 8,
):
    """Generate the full San Francisco poster and save to disk.

    Composes all layers (terrain, neighborhoods, contours, parks, bridges)
    into an 18x18 inch figure, adds title and attribution, then saves both
    a full-resolution TIFF and a 200 DPI PNG preview.
    """
    # --- Load data ---
    logger.info("Loading data...")
    with open(neighborhoods_file, "r") as f:
        neighborhoods = parse_neighborhoods(json.load(f))
    logger.info(f"  {len(neighborhoods)} neighborhoods")

    elevation_data = load_elevation_from_file(elevation_file)
    logger.info(f"  {len(elevation_data.isolines)} elevation isolines")

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # --- Layer 1: Neighborhoods + hillshade + contours ---
    add_combined_visualization_to_axis(
        ax,
        neighborhoods,
        elevation_data,
        max_colors=max_colors,
        show_neighborhood_labels=True,
        palette=palette,
        contour_style="darken",
        dem_path=dem_path,
        elevation_linewidth=0.55,
        colored_hillshade=True,
        saturation=0.65,
        elevation_alpha=0.45,
    )

    # --- Build city boundary (used by parks + surrounding terrain) ---
    from shapely.ops import unary_union

    _, all_polys, _ = build_neighborhood_index(neighborhoods)
    city_boundary = unary_union(all_polys)

    # --- Layer 2: Parks ---
    add_parks(ax, city_boundary=city_boundary)

    # --- Layer 3: Surrounding terrain (gray hillshade outside SF) ---
    add_surrounding_terrain(ax, dem_path=dem_path, city_boundary=city_boundary)

    # --- Layer 4: Bridges ---
    add_bridges(ax)

    # --- Manual park labels (rotated to match park orientation) ---
    # These parks are too narrow for auto-labeling to place well.
    park_label_style = dict(
        ha="center", va="center",
        color="#1a4a1a", fontweight="bold", fontfamily="sans-serif",
        fontstyle="italic", zorder=8,
        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
    )
    # Park Presidio Blvd: long N-S strip between Inner/Outer Richmond
    # Rotation 93.5 deg computed from park coordinate data
    ax.text(-122.4735, 37.7803, "PARK PRESIDIO BLVD",
            fontsize=5, rotation=93.5, **park_label_style)
    # Lower Great Highway: long N-S strip along the west coast
    # Rotation 96 deg computed from park coordinate data
    ax.text(-122.5107, 37.7597, "LOWER GREAT HIGHWAY",
            fontsize=7, rotation=96, **park_label_style)

    # --- Clean up axes ---
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Push map content toward the bottom by extending ylim upward,
    # creating space for the title above the map.
    ylo, yhi = ax.get_ylim()
    y_span = yhi - ylo
    ax.set_ylim(ylo - y_span * 0.02, yhi + y_span * 0.25)

    # --- Poster title (in data coords, in the empty space above the map) ---
    xlo, xhi = ax.get_xlim()
    title_x = (xlo + xhi) / 2
    title_y = yhi + y_span * 0.03
    ax.text(
        title_x, title_y,
        "S A N   F R A N C I S C O",
        ha="center", va="bottom",
        fontsize=64, fontweight="bold", fontfamily="sans-serif",
        color="#2a2218", zorder=10,
    )

    # --- Attribution (bottom-right corner) ---
    fig.text(
        0.97, 0.015,
        "DEM: USGS 10 m\nContours & Neighborhoods: SF Open Data",
        ha="right", va="bottom", multialignment="right",
        fontsize=9, color="#9a9080", fontfamily="sans-serif",
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    # --- Save outputs ---
    logger.info(f"Saving to {save_path} at {dpi} DPI...")
    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
    logger.info(f"Done — {save_path}")

    preview_path = save_path.rsplit(".", 1)[0] + "_preview.png"
    logger.info(f"Saving preview to {preview_path} at 200 DPI...")
    fig.savefig(preview_path, dpi=200, facecolor=fig.get_facecolor())
    logger.info(f"Done — {preview_path}")

    return fig


if __name__ == "__main__":
    make_poster()
