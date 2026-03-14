#!/usr/bin/env python3
"""
Poster-quality map of Manhattan: neighborhoods + hillshade.

Layers (bottom to top):
  1. Hillshade from USGS 10 m DEM (gray for all land, colored for Manhattan)
  2. Neighborhood borders (white edges)
  3. Parks — green polygons with labels
  4. Bridges — side-elevation profiles
  5. Neighborhood labels, geographic labels, title, attribution

Usage:
    python make_map.py                           # defaults: 14.4x18 in @ 600 DPI
    python make_map.py --width 18 --height 18    # square poster
    python make_map.py --dpi 300                  # lower resolution for testing
    python make_map.py --palette nordic           # different color scheme
"""

import argparse
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import json
import logging
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from process_neighborhoods import (
    parse_neighborhoods, build_neighborhood_index,
    get_neighborhood_color_map, add_smart_neighborhood_labels,
)
from process_terrain import add_hillshade
from process_elevation import (
    load_elevation_from_file, add_contour_lines,
    build_contour_color_index_from_neighborhoods,
    build_contour_color_index_from_color_source,
)
from process_zoning import (
    ZoneCategory, DEFAULT_ZONE_COLORS,
    load_nyc_zoning_color_source, load_nyc_land_use_color_source,
    add_zoning_legend,
    add_nyc_special_purpose, add_nyc_commercial_overlay,
)
from draw_bridges import draw_bridges
from colors import BG_COLOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Parks
# ---------------------------------------------------------------------------

def add_parks(ax, parks_file="data/manhattan_parks.json", city_boundary=None,
              min_acres_for_label=3.0, park_color="#1e7a2a", park_alpha=0.75,
              park_labels=True):
    """Add park polygons with overlap-aware labels."""
    from shapely.geometry import shape as shapely_shape, MultiPolygon, Polygon

    with open(parks_file, "r") as f:
        parks = json.load(f)

    # Parks handled as neighborhoods (skip polygon + label)
    skip_entirely = {"Central Park"}

    patches_list = []
    labeled_parks = []

    for park in parks:
        geom_data = park.get("multipolygon")
        if not geom_data:
            continue

        name = park.get("signname", "")
        if not name or name in skip_entirely:
            continue

        try:
            park_geom = shapely_shape(geom_data)
            if not park_geom.is_valid:
                park_geom = park_geom.buffer(0)
            if city_boundary is not None:
                park_geom = park_geom.intersection(city_boundary)
            if park_geom.is_empty:
                continue

            geom_polys = []
            if isinstance(park_geom, Polygon):
                geom_polys = [park_geom]
            elif isinstance(park_geom, MultiPolygon):
                geom_polys = list(park_geom.geoms)
            elif hasattr(park_geom, "geoms"):
                geom_polys = [g for g in park_geom.geoms if isinstance(g, Polygon)]

            for poly in geom_polys:
                coords = np.array(poly.exterior.coords)
                patches_list.append(mpatches.Polygon(coords, closed=True))

            acres = float(park.get("acres", 0) or 0)
            if acres >= min_acres_for_label and not park_geom.is_empty:
                centroid = park_geom.centroid
                labeled_parks.append((name, centroid.x, centroid.y, acres))

        except Exception:
            continue

    # Derive edge color by darkening the fill color
    r, g, b = int(park_color[1:3], 16), int(park_color[3:5], 16), int(park_color[5:7], 16)
    edge_color = f"#{int(r*0.7):02x}{int(g*0.7):02x}{int(b*0.7):02x}"

    if patches_list:
        pc = PatchCollection(
            patches_list,
            facecolors=park_color,
            edgecolors=edge_color,
            linewidths=0.6,
            alpha=park_alpha,
            zorder=3,
        )
        ax.add_collection(pc)

    if not park_labels:
        logger.info(f"  Added {len(patches_list)} park polygons (labels disabled)")
        return

    # Overlap-aware labeling
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

    # Derive label color by darkening fill further
    label_color = f"#{int(r*0.5):02x}{int(g*0.5):02x}{int(b*0.5):02x}"

    def overlaps_existing(cx, cy, label, fs):
        tmp = ax.text(cx, cy, label, fontsize=fs, ha="center", va="center",
                      fontweight="bold", fontfamily="sans-serif", fontstyle="italic")
        bbox = tmp.get_window_extent(renderer)
        tmp.remove()
        data_bbox = inv_transform.transform_bbox(bbox)
        x0, y0, x1, y1 = data_bbox.x0, data_bbox.y0, data_bbox.x1, data_bbox.y1
        for ex0, ey0, ex1, ey1 in existing_boxes:
            if not (x1 < ex0 or x0 > ex1 or y1 < ey0 or y0 > ey1):
                return True
        return False

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
            color=label_color, fontweight="bold", fontfamily="sans-serif",
            fontstyle="italic", zorder=15,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )

        bbox = txt.get_window_extent(renderer)
        data_bbox = inv_transform.transform_bbox(bbox)
        existing_boxes.append((data_bbox.x0, data_bbox.y0,
                               data_bbox.x1, data_bbox.y1))

    logger.info(
        f"  Added {len(patches_list)} park polygons, "
        f"labeled {len(labeled_parks) - skipped} parks (skipped {skipped} overlaps)"
    )


# ---------------------------------------------------------------------------
# Buildings (isometric shadow casting)
# ---------------------------------------------------------------------------

def add_buildings(ax, buildings_file="data/manhattan_buildings.json",
                  city_boundary=None,
                  shadow_color=(0.12, 0.10, 0.08),
                  shadow_alpha=0.45,
                  building_edge_color=(0.40, 0.35, 0.30, 0.45),
                  building_edge_width=0.15,
                  max_shadow_deg=0.003,
                  min_shadow_height=20.0):
    """Add building footprints with isometric shadow casting.

    Taller buildings cast longer shadows to the SE (sun from NW).
    Shadow length is proportional to roof height.
    """
    from shapely.geometry import shape as shapely_shape, MultiPolygon, Polygon

    with open(buildings_file, "r") as f:
        buildings = json.load(f)

    # Longitude correction at NYC latitude (~40.78°N)
    cos_lat = np.cos(np.radians(40.78))
    # Shadow direction: 135° (SE), corrected for lon/lat aspect ratio
    shadow_dir_lon = max_shadow_deg * 0.707 / cos_lat
    shadow_dir_lat = -max_shadow_deg * 0.707

    shadow_patches = []
    building_patches = []
    max_h = 1.0  # will be updated

    # First pass: collect all valid polygons + heights
    entries = []
    for bldg in buildings:
        geom_data = bldg.get("the_geom")
        if not geom_data:
            continue
        height = float(bldg.get("height_roof", 0) or 0)
        try:
            geom = shapely_shape(geom_data)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if city_boundary is not None:
                geom = geom.intersection(city_boundary)
            if geom.is_empty:
                continue

            polys = []
            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            elif hasattr(geom, "geoms"):
                polys = [g for g in geom.geoms if isinstance(g, Polygon)]

            for poly in polys:
                coords = np.array(poly.exterior.coords)
                if len(coords) >= 3:
                    entries.append((coords, height))
                    if height > max_h:
                        max_h = height
        except Exception:
            continue

    logger.info(f"  Parsed {len(entries)} building polygons "
                f"(max height: {max_h:.0f} ft)")

    # Second pass: create shadow + building patches
    for coords, height in entries:
        # Building outline (all buildings)
        building_patches.append(mpatches.Polygon(coords, closed=True))

        # Shadow (only for buildings above threshold)
        if height >= min_shadow_height:
            h_frac = height / max_h
            offset_lon = shadow_dir_lon * h_frac
            offset_lat = shadow_dir_lat * h_frac
            shadow_coords = coords.copy()
            shadow_coords[:, 0] += offset_lon
            shadow_coords[:, 1] += offset_lat
            shadow_patches.append(mpatches.Polygon(shadow_coords, closed=True))

    # Render shadows first (below buildings)
    if shadow_patches:
        sc = PatchCollection(
            shadow_patches,
            facecolors=[(*shadow_color, shadow_alpha)] * len(shadow_patches),
            edgecolors="none",
            linewidths=0,
            zorder=2,
        )
        ax.add_collection(sc)

    # Render building outlines (no fill, thin edges)
    if building_patches:
        bc = PatchCollection(
            building_patches,
            facecolors="none",
            edgecolors=[building_edge_color] * len(building_patches),
            linewidths=building_edge_width,
            zorder=3,
        )
        ax.add_collection(bc)

    n_shadows = len(shadow_patches)
    logger.info(f"  Added {len(building_patches)} buildings, "
                f"{n_shadows} shadows (height >= {min_shadow_height} ft)")


# ---------------------------------------------------------------------------
# Bridges
# ---------------------------------------------------------------------------

def add_bridges(ax, bridge_labels=False):
    """Draw side-elevation bridge profiles for NYC bridges."""
    bridges = [
        dict(name="BROOKLYN\nBRIDGE",
             start=[-74.0000, 40.7084], end=[-73.9940, 40.7037],
             tower_fracs=[0.27, 0.73],
             color="#8C7A6B", label_color="#6B5D4F"),
        dict(name="MANHATTAN\nBRIDGE",
             start=[-73.9921, 40.7101], end=[-73.9888, 40.7043],
             tower_fracs=[0.24, 0.76],
             color="#B8AFA3", label_color="#8C8578"),
        dict(name="WILLIAMSBURG\nBRIDGE",
             start=[-73.9761, 40.7166], end=[-73.9670, 40.7158],
             tower_fracs=[0.21, 0.79],
             color="#B8AFA3", label_color="#8C8578"),
        dict(name="QUEENSBORO\nBRIDGE",
             start=[-73.9583, 40.7590], end=[-73.9514, 40.7542],
             tower_fracs=[0.25, 0.75],
             color="#B8AFA3", label_color="#8C8578"),
        dict(name="GEORGE WASHINGTON\nBRIDGE",
             start=[-73.9464, 40.8508], end=[-73.9594, 40.8541],
             tower_fracs=[0.13, 0.87],
             color="#B8AFA3", label_color="#8C8578"),
        dict(name="RFK BRIDGE",
             start=[-73.9295, 40.8012], end=[-73.9263, 40.8006],
             tower_fracs=[0.25, 0.75],
             color="#B8AFA3", label_color="#8C8578"),
        dict(name="HENRY HUDSON\nBRIDGE",
             start=[-73.9229, 40.8773], end=[-73.9219, 40.8779],
             tower_fracs=[0.25, 0.75],
             color="#B8AFA3", label_color="#8C8578"),
    ]
    draw_bridges(ax, bridges, bridge_labels=bridge_labels)


# ---------------------------------------------------------------------------
# Tunnels
# ---------------------------------------------------------------------------

def add_tunnels(ax, dem_path="data/manhattan_dem.tif",
                tunnels_file="data/manhattan_tunnels.json"):
    """Draw tunnel routes as dashed lines with portal markers.

    Reads centerline routes from GeoJSON, uses the DEM to find shoreline
    crossings (where the tunnel goes underwater), and draws:
      - dashed line for the underwater/underground portion
      - filled circles at the portal entrances
      - labels

    All visual sizes derive from the axes extent and figure dimensions
    so scaling works automatically.
    """
    import rasterio

    fig = ax.get_figure()
    fig_h = fig.get_size_inches()[1]
    ylim = ax.get_ylim()
    deg_per_inch = (ylim[1] - ylim[0]) / fig_h
    lw_scale = fig_h / 18.0

    # Load DEM for shoreline detection
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        bounds = src.bounds

    def _elev_at(lon, lat):
        """Sample DEM elevation at a geographic coordinate."""
        if not (bounds.left <= lon <= bounds.right and
                bounds.bottom <= lat <= bounds.top):
            return -9999
        col = int((lon - bounds.left) / (bounds.right - bounds.left) * elev.shape[1])
        row = int((bounds.top - lat) / (bounds.top - bounds.bottom) * elev.shape[0])
        col = min(col, elev.shape[1] - 1)
        row = min(row, elev.shape[0] - 1)
        return elev[row, col]

    def _find_portal(coords, threshold=1.0):
        """Walk along coords and find the first point where elevation > threshold.

        Returns the index of the portal point (first land point from this end).
        """
        for i, (lon, lat) in enumerate(coords):
            if _elev_at(lon, lat) > threshold:
                return i
        return 0

    # Load tunnel centerlines (take the short approximations)
    with open(tunnels_file, "r") as f:
        data = json.load(f)

    # Use only the short centerline approximations (< 15 points)
    tunnel_defs = [
        ("HOLLAND\nTUNNEL", "#8C7A6B"),
        ("LINCOLN\nTUNNEL", "#8C7A6B"),
        ("QUEENS-MIDTOWN\nTUNNEL", "#8C7A6B"),
        ("BROOKLYN-BATTERY\nTUNNEL", "#8C7A6B"),
    ]
    centerlines = []
    seen_names = set()
    for feat in data["features"]:
        coords = feat["geometry"]["coordinates"]
        name = feat["properties"].get("name", "")
        # Take only the short centerline versions
        if len(coords) < 15 and name not in seen_names:
            centerlines.append((name, coords))
            seen_names.add(name)

    zz = 9
    portal_size = max(2.0, 3.5 * lw_scale)

    for (raw_name, coords), (label, color) in zip(centerlines, tunnel_defs):
        coords = [list(c) for c in coords]

        # Find portal indices (where tunnel enters land from each end)
        portal_start = _find_portal(coords)
        portal_end = len(coords) - 1 - _find_portal(list(reversed(coords)))

        # Ensure we have a valid segment
        if portal_start >= portal_end:
            portal_start = 0
            portal_end = len(coords) - 1

        # Extract the segment between portals (plus a few points of land overlap)
        i0 = max(0, portal_start - 1)
        i1 = min(len(coords) - 1, portal_end + 1)
        segment = coords[i0:i1 + 1]

        if len(segment) < 2:
            continue

        xs = [p[0] for p in segment]
        ys = [p[1] for p in segment]

        # Dashed line for the tunnel route
        ax.plot(xs, ys, color=color, lw=1.4 * lw_scale,
                linestyle=(0, (4, 3)), zorder=zz, alpha=0.7,
                solid_capstyle="round", dash_capstyle="round")

        # Portal markers (filled circles at each end)
        for px, py in [(xs[0], ys[0]), (xs[-1], ys[-1])]:
            ax.plot(px, py, 'o', color=color, markersize=portal_size,
                    zorder=zz + 1, markeredgecolor="white",
                    markeredgewidth=0.4 * lw_scale)

        # Find the underwater portion of the segment and label at its center
        water_pts = [(x, y) for x, y in zip(xs, ys)
                     if _elev_at(x, y) <= 1.0]
        if water_pts:
            # Midpoint of underwater section
            wi = len(water_pts) // 2
            mx, my = water_pts[wi]
            # Direction from neighboring water points for rotation
            w0 = max(0, wi - 1)
            w1 = min(len(water_pts) - 1, wi + 1)
            dx = water_pts[w1][0] - water_pts[w0][0]
            dy = water_pts[w1][1] - water_pts[w0][1]
        else:
            # Fallback: midpoint of full segment
            mid_idx = len(segment) // 2
            mx, my = segment[mid_idx]
            dx = xs[-1] - xs[0]
            dy = ys[-1] - ys[0]
        angle = np.degrees(np.arctan2(dy, dx))
        # Keep text readable (not upside down)
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        tunnel_length = ((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2) ** 0.5
        bridge_inches = tunnel_length / deg_per_inch
        fontsize = max(3.5, min(5.0, 3.5 + 1.5 * bridge_inches)) * lw_scale

        ax.text(mx, my, label,
                ha="center", va="center",
                fontsize=fontsize, color=color, rotation=angle,
                fontweight="bold", fontfamily="sans-serif",
                fontstyle="italic",
                path_effects=[pe.withStroke(linewidth=1.5 * lw_scale,
                                            foreground="white")],
                zorder=zz + 5, linespacing=1.1)

    logger.info(f"  Added {len(centerlines)} tunnels")


# ---------------------------------------------------------------------------
# Geographic labels
# ---------------------------------------------------------------------------

def add_geographic_labels(ax):
    """Add labels for surrounding boroughs and NJ."""
    label_style = dict(
        ha="center", va="center", fontweight="bold", fontfamily="sans-serif",
        color="#6a6a6a",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=8,
    )
    ax.text(-73.940, 40.700, "BROOKLYN", fontsize=12, **label_style)
    ax.text(-73.920, 40.760, "QUEENS", fontsize=12, **label_style)
    ax.text(-73.915, 40.845, "THE BRONX", fontsize=12, **label_style)
    ax.text(-74.043, 40.780, "NEW\nJERSEY", fontsize=12,
            linespacing=1.1, **label_style)
    ax.text(-74.030, 40.836, "FORT LEE", fontsize=6, **label_style)
    ax.text(-73.9435, 40.7692, "ROOSEVELT\nISLAND", fontsize=4,
            linespacing=1.1, rotation=61, **label_style)

    # White border between Brooklyn and Queens (approximate Newtown Creek + boundary)
    bk_q_border = np.array([
        [-73.961, 40.7305], [-73.952, 40.730], [-73.945, 40.728],
        [-73.938, 40.727], [-73.932, 40.726], [-73.928, 40.723],
        [-73.922, 40.718], [-73.916, 40.713], [-73.910, 40.709],
        [-73.904, 40.705], [-73.900, 40.702],
    ])
    ax.plot(bk_q_border[:, 0], bk_q_border[:, 1],
            color="white", linewidth=1.2, zorder=3, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Main poster composition
# ---------------------------------------------------------------------------

from constants import PIXELS_PER_DEGREE


def make_poster(
    neighborhoods_file: str = "data/manhattan_neighborhoods.json",
    dem_path: str = "data/manhattan_dem.tif",
    save_path: str = "images/manhattan_poster.tiff",
    figsize=(14.4, 18.0),
    dpi: int = 600,
    scale_factor: float = 0.2,
    center_lon: float = -73.98,
    center_lat: float = 40.78,
    palette: str = "earthy",
    max_colors: int = 5,
    neighborhood_labels: bool = True,
    park_color: str = "#1e7a2a",
    park_alpha: float = 0.75,
    park_labels: bool = False,
    elevation_file: str = "data/manhattan_elevation.json",
    show_minor_contours: bool = False,
    show_major_contours: bool = False,
    contour_interval: float = 10.0,
    contour_major_interval: float = 50.0,
    contour_color: str = None,
    contour_linewidth: float = 0.35,
    contour_major_linewidth: float = 0.8,
    contour_alpha: float = 0.45,
    show_zoning: bool = False,
    show_land_use: bool = False,
    show_buildings: bool = False,
    show_zoning_special: bool = False,
    show_zoning_commercial: bool = False,
    zoning_alpha: float = 0.8,
    zone_colors: dict = None,
    bridge_labels: bool = False,
    show_zoning_legend: bool = True,
    zoning_legend_loc: str = "upper left",
    show_title: bool = True,
    show_attribution: bool = True,
    save_full: bool = False,
):
    """Generate the full Manhattan poster and save to disk."""
    # --- Compute view extent from parameters ---
    extent_w = figsize[0] * dpi * scale_factor / PIXELS_PER_DEGREE
    extent_h = figsize[1] * dpi * scale_factor / PIXELS_PER_DEGREE
    view_xmin = center_lon - extent_w / 2
    view_xmax = center_lon + extent_w / 2
    view_ymin = center_lat - extent_h / 2
    view_ymax = center_lat + extent_h / 2
    logger.info(f"View extent: ({view_xmin:.3f}, {view_ymin:.3f}) "
                f"to ({view_xmax:.3f}, {view_ymax:.3f}) "
                f"({extent_w:.3f} x {extent_h:.3f} deg)")

    logger.info("Loading data...")
    with open(neighborhoods_file, "r") as f:
        neighborhoods = parse_neighborhoods(json.load(f))
    logger.info(f"  {len(neighborhoods)} neighborhoods")

    from shapely.ops import unary_union

    names, polys, tree = build_neighborhood_index(neighborhoods)
    city_boundary = unary_union(polys)
    if palette == "transparent":
        color_map = None
    else:
        color_map, _ = get_neighborhood_color_map(neighborhoods, max_colors, palette=palette)

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Set view extent from computed parameters
    ax.set_xlim(view_xmin, view_xmax)
    ax.set_ylim(view_ymin, view_ymax)

    # --- Layer 1: Hillshade ---
    # Color source: land use or zoning replaces neighborhood coloring when enabled.
    zoning_cats = []
    legend_title = "Zoning Districts"
    if show_land_use:
        color_source, zoning_cats = load_nyc_land_use_color_source(
            zone_colors=zone_colors)
        legend_title = "Land Use"
    elif show_zoning:
        color_source, zoning_cats = load_nyc_zoning_color_source(
            zone_colors=zone_colors)
    else:
        color_source = None

    add_hillshade(
        ax, dem_path, names=names, polys=polys,
        neighborhood_color_map=color_map if not color_source else None,
        color_source=color_source,
        color_source_clip=city_boundary if color_source else None,
        color_source_alpha=zoning_alpha,
        saturation=0.65, z_factor=6.0,
    )

    if zoning_cats and show_zoning_legend:
        add_zoning_legend(ax, zoning_cats, zone_colors,
                          loc=zoning_legend_loc, title=legend_title)

    # Additional overlays (semi-transparent, on top of shaded zoning)
    if show_zoning_special:
        add_nyc_special_purpose(ax, alpha=zoning_alpha * 0.5)
    if show_zoning_commercial:
        add_nyc_commercial_overlay(ax, alpha=zoning_alpha * 0.5)

    # --- Layer 2: Neighborhood borders ---
    border_patches = []
    for n in neighborhoods:
        for polygon_coords in n.geometry.coordinates:
            for ring_coords in polygon_coords:
                border_patches.append(mpatches.Polygon(np.array(ring_coords), closed=True))
    ax.add_collection(PatchCollection(border_patches, facecolors='none',
                                      edgecolors='white', linewidths=0.6,
                                      alpha=0.5, zorder=2))

    # --- Layer 3: Elevation contours ---
    if show_minor_contours or show_major_contours:
        try:
            elevation_data = load_elevation_from_file(elevation_file)
            if color_source:
                color_idx = build_contour_color_index_from_color_source(color_source)
            elif color_map is not None:
                color_idx = build_contour_color_index_from_neighborhoods(
                    names, polys, tree, color_map)
            else:
                color_idx = None
            if color_idx is not None:
                add_contour_lines(
                    ax, elevation_data, color_idx,
                    contour_interval=contour_interval,
                    major_interval=contour_major_interval,
                    linewidth=contour_linewidth,
                    major_linewidth=contour_major_linewidth,
                    alpha=contour_alpha,
                    show_minor=show_minor_contours,
                    show_major=show_major_contours,
                    color_override=contour_color,
                )
        except FileNotFoundError:
            logger.warning(f"Elevation data not found at {elevation_file} — skipping contours")

    # --- Layer 3b: Buildings with isometric shadows ---
    if show_buildings:
        add_buildings(ax, city_boundary=city_boundary)

    # --- Neighborhood labels ---
    if neighborhood_labels:
        font_scale = figsize[0] / 16.0
        dummy_colors = list(range(len(neighborhoods)))
        add_smart_neighborhood_labels(ax, neighborhoods, dummy_colors,
                                      lambda x: (0.5, 0.5, 0.5), font_scale=font_scale)

    # --- Parks ---
    add_parks(ax, city_boundary=city_boundary,
              park_color=park_color, park_alpha=park_alpha,
              park_labels=park_labels)

    # --- Geographic labels ---
    add_geographic_labels(ax)

    # --- Bridges and tunnels ---
    add_bridges(ax, bridge_labels=bridge_labels)
    add_tunnels(ax, dem_path=dem_path)

    # --- Title and attribution ---
    if show_title:
        ax.text(
            0.5, 0.97,
            "M A N H A T T A N",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=52, fontweight="bold", fontfamily="sans-serif",
            color="#2a2218", zorder=10,
        )
    if show_attribution:
        ax.text(
            0.975, 0.025,
            "DEM: USGS 10 m  |  Neighborhoods: NYC Open Data",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8, color="#9a9080", fontfamily="sans-serif", zorder=10,
        )

    # --- Save ---
    if save_full:
        logger.info(f"Saving full-resolution to {save_path} at {dpi} DPI...")
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    transparent=False, pad_inches=0,
                    pil_kwargs={'compression': 'tiff_lzw'})
        logger.info(f"Done — {save_path}")

    preview_path = save_path.rsplit(".", 1)[0] + "_preview.png"
    logger.info(f"Saving preview to {preview_path} at 200 DPI...")
    fig.savefig(preview_path, dpi=200, facecolor=fig.get_facecolor(),
                transparent=False, pad_inches=0)
    logger.info(f"Done — {preview_path}")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Manhattan poster map")
    parser.add_argument("--width", type=float, default=14.4,
                        help="Figure width in inches (default: 14.4)")
    parser.add_argument("--height", type=float, default=18.0,
                        help="Figure height in inches (default: 18.0)")
    parser.add_argument("--dpi", type=int, default=600,
                        help="Output DPI (default: 600)")
    parser.add_argument("--scale-factor", type=float, default=0.2,
                        help="DEM-to-figure pixel ratio (default: 0.2)")
    parser.add_argument("--center-lon", type=float, default=-73.98,
                        help="Map center longitude (default: -73.98)")
    parser.add_argument("--center-lat", type=float, default=40.78,
                        help="Map center latitude (default: 40.78)")
    parser.add_argument("--palette", type=str, default="earthy",
                        help="Color palette (default: earthy)")
    parser.add_argument("--max-colors", type=int, default=5,
                        help="Max colors for graph coloring (default: 5)")
    parser.add_argument("--no-neighborhood-labels", action="store_true", default=False,
                        help="Hide neighborhood labels")
    parser.add_argument("--park-color", type=str, default="#1e7a2a",
                        help="Park fill color as hex (default: #1e7a2a)")
    parser.add_argument("--park-alpha", type=float, default=0.75,
                        help="Park overlay opacity (default: 0.75)")
    parser.add_argument("--park-labels", action="store_true", default=False,
                        help="Show park labels")
    parser.add_argument("--bridge-labels", action="store_true", default=False,
                        help="Show bridge labels")
    # Contour arguments
    parser.add_argument("--minor-contours", action="store_true", default=False,
                        dest="minor_contours",
                        help="Enable minor contour lines (default: disabled)")
    parser.add_argument("--no-minor-contours", action="store_false", dest="minor_contours",
                        help="Disable minor contour lines")
    parser.add_argument("--major-contours", action="store_true", default=False,
                        dest="major_contours",
                        help="Enable major contour lines (default: disabled)")
    parser.add_argument("--no-major-contours", action="store_false", dest="major_contours",
                        help="Disable major contour lines")
    parser.add_argument("--contour-interval", type=float, default=10.0,
                        help="Meters between minor contour lines (default: 10)")
    parser.add_argument("--contour-major-interval", type=float, default=50.0,
                        help="Meters for major contours (default: 50)")
    parser.add_argument("--contour-color", type=str, default=None,
                        help="Single contour color override hex (default: adapt to color map)")
    parser.add_argument("--contour-linewidth", type=float, default=0.35,
                        help="Minor contour linewidth (default: 0.35)")
    parser.add_argument("--contour-major-linewidth", type=float, default=0.8,
                        help="Major contour linewidth (default: 0.8)")
    parser.add_argument("--contour-alpha", type=float, default=0.45,
                        help="Contour opacity (default: 0.45)")
    # Zoning arguments
    parser.add_argument("--show-zoning", action="store_true", default=False,
                        help="Enable zoning districts overlay")
    parser.add_argument("--show-land-use", action="store_true", default=False,
                        help="Enable parcel-level land use overlay (MapPLUTO)")
    parser.add_argument("--show-buildings", action="store_true", default=False,
                        help="Enable building footprints with isometric shadows")
    parser.add_argument("--show-zoning-special", action="store_true", default=False,
                        help="Enable special purpose districts overlay")
    parser.add_argument("--show-zoning-commercial", action="store_true", default=False,
                        help="Enable commercial overlays")
    parser.add_argument("--zoning-alpha", type=float, default=0.8,
                        help="Zoning overlay opacity (default: 0.8)")
    parser.add_argument("--no-zoning-legend", action="store_true", default=False,
                        help="Hide zoning legend")
    parser.add_argument("--zoning-legend-loc", type=str, default="upper left",
                        help="Zoning legend location (default: 'upper left')")
    parser.add_argument("--zoning-residential-color", type=str, default="#C06830")
    parser.add_argument("--zoning-commercial-color", type=str, default="#1E7898")
    parser.add_argument("--zoning-industrial-color", type=str, default="#786050")
    parser.add_argument("--zoning-mixed-use-color", type=str, default="#C09868")
    parser.add_argument("--zoning-public-color", type=str, default="#488A44")
    parser.add_argument("--zoning-park-color", type=str, default="#488A44")
    parser.add_argument("--zoning-other-color", type=str, default="#B8AFA3")
    parser.add_argument("--no-title", action="store_true", default=False,
                        help="Hide title text")
    parser.add_argument("--no-attribution", action="store_true", default=False,
                        help="Hide attribution text")
    parser.add_argument("--print", dest="save_full", action="store_true", default=False,
                        help="Save full-resolution TIFF for printing (preview only by default)")
    args = parser.parse_args()

    zone_colors = {
        ZoneCategory.RESIDENTIAL: args.zoning_residential_color,
        ZoneCategory.COMMERCIAL:  args.zoning_commercial_color,
        ZoneCategory.INDUSTRIAL:  args.zoning_industrial_color,
        ZoneCategory.MIXED_USE:   args.zoning_mixed_use_color,
        ZoneCategory.PUBLIC:      args.zoning_public_color,
        ZoneCategory.PARK:        args.zoning_park_color,
        ZoneCategory.OTHER:       args.zoning_other_color,
    }

    make_poster(
        figsize=(args.width, args.height),
        dpi=args.dpi,
        scale_factor=args.scale_factor,
        center_lon=args.center_lon,
        center_lat=args.center_lat,
        palette=args.palette,
        max_colors=args.max_colors,
        neighborhood_labels=not args.no_neighborhood_labels,
        park_color=args.park_color,
        park_alpha=args.park_alpha,
        park_labels=args.park_labels,
        bridge_labels=args.bridge_labels,
        show_minor_contours=args.minor_contours,
        show_major_contours=args.major_contours,
        contour_interval=args.contour_interval,
        contour_major_interval=args.contour_major_interval,
        contour_color=args.contour_color,
        contour_linewidth=args.contour_linewidth,
        contour_major_linewidth=args.contour_major_linewidth,
        contour_alpha=args.contour_alpha,
        show_zoning=args.show_zoning,
        show_land_use=args.show_land_use,
        show_buildings=args.show_buildings,
        show_zoning_special=args.show_zoning_special,
        show_zoning_commercial=args.show_zoning_commercial,
        zoning_alpha=args.zoning_alpha,
        zone_colors=zone_colors,
        show_zoning_legend=not args.no_zoning_legend,
        zoning_legend_loc=args.zoning_legend_loc,
        show_title=not args.no_title,
        show_attribution=not args.no_attribution,
        save_full=args.save_full,
    )
