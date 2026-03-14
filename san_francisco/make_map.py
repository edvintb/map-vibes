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

Usage:
    python make_map.py                           # defaults: 18x18 in @ 600 DPI
    python make_map.py --width 24 --height 24    # larger poster
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
from process_elevation import (
    load_elevation_from_file, add_contour_lines,
    build_contour_color_index_from_neighborhoods,
    build_contour_color_index_from_color_source,
)
from process_terrain import add_hillshade
from process_zoning import (
    ZoneCategory, DEFAULT_ZONE_COLORS,
    load_sf_zoning_color_source, load_sf_land_use_color_source,
    add_zoning_legend,
)
from draw_bridges import draw_bridges
from colors import darken_hex as _darken, BG_COLOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# ---------------------------------------------------------------------------
# Parks
# ---------------------------------------------------------------------------

def add_parks(
    ax,
    parks_file: str = "data/sf_park.json",
    city_boundary=None,
    min_acres_for_label: float = 3.0,
    park_color: str = "#1e7a2a",
    park_alpha: float = 0.75,
    park_labels: bool = True,
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
    skip_label_only = ("Lower Great Highway", "Yacht Harbor")

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

    # Derive label color by darkening fill further
    label_color = f"#{int(r*0.5):02x}{int(g*0.5):02x}{int(b*0.5):02x}"

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
            color=label_color, fontweight="bold", fontfamily="sans-serif",
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
# Geographic labels (Marin County, San Mateo County, islands)
# ---------------------------------------------------------------------------

def add_geographic_labels(ax):
    """Add labels for surrounding counties, islands, and peninsulas."""
    label_style = dict(
        ha="center", va="center", fontweight="bold", fontfamily="sans-serif",
        color="#6a6a6a",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=8,
    )
    ax.text(-122.513, 37.862, "MARIN COUNTY", fontsize=9,
            ha="left", **{k: v for k, v in label_style.items() if k != "ha"})
    ax.text(-122.447, 37.705, "SAN MATEO COUNTY", fontsize=9, **label_style)
    ax.text(-122.4204, 37.8284, "ALCATRAZ\nISLAND", fontsize=6,
            linespacing=1.1, **label_style)
    ax.text(-122.431, 37.862, "ANGEL\nISLAND", fontsize=7,
            linespacing=1.1, **label_style)
    ax.text(-122.463, 37.865, "TIBURON", fontsize=6, **label_style)


# ---------------------------------------------------------------------------
# Bridges (side-elevation profiles)
# ---------------------------------------------------------------------------

def add_bridges(ax, golden_gate_color="#B5564E", bay_bridge_color="#B8AFA3",
                bridge_labels=False):
    """Draw side-elevation bridge profiles for Golden Gate and Bay Bridge."""
    bridges = [
        dict(name="GOLDEN GATE\nBRIDGE",
             start=[-122.47700, 37.81050], end=[-122.47970, 37.82630],
             tower_fracs=[0.175, 0.825],
             color=golden_gate_color, label_color=_darken(golden_gate_color, 0.65),
             label_side=-1, label_offset=(-0.003, 0.0)),
        dict(name="BAY BRIDGE",
             start=[-122.38800, 37.78870], end=[-122.36700, 37.80830],
             tower_fracs=[0.167, 0.50, 0.833],
             color=bay_bridge_color, label_color=_darken(bay_bridge_color, 0.65),
             label_side=-1),
    ]
    draw_bridges(ax, bridges, bridge_labels=bridge_labels)


# ---------------------------------------------------------------------------
# Main poster composition
# ---------------------------------------------------------------------------

from constants import PIXELS_PER_DEGREE


def make_poster(
    neighborhoods_file: str = "data/sf_neighborhoods.json",
    elevation_file: str = "data/sf_elevation.json",
    dem_path: str = "data/sf_dem_10m_marin.tif",
    save_path: str = "images/sf_combined_poster.tiff",
    figsize=(18, 18),
    dpi: int = 600,
    scale_factor: float = 0.17,
    center_lon: float = -122.435,
    center_lat: float = 37.785,
    palette: str = "earthy",
    max_colors: int = 8,
    neighborhood_labels: bool = True,
    park_color: str = "#1e7a2a",
    park_alpha: float = 0.75,
    park_labels: bool = False,
    golden_gate_color: str = "#B5564E",
    bay_bridge_color: str = "#B8AFA3",
    bridge_labels: bool = False,
    show_minor_contours: bool = True,
    show_major_contours: bool = True,
    contour_interval: float = 10.0,
    contour_major_interval: float = 50.0,
    contour_color: str = None,
    contour_linewidth: float = 0.55,
    contour_major_linewidth: float = 1.2,
    contour_alpha: float = 0.45,
    show_zoning: bool = False,
    show_land_use: bool = False,
    zoning_alpha: float = 0.8,
    zone_colors: dict = None,
    show_zoning_legend: bool = True,
    zoning_legend_loc: str = "upper left",
    show_title: bool = True,
    show_attribution: bool = True,
    save_full: bool = False,
):
    """Generate the full San Francisco poster and save to disk.

    Composes all layers (terrain, neighborhoods, contours, parks, bridges)
    into a poster figure, adds title and attribution, then saves both
    a full-resolution TIFF and a 200 DPI PNG preview.
    """
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

    # --- Load data ---
    logger.info("Loading data...")
    with open(neighborhoods_file, "r") as f:
        neighborhoods = parse_neighborhoods(json.load(f))
    logger.info(f"  {len(neighborhoods)} neighborhoods")

    elevation_data = load_elevation_from_file(elevation_file)
    logger.info(f"  {len(elevation_data.isolines)} elevation isolines")

    # --- Build spatial data ---
    from shapely.ops import unary_union

    names, polys, tree = build_neighborhood_index(neighborhoods)
    city_boundary = unary_union(polys)
    if palette == "transparent":
        color_map = None
    else:
        color_map, _ = get_neighborhood_color_map(neighborhoods, max_colors, palette=palette)

    # --- Create figure ---
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
    # Color source: zoning/land-use replaces neighborhood coloring when enabled.
    zoning_cats = []
    if show_zoning:
        color_source, zoning_cats = load_sf_zoning_color_source(
            zone_colors=zone_colors)
    elif show_land_use:
        color_source, zoning_cats = load_sf_land_use_color_source(
            zone_colors=zone_colors)
    else:
        color_source = None

    add_hillshade(
        ax, dem_path, names=names, polys=polys,
        neighborhood_color_map=color_map if not color_source else None,
        color_source=color_source,
        color_source_clip=city_boundary if color_source else None,
        color_source_alpha=zoning_alpha,
        saturation=0.65,
    )

    if zoning_cats and show_zoning_legend:
        legend_title = "Zoning Districts" if show_zoning else "Land Use"
        add_zoning_legend(ax, zoning_cats, zone_colors,
                          loc=zoning_legend_loc, title=legend_title)

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
    if (show_minor_contours or show_major_contours) and elevation_data:
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
                clip_boundary=city_boundary,
            )

    # --- Layer 4: Neighborhood labels ---
    if neighborhood_labels:
        font_scale = figsize[0] / 16.0
        dummy_colors = list(range(len(neighborhoods)))
        add_smart_neighborhood_labels(ax, neighborhoods, dummy_colors,
                                      lambda x: (0.5, 0.5, 0.5), font_scale=font_scale)

    # --- Parks ---
    add_parks(ax, city_boundary=city_boundary,
              park_color=park_color, park_alpha=park_alpha,
              park_labels=park_labels)

    # --- Geographic labels for surrounding areas ---
    add_geographic_labels(ax)

    # --- Layer 4: Bridges ---
    # "neutral" maps to the flat-surface hillshade gray
    hillshade_neutral = "#CAC8C6"
    if golden_gate_color == "neutral":
        golden_gate_color = hillshade_neutral
    if bay_bridge_color == "neutral":
        bay_bridge_color = hillshade_neutral
    add_bridges(ax, golden_gate_color=golden_gate_color,
                bay_bridge_color=bay_bridge_color,
                bridge_labels=bridge_labels)

    # --- Manual park labels (rotated to match park orientation) ---
    # These parks are too narrow for auto-labeling to place well.
    if park_labels:
        r, g, b = int(park_color[1:3], 16), int(park_color[3:5], 16), int(park_color[5:7], 16)
        manual_label_color = f"#{int(r*0.5):02x}{int(g*0.5):02x}{int(b*0.5):02x}"
        park_label_style = dict(
            ha="center", va="center",
            color=manual_label_color, fontweight="bold", fontfamily="sans-serif",
            fontstyle="italic", zorder=8,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )
        # Park Presidio Blvd: long N-S strip between Inner/Outer Richmond
        ax.text(-122.4735, 37.7803, "PARK PRESIDIO BLVD",
                fontsize=5, rotation=93.5, **park_label_style)
        # Lower Great Highway: long N-S strip along the west coast
        ax.text(-122.5107, 37.7597, "LOWER GREAT HIGHWAY",
                fontsize=7, rotation=96, **park_label_style)
        # Yacht Harbor and Marina Green: label placed north, off the coast
        ax.text(-122.4412, 37.8108, "YACHT HARBOR\n& MARINA GREEN",
                fontsize=5.5, **park_label_style)

    # --- Title and attribution ---
    if show_title:
        ax.text(
            0.5, 0.85,
            "S A N   F R A N C I S C O",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=64, fontweight="bold", fontfamily="sans-serif",
            color="#2a2218", zorder=10,
        )
    if show_attribution:
        ax.text(
            0.975, 0.025,
            "DEM: USGS 10 m\nContours & Neighborhoods: SF Open Data",
            transform=ax.transAxes,
            ha="right", va="bottom", multialignment="right",
            fontsize=9, color="#9a9080", fontfamily="sans-serif", zorder=10,
        )

    # --- Save outputs ---
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
    parser = argparse.ArgumentParser(description="Generate SF poster map")
    parser.add_argument("--width", type=float, default=18.0,
                        help="Figure width in inches (default: 18.0)")
    parser.add_argument("--height", type=float, default=18.0,
                        help="Figure height in inches (default: 18.0)")
    parser.add_argument("--dpi", type=int, default=600,
                        help="Output DPI (default: 600)")
    parser.add_argument("--scale-factor", type=float, default=0.17,
                        help="DEM-to-figure pixel ratio (default: 0.17)")
    parser.add_argument("--center-lon", type=float, default=-122.435,
                        help="Map center longitude (default: -122.435)")
    parser.add_argument("--center-lat", type=float, default=37.785,
                        help="Map center latitude (default: 37.785)")
    parser.add_argument("--palette", type=str, default="earthy",
                        help="Color palette (default: earthy)")
    parser.add_argument("--max-colors", type=int, default=8,
                        help="Max colors for graph coloring (default: 8)")
    parser.add_argument("--no-neighborhood-labels", action="store_true", default=False,
                        help="Hide neighborhood labels")
    parser.add_argument("--park-color", type=str, default="#1e7a2a",
                        help="Park fill color as hex (default: #1e7a2a)")
    parser.add_argument("--park-alpha", type=float, default=0.75,
                        help="Park overlay opacity (default: 0.75)")
    parser.add_argument("--park-labels", action="store_true", default=False,
                        help="Show park labels")
    parser.add_argument("--golden-gate-color", type=str, default="#B5564E",
                        help="Golden Gate Bridge color as hex (default: #B5564E)")
    parser.add_argument("--bay-bridge-color", type=str, default="#B8AFA3",
                        help="Bay Bridge color as hex (default: #B8AFA3)")
    parser.add_argument("--bridge-labels", action="store_true", default=False,
                        help="Show bridge labels")
    # Contour arguments
    parser.add_argument("--minor-contours", action="store_true", default=True,
                        dest="minor_contours",
                        help="Enable minor contour lines (default: enabled)")
    parser.add_argument("--no-minor-contours", action="store_false", dest="minor_contours",
                        help="Disable minor contour lines")
    parser.add_argument("--major-contours", action="store_true", default=True,
                        dest="major_contours",
                        help="Enable major contour lines (default: enabled)")
    parser.add_argument("--no-major-contours", action="store_false", dest="major_contours",
                        help="Disable major contour lines")
    parser.add_argument("--contour-interval", type=float, default=10.0,
                        help="Meters between minor contour lines (default: 10)")
    parser.add_argument("--contour-major-interval", type=float, default=50.0,
                        help="Meters for major contours (default: 50)")
    parser.add_argument("--contour-color", type=str, default=None,
                        help="Single contour color override hex (default: adapt to color map)")
    parser.add_argument("--contour-linewidth", type=float, default=0.55,
                        help="Minor contour linewidth (default: 0.55)")
    parser.add_argument("--contour-major-linewidth", type=float, default=1.2,
                        help="Major contour linewidth (default: 1.2)")
    parser.add_argument("--contour-alpha", type=float, default=0.45,
                        help="Contour opacity (default: 0.45)")
    # Zoning arguments
    parser.add_argument("--show-zoning", action="store_true", default=False,
                        help="Enable zoning districts overlay")
    parser.add_argument("--show-land-use", action="store_true", default=False,
                        help="Enable land use overlay")
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
        golden_gate_color=args.golden_gate_color,
        bay_bridge_color=args.bay_bridge_color,
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
        zoning_alpha=args.zoning_alpha,
        zone_colors=zone_colors,
        show_zoning_legend=not args.no_zoning_legend,
        zoning_legend_loc=args.zoning_legend_loc,
        show_title=not args.no_title,
        show_attribution=not args.no_attribution,
        save_full=args.save_full,
    )
