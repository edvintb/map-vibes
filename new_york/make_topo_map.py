#!/usr/bin/env python3
"""
Topographic map of Manhattan: gray hillshade + neighborhood labels + bridges.

Layers (bottom to top):
  1. Gray hillshade from USGS DEM (all land, ocean transparent)
  2. Neighborhood borders (background-colored lines)
  3. Bridges
  4. Neighborhood labels, geographic labels, title, attribution

Output: poster at 600 DPI (TIFF) + 200 DPI preview (PNG).
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import json
import logging
import numpy as np

from process_neighborhoods import (
    parse_neighborhoods,
    build_neighborhood_index,
    add_smart_neighborhood_labels,
)
from process_terrain import add_hillshade
from make_map import add_bridges, add_geographic_labels, DEM_BBOX, POSTER_WIDTH, POSTER_HEIGHT, DEG_PER_INCH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BG_COLOR = "#faf8f4"


def add_neighborhood_borders(ax, neighborhoods):
    """Draw neighborhood polygons with background-colored edges."""
    patches = []
    for neighborhood in neighborhoods:
        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                coords_array = np.array(ring_coords)
                patches.append(mpatches.Polygon(coords_array, closed=True))

    coll = PatchCollection(
        patches,
        facecolors="none",
        edgecolors=BG_COLOR,
        linewidths=1.2,
        zorder=2,
    )
    ax.add_collection(coll)
    logger.info(f"  Added {len(patches)} neighborhood border polygons")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_topo_map(
    neighborhoods_file: str = "data/manhattan_neighborhoods.json",
    dem_path: str = "data/manhattan_dem.tif",
    save_path: str = "images/manhattan_topo_poster.tiff",
    figsize=(POSTER_WIDTH, POSTER_HEIGHT),
    dpi: int = 600,
):
    """Generate a gray-hillshade topographic poster of Manhattan."""

    logger.info("Loading data...")
    with open(neighborhoods_file, "r") as f:
        neighborhoods = parse_neighborhoods(json.load(f))

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Set view to DEM bounding box
    ax.set_xlim(DEM_BBOX[0], DEM_BBOX[2])
    ax.set_ylim(DEM_BBOX[1], DEM_BBOX[3])

    # --- Map layers ---
    add_hillshade(ax, dem_path)
    add_neighborhood_borders(ax, neighborhoods)
    add_bridges(ax)

    logger.info("  Adding neighborhood labels...")
    font_scale = fig.get_size_inches()[0] / 16.0
    dummy_colors = list(range(len(neighborhoods)))
    add_smart_neighborhood_labels(
        ax, neighborhoods, dummy_colors, lambda x: (0.5, 0.5, 0.5),
        font_scale=font_scale,
    )
    add_geographic_labels(ax)

    # --- Title and attribution ---
    ax.text(
        0.5, 0.97,
        "M A N H A T T A N",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=52, fontweight="bold", fontfamily="sans-serif",
        color="#2a2218", zorder=10,
    )
    ax.text(
        0.98, 0.005,
        "DEM: USGS 10 m  |  Neighborhoods: NYC Open Data",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="#9a9080", fontfamily="sans-serif", zorder=10,
    )

    # --- Save ---
    logger.info(f"Saving to {save_path} at {dpi} DPI...")
    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
    logger.info(f"Done — {save_path}")

    preview_path = save_path.rsplit(".", 1)[0] + "_preview.png"
    logger.info(f"Saving preview to {preview_path} at 200 DPI...")
    fig.savefig(preview_path, dpi=200, facecolor=fig.get_facecolor())
    logger.info(f"Done — {preview_path}")

    return fig


if __name__ == "__main__":
    make_topo_map()
