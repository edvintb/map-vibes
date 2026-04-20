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

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

import matplotlib

matplotlib.use("Agg")
import json
import logging

import matplotlib.pyplot as plt
from make_map import PIXELS_PER_DEGREE, add_bridges, add_geographic_labels

from common.colors import BG_COLOR
from common.process_neighborhoods import (
    add_neighborhood_borders,
    add_smart_neighborhood_labels,
    parse_neighborhoods,
)
from common.process_terrain import add_hillshade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_topo_map(
    neighborhoods_file: str = "data/manhattan_neighborhoods.json",
    dem_path: str = "data/manhattan_dem.tif",
    save_path: str = "images/manhattan_topo_poster.tiff",
    figsize=(14.4, 18.0),
    dpi: int = 600,
    scale_factor: float = 0.2,
    center_lon: float = -73.98,
    center_lat: float = 40.78,
):
    """Generate a gray-hillshade topographic poster of Manhattan."""

    # Compute view extent (same logic as make_poster)
    extent_w = figsize[0] * dpi * scale_factor / PIXELS_PER_DEGREE
    extent_h = figsize[1] * dpi * scale_factor / PIXELS_PER_DEGREE
    view_xmin = center_lon - extent_w / 2
    view_xmax = center_lon + extent_w / 2
    view_ymin = center_lat - extent_h / 2
    view_ymax = center_lat + extent_h / 2

    logger.info("Loading data...")
    with open(neighborhoods_file) as f:
        neighborhoods = parse_neighborhoods(json.load(f))

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Set view to computed bounding box
    ax.set_xlim(view_xmin, view_xmax)
    ax.set_ylim(view_ymin, view_ymax)

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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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
