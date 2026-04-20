#!/usr/bin/env python3
"""
Topographic map of San Francisco: gray hillshade + neighborhood labels + bridges.

Same style as the surrounding terrain in make_map.py (Marin County look),
applied to the entire DEM including SF proper. No neighborhood colors,
no elevation isolines, no parks.

Layers (bottom to top):
  1. Gray hillshade from 10 m DEM (all land, ocean transparent)
  2. Neighborhood borders (background-colored lines between neighborhoods)
  3. Bridges — side-elevation profiles (Golden Gate + Bay Bridge)
  4. Neighborhood labels, geographic labels, title, attribution

Output: 18x18 inch poster at 600 DPI (TIFF) + 200 DPI preview (PNG).
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

import matplotlib

matplotlib.use("Agg")
import json
import logging

import matplotlib.pyplot as plt
from make_map import add_bridges, add_geographic_labels

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
    neighborhoods_file: str = "data/sf_neighborhoods.json",
    dem_path: str = "data/sf_dem_10m_marin.tif",
    save_path: str = "images/sf_topo_poster.tiff",
    figsize=(18, 18),
    dpi: int = 600,
):
    """Generate a gray-hillshade topographic poster of San Francisco."""

    logger.info("Loading data...")
    with open(neighborhoods_file) as f:
        neighborhoods = parse_neighborhoods(json.load(f))

    # --- Figure and axes ---
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

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
        0.5, 0.85,
        "S A N   F R A N C I S C O",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=64, fontweight="bold", fontfamily="sans-serif",
        color="#2a2218", zorder=10,
    )
    ax.text(
        0.98, 0.01,
        "DEM: USGS 10 m  |  Neighborhoods: SF Open Data",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, color="#9a9080", fontfamily="sans-serif", zorder=10,
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
