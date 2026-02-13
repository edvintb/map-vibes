#!/usr/bin/env python3
"""
Poster-quality hillshade + contour lines visualization of San Francisco.

Renders a high-resolution terrain map combining:
  - Hillshade relief from 10m DEM (Swiss-style blended)
  - Elevation-tinted hypsometric color wash
  - Contour lines at multiple intervals
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import rasterio
import shapely
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from shapely.ops import unary_union

from process_elevation import load_elevation_from_file
from process_neighborhoods import parse_neighborhoods
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_city_boundary(neighborhoods_file="data/sf_neighborhoods.json"):
    """Build a unified city boundary from neighborhood polygons."""
    from shapely.geometry import Polygon as ShapelyPolygon
    with open(neighborhoods_file, 'r') as f:
        neighborhoods = parse_neighborhoods(json.load(f))
    polys = []
    for n in neighborhoods:
        for polygon_coords in n.geometry.coordinates:
            for ring_coords in polygon_coords:
                if len(ring_coords) >= 3:
                    try:
                        p = ShapelyPolygon(ring_coords)
                        if p.is_valid:
                            polys.append(p)
                    except Exception:
                        continue
    return unary_union(polys)


def compute_hillshade_and_blend(dem_path, terrain_cmap, vmin, vmax,
                                 azimuth=315, altitude=45, z_factor=2.0):
    """
    Compute hillshade from DEM and blend it with the hypsometric tint using
    matplotlib's LightSource.shade for proper Swiss-style relief.

    Returns (blended_rgba, bounds, elevation_array, hillshade_array).
    """
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        bounds = src.bounds

    ls = LightSource(azdeg=azimuth, altdeg=altitude)

    # Normalize elevation for colormap
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    elev_normed = norm(elev)

    # Get the RGB from the terrain colormap
    terrain_rgb = terrain_cmap(elev_normed)[:, :, :3]

    # Compute gradient for hillshade
    dy, dx = np.gradient(elev * z_factor, 10.0)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    hillshade = (
        np.sin(alt_rad) * np.cos(slope) +
        np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    hillshade = np.clip(hillshade, 0, 1)

    # Blend: use LightSource's shade_rgb for nicer results
    blended = ls.shade_rgb(terrain_rgb, elev * z_factor, blend_mode='soft',
                           vert_exag=1.0, dx=10.0, dy=10.0)

    return blended, bounds, elev, hillshade


def mask_rgba_to_boundary(rgb_array, bounds, city_boundary):
    """Mask an RGB(A) raster to the city boundary — set outside pixels to NaN."""
    rows, cols = rgb_array.shape[:2]
    lon = np.linspace(bounds.left, bounds.right, cols)
    lat = np.linspace(bounds.top, bounds.bottom, rows)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    pts = shapely.points(lon_grid.ravel(), lat_grid.ravel())
    inside = shapely.contains(city_boundary, pts).reshape(rows, cols)

    # Create RGBA with alpha channel
    rgba = np.zeros((*rgb_array.shape[:2], 4), dtype=float)
    rgba[:, :, :3] = rgb_array[:, :, :3]
    rgba[:, :, 3] = inside.astype(float)
    return rgba


def build_terrain_colormap():
    """
    Swiss-style hypsometric tint: lowlands in rich greens,
    mid-elevations warm golden-brown, peaks bright tan.
    """
    return LinearSegmentedColormap.from_list('sf_terrain', [
        (0.00, '#4a7c59'),   # rich green (lowlands / parks)
        (0.12, '#6a9e5c'),   # bright green
        (0.25, '#8db86b'),   # light green
        (0.38, '#b8cc73'),   # yellow-green
        (0.50, '#ddd68a'),   # pale golden
        (0.62, '#d4b76a'),   # warm gold
        (0.74, '#c49552'),   # ochre
        (0.85, '#b07a42'),   # warm brown
        (0.94, '#946038'),   # deep sienna
        (1.00, '#7a4c30'),   # dark earth peak
    ])


def render_poster(dem_path="data/sf_dem_10m_marin.tif",
                  elevation_file="data/sf_elevation.json",
                  neighborhoods_file="data/sf_neighborhoods.json",
                  save_path="images/sf_hillshade_contours_poster.png",
                  figsize=(24, 30),
                  dpi=300):
    """Render the poster-quality hillshade + contours map."""

    logger.info("Loading data...")
    city_boundary = load_city_boundary(neighborhoods_file)
    elevation_data = load_elevation_from_file(elevation_file)
    logger.info(f"  {len(elevation_data.isolines)} contour isolines loaded")

    # Get elevation stats for normalization
    with rasterio.open(dem_path) as src:
        elev_raw = src.read(1).astype(float)
    # Use the full range but clip outliers
    valid = elev_raw[elev_raw > -999]
    vmin, vmax = np.percentile(valid, [1, 99])
    logger.info(f"  Elevation range for coloring: {vmin:.1f} – {vmax:.1f} m")

    terrain_cmap = build_terrain_colormap()

    # --- Compute blended hillshade + hypsometric tint ---
    logger.info("Computing hillshade + terrain blend...")
    blended, bounds, elev, hillshade = compute_hillshade_and_blend(
        dem_path, terrain_cmap, vmin, vmax,
        azimuth=315, altitude=42, z_factor=3.0
    )

    logger.info("Masking to city boundary...")
    rgba = mask_rgba_to_boundary(blended, bounds, city_boundary)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # --- Create figure ---
    bg_color = '#111214'
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Single blended terrain layer
    logger.info("Rendering terrain...")
    ax.imshow(rgba, extent=extent, origin='upper',
              interpolation='bilinear', zorder=1)

    # --- Contour lines ---
    logger.info("Preparing contour lines...")
    min_elev, max_elev = elevation_data.get_elevation_range()
    elev_span = max_elev - min_elev if max_elev > min_elev else 1.0

    major_interval = 50.0
    lines_major = []
    lines_minor = []
    colors_major = []
    colors_minor = []
    lw_major = []
    lw_minor = []

    for isoline in elevation_data.isolines:
        coords = isoline.coordinates
        if len(coords) < 2:
            continue
        ev = isoline.elevation_value
        if ev is None:
            continue

        t = max(0.0, min(1.0, (ev - min_elev) / elev_span))

        is_major = (abs(ev % major_interval) < 0.5) or \
                   (abs(ev % major_interval - major_interval) < 0.5)

        # Classic topo-map style: dark sienna/brown contour lines
        # Darker at low elevation (contrast on green), slightly lighter at peaks
        r = 0.35 + 0.08 * t
        g = 0.18 + 0.06 * t
        b = 0.10 + 0.04 * t
        color = (r, g, b)

        if is_major:
            lines_major.append(coords)
            colors_major.append(color)
            lw_major.append(0.8 + 0.5 * t)
        else:
            lines_minor.append(coords)
            colors_minor.append(color)
            lw_minor.append(0.25 + 0.15 * t)

    # Draw minor contours first (subtle)
    logger.info(f"  Drawing {len(lines_minor)} minor contour lines...")
    if lines_minor:
        lc_minor = LineCollection(lines_minor, colors=colors_minor,
                                  linewidths=lw_minor, alpha=0.40, zorder=3)
        ax.add_collection(lc_minor)

    # Draw major contours (prominent)
    logger.info(f"  Drawing {len(lines_major)} major contour lines...")
    if lines_major:
        lc_major = LineCollection(lines_major, colors=colors_major,
                                  linewidths=lw_major, alpha=0.75, zorder=4)
        ax.add_collection(lc_major)

    # --- Styling ---
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set tight bounds from city boundary
    bx = city_boundary.bounds  # (minx, miny, maxx, maxy)
    margin_x = (bx[2] - bx[0]) * 0.025
    margin_y = (bx[3] - bx[1]) * 0.025
    ax.set_xlim(bx[0] - margin_x, bx[2] + margin_x)
    ax.set_ylim(bx[1] - margin_y, bx[3] + margin_y)

    # --- Title ---
    ax.text(0.5, 0.975, 'S A N   F R A N C I S C O',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=48, fontweight='bold', fontfamily='sans-serif',
            color='#ece4d8',
            path_effects=[pe.withStroke(linewidth=4, foreground=bg_color)],
            zorder=10)

    ax.text(0.5, 0.947, 'TERRAIN  RELIEF  &  ELEVATION  CONTOURS',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=16, fontweight='normal', fontfamily='sans-serif',
            color='#9a9080',
            path_effects=[pe.withStroke(linewidth=2, foreground=bg_color)],
            zorder=10)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=terrain_cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.18, 0.045, 0.64, 0.01])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Elevation (m)', fontsize=13, color='#9a9080', labelpad=8)
    cbar.ax.tick_params(colors='#9a9080', labelsize=10)
    cbar.outline.set_edgecolor('#444444')
    cbar.outline.set_linewidth(0.5)

    # Contour legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#6e3a20', linewidth=2.0, alpha=0.75, label='Major contours (50 m)'),
        Line2D([0], [0], color='#5a2e1a', linewidth=0.8, alpha=0.40, label='Minor contours'),
    ]
    leg = ax.legend(handles=legend_elements, loc='lower left',
                    frameon=True, facecolor='#1a1c1e', edgecolor='#444444',
                    labelcolor='#9a9080', fontsize=11, framealpha=0.85)
    leg.set_zorder(10)

    # Attribution
    fig.text(0.97, 0.018, 'DEM: USGS 10 m  |  Contours: SF Open Data',
             ha='right', va='bottom', fontsize=9, color='#555555',
             fontfamily='sans-serif')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.07)

    # Save
    logger.info(f"Saving poster to {save_path} at {dpi} DPI...")
    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    logger.info("Done!")

    return fig


if __name__ == "__main__":
    render_poster()
