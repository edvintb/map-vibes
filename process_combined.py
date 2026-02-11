#!/usr/bin/env python3
"""
Combined visualization: neighborhoods + elevation contours + hillshade.

Orchestrates the domain-specific modules (neighborhood_processing,
elevation_processing, terrain_processing) to compose a layered map.
"""

import colorsys
import logging
import numpy as np
from typing import List, Optional
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes

from process_neighborhoods import (
    Neighborhood,
    get_neighborhood_color_map,
    build_neighborhood_index,
    add_smart_neighborhood_labels,
)
from process_elevation import ElevationData, ElevationIsoline
from process_terrain import add_filled_contours, add_hillshade, add_colored_hillshade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contour coloring helpers
# ---------------------------------------------------------------------------

def _darken_hex(hex_color: str, factor: float = 0.55):
    """Darken a hex color by multiplying RGB channels."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r * factor, g * factor, b * factor)


def _elevation_tint(hex_color: str, t: float):
    """
    Map a neighborhood color through an elevation gradient.

    t=0 (low elevation): light, desaturated wash
    t=1 (high elevation): dark, saturated
    """
    h_hex = hex_color.lstrip('#')
    r, g, b = (int(h_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    new_l = 0.68 - 0.53 * t
    new_s = 0.40 + 0.45 * t

    return colorsys.hls_to_rgb(h, new_l, new_s)


def _extract_line_coords(geom):
    """Extract coordinate lists from a Shapely geometry."""
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        if len(geom.coords) >= 2:
            return [list(geom.coords)]
        return []
    if isinstance(geom, MultiLineString):
        return [list(line.coords) for line in geom.geoms if len(line.coords) >= 2]
    if hasattr(geom, 'geoms'):
        result = []
        for sub in geom.geoms:
            result.extend(_extract_line_coords(sub))
        return result
    return []


# ---------------------------------------------------------------------------
# Main composition
# ---------------------------------------------------------------------------

def add_combined_visualization_to_axis(ax: Axes,
                                      neighborhoods: List[Neighborhood],
                                      elevation_data: ElevationData,
                                      max_colors: int = 8,
                                      show_neighborhood_labels: bool = True,
                                      elevation_alpha: float = 0.8,
                                      elevation_linewidth: float = 0.35,
                                      neighborhood_alpha: float = 0.3,
                                      palette: str = 'earthy',
                                      contour_style: str = 'tint',
                                      dem_path: Optional[str] = None,
                                      hillshade_alpha: Optional[float] = None,
                                      colored_hillshade: bool = False,
                                      saturation: float = 1.0) -> Axes:
    """
    Add combined neighborhood + elevation + hillshade visualization to an axis.

    Composes terrain base, neighborhood fills, contour lines, and labels.
    """
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    logger.info("Creating combined visualization...")

    # Build spatial index and city boundary
    names, polys, tree = build_neighborhood_index(neighborhoods)
    city_boundary = unary_union(polys)

    # Get neighborhood colors
    neighborhood_color_map, color_palette = get_neighborhood_color_map(
        neighborhoods, max_colors, palette=palette
    )

    # Terrain base and neighborhood backgrounds
    if dem_path and colored_hillshade:
        add_colored_hillshade(ax, dem_path, neighborhoods,
                              neighborhood_color_map, city_boundary,
                              names, polys, tree, saturation=saturation)
        # Thin white neighborhood borders
        border_patches = []
        for neighborhood in neighborhoods:
            for polygon_coords in neighborhood.geometry.coordinates:
                for ring_coords in polygon_coords:
                    coords_array = np.array(ring_coords)
                    border_patches.append(mpatches.Polygon(coords_array, closed=True))
        border_coll = PatchCollection(border_patches, facecolors='none',
                                      edgecolors='white', linewidths=0.6,
                                      alpha=0.5, zorder=2)
        ax.add_collection(border_coll)
    else:
        if dem_path:
            hs_alpha = hillshade_alpha if hillshade_alpha is not None else 0.45
            add_hillshade(ax, dem_path, city_boundary, alpha=hs_alpha)
        else:
            add_filled_contours(ax, elevation_data, city_boundary)

        # Neighborhood backgrounds with low alpha
        logger.info("  Drawing neighborhood backgrounds...")
        patches_list = []
        patch_colors = []
        for neighborhood in neighborhoods:
            color = neighborhood_color_map[neighborhood.name]
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
            for polygon_coords in neighborhood.geometry.coordinates:
                for ring_coords in polygon_coords:
                    coords_array = np.array(ring_coords)
                    polygon = mpatches.Polygon(coords_array, closed=True)
                    patches_list.append(polygon)
                    patch_colors.append(rgb)
        patch_collection = PatchCollection(patches_list, alpha=neighborhood_alpha,
                                          edgecolors='white', linewidths=0.5,
                                          zorder=2)
        patch_collection.set_facecolor(patch_colors)
        ax.add_collection(patch_collection)

    # Contour lines colored per-neighborhood
    logger.info(f"    Indexed {len(polys)} neighborhood polygons")

    min_elev, max_elev = elevation_data.get_elevation_range()
    elev_span = (max_elev - min_elev) if (max_elev and min_elev and max_elev > min_elev) else 1.0

    major_interval = 50.0
    lines_minor = []
    colors_minor = []
    lw_minor = []
    lines_major = []
    colors_major = []
    lw_major = []

    if elevation_linewidth <= 0:
        logger.info("  Skipping contour lines (linewidth=0)")
    else:
        logger.info("  Intersecting isolines with neighborhoods...")
        total = len(elevation_data.isolines)

        for i, isoline in enumerate(elevation_data.isolines):
            if (i + 1) % 2000 == 0:
                logger.info(f"    Processed {i + 1}/{total} isolines...")

            coords = isoline.coordinates
            if len(coords) < 2:
                continue

            elev = isoline.elevation_value or 0.0
            t = max(0.0, min(1.0, (elev - min_elev) / elev_span))

            is_major = (abs(elev % major_interval) < 0.5) or \
                       (abs(elev % major_interval - major_interval) < 0.5)

            lw = elevation_linewidth + 0.2 * t
            if is_major:
                lw = elevation_linewidth * 2.2 + 0.3 * t

            line = LineString(coords)
            candidate_indices = tree.query(line)

            for idx in candidate_indices:
                try:
                    intersection = line.intersection(polys[idx])
                    segments = _extract_line_coords(intersection)
                    if segments:
                        hex_color = neighborhood_color_map.get(names[idx], '#808080')
                        if contour_style == 'darken':
                            darken_factor = 0.7 - 0.4 * t
                            if is_major:
                                darken_factor *= 0.75
                            rgb = _darken_hex(hex_color, darken_factor)
                        else:
                            rgb = _elevation_tint(hex_color, t)
                        for seg in segments:
                            if is_major:
                                lines_major.append(seg)
                                colors_major.append(rgb)
                                lw_major.append(lw)
                            else:
                                lines_minor.append(seg)
                                colors_minor.append(rgb)
                                lw_minor.append(lw)
                except Exception:
                    continue

    # Draw contour collections
    logger.info(f"  Drawing {len(lines_minor)} minor + {len(lines_major)} major contour segments...")
    if lines_minor:
        lc = LineCollection(lines_minor, colors=colors_minor, linewidths=lw_minor,
                          alpha=elevation_alpha, zorder=5)
        ax.add_collection(lc)

    if lines_major:
        major_alpha = min(1.0, elevation_alpha * 1.8)
        lc = LineCollection(lines_major, colors=colors_major, linewidths=lw_major,
                          alpha=major_alpha, zorder=6)
        ax.add_collection(lc)

    # Neighborhood labels
    if show_neighborhood_labels:
        logger.info("  Adding neighborhood labels...")
        def simple_cmap(x):
            idx = int(x * (len(color_palette) - 1))
            idx = min(idx, len(color_palette) - 1)
            hex_color = color_palette[idx].lstrip('#')
            return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

        neighborhood_colors_list = []
        for neighborhood in neighborhoods:
            color = neighborhood_color_map[neighborhood.name]
            neighborhood_colors_list.append(color_palette.index(color))

        fig_width = ax.figure.get_size_inches()[0]
        font_scale = fig_width / 16.0

        add_smart_neighborhood_labels(ax, neighborhoods, neighborhood_colors_list, simple_cmap,
                                      font_scale=font_scale)

    # Set axis limits
    all_lons = []
    all_lats = []
    for neighborhood in neighborhoods:
        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                for lon, lat in ring_coords:
                    all_lons.append(lon)
                    all_lats.append(lat)

    if all_lons and all_lats:
        lon_margin = (max(all_lons) - min(all_lons)) * 0.02
        lat_margin = (max(all_lats) - min(all_lats)) * 0.02
        ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
        ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)

    logger.info("Combined visualization complete!")

    return ax
