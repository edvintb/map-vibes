#!/usr/bin/env python3
"""
Elevation data loading, dataclasses, and visualization for SF elevation isolines.

Combines data models (ElevationIsoline, ElevationData) with loading functions
and matplotlib rendering utilities for contour visualization.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .colors import darken_rgb as _darken_rgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class IsolineType(Enum):
    """Enumeration for isoline types found in the elevation data."""
    NORMAL_800 = "800 - Normal"
    NORMAL_400 = "400 - Normal"
    NORMAL_200 = "200 - Normal"
    NORMAL_100 = "100 - Normal"
    NORMAL_50 = "50 - Normal"
    NORMAL_25 = "25 - Normal"
    NORMAL_10 = "10 - Normal"
    NORMAL_5 = "5 - Normal"
    NORMAL_2 = "2 - Normal"
    NORMAL_1 = "1 - Normal"


@dataclass
class ElevationLineGeometry:
    """Represents the geometry of an elevation isoline (GeoJSON LineString)."""
    type: str  # Should be "LineString"
    coordinates: list[list[float]]  # List of [longitude, latitude] pairs


@dataclass
class ElevationIsoline:
    """Represents an elevation isoline from the SF Elevation dataset."""

    # Core identifiers
    objectid: str

    # Elevation information
    elevation: str  # Elevation value as string (e.g., "-25", "100")
    isoline_ty: str | None = None  # Isoline type (e.g., "800 - Normal")

    # Geometric properties
    shape__len: str | None = None  # Shape length as string

    # Geometry
    the_geom: ElevationLineGeometry | None = None

    @property
    def elevation_value(self) -> float | None:
        """Get elevation as a float value."""
        try:
            return float(self.elevation) if self.elevation else None
        except (ValueError, TypeError):
            return None

    @property
    def shape_length(self) -> float | None:
        """Get shape length as a float value."""
        try:
            return float(self.shape__len) if self.shape__len else None
        except (ValueError, TypeError):
            return None

    @property
    def coordinates(self) -> list[tuple[float, float]]:
        """Get coordinates as a list of (longitude, latitude) tuples."""
        if self.the_geom and self.the_geom.coordinates:
            return [(coord[0], coord[1]) for coord in self.the_geom.coordinates]
        return []

    @property
    def start_coordinate(self) -> tuple[float, float] | None:
        """Get the starting coordinate (longitude, latitude)."""
        coords = self.coordinates
        return coords[0] if coords else None

    @property
    def end_coordinate(self) -> tuple[float, float] | None:
        """Get the ending coordinate (longitude, latitude)."""
        coords = self.coordinates
        return coords[-1] if coords else None

    @property
    def is_closed_loop(self) -> bool:
        """Check if the isoline forms a closed loop."""
        coords = self.coordinates
        if len(coords) < 3:
            return False
        return coords[0] == coords[-1]

    @property
    def coordinate_count(self) -> int:
        """Get the number of coordinate points in this isoline."""
        return len(self.coordinates)

    def __hash__(self) -> int:
        return hash(self.objectid)


@dataclass
class ElevationData:
    """Container for all elevation isoline data."""
    isolines: list[ElevationIsoline]

    def __len__(self) -> int:
        return len(self.isolines)

    def filter_by_elevation_range(self, min_elevation: float, max_elevation: float) -> 'ElevationData':
        """Filter isolines by elevation range."""
        filtered = []
        for isoline in self.isolines:
            elev = isoline.elevation_value
            if elev is not None and min_elevation <= elev <= max_elevation:
                filtered.append(isoline)
        return ElevationData(isolines=filtered)

    def filter_by_elevation_value(self, elevation: float, tolerance: float = 0.1) -> 'ElevationData':
        """Filter isolines by specific elevation value with tolerance."""
        filtered = []
        for isoline in self.isolines:
            elev = isoline.elevation_value
            if elev is not None and abs(elev - elevation) <= tolerance:
                filtered.append(isoline)
        return ElevationData(isolines=filtered)

    def get_elevation_values(self) -> list[float]:
        """Get all unique elevation values, sorted."""
        elevations = set()
        for isoline in self.isolines:
            elev = isoline.elevation_value
            if elev is not None:
                elevations.add(elev)
        return sorted(list(elevations))

    def get_elevation_range(self) -> tuple[float | None, float | None]:
        """Get the minimum and maximum elevation values."""
        elevations = self.get_elevation_values()
        if not elevations:
            return None, None
        return elevations[0], elevations[-1]

    def get_isolines_by_type(self, isoline_type: str) -> 'ElevationData':
        """Filter isolines by isoline type."""
        filtered = [isoline for isoline in self.isolines
                   if isoline.isoline_ty == isoline_type]
        return ElevationData(isolines=filtered)

    def get_isoline_types(self) -> list[str]:
        """Get all unique isoline types."""
        types = set()
        for isoline in self.isolines:
            if isoline.isoline_ty:
                types.add(isoline.isoline_ty)
        return sorted(list(types))


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_elevation_from_json(json_data: list[dict]) -> ElevationData:
    """Load elevation data from parsed JSON (list of dicts)."""
    isolines = []
    for isoline_dict in json_data:
        geom_data = isoline_dict.get('the_geom')
        line_geometry = None
        if geom_data:
            line_geometry = ElevationLineGeometry(
                type=geom_data['type'],
                coordinates=geom_data['coordinates']
            )

        isoline_dict_copy = isoline_dict.copy()
        isoline_dict_copy.pop('the_geom', None)

        isoline = ElevationIsoline(
            the_geom=line_geometry,
            **isoline_dict_copy
        )
        isolines.append(isoline)

    return ElevationData(isolines=isolines)


def load_elevation_from_file(filename: str) -> ElevationData:
    """Load elevation data from a JSON file."""
    with open(filename) as f:
        json_data = json.load(f)
    return load_elevation_from_json(json_data)


# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------

DEFAULT_ELEVATION_COLORMAP = 'terrain'

ELEVATION_COLORS = {
    'below_sea_level': '#1f4e79',
    'sea_level': '#2e86ab',
    'low_elevation': '#a23b72',
    'medium_elevation': '#f18f01',
    'high_elevation': '#c73e1d',
    'peaks': '#8b0000'
}

ISOLINE_TYPE_COLORS = {
    '800 - Normal': '#2e86ab',
    '810 - Depression': '#8b0000',
    '820 - Intermediate Normal': '#a23b72',
    '830 - Intermediate Depression': '#c73e1d'
}


def get_elevation_color(elevation_value: float | None,
                       colormap: str = DEFAULT_ELEVATION_COLORMAP) -> str:
    """Get color for an elevation value based on a colormap."""
    if elevation_value is None:
        return '#808080'

    min_elev, max_elev = -50, 950
    normalized = (elevation_value - min_elev) / (max_elev - min_elev)
    normalized = max(0, min(1, normalized))

    try:
        cmap = plt.get_cmap(colormap)
    except AttributeError:
        cmap = cm.get_cmap(colormap)
    rgba = cmap(normalized)

    return f'#{int(rgba[0] * 255):02x}{int(rgba[1] * 255):02x}{int(rgba[2] * 255):02x}'


def get_elevation_color_by_range(elevation_value: float | None) -> str:
    """Get color for an elevation value based on predefined ranges."""
    if elevation_value is None:
        return '#808080'

    if elevation_value < 0:
        return ELEVATION_COLORS['below_sea_level']
    elif elevation_value < 10:
        return ELEVATION_COLORS['sea_level']
    elif elevation_value < 100:
        return ELEVATION_COLORS['low_elevation']
    elif elevation_value < 300:
        return ELEVATION_COLORS['medium_elevation']
    elif elevation_value < 600:
        return ELEVATION_COLORS['high_elevation']
    else:
        return ELEVATION_COLORS['peaks']


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def add_isoline_to_axis(ax: Axes,
                       isoline: ElevationIsoline,
                       color: str | None = None,
                       linewidth: float = 1.0,
                       alpha: float = 0.8,
                       style: str = 'solid',
                       color_by: str = 'elevation',
                       **kwargs) -> Axes:
    """Add a single elevation isoline to a matplotlib axis."""
    if not isoline.coordinates:
        logger.warning(f"Isoline {isoline.objectid} has no coordinates")
        return ax

    if color is None:
        if color_by == 'elevation':
            color = get_elevation_color(isoline.elevation_value)
        elif color_by == 'type' and isoline.isoline_ty:
            color = ISOLINE_TYPE_COLORS.get(isoline.isoline_ty, '#808080')
        else:
            color = '#2e86ab'

    coords = isoline.coordinates
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]

    ax.plot(x_coords, y_coords,
           color=color,
           linewidth=linewidth,
           alpha=alpha,
           linestyle=style,
           **kwargs)

    return ax


def add_isolines_to_axis(ax: Axes,
                        isolines: list[ElevationIsoline],
                        color: str | None = None,
                        color_by: str = 'elevation',
                        linewidth: float = 0.8,
                        alpha: float = 0.7,
                        colormap: str = DEFAULT_ELEVATION_COLORMAP,
                        filter_elevation_range: tuple[float, float] | None = None,
                        filter_isoline_type: str | None = None,
                        **kwargs) -> Axes:
    """Add multiple elevation isolines to a matplotlib axis efficiently."""
    if not isolines:
        logger.warning("No isolines provided to visualize")
        return ax

    filtered_isolines = isolines

    if filter_elevation_range:
        min_elev, max_elev = filter_elevation_range
        filtered_isolines = [iso for iso in filtered_isolines
                           if iso.elevation_value is not None and
                           min_elev <= iso.elevation_value <= max_elev]

    if filter_isoline_type:
        filtered_isolines = [iso for iso in filtered_isolines
                           if iso.isoline_ty == filter_isoline_type]

    logger.info(f"Visualizing {len(filtered_isolines)} isolines (filtered from {len(isolines)})")

    if not filtered_isolines:
        logger.warning("No isolines remain after filtering")
        return ax

    if color_by == 'single' or color is not None:
        lines = []
        for isoline in filtered_isolines:
            if isoline.coordinates:
                lines.append(isoline.coordinates)

        if lines:
            line_color = color if color else '#2e86ab'
            lc = LineCollection(lines, colors=line_color, linewidths=linewidth,
                              alpha=alpha, **kwargs)
            ax.add_collection(lc)

    else:
        lines = []
        colors = []

        for isoline in filtered_isolines:
            if not isoline.coordinates:
                continue

            lines.append(isoline.coordinates)

            if color_by == 'elevation':
                line_color = get_elevation_color(isoline.elevation_value, colormap)
            elif color_by == 'type' and isoline.isoline_ty:
                line_color = ISOLINE_TYPE_COLORS.get(isoline.isoline_ty, '#808080')
            else:
                line_color = '#2e86ab'

            colors.append(line_color)

        if lines:
            lc = LineCollection(lines, colors=colors, linewidths=linewidth,
                              alpha=alpha, **kwargs)
            ax.add_collection(lc)

    all_x = []
    all_y = []
    for isoline in filtered_isolines:
        coords = isoline.coordinates
        if coords:
            all_x.extend([c[0] for c in coords])
            all_y.extend([c[1] for c in coords])

    if all_x and all_y:
        ax.set_xlim(min(all_x), max(all_x))
        ax.set_ylim(min(all_y), max(all_y))

    return ax


def visualize_elevation_data(elevation_data: ElevationData,
                           figsize: tuple[int, int] = (12, 10),
                           color_by: str = 'elevation',
                           linewidth: float = 0.6,
                           alpha: float = 0.7,
                           colormap: str = DEFAULT_ELEVATION_COLORMAP,
                           filter_elevation_range: tuple[float, float] | None = None,
                           filter_isoline_type: str | None = None,
                           title: str = "San Francisco Elevation Contours",
                           save_path: str | None = None,
                           dpi: int = 150,
                           **kwargs) -> Figure:
    """Create a complete visualization of elevation data."""
    fig, ax = plt.subplots(figsize=figsize)

    add_isolines_to_axis(
        ax,
        elevation_data.isolines,
        color_by=color_by,
        linewidth=linewidth,
        alpha=alpha,
        colormap=colormap,
        filter_elevation_range=filter_elevation_range,
        filter_isoline_type=filter_isoline_type,
        **kwargs
    )

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, pad=20)

    if color_by == 'elevation' and not filter_elevation_range:
        min_elev, max_elev = elevation_data.get_elevation_range()
        if min_elev is not None and max_elev is not None:
            try:
                cmap = plt.get_cmap(colormap)
            except AttributeError:
                cmap = cm.get_cmap(colormap)
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                     norm=Normalize(vmin=min_elev, vmax=max_elev))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Elevation (meters)', rotation=270, labelpad=20)

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving elevation visualization to {save_path}")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')

    return fig


def add_elevation_contours_to_axis(ax: Axes,
                                 filename: str = "data/sf_elevation.json",
                                 **kwargs) -> Axes:
    """Convenience: load elevation data and add contours to an existing axis."""
    elevation_data = load_elevation_from_file(filename)
    return add_isolines_to_axis(ax, elevation_data.isolines, **kwargs)


def create_elevation_legend(ax: Axes, color_by: str = 'elevation') -> None:
    """Create a legend for elevation visualization."""
    from matplotlib.lines import Line2D

    legend_elements = []

    if color_by == 'elevation':
        elevation_ranges = [
            ('Below Sea Level', ELEVATION_COLORS['below_sea_level']),
            ('Sea Level (0-10m)', ELEVATION_COLORS['sea_level']),
            ('Low Hills (10-100m)', ELEVATION_COLORS['low_elevation']),
            ('Medium Hills (100-300m)', ELEVATION_COLORS['medium_elevation']),
            ('High Hills (300-600m)', ELEVATION_COLORS['high_elevation']),
            ('Peaks (600m+)', ELEVATION_COLORS['peaks'])
        ]
        for label, color in elevation_ranges:
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))

    elif color_by == 'type':
        for isoline_type, color in ISOLINE_TYPE_COLORS.items():
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=isoline_type))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right',
                 bbox_to_anchor=(1, 1), title=color_by.title())


def visualize_elevation_by_range(elevation_data: ElevationData,
                               min_elevation: float,
                               max_elevation: float,
                               **kwargs) -> Figure | None:
    """Visualize isolines in a specific elevation range."""
    filtered_data = elevation_data.filter_by_elevation_range(min_elevation, max_elevation)

    if not filtered_data.isolines:
        logger.warning(f"No isolines found in elevation range {min_elevation}-{max_elevation}m")
        return None

    title = f"Elevation Contours: {min_elevation}m to {max_elevation}m"
    return visualize_elevation_data(filtered_data, title=title, **kwargs)


def visualize_elevation_by_type(elevation_data: ElevationData,
                              isoline_type: str,
                              **kwargs) -> Figure | None:
    """Visualize isolines of a specific type."""
    filtered_data = elevation_data.get_isolines_by_type(isoline_type)

    if not filtered_data.isolines:
        logger.warning(f"No isolines found for type: {isoline_type}")
        return None

    return visualize_elevation_data(
        filtered_data,
        title=f"Elevation Contours: {isoline_type}",
        color_by='type',
        **kwargs
    )


def _elevation_tint(hex_color: str, t: float):
    """Map a neighborhood color through an elevation gradient.

    t=0 (low elevation): light, desaturated
    t=1 (high elevation): dark, saturated
    """
    import colorsys
    h_hex = hex_color.lstrip('#')
    r, g, b = (int(h_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, _lightness, _sat = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, 0.68 - 0.53 * t, 0.40 + 0.45 * t)


def _extract_line_coords(geom):
    """Extract coordinate lists from a Shapely geometry."""
    from shapely.geometry import LineString, MultiLineString
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [list(geom.coords)] if len(geom.coords) >= 2 else []
    if isinstance(geom, MultiLineString):
        return [list(line.coords) for line in geom.geoms if len(line.coords) >= 2]
    if hasattr(geom, 'geoms'):
        result = []
        for sub in geom.geoms:
            result.extend(_extract_line_coords(sub))
        return result
    return []


# ---------------------------------------------------------------------------
# Unified contour color index builders
# ---------------------------------------------------------------------------

def build_contour_color_index_from_neighborhoods(
    names: list, polys: list, tree, neighborhood_color_map: dict,
):
    """Wrap neighborhood data into the unified color index format.

    Returns (hex_colors, polys, tree) where hex_colors[i] is the color
    for polys[i].
    """
    hex_colors = [neighborhood_color_map.get(name, '#808080') for name in names]
    return (hex_colors, polys, tree)


def build_contour_color_index_from_color_source(color_source: dict):
    """Flatten a color_source dict into the unified color index format.

    color_source is {hex_color: [shapely_geom, ...]}.  Returns
    (hex_colors, all_polys, tree) with a new STRtree built from all geometries.
    """
    from shapely.strtree import STRtree

    hex_colors = []
    all_polys = []
    for hex_color, geom_list in color_source.items():
        for geom in geom_list:
            if not geom.is_empty:
                hex_colors.append(hex_color)
                all_polys.append(geom)
    tree = STRtree(all_polys)
    return (hex_colors, all_polys, tree)


# ---------------------------------------------------------------------------
# Unified contour line renderer
# ---------------------------------------------------------------------------

def add_contour_lines(
    ax: Axes,
    elevation_data: ElevationData,
    color_index,
    *,
    contour_interval: float = 10.0,
    major_interval: float = 50.0,
    linewidth: float = 0.35,
    major_linewidth: float = 0.8,
    alpha: float = 0.45,
    show_minor: bool = True,
    show_major: bool = True,
    color_override: str | None = None,
    contour_style: str = "darken",
    gap_color: str | None = "#B8B7B5",
    clip_boundary=None,
):
    """Draw elevation contour lines colored by a unified color index.

    The color index is a (hex_colors, polys, tree) tuple from either
    build_contour_color_index_from_neighborhoods() or
    build_contour_color_index_from_color_source().

    Each isoline is clipped to polygons via STRtree intersection and colored
    based on the polygon's assigned color, darkened or tinted by elevation.
    Gaps between polygons are filled with gap_color (default: gray matching
    the surrounding hillshade).
    """
    from shapely.geometry import LineString

    if not show_minor and not show_major:
        return

    hex_colors, polys, tree = color_index

    min_elev, max_elev = elevation_data.get_elevation_range()
    elev_span = (max_elev - min_elev) if (max_elev and min_elev and max_elev > min_elev) else 1.0

    # Filter isolines by interval
    filtered = []
    for isoline in elevation_data.isolines:
        elev = isoline.elevation_value
        if elev is None or len(isoline.coordinates) < 2:
            continue
        remainder = abs(elev % contour_interval)
        if remainder > 0.5 and abs(remainder - contour_interval) > 0.5:
            continue
        filtered.append(isoline)

    if not filtered:
        logger.info("  No contour lines after filtering")
        return

    logger.info(f"  Processing {len(filtered)} contour lines "
                f"(interval={contour_interval}m, major={major_interval}m)...")

    lines_minor, colors_minor, lw_minor = [], [], []
    lines_major, colors_major, lw_major = [], [], []

    for i, isoline in enumerate(filtered):
        if (i + 1) % 2000 == 0:
            logger.info(f"    Processed {i + 1}/{len(filtered)} isolines...")

        coords = isoline.coordinates
        elev = isoline.elevation_value or 0.0
        t = max(0.0, min(1.0, (elev - min_elev) / elev_span))

        is_major = (abs(elev % major_interval) < 0.5) or \
                   (abs(elev % major_interval - major_interval) < 0.5)

        if is_major and not show_major:
            continue
        if not is_major and not show_minor:
            continue

        lw = linewidth + 0.2 * t
        if is_major:
            lw = major_linewidth + 0.3 * t

        line = LineString(coords)

        # Clip to city boundary if provided
        if clip_boundary is not None:
            try:
                line = line.intersection(clip_boundary)
                if line.is_empty:
                    continue
            except Exception:
                continue
            # Update coords for color_override path
            clipped_segments = _extract_line_coords(line)
            if not clipped_segments:
                continue
        else:
            clipped_segments = None

        if color_override is not None:
            # Single color mode — no spatial lookup needed
            if contour_style == 'darken':
                darken_factor = 0.7 - 0.4 * t
                if is_major:
                    darken_factor *= 0.75
                rgb = _darken_rgb(color_override, darken_factor)
            else:
                rgb = _elevation_tint(color_override, t)
            for seg in (clipped_segments if clipped_segments else [coords]):
                if is_major:
                    lines_major.append(seg)
                    colors_major.append(rgb)
                    lw_major.append(lw)
                else:
                    lines_minor.append(seg)
                    colors_minor.append(rgb)
                    lw_minor.append(lw)
            continue

        # Spatial lookup — clip to polygons and collect covered geometry
        candidate_indices = tree.query(line)

        covered_geoms = []
        for idx in candidate_indices:
            try:
                intersection = line.intersection(polys[idx])
                segments = _extract_line_coords(intersection)
                if not segments:
                    continue
                covered_geoms.append(intersection)
                hex_color = hex_colors[idx]
                if contour_style == 'darken':
                    darken_factor = 0.7 - 0.4 * t
                    if is_major:
                        darken_factor *= 0.75
                    rgb = _darken_rgb(hex_color, darken_factor)
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

        # Gap segments: uncovered portions of the isoline
        if gap_color is not None and covered_geoms:
            try:
                from shapely.ops import unary_union
                covered = unary_union(covered_geoms)
                gap_geom = line.difference(covered)
                gap_segments = _extract_line_coords(gap_geom)
                if gap_segments:
                    gap_rgb = _darken_rgb(gap_color, 0.5 - 0.3 * t)
                    for seg in gap_segments:
                        if is_major:
                            lines_major.append(seg)
                            colors_major.append(gap_rgb)
                            lw_major.append(lw)
                        else:
                            lines_minor.append(seg)
                            colors_minor.append(gap_rgb)
                            lw_minor.append(lw)
            except Exception:
                pass
        elif gap_color is not None and not covered_geoms:
            # Entire isoline is in a gap — render it all in gap color
            gap_rgb = _darken_rgb(gap_color, 0.5 - 0.3 * t)
            if is_major:
                lines_major.append(coords)
                colors_major.append(gap_rgb)
                lw_major.append(lw)
            else:
                lines_minor.append(coords)
                colors_minor.append(gap_rgb)
                lw_minor.append(lw)

    logger.info(f"  Drawing {len(lines_minor)} minor + {len(lines_major)} major contour segments...")
    if lines_minor:
        lc = LineCollection(lines_minor, colors=colors_minor, linewidths=lw_minor,
                            alpha=alpha, zorder=5)
        ax.add_collection(lc)
    if lines_major:
        lc = LineCollection(lines_major, colors=colors_major, linewidths=lw_major,
                            alpha=min(1.0, alpha * 1.8), zorder=6)
        ax.add_collection(lc)


# ---------------------------------------------------------------------------
# Legacy contour renderer (kept for backward compatibility)
# ---------------------------------------------------------------------------

def add_neighborhood_contours(ax: Axes, elevation_data: ElevationData,
                              names: list, polys: list, tree,
                              neighborhood_color_map: dict,
                              contour_style: str = "darken",
                              linewidth: float = 0.35,
                              alpha: float = 0.8,
                              major_interval: float = 50.0):
    """Draw elevation contour lines colored per-neighborhood.

    Each isoline is clipped to neighborhood polygons and colored based on
    the neighborhood's assigned color, darkened or tinted by elevation.
    """
    from shapely.geometry import LineString

    min_elev, max_elev = elevation_data.get_elevation_range()
    elev_span = (max_elev - min_elev) if (max_elev and min_elev and max_elev > min_elev) else 1.0

    lines_minor, colors_minor, lw_minor = [], [], []
    lines_major, colors_major, lw_major = [], [], []

    if linewidth <= 0:
        logger.info("  Skipping contour lines (linewidth=0)")
        return

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

        lw = linewidth + 0.2 * t
        if is_major:
            lw = linewidth * 2.2 + 0.3 * t

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
                        rgb = _darken_rgb(hex_color, darken_factor)
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

    logger.info(f"  Drawing {len(lines_minor)} minor + {len(lines_major)} major contour segments...")
    if lines_minor:
        lc = LineCollection(lines_minor, colors=colors_minor, linewidths=lw_minor,
                            alpha=alpha, zorder=5)
        ax.add_collection(lc)
    if lines_major:
        lc = LineCollection(lines_major, colors=colors_major, linewidths=lw_major,
                            alpha=min(1.0, alpha * 1.8), zorder=6)
        ax.add_collection(lc)
