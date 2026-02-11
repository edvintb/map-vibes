#!/usr/bin/env python3
"""
Elevation data loading, dataclasses, and visualization for SF elevation isolines.

Combines data models (ElevationIsoline, ElevationData) with loading functions
and matplotlib rendering utilities for contour visualization.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

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
    coordinates: List[List[float]]  # List of [longitude, latitude] pairs


@dataclass
class ElevationIsoline:
    """Represents an elevation isoline from the SF Elevation dataset."""

    # Core identifiers
    objectid: str

    # Elevation information
    elevation: str  # Elevation value as string (e.g., "-25", "100")
    isoline_ty: Optional[str] = None  # Isoline type (e.g., "800 - Normal")

    # Geometric properties
    shape__len: Optional[str] = None  # Shape length as string

    # Geometry
    the_geom: Optional[ElevationLineGeometry] = None

    @property
    def elevation_value(self) -> Optional[float]:
        """Get elevation as a float value."""
        try:
            return float(self.elevation) if self.elevation else None
        except (ValueError, TypeError):
            return None

    @property
    def shape_length(self) -> Optional[float]:
        """Get shape length as a float value."""
        try:
            return float(self.shape__len) if self.shape__len else None
        except (ValueError, TypeError):
            return None

    @property
    def coordinates(self) -> List[Tuple[float, float]]:
        """Get coordinates as a list of (longitude, latitude) tuples."""
        if self.the_geom and self.the_geom.coordinates:
            return [(coord[0], coord[1]) for coord in self.the_geom.coordinates]
        return []

    @property
    def start_coordinate(self) -> Optional[Tuple[float, float]]:
        """Get the starting coordinate (longitude, latitude)."""
        coords = self.coordinates
        return coords[0] if coords else None

    @property
    def end_coordinate(self) -> Optional[Tuple[float, float]]:
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
    isolines: List[ElevationIsoline]

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

    def get_elevation_values(self) -> List[float]:
        """Get all unique elevation values, sorted."""
        elevations = set()
        for isoline in self.isolines:
            elev = isoline.elevation_value
            if elev is not None:
                elevations.add(elev)
        return sorted(list(elevations))

    def get_elevation_range(self) -> Tuple[Optional[float], Optional[float]]:
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

    def get_isoline_types(self) -> List[str]:
        """Get all unique isoline types."""
        types = set()
        for isoline in self.isolines:
            if isoline.isoline_ty:
                types.add(isoline.isoline_ty)
        return sorted(list(types))


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_elevation_from_json(json_data: List[dict]) -> ElevationData:
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
    with open(filename, 'r') as f:
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


def get_elevation_color(elevation_value: Optional[float],
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

    return '#{:02x}{:02x}{:02x}'.format(
        int(rgba[0] * 255),
        int(rgba[1] * 255),
        int(rgba[2] * 255)
    )


def get_elevation_color_by_range(elevation_value: Optional[float]) -> str:
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
                       color: Optional[str] = None,
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
                        isolines: List[ElevationIsoline],
                        color: Optional[str] = None,
                        color_by: str = 'elevation',
                        linewidth: float = 0.8,
                        alpha: float = 0.7,
                        colormap: str = DEFAULT_ELEVATION_COLORMAP,
                        filter_elevation_range: Optional[Tuple[float, float]] = None,
                        filter_isoline_type: Optional[str] = None,
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
                           figsize: Tuple[int, int] = (12, 10),
                           color_by: str = 'elevation',
                           linewidth: float = 0.6,
                           alpha: float = 0.7,
                           colormap: str = DEFAULT_ELEVATION_COLORMAP,
                           filter_elevation_range: Optional[Tuple[float, float]] = None,
                           filter_isoline_type: Optional[str] = None,
                           title: str = "San Francisco Elevation Contours",
                           save_path: Optional[str] = None,
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
                               **kwargs) -> Optional[Figure]:
    """Visualize isolines in a specific elevation range."""
    filtered_data = elevation_data.filter_by_elevation_range(min_elevation, max_elevation)

    if not filtered_data.isolines:
        logger.warning(f"No isolines found in elevation range {min_elevation}-{max_elevation}m")
        return None

    title = f"Elevation Contours: {min_elevation}m to {max_elevation}m"
    return visualize_elevation_data(filtered_data, title=title, **kwargs)


def visualize_elevation_by_type(elevation_data: ElevationData,
                              isoline_type: str,
                              **kwargs) -> Optional[Figure]:
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


if __name__ == "__main__":
    try:
        elevation_data = load_elevation_from_file('data/sf_elevation.json')
        logger.info(f"Loaded {len(elevation_data.isolines)} elevation isolines")

        min_elev, max_elev = elevation_data.get_elevation_range()
        logger.info(f"Elevation range: {min_elev} to {max_elev}")
        logger.info(f"Isoline types: {elevation_data.get_isoline_types()}")

        fig = visualize_elevation_data(
            elevation_data,
            figsize=(14, 11),
            color_by='elevation',
            title="San Francisco Elevation Contours",
            save_path="images/sf_elevation_contours.png"
        )
        create_elevation_legend(fig.axes[0], 'elevation')
        plt.show()

    except FileNotFoundError:
        logger.error("sf_elevation.json not found.")
    except Exception as e:
        logger.error(f"Error: {e}")
