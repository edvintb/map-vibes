#!/usr/bin/env python3
"""
Street data loading, dataclasses, and visualization for SF street centerlines.

Combines data models (Street, StreetsData) with loading functions
and matplotlib rendering utilities for street network visualization.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class OnewayDirection(Enum):
    """Enumeration for oneway direction values."""
    BIDIRECTIONAL = "B"
    TO_DIRECTION = "T"
    FROM_DIRECTION = "F"


class StreetLayer(Enum):
    """Enumeration for common street layer values."""
    STREETS = "STREETS"
    PRIVATE = "PRIVATE"
    FREEWAYS = "FREEWAYS"
    PARKS_NPS_PRESIDIO = "PARKS_NPS_PRESIDIO"
    UPROW = "UPROW"
    PAPER = "PAPER"
    STREETS_PEDESTRI = "STREETS_PEDESTRI"
    STREETS_TI = "STREETS_TI"
    PARKS = "PARKS"
    PAPER_FWYS = "PAPER_FWYS"
    PAPER_WATER = "PAPER_WATER"
    STREETS_HUNTERSP = "STREETS_HUNTERSP"
    PRIVATE_PARKING = "PRIVATE_PARKING"
    STREETS_YBI = "STREETS_YBI"
    PARKS_NPS_FTMASON = "PARKS_NPS_FTMASON"
    PSEUDO = "PSEUDO"


@dataclass
class LineGeometry:
    """Represents the geometry of a street line (GeoJSON LineString)."""
    type: str  # Should be "LineString"
    coordinates: List[List[float]]  # List of [longitude, latitude] pairs


@dataclass
class Street:
    """Represents a street segment from the SF Streets dataset."""

    # Core identifiers
    cnn: str

    # Address ranges
    lf_fadd: Optional[str] = None
    lf_toadd: Optional[str] = None
    rt_fadd: Optional[str] = None
    rt_toadd: Optional[str] = None

    # Street names
    street: Optional[str] = None
    st_type: Optional[str] = None
    streetname: Optional[str] = None
    streetname_gc: Optional[str] = None
    street_gc: Optional[str] = None

    # Intersection streets
    f_st: Optional[str] = None
    t_st: Optional[str] = None

    # Node connections
    f_node_cnn: Optional[str] = None
    t_node_cnn: Optional[str] = None

    # Status flags
    accepted: Optional[bool] = None
    active: Optional[bool] = None

    # Classification
    classcode: Optional[str] = None
    layer: Optional[str] = None
    jurisdiction: Optional[str] = None
    oneway: Optional[str] = None

    # Geographic information
    nhood: Optional[str] = None
    analysis_neighborhood: Optional[str] = None
    supervisor_district: Optional[str] = None
    zip_code: Optional[str] = None

    # Geometry
    line: Optional[LineGeometry] = None

    # Dates and metadata
    date_added: Optional[str] = None
    date_altered: Optional[str] = None
    date_dropped: Optional[str] = None
    gds_chg_id_add: Optional[str] = None
    gds_chg_id_altered: Optional[str] = None
    gds_chg_id_dropped: Optional[str] = None
    data_as_of: Optional[str] = None
    data_loaded_at: Optional[str] = None

    @property
    def full_street_name(self) -> str:
        """Get the full street name including type."""
        if self.streetname:
            return self.streetname
        elif self.street and self.st_type:
            return f"{self.street} {self.st_type}"
        elif self.street:
            return self.street
        else:
            return f"Street {self.cnn}"

    @property
    def coordinates(self) -> List[Tuple[float, float]]:
        """Get coordinates as a list of (longitude, latitude) tuples."""
        if self.line and self.line.coordinates:
            return [(coord[0], coord[1]) for coord in self.line.coordinates]
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
    def is_bidirectional(self) -> bool:
        """Check if the street is bidirectional."""
        return self.oneway == OnewayDirection.BIDIRECTIONAL.value

    @property
    def is_active_and_accepted(self) -> bool:
        """Check if the street is both active and accepted."""
        return bool(self.active and self.accepted)

    def get_address_range_left(self) -> Optional[Tuple[str, str]]:
        """Get the left side address range as (from, to) tuple."""
        if self.lf_fadd and self.lf_toadd:
            return (self.lf_fadd, self.lf_toadd)
        return None

    def get_address_range_right(self) -> Optional[Tuple[str, str]]:
        """Get the right side address range as (from, to) tuple."""
        if self.rt_fadd and self.rt_toadd:
            return (self.rt_fadd, self.rt_toadd)
        return None

    def __hash__(self) -> int:
        return hash(self.cnn)


@dataclass
class StreetsData:
    """Container for the complete streets dataset."""
    streets: List[Street]

    def filter_by_neighborhood(self, neighborhood: str) -> List[Street]:
        """Filter streets by neighborhood name."""
        return [s for s in self.streets if s.nhood == neighborhood]

    def filter_by_active(self, active: bool = True) -> List[Street]:
        """Filter streets by active status."""
        return [s for s in self.streets if s.active == active]

    def filter_by_accepted(self, accepted: bool = True) -> List[Street]:
        """Filter streets by accepted status."""
        return [s for s in self.streets if s.accepted == accepted]

    def filter_by_layer(self, layer: str) -> List[Street]:
        """Filter streets by layer."""
        return [s for s in self.streets if s.layer == layer]

    def filter_by_street_type(self, street_type: str) -> List[Street]:
        """Filter streets by street type."""
        return [s for s in self.streets if s.st_type == street_type]

    def filter_by_classcode(self, classcode: int) -> List[Street]:
        """Filter streets by classification code.

        Args:
            classcode: Integer code representing street type:
                0 = Other (private streets, paper street, etc.)
                1 = Freeway
                2 = Major street/Highway
                3 = Arterial street
                4 = Collector Street
                5 = Residential Street
                6 = Freeway Ramp
        """
        if not isinstance(classcode, int) or classcode not in range(7):
            raise ValueError("classcode must be an integer between 0 and 6")
        return [s for s in self.streets if s.classcode == str(classcode)]

    def get_neighborhoods(self) -> List[str]:
        """Get a sorted list of unique neighborhoods."""
        neighborhoods = set(s.nhood for s in self.streets if s.nhood)
        return sorted(list(neighborhoods))

    def get_street_types(self) -> List[str]:
        """Get a sorted list of unique street types."""
        street_types = set(s.st_type for s in self.streets if s.st_type)
        return sorted(list(street_types))

    def get_layers(self) -> List[str]:
        """Get a sorted list of unique layers."""
        layers = set(s.layer for s in self.streets if s.layer)
        return sorted(list(layers))

    def search_by_name(self, name: str, case_sensitive: bool = False) -> List[Street]:
        """Search streets by name (partial match)."""
        if not case_sensitive:
            name = name.lower()

        results = []
        for street in self.streets:
            street_name = street.full_street_name
            if not case_sensitive:
                street_name = street_name.lower()

            if name in street_name:
                results.append(street)

        return results


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_streets_from_json(json_data: List[dict]) -> StreetsData:
    """Load streets data from parsed JSON (list of dicts)."""
    streets = []
    for street_dict in json_data:
        line_data = street_dict.get('line')
        line_geometry = None
        if line_data:
            line_geometry = LineGeometry(
                type=line_data['type'],
                coordinates=line_data['coordinates']
            )

        street_dict_copy = street_dict.copy()
        street_dict_copy.pop('line', None)

        street = Street(
            line=line_geometry,
            **street_dict_copy
        )
        streets.append(street)

    return StreetsData(streets=streets)


# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------

DEFAULT_STREET_COLORS = {
    'ST': '#2E86AB',
    'AVE': '#A23B72',
    'BLVD': '#F18F01',
    'HWY': '#C73E1D',
    'WAY': '#6A994E',
    'DR': '#7209B7',
    'LN': '#F77F00',
    'CT': '#FCBF49',
    'CIR': '#90E0EF',
    'PL': '#F72585',
    'ALY': '#4D5057',
    'RAMP': '#FF6B6B',
    'LOOP': '#4ECDC4',
    'TER': '#45B7D1',
    'default': '#666666'
}

DEFAULT_LAYER_COLORS = {
    StreetLayer.STREETS.value: '#2E86AB',
    StreetLayer.FREEWAYS.value: '#C73E1D',
    StreetLayer.PRIVATE.value: '#A23B72',
    StreetLayer.PARKS_NPS_PRESIDIO.value: '#6A994E',
    StreetLayer.UPROW.value: '#F18F01',
    'default': '#666666'
}

DEFAULT_ONEWAY_COLORS = {
    OnewayDirection.BIDIRECTIONAL.value: '#2E86AB',
    OnewayDirection.TO_DIRECTION.value: '#F18F01',
    OnewayDirection.FROM_DIRECTION.value: '#A23B72',
    'default': '#666666'
}


def get_street_color_by_type(street_type: Optional[str]) -> str:
    """Get color for a street based on its type."""
    if not street_type:
        return DEFAULT_STREET_COLORS['default']
    return DEFAULT_STREET_COLORS.get(street_type, DEFAULT_STREET_COLORS['default'])


def get_street_color_by_layer(layer: Optional[str]) -> str:
    """Get color for a street based on its layer."""
    if not layer:
        return DEFAULT_LAYER_COLORS['default']
    return DEFAULT_LAYER_COLORS.get(layer, DEFAULT_LAYER_COLORS['default'])


def get_street_color_by_oneway(oneway: Optional[str]) -> str:
    """Get color for a street based on its oneway direction."""
    if not oneway:
        return DEFAULT_ONEWAY_COLORS['default']
    return DEFAULT_ONEWAY_COLORS.get(oneway, DEFAULT_ONEWAY_COLORS['default'])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def add_street_to_axis(ax: Axes, street: Street,
                      color: Optional[str] = None,
                      linewidth: float = 1.0,
                      alpha: float = 0.8,
                      style: str = 'solid',
                      **kwargs) -> Axes:
    """Add a single street to a matplotlib axis."""
    if not street.coordinates:
        logger.warning(f"Street {street.full_street_name} has no coordinates")
        return ax

    coords = street.coordinates
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]

    if color is None:
        color = get_street_color_by_type(street.st_type)

    ax.plot(x_coords, y_coords,
           color=color,
           linewidth=linewidth,
           alpha=alpha,
           linestyle=style,
           **kwargs)
    return ax


def add_street_names_to_axis(ax: Axes,
                            streets: List[Street],
                            fontsize: float = 8,
                            alpha: float = 0.7) -> None:
    """Add street names at centroids, each name appearing only once."""
    street_groups = {}

    for street in streets:
        name = street.full_street_name
        if not street.coordinates:
            continue

        if name not in street_groups:
            street_groups[name] = []
        street_groups[name].extend(street.coordinates)

    for name, all_coords in street_groups.items():
        if not all_coords:
            continue

        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]

        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)

        ax.text(centroid_x, centroid_y, name,
               fontsize=fontsize, alpha=alpha,
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))


def add_streets_to_axis(ax: Axes,
                       streets: List[Street],
                       color: Optional[str] = None,
                       color_by: str = 'type',
                       linewidth: float = 1.0,
                       alpha: float = 0.8,
                       filter_active: bool = True,
                       filter_accepted: bool = True,
                       show_names: bool = False,
                       name_fontsize: float = 8,
                       name_alpha: float = 0.7,
                       **kwargs) -> Axes:
    """Add multiple streets to a matplotlib axis efficiently."""
    if not streets:
        logger.warning("No streets provided to visualize")
        return ax

    filtered_streets = streets

    logger.info(f"Visualizing {len(filtered_streets)} streets (filtered from {len(streets)}): {color_by} {color}")

    if color:
        street_groups = {color: filtered_streets}
    else:
        street_groups = group_streets_by_color(filtered_streets, color_by)
        legend = create_street_legend(ax, color_by, filtered_streets)
        ax.legend(handles=legend, loc='upper right', bbox_to_anchor=(1, 1), title=color_by.title())

    for color, group_streets in street_groups.items():
        lines = []
        for street in group_streets:
            if street.coordinates:
                coords = street.coordinates
                lines.append([(coord[0], coord[1]) for coord in coords])

        if lines:
            black_lc = LineCollection(lines, colors='black', linewidths=linewidth+0.25, alpha=min(1, alpha + 0.25), **kwargs)
            ax.add_collection(black_lc)
            gray_lc = LineCollection(lines, colors=color, linewidths=linewidth, alpha=alpha, **kwargs)
            ax.add_collection(gray_lc)

    if show_names:
        add_street_names_to_axis(ax, filtered_streets, name_fontsize, name_alpha)

    return ax


def visualize_streets_data(streets_data: StreetsData,
                          figsize: Tuple[int, int] = (12, 10),
                          color_by: str = 'type',
                          linewidth: float = 0.8,
                          alpha: float = 0.7,
                          filter_active: bool = True,
                          filter_accepted: bool = True,
                          title: str = "San Francisco Streets",
                          save_path: Optional[str] = None,
                          dpi: int = 150,
                          show_names: bool = False,
                          name_fontsize: float = 8,
                          name_alpha: float = 0.7,
                          **kwargs) -> Figure:
    """Create a complete visualization of streets data."""
    logger.info(f"Creating streets visualization with {len(streets_data.streets)} streets")

    fig, ax = plt.subplots(figsize=figsize)

    add_streets_to_axis(ax, streets_data.streets,
                       color_by=color_by,
                       linewidth=linewidth,
                       alpha=alpha,
                       filter_active=filter_active,
                       filter_accepted=filter_accepted,
                       show_names=show_names,
                       name_fontsize=name_fontsize,
                       name_alpha=name_alpha,
                       **kwargs)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving visualization to {save_path}")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

    return fig


def group_streets_by_color(streets: List[Street], color_by: str) -> Dict[str, List[Street]]:
    """Group streets by color for efficient rendering."""
    groups = {}

    for street in streets:
        if color_by == 'type':
            color = get_street_color_by_type(street.st_type)
        elif color_by == 'layer':
            color = get_street_color_by_layer(street.layer)
        elif color_by == 'oneway':
            color = get_street_color_by_oneway(street.oneway)
        else:
            color = DEFAULT_STREET_COLORS['default']

        if color not in groups:
            groups[color] = []
        groups[color].append(street)

    return groups


def calculate_street_length(street: Street, method: str = 'bounding_box') -> float:
    """Calculate the length of a street based on its coordinates."""
    if not street.coordinates or len(street.coordinates) < 2:
        return 0.0

    coords = street.coordinates

    if method == 'bounding_box':
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        dx = max(x_coords) - min(x_coords)
        dy = max(y_coords) - min(y_coords)
        return np.sqrt(dx*dx + dy*dy)

    elif method == 'euclidean':
        total_length = 0.0
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            total_length += np.sqrt(dx*dx + dy*dy)
        return total_length

    elif method == 'manhattan':
        total_length = 0.0
        for i in range(1, len(coords)):
            dx = abs(coords[i][0] - coords[i-1][0])
            dy = abs(coords[i][1] - coords[i-1][1])
            total_length += dx + dy
        return total_length

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bounding_box', 'euclidean', or 'manhattan'")


def calculate_streets_lengths(streets: List[Street], method: str = 'bounding_box') -> Dict[str, float]:
    """Calculate lengths for multiple streets."""
    lengths = {}
    for street in streets:
        if street.coordinates:
            lengths[street.cnn] = calculate_street_length(street, method)
        else:
            lengths[street.cnn] = 0.0
    return lengths


def find_longest_streets(streets: List[Street], n: int = 10,
                        method: str = 'bounding_box') -> Tuple[List[Street], List[float]]:
    """Find the n longest streets in a list."""
    street_lengths = []
    for street in streets:
        if street.coordinates:
            length = calculate_street_length(street, method)
            street_lengths.append((street, length))

    street_lengths.sort(key=lambda x: x[1], reverse=True)
    streets_out, lengths = zip(*street_lengths[:n])
    return streets_out, lengths


def create_street_legend(ax: Axes, color_by: str,
                        streets: Optional[List[Street]] = None) -> List[Line2D]:
    """Create a legend for the street visualization."""
    if color_by == 'type':
        legend_items = [(k, v) for k, v in DEFAULT_STREET_COLORS.items() if k != 'default']
    elif color_by == 'layer':
        legend_items = [(k, v) for k, v in DEFAULT_LAYER_COLORS.items() if k != 'default']
    elif color_by == 'oneway':
        legend_items = [
            ('Bidirectional', DEFAULT_ONEWAY_COLORS[OnewayDirection.BIDIRECTIONAL.value]),
            ('To Direction', DEFAULT_ONEWAY_COLORS[OnewayDirection.TO_DIRECTION.value]),
            ('From Direction', DEFAULT_ONEWAY_COLORS[OnewayDirection.FROM_DIRECTION.value])
        ]
    else:
        return []

    legend_elements = [Line2D([0], [0], color=color, lw=2, label=label)
                      for label, color in legend_items]

    return legend_elements


def visualize_neighborhood_streets(streets_data: StreetsData, neighborhood: str,
                                 **kwargs) -> Optional[Figure]:
    """Visualize streets in a specific neighborhood."""
    neighborhood_streets = streets_data.filter_by_neighborhood(neighborhood)
    if not neighborhood_streets:
        logger.warning(f"No streets found for neighborhood: {neighborhood}")
        return None

    return visualize_streets_data(
        StreetsData(neighborhood_streets),
        title=f"Streets in {neighborhood}",
        **kwargs
    )


def visualize_streets_by_type(streets_data: StreetsData, street_type: str,
                             **kwargs) -> Optional[Figure]:
    """Visualize streets of a specific type."""
    type_streets = [s for s in streets_data.streets if s.st_type == street_type]
    if not type_streets:
        logger.warning(f"No streets found for type: {street_type}")
        return None

    return visualize_streets_data(
        StreetsData(type_streets),
        title=f"{street_type} Streets in San Francisco",
        color_by='single',
        **kwargs
    )


if __name__ == "__main__":
    with open('data/sf_streets.json', 'r') as f:
        streets_data = load_streets_from_json(json.load(f))

    print(f"Loaded {len(streets_data.streets)} street segments")

    if streets_data.streets:
        first_street = streets_data.streets[0]
        print(f"First street: {first_street.streetname} ({first_street.street} {first_street.st_type})")
        print(f"Neighborhood: {first_street.nhood}")
        if first_street.line:
            print(f"Coordinates: {len(first_street.line.coordinates)} points")
