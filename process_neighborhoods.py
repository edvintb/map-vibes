import colorsys
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import IdentityTransform, Affine2D

sf_map_data_url: str = "https://data.sfgov.org/resource/gfpk-269f.json"

class GeometryType(Enum):
    """Enum for geometry types found in the SF neighborhood data."""
    MULTIPOLYGON = "MultiPolygon"


@dataclass
class Geometry:
    type: GeometryType
    coordinates: List[List[List[Tuple[float, float]]]]  # Simplified: innermost level is (longitude, latitude)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Geometry':
        # Convert coordinate lists to tuples at the innermost level
        def convert_coordinates(coords) -> List[List[List[Tuple[float, float]]]]:
            result = []
            for polygon in coords:  # Each polygon in the MultiPolygon
                polygon_rings = []
                for ring in polygon:  # Each ring in the polygon (exterior + holes)
                    ring_coords = []
                    for coord_pair in ring:  # Each coordinate pair [longitude, latitude]
                        ring_coords.append((float(coord_pair[0]), float(coord_pair[1])))
                    polygon_rings.append(ring_coords)
                result.append(polygon_rings)
            return result

        return cls(
            type=GeometryType(data['type']),
            coordinates=convert_coordinates(data['coordinates'])
        )

@dataclass
class Neighborhood:
    name: str
    link: Optional[str]
    geometry: Geometry
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Neighborhood':
        return cls(
            name=data['name'],
            link=data.get('link'),  # Using get() in case link is missing
            geometry=Geometry.from_dict(data['the_geom'])
        )

def parse_neighborhoods(json_data: List[Dict[str, Any]]) -> List[Neighborhood]:
    """Parse a list of neighborhood JSON data into Neighborhood objects"""
    return [Neighborhood.from_dict(item) for item in json_data]

def find_adjacent_neighborhoods(neighborhoods: List[Neighborhood], logger=None) -> Dict[int, List[int]]:
    """
    Find which neighborhoods are adjacent (share a boundary or are very close).
    Returns a dictionary mapping neighborhood index to list of adjacent neighborhood indices.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if logger is None:
        logger = logging.getLogger('sf_neighborhood')

    logger.info("🔍 Finding adjacent neighborhoods with improved detection...")

    # Convert our neighborhoods to Shapely polygons for easier geometric operations
    shapely_polygons = []
    centroids = []

    for i, neighborhood in enumerate(neighborhoods):
        # Combine all polygons in the MultiPolygon into one
        polygons = []
        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                # Shapely expects (x, y) coordinates, which is (longitude, latitude)
                coords = [(lon, lat) for lon, lat in ring_coords]
                if len(coords) >= 3:  # Need at least 3 points for a polygon
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid:
                            polygons.append(poly)
                    except:
                        continue

        if polygons:
            # Union all polygons for this neighborhood
            try:
                combined = unary_union(polygons)
                if combined.is_valid:
                    shapely_polygons.append(combined)
                    centroids.append(combined.centroid)
                else:
                    shapely_polygons.append(None)
                    centroids.append(None)
            except:
                shapely_polygons.append(None)
                centroids.append(None)
        else:
            shapely_polygons.append(None)
            centroids.append(None)

    # Find adjacencies using multiple methods
    adjacency_dict = {i: set() for i in range(len(neighborhoods))}

    for i in range(len(neighborhoods)):
        if i % 20 == 0:
            logger.info(f"  Checking adjacencies for neighborhood {i+1}/{len(neighborhoods)}")

        if shapely_polygons[i] is None:
            continue

        for j in range(i + 1, len(neighborhoods)):
            if shapely_polygons[j] is None:
                continue

            try:
                poly_i = shapely_polygons[i]
                poly_j = shapely_polygons[j]

                # Method 1: Direct touching
                if poly_i.touches(poly_j):
                    adjacency_dict[i].add(j)
                    adjacency_dict[j].add(i)
                    continue

                # Method 2: Check if they intersect (overlap)
                if poly_i.intersects(poly_j):
                    adjacency_dict[i].add(j)
                    adjacency_dict[j].add(i)
                    continue

                # Method 3: Check distance between boundaries (for very close neighborhoods)
                distance = poly_i.distance(poly_j)
                # Use a small threshold for SF neighborhoods (about 10 meters in degrees)
                if distance < 0.0001:  # Very small distance threshold
                    adjacency_dict[i].add(j)
                    adjacency_dict[j].add(i)
                    continue

                # Method 4: Check if centroids are close (for small neighborhoods)
                if centroids[i] and centroids[j]:
                    centroid_distance = centroids[i].distance(centroids[j])
                    if centroid_distance < 0.01:  # Close centroids
                        adjacency_dict[i].add(j)
                        adjacency_dict[j].add(i)

            except Exception:
                # Skip if there's a geometric error
                continue

    # Convert sets back to lists
    adjacency_dict = {i: list(adj_set) for i, adj_set in adjacency_dict.items()}

    # Print some statistics
    total_adjacencies = sum(len(adj_list) for adj_list in adjacency_dict.values()) // 2
    logger.info(f"  Found {total_adjacencies} adjacency relationships")

    # Show neighborhoods with most connections
    max_connections = max(len(adj_list) for adj_list in adjacency_dict.values())
    logger.info(f"  Maximum connections for any neighborhood: {max_connections}")

    return adjacency_dict

def color_neighborhoods_greedy(adjacency_dict: Dict[int, List[int]], num_neighborhoods: int,
                             force_more_colors: bool = False, max_colors: Optional[int] = None,
                             logger=None, seed: int = 0) -> List[int]:
    """
    Color neighborhoods using randomized graph coloring.
    Processes neighborhoods in random order and assigns a random valid color.
    Returns a list where index i contains the color number for neighborhood i.

    Args:
        adjacency_dict: Dictionary mapping neighborhood indices to their adjacent neighbors
        num_neighborhoods: Total number of neighborhoods
        force_more_colors: Ignored (kept for API compatibility)
        max_colors: Maximum number of colors to use (None for unlimited)
        logger: Logger instance to use for output
        seed: Random seed for reproducibility
    """
    import random
    if logger is None:
        logger = logging.getLogger('sf_neighborhood')

    logger.info("🎨 Applying graph coloring algorithm...")

    rng = random.Random(seed)
    n_colors = max_colors if max_colors else min(8, max(5, num_neighborhoods // 4))

    # Initialize all neighborhoods as uncolored (-1)
    colors = [-1] * num_neighborhoods

    # Process neighborhoods in random order
    order = list(range(num_neighborhoods))
    rng.shuffle(order)

    for i in order:
        # Get colors of adjacent neighborhoods
        adjacent_colors = set()
        for adj in adjacency_dict[i]:
            if colors[adj] != -1:
                adjacent_colors.add(colors[adj])

        # Collect valid colors (not used by neighbors)
        valid = [c for c in range(n_colors) if c not in adjacent_colors]

        if valid:
            colors[i] = rng.choice(valid)
        else:
            # All colors conflict — pick least-used among all colors
            color_counts = [0] * n_colors
            for c in colors:
                if 0 <= c < n_colors:
                    color_counts[c] += 1
            colors[i] = min(range(n_colors), key=lambda c: color_counts[c])

    # Repair any conflicts from the greedy fallback
    for _ in range(100):
        conflicts = False
        for i in range(num_neighborhoods):
            for adj in adjacency_dict[i]:
                if colors[i] == colors[adj]:
                    # Try to recolor i
                    adj_colors = {colors[a] for a in adjacency_dict[i] if colors[a] != -1}
                    valid = [c for c in range(n_colors) if c not in adj_colors]
                    if valid:
                        colors[i] = rng.choice(valid)
                        conflicts = True
                        break
            if conflicts:
                break
        if not conflicts:
            break

    num_colors_used = max(colors) + 1
    logger.info(f"  Successfully colored with {num_colors_used} colors")

    return colors

# Named palettes — each has 8 muted, distinguishable colors
PALETTES = {
    'earthy': [              # warm sandy/beige/orange tones
        '#C9B99A',  '#D4A373',  '#B87F45',  '#C47E5A',  '#8B4225',
    ],
    'nordic': [              # cool Scandinavian tones
        '#7B9EA8',  '#D4A373',  '#A3B18A',  '#BC6C6C',
        '#C8B8A9',  '#5B7B7A',  '#8C7A6B',  '#B5C4B1',
    ],
    'coastal': [             # sea, sand, and sky
        '#6B93A1',  '#D9C5A0',  '#8FB3A0',  '#C47E6E',
        '#B5C8D0',  '#7A8B6E',  '#AE9B82',  '#5E7E8A',
    ],
    'dusk': [                # twilight muted purples / warm neutrals
        '#8E7B9B',  '#C4956A',  '#7A9B8E',  '#B07482',
        '#B8AFA3',  '#6A7F94',  '#9B8E6E',  '#A38DA3',
    ],
    'clay': [                # warm terracotta studio palette
        '#C4785C',  '#7E9B89',  '#D4B483',  '#8B6F5E',
        '#A3B5A6',  '#B8957A',  '#6E7F6E',  '#C9A68C',
    ],
    'mineral': [             # geological / stone-inspired
        '#8A9BA3',  '#BFA87A',  '#7B8C72',  '#A3747A',
        '#C5BDB0',  '#5E7E73',  '#9A8868',  '#7A8FA0',
    ],
    'industrial': [          # industrial chic — concrete, rust, steel, patina
        '#8C8C8C',  '#B5703C',  '#4A5859',  '#A39482',
        '#6E7B8B',  '#C4956A',  '#3E4E50',  '#9B8578',
    ],
}


def create_distinct_colors(n, palette='earthy'):
    """Create n maximally distinct colors from a named palette.

    Available palettes: earthy, nordic, coastal, dusk, clay, mineral, industrial
    """
    base_colors = list(PALETTES.get(palette, PALETTES['earthy']))

    if n <= len(base_colors):
        return base_colors[:n]

    # Generate additional colors in HSV space if more are needed
    additional_needed = n - len(base_colors)
    for i in range(additional_needed):
        hue_base = [0.08, 0.25, 0.33, 0.15, 0.55, 0.75]
        hue = hue_base[i % len(hue_base)] + (i // len(hue_base)) * 0.05
        saturation = 0.35 + (i % 3) * 0.15
        value = 0.65 + (i % 4) * 0.08
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        base_colors.append(hex_color)

    return base_colors[:n]

def calculate_polygon_area(coords):
    """Calculate the area of a polygon using the shoelace formula."""
    if len(coords) < 3:
        return 0

    area = 0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2

def get_polygon_centroid(coords):
    """Calculate the centroid of a polygon."""
    if len(coords) < 3:
        return None

    area = calculate_polygon_area(coords)
    if area == 0:
        return None

    cx = 0
    cy = 0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        # The correct formula uses the cross product of consecutive vertices
        cross_product = coords[i][0] * coords[j][1] - coords[j][0] * coords[i][1]
        cx += (coords[i][0] + coords[j][0]) * cross_product
        cy += (coords[i][1] + coords[j][1]) * cross_product

    # The sign was being applied incorrectly
    # We need to divide by 6 times the signed area
    signed_area = 6 * area
    cx = cx / signed_area
    cy = cy / signed_area
    return (-cx, -cy)

def get_polygon_bounds(coords):
    """Get the bounding box of a polygon."""
    if not coords:
        return None

    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    return {
        'min_lon': min(lons),
        'max_lon': max(lons),
        'min_lat': min(lats),
        'max_lat': max(lats),
        'width': max(lons) - min(lons),
        'height': max(lats) - min(lats)
    }

def calculate_optimal_font_size(text, bounds, base_font_size=8):
    """Calculate optimal font size to fit text within polygon bounds."""
    if not bounds:
        return 6  # Default small size

    # Simplified calculation based on polygon dimensions
    # Use the smaller dimension to determine font size
    min_dimension = min(bounds['width'], bounds['height'])

    # Scale font size based on polygon size
    # Rough conversion: 0.01 degrees ≈ reasonable text size
    if min_dimension > 0.02:  # Large polygon
        return min(10, max(6, int(min_dimension * 200)))
    elif min_dimension > 0.01:  # Medium polygon
        return 6
    else:  # Small polygon
        return 4

def fit_text_to_polygon(text, polygon_coords, min_font_size=5, max_font_size=16):
    """
    Intelligently fit text into a polygon.
    
    Args:
        text: The text to fit
        polygon_coords: List of (x,y) coordinates defining the polygon
        min_font_size: Minimum acceptable font size
        max_font_size: Maximum font size to try
        
    Returns:
        tuple: (formatted_text, centroid, font_size)
    """
    # Get polygon bounds and centroid
    bounds = get_polygon_bounds(polygon_coords)
    centroid = get_polygon_centroid(polygon_coords)
    
    if not bounds or not centroid:
        return text, centroid, min_font_size
    
    # Calculate aspect ratio of the polygon
    aspect_ratio = bounds['width'] / bounds['height'] if bounds['height'] > 0 else 1
    
    # Determine optimal line length based on aspect ratio
    if aspect_ratio > 1.5:  # Wide polygon
        target_chars_per_line = max(5, min(15, len(text) // 2))
    elif aspect_ratio < 0.67:  # Tall polygon
        target_chars_per_line = max(3, min(8, len(text) // 3))
    else:  # Roughly square
        target_chars_per_line = max(5, min(10, len(text) // 2))
    
    # Split text into words
    words = text.split()
    
    # Format text with line breaks
    formatted_text = ""
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if len(test_line) <= target_chars_per_line:
            current_line = test_line
        else:
            formatted_text += current_line + "\n"
            current_line = word
    
    # Add the last line
    if current_line:
        formatted_text += current_line
    
    # Calculate optimal font size based on polygon size
    # Rough estimate: 1 degree of longitude/latitude ≈ 111km at equator
    # We want text to take up about 70% of the polygon's smaller dimension
    min_dimension = min(bounds['width'], bounds['height'])
    
    # Scale factor: 0.01 degrees ≈ reasonable text size for a medium neighborhood
    # This is a heuristic that can be adjusted
    optimal_font_size = min_dimension * 300  # Scale factor

    # Constrain to min/max font size
    font_size = max(min_font_size, min(max_font_size, optimal_font_size))
    
    # Round to nearest 0.5
    font_size = round(font_size * 2) / 2
    
    return formatted_text, centroid, font_size

def add_smart_neighborhood_labels(ax, neighborhoods, neighborhood_colors, cmap, logger=None,
                                  font_scale: float = 1.0):
    """Add neighborhood labels with smart text fitting.

    Args:
        font_scale: Multiplier for font sizes (e.g. 2.0 for double-size figures).
    """
    if logger is None:
        logger = logging.getLogger('sf_neighborhood')

    import matplotlib.patheffects as pe

    # Sort neighborhoods by area to prioritize labeling larger ones
    neighborhood_data = []
    for i, neighborhood in enumerate(neighborhoods):
        largest_area = 0
        best_polygon = None

        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                area = calculate_polygon_area(ring_coords)
                if area > largest_area:
                    largest_area = area
                    best_polygon = ring_coords

        if best_polygon:
            neighborhood_data.append((i, neighborhood, largest_area, best_polygon))

    # Sort by area (largest first) and limit to a reasonable number
    neighborhood_data.sort(key=lambda x: x[2], reverse=True)
    max_labels = min(len(neighborhoods), 150)

    stroke_width = 2 * font_scale

    for i, neighborhood, _, polygon in neighborhood_data[:max_labels]:
        # Get optimally formatted text and position
        display_name = neighborhood.name
        formatted_text, centroid, font_size = fit_text_to_polygon(display_name, polygon)

        if not centroid:
            continue

        ax.text(centroid[0], centroid[1], formatted_text.upper(),
               fontsize=font_size * font_scale, ha='center', va='center', zorder=10,
               color='white',
               multialignment='center', weight='bold', fontfamily='sans-serif',
               path_effects=[pe.withStroke(linewidth=stroke_width, foreground='#333333')])

def add_neighborhood_labels_as_paths(ax, neighborhoods, neighborhood_colors, cmap, logger=None):
    """Add neighborhood labels as paths (more memory efficient)."""
    if logger is None:
        logger = logging.getLogger('sf_neighborhood')

    # logger.info("   Adding labels as paths...")

    # Sort neighborhoods by area
    neighborhood_data = []
    for i, neighborhood in enumerate(neighborhoods):
        largest_area = 0
        best_centroid = None

        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                area = calculate_polygon_area(ring_coords)
                if area > largest_area:
                    largest_area = area
                    best_centroid = get_polygon_centroid(ring_coords)

        if best_centroid:
            neighborhood_data.append((i, neighborhood, largest_area, best_centroid))

    # Sort by area and limit
    neighborhood_data.sort(key=lambda x: x[2], reverse=True)
    max_labels = min(len(neighborhoods), 50)

    for i, neighborhood, _, best_centroid in neighborhood_data[:max_labels]:
        display_name = neighborhood.name
        if len(display_name) > 20:
            display_name = display_name[:17] + "..."

        # logger.info(f"      Adding label for {display_name} at {best_centroid}")

        ax.scatter(best_centroid[0], best_centroid[1], color='red', s=10)

        # Create text path
        text_path = TextPath((0, 0), display_name, size=8)

        # Create patch from path
        p = PathPatch(text_path, facecolor='black', edgecolor='red',
                     transform=IdentityTransform() +
                     Affine2D().translate(best_centroid[0], best_centroid[1]),
                     zorder=10)

        # Add to axes
        ax.add_patch(p)

def setup_logger(level=logging.INFO):
    """Configure the sf_neighborhood logger with the specified level.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG, etc.)
              or string ('INFO', 'DEBUG', etc.)
    """
    # Convert string level to numeric if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add a console handler with the specified level
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set the logger level
    logger.setLevel(level)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger

# Create a module-level logger
logger = logging.getLogger('sf_neighborhood')

# Initialize with default level
setup_logger()

def visualize_neighborhoods(
        neighborhoods: List[Neighborhood],
        ax: Axes,
        use_adjacency_coloring: bool = True,
        max_colors: Optional[int] = None,
        show_labels: bool = True
) -> Axes:
    """
    Add neighborhood visualization to a matplotlib axis.

    Args:
        neighborhoods: List of Neighborhood objects to visualize
        ax: Matplotlib axis to draw on
        use_adjacency_coloring: If True, use graph coloring to ensure adjacent neighborhoods have different colors
        max_colors: Maximum number of colors to use (None for unlimited, minimum 4 recommended)
        show_labels: If True, add neighborhood names scaled to fit within polygons
        
    Returns:
        matplotlib axis object
    """

    # Determine coloring strategy
    if use_adjacency_coloring:
        # Find adjacencies and apply graph coloring
        adjacency_dict = find_adjacent_neighborhoods(neighborhoods)
        neighborhood_colors = color_neighborhoods_greedy(adjacency_dict, len(neighborhoods),
                                                        force_more_colors=(max_colors is None),
                                                        max_colors=max_colors)

        # Create a color palette with many distinct colors
        num_colors_needed = max(neighborhood_colors) + 1
        if max_colors is not None:
            num_colors_to_use = max_colors
            logger.info(f"🎨 Using {num_colors_to_use} colors (requested max: {max_colors})")
        else:
            num_colors_to_use = num_colors_needed
            logger.info(f"🎨 Using {num_colors_needed} distinct colors for proper adjacency coloring")

        # Get distinct colors
        distinct_colors = create_distinct_colors(num_colors_to_use)

        # Create a custom colormap
        def custom_cmap(x):
            if num_colors_to_use == 1:
                return distinct_colors[0]
            color_idx = int(x * (num_colors_to_use - 1))
            color_idx = min(color_idx, num_colors_to_use - 1)
            hex_color = distinct_colors[color_idx]
            # Convert hex to RGB tuple
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

        cmap = custom_cmap
    else:
        # Use simple sequential coloring
        cmap = cm.get_cmap('Set3')
        neighborhood_colors = list(range(len(neighborhoods)))

    patches_list = []
    patch_colors = []
    neighborhood_names = []

    logger.info(f"🖼️  Rendering {len(neighborhoods)} neighborhoods...")

    for i, neighborhood in enumerate(neighborhoods):
        color_index = neighborhood_colors[i]
        color = cmap(color_index / max(neighborhood_colors))

        # Extract coordinates for each polygon in the MultiPolygon
        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                # Convert tuples to numpy array for matplotlib
                coords_array = np.array(ring_coords)

                # Create a polygon patch
                polygon = patches.Polygon(coords_array, closed=True)
                patches_list.append(polygon)
                patch_colors.append(color)
                neighborhood_names.append(neighborhood.name)

    # Create patch collection with colors
    patch_collection = PatchCollection(patches_list, alpha=0.7, edgecolors='white', linewidths=0.5)
    patch_collection.set_facecolor(patch_colors)

    # Add patches to the plot
    ax.add_collection(patch_collection)

    # Add neighborhood labels if requested
    if show_labels:
        logger.info("📝 Adding neighborhood labels...")
        add_smart_neighborhood_labels(ax, neighborhoods, neighborhood_colors, cmap, logger)

    logger.info(f"Successfully rendered {len(neighborhoods)} neighborhoods!")

    # Calculate bounds from all coordinates
    all_lons = []
    all_lats = []
    for neighborhood in neighborhoods:
        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                for lon, lat in ring_coords:
                    all_lons.append(lon)
                    all_lats.append(lat)

    # Set plot limits with some padding
    lon_margin = (max(all_lons) - min(all_lons)) * 0.05
    lat_margin = (max(all_lats) - min(all_lats)) * 0.05

    ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
    ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)

    return ax


def get_neighborhood_color_map(neighborhoods: List[Neighborhood],
                               max_colors: int = 8,
                               palette: str = 'earthy') -> Tuple[dict, list]:
    """
    Generate a color map for neighborhoods using graph coloring.

    Returns:
        Tuple of (neighborhood_to_color dict, color_palette list)
    """
    logger.info("Computing neighborhood colors...")

    adjacency_dict = find_adjacent_neighborhoods(neighborhoods)
    neighborhood_colors = color_neighborhoods_greedy(
        adjacency_dict,
        len(neighborhoods),
        force_more_colors=True,
        max_colors=max_colors
    )

    num_colors_needed = max(neighborhood_colors) + 1
    color_palette = create_distinct_colors(num_colors_needed, palette=palette)

    neighborhood_color_map = {}
    for i, neighborhood in enumerate(neighborhoods):
        color_idx = neighborhood_colors[i]
        neighborhood_color_map[neighborhood.name] = color_palette[color_idx]

    logger.info(f"  Using {num_colors_needed} distinct colors")

    return neighborhood_color_map, color_palette


def add_neighborhood_fills(ax, neighborhoods, color_map, alpha=0.3,
                           edge_color='white', linewidth=0.5, zorder=2):
    """Draw colored neighborhood polygons on an axis."""
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    patches = []
    colors = []
    for n in neighborhoods:
        hex_color = color_map[n.name].lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        for polygon_coords in n.geometry.coordinates:
            for ring_coords in polygon_coords:
                patches.append(mpatches.Polygon(np.array(ring_coords), closed=True))
                colors.append(rgb)
    coll = PatchCollection(patches, alpha=alpha, edgecolors=edge_color,
                           linewidths=linewidth, zorder=zorder)
    coll.set_facecolor(colors)
    ax.add_collection(coll)


def build_neighborhood_index(neighborhoods: List[Neighborhood]) -> Tuple[list, list, 'STRtree']:
    """
    Build Shapely polygons and an STRtree spatial index for neighborhoods.

    Returns:
        Tuple of (names list, polygons list, STRtree index)
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    names = []
    polys = []

    for neighborhood in neighborhoods:
        polygons = []
        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                if len(ring_coords) >= 3:
                    try:
                        poly = Polygon(ring_coords)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception:
                        continue

        if polygons:
            try:
                combined = unary_union(polygons)
                if combined.is_valid:
                    names.append(neighborhood.name)
                    polys.append(combined)
            except Exception:
                continue

    tree = STRtree(polys)
    return names, polys, tree


def add_neighborhood_borders(ax, neighborhoods, bg_color="#faf8f4", linewidth=1.2, zorder=2):
    """Draw neighborhood polygons with background-colored edges (no fill)."""
    logger = logging.getLogger(__name__)
    border_patches = []
    for neighborhood in neighborhoods:
        for polygon_coords in neighborhood.geometry.coordinates:
            for ring_coords in polygon_coords:
                coords_array = np.array(ring_coords)
                border_patches.append(patches.Polygon(coords_array, closed=True))

    coll = PatchCollection(
        border_patches,
        facecolors="none",
        edgecolors=bg_color,
        linewidths=linewidth,
        zorder=zorder,
    )
    ax.add_collection(coll)
    logger.info(f"  Added {len(border_patches)} neighborhood border polygons")
