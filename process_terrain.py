#!/usr/bin/env python3
"""
DEM-based terrain rendering: hillshade, colored hillshade, and filled contours.

These functions render terrain from raster DEM data or interpolated elevation
isolines onto matplotlib axes, optionally clipped to a city boundary.
"""

import logging
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from process_elevation import ElevationData
    from process_neighborhoods import Neighborhood

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_filled_contours(ax: Axes, elevation_data: 'ElevationData',
                        city_boundary, alpha: float = 0.35,
                        grid_res: int = 500, n_levels: int = 25):
    """
    Add filled contour layer interpolated from isoline data.

    Args:
        ax: Matplotlib axis
        elevation_data: ElevationData with isolines
        city_boundary: Shapely geometry for clipping
        alpha: Fill transparency
        grid_res: Grid resolution (pixels per side)
        n_levels: Number of contour levels
    """
    from scipy.interpolate import griddata
    import shapely

    logger.info("  Building filled contour surface...")

    xs, ys, zs = [], [], []
    for isoline in elevation_data.isolines:
        elev = isoline.elevation_value
        if elev is None:
            continue
        coords = isoline.coordinates
        step = max(1, len(coords) // 30)
        for coord in coords[::step]:
            xs.append(coord[0])
            ys.append(coord[1])
            zs.append(elev)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    logger.info(f"    Sampled {len(xs)} elevation points")

    grid_x = np.linspace(xs.min(), xs.max(), grid_res)
    grid_y = np.linspace(ys.min(), ys.max(), grid_res)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    logger.info("    Interpolating elevation grid...")
    grid_z = griddata((xs, ys), zs, (grid_xx, grid_yy), method='linear')

    elev_cmap = LinearSegmentedColormap.from_list('earth_elev', [
        (0.0, '#f5f0e8'),
        (0.25, '#e8dcc8'),
        (0.5, '#c4a86c'),
        (0.75, '#8b6d3f'),
        (1.0, '#4a3520'),
    ])

    logger.info("    Masking grid to city boundary...")
    points = shapely.points(grid_xx.ravel(), grid_yy.ravel())
    inside = shapely.contains(city_boundary, points).reshape(grid_z.shape)
    grid_z[~inside] = np.nan

    levels = np.linspace(np.nanmin(zs), np.nanmax(zs), n_levels)
    logger.info("    Rendering filled contours...")
    cf = ax.contourf(grid_xx, grid_yy, grid_z, levels=levels,
                     cmap=elev_cmap, alpha=alpha, zorder=1, extend='both')

    return cf


def add_hillshade(ax: Axes, dem_path: str, city_boundary, azimuth: float = 315,
                  altitude: float = 45, alpha: float = 0.45):
    """
    Add a gray hillshade layer from a GeoTIFF DEM, clipped to the city boundary.

    Args:
        ax: Matplotlib axis
        dem_path: Path to GeoTIFF DEM file
        city_boundary: Shapely geometry for masking
        azimuth: Sun azimuth in degrees (315 = NW)
        altitude: Sun altitude in degrees
        alpha: Layer transparency
    """
    import rasterio
    import shapely

    logger.info("  Computing hillshade from DEM...")
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        bounds = src.bounds

    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    dy, dx = np.gradient(elev, 10.0)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)

    hillshade = (
        np.sin(alt_rad) * np.cos(slope) +
        np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    hillshade = np.clip(hillshade, 0, 1)

    logger.info("    Masking hillshade to city boundary...")
    rows, cols = hillshade.shape
    lon = np.linspace(bounds.left, bounds.right, cols)
    lat = np.linspace(bounds.top, bounds.bottom, rows)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    pts = shapely.points(lon_grid.ravel(), lat_grid.ravel())
    inside = shapely.contains(city_boundary, pts).reshape(rows, cols)

    hillshade_masked = np.where(inside, hillshade, np.nan)

    logger.info("    Rendering hillshade...")
    ax.imshow(hillshade_masked, cmap='gray', alpha=alpha,
              extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
              origin='upper', vmin=0, vmax=1, zorder=0, interpolation='bilinear')


def add_colored_hillshade(ax: Axes, dem_path: str,
                          neighborhoods: List['Neighborhood'],
                          neighborhood_color_map: dict, city_boundary,
                          names: list, polys: list, tree,
                          azimuth: float = 315, altitude: float = 45,
                          z_factor: float = 1.5, saturation: float = 1.0):
    """
    Render hillshade tinted per-neighborhood: each neighborhood's terrain is
    shaded in its own earthy color instead of plain gray.

    Args:
        ax: Matplotlib axis
        dem_path: Path to GeoTIFF DEM
        neighborhoods: List of Neighborhood objects
        neighborhood_color_map: {name: '#hex'} mapping
        city_boundary: Shapely geometry (union of all neighborhoods)
        names, polys, tree: Spatial index from build_neighborhood_index
        azimuth, altitude: Sun position for hillshade
        z_factor: Vertical exaggeration
        saturation: Color saturation (0-1, lower = more washed out)
    """
    import rasterio
    import shapely

    logger.info("  Computing colored hillshade from DEM...")
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        bounds = src.bounds

    rows, cols = elev.shape

    # Compute hillshade
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(elev * z_factor, 10.0)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hillshade = (
        np.sin(alt_rad) * np.cos(slope) +
        np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    hillshade = np.clip(hillshade, 0, 1)

    # Build coordinate grids
    lon = np.linspace(bounds.left, bounds.right, cols)
    lat = np.linspace(bounds.top, bounds.bottom, rows)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Determine which pixels are inside city
    logger.info("    Assigning pixels to neighborhoods...")
    pts_flat = shapely.points(lon_grid.ravel(), lat_grid.ravel())
    inside_city = shapely.contains(city_boundary, pts_flat).reshape(rows, cols)

    # Build RGBA image: neighborhood color modulated by hillshade
    rgba = np.zeros((rows, cols, 4), dtype=float)

    for idx, poly in enumerate(polys):
        name = names[idx]
        hex_color = neighborhood_color_map.get(name, '#808080')
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
        # Wash out toward light warm tone (blend toward paper white)
        if saturation < 1.0:
            wr, wg, wb = 0.96, 0.94, 0.91
            r = r * saturation + wr * (1 - saturation)
            g = g * saturation + wg * (1 - saturation)
            b = b * saturation + wb * (1 - saturation)

        # Get bounding box of this polygon in pixel coords
        pbounds = poly.bounds
        col_start = max(0, int((pbounds[0] - bounds.left) / (bounds.right - bounds.left) * cols) - 1)
        col_end = min(cols, int((pbounds[2] - bounds.left) / (bounds.right - bounds.left) * cols) + 2)
        row_start = max(0, int((bounds.top - pbounds[3]) / (bounds.top - bounds.bottom) * rows) - 1)
        row_end = min(rows, int((bounds.top - pbounds[1]) / (bounds.top - bounds.bottom) * rows) + 2)

        if col_start >= col_end or row_start >= row_end:
            continue

        sub_lon = lon_grid[row_start:row_end, col_start:col_end]
        sub_lat = lat_grid[row_start:row_end, col_start:col_end]
        sub_pts = shapely.points(sub_lon.ravel(), sub_lat.ravel())
        sub_inside = shapely.contains(poly, sub_pts).reshape(sub_lon.shape)

        sub_hs = hillshade[row_start:row_end, col_start:col_end]

        # Light-to-dark shading
        shade = 0.55 + 0.45 * sub_hs

        for c_idx, c_val in enumerate([r, g, b]):
            channel = rgba[row_start:row_end, col_start:col_end, c_idx]
            np.putmask(channel, sub_inside, c_val * shade)
            rgba[row_start:row_end, col_start:col_end, c_idx] = channel

        alpha_ch = rgba[row_start:row_end, col_start:col_end, 3]
        np.putmask(alpha_ch, sub_inside, 1.0)
        rgba[row_start:row_end, col_start:col_end, 3] = alpha_ch

    # Pixels outside all neighborhoods but inside city: neutral gray shade
    unassigned = inside_city & (rgba[:, :, 3] == 0)
    if unassigned.any():
        shade = 0.35 + 0.65 * hillshade
        for c in range(3):
            rgba[:, :, c] = np.where(unassigned, 0.7 * shade, rgba[:, :, c])
        rgba[:, :, 3] = np.where(unassigned, 1.0, rgba[:, :, 3])

    logger.info("    Rendering colored hillshade...")
    ax.imshow(rgba,
              extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
              origin='upper', zorder=0, interpolation='bilinear')
