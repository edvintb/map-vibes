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
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from process_elevation import ElevationData
    from process_neighborhoods import Neighborhood

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_and_compute_hillshade(dem_path: str, azimuth: float = 315,
                                altitude: float = 45, z_factor: float = 1.0):
    """Load DEM and compute hillshade array.

    Returns (elevation, hillshade, bounds, lon_grid, lat_grid).
    """
    import rasterio

    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        bounds = src.bounds

    rows, cols = elev.shape
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(elev * z_factor, 10.0)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hillshade = np.clip(
        np.sin(alt_rad) * np.cos(slope) +
        np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect),
        0, 1,
    )

    lon = np.linspace(bounds.left, bounds.right, cols)
    lat = np.linspace(bounds.top, bounds.bottom, rows)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    return elev, hillshade, bounds, lon_grid, lat_grid


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


def add_hillshade(ax: Axes, dem_path: str,
                  names: list = None, polys: list = None,
                  neighborhood_color_map: dict = None,
                  color_source: dict = None,
                  color_source_clip=None,
                  color_source_alpha: float = 1.0,
                  azimuth: float = 315, altitude: float = 45,
                  z_factor: float = 3.0, saturation: float = 1.0,
                  alpha: float = 1.0,
                  surrounding_alpha: float = 0.4):
    """Add hillshade layer from a GeoTIFF DEM.

    All land (elevation > 1 m) is rendered as gray hillshade; ocean is
    transparent.

    Color sources (mutually exclusive):
      - neighborhood_color_map + names + polys: raster approach with
        supersampled anti-aliased edges (original neighborhood coloring).
      - color_source: Dict[str, List[shapely_geom]] mapping hex color to
        geometry lists.  Rendered as vector PatchCollections (sharp edges)
        with a hillshade darkening overlay on top.  Preserves terrain detail
        at vector resolution.
    """
    import shapely
    from scipy.ndimage import zoom as _ndi_zoom

    SS = 2  # Supersample factor for anti-aliased edges

    colored = neighborhood_color_map is not None and color_source is None
    logger.info(f"  Computing {'colored' if colored else 'gray'} hillshade from DEM...")

    elev, hillshade, bounds, lon_grid, lat_grid = _load_and_compute_hillshade(
        dem_path, azimuth, altitude, z_factor)

    rows, cols = hillshade.shape
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # Anti-aliased land mask via supersampling: upsample elevation,
    # threshold at high res, then downsample by block-averaging.
    elev_ss = _ndi_zoom(elev, SS, order=1)
    land_ss = (elev_ss > 1.0).astype(np.float32)
    land_alpha = land_ss.reshape(rows, SS, cols, SS).mean(axis=(1, 3))

    # Gray base for all land, washed toward paper white
    paper = np.array([0.96, 0.94, 0.91])
    shade = 0.55 + 0.45 * hillshade
    gray_val = 0.72 * shade

    rgba = np.zeros((rows, cols, 4), dtype=float)
    for c in range(3):
        rgba[:, :, c] = gray_val * 0.7 + paper[c] * 0.3

    # Mute surrounding terrain outside the city boundary
    if color_source_clip is not None and surrounding_alpha < 1.0:
        clip_pts = shapely.points(lon_grid.ravel(), lat_grid.ravel())
        inside_city = shapely.contains(
            color_source_clip, clip_pts).reshape(rows, cols)
        alpha_map = np.where(inside_city, alpha, alpha * surrounding_alpha)
        rgba[:, :, 3] = land_alpha * alpha_map
    else:
        rgba[:, :, 3] = land_alpha * alpha

    # Raster neighborhood coloring (original path)
    if colored:
        logger.info("    Assigning pixels to neighborhoods...")
        for idx, poly in enumerate(polys):
            name = names[idx]
            hex_color = neighborhood_color_map.get(name, '#808080')
            h = hex_color.lstrip('#')
            r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
            if saturation < 1.0:
                wr, wg, wb = 0.96, 0.94, 0.91
                r = r * saturation + wr * (1 - saturation)
                g = g * saturation + wg * (1 - saturation)
                b = b * saturation + wb * (1 - saturation)

            pbounds = poly.bounds
            col_start = max(0, int((pbounds[0] - bounds.left) / (bounds.right - bounds.left) * cols) - 1)
            col_end = min(cols, int((pbounds[2] - bounds.left) / (bounds.right - bounds.left) * cols) + 2)
            row_start = max(0, int((bounds.top - pbounds[3]) / (bounds.top - bounds.bottom) * rows) - 1)
            row_end = min(rows, int((bounds.top - pbounds[1]) / (bounds.top - bounds.bottom) * rows) + 2)

            if col_start >= col_end or row_start >= row_end:
                continue

            nr = row_end - row_start
            nc = col_end - col_start

            # Supersample the contains test for anti-aliased edges
            sub_lon = lon_grid[row_start:row_end, col_start:col_end]
            sub_lat = lat_grid[row_start:row_end, col_start:col_end]
            ss_lon = _ndi_zoom(sub_lon, SS, order=1)[:nr*SS, :nc*SS]
            ss_lat = _ndi_zoom(sub_lat, SS, order=1)[:nr*SS, :nc*SS]
            ss_pts = shapely.points(ss_lon.ravel(), ss_lat.ravel())
            ss_inside = shapely.contains(poly, ss_pts).reshape(nr*SS, nc*SS)
            # Block-average back to original resolution
            blend = ss_inside.astype(np.float32).reshape(
                nr, SS, nc, SS).mean(axis=(1, 3))

            sub_hs = hillshade[row_start:row_end, col_start:col_end]
            sub_shade = 0.55 + 0.45 * sub_hs

            for c_idx, c_val in enumerate([r, g, b]):
                channel = rgba[row_start:row_end, col_start:col_end, c_idx]
                colored_val = c_val * sub_shade
                channel[:] = blend * colored_val + (1.0 - blend) * channel
                rgba[row_start:row_end, col_start:col_end, c_idx] = channel

    logger.info("    Rendering hillshade...")
    ax.imshow(rgba, extent=extent, origin='upper', zorder=0, interpolation='bilinear')

    # Vector color source path: PatchCollections + hillshade darkening overlay
    if color_source is not None:
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection
        from shapely.geometry import Polygon, MultiPolygon

        logger.info("    Rendering vector color source...")
        for hex_color, geom_list in color_source.items():
            h = hex_color.lstrip('#')
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
            if saturation < 1.0:
                r = r * saturation + paper[0] * (1 - saturation)
                g = g * saturation + paper[1] * (1 - saturation)
                b = b * saturation + paper[2] * (1 - saturation)

            patches_list = []
            for geom in geom_list:
                if geom.is_empty:
                    continue
                sub_polys = []
                if isinstance(geom, Polygon):
                    sub_polys = [geom]
                elif isinstance(geom, MultiPolygon):
                    sub_polys = list(geom.geoms)
                elif hasattr(geom, "geoms"):
                    sub_polys = [p for p in geom.geoms if isinstance(p, Polygon)]
                for poly in sub_polys:
                    coords = np.array(poly.exterior.coords)
                    if len(coords) >= 3:
                        patches_list.append(mpatches.Polygon(coords, closed=True))

            if patches_list:
                pc = PatchCollection(
                    patches_list,
                    facecolors=(r, g, b),
                    edgecolors='none',
                    linewidths=0,
                    alpha=color_source_alpha,
                    zorder=1,
                )
                ax.add_collection(pc)

        # Hillshade darkening overlay: black with alpha = (1 - shade),
        # masked to land only so zoning doesn't shade water.
        logger.info("    Rendering hillshade shading overlay...")
        land_mask = (land_alpha > 0).astype(np.float32)

        overlay = np.zeros((rows, cols, 4), dtype=float)
        overlay[:, :, 3] = (1.0 - shade) * land_mask
        ax.imshow(overlay, extent=extent, origin='upper', zorder=1.5,
                  interpolation='bilinear')

        # Water mask: paper-colored layer that hides any color patches
        # bleeding over water (DEM elevation <= 1 m).
        water = (land_alpha == 0).astype(np.float32)
        water_rgba = np.zeros((rows, cols, 4), dtype=float)
        water_rgba[:, :, 0] = paper[0]
        water_rgba[:, :, 1] = paper[1]
        water_rgba[:, :, 2] = paper[2]
        water_rgba[:, :, 3] = water
        ax.imshow(water_rgba, extent=extent, origin='upper', zorder=1.8,
                  interpolation='bilinear')
