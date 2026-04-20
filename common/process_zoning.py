#!/usr/bin/env python3
"""
Zoning data loading, classification, and color source generation for poster maps.

Supports:
  - SF zoning districts (Socrata JSON with the_geom MultiPolygon)
  - SF land use parcels  (Socrata JSON with the_geom MultiPolygon)
  - NYC zoning districts  (GeoJSON FeatureCollection, Polygon geometry)
  - NYC special purpose districts (GeoJSON, single-color overlay)
  - NYC commercial overlays       (GeoJSON, single-color overlay)

The main output is a "color_source" dict: {hex_color: [shapely geometries]},
which is passed to add_hillshade(color_source=...) for rendering as vector
PatchCollections with hillshade terrain shading.
"""

import json
import logging
from enum import Enum

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import shape as shapely_shape

from .colors import darken_hex as _darken_hex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zone categories and default colors
# ---------------------------------------------------------------------------

class ZoneCategory(Enum):
    """Normalized zoning/land-use category. Cross-city: SF + NYC source data
    both map into this enum so the rendering and legend code stays shared."""
    RESIDENTIAL = "Residential"
    COMMERCIAL = "Commercial"
    INDUSTRIAL = "Industrial"
    MIXED_USE = "Mixed Use"
    PUBLIC = "Public"
    PARK = "Park"
    OTHER = "Other"
    # Detailed land-use subcategories (PLUTO)
    RESIDENTIAL_LOW = "1-2 Family"
    RESIDENTIAL_MID = "Walk-Up"
    RESIDENTIAL_HIGH = "Elevator"
    TRANSPORTATION = "Transportation"
    PARKING = "Parking"
    VACANT = "Vacant"


DEFAULT_ZONE_COLORS: dict[ZoneCategory, str] = {
    ZoneCategory.RESIDENTIAL: "#C06830",
    ZoneCategory.COMMERCIAL:  "#1E7898",
    ZoneCategory.INDUSTRIAL:  "#786050",
    ZoneCategory.MIXED_USE:   "#C09868",
    ZoneCategory.PUBLIC:      "#488A44",
    ZoneCategory.PARK:        "#488A44",
    ZoneCategory.OTHER:       "#B8AFA3",
    # Detailed land-use subcategory colors
    ZoneCategory.RESIDENTIAL_LOW:  "#D4915E",
    ZoneCategory.RESIDENTIAL_MID:  "#C06830",
    ZoneCategory.RESIDENTIAL_HIGH: "#A0522D",
    ZoneCategory.TRANSPORTATION:   "#9E8B7E",
    ZoneCategory.PARKING:          "#C4B8A8",
    ZoneCategory.VACANT:           "#D4C5B0",
}


# ---------------------------------------------------------------------------
# Category mapping — SF zoning districts (gen field)
# ---------------------------------------------------------------------------

SF_ZONING_MAP: dict[str, ZoneCategory] = {
    "Residential":  ZoneCategory.RESIDENTIAL,
    "Commercial":   ZoneCategory.COMMERCIAL,
    "Industrial":   ZoneCategory.INDUSTRIAL,
    "Mixed":        ZoneCategory.MIXED_USE,
    "Mixed Use":    ZoneCategory.MIXED_USE,
    "Public":       ZoneCategory.PUBLIC,
}


# ---------------------------------------------------------------------------
# Category mapping — SF land use (landuse field)
# ---------------------------------------------------------------------------

SF_LAND_USE_MAP: dict[str, ZoneCategory] = {
    "RESIDENT":    ZoneCategory.RESIDENTIAL,
    "RETAIL/ENT":  ZoneCategory.COMMERCIAL,
    "VISITOR":     ZoneCategory.COMMERCIAL,
    "PDR":         ZoneCategory.INDUSTRIAL,
    "MIXED":       ZoneCategory.MIXED_USE,
    "MIXRES":      ZoneCategory.MIXED_USE,
    "CIE":         ZoneCategory.PUBLIC,
    "MIPS":        ZoneCategory.PUBLIC,
    "MED":         ZoneCategory.PUBLIC,
    "OpenSpace":   ZoneCategory.PARK,
}

_SF_LAND_USE_SKIP = {"VACANT", "MISSING DATA", ""}


# ---------------------------------------------------------------------------
# Category mapping — NYC zoning districts (ZONEDIST field)
# ---------------------------------------------------------------------------

def _classify_nyc_zonedist(zonedist: str) -> ZoneCategory | None:
    """Classify a NYC ZONEDIST string into a ZoneCategory."""
    if not zonedist:
        return None
    zd = zonedist.strip()

    if "/" in zd:
        parts = zd.split("/")
        has_m = any(p.startswith("M") for p in parts)
        has_r = any(p.startswith("R") for p in parts)
        if has_m and has_r:
            return ZoneCategory.MIXED_USE

    if zd == "PARK":
        return ZoneCategory.PARK
    if zd.startswith("R"):
        return ZoneCategory.RESIDENTIAL
    if zd.startswith("C"):
        return ZoneCategory.COMMERCIAL
    if zd.startswith("M"):
        return ZoneCategory.INDUSTRIAL
    return ZoneCategory.OTHER


# ---------------------------------------------------------------------------
# Data loading — Socrata JSON (SF)
# ---------------------------------------------------------------------------

def _load_socrata_zoning(
    path: str, geom_key: str, category_key: str,
    category_map: dict[str, ZoneCategory],
    skip_values: set | None = None,
) -> dict[ZoneCategory, list]:
    """Load SF Socrata JSON, return {ZoneCategory: [shapely geom, ...]}."""
    with open(path) as f:
        records = json.load(f)

    result: dict[ZoneCategory, list] = {cat: [] for cat in ZoneCategory}
    skipped = 0

    for record in records:
        geom_data = record.get(geom_key)
        if not geom_data:
            skipped += 1
            continue
        cat_value = record.get(category_key, "")
        if skip_values and cat_value in skip_values:
            continue
        category = category_map.get(cat_value)
        if category is None:
            continue
        try:
            geom = shapely_shape(geom_data)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not geom.is_empty:
                result[category].append(geom)
        except Exception:
            continue

    total = sum(len(v) for v in result.values())
    logger.info(f"  Loaded {total} zoning geometries from {path} "
                f"(skipped {skipped} without geometry)")
    return result


# ---------------------------------------------------------------------------
# Data loading — GeoJSON (NYC)
# ---------------------------------------------------------------------------

def _load_geojson_zoning(path: str, classify_fn) -> dict[ZoneCategory, list]:
    """Load GeoJSON FeatureCollection, return {ZoneCategory: [shapely geom, ...]}."""
    with open(path) as f:
        data = json.load(f)

    result: dict[ZoneCategory, list] = {cat: [] for cat in ZoneCategory}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        category = classify_fn(props)
        if category is None:
            continue
        geom_data = feat.get("geometry")
        if not geom_data:
            continue
        try:
            geom = shapely_shape(geom_data)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not geom.is_empty:
                result[category].append(geom)
        except Exception:
            continue

    total = sum(len(v) for v in result.values())
    logger.info(f"  Loaded {total} zoning geometries from {path}")
    return result


def _load_geojson_single_layer(path: str) -> list:
    """Load a GeoJSON FeatureCollection as a flat list of Shapely geometries."""
    with open(path) as f:
        data = json.load(f)

    geoms = []
    for feat in data.get("features", []):
        geom_data = feat.get("geometry")
        if not geom_data:
            continue
        try:
            geom = shapely_shape(geom_data)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not geom.is_empty:
                geoms.append(geom)
        except Exception:
            continue

    logger.info(f"  Loaded {len(geoms)} geometries from {path}")
    return geoms


# ---------------------------------------------------------------------------
# Color source builders
# ---------------------------------------------------------------------------

def _build_color_source(
    data: dict[ZoneCategory, list],
    zone_colors: dict[ZoneCategory, str] | None = None,
) -> tuple[dict[str, list], list[ZoneCategory]]:
    """Convert categorized data to a color_source dict for add_hillshade.

    Returns (color_source, categories_present) where:
      - color_source: {hex_color: [shapely geom, ...]}
      - categories_present: list of ZoneCategory with data
    """
    if zone_colors is None:
        zone_colors = DEFAULT_ZONE_COLORS

    color_source: dict[str, list] = {}
    categories_present = []

    for cat in ZoneCategory:
        geoms = data.get(cat, [])
        if not geoms:
            continue
        color = zone_colors.get(cat, DEFAULT_ZONE_COLORS[cat])
        color_source.setdefault(color, []).extend(geoms)
        categories_present.append(cat)

    return color_source, categories_present


def load_sf_zoning_color_source(
    path: str = "data/sf_zoning.json",
    zone_colors: dict[ZoneCategory, str] | None = None,
) -> tuple[dict[str, list], list[ZoneCategory]]:
    """Load SF zoning districts and return color_source for add_hillshade."""
    logger.info("Loading SF zoning districts...")
    data = _load_socrata_zoning(path, "the_geom", "gen", SF_ZONING_MAP)
    return _build_color_source(data, zone_colors)


def load_sf_land_use_color_source(
    path: str = "data/sf_land_use.json",
    zone_colors: dict[ZoneCategory, str] | None = None,
) -> tuple[dict[str, list], list[ZoneCategory]]:
    """Load SF land use parcels and return color_source for add_hillshade."""
    logger.info("Loading SF land use...")
    data = _load_socrata_zoning(
        path, "the_geom", "landuse", SF_LAND_USE_MAP,
        skip_values=_SF_LAND_USE_SKIP,
    )
    return _build_color_source(data, zone_colors)


def load_nyc_zoning_color_source(
    path: str = "data/manhattan_zoning.json",
    zone_colors: dict[ZoneCategory, str] | None = None,
) -> tuple[dict[str, list], list[ZoneCategory]]:
    """Load NYC zoning districts and return color_source for add_hillshade."""
    logger.info("Loading NYC zoning districts...")
    data = _load_geojson_zoning(
        path,
        lambda props: _classify_nyc_zonedist(props.get("ZONEDIST", "")),
    )
    return _build_color_source(data, zone_colors)


# ---------------------------------------------------------------------------
# Category mapping — NYC PLUTO land use (LandUse field, parcel-level)
# ---------------------------------------------------------------------------

NYC_PLUTO_LAND_USE_MAP: dict[str, ZoneCategory] = {
    "01": ZoneCategory.RESIDENTIAL_LOW,   # One & Two Family Buildings
    "02": ZoneCategory.RESIDENTIAL_MID,   # Multi-Family Walk-Up
    "03": ZoneCategory.RESIDENTIAL_HIGH,  # Multi-Family Elevator
    "04": ZoneCategory.MIXED_USE,         # Mixed Residential & Commercial
    "05": ZoneCategory.COMMERCIAL,        # Commercial & Office
    "06": ZoneCategory.INDUSTRIAL,        # Industrial & Manufacturing
    "07": ZoneCategory.TRANSPORTATION,    # Transportation & Utility
    "08": ZoneCategory.PUBLIC,            # Public Facilities & Institutions
    "09": ZoneCategory.PARK,              # Open Space & Outdoor Recreation
    "10": ZoneCategory.PARKING,           # Parking Facilities
    "11": ZoneCategory.VACANT,            # Vacant Land
}


def _classify_nyc_pluto(props: dict) -> ZoneCategory | None:
    """Classify a PLUTO record by its LandUse code."""
    lu = (props.get("LandUse") or "").strip()
    return NYC_PLUTO_LAND_USE_MAP.get(lu)


def load_nyc_land_use_color_source(
    path: str = "data/manhattan_land_use.json",
    zone_colors: dict[ZoneCategory, str] | None = None,
) -> tuple[dict[str, list], list[ZoneCategory]]:
    """Load NYC MapPLUTO parcel-level land use and return color_source."""
    logger.info("Loading NYC parcel-level land use (MapPLUTO)...")
    data = _load_geojson_zoning(path, _classify_nyc_pluto)
    return _build_color_source(data, zone_colors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geom_to_patches(geom) -> list[mpatches.Polygon]:
    """Convert a Shapely geometry to matplotlib Polygons."""
    if geom.is_empty:
        return []
    polys = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
    result = []
    for poly in polys:
        coords = np.array(poly.exterior.coords)
        if len(coords) >= 3:
            result.append(mpatches.Polygon(coords, closed=True))
    return result


# ---------------------------------------------------------------------------
# Single-color overlays (NYC special purpose / commercial)
# ---------------------------------------------------------------------------

def add_single_color_overlay(
    ax: Axes, geoms: list,
    color: str = "#B8AFA3", alpha: float = 0.35,
    linewidth: float = 0.2, zorder: int = 3,
    label: str = "",
):
    """Render geometries as a single-color semi-transparent overlay."""
    edge_color = _darken_hex(color, 0.7)
    patches_list = []
    for geom in geoms:
        patches_list.extend(_geom_to_patches(geom))
    if not patches_list:
        logger.info(f"  {label or 'Overlay'}: no patches to render")
        return
    pc = PatchCollection(
        patches_list,
        facecolors=color, edgecolors=edge_color,
        linewidths=linewidth, alpha=alpha, zorder=zorder,
    )
    ax.add_collection(pc)
    logger.info(f"  {label or 'Overlay'}: {len(patches_list)} patches")


def add_nyc_special_purpose(
    ax: Axes, path: str = "data/manhattan_zoning_special.json",
    color: str = "#A3B18A", alpha: float = 0.35, zorder: int = 3,
):
    """Load and render NYC special purpose districts as a semi-transparent overlay."""
    logger.info("Adding NYC special purpose districts...")
    geoms = _load_geojson_single_layer(path)
    add_single_color_overlay(ax, geoms, color=color, alpha=alpha,
                             zorder=zorder, label="Special Purpose Districts")


def add_nyc_commercial_overlay(
    ax: Axes, path: str = "data/manhattan_zoning_commercial.json",
    color: str = "#7B9EA8", alpha: float = 0.35, zorder: int = 3,
):
    """Load and render NYC commercial overlays as a semi-transparent overlay."""
    logger.info("Adding NYC commercial overlays...")
    geoms = _load_geojson_single_layer(path)
    add_single_color_overlay(ax, geoms, color=color, alpha=alpha,
                             zorder=zorder, label="Commercial Overlays")


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def add_zoning_legend(
    ax: Axes,
    categories: list[ZoneCategory],
    zone_colors: dict[ZoneCategory, str] | None = None,
    loc: str = "upper left",
    fontsize: float = 9.0,
    title: str = "Zoning",
    ncol: int = 3,
    show_unzoned: bool = True,
    unzoned_color: str = "#B8B7B5",
    unzoned_label: str = "Surrounding",
):
    """Add a legend with colored Patch handles for zoning categories."""
    if zone_colors is None:
        zone_colors = DEFAULT_ZONE_COLORS
    handles = []
    for cat in categories:
        color = zone_colors.get(cat, DEFAULT_ZONE_COLORS[cat])
        patch = mpatches.Patch(facecolor=color, edgecolor=_darken_hex(color, 0.7),
                               alpha=0.8, label=cat.value)
        handles.append(patch)
    if show_unzoned:
        patch = mpatches.Patch(facecolor=unzoned_color,
                               edgecolor=_darken_hex(unzoned_color, 0.7),
                               alpha=0.8, label=unzoned_label)
        handles.append(patch)
    ax.legend(
        handles=handles, loc=loc, fontsize=fontsize,
        ncol=ncol,
        framealpha=0.85, edgecolor="#cccccc",
        fancybox=True, borderpad=0.8, borderaxespad=2.0,
    )
