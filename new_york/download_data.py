#!/usr/bin/env python3
"""Download all data files required for the Manhattan poster map.

Data sources:
  - NYC Open Data (Socrata API): neighborhood boundaries (NTA), parks
  - USGS 3DEP ImageServer: ~10 m DEM covering Manhattan + surrounding area

Usage:
    python download_data.py              # download everything
    python download_data.py --skip-dem   # skip the large DEM download
    python download_data.py --extent 0.4 # download a wider area

Downloads a generous DEM area at native USGS 1/3 arc-second resolution
(10800 px/degree). The make_map script crops to the desired view.
"""

import argparse
import json
import logging
import os
import urllib.parse
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from constants import PIXELS_PER_DEGREE, DEFAULT_EXTENT, DATA_DIR

# City-specific defaults
CENTER_LON = -73.98
CENTER_LAT = 40.78


# ---------------------------------------------------------------------------
# NYC Open Data datasets (Socrata API)
# ---------------------------------------------------------------------------

def download_neighborhoods():
    """Download Manhattan NTA boundaries from NYC Open Data.

    Dataset: 2020 Neighborhood Tabulation Areas (9nt8-h7nd)
    Filters to Manhattan only (boroname='Manhattan').
    Renames 'ntaname' -> 'name' to match the shared parse_neighborhoods format.
    """
    path = os.path.join(DATA_DIR, "manhattan_neighborhoods.json")
    if os.path.exists(path):
        logger.info(f"  {path} already exists, skipping")
        return

    try:
        from sodapy import Socrata
    except ImportError:
        logger.error("sodapy not installed. Run: pip install sodapy")
        return

    logger.info("  Downloading Manhattan NTA boundaries...")
    client = Socrata("data.cityofnewyork.us", None)
    results = client.get("9nt8-h7nd", where="boroname='Manhattan'", limit=5000)

    # Rename fields to match expected format
    for record in results:
        if 'ntaname' in record:
            record['name'] = record.pop('ntaname')

    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"  -> manhattan_neighborhoods.json ({len(results)} records)")


def download_parks():
    """Download Manhattan parks from NYC Open Data.

    Dataset: Parks Properties (enfh-gkve)
    Filters to Manhattan (borough='M').
    """
    path = os.path.join(DATA_DIR, "manhattan_parks.json")
    if os.path.exists(path):
        logger.info(f"  {path} already exists, skipping")
        return

    try:
        from sodapy import Socrata
    except ImportError:
        logger.error("sodapy not installed. Run: pip install sodapy")
        return

    logger.info("  Downloading Manhattan parks...")
    client = Socrata("data.cityofnewyork.us", None)
    results = client.get("enfh-gkve", where="borough='M'", limit=5000)

    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"  -> manhattan_parks.json ({len(results)} records)")


# ---------------------------------------------------------------------------
# NYC Zoning (ArcGIS FeatureServer — paginated)
# ---------------------------------------------------------------------------

def _download_arcgis_features(url, path, label, where="1=1", out_fields="*"):
    """Download all features from an ArcGIS FeatureServer with pagination."""
    if os.path.exists(path):
        logger.info(f"  {path} already exists, skipping")
        return

    logger.info(f"  Downloading {label}...")
    all_features = []
    offset = 0
    batch_size = 2000
    where_encoded = urllib.parse.quote(where)

    while True:
        query_url = (
            f"{url}/query?where={where_encoded}&outFields={out_fields}&outSR=4326"
            f"&f=geojson&resultOffset={offset}&resultRecordCount={batch_size}"
        )
        req = urllib.request.Request(query_url)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())

        features = data.get("features", [])
        if not features:
            break
        all_features.extend(features)
        logger.info(f"    fetched {len(all_features)} features...")
        offset += batch_size

        # ArcGIS signals no more results by returning fewer than requested
        if len(features) < batch_size:
            break

    geojson = {"type": "FeatureCollection", "features": all_features}
    with open(path, "w") as f:
        json.dump(geojson, f)
    logger.info(f"  -> {os.path.basename(path)} ({len(all_features)} features)")


def download_zoning():
    """Download NYC zoning districts, special purpose districts, and commercial overlays."""
    arcgis_base = "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services"

    datasets = [
        ("manhattan_zoning.json", f"{arcgis_base}/nyzd/FeatureServer/0",
         "NYC Zoning Districts"),
        ("manhattan_zoning_special.json", f"{arcgis_base}/nysp/FeatureServer/0",
         "NYC Special Purpose Districts"),
        ("manhattan_zoning_commercial.json", f"{arcgis_base}/nyco/FeatureServer/0",
         "NYC Commercial Overlay Districts"),
    ]

    for filename, url, label in datasets:
        path = os.path.join(DATA_DIR, filename)
        _download_arcgis_features(url, path, label)


def download_buildings():
    """Download Manhattan building footprints with heights from NYC Open Data.

    Dataset: Building Footprints (5zhs-2jue)
    Contains ~47,000 buildings in Manhattan with LiDAR-derived roof heights.
    Filters to Manhattan via BIN prefix (1xxxxxxx).
    """
    path = os.path.join(DATA_DIR, "manhattan_buildings.json")
    if os.path.exists(path):
        logger.info(f"  {path} already exists, skipping")
        return

    try:
        from sodapy import Socrata
    except ImportError:
        logger.error("sodapy not installed. Run: pip install sodapy")
        return

    logger.info("  Downloading Manhattan building footprints...")
    client = Socrata("data.cityofnewyork.us", None)

    all_results = []
    offset = 0
    batch_size = 50000

    while True:
        results = client.get(
            "5zhs-2jue",
            where="bin >= 1000000 AND bin < 2000000",
            select="the_geom, height_roof, ground_elevation",
            limit=batch_size,
            offset=offset,
        )
        if not results:
            break
        all_results.extend(results)
        logger.info(f"    fetched {len(all_results)} buildings...")
        offset += batch_size
        if len(results) < batch_size:
            break

    with open(path, "w") as f:
        json.dump(all_results, f)
    logger.info(f"  -> manhattan_buildings.json ({len(all_results)} records)")


def download_land_use():
    """Download MapPLUTO parcel-level land use for Manhattan from NYC DCP ArcGIS.

    MapPLUTO contains ~43,000 individual tax lots in Manhattan, each with a
    LandUse code (01-11) and polygon geometry — far more detailed than the
    ~1,000 zoning districts.
    """
    arcgis_base = "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services"
    path = os.path.join(DATA_DIR, "manhattan_land_use.json")
    _download_arcgis_features(
        f"{arcgis_base}/MapPLUTO/FeatureServer/0",
        path,
        "NYC MapPLUTO (Manhattan parcel-level land use)",
        where="Borough='MN'",
        out_fields="LandUse",
    )


# ---------------------------------------------------------------------------
# USGS 3DEP DEM
# ---------------------------------------------------------------------------

def download_dem(center_lon, center_lat, extent):
    """Download DEM from USGS 3DEP ImageServer.

    Downloads a square area of `extent` degrees per side, centered on
    (center_lon, center_lat), at native USGS 1/3 arc-second resolution.
    """
    path = os.path.join(DATA_DIR, "manhattan_dem.tif")
    if os.path.exists(path):
        logger.info(f"  {path} already exists, skipping")
        return

    half = extent / 2
    xmin = center_lon - half
    ymin = center_lat - half
    xmax = center_lon + half
    ymax = center_lat + half
    dem_width = round(extent * PIXELS_PER_DEGREE)
    dem_height = round(extent * PIXELS_PER_DEGREE)

    logger.info(f"  DEM: {dem_width} x {dem_height} px, "
                f"bbox: ({xmin:.3f}, {ymin:.3f}) to ({xmax:.3f}, {ymax:.3f})")
    logger.info("  Downloading USGS 3DEP DEM for Manhattan area...")
    url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/"
        "3DEPElevation/ImageServer/exportImage"
        f"?bbox={xmin},{ymin},{xmax},{ymax}"
        f"&bboxSR=4326&imageSR=4326"
        f"&size={dem_width},{dem_height}"
        f"&format=tiff&pixelType=F32&noData=-9999"
        f"&interpolation=+RSP_BilinearInterpolation"
        f"&f=image"
    )
    urllib.request.urlretrieve(url, path)
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"  -> manhattan_dem.tif ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download data for Manhattan poster map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--center-lon", type=float, default=CENTER_LON,
                        help=f"DEM center longitude (default: {CENTER_LON})")
    parser.add_argument("--center-lat", type=float, default=CENTER_LAT,
                        help=f"DEM center latitude (default: {CENTER_LAT})")
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT,
                        help=f"DEM extent in degrees per side (default: {DEFAULT_EXTENT})")
    parser.add_argument("--skip-dem", action="store_true",
                        help="Skip the large DEM download")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("Downloading NYC Open Data...")
    download_neighborhoods()
    download_parks()

    logger.info("Downloading NYC Zoning Data...")
    download_zoning()

    logger.info("Downloading NYC Land Use Data (MapPLUTO)...")
    download_land_use()

    logger.info("Downloading NYC Building Footprints...")
    download_buildings()

    if not args.skip_dem:
        logger.info("Downloading USGS 3DEP DEM...")
        download_dem(args.center_lon, args.center_lat, args.extent)
    else:
        logger.info("Skipping DEM download (--skip-dem)")

    logger.info("Done! All data saved to data/")


if __name__ == "__main__":
    main()
