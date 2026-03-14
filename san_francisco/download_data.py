#!/usr/bin/env python3
"""Download all data files required by make_map.py into data/.

Data sources:
  - SF Open Data (Socrata API): neighborhoods, streets, elevation, parks
  - USGS 3DEP ImageServer: 10 m DEM covering SF + surrounding area

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
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import PIXELS_PER_DEGREE, DEFAULT_EXTENT, DATA_DIR

# City-specific defaults
CENTER_LON = -122.435
CENTER_LAT = 37.785

# ---------------------------------------------------------------------------
# SF Open Data datasets (Socrata API)
# ---------------------------------------------------------------------------

SF_OPEN_DATA = {
    "sf_neighborhoods.json": {
        "dataset_id": "gfpk-269f",
        "description": "SF Find Neighborhoods (117 neighborhoods with polygons)",
    },
    "sf_elevation.json": {
        "dataset_id": "6d73-6c4f",
        "description": "SF Elevation Contours (14,151 isolines)",
    },
    "sf_park.json": {
        "dataset_id": "gtr9-ntp6",
        "description": "SF Recreation & Parks Properties (245 parks)",
    },
    "sf_streets.json": {
        "dataset_id": "3psu-pn9h",
        "description": "SF Street Centerlines (17,000+ segments)",
    },
    "sf_zoning.json": {
        "dataset_id": "3i4a-hu95",
        "description": "SF Zoning Districts (polygons with zoning codes + gen category)",
    },
    "sf_zoning_height.json": {
        "dataset_id": "kxax-c386",
        "description": "SF Height and Bulk Districts",
    },
    "sf_zoning_special.json": {
        "dataset_id": "ry69-hnut",
        "description": "SF Special Use Districts",
    },
    "sf_land_use.json": {
        "dataset_id": "ygi5-84iq",
        "description": "SF Land Use 2020 (parcel-level)",
    },
}


def download_sf_open_data():
    """Download all datasets from SF Open Data via Socrata API."""
    try:
        from sodapy import Socrata
    except ImportError:
        logger.error("sodapy not installed. Run: pip install sodapy")
        return

    client = Socrata("data.sfgov.org", None)

    for filename, info in SF_OPEN_DATA.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            logger.info(f"  {filename} already exists, skipping")
            continue

        logger.info(f"  Downloading {info['description']}...")
        results = client.get(info["dataset_id"], limit=50000)
        with open(path, "w") as f:
            json.dump(results, f)
        logger.info(f"  -> {filename} ({len(results)} records)")


# ---------------------------------------------------------------------------
# USGS 3DEP DEM
# ---------------------------------------------------------------------------

def download_dem(center_lon, center_lat, extent):
    """Download DEM from USGS 3DEP ImageServer.

    Downloads a square area of `extent` degrees per side, centered on
    (center_lon, center_lat), at native USGS 1/3 arc-second resolution.
    """
    path = os.path.join(DATA_DIR, "sf_dem_10m_marin.tif")
    if os.path.exists(path):
        logger.info("  sf_dem_10m_marin.tif already exists, skipping")
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
    logger.info("  Downloading USGS 3DEP DEM...")
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
    logger.info(f"  -> sf_dem_10m_marin.tif ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download data for SF poster map",
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

    logger.info("Downloading SF Open Data...")
    download_sf_open_data()

    if not args.skip_dem:
        logger.info("Downloading USGS 3DEP DEM...")
        download_dem(args.center_lon, args.center_lat, args.extent)
    else:
        logger.info("Skipping DEM download (--skip-dem)")

    logger.info("Done! All data saved to data/")


if __name__ == "__main__":
    main()
