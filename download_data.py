#!/usr/bin/env python3
"""Download all data files required by make_poster.py into data/.

Data sources:
  - SF Open Data (Socrata API): neighborhoods, streets, elevation, parks
  - USGS 3DEP ImageServer: 10 m DEM covering SF + Marin + San Mateo

Usage:
    python download_data.py          # download everything
    python download_data.py --skip-dem  # skip the large DEM download
"""

import argparse
import json
import logging
import os
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"

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
# USGS 3DEP DEM (10 m resolution)
# ---------------------------------------------------------------------------

# Bounding box covering SF + Marin County + San Mateo County + islands
DEM_BBOX = {
    "xmin": -122.52,
    "ymin": 37.70,
    "xmax": -122.35,
    "ymax": 37.87,
}
DEM_SIZE = 1850  # pixels per side


def download_dem():
    """Download 10 m DEM from USGS 3DEP ImageServer.

    Covers San Francisco, Marin Headlands, Angel Island, Tiburon,
    Alcatraz, and northern San Mateo County.
    """
    path = os.path.join(DATA_DIR, "sf_dem_10m_marin.tif")
    if os.path.exists(path):
        logger.info("  sf_dem_10m_marin.tif already exists, skipping")
        return

    logger.info("  Downloading USGS 3DEP 10m DEM...")
    bbox = DEM_BBOX
    url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/"
        "3DEPElevation/ImageServer/exportImage"
        f"?bbox={bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"
        f"&bboxSR=4326&imageSR=4326"
        f"&size={DEM_SIZE},{DEM_SIZE}"
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
    parser = argparse.ArgumentParser(description="Download data for SF poster map")
    parser.add_argument("--skip-dem", action="store_true",
                        help="Skip the large DEM download (~13 MB)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("Downloading SF Open Data...")
    download_sf_open_data()

    if not args.skip_dem:
        logger.info("Downloading USGS 3DEP DEM...")
        download_dem()
    else:
        logger.info("Skipping DEM download (--skip-dem)")

    logger.info("Done! All data saved to data/")


if __name__ == "__main__":
    main()
