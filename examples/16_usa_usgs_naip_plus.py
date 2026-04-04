"""
16 — USA Central Valley (California), USGS NAIP Plus with Delineate-Anything

Delineates large commercial agricultural fields in California's Central Valley
using the USGS NAIP Plus ImageServer (same NAIP data as GEE, from
https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer)
rather than the GEE-backed NAIP source.

This example is intended to demonstrate agribound's non-GEE high-resolution
local-raster acquisition path: the AOI is queried from the USGS ImageServer,
exported to a local GeoTIFF, and then passed into the existing delineation
engine pipeline.

Estimated runtime:  ~30-60 minutes (1 year, 1 m-ish high-resolution imagery, GPU recommended).

Prerequisites:
    pip install "agribound[delineate-anything]"

Notes:
    - No GEE authentication is required for this example.
    - LULC filtering is disabled here to keep the workflow purely non-GEE.
    - If the standalone Delineate-Anything repository is not present
      locally, agribound falls back to direct YOLO inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import agribound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/usa_central_valley_usgs_naip_plus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "usgs-naip-plus"
ENGINE = "delineate-anything"
YEAR = 2022
USGS_STATE = "CA"


def create_study_area() -> str:
    """Create a study area GeoJSON in California's Central Valley."""
    # AOI near Fresno, CA. Reuses the same general region as the GEE-backed
    # NAIP example for easier comparison.
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-119.85, 36.70],
                            [-119.75, 36.70],
                            [-119.75, 36.78],
                            [-119.85, 36.78],
                            [-119.85, 36.70],
                        ]
                    ],
                },
                "properties": {"name": "Central Valley AOI"},
            }
        ],
    }
    path = OUTPUT_DIR / "central_valley_aoi.geojson"
    path.write_text(json.dumps(aoi, indent=2), encoding="utf-8")
    return str(path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="USA Central Valley USGS NAIP Plus high-resolution field boundary delineation."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=YEAR,
        help="Target imagery year.",
    )
    parser.add_argument(
        "--usgs-state",
        default=USGS_STATE,
        help="Two-letter state code used in ImageServer candidate filtering.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for the delineation engine.",
    )
    return parser.parse_args()


def main() -> None:
    """Run high-resolution field delineation from the USGS NAIP Plus ImageServer."""
    args = parse_args()
    study_area = create_study_area()

    print(f"Delineating fields from USGS NAIP Plus {args.year}...")
    output_path = OUTPUT_DIR / f"fields_usgs_naip_plus_{args.year}.gpkg"

    gdf = agribound.delineate(
        study_area=study_area,
        source=SOURCE,
        year=args.year,
        engine=ENGINE,
        output_path=str(output_path),
        usgs_state=args.usgs_state,
        device=args.device,
        # Keep this example purely non-GEE.
        lulc_filter=False,
        # Large commercial fields in the Central Valley.
        min_field_area_m2=10000,
        simplify_tolerance=3.0,
        n_workers=8,
    )

    print(f"\nDelineated {len(gdf)} fields")
    print(f"Output: {output_path}")

    if "metrics:area" in gdf.columns and len(gdf) > 0:
        area_ha = gdf["metrics:area"].sum() / 10000
        mean_ha = gdf["metrics:area"].mean() / 10000
        print(f"Total agricultural area: {area_ha:,.1f} ha")
        print(f"Average field size: {mean_ha:,.1f} ha")

    agribound.show_boundaries(
        gdf,
        basemap="Google.Satellite",
        output_html=str(OUTPUT_DIR / "map_central_valley_usgs_naip_plus.html"),
    )
    print(f"\nMap saved to {OUTPUT_DIR / 'map_central_valley_usgs_naip_plus.html'}")


if __name__ == "__main__":
    main()
