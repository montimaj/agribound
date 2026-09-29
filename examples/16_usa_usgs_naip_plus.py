"""
16 — USA, Central Valley (California): USGS NAIP Plus with Delineate-Anything

Delineates fields in the San Joaquin Valley (Fresno County) from the USGS
NAIP Plus ImageServer
(https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer)
with the Delineate-Anything engine (default model ``large_v2``, label-free).
This is the source path that needs no Earth Engine: the footprints
intersecting the study area are selected in the ImageServer catalogue, the
area is exported to a local GeoTIFF (R, G, B, NIR, 8-bit) and passed to the
engine.

Data:
    - The ImageServer holds only the latest NAIP/HRO vintage of each state
      (years 2012-2023), not a historical archive. For this study area the
      catalogue has only 2022 imagery (nine 4-band 0.6 m footprints; query on
      2026-09-27), so any other ``--year`` raises a ValueError that lists the
      available years (``usgs_allow_year_fallback=True`` also accepts
      year +/- 1).
    - The export uses the finest resolution among the selected footprints
      (0.3-0.6 m depending on the state vintage): about 15,500 x 19,000 pixels
      at 0.6 m for this 0.1 x 0.1 degree box (120.1-120.0 W, 36.3-36.4 N).
      Example 07 runs the same box from Earth Engine NAIP at 1 m.
    - The LULC crop filter needs Earth Engine, so it is disabled here to
      keep the run free of Earth Engine.

Estimated runtime (not measured for 1.0): ~30-60 minutes (sub-metre raster, GPU
recommended).

Prerequisites:
    pip install "agribound[delineate-anything]"
    Run from the repository root: python examples/16_usa_usgs_naip_plus.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import agribound

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/usa_central_valley_usgs_naip_plus")
SOURCE = "usgs-naip-plus"
ENGINE = "delineate-anything"
YEAR = 2022
USGS_STATE = "CA"

AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-120.10, 36.30],
                        [-120.00, 36.30],
                        [-120.00, 36.40],
                        [-120.10, 36.40],
                        [-120.10, 36.30],
                    ]
                ],
            },
            "properties": {"name": "San Joaquin Valley AOI (Fresno County)"},
        }
    ],
}


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Central Valley: USGS NAIP Plus + DA.")
    parser.add_argument("--year", type=int, default=YEAR, help="Imagery year.")
    parser.add_argument(
        "--usgs-state", default=USGS_STATE, help="Two-letter state code of the footprints."
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device."
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def map_ready(gdf):
    """Copy of *gdf* with datetime columns as ISO-8601 text, for an HTML map.

    leafmap 0.63 cannot write pandas Timestamp values (such as agribound's
    ``determination:datetime`` column) into the map's HTML: it raises a JSON
    serialisation error.
    """
    import pandas as pd

    out = gdf.copy()
    for column in out.columns:
        if column != out.geometry.name and pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].map(lambda t: t.isoformat() if pd.notna(t) else None)
    return out


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study_area = OUTPUT_DIR / "central_valley_aoi.geojson"
    study_area.write_text(json.dumps(AOI, indent=2), encoding="utf-8")

    output_path = OUTPUT_DIR / f"fields_{SOURCE}_{ENGINE}_{args.year}.gpkg"
    print(f"Delineating fields from USGS NAIP Plus {args.year} -> {output_path}")
    gdf = agribound.delineate(
        study_area=str(study_area),
        source=SOURCE,
        year=args.year,
        engine=ENGINE,
        output_path=str(output_path),
        usgs_state=args.usgs_state,
        device=args.device,
        lulc_filter=False,  # the LULC filter needs Earth Engine
        min_field_area_m2=10000,  # the minimum polygon area chosen for this example
        simplify_tolerance=3.0,
    )

    print(f"\nDelineated {len(gdf)} fields")
    if len(gdf) and "metrics:area" in gdf.columns:
        print(f"Total area: {gdf['metrics:area'].sum() / 10000:,.1f} ha")
        print(f"Mean field size: {gdf['metrics:area'].mean() / 10000:,.1f} ha")

    web_map = agribound.show_boundaries(
        map_ready(gdf),
        basemap="Google.Satellite",
        output_html=str(OUTPUT_DIR / "map_central_valley_usgs_naip_plus.html"),
    )
    show_in_notebook(web_map)
    print(f"\nMap saved to {OUTPUT_DIR / 'map_central_valley_usgs_naip_plus.html'}")


if __name__ == "__main__":
    main()
