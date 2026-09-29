"""
06 — Western Kenya Smallholder Fields: Sentinel-2 with the FTW Engine

Delineates smallholder fields in Kakamega (Western Kenya) with the FTW engine
on Sentinel-2 and compares four minimum-area thresholds
(``min_field_area_m2`` = 100, 500, 1000 and 2500 m^2).

Model: the ftw-tools registry default ``FTW_PRUE_EFNET_B5`` (two seasonal
windows of R, G, B, NIR). agribound has no country-specific FTW models; the
FTW benchmark the model was trained on includes Kenya
(``ftw_tools.settings.ALL_COUNTRIES``).

Study area: a 0.1 x 0.1 degree box (34.7-34.8 E, 0.4-0.5 N) in Kakamega
County (FAO GAUL 2015: ~92 % in the former Kakamega and ~8 % in the former
Lugari district, both now part of the county); ESA WorldCover 2021
classifies ~69 % of it as cropland (queries on 2026-09-27). "Smallholder":
the County Government of Kakamega gives an average farm size of 1.5 acres
(about 0.6 ha) for small-scale holders and 10 acres for large-scale holders,
county-wide (https://kakamega.go.ke/economy/, accessed 2026-09-27; the same
page names Lugari as a sub-county); field sizes inside this box were not
measured. The four runs write separate outputs; the composites and FTW
window composites are cached and shared between them.

The LULC crop filter is on (Dynamic World is selected outside the US) and
needs Earth Engine, like the Sentinel-2 composites.

Estimated runtime (not measured for 1.0): ~10-20 minutes (1 year, small AOI,
GPU).

Prerequisites:
    pip install "agribound[gee,ftw]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/06_kenya_smallholder_ftw.py
"""

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
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/kenya_smallholder")
SOURCE = "sentinel2"
ENGINE = "ftw"
YEAR = 2023
THRESHOLDS_M2 = [100, 500, 1000, 2500]

AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [34.70, 0.40],
                        [34.80, 0.40],
                        [34.80, 0.50],
                        [34.70, 0.50],
                        [34.70, 0.40],
                    ]
                ],
            },
            "properties": {"name": "Kakamega, Western Kenya"},
        }
    ],
}


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Western Kenya: FTW with four area thresholds.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study_area = str(OUTPUT_DIR / "kenya_aoi.geojson")
    Path(study_area).write_text(json.dumps(AOI))

    results = {}
    for min_area in THRESHOLDS_M2:
        output_path = OUTPUT_DIR / f"fields_{SOURCE}_{ENGINE}_{YEAR}_minarea{min_area}.gpkg"
        print(f"\nmin_field_area_m2={min_area} -> {output_path}")
        gdf = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=YEAR,
            engine=ENGINE,
            output_path=str(output_path),
            gee_project=args.gee_project,
            min_area=min_area,
            simplify=1.0,
        )
        results[min_area] = gdf
        print(f"  {len(gdf)} fields")

    print(f"\n{'=' * 60}\nEffect of min_field_area_m2 on the output\n{'=' * 60}")
    for threshold, gdf in results.items():
        mean_area = gdf["metrics:area"].mean() if len(gdf) else 0.0
        print(f"  {threshold:>6} m^2: {len(gdf):>6} fields, mean area {mean_area:>9,.0f} m^2")

    from agribound.visualize import show_comparison

    web_map = show_comparison(
        list(results.values()),
        labels=[f"min {t} m2" for t in results],
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_kenya_thresholds.html"),
    )
    show_in_notebook(web_map)
    print(f"\nComparison map: {OUTPUT_DIR / 'map_kenya_thresholds.html'}")


if __name__ == "__main__":
    main()
