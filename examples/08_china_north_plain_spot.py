"""
08 — North China Plain: SPOT 6/7 with Delineate-Anything

Delineates fields in Hengshui (Hebei Province) from a SPOT 6/7 multispectral
composite (6 m) with the Delineate-Anything engine (default model
``large_v2``, label-free).

Data:
    - SPOT 6/7 (AIRBUS/SPOT6_7), 2023, scenes with <= 15 % cloud cover. SPOT
      composites hold per-band medians of raw digital numbers (radiometry
      unverified); Delineate-Anything stretches each band to 8 bits with
      scene-wide 1-99 percentiles, so it does not depend on the DN scale.
      The collection ends on 2023-11-15.
    - Study area: a 0.1 x 0.1 degree box (115.4-115.5 E, 37.6-37.7 N). On
      2026-09-27 five SPOT scenes with <= 15 % cloud cover covered all of it
      in 2023, and ESA WorldCover 2021 classifies ~89 % of it as cropland.

SPOT 6/7 access is restricted to select Earth Engine users (internal DRI
use). External users who need SPOT-based field boundaries can contact the
package author to request processing. Without access, the run fails with the
Earth Engine error, which the script prints.

The LULC crop filter is on (Dynamic World is selected outside the US).

Estimated runtime (not measured for 1.0): ~10-20 minutes (1 year, 6 m, GPU).

Prerequisites:
    pip install "agribound[gee,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/08_china_north_plain_spot.py
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import agribound

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*STAC entry.*")
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
OUTPUT_DIR = Path("outputs/china_north_plain")
SOURCE = "spot"
ENGINE = "delineate-anything"
YEAR = 2023

AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [115.40, 37.60],
                        [115.50, 37.60],
                        [115.50, 37.70],
                        [115.40, 37.70],
                        [115.40, 37.60],
                    ]
                ],
            },
            "properties": {"name": "North China Plain AOI (Hengshui, Hebei)"},
        }
    ],
}


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="North China Plain: SPOT 6/7 + DA.")
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


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study_area = str(OUTPUT_DIR / "north_china_aoi.geojson")
    Path(study_area).write_text(json.dumps(AOI))

    output_path = OUTPUT_DIR / f"fields_{SOURCE}_{ENGINE}_{YEAR}.gpkg"
    print(f"Delineating fields from SPOT 6/7 ({YEAR}) -> {output_path}")
    try:
        gdf = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=YEAR,
            engine=ENGINE,
            output_path=str(output_path),
            gee_project=args.gee_project,
            cloud_cover_max=15,
            min_area=3000,
            simplify=2.0,
        )
    except Exception as exc:
        print(f"\nThe run failed: {type(exc).__name__}: {exc}")
        print(
            "If this is an Earth Engine permission error: SPOT 6/7 is restricted to select "
            "users; contact the agribound author for processing."
        )
        return

    print(f"\nDelineated {len(gdf)} fields")
    if len(gdf) and "metrics:area" in gdf.columns:
        print(f"Total area: {gdf['metrics:area'].sum() / 10000:,.1f} ha")
        print(f"Mean field size: {gdf['metrics:area'].mean() / 10000:,.2f} ha")

    web_map = agribound.show_boundaries(
        map_ready(gdf),
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_north_china.html"),
    )
    show_in_notebook(web_map)
    print(f"\nMap saved to {OUTPUT_DIR / 'map_north_china.html'}")


if __name__ == "__main__":
    main()
