"""
07 — USA, Central Valley (California): NAIP with Delineate-Anything

Delineates fields in the San Joaquin Valley (Fresno County) from NAIP aerial
imagery with the Delineate-Anything engine (default model ``large_v2``,
label-free).

Data:
    - NAIP (USDA/NAIP/DOQQ in Earth Engine, R, G, B, NIR, 8-bit), 2022. NAIP is
      acquired every 2-3 years per state; Earth Engine has 2002-2023. The
      composite is exported at ``naip_resolution_m`` = 1.0 m (default; the
      native GSD is 0.6 m in most states since 2018).
    - Study area: a 0.1 x 0.1 degree box (120.1-120.0 W, 36.3-36.4 N, about
      9 x 11 km); USDA CDL 2022 marks ~95 % of it as cultivated (query on
      2026-09-27). At 1 m the composite is about 9,300 x 11,400 pixels.

The LULC crop filter is on (NLCD is selected in the conterminous US) and
needs Earth Engine, like the NAIP composite.

Estimated runtime (not measured for 1.0): ~20-40 minutes (1 m raster, GPU).

Prerequisites:
    pip install "agribound[gee,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/07_usa_naip_high_res.py
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
OUTPUT_DIR = Path("outputs/usa_central_valley")
SOURCE = "naip"
ENGINE = "delineate-anything"
YEAR = 2022

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
    parser = argparse.ArgumentParser(description="Central Valley: NAIP + Delineate-Anything.")
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
    study_area = str(OUTPUT_DIR / "central_valley_aoi.geojson")
    Path(study_area).write_text(json.dumps(AOI))

    output_path = OUTPUT_DIR / f"fields_{SOURCE}_{ENGINE}_{YEAR}.gpkg"
    print(f"Delineating fields from NAIP {YEAR} -> {output_path}")
    gdf = agribound.delineate(
        study_area=study_area,
        source=SOURCE,
        year=YEAR,
        engine=ENGINE,
        output_path=str(output_path),
        gee_project=args.gee_project,
        min_area=10000,  # m^2; the minimum polygon area chosen for this example
        simplify=3.0,
    )

    meta = gdf.attrs.get("engine_meta", {})
    print(f"\nDelineate-Anything model: {meta.get('model_key')} (backend {meta.get('backend')})")
    print(f"Delineated {len(gdf)} fields")
    if len(gdf) and "metrics:area" in gdf.columns:
        print(f"Total area: {gdf['metrics:area'].sum() / 10000:,.1f} ha")
        print(f"Mean field size: {gdf['metrics:area'].mean() / 10000:,.1f} ha")

    web_map = agribound.show_boundaries(
        map_ready(gdf),
        basemap="Google.Satellite",
        output_html=str(OUTPUT_DIR / "map_central_valley.html"),
    )
    show_in_notebook(web_map)
    print(f"\nMap saved to {OUTPUT_DIR / 'map_central_valley.html'}")


if __name__ == "__main__":
    main()
