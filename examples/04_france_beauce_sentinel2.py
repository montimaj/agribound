"""
04 — France, Beauce Region: Sentinel-2 with the FTW Engine

Delineates fields in the Beauce (Eure-et-Loir) with the Fields of
The World (FTW) engine on Sentinel-2, without fine-tuning.

The FTW engine runs the ftw-tools registry default model
(``FTW_PRUE_EFNET_B5`` in ftw-tools 2.0.0b5: PRUE U-Net, EfficientNet-B5
encoder). It takes two input windows of R, G, B, NIR, centred on the start and
end of the growing season from FTW's crop calendar; each window is a median
Sentinel-2 composite of +/- 30 days, built and cached separately. The window
dates are printed from ``gdf.attrs["engine_meta"]``. The FTW benchmark the
model was trained on includes France (``ftw_tools.settings.ALL_COUNTRIES``).
Other models: ``engine_params={"model": ...}`` (``agribound list-ftw-models``).

The LULC crop filter is on (Dynamic World is selected outside the US) and
needs Earth Engine, like the Sentinel-2 composites.

Estimated runtime (not measured for 1.0): ~15-30 minutes (1 year, GPU).

Prerequisites:
    pip install "agribound[gee,ftw]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/04_france_beauce_sentinel2.py
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
OUTPUT_DIR = Path("outputs/france_beauce")
SOURCE = "sentinel2"
ENGINE = "ftw"
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
                        [1.40, 48.10],
                        [1.55, 48.10],
                        [1.55, 48.20],
                        [1.40, 48.20],
                        [1.40, 48.10],
                    ]
                ],
            },
            "properties": {"name": "Beauce AOI (Eure-et-Loir)"},
        }
    ],
}


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Beauce, France: Sentinel-2 + FTW.")
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
    study_area = str(OUTPUT_DIR / "beauce_aoi.geojson")
    Path(study_area).write_text(json.dumps(AOI))

    output_path = OUTPUT_DIR / f"fields_{SOURCE}_{ENGINE}_{YEAR}.gpkg"
    print(f"Delineating fields for Beauce, France ({YEAR}) -> {output_path}")
    gdf = agribound.delineate(
        study_area=study_area,
        source=SOURCE,
        year=YEAR,
        engine=ENGINE,
        output_path=str(output_path),
        gee_project=args.gee_project,
        min_area=5000,  # m^2; the minimum polygon area chosen for this example
        simplify=2.5,
    )

    meta = gdf.attrs.get("engine_meta", {})
    print(f"\nFTW model: {meta.get('model')} ({meta.get('n_windows')} window(s))")
    for key, window in (meta.get("windows") or {}).items():
        if isinstance(window, dict) and "start" in window:
            print(f"  Window {key.upper()}: {window['start']} to {window['end']}")
    print(f"Delineated {len(gdf)} fields")
    if len(gdf) and "metrics:area" in gdf.columns:
        print(f"Total area: {gdf['metrics:area'].sum() / 10000:,.1f} ha")
        print(f"Mean field size: {gdf['metrics:area'].mean() / 10000:,.1f} ha")

    web_map = agribound.show_boundaries(
        map_ready(gdf),
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_beauce.html"),
    )
    show_in_notebook(web_map)
    print(f"\nMap saved to {OUTPUT_DIR / 'map_beauce.html'}")


if __name__ == "__main__":
    main()
