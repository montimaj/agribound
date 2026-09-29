"""
11 — Mississippi Alluvial Plain: SPOT 6/7 Time Series with Delineate-Anything

Delineates fields near Greenville (Washington County, Mississippi) from
SPOT 6/7 multispectral composites (6 m) for 2021, 2022 and 2023 with the
Delineate-Anything engine (default model ``large_v2``, label-free), then
measures how much the boundaries of consecutive years agree.

Data:
    - SPOT 6/7 (AIRBUS/SPOT6_7), scenes with <= 15 % cloud cover. On
      2026-09-27 the footprints of those scenes covered the whole box in each
      year (16 scenes in 2021, 2 in 2022, 12 in 2023). Composites hold
      per-band medians of raw digital numbers (radiometry unverified).
    - Study area: 0.2 x 0.15 degrees (91.1-90.9 W, 33.30-33.45 N); USDA CDL
      2022 marks ~62 % of it as cultivated (query on 2026-09-27).

Year-to-year agreement: ``agribound.evaluate.evaluate(later, earlier)`` with
one-to-one IoU matching (IoU >= 0.5). This is agreement between two
predictions, not accuracy: neither year is a reference.

SPOT 6/7 access is restricted to select Earth Engine users (internal DRI
use); external users can contact the package author to request processing.
The LULC crop filter is on (NLCD is selected in the conterminous US).
Each year writes its own output file.

Estimated runtime (not measured for 1.0): ~10-20 minutes per year (6 m, GPU).

Prerequisites:
    pip install "agribound[gee,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/11_mississippi_alluvial_plain_spot.py
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import agribound
from agribound.evaluate import evaluate

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
OUTPUT_DIR = Path("outputs/mississippi_alluvial_plain")
SOURCE = "spot"
ENGINE = "delineate-anything"
YEARS = [2021, 2022, 2023]

AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-91.10, 33.30],
                        [-90.90, 33.30],
                        [-90.90, 33.45],
                        [-91.10, 33.45],
                        [-91.10, 33.30],
                    ]
                ],
            },
            "properties": {"name": "Mississippi Alluvial Plain AOI (near Greenville, MS)"},
        }
    ],
}


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Mississippi Alluvial Plain: SPOT time series.")
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
    study_area = str(OUTPUT_DIR / "map_aoi.geojson")
    Path(study_area).write_text(json.dumps(AOI))

    all_results = {}
    for year in YEARS:
        output_path = OUTPUT_DIR / f"fields_{SOURCE}_{ENGINE}_{year}.gpkg"
        print(f"\n{'=' * 60}\n{year}: SPOT 6/7 -> {output_path}\n{'=' * 60}")
        try:
            gdf = agribound.delineate(
                study_area=study_area,
                source=SOURCE,
                year=year,
                engine=ENGINE,
                output_path=str(output_path),
                gee_project=args.gee_project,
                cloud_cover_max=15,
                min_area=10000,  # m^2; the minimum polygon area chosen for this example
                simplify=3.0,
            )
        except Exception as exc:
            print(f"  {year} failed: {type(exc).__name__}: {exc}")
            print(
                "  If this is an Earth Engine permission error: SPOT 6/7 is restricted to "
                "select users; contact the agribound author for processing."
            )
            continue
        all_results[year] = gdf
        n = len(gdf)
        mean_ha = gdf["metrics:area"].mean() / 10000 if n else 0.0
        print(f"  {n} fields, mean {mean_ha:,.1f} ha")

    if len(all_results) >= 2:
        print(f"\n{'=' * 60}\nYear-to-year agreement (one-to-one, IoU >= 0.5)\n{'=' * 60}")
        years_done = sorted(all_results)
        for y1, y2 in zip(years_done[:-1], years_done[1:], strict=True):
            m = evaluate(all_results[y2], all_results[y1])
            print(
                f"  {y1} -> {y2}: matched {m['count_tp']} of {m['count_reference']} "
                f"({y1}) and {m['count_predicted']} ({y2}) polygons; "
                f"F1={m['f1']:.3f}, mean IoU of matches={m['iou_mean']:.3f}"
            )

    if not all_results:
        print("\nNo year succeeded; no maps written.")
        return
    from agribound.visualize import show_comparison

    web_map = show_comparison(
        [all_results[y] for y in sorted(all_results)],
        labels=[str(y) for y in sorted(all_results)],
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_spot_timeseries.html"),
    )
    show_in_notebook(web_map)
    print(f"\nTime series map: {OUTPUT_DIR / 'map_spot_timeseries.html'}")
    web_map = agribound.show_boundaries(
        map_ready(all_results[max(all_results)]),
        basemap="Google.Satellite",
        output_html=str(OUTPUT_DIR / "map_spot_latest.html"),
    )
    show_in_notebook(web_map)
    print(f"Latest year map: {OUTPUT_DIR / 'map_spot_latest.html'}")


if __name__ == "__main__":
    main()
