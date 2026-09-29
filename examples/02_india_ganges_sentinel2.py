"""
02 — India, Nadia District (West Bengal): Four Label-Free Approaches

Runs four approaches that need no training labels on the same study area in
Nadia district, West Bengal:

    1. FTW on Sentinel-2 (2024) — the global default FTW model
       (``FTW_PRUE_EFNET_B5``, two seasonal windows). agribound has no
       country-specific FTW models; the FTW benchmark the model was trained
       on includes India (``ftw_tools.settings.ALL_COUNTRIES``).
    2. Google Satellite Embedding (AlphaEarth, 64-D, 2024) — K-means clustering.
    3. TESSERA v1 embeddings (128-D, 2024) — K-means clustering. TESSERA v1 is
       near-global only for 2024; all four v1 tiles of this box exist for 2024.
    4. SPOT 6/7 panchromatic (1.5 m) with Delineate-Anything — SPOT ends in
       November 2023, so this run uses 2020 (restricted access, see below).

Clustering (2, 3) gives land-cover segments, not field instances; the LULC
crop filter (on by default; Dynamic World is selected outside the US) removes
segments with a low crop value. Approaches 1-3 use 2024 and approach 4 uses
2020, so the comparison mixes years.

SPOT 6/7 (AIRBUS/SPOT6_7) is restricted to select Earth Engine users (internal
DRI use). If your project has no access, approach 4 fails and the script
continues with the other three.

Estimated runtime (not measured for 1.0): ~15-30 minutes with a GPU (FTW,
Delineate-Anything); the embedding runs use the CPU.

Prerequisites:
    pip install "agribound[gee,ftw,tessera,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/02_india_ganges_sentinel2.py
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

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")
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
OUTPUT_DIR = Path("outputs/india_nadia")
YEAR = 2024
SPOT_YEAR = 2020  # AIRBUS/SPOT6_7 covers 2012-10-17 to 2023-11-15
MIN_AREA = 100  # m^2; the minimum polygon area chosen for this example

# Study area: Nadia district, West Bengal
STUDY_AREA_BBOX = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [88.35, 23.35],
                        [88.50, 23.35],
                        [88.50, 23.50],
                        [88.35, 23.50],
                        [88.35, 23.35],
                    ]
                ],
            },
            "properties": {"name": "Nadia District, West Bengal"},
        }
    ],
}


def run(label, results, **kwargs):
    """Run agribound.delineate(); store the result under *label* or report the error."""
    try:
        gdf = agribound.delineate(**kwargs)
    except Exception as exc:
        print(f"  {label} failed: {type(exc).__name__}: {exc}")
        return None
    results[label] = gdf
    print(f"  {label}: {len(gdf)} polygons -> {kwargs['output_path']}")
    return gdf


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Nadia District: four label-free approaches.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument("--skip-spot", action="store_true", help="Skip the SPOT-Pan run.")
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
    study_area = str(OUTPUT_DIR / "nadia_aoi.geojson")
    Path(study_area).write_text(json.dumps(STUDY_AREA_BBOX))
    common = dict(study_area=study_area, gee_project=args.gee_project, min_area=MIN_AREA)
    results = {}

    print(f"{'=' * 60}\n1. FTW on Sentinel-2 ({YEAR})\n{'=' * 60}")
    gdf_ftw = run(
        "FTW (Sentinel-2)",
        results,
        **common,
        source="sentinel2",
        year=YEAR,
        engine="ftw",
        output_path=str(OUTPUT_DIR / f"fields_sentinel2_ftw_{YEAR}.gpkg"),
        cloud_cover_max=30,
        simplify=1.0,
    )
    if gdf_ftw is not None:
        meta = gdf_ftw.attrs.get("engine_meta", {})
        print(f"  FTW model: {meta.get('model')}")
        for key, window in (meta.get("windows") or {}).items():
            if isinstance(window, dict) and "start" in window:
                print(f"  Window {key.upper()}: {window['start']} to {window['end']}")

    print(f"\n{'=' * 60}\n2. Google Satellite Embedding ({YEAR})\n{'=' * 60}")
    run(
        "Google embedding",
        results,
        **common,
        source="google-embedding",
        year=YEAR,
        engine="embedding",
        output_path=str(OUTPUT_DIR / f"fields_google-embedding_embedding_{YEAR}.gpkg"),
        device="cpu",
    )

    print(f"\n{'=' * 60}\n3. TESSERA v1 embeddings ({YEAR})\n{'=' * 60}")
    run(
        "TESSERA embedding",
        results,
        **common,
        source="tessera-embedding",
        tessera_version="v1",
        year=YEAR,
        engine="embedding",
        output_path=str(OUTPUT_DIR / f"fields_tessera-embedding_embedding_{YEAR}.gpkg"),
        device="cpu",
        engine_params={"n_clusters": 8},
    )

    if not args.skip_spot:
        print(f"\n{'=' * 60}\n4. SPOT-Pan 1.5 m + Delineate-Anything ({SPOT_YEAR})\n{'=' * 60}")
        run(
            f"SPOT-Pan DA ({SPOT_YEAR})",
            results,
            **common,
            source="spot-pan",
            year=SPOT_YEAR,
            engine="delineate-anything",
            output_path=str(OUTPUT_DIR / f"fields_spot-pan_delineate-anything_{SPOT_YEAR}.gpkg"),
            cloud_cover_max=15,
            simplify=1.0,
        )

    print(f"\n{'=' * 60}\nComparison\n{'=' * 60}")
    print(f"  {'Method':<30} {'Polygons':>9} {'Area (ha)':>12}")
    for label, gdf in results.items():
        area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0.0
        print(f"  {label:<30} {len(gdf):>9} {area:>12,.1f}")

    if not results:
        print("\nNo run succeeded; no map written.")
        return
    from agribound.visualize import show_comparison

    map_path = OUTPUT_DIR / "map_ftw_google_tessera_spot.html"
    web_map = show_comparison(
        list(results.values()),
        labels=list(results.keys()),
        basemap="Esri.WorldImagery",
        output_html=str(map_path),
    )
    show_in_notebook(web_map)
    print(f"\n  Map: {map_path}")


if __name__ == "__main__":
    main()
