"""
09 — Ensemble Comparison: Delineate-Anything and FTW on the Same AOI (Andalusia)

Runs two label-free engines on the same Sentinel-2 study area near Carmona
(Seville, Spain), then combines their saved outputs with the three merge
strategies of the ensemble engine:

    - ``intersection``: areas that every member covers;
    - ``union``: all members' polygons, duplicates fused (IoU >= 0.3 or
      containment >= 0.8);
    - ``vote``: pixels covered by at least ``min_votes`` members, polygonised.
      ``min_votes = max(min(2, n), ceil(vote_threshold * n))`` for the ``n``
      members with polygons, so with two members and ``vote_threshold=0.3``
      both must agree.

The merges here call the ensemble engine's static merge functions on the two
saved member outputs (no engine is re-run). They are followed only by the
area filter, not by the pipeline's smoothing, simplification, LULC filter or
metadata columns. On saved outputs the vote uses a 10 m grid over the
members' extent; ``engine="ensemble"`` votes on the input raster's own grid.
``--run-ensemble-engine`` also runs the full pipeline with
``engine="ensemble"`` (the members are run again inside it).

Data: Sentinel-2 L2A, 2024. Study area: a 0.15 x 0.1 degree box
(5.55-5.40 W, 37.4-37.5 N); ESA WorldCover 2021 classifies ~90 % of it as
cropland (query on 2026-09-27). The LULC crop filter is on for the member
runs (Dynamic World is selected outside the US).

Estimated runtime (not measured for 1.0): ~20-40 minutes (two engines, GPU
recommended).

Prerequisites:
    pip install "agribound[gee,delineate-anything,ftw]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/09_ensemble_comparison.py
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
OUTPUT_DIR = Path("outputs/ensemble_comparison")
SOURCE = "sentinel2"
YEAR = 2024
ENGINES = ["delineate-anything", "ftw"]
VOTE_THRESHOLD = 0.3
MIN_AREA_M2 = 2500

AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-5.55, 37.40],
                        [-5.40, 37.40],
                        [-5.40, 37.50],
                        [-5.55, 37.50],
                        [-5.55, 37.40],
                    ]
                ],
            },
            "properties": {"name": "Carmona AOI (Seville, Andalusia)"},
        }
    ],
}


def area_ha(gdf):
    """Total polygon area in hectares (equal-area EPSG:6933)."""
    from agribound.io.crs import get_equal_area_crs

    if len(gdf) == 0:
        return 0.0
    return float(gdf.geometry.to_crs(get_equal_area_crs()).area.sum() / 10000)


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Delineate-Anything + FTW ensemble merges.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument(
        "--run-ensemble-engine",
        action="store_true",
        help="Also run the full pipeline with engine='ensemble' (vote).",
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
    study_area = str(OUTPUT_DIR / "andalusia_aoi.geojson")
    Path(study_area).write_text(json.dumps(AOI))
    common = dict(
        study_area=study_area,
        source=SOURCE,
        year=YEAR,
        gee_project=args.gee_project,
        min_area=MIN_AREA_M2,
        simplify=2.0,
    )

    # --- Members ---------------------------------------------------------------
    members = {}
    for engine_name in ENGINES:
        print(f"\n{'=' * 60}\nEngine: {engine_name}\n{'=' * 60}")
        output_path = OUTPUT_DIR / f"fields_{SOURCE}_{engine_name}_{YEAR}.gpkg"
        try:
            gdf = agribound.delineate(**common, engine=engine_name, output_path=str(output_path))
        except Exception as exc:
            print(f"  {engine_name} failed: {type(exc).__name__}: {exc}")
            continue
        members[engine_name] = gdf
        print(f"  {engine_name}: {len(gdf)} fields -> {output_path}")

    results = dict(members)

    # --- Merge strategies on the saved member outputs -------------------------------
    if len(members) >= 2:
        from agribound.engines.ensemble import EnsembleEngine
        from agribound.postprocess import filter_polygons

        merged = {
            "intersection": EnsembleEngine._merge_intersection(members),
            "union": EnsembleEngine._merge_union(members),
            "vote": EnsembleEngine._merge_vote(members, threshold=VOTE_THRESHOLD),
        }
        for strategy, gdf in merged.items():
            gdf = filter_polygons(gdf, min_area_m2=MIN_AREA_M2)
            path = OUTPUT_DIR / f"fields_{SOURCE}_merge-{strategy}_{YEAR}.gpkg"
            if len(gdf):
                gdf.to_file(path, driver="GPKG", layer="fields")
            results[f"merge: {strategy}"] = gdf
            print(f"  merge '{strategy}': {len(gdf)} polygons")
    else:
        print("\nFewer than two members succeeded; no merges.")

    # --- Optional: the ensemble engine in the full pipeline -----------------------------
    if args.run_ensemble_engine:
        print(f"\n{'=' * 60}\nengine='ensemble' (vote) in the pipeline\n{'=' * 60}")
        output_path = OUTPUT_DIR / f"fields_{SOURCE}_ensemble-vote_{YEAR}.gpkg"
        gdf = agribound.delineate(
            **common,
            engine="ensemble",
            output_path=str(output_path),
            engine_params={
                "engines": ENGINES,
                "merge_strategy": "vote",
                "vote_threshold": VOTE_THRESHOLD,
            },
        )
        results["engine=ensemble (vote)"] = gdf
        print(f"  {len(gdf)} fields -> {output_path}")

    # --- Summary and map ------------------------------------------------------------
    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    for name, gdf in results.items():
        n = len(gdf)
        mean_ha = area_ha(gdf) / n if n else 0.0
        print(f"  {name:<28} {n:>6} polygons, mean {mean_ha:>6.1f} ha")

    if not results:
        print("\nNo run succeeded; no map written.")
        return
    from agribound.visualize import show_comparison

    web_map = show_comparison(
        list(results.values()),
        labels=list(results.keys()),
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_ensemble_comparison.html"),
    )
    show_in_notebook(web_map)
    print(f"\nComparison map: {OUTPUT_DIR / 'map_ensemble_comparison.html'}")


if __name__ == "__main__":
    main()
