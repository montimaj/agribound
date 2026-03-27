"""
09 — Ensemble Comparison: Multiple Engines on the Same AOI

Runs multiple delineation engines on the same study area and compares
results. Demonstrates the ensemble engine with different merge strategies
and per-engine visual comparison.

Estimated runtime: ~30–60 minutes (runs 2–3 engines, GPU recommended).

Prerequisites:
    pip install agribound[gee,delineate-anything,ftw,geoai]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
from pathlib import Path

import agribound

# --- Configuration ---
OUTPUT_DIR = Path("outputs/ensemble_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "sentinel2"
YEAR = 2024
ENGINES = ["delineate-anything", "ftw", "geoai"]
SAM_REFINE = True


def create_study_area():
    """Create a moderate-sized study area."""
    import json

    # AOI in southern Spain (Andalusia) — mix of field sizes
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-3.80, 37.75],
                            [-3.65, 37.75],
                            [-3.65, 37.85],
                            [-3.80, 37.85],
                            [-3.80, 37.75],
                        ]
                    ],
                },
                "properties": {"name": "Andalusia AOI"},
            }
        ],
    }
    path = OUTPUT_DIR / "andalusia_aoi.geojson"
    with open(path, "w") as f:
        json.dump(aoi, f)
    return str(path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ensemble comparison of multiple delineation engines."
    )
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    """Compare multiple engines and run ensemble."""
    args = parse_args()
    gee_project = args.gee_project

    study_area = create_study_area()
    results = {}

    # --- Run each engine individually ---
    for engine_name in ENGINES:
        print(f"\n{'=' * 60}")
        print(f"Running engine: {engine_name}")
        print(f"{'=' * 60}")

        output_path = OUTPUT_DIR / f"fields_{engine_name}.gpkg"

        try:
            gdf = agribound.delineate(
                study_area=study_area,
                source=SOURCE,
                year=YEAR,
                engine=engine_name,
                output_path=str(output_path),
                gee_project=gee_project,
                engine_params={"sam_refine": SAM_REFINE},
                min_area=2500,
                simplify=2.0,
            )
            results[engine_name] = gdf
            print(f"  {engine_name}: {len(gdf)} fields")
        except Exception as exc:
            print(f"  {engine_name} failed: {exc}")

    # --- Run ensemble engine ---
    print(f"\n{'=' * 60}")
    print("Running ensemble (vote strategy)")
    print(f"{'=' * 60}")

    try:
        ensemble_gdf = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=YEAR,
            engine="ensemble",
            output_path=str(OUTPUT_DIR / "fields_ensemble.gpkg"),
            gee_project=gee_project,
            engine_params={
                "engines": ENGINES,
                "merge_strategy": "vote",
                "sam_refine": SAM_REFINE,
            },
            min_area=2500,
        )
        results["ensemble"] = ensemble_gdf
        print(f"  ensemble: {len(ensemble_gdf)} fields")
    except Exception as exc:
        print(f"  Ensemble failed: {exc}")

    # --- Comparison summary ---
    print(f"\n{'=' * 60}")
    print("Engine Comparison Summary")
    print(f"{'=' * 60}")
    for name, gdf in results.items():
        n = len(gdf)
        avg_area = gdf["metrics:area"].mean() / 10000 if "metrics:area" in gdf.columns else 0
        print(f"  {name:<25} {n:>5} fields, avg {avg_area:>6.1f} ha")

    # --- Visualization ---
    if len(results) >= 2:
        from agribound.visualize import show_comparison

        show_comparison(
            list(results.values()),
            labels=list(results.keys()),
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_ensemble_comparison.html"),
        )
        print(f"\nComparison map saved to {OUTPUT_DIR / 'map_ensemble_comparison.html'}")


if __name__ == "__main__":
    main()
