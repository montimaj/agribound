"""
06 — Kenya Smallholder Fields, Sentinel-2 with FTW Engine

Delineates smallholder agricultural fields in Central Kenya using Sentinel-2
imagery and the FTW engine with a Kenya-specific model. Demonstrates
min_field_area tuning for smallholder agriculture.

Estimated runtime: ~10–20 minutes (1 year, small AOI, GPU).

Prerequisites:
    pip install agribound[gee,ftw]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
from pathlib import Path

import agribound

# --- Configuration ---
OUTPUT_DIR = Path("outputs/kenya_smallholder")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "sentinel2"
ENGINE = "ftw"
YEAR = 2023


def create_study_area():
    """Create a study area GeoJSON in Central Kenya."""
    import json

    # AOI near Nyeri, Central Kenya
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [36.90, -0.50],
                            [37.05, -0.50],
                            [37.05, -0.35],
                            [36.90, -0.35],
                            [36.90, -0.50],
                        ]
                    ],
                },
                "properties": {"name": "Central Kenya AOI"},
            }
        ],
    }
    path = OUTPUT_DIR / "kenya_aoi.geojson"
    with open(path, "w") as f:
        json.dump(aoi, f)
    return str(path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Kenya smallholder Sentinel-2 field boundary delineation."
    )
    parser.add_argument(
        "--gee-project", default=None, help="GEE project ID."
    )
    return parser.parse_args()


def main():
    """Run field delineation for Central Kenya with different area thresholds."""
    args = parse_args()
    gee_project = args.gee_project

    study_area = create_study_area()

    # Compare different min_area thresholds for smallholder fields
    thresholds = [100, 500, 1000, 2500]
    results = {}

    for min_area in thresholds:
        print(f"\nDelineating with min_area={min_area} m2...")
        output_path = OUTPUT_DIR / f"fields_minarea_{min_area}.gpkg"

        gdf = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=YEAR,
            engine=ENGINE,
            output_path=str(output_path),
            gee_project=gee_project,
            composite_method="median",
            min_area=min_area,
            simplify=1.0,
        )
        results[min_area] = gdf
        print(f"  min_area={min_area}: {len(gdf)} fields detected")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Effect of min_field_area on detection count")
    print(f"{'='*60}")
    for threshold, gdf in results.items():
        avg_area = gdf["metrics:area"].mean() if "metrics:area" in gdf.columns else 0
        print(f"  {threshold:>6} m2: {len(gdf):>5} fields, avg area {avg_area:>8,.0f} m2")

    # --- Visualization: overlay all thresholds ---
    from agribound.visualize import show_comparison

    show_comparison(
        list(results.values()),
        labels=[f"min={t} m2" for t in thresholds],
        basemap="Google.Satellite",
        output_html=str(OUTPUT_DIR / "map_kenya_thresholds.html"),
    )
    print(f"\nComparison map saved to {OUTPUT_DIR / 'map_kenya_thresholds.html'}")


if __name__ == "__main__":
    main()
