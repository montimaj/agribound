"""
11 — Mississippi Alluvial Plain, SPOT 6/7 with Delineate-Anything

Delineates agricultural fields in the Mississippi Alluvial Plain (MAP) using
SPOT 6/7 imagery at 6m resolution and the Delineate-Anything engine. The MAP
is one of the most productive agricultural regions in the US, dominated by
row crops (cotton, soybeans, rice, corn) with large, regular field patterns.

Estimated runtime: ~15–30 minutes (1 year, 6m resolution, GPU).

Prerequisites:
    pip install agribound[gee,delineate-anything]
    agribound auth --project YOUR_GEE_PROJECT

Note:
    SPOT 6/7 GEE access is restricted to select users (internal DRI use
    only). External users can contact the agribound author to request
    field boundary processing for their study area.
"""

import argparse
from pathlib import Path

import agribound
from agribound.evaluate import evaluate

# --- Configuration ---
OUTPUT_DIR = Path("outputs/mississippi_alluvial_plain")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "spot"
ENGINE = "delineate-anything"
YEARS = [2021, 2022, 2023]


def create_study_area():
    """Create a study area GeoJSON in the Mississippi Alluvial Plain."""
    import json

    # AOI in the central MAP near Greenville, Mississippi
    aoi = {
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
                "properties": {"name": "Mississippi Alluvial Plain AOI"},
            }
        ],
    }
    path = OUTPUT_DIR / "map_aoi.geojson"
    with open(path, "w") as f:
        json.dump(aoi, f)
    return str(path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Mississippi Alluvial Plain SPOT 6/7 field boundary delineation."
    )
    parser.add_argument(
        "--gee-project", default=None, help="GEE project ID (auto-detected from gcloud config if not set)."
    )
    return parser.parse_args()


def main():
    """Run SPOT-based field delineation for the Mississippi Alluvial Plain."""
    args = parse_args()
    gee_project = args.gee_project

    study_area = create_study_area()
    all_results = {}

    for year in YEARS:
        print(f"\n{'='*60}")
        print(f"Processing {year} — SPOT 6/7 at 6m resolution")
        print(f"{'='*60}")

        output_path = OUTPUT_DIR / f"fields_spot_{year}.gpkg"

        try:
            gdf = agribound.delineate(
                study_area=study_area,
                source=SOURCE,
                year=year,
                engine=ENGINE,
                output_path=str(output_path),
                gee_project=gee_project,
                cloud_cover_max=15,
                # Large row-crop fields in the MAP
                min_area=10000,
                simplify=3.0,
                device="auto",
            )
            all_results[year] = gdf
            print(f"  {year}: {len(gdf)} fields delineated")

            if "metrics:area" in gdf.columns:
                area_ha = gdf["metrics:area"].sum() / 10000
                avg_ha = gdf["metrics:area"].mean() / 10000
                print(f"  Total area: {area_ha:,.1f} ha")
                print(f"  Average field size: {avg_ha:,.1f} ha")

        except Exception as exc:
            print(f"\n  SPOT access error: {exc}")
            print(
                "  SPOT 6/7 is restricted to select GEE users. "
                "Contact the agribound author for processing assistance."
            )
            return

    # --- Cross-year comparison ---
    if len(all_results) >= 2:
        print(f"\n{'='*60}")
        print("Cross-Year Comparison")
        print(f"{'='*60}")
        print(f"  {'Year':<6} {'Fields':>6} {'Total (ha)':>12} {'Avg (ha)':>10}")
        print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*10}")
        for year, gdf in sorted(all_results.items()):
            area_ha = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
            avg_ha = gdf["metrics:area"].mean() / 10000 if "metrics:area" in gdf.columns else 0
            print(f"  {year:<6} {len(gdf):>6} {area_ha:>12,.1f} {avg_ha:>10,.1f}")

        # Cross-year evaluation: how stable are boundaries year to year?
        years_sorted = sorted(all_results.keys())
        for i in range(len(years_sorted) - 1):
            y1, y2 = years_sorted[i], years_sorted[i + 1]
            metrics = evaluate(all_results[y2], all_results[y1])
            print(f"\n  Stability {y1}→{y2}: "
                  f"IoU={metrics['iou_mean']:.3f} "
                  f"F1={metrics['f1']:.3f} "
                  f"P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f}")

    # --- Visualization ---
    if all_results:
        from agribound.visualize import show_comparison

        # Multi-year overlay
        m = show_comparison(
            list(all_results.values()),
            labels=[str(y) for y in sorted(all_results.keys())],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_spot_timeseries.html"),
        )
        print(f"\nTime series map saved to {OUTPUT_DIR / 'map_spot_timeseries.html'}")

        # Latest year standalone
        latest_year = max(all_results.keys())
        m_latest = agribound.show_boundaries(
            all_results[latest_year],
            basemap="Google.Satellite",
            output_html=str(OUTPUT_DIR / "map_spot_latest.html"),
        )
        print(f"Latest year map saved to {OUTPUT_DIR / 'map_spot_latest.html'}")


if __name__ == "__main__":
    main()
