"""
03 — Australia Murray-Darling Basin, HLS with Prithvi Engine

Delineates large-scale irrigated agricultural fields in the Murray-Darling
Basin using Harmonized Landsat-Sentinel (HLS) imagery and the Prithvi
foundation model in embedding mode.

Estimated runtime: ~45–90 minutes (3 years, GPU recommended).

Prerequisites:
    pip install agribound[gee,prithvi]
    agribound auth --project YOUR_GEE_PROJECT
"""

from pathlib import Path

import agribound

# --- Configuration ---
GEE_PROJECT = "your-gee-project"
OUTPUT_DIR = Path("outputs/australia_murray_darling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2022, 2025)
SOURCE = "hls"
ENGINE = "prithvi"


def create_study_area():
    """Create a study area GeoJSON in the Murray-Darling Basin."""
    import json

    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [145.50, -35.00],
                            [145.65, -35.00],
                            [145.65, -34.85],
                            [145.50, -34.85],
                            [145.50, -35.00],
                        ]
                    ],
                },
                "properties": {"name": "Murray-Darling Basin AOI"},
            }
        ],
    }
    path = OUTPUT_DIR / "murray_darling_aoi.geojson"
    with open(path, "w") as f:
        json.dump(aoi, f)
    return str(path)


def main():
    """Run field delineation for the Murray-Darling Basin."""
    study_area = create_study_area()
    all_results = {}

    for year in YEARS:
        print(f"\nProcessing {year}...")
        output_path = OUTPUT_DIR / f"fields_hls_{year}.gpkg"

        gdf = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=year,
            engine=ENGINE,
            output_path=str(output_path),
            gee_project=GEE_PROJECT,
            composite_method="median",
            # Large irrigated fields
            min_area=5000,
            simplify=3.0,
        )
        all_results[year] = gdf
        print(f"  {year}: {len(gdf)} fields delineated")

    # --- Visualization ---
    if all_results:
        from agribound.visualize import show_comparison

        boundaries = list(all_results.values())
        labels = [str(y) for y in all_results.keys()]

        m = show_comparison(
            boundaries,
            labels=labels,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_murray_darling.html"),
        )
        print(f"\nComparison map saved to {OUTPUT_DIR / 'map_murray_darling.html'}")


if __name__ == "__main__":
    main()
