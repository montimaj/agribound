"""
02 — India Ganges Plain, Sentinel-2 with FTW Engine

Delineates smallholder agricultural fields in the Ganges Plain using
Sentinel-2 imagery and the FTW (Fields of the World) engine with a
country-specific model for India.

Estimated runtime: ~30–60 minutes (5 years, small AOI, GPU).

Prerequisites:
    pip install agribound[gee,ftw]
    agribound auth --project YOUR_GEE_PROJECT
"""

from pathlib import Path

import agribound

# --- Configuration ---
GEE_PROJECT = "your-gee-project"
OUTPUT_DIR = Path("outputs/india_ganges")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Small AOI in the Ganges Plain (Uttar Pradesh)
STUDY_AREA_GEOJSON = OUTPUT_DIR / "ganges_aoi.geojson"

YEARS = range(2020, 2025)
SOURCE = "sentinel2"
ENGINE = "ftw"


def create_study_area():
    """Create a small study area GeoJSON in the Ganges Plain."""
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
                            [80.90, 26.80],
                            [81.00, 26.80],
                            [81.00, 26.90],
                            [80.90, 26.90],
                            [80.90, 26.80],
                        ]
                    ],
                },
                "properties": {"name": "Ganges Plain AOI"},
            }
        ],
    }
    with open(STUDY_AREA_GEOJSON, "w") as f:
        json.dump(aoi, f)
    return str(STUDY_AREA_GEOJSON)


def main():
    """Run field boundary delineation for the Ganges Plain."""
    study_area = create_study_area()
    all_results = {}

    for year in YEARS:
        print(f"\nProcessing {year}...")
        output_path = OUTPUT_DIR / f"fields_s2_{year}.gpkg"

        gdf = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=year,
            engine=ENGINE,
            output_path=str(output_path),
            gee_project=GEE_PROJECT,
            composite_method="median",
            cloud_cover_max=30,
            # Smallholder fields — lower minimum area
            min_area=500,
            simplify=1.0,
        )
        all_results[year] = gdf
        print(f"  {year}: {len(gdf)} fields delineated")

    # --- Visualization ---
    if all_results:
        latest_year = max(all_results.keys())
        m = agribound.show_boundaries(
            all_results[latest_year],
            basemap="Google.Satellite",
            output_html=str(OUTPUT_DIR / "map_ganges.html"),
        )
        print(f"\nMap saved to {OUTPUT_DIR / 'map_ganges.html'}")


if __name__ == "__main__":
    main()
