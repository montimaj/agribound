"""
04 — France Beauce Region, Sentinel-2 with GeoAI Engine

Delineates large-field European agriculture in the Beauce region of France
using Sentinel-2 imagery and geoai's AgricultureFieldDelineator (Mask R-CNN).

Estimated runtime: ~15–30 minutes (1 year, GPU).

Prerequisites:
    pip install agribound[gee,geoai]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
from pathlib import Path

import agribound

# --- Configuration ---
OUTPUT_DIR = Path("outputs/france_beauce")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "sentinel2"
ENGINE = "geoai"
YEAR = 2023

# Set to True to refine boundaries with SAM2
SAM_REFINE = True


def create_study_area():
    """Create a study area GeoJSON in the Beauce region."""
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
                            [1.40, 48.10],
                            [1.55, 48.10],
                            [1.55, 48.20],
                            [1.40, 48.20],
                            [1.40, 48.10],
                        ]
                    ],
                },
                "properties": {"name": "Beauce AOI"},
            }
        ],
    }
    path = OUTPUT_DIR / "beauce_aoi.geojson"
    with open(path, "w") as f:
        json.dump(aoi, f)
    return str(path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="France Beauce region Sentinel-2 field boundary delineation."
    )
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    """Run field delineation for the Beauce region."""
    args = parse_args()
    gee_project = args.gee_project

    study_area = create_study_area()

    print(f"Delineating fields for Beauce, France ({YEAR})...")
    output_path = OUTPUT_DIR / f"fields_s2_{YEAR}.gpkg"

    gdf = agribound.delineate(
        study_area=study_area,
        source=SOURCE,
        year=YEAR,
        engine=ENGINE,
        output_path=str(output_path),
        gee_project=gee_project,
        composite_method="median",
        # European large fields
        min_area=5000,
        simplify=2.5,
        engine_params={"sam_refine": SAM_REFINE},
    )

    print(f"\nDelineated {len(gdf)} fields")
    if "metrics:area" in gdf.columns:
        area_ha = gdf["metrics:area"].sum() / 10000
        avg_ha = gdf["metrics:area"].mean() / 10000
        print(f"Total area: {area_ha:,.1f} ha")
        print(f"Average field size: {avg_ha:,.1f} ha")

    # --- Visualization ---
    agribound.show_boundaries(
        gdf,
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_beauce.html"),
    )
    print(f"\nMap saved to {OUTPUT_DIR / 'map_beauce.html'}")


if __name__ == "__main__":
    main()
