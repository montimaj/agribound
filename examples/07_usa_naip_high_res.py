"""
07 — USA Central Valley (California), NAIP 1m with Delineate-Anything

Delineates large commercial agricultural fields in California's Central Valley
using NAIP imagery at 1m resolution and the Delineate-Anything (YOLO) engine.

Estimated runtime: ~20–40 minutes (1 year, 1m resolution = large raster, GPU).

Prerequisites:
    pip install agribound[gee,delineate-anything]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
from pathlib import Path

import agribound

# --- Configuration ---
OUTPUT_DIR = Path("outputs/usa_central_valley")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "naip"
ENGINE = "delineate-anything"
YEAR = 2022


def create_study_area():
    """Create a study area GeoJSON in the Central Valley."""
    import json

    # AOI near Fresno, CA
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-119.85, 36.70],
                            [-119.75, 36.70],
                            [-119.75, 36.78],
                            [-119.85, 36.78],
                            [-119.85, 36.70],
                        ]
                    ],
                },
                "properties": {"name": "Central Valley AOI"},
            }
        ],
    }
    path = OUTPUT_DIR / "central_valley_aoi.geojson"
    with open(path, "w") as f:
        json.dump(aoi, f)
    return str(path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="USA Central Valley NAIP high-resolution field boundary delineation."
    )
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    """Run high-resolution NAIP field delineation."""
    args = parse_args()
    gee_project = args.gee_project

    study_area = create_study_area()

    print(f"Delineating fields from NAIP {YEAR} at 1m resolution...")
    output_path = OUTPUT_DIR / f"fields_naip_{YEAR}.gpkg"

    gdf = agribound.delineate(
        study_area=study_area,
        source=SOURCE,
        year=YEAR,
        engine=ENGINE,
        output_path=str(output_path),
        gee_project=gee_project,
        # Large commercial fields
        min_area=10000,
        simplify=3.0,
        n_workers=8,
    )

    print(f"\nDelineated {len(gdf)} fields")
    if "metrics:area" in gdf.columns:
        area_ha = gdf["metrics:area"].sum() / 10000
        print(f"Total agricultural area: {area_ha:,.1f} ha")
        print(f"Average field size: {gdf['metrics:area'].mean() / 10000:,.1f} ha")

    # --- Visualization ---
    agribound.show_boundaries(
        gdf,
        basemap="Google.Satellite",
        output_html=str(OUTPUT_DIR / "map_central_valley.html"),
    )
    print(f"\nMap saved to {OUTPUT_DIR / 'map_central_valley.html'}")


if __name__ == "__main__":
    main()
