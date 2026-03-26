"""
08 — China North Plain, SPOT 6/7 with Delineate-Anything

Delineates agricultural fields in the North China Plain using SPOT 6/7
imagery at 6m resolution. SPOT access is restricted to select GEE users
(internal DRI use only). External users can contact the package author
to request processing on their behalf.

Estimated runtime: ~15–30 minutes (1 year, 6m resolution, GPU).

Prerequisites:
    pip install agribound[gee,delineate-anything]
    agribound auth --project YOUR_GEE_PROJECT

Note:
    SPOT 6/7 GEE access is restricted. If you receive an access error,
    contact the agribound author to request field boundary processing
    for your study area.
"""

from pathlib import Path

import agribound

# --- Configuration ---
GEE_PROJECT = "your-gee-project"
OUTPUT_DIR = Path("outputs/china_north_plain")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = "spot"
ENGINE = "delineate-anything"
YEAR = 2022


def create_study_area():
    """Create a study area GeoJSON in the North China Plain."""
    import json

    # AOI near Shijiazhuang, Hebei Province
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [114.40, 37.95],
                            [114.55, 37.95],
                            [114.55, 38.05],
                            [114.40, 38.05],
                            [114.40, 37.95],
                        ]
                    ],
                },
                "properties": {"name": "North China Plain AOI"},
            }
        ],
    }
    path = OUTPUT_DIR / "north_china_aoi.geojson"
    with open(path, "w") as f:
        json.dump(aoi, f)
    return str(path)


def main():
    """Run SPOT-based field delineation for the North China Plain."""
    study_area = create_study_area()

    print(f"Delineating fields from SPOT 6/7 ({YEAR})...")
    print("Note: SPOT access is restricted. See docstring for details.")
    output_path = OUTPUT_DIR / f"fields_spot_{YEAR}.gpkg"

    try:
        gdf = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=YEAR,
            engine=ENGINE,
            output_path=str(output_path),
            gee_project=GEE_PROJECT,
            cloud_cover_max=15,
            min_area=3000,
            simplify=2.0,
        )

        print(f"\nDelineated {len(gdf)} fields")
        if "metrics:area" in gdf.columns:
            area_ha = gdf["metrics:area"].sum() / 10000
            print(f"Total agricultural area: {area_ha:,.1f} ha")

        # --- Visualization ---
        m = agribound.show_boundaries(
            gdf,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_north_china.html"),
        )
        print(f"\nMap saved to {OUTPUT_DIR / 'map_north_china.html'}")

    except Exception as exc:
        print(f"\nSPOT access error: {exc}")
        print(
            "SPOT 6/7 is restricted to select GEE users. "
            "Contact the agribound author for processing assistance."
        )


if __name__ == "__main__":
    main()
