"""
10 — Local GeoTIFF Quickstart

Minimal example: load a local GeoTIFF file, run field boundary delineation,
and visualize results. No GEE authentication required.

Estimated runtime: ~2–5 minutes (single small TIF, GPU or CPU).

Prerequisites:
    pip install agribound[delineate-anything]
"""

from pathlib import Path

import agribound

# --- Configuration ---
# Replace with path to your GeoTIFF
LOCAL_TIF = "path/to/your/satellite_image.tif"
STUDY_AREA = "path/to/your/study_area.geojson"
OUTPUT_DIR = Path("outputs/local_quickstart")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Delineate field boundaries from a local GeoTIFF in 5 lines."""
    output_path = OUTPUT_DIR / "fields.gpkg"

    # --- Delineate in 5 lines ---
    gdf = agribound.delineate(
        study_area=STUDY_AREA,
        source="local",
        engine="delineate-anything",
        local_tif_path=LOCAL_TIF,
        output_path=str(output_path),
    )

    print(f"Delineated {len(gdf)} field boundaries")
    print(f"Output: {output_path}")

    # --- Optional: view results ---
    if len(gdf) > 0:
        # Print summary statistics
        if "metrics:area" in gdf.columns:
            print(f"Total area: {gdf['metrics:area'].sum() / 10000:,.1f} ha")
            print(f"Average field: {gdf['metrics:area'].mean() / 10000:,.2f} ha")
            print(f"Smallest field: {gdf['metrics:area'].min():,.0f} m2")
            print(f"Largest field: {gdf['metrics:area'].max() / 10000:,.1f} ha")

        # Interactive map with the local TIF as backdrop
        m = agribound.show_boundaries(
            gdf,
            satellite_tif=LOCAL_TIF,
            output_html=str(OUTPUT_DIR / "map.html"),
        )
        print(f"\nMap saved to {OUTPUT_DIR / 'map.html'}")


if __name__ == "__main__":
    main()
