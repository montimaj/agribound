"""
05 — Argentine Pampas, Google/TESSERA Embeddings (CPU-only)

Delineates agricultural fields in a subset of the Argentine Pampas (within the
Rio de la Plata basin) using pre-computed satellite embeddings (Google 64-D and
TESSERA 128-D) with unsupervised K-means clustering. No GPU required.

The study area is a ~50 km bbox over the Pampas agricultural heartland near
Pergamino, Buenos Aires Province — small enough for practical embedding
downloads (~1–2 GB) while still covering large-scale cropping.

Estimated runtime: ~10–20 minutes (5 years, CPU only).

Prerequisites:
    pip install agribound[gee,tessera]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="geedim")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="geedim")

import agribound

# --- Configuration ---
OUTPUT_DIR = Path("outputs/pampas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Study area: ~50 km bbox over the Argentine Pampas near Pergamino, Buenos Aires
# Covers intensive soybean/corn/wheat cropping in the Pampas agricultural heartland
STUDY_AREA_BBOX = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-60.8, -34.0],
                        [-60.3, -34.0],
                        [-60.3, -33.5],
                        [-60.8, -33.5],
                        [-60.8, -34.0],
                    ]
                ],
            },
            "properties": {"name": "Pampas (Pergamino, Buenos Aires)"},
        }
    ],
}

YEARS = range(2020, 2025)

# Set to True to refine boundaries with SAM2
SAM_REFINE = True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Argentine Pampas embedding-based field boundary delineation."
    )
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    """Run embedding-based field delineation for the Argentine Pampas."""
    args = parse_args()
    gee_project = args.gee_project

    # Write study area bbox to GeoJSON
    import json

    study_area_path = str(OUTPUT_DIR / "study_area.geojson")
    with open(study_area_path, "w") as f:
        json.dump(STUDY_AREA_BBOX, f)

    all_results = {}

    # --- Google Satellite Embeddings ---
    print("=" * 60)
    print("Part 1: Google Satellite Embeddings (64-D)")
    print("=" * 60)

    for year in YEARS:
        print(f"\nProcessing {year} (Google embeddings)...")
        output_path = OUTPUT_DIR / f"fields_google_{year}.gpkg"

        gdf = agribound.delineate(
            study_area=study_area_path,
            source="google-embedding",
            year=year,
            engine="embedding",
            output_path=str(output_path),
            gee_project=gee_project,
            device="cpu",
            min_area=5000,
            engine_params={"sam_refine": SAM_REFINE},
        )
        all_results[f"google_{year}"] = gdf
        print(f"  {year}: {len(gdf)} fields delineated")

    # --- TESSERA Embeddings ---
    print(f"\n{'=' * 60}")
    print("Part 2: TESSERA Embeddings (128-D)")
    print("=" * 60)

    for year in YEARS:
        print(f"\nProcessing {year} (TESSERA embeddings)...")
        output_path = OUTPUT_DIR / f"fields_tessera_{year}.gpkg"

        gdf = agribound.delineate(
            study_area=study_area_path,
            source="tessera-embedding",
            year=year,
            engine="embedding",
            output_path=str(output_path),
            gee_project=gee_project,
            device="cpu",
            min_area=5000,
            engine_params={"sam_refine": SAM_REFINE},
        )
        all_results[f"tessera_{year}"] = gdf
        print(f"  {year}: {len(gdf)} fields delineated")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  {'Source':<20} {'Year':<6} {'Fields':>6} {'Area (ha)':>12}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 12}")
    for key, gdf in sorted(all_results.items()):
        area_ha = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        source, year = key.rsplit("_", 1)
        print(f"  {source:<20} {year:<6} {len(gdf):>6} {area_ha:>12,.1f}")

    # --- Comparison: Google vs TESSERA for latest year ---
    latest = max(YEARS)
    google_gdf = all_results.get(f"google_{latest}")
    tessera_gdf = all_results.get(f"tessera_{latest}")

    if google_gdf is not None and tessera_gdf is not None:
        from agribound.visualize import show_comparison

        show_comparison(
            [google_gdf, tessera_gdf],
            labels=[f"Google {latest}", f"TESSERA {latest}"],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_comparison.html"),
        )
        print(f"\nComparison map saved to {OUTPUT_DIR / 'map_comparison.html'}")

    # --- Time series for Google embeddings ---
    google_gdfs = [all_results[f"google_{y}"] for y in YEARS if f"google_{y}" in all_results]
    if len(google_gdfs) >= 2:
        from agribound.visualize import show_comparison

        show_comparison(
            google_gdfs,
            labels=[str(y) for y in YEARS],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_timeseries.html"),
        )
        print(f"Time series map saved to {OUTPUT_DIR / 'map_timeseries.html'}")


if __name__ == "__main__":
    main()
