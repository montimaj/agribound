"""
03 — Australia Murray-Darling Basin, HLS with Prithvi Engine

Compares two Prithvi-based delineation modes on the Murray-Darling Basin
using Harmonized Landsat-Sentinel (HLS) imagery:

    1. Prithvi ViT embedding — full encoder feature extraction + clustering
    2. PCA baseline — spectral PCA + clustering (no model, CPU only)

Both run on the same AOI for 2022–2024.

Note: The ViT embedding mode (without fine-tuning) tends to produce very
few, over-merged fields because the raw encoder features do not
distinguish individual field boundaries well. PCA mode typically produces
far more realistic results out of the box. For production use with ViT
embeddings, fine-tuning on reference boundaries (``fine_tune=True``) is
recommended — see Example 01 for a fine-tuning workflow.

Estimated runtime: ~45–90 minutes (3 years, GPU recommended for ViT mode).

Prerequisites:
    pip install agribound[gee,prithvi]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
import logging
from pathlib import Path

import agribound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/australia_murray_darling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2022, 2025)
YEARS = range(2022, 2023)  # quick test for 1 year; comment this out for full 3-year run
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
                            [149.70, -30.30],
                            [149.85, -30.30],
                            [149.85, -30.15],
                            [149.70, -30.15],
                            [149.70, -30.30],
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Australia Murray-Darling Basin HLS field boundary delineation."
    )
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    """Run field delineation for the Murray-Darling Basin.

    Compares two Prithvi modes on the same AOI:
    1. **ViT embedding** — full Prithvi encoder (requires ``transformers``)
    2. **PCA baseline** — spectral PCA clustering (no model needed)
    """
    args = parse_args()
    gee_project = args.gee_project

    study_area = create_study_area()
    comparison_results = {}

    for year in YEARS:
        print(f"\n{'=' * 60}")
        print(f"Year {year}")
        print(f"{'=' * 60}")

        # ── Prithvi ViT embedding mode ────────────────────────────
        print("\n  Mode 1: Prithvi ViT embeddings")
        vit_path = OUTPUT_DIR / f"fields_hls_vit_{year}.gpkg"

        gdf_vit = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=year,
            engine=ENGINE,
            output_path=str(vit_path),
            gee_project=gee_project,
            composite_method="median",
            min_area=5000,
            simplify=3.0,
            engine_params={"mode": "embed"},
        )
        comparison_results[f"ViT {year}"] = gdf_vit
        print(f"    ViT: {len(gdf_vit)} fields")

        # ── PCA baseline mode ─────────────────────────────────────
        print("\n  Mode 2: PCA baseline")
        pca_path = OUTPUT_DIR / f"fields_hls_pca_{year}.gpkg"

        gdf_pca = agribound.delineate(
            study_area=study_area,
            source=SOURCE,
            year=year,
            engine=ENGINE,
            output_path=str(pca_path),
            gee_project=gee_project,
            composite_method="median",
            min_area=5000,
            simplify=3.0,
            engine_params={"mode": "pca"},
        )
        comparison_results[f"PCA {year}"] = gdf_pca
        print(f"    PCA: {len(gdf_pca)} fields")

    # --- Comparison ---
    print(f"\n{'=' * 60}")
    print("Comparison")
    print(f"{'=' * 60}")

    print(f"\n  {'Method':<25} {'Fields':>8} {'Area (ha)':>12}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 12}")
    for label, gdf in comparison_results.items():
        area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        print(f"  {label:<25} {len(gdf):>8} {area:>12,.1f}")

    # --- Visualization ---
    if comparison_results:
        from agribound.visualize import show_comparison

        show_comparison(
            list(comparison_results.values()),
            labels=list(comparison_results.keys()),
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_murray_darling.html"),
        )
        print(f"\n  Map: {OUTPUT_DIR / 'map_murray_darling.html'}")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # Force exit — geedim\'s async runner hangs on cleanup
