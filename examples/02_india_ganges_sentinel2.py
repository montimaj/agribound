"""
02 — India Nadia District (West Bengal), Multi-Approach Comparison

Compares field boundary delineation in Nadia district, West Bengal using
four approaches:
    1. FTW (Sentinel-2) — supervised, country-specific model for India
    2. Google embeddings (64-D) — unsupervised clustering + LULC filtering
    3. TESSERA embeddings (128-D) — unsupervised clustering + LULC filtering
    4. SPOT Panchromatic (1.5 m) — high-res grayscale with Delineate-Anything

All run on the same study area for 2024.

Estimated runtime: ~15–30 minutes (GPU recommended).

Prerequisites:
    pip install agribound[gee,ftw,tessera,delineate-anything]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import agribound

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")

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
OUTPUT_DIR = Path("outputs/india_nadia")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2024
MIN_AREA = 100  # Smallholder fields — lower minimum area

# Study area: Nadia district, West Bengal (dense rice paddies)
STUDY_AREA_BBOX = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [88.35, 23.35],
                        [88.50, 23.35],
                        [88.50, 23.50],
                        [88.35, 23.50],
                        [88.35, 23.35],
                    ]
                ],
            },
            "properties": {"name": "Nadia District, West Bengal"},
        }
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="India Nadia District: FTW vs TESSERA comparison.")
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    gee_project = args.gee_project

    study_area_path = str(OUTPUT_DIR / "nadia_aoi.geojson")
    with open(study_area_path, "w") as f:
        json.dump(STUDY_AREA_BBOX, f)

    results = {}

    # ================================================================
    # Approach 1: FTW (Sentinel-2, supervised)
    # ================================================================
    print(f"{'=' * 60}")
    print("Approach 1: FTW on Sentinel-2 (supervised)")
    print(f"{'=' * 60}")

    ftw_path = OUTPUT_DIR / f"fields_ftw_s2_{YEAR}.gpkg"

    gdf_ftw = agribound.delineate(
        study_area=study_area_path,
        source="sentinel2",
        year=YEAR,
        engine="ftw",
        output_path=str(ftw_path),
        gee_project=gee_project,
        composite_method="median",
        cloud_cover_max=30,
        min_area=MIN_AREA,
        simplify=1.0,
    )
    results["FTW (Sentinel-2)"] = gdf_ftw
    print(f"  FTW: {len(gdf_ftw)} fields")

    # ================================================================
    # Approach 2: Google embeddings (unsupervised + LULC filter)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Approach 2: Google embeddings (unsupervised + LULC filter)")
    print(f"{'=' * 60}")

    google_path = OUTPUT_DIR / f"fields_google_{YEAR}.gpkg"

    gdf_google = agribound.delineate(
        study_area=study_area_path,
        source="google-embedding",
        year=YEAR,
        engine="embedding",
        output_path=str(google_path),
        gee_project=gee_project,
        device="cpu",
        min_area=MIN_AREA
    )
    results["Google (unsupervised)"] = gdf_google
    print(f"  Google: {len(gdf_google)} fields")

    # ================================================================
    # Approach 3: TESSERA embeddings (unsupervised + LULC filter)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Approach 3: TESSERA embeddings (unsupervised + LULC filter)")
    print(f"{'=' * 60}")

    tessera_path = OUTPUT_DIR / f"fields_tessera_{YEAR}.gpkg"

    gdf_tessera = agribound.delineate(
        study_area=study_area_path,
        source="tessera-embedding",
        year=YEAR,
        engine="embedding",
        output_path=str(tessera_path),
        gee_project=gee_project,
        device="cpu",
        min_area=MIN_AREA,
        engine_params={"n_clusters": 8},
    )
    results["TESSERA (unsupervised)"] = gdf_tessera
    print(f"  TESSERA: {len(gdf_tessera)} fields")

    # ================================================================
    # Approach 4: SPOT Panchromatic (1.5 m, restricted access)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Approach 4: SPOT Panchromatic (1.5 m) + Delineate-Anything")
    print(f"{'=' * 60}")

    spot_pan_path = OUTPUT_DIR / "fields_spot_pan_2023.gpkg"

    try:
        gdf_spot = agribound.delineate(
            study_area=study_area_path,
            source="spot-pan",
            year=2020,  # SPOT available through 2023
            engine="delineate-anything",
            output_path=str(spot_pan_path),
            gee_project=gee_project,
            composite_method="median",
            cloud_cover_max=15,
            min_area=MIN_AREA,
            simplify=1.0,
        )
        results["SPOT Pan (1.5 m)"] = gdf_spot
        print(f"  SPOT Pan: {len(gdf_spot)} fields")
    except Exception as exc:
        print(f"  SPOT Pan failed (restricted access): {exc}")

    # ================================================================
    # Comparison
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Comparison")
    print(f"{'=' * 60}")

    print(f"\n  {'Method':<30} {'Fields':>8} {'Area (ha)':>12}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 12}")

    for label, gdf in results.items():
        area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        print(f"  {label:<30} {len(gdf):>8} {area:>12,.1f}")

    # ================================================================
    # Visualization
    # ================================================================
    from agribound.visualize import show_comparison

    show_comparison(
        list(results.values()),
        labels=list(results.keys()),
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_ftw_google_tessera_spot.html"),
    )
    print(f"\n  Map: {OUTPUT_DIR / 'map_ftw_google_tessera_spot.html'}")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # Force exit — geedim's async runner hangs on cleanup
