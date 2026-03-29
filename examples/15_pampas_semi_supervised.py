"""
15 — Semi-Supervised Field Delineation (Pampas, Argentina)

Demonstrates a fully automated pipeline that requires **no human-labeled
reference boundaries**:

    1. Download Google Satellite Embeddings → K-means clustering → all polygons
    2. LULC crop filter (Dynamic World) → crop-only polygons
    3. Download Sentinel-2 RGB composite
    4. SAM2 refines each crop polygon's boundary using Sentinel-2 imagery

The key insight: embedding clusters provide spectral boundaries but no
semantics.  The LULC filter keeps only agricultural polygons.  SAM2 then
refines the rough cluster boundaries to pixel-accurate field edges using
the Sentinel-2 imagery.  No fine-tuning or model training required.

The study area is a region near Pergamino, Buenos Aires Province,
covering intensive soybean/corn/wheat cropping.

Estimated runtime: ~15–30 minutes (embedding download + S2 download +
SAM2 refinement, GPU recommended).

Prerequisites:
    pip install agribound[gee,samgeo]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", message=".*organizePolygons.*")

import geopandas as gpd

import agribound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/pampas_semi_supervised")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Study area: Pampas region near Pergamino, Buenos Aires
# Coordinates extracted from pampas_region.geojson
STUDY_AREA_BBOX = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-60.552058, -33.789222],
                        [-60.274083, -33.734778],
                        [-60.242050, -33.903368],
                        [-60.363950, -33.995060],
                        [-60.505737, -34.015376],
                        [-60.552058, -33.789222],
                    ]
                ],
            },
            "properties": {"name": "Pampas (Pergamino, Buenos Aires)"},
        }
    ],
}

YEAR = 2020
MIN_AREA = 5000  # m²
CROP_PROB_THRESHOLD = 0.3  # LULC crop probability threshold

# SAM2 refinement
SAM_MODEL = "large"
SAM_BATCH_SIZE = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semi-supervised field delineation (embeddings + SAM2)."
    )
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    gee_project = args.gee_project
    start_time = time.time()

    study_area_path = str(OUTPUT_DIR / "study_area.geojson")
    with open(study_area_path, "w") as f:
        json.dump(STUDY_AREA_BBOX, f)

    # ================================================================
    # Phase 1: Embedding clustering → all polygons (no LULC filter)
    # ================================================================
    print(f"{'=' * 70}")
    print("Phase 1: Embedding clustering → all polygons")
    print(f"{'=' * 70}")

    all_clusters_path = OUTPUT_DIR / f"all_clusters_google_{YEAR}.gpkg"

    if all_clusters_path.exists():
        print(f"  Using cached clusters: {all_clusters_path}")
        all_clusters_gdf = gpd.read_file(all_clusters_path)
    else:
        print(f"  Downloading Google embeddings and clustering ({YEAR})...")
        all_clusters_gdf = agribound.delineate(
            study_area=study_area_path,
            source="google-embedding",
            year=YEAR,
            engine="embedding",
            output_path=str(all_clusters_path),
            gee_project=gee_project,
            device="cpu",
            min_area=MIN_AREA,
            lulc_filter=False,  # Disable — we filter manually in Phase 2
            engine_params={},
        )

    print(f"  All clusters: {len(all_clusters_gdf)} polygons")

    # ================================================================
    # Phase 2: LULC crop filter → crop-only polygons
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: LULC crop filter (threshold={CROP_PROB_THRESHOLD})")
    print(f"{'=' * 70}")

    from agribound.config import AgriboundConfig

    crop_path = OUTPUT_DIR / f"fields_crop_clusters_{YEAR}.gpkg"

    if crop_path.exists():
        print(f"  Using cached crop polygons: {crop_path}")
        pseudo_gdf = gpd.read_file(crop_path)
    else:
        from agribound.postprocess.lulc_filter import filter_by_lulc

        filter_config = AgriboundConfig(
            study_area=study_area_path,
            source="google-embedding",
            year=YEAR,
            output_path=str(all_clusters_path),
            gee_project=gee_project,
            lulc_crop_threshold=CROP_PROB_THRESHOLD,
        )
        pseudo_gdf = filter_by_lulc(all_clusters_gdf, filter_config)
        pseudo_gdf.to_file(crop_path, driver="GPKG", layer="fields")

    print(f"  Crop polygons: {len(pseudo_gdf)} (filtered from {len(all_clusters_gdf)})")

    # ================================================================
    # Phase 3: Download Sentinel-2 composite
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Download Sentinel-2 composite")
    print(f"{'=' * 70}")

    # Build S2 study area from pseudo-label extent
    ref_bounds_gdf = pseudo_gdf.to_crs(epsg=4326)
    bounds = ref_bounds_gdf.total_bounds
    s2_study_area_path = str(OUTPUT_DIR / "study_area_s2.geojson")
    s2_bbox = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [bounds[0], bounds[1]],
                            [bounds[2], bounds[1]],
                            [bounds[2], bounds[3]],
                            [bounds[0], bounds[3]],
                            [bounds[0], bounds[1]],
                        ]
                    ],
                },
                "properties": {"name": "S2 composite extent"},
            }
        ],
    }
    with open(s2_study_area_path, "w") as f:
        json.dump(s2_bbox, f)

    # Download S2 composite via the pipeline's composite builder
    from agribound.composites import get_composite_builder

    s2_config = AgriboundConfig(
        study_area=s2_study_area_path,
        source="sentinel2",
        year=YEAR,
        output_path=str(OUTPUT_DIR / "s2_composite.gpkg"),
        gee_project=gee_project,
        composite_method="median",
        cloud_cover_max=20,
        date_range=(f"{YEAR}-10-01", f"{YEAR}-10-31"),
    )
    builder = get_composite_builder("sentinel2")
    s2_raster = builder.build(s2_config)
    print(f"  Sentinel-2 composite: {s2_raster}")

    # ================================================================
    # Phase 4: SAM2 boundary refinement
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 4: SAM2 boundary refinement on crop polygons")
    print(f"{'=' * 70}")

    output_path = OUTPUT_DIR / f"fields_sam2_{YEAR}.gpkg"

    if output_path.exists():
        print(f"  Using cached SAM2 output: {output_path}")
        refined_gdf = gpd.read_file(output_path)
    else:
        try:
            from agribound.engines.samgeo_engine import refine_boundaries
            from agribound.postprocess.simplify import (
                simplify_polygons,
                smooth_polygons,
            )

            print(f"  Refining {len(pseudo_gdf)} crop polygons with SAM2...")
            sam_config = AgriboundConfig(
                source="sentinel2",
                engine="embedding",
                year=YEAR,
                study_area=s2_study_area_path,
                output_path=str(output_path),
                engine_params={
                    "sam_model": SAM_MODEL,
                    "sam_batch_size": SAM_BATCH_SIZE,
                },
                device="auto",
            )
            refined_gdf = refine_boundaries(pseudo_gdf, s2_raster, sam_config)
            refined_gdf = smooth_polygons(refined_gdf, iterations=3)
            refined_gdf = simplify_polygons(refined_gdf, tolerance=2.0)
            refined_gdf.to_file(output_path, driver="GPKG", layer="fields")
            print(f"  SAM2 refined → {len(refined_gdf)} fields")
        except Exception as exc:
            print(f"  SAM2 failed: {exc}")
            refined_gdf = pseudo_gdf

    # ================================================================
    # Phase 5: Comparison
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 5: Comparison")
    print(f"{'=' * 70}")

    print(f"\n  {'Method':<35} {'Fields':>8} {'Area (ha)':>12}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 12}")

    for label, gdf in [
        ("All embedding clusters", all_clusters_gdf),
        ("Crop-filtered polygons", pseudo_gdf),
        ("SAM2 refined", refined_gdf),
    ]:
        area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        print(f"  {label:<35} {len(gdf):>8} {area:>12,.1f}")

    # Visualization
    from agribound.visualize import show_comparison

    show_comparison(
        [all_clusters_gdf, pseudo_gdf, refined_gdf],
        labels=[
            "All Clusters",
            "Crop-Filtered",
            "SAM2 Refined (final)",
        ],
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_comparison.html"),
    )
    print(f"\n  Comparison map: {OUTPUT_DIR / 'map_comparison.html'}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # Force exit — geedim's async runner hangs on cleanup
