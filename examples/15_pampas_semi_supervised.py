"""
15 — Semi-Supervised DINOv3 Field Delineation (Pampas, Argentina)

Demonstrates a fully automated semi-supervised pipeline that requires **no
human-labeled reference boundaries**:

    1. Download Google Satellite Embeddings → K-means clustering → all polygons
    2. LULC crop filter (Dynamic World) → crop-only pseudo-labels
    3. Download Sentinel-2 RGB composite for the same area/year
    4. Fine-tune DINOv3 on the Sentinel-2 imagery using crop pseudo-labels
    5. Run DINOv3 inference → cleaner field boundaries (LULC filter applied)
    6. SAM2 per-field boundary refinement → final output

The key insight: embedding clusters provide precise spectral boundaries but
no semantics (field vs road vs water).  The pipeline's LULC filter uses
Dynamic World crop probability (or NLCD for US / C3S for pre-Sentinel) to
keep only agricultural polygons.  Combining both gives us crop-only
pseudo-labels with good boundaries.

The study area is a ~10 km bbox near Pergamino, Buenos Aires Province,
covering intensive soybean/corn/wheat cropping.

Estimated runtime: ~30–60 minutes (embedding download + fine-tuning +
inference + SAM2, GPU recommended).

Prerequisites:
    pip install agribound[gee,geoai,samgeo]
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

# Study area: ~10 km bbox near Pergamino, Buenos Aires (Pampas heartland)
STUDY_AREA_BBOX = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-60.65, -33.85],
                        [-60.55, -33.85],
                        [-60.55, -33.75],
                        [-60.65, -33.75],
                        [-60.65, -33.85],
                    ]
                ],
            },
            "properties": {"name": "Pampas (Pergamino, Buenos Aires)"},
        }
    ],
}

YEAR = 2020
FINE_TUNE_EPOCHS = 20
BATCH_SIZE = 8
MIN_AREA = 5000  # m²
CROP_PROB_THRESHOLD = 0.3  # LULC crop probability threshold

# SAM2 refinement
SAM_MODEL = "large"
SAM_BATCH_SIZE = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semi-supervised DINOv3 field delineation."
    )
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    gee_project = args.gee_project
    start_time = time.time()

    # Write study area
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
    # Phase 2: LULC crop filter → crop-only pseudo-labels
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: LULC crop filter (threshold={CROP_PROB_THRESHOLD})")
    print(f"{'=' * 70}")

    from agribound.config import AgriboundConfig
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

    print(
        f"  Crop pseudo-labels: {len(pseudo_gdf)} polygons "
        f"(filtered from {len(all_clusters_gdf)})"
    )

    # Save crop-filtered pseudo-labels as reference boundaries
    ref_path = str(OUTPUT_DIR / "pseudo_reference_crops.gpkg")
    pseudo_gdf.to_file(ref_path, driver="GPKG", layer="fields")

    # ================================================================
    # Phase 3: Fine-tune DINOv3 on Sentinel-2 using crop pseudo-labels
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: DINOv3 fine-tuning on Sentinel-2 (crop pseudo-labels)")
    print(f"{'=' * 70}")

    dinov3_lulc_path = OUTPUT_DIR / f"fields_dinov3_{YEAR}_lulc.gpkg"
    dinov3_path = OUTPUT_DIR / f"fields_dinov3_{YEAR}.gpkg"
    pipeline_path = OUTPUT_DIR / f"fields_dinov3_{YEAR}_pipeline.gpkg"

    if dinov3_path.exists():
        print(f"  Using cached DINOv3 output: {dinov3_path}")
        dinov3_gdf = gpd.read_file(dinov3_path)
    else:
        print(f"  Fine-tuning DINOv3 for {FINE_TUNE_EPOCHS} epochs...")
        dinov3_gdf = agribound.delineate(
            study_area=study_area_path,
            source="sentinel2",
            year=YEAR,
            engine="dinov3",
            output_path=str(pipeline_path),
            gee_project=gee_project,
            min_area=MIN_AREA,
            simplify=2.0,
            device="auto",
            reference_boundaries=ref_path,
            fine_tune=True,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            composite_method="median",
            cloud_cover_max=20,
            date_range=(f"{YEAR}-10-01", f"{YEAR}-10-31"),
            lulc_filter=True,
            lulc_crop_threshold=CROP_PROB_THRESHOLD,
            engine_params={"batch_size": BATCH_SIZE},
        )

        # Save LULC-filtered, pre-SAM result
        dinov3_gdf.to_file(dinov3_lulc_path, driver="GPKG", layer="fields")
        print(f"  LULC-filtered: {len(dinov3_gdf)} fields → {dinov3_lulc_path.name}")

        # ============================================================
        # Phase 4: SAM2 boundary refinement
        # ============================================================
        print(f"\n{'=' * 70}")
        print("Phase 4: SAM2 boundary refinement")
        print(f"{'=' * 70}")

        try:
            from agribound.engines.samgeo_engine import refine_boundaries

            raster_cache = OUTPUT_DIR / ".agribound_cache"
            raster_candidates = sorted(
                raster_cache.glob(f"*sentinel2*{YEAR}*.tif")
            )

            if raster_candidates:
                print(f"  Refining {len(dinov3_gdf)} fields with SAM2...")
                sam_config = AgriboundConfig(
                    source="sentinel2",
                    engine="dinov3",
                    year=YEAR,
                    study_area=study_area_path,
                    output_path=str(dinov3_path),
                    engine_params={
                        "sam_model": SAM_MODEL,
                        "sam_batch_size": SAM_BATCH_SIZE,
                    },
                    device="auto",
                )
                dinov3_gdf = refine_boundaries(
                    dinov3_gdf, str(raster_candidates[0]), sam_config
                )

                from agribound.postprocess.simplify import (
                    simplify_polygons,
                    smooth_polygons,
                )

                dinov3_gdf = smooth_polygons(dinov3_gdf, iterations=3)
                dinov3_gdf = simplify_polygons(dinov3_gdf, tolerance=2.0)
                print(f"  SAM2 refined → {len(dinov3_gdf)} fields")
            else:
                print("  No raster found for SAM2, skipping")
        except Exception as exc:
            print(f"  SAM2 failed: {exc}")

        # Write final output only after all steps complete
        dinov3_gdf.to_file(dinov3_path, driver="GPKG", layer="fields")

        # Clean up pipeline temp file
        if pipeline_path.exists():
            pipeline_path.unlink()

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
        ("Crop-filtered pseudo-labels", pseudo_gdf),
        ("DINOv3 + SAM2", dinov3_gdf),
    ]:
        area = (
            gdf["metrics:area"].sum() / 10000
            if "metrics:area" in gdf.columns
            else 0
        )
        print(f"  {label:<35} {len(gdf):>8} {area:>12,.1f}")

    # Visualization
    from agribound.visualize import show_comparison

    show_comparison(
        [all_clusters_gdf, pseudo_gdf, dinov3_gdf],
        labels=[
            "All Clusters",
            "Crop-Filtered Pseudo-Labels",
            "DINOv3 + SAM2 (final)",
        ],
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_comparison.html"),
    )
    print(f"\n  Comparison map: {OUTPUT_DIR / 'map_comparison.html'}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
