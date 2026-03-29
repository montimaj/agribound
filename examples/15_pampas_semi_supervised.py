"""
15 — Automated Field Delineation (Pampas, Argentina)

Demonstrates a fully automated pipeline that requires **no human-labeled
reference boundaries or model training**:

    1. Download Google (64-D) and TESSERA (128-D) embeddings → K-means clustering
    2. LULC crop filter (Dynamic World) → crop-only polygons per embedding
    3. Download Sentinel-2 (10 m) composite
    4. SAM2 refines crop polygons from each embedding using S2 imagery
    5. 6-way comparison: 2 embeddings × (raw, LULC-filtered, SAM2/S2)

The key insight: embedding clusters provide spectral boundaries but no
semantics.  The LULC filter keeps only agricultural polygons.  SAM2 then
refines the rough cluster boundaries to pixel-accurate field edges using
Sentinel-2 imagery.  Comparing Google vs TESSERA embeddings reveals how
embedding source affects boundary quality.

The study area is a region near Pergamino, Buenos Aires Province,
covering intensive soybean/corn/wheat cropping.

Estimated runtime: ~15–30 minutes (embedding + S2 download + SAM2
refinement, GPU recommended).

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

YEAR = 2024
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

    # Google Satellite Embeddings (64-D)
    google_clusters_path = OUTPUT_DIR / f"all_clusters_google_{YEAR}.gpkg"

    if google_clusters_path.exists():
        print(f"  Using cached Google clusters: {google_clusters_path}")
        google_clusters_gdf = gpd.read_file(google_clusters_path)
    else:
        print(f"  Downloading Google embeddings and clustering ({YEAR})...")
        google_clusters_gdf = agribound.delineate(
            study_area=study_area_path,
            source="google-embedding",
            year=YEAR,
            engine="embedding",
            output_path=str(google_clusters_path),
            gee_project=gee_project,
            device="cpu",
            min_area=MIN_AREA,
            lulc_filter=False,
            engine_params={},
        )

    print(f"  Google clusters: {len(google_clusters_gdf)} polygons")

    # TESSERA Embeddings (128-D)
    # Note: TESSERA coverage varies by region/year (2017-2025).
    # See https://github.com/ucam-eo/geotessera for availability.
    tessera_clusters_path = OUTPUT_DIR / f"all_clusters_tessera_{YEAR}.gpkg"

    if tessera_clusters_path.exists():
        print(f"  Using cached TESSERA clusters: {tessera_clusters_path}")
        tessera_clusters_gdf = gpd.read_file(tessera_clusters_path)
    else:
        print(f"  Downloading TESSERA embeddings and clustering ({YEAR})...")
        tessera_clusters_gdf = agribound.delineate(
            study_area=study_area_path,
            source="tessera-embedding",
            year=YEAR,
            engine="embedding",
            output_path=str(tessera_clusters_path),
            gee_project=gee_project,
            device="cpu",
            min_area=MIN_AREA,
            lulc_filter=False,
            engine_params={},
        )

    print(f"  TESSERA clusters: {len(tessera_clusters_gdf)} polygons")

    # ================================================================
    # Phase 2: LULC crop filter → crop-only polygons
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: LULC crop filter (threshold={CROP_PROB_THRESHOLD})")
    print(f"{'=' * 70}")

    from agribound.config import AgriboundConfig
    from agribound.postprocess.lulc_filter import filter_by_lulc

    def _lulc_filter(clusters_gdf, source_name, cache_name):
        """LULC-filter clusters, with caching."""
        path = OUTPUT_DIR / f"fields_crop_{cache_name}_{YEAR}.gpkg"
        if path.exists():
            print(f"  Using cached {cache_name} crop polygons: {path}")
            return gpd.read_file(path)
        config = AgriboundConfig(
            study_area=study_area_path,
            source=source_name,
            year=YEAR,
            output_path=str(path),
            gee_project=gee_project,
            lulc_crop_threshold=CROP_PROB_THRESHOLD,
        )
        result = filter_by_lulc(clusters_gdf, config)
        result.to_file(path, driver="GPKG", layer="fields")
        return result

    google_crop_gdf = _lulc_filter(google_clusters_gdf, "google-embedding", "google")
    print(f"  Google crop polygons: {len(google_crop_gdf)} (from {len(google_clusters_gdf)})")

    tessera_crop_gdf = _lulc_filter(tessera_clusters_gdf, "tessera-embedding", "tessera")
    print(f"  TESSERA crop polygons: {len(tessera_crop_gdf)} (from {len(tessera_clusters_gdf)})")

    # ================================================================
    # Phase 3: Download Sentinel-2 composite
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Download Sentinel-2 composite")
    print(f"{'=' * 70}")

    # Build study area bbox from crop polygon extent
    import pandas as pd

    all_crop_gdf = pd.concat([google_crop_gdf, tessera_crop_gdf], ignore_index=True)
    ref_bounds_gdf = all_crop_gdf.to_crs(epsg=4326)
    bounds = ref_bounds_gdf.total_bounds
    composite_study_area_path = str(OUTPUT_DIR / "study_area_composite.geojson")
    composite_bbox = {
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
                "properties": {"name": "Composite extent"},
            }
        ],
    }
    with open(composite_study_area_path, "w") as f:
        json.dump(composite_bbox, f)

    from agribound.composites import get_composite_builder

    s2_config = AgriboundConfig(
        study_area=composite_study_area_path,
        source="sentinel2",
        year=YEAR,
        output_path=str(OUTPUT_DIR / "s2_composite.gpkg"),
        gee_project=gee_project,
        composite_method="median",
        cloud_cover_max=20,
        date_range=(f"{YEAR}-10-01", f"{YEAR}-10-31"),
    )
    s2_raster = get_composite_builder("sentinel2").build(s2_config)
    print(f"  Sentinel-2 composite: {s2_raster}")

    # ================================================================
    # Phase 4: SAM2 boundary refinement (S2)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 4: SAM2 boundary refinement on crop polygons")
    print(f"{'=' * 70}")

    from agribound.engines.samgeo_engine import refine_boundaries
    from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

    def _run_sam2(crop_gdf, label, raster_path, out_path):
        """Run SAM2 refinement for a single embedding set."""
        if out_path.exists():
            print(f"  Using cached {out_path.name}")
            return gpd.read_file(out_path)
        try:
            print(f"  Refining {len(crop_gdf)} {label} polygons with SAM2...")
            sam_config = AgriboundConfig(
                source="sentinel2",
                engine="embedding",
                year=YEAR,
                study_area=composite_study_area_path,
                output_path=str(out_path),
                engine_params={
                    "sam_model": SAM_MODEL,
                    "sam_batch_size": SAM_BATCH_SIZE,
                },
                device="auto",
            )
            result = refine_boundaries(crop_gdf, raster_path, sam_config)
            result = smooth_polygons(result, iterations=3)
            result = simplify_polygons(result, tolerance=2.0)
            result.to_file(out_path, driver="GPKG", layer="fields")
            print(f"  → {len(result)} fields")
            return result
        except Exception as exc:
            print(f"  SAM2 failed: {exc}")
            return None

    # Google crops + S2
    google_s2_gdf = _run_sam2(
        google_crop_gdf,
        "Google",
        s2_raster,
        OUTPUT_DIR / f"fields_sam2_google_s2_{YEAR}.gpkg",
    )

    # TESSERA crops + S2
    tessera_s2_gdf = _run_sam2(
        tessera_crop_gdf,
        "TESSERA",
        s2_raster,
        OUTPUT_DIR / f"fields_sam2_tessera_s2_{YEAR}.gpkg",
    )

    # ================================================================
    # Phase 5: Comparison
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 5: Comparison")
    print(f"{'=' * 70}")

    print(f"\n  {'Method':<35} {'Fields':>8} {'Area (ha)':>12}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 12}")

    comparison_items = [
        ("Google clusters (raw)", google_clusters_gdf),
        ("Google crops (LULC)", google_crop_gdf),
        ("TESSERA clusters (raw)", tessera_clusters_gdf),
        ("TESSERA crops (LULC)", tessera_crop_gdf),
    ]
    if google_s2_gdf is not None:
        comparison_items.append(("Google + SAM2/S2 (10 m)", google_s2_gdf))
    if tessera_s2_gdf is not None:
        comparison_items.append(("TESSERA + SAM2/S2 (10 m)", tessera_s2_gdf))

    for label, gdf in comparison_items:
        area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        print(f"  {label:<35} {len(gdf):>8} {area:>12,.1f}")

    # Visualization
    from agribound.visualize import show_comparison

    comp_gdfs = [item[1] for item in comparison_items]
    comp_labels = [item[0] for item in comparison_items]

    show_comparison(
        comp_gdfs,
        labels=comp_labels,
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
