"""
15 — Automated Field Delineation (Pampas, Argentina)

Demonstrates a fully automated pipeline that requires **no human-labeled
reference boundaries or model training**:

    1. Download Google (64-D) and TESSERA (128-D) embeddings → K-means clustering
    2. LULC crop filter (Dynamic World) → crop-only polygons per embedding
    3. Download Sentinel-2 (10 m) composite
    4. SAM2 refines crop polygons from each embedding using S2 imagery
    5. SAM2 on TESSERA native bands 1-3 (no S2 temporal mismatch)
    6. Improved SAM2: fix geometries, explode multi-part polygons,
       separate large fields before refinement
    7. Full comparison across all methods

TESSERA produces more accurate field boundaries than Google because it
encodes temporal crop signatures from Sentinel-1/2 time series, while
Google's annual summary averages out intra-seasonal differences.
The LULC-filtered output is already highly accurate; SAM2 refinement
sharpens edges but can merge/drop fields if polygons are large or
unclosed.  Phase 7 addresses this by fixing geometries, exploding
multi-part polygons, and skipping SAM2 for oversized fields.

Note: TESSERA coverage varies by region/year (2017–2025).
See https://github.com/ucam-eo/geotessera

The study area is a region near Pergamino, Buenos Aires Province,
covering intensive soybean/corn/wheat cropping.

Estimated runtime: ~15–30 minutes (embedding + S2 download + SAM2
refinement, GPU recommended).

Prerequisites:
    pip install agribound[gee,samgeo,tessera]
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

# Embedding sources to run
EMBEDDING_SOURCES = {
    "google": {"source": "google-embedding", "year": YEAR},
    "tessera": {"source": "tessera-embedding", "year": 2024},  # Best coverage year
}


def parse_args():
    parser = argparse.ArgumentParser(description="Automated field delineation (embeddings + SAM2).")
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    gee_project = args.gee_project
    start_time = time.time()

    study_area_path = str(OUTPUT_DIR / "study_area.geojson")
    with open(study_area_path, "w") as f:
        json.dump(STUDY_AREA_BBOX, f)

    from agribound.config import AgriboundConfig

    # ================================================================
    # Phase 1: Embedding clustering → all polygons (no LULC filter)
    # ================================================================
    print(f"{'=' * 70}")
    print("Phase 1: Embedding clustering → all polygons")
    print(f"{'=' * 70}")

    cluster_results = {}  # {name: gdf}

    for name, cfg in EMBEDDING_SOURCES.items():
        clusters_path = OUTPUT_DIR / f"all_clusters_{name}_{cfg['year']}.gpkg"

        if clusters_path.exists():
            print(f"  Using cached {name} clusters: {clusters_path}")
            cluster_results[name] = gpd.read_file(clusters_path)
        else:
            print(f"  Downloading {name} embeddings ({cfg['year']})...")
            try:
                gdf = agribound.delineate(
                    study_area=study_area_path,
                    source=cfg["source"],
                    year=cfg["year"],
                    engine="embedding",
                    output_path=str(clusters_path),
                    gee_project=gee_project,
                    device="cpu",
                    min_area=MIN_AREA,
                    lulc_filter=False,
                    engine_params={},
                )
                cluster_results[name] = gdf
            except Exception as exc:
                print(f"  {name} FAILED: {exc}")

        if name in cluster_results:
            print(f"  {name}: {len(cluster_results[name])} polygons")

    # ================================================================
    # Phase 2: LULC crop filter → crop-only polygons
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: LULC crop filter (threshold={CROP_PROB_THRESHOLD})")
    print(f"{'=' * 70}")

    from agribound.postprocess.lulc_filter import filter_by_lulc

    crop_results = {}  # {name: gdf}

    for name, clusters_gdf in cluster_results.items():
        src_cfg = EMBEDDING_SOURCES[name]
        crop_path = OUTPUT_DIR / f"fields_crop_{name}_{src_cfg['year']}.gpkg"

        if crop_path.exists():
            print(f"  Using cached {name} crop polygons: {crop_path}")
            crop_results[name] = gpd.read_file(crop_path)
        else:
            filter_config = AgriboundConfig(
                study_area=study_area_path,
                source=src_cfg["source"],
                year=src_cfg["year"],
                output_path=str(crop_path),
                gee_project=gee_project,
                lulc_crop_threshold=CROP_PROB_THRESHOLD,
            )
            crop_results[name] = filter_by_lulc(clusters_gdf, filter_config)
            crop_results[name].to_file(crop_path, driver="GPKG", layer="fields")

        print(f"  {name}: {len(crop_results[name])} crop polygons (from {len(clusters_gdf)})")

    # ================================================================
    # Phase 3: Download Sentinel-2 composite
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Download Sentinel-2 composite")
    print(f"{'=' * 70}")

    # Use the largest crop extent across all embeddings
    import pandas as pd

    all_crop_4326 = [gdf.to_crs(epsg=4326) for gdf in crop_results.values()]
    all_crop_gdf = pd.concat(all_crop_4326, ignore_index=True)
    bounds = all_crop_gdf.total_bounds
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
    # Phase 4: SAM2 boundary refinement
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 4: SAM2 boundary refinement on crop polygons")
    print(f"{'=' * 70}")

    from agribound.engines.samgeo_engine import refine_boundaries
    from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

    sam2_results = {}  # {name: gdf}

    for name, crop_gdf in crop_results.items():
        out_path = OUTPUT_DIR / f"fields_sam2_{name}_s2_{YEAR}.gpkg"

        if out_path.exists():
            print(f"  Using cached {name} SAM2 output: {out_path.name}")
            sam2_results[name] = gpd.read_file(out_path)
        else:
            try:
                print(f"  Refining {len(crop_gdf)} {name} polygons with SAM2...")
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
                result = refine_boundaries(crop_gdf, s2_raster, sam_config)
                result = smooth_polygons(result, iterations=3)
                result = simplify_polygons(result, tolerance=2.0)
                result.to_file(out_path, driver="GPKG", layer="fields")
                sam2_results[name] = result
                print(f"  {name} SAM2 → {len(result)} fields")
            except Exception as exc:
                print(f"  {name} SAM2 failed: {exc}")

    # ================================================================
    # Phase 5: Comparison
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 5: Comparison")
    print(f"{'=' * 70}")

    print(f"\n  {'Method':<35} {'Fields':>8} {'Area (ha)':>12}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 12}")

    comparison_items = []
    for name in EMBEDDING_SOURCES:
        if name in cluster_results:
            comparison_items.append((f"{name} clusters (raw)", cluster_results[name]))
        if name in crop_results:
            comparison_items.append((f"{name} crops (LULC)", crop_results[name]))
        if name in sam2_results:
            comparison_items.append((f"{name} + SAM2/S2", sam2_results[name]))

    for label, gdf in comparison_items:
        area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        print(f"  {label:<35} {len(gdf):>8} {area:>12,.1f}")

    # Visualization
    from agribound.visualize import show_comparison

    show_comparison(
        [item[1] for item in comparison_items],
        labels=[item[0] for item in comparison_items],
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_comparison.html"),
    )
    print(f"\n  Comparison map: {OUTPUT_DIR / 'map_comparison.html'}")

    # ================================================================
    # Phase 6: SAM2 on TESSERA bands 1-3 (experimental)
    # ================================================================
    # The S2 median composite averages multiple overpass dates, which
    # may not match the temporal signal TESSERA captured.  Using
    # TESSERA's first 3 embedding bands as pseudo-RGB for SAM2 gives
    # a more temporally consistent refinement.
    if "tessera" in crop_results:
        print(f"\n{'=' * 70}")
        print("Phase 6: SAM2 on TESSERA bands 1-3 (experimental)")
        print(f"{'=' * 70}")

        tessera_sam2_path = OUTPUT_DIR / f"fields_sam2_tessera_native_{YEAR}.gpkg"

        if tessera_sam2_path.exists():
            print(f"  Using cached: {tessera_sam2_path.name}")
            tessera_native_gdf = gpd.read_file(tessera_sam2_path)
        else:
            try:
                # Find the mosaicked TESSERA raster
                tessera_raster = str(
                    OUTPUT_DIR
                    / ".agribound_cache"
                    / f"tessera_embedding_{EMBEDDING_SOURCES['tessera']['year']}.tif"
                )

                print(
                    f"  Refining {len(crop_results['tessera'])} TESSERA "
                    f"polygons with SAM2 (bands 1-3 as pseudo-RGB)..."
                )
                sam_config = AgriboundConfig(
                    # Use "local" source so SAM2 takes bands 1,2,3 directly
                    source="local",
                    local_tif_path=tessera_raster,
                    engine="embedding",
                    year=YEAR,
                    study_area=composite_study_area_path,
                    output_path=str(tessera_sam2_path),
                    engine_params={
                        "sam_model": SAM_MODEL,
                        "sam_batch_size": SAM_BATCH_SIZE,
                    },
                    device="auto",
                )
                tessera_native_gdf = refine_boundaries(
                    crop_results["tessera"], tessera_raster, sam_config
                )
                tessera_native_gdf = smooth_polygons(tessera_native_gdf, iterations=3)
                tessera_native_gdf = simplify_polygons(tessera_native_gdf, tolerance=2.0)
                tessera_native_gdf.to_file(tessera_sam2_path, driver="GPKG", layer="fields")
                print(f"  TESSERA native SAM2 → {len(tessera_native_gdf)} fields")

                # Add to comparison
                comparison_items.append(("TESSERA + SAM2/native", tessera_native_gdf))

                # Updated comparison table
                print(f"\n  {'Method':<35} {'Fields':>8} {'Area (ha)':>12}")
                print(f"  {'-' * 35} {'-' * 8} {'-' * 12}")
                for label, gdf in comparison_items:
                    area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
                    print(f"  {label:<35} {len(gdf):>8} {area:>12,.1f}")

                # Updated map
                show_comparison(
                    [item[1] for item in comparison_items],
                    labels=[item[0] for item in comparison_items],
                    basemap="Esri.WorldImagery",
                    output_html=str(OUTPUT_DIR / "map_comparison_with_native.html"),
                )
                print(f"\n  Updated map: {OUTPUT_DIR / 'map_comparison_with_native.html'}")
            except Exception as exc:
                print(f"  TESSERA native SAM2 failed: {exc}")

    # ================================================================
    # Phase 7: Improved SAM2 on TESSERA (fix geometry + explode)
    # ================================================================
    # The LULC-filtered output is already accurate, but SAM2 drops
    # fields because: (1) large polygons covering multiple pivots get
    # a single box prompt, (2) unclosed/invalid geometries produce
    # oversized bounding boxes, (3) small fields fall below
    # MIN_CROP_SIZE.  Fix: clean geometries, explode multi-part
    # polygons, and filter before SAM2.
    if "tessera" in crop_results:
        print(f"\n{'=' * 70}")
        print("Phase 7: Improved SAM2 on TESSERA (geometry fixes)")
        print(f"{'=' * 70}")

        improved_sam2_path = OUTPUT_DIR / f"fields_sam2_tessera_improved_{YEAR}.gpkg"

        if improved_sam2_path.exists():
            print(f"  Using cached: {improved_sam2_path.name}")
            improved_gdf = gpd.read_file(improved_sam2_path)
        else:
            try:
                from shapely.validation import make_valid

                tessera_raster = str(
                    OUTPUT_DIR
                    / ".agribound_cache"
                    / f"tessera_embedding_{EMBEDDING_SOURCES['tessera']['year']}.tif"
                )

                crop_gdf = crop_results["tessera"].copy()
                n_before = len(crop_gdf)

                # Step 1: Fix invalid geometries
                crop_gdf["geometry"] = crop_gdf.geometry.apply(
                    lambda g: make_valid(g) if not g.is_valid else g
                )
                crop_gdf = crop_gdf[~crop_gdf.geometry.is_empty].copy()
                print(f"  After fixing invalid geometries: {len(crop_gdf)} fields")

                # Step 2: Explode multi-part polygons so each part gets
                # its own SAM2 box prompt (separates merged pivots)
                crop_gdf = crop_gdf.explode(index_parts=False).reset_index(drop=True)
                crop_gdf = crop_gdf[crop_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                print(f"  After explode: {len(crop_gdf)} fields (was {n_before})")

                # Step 3: Remove very large polygons that likely contain
                # multiple fields — SAM2 can't separate them with a
                # single box prompt.  Re-add them unchanged after SAM2.
                crop_gdf_metric = crop_gdf.to_crs(epsg=32720)  # UTM 20S
                areas = crop_gdf_metric.geometry.area
                max_field_area = 500_000  # 50 ha — larger than any single pivot
                large_mask = areas > max_field_area
                small_gdf = crop_gdf[~large_mask].copy()
                large_gdf = crop_gdf[large_mask].copy()
                print(
                    f"  {len(small_gdf)} fields for SAM2, {len(large_gdf)} large fields kept as-is"
                )

                # Step 4: Run SAM2 on the cleaned, exploded, size-filtered set
                print(f"  Refining {len(small_gdf)} fields with SAM2...")
                sam_config = AgriboundConfig(
                    source="local",
                    local_tif_path=tessera_raster,
                    engine="embedding",
                    year=YEAR,
                    study_area=composite_study_area_path,
                    output_path=str(improved_sam2_path),
                    engine_params={
                        "sam_model": SAM_MODEL,
                        "sam_batch_size": SAM_BATCH_SIZE,
                    },
                    device="auto",
                )
                refined_small = refine_boundaries(small_gdf, tessera_raster, sam_config)
                refined_small = smooth_polygons(refined_small, iterations=3)
                refined_small = simplify_polygons(refined_small, tolerance=2.0)

                # Step 5: Combine SAM2-refined small fields + unchanged large fields
                import pandas as pd

                improved_gdf = pd.concat([refined_small, large_gdf], ignore_index=True)
                improved_gdf = improved_gdf[
                    ~improved_gdf.geometry.is_empty & improved_gdf.geometry.is_valid
                ].reset_index(drop=True)

                improved_gdf.to_file(improved_sam2_path, driver="GPKG", layer="fields")
                print(
                    f"  Improved SAM2 → {len(improved_gdf)} fields "
                    f"({len(refined_small)} refined + {len(large_gdf)} kept)"
                )

                # Add to comparison
                comparison_items.append(("TESSERA + SAM2/improved", improved_gdf))

                # Final comparison table
                print(f"\n  {'Method':<35} {'Fields':>8} {'Area (ha)':>12}")
                print(f"  {'-' * 35} {'-' * 8} {'-' * 12}")
                for label, gdf in comparison_items:
                    area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
                    print(f"  {label:<35} {len(gdf):>8} {area:>12,.1f}")

                # Final map
                show_comparison(
                    [item[1] for item in comparison_items],
                    labels=[item[0] for item in comparison_items],
                    basemap="Esri.WorldImagery",
                    output_html=str(OUTPUT_DIR / "map_final.html"),
                )
                print(f"\n  Final map: {OUTPUT_DIR / 'map_final.html'}")
            except Exception as exc:
                print(f"  Improved SAM2 failed: {exc}")
                import traceback

                traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # Force exit — geedim's async runner hangs on cleanup
