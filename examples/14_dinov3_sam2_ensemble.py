"""
14 — DINOv3 + SAM2 Multi-Source Comparison (Eastern Lea County, NM)

Runs DINOv3 (fine-tuned, SAT-493M satellite-pretrained) across five
satellite sources with per-source SAM2 boundary refinement, then
compares per-source results against NMOSE reference boundaries.

Pipeline per source:
    1. Download composite from GEE
    2. Fine-tune DINOv3 on NMOSE reference boundaries
    3. Run DINOv3 inference → field polygons (LULC-filtered)
    4. SAM2 boundary refinement (per-source, using native raster)
    5. Evaluate against NMOSE reference

Estimated runtime: ~1–2 hours (5 sources × fine-tuning + inference, GPU
recommended).

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

warnings.filterwarnings("ignore", message=".*organizePolygons.*")

import geopandas as gpd

import agribound
from agribound.evaluate import evaluate

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
NMOSE_SHAPEFILE = "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
OUTPUT_DIR = Path("outputs/lea_county_dinov3_sam2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CRS = "EPSG:26913"  # Match NMOSE reference CRS (NAD83 / UTM zone 13N)

COUNTY_CODE = "25"  # Lea County
FINE_TUNE_EPOCHS = 30  # DINOv3 full fine-tuning; early stopping patience=10
BATCH_SIZE = 8
YEARS = [2020, 2021, 2022]
# SAM2 refinement per source (each source refined against its own raster)
SAM_REFINE = True
SAM_MODEL = "large"  # Per-field cropping makes large model feasible
SAM_BATCH_SIZE = 100  # Log interval

# Sources to run DINOv3 on (all that support RGB)
SOURCES = ["sentinel2", "landsat", "hls", "naip", "spot"]

# Year availability constraints
SOURCE_YEAR_RANGE = {
    "sentinel2": (2017, 2025),
    "landsat": (1985, 2025),
    "hls": (2013, 2025),
    "naip": (2003, 2025),
    "spot": (2012, 2023),
}


def create_study_area(shapefile_path, county_code, output_dir):
    """Extract eastern Lea County study area and reference boundaries.

    Uses a ~20×22 km bbox over eastern Lea County where center pivots are
    dense.  This keeps NAIP (1 m) and SPOT (1.5 m) runtimes practical
    while still covering a diverse agricultural landscape.
    """
    gdf = gpd.read_file(shapefile_path)
    county_gdf = gdf[gdf["County"] == county_code].copy()
    if len(county_gdf) == 0:
        raise ValueError(f"No records for County {county_code}")

    # Eastern Lea County bbox (center pivot area)
    bbox_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-103.25, 32.75],
                            [-103.05, 32.75],
                            [-103.05, 32.95],
                            [-103.25, 32.95],
                            [-103.25, 32.75],
                        ]
                    ],
                },
                "properties": {"name": f"Eastern Lea County (County {county_code})"},
            }
        ],
    }
    study_area_path = output_dir / "study_area.geojson"
    with open(study_area_path, "w") as f:
        json.dump(bbox_geojson, f)

    # Clip reference boundaries to the study area bbox
    from shapely.geometry import box

    bbox_geom = box(-103.25, 32.75, -103.05, 32.95)
    county_4326 = county_gdf.to_crs(epsg=4326)
    ref_clipped = county_4326[county_4326.intersects(bbox_geom)].copy()
    ref_clipped = ref_clipped.to_crs(county_gdf.crs)

    ref_path = output_dir / "reference.gpkg"
    if not ref_path.exists():
        ref_clipped.to_file(ref_path, driver="GPKG", layer="fields")

    return str(study_area_path), ref_clipped, str(ref_path)


def run_dinov3(source, year, study_area, gee_project, ref_path):
    """Run DINOv3 fine-tuning + inference (with LULC filter) + SAM2 refinement.

    The pipeline's built-in LULC filter removes non-agricultural polygons
    automatically.  SAM2 then refines only the crop field boundaries.

    Saves two GPKGs per source/year:
      1. *_lulc.gpkg  — LULC-filtered model output (pre-SAM)
      2. *.gpkg       — final (after SAM2 refinement)
    """
    output_path = OUTPUT_DIR / f"fields_dinov3_{source}_{year}.gpkg"
    lulc_path = OUTPUT_DIR / f"fields_dinov3_{source}_{year}_lulc.gpkg"

    if output_path.exists():
        return gpd.read_file(output_path), output_path

    # Use a temp path for the pipeline so it doesn't write to final output_path.
    # The final file is only written after all steps (including SAM2) complete.
    pipeline_path = OUTPUT_DIR / f"fields_dinov3_{source}_{year}_pipeline.gpkg"

    kwargs = dict(
        study_area=study_area,
        source=source,
        year=year,
        engine="dinov3",
        output_path=str(pipeline_path),
        gee_project=gee_project,
        min_area=2500,
        simplify=2.0,
        device="auto",
        reference_boundaries=ref_path,
        fine_tune=True,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        engine_params={"batch_size": BATCH_SIZE},
    )

    # Source-specific composite parameters
    if source in ("sentinel2", "landsat", "hls"):
        kwargs["composite_method"] = "median"
        kwargs["cloud_cover_max"] = 20
        kwargs["date_range"] = (f"{year}-10-01", f"{year}-10-31")
    elif source == "spot":
        kwargs["composite_method"] = "median"
        kwargs["cloud_cover_max"] = 15
    elif source == "naip":
        kwargs["min_area"] = 5000

    # Pipeline runs: delineation → post-process → LULC filter → export to temp
    gdf = agribound.delineate(**kwargs)

    # Reproject to match NMOSE reference CRS
    if gdf.crs is not None and str(gdf.crs) != OUTPUT_CRS:
        gdf = gdf.to_crs(OUTPUT_CRS)

    # Save LULC-filtered, pre-SAM result
    gdf.to_file(lulc_path, driver="GPKG", layer="fields")
    print(f"    LULC-filtered: {len(gdf)} fields → {lulc_path.name}")

    # SAM2 refinement using this source's own raster
    if SAM_REFINE:
        try:
            from agribound.config import AgriboundConfig
            from agribound.engines.samgeo_engine import refine_boundaries

            raster_cache = OUTPUT_DIR / ".agribound_cache"
            raster_candidates = sorted(raster_cache.glob(f"*{source}*{year}*.tif"))
            if raster_candidates:
                print(f"    SAM2 refining {len(gdf)} fields with {source} raster...")
                sam_config = AgriboundConfig(
                    source=source,
                    engine="dinov3",
                    year=year,
                    study_area=study_area,
                    output_path=str(output_path),
                    engine_params={
                        "sam_model": SAM_MODEL,
                        "sam_batch_size": SAM_BATCH_SIZE,
                    },
                    device="auto",
                )
                gdf = refine_boundaries(gdf, str(raster_candidates[0]), sam_config)

                from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

                gdf = smooth_polygons(gdf, iterations=3)
                gdf = simplify_polygons(gdf, tolerance=2.0)
                print(f"    SAM2 refined → {len(gdf)} fields")
            else:
                print(f"    No raster found for SAM2 ({source}), skipping")
        except Exception as exc:
            print(f"    SAM2 failed for {source}: {exc}")

    # Write final output only after all steps complete
    gdf.to_file(output_path, driver="GPKG", layer="fields")

    # Clean up pipeline temp file
    if pipeline_path.exists():
        pipeline_path.unlink()

    return gdf, output_path


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 + SAM2 multi-source ensemble.")
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    gee_project = args.gee_project
    start_time = time.time()

    # --- Study area ---
    study_area, ref_gdf, ref_path = create_study_area(NMOSE_SHAPEFILE, COUNTY_CODE, OUTPUT_DIR)
    print(f"Study area: Lea County ({len(ref_gdf)} reference polygons)")
    print(f"Sources: {', '.join(SOURCES)}")
    print(f"Years: {YEARS}")
    print(f"Fine-tuning: {FINE_TUNE_EPOCHS} epochs, early stopping")
    print(f"SAM2 refinement: {SAM_REFINE} (model={SAM_MODEL})")

    # ================================================================
    # Phase 1: DINOv3 per source per year
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 1: DINOv3 delineation per source")
    print(f"{'=' * 70}")

    all_results = {}  # {year: {"source": gdf}}

    for year in YEARS:
        print(f"\n--- Year {year} ---")
        all_results[year] = {}

        for source in SOURCES:
            yr_min, yr_max = SOURCE_YEAR_RANGE[source]
            if year < yr_min or year > yr_max:
                continue

            tag = f"{source}/dinov3"
            print(f"  {tag}: starting...", flush=True)
            try:
                gdf, _ = run_dinov3(source, year, study_area, gee_project, ref_path)
                all_results[year][source] = gdf
                print(f"  {tag}: {len(gdf)} fields")
            except Exception as exc:
                print(f"  {tag}: FAILED — {exc}")

    # ================================================================
    # Phase 2: Evaluation
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 2: Evaluation against NMOSE reference")
    print(f"{'=' * 70}")

    header = f"  {'Year':<6} {'Source':<20} {'Fields':>6} {'F1':>6} {'IoU':>6} {'P':>6} {'R':>6}"
    print(f"\n{header}")
    print(f"  {'-' * 6} {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")

    for year in YEARS:
        for source, gdf in sorted(all_results.get(year, {}).items()):
            try:
                m = evaluate(gdf, ref_gdf)
                print(
                    f"  {year:<6} {source:<20} {len(gdf):>6} "
                    f"{m['f1']:.3f} {m['iou_mean']:.3f} "
                    f"{m['precision']:.3f} {m['recall']:.3f}"
                )
            except Exception:
                pass

    # ================================================================
    # Phase 3: Visualization
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Generating maps...")
    print(f"{'=' * 70}")

    from agribound.visualize import show_comparison

    latest_year = max(all_results.keys())
    latest_sources = all_results.get(latest_year, {})
    if latest_sources:
        # Per-source comparison + reference
        comp_gdfs = list(latest_sources.values()) + [ref_gdf]
        comp_labels = list(latest_sources.keys()) + ["NMOSE Reference"]
        show_comparison(
            comp_gdfs,
            labels=comp_labels,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_source_comparison.html"),
        )
        print(f"  Source comparison: {OUTPUT_DIR / 'map_source_comparison.html'}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # Force exit — geedim's async runner hangs on cleanup
