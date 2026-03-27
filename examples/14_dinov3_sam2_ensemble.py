"""
14 — DINOv3 + SAM2 Multi-Source Ensemble (Lea County, NM)

Focused ensemble using only DINOv3 (fine-tuned) across multiple satellite
sources, with SAM2 boundary refinement on the final ensemble output.

DINOv3's powerful ViT backbone adapts well to each sensor's characteristics
via fine-tuning, and SAM2 produces pixel-accurate boundaries.  Running the
same architecture across Sentinel-2, Landsat, HLS, NAIP, and SPOT provides
diversity through the imagery itself rather than through different model
architectures.

Pipeline per source:
    1. Download composite from GEE
    2. Fine-tune DINOv3 on NMOSE reference boundaries
    3. Run DINOv3 inference → field polygons
    4. (Repeat for each source)

Then:
    5. Grand ensemble: majority-vote merge across all sources
    6. SAM2 boundary refinement on the ensemble output
    7. Evaluate against NMOSE reference

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
VOTE_THRESHOLD = 0.3  # Fraction of sources that must agree

# SAM2 refinement on ensemble output
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
    """Extract Lea County study area and reference boundaries."""
    gdf = gpd.read_file(shapefile_path)
    county_gdf = gdf[gdf["County"] == county_code].copy()
    if len(county_gdf) == 0:
        raise ValueError(f"No records for County {county_code}")

    county_4326 = county_gdf.to_crs(epsg=4326)
    bounds = county_4326.total_bounds
    bbox_geojson = {
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
                "properties": {"name": f"Lea County (County {county_code})"},
            }
        ],
    }
    study_area_path = output_dir / "study_area.geojson"
    with open(study_area_path, "w") as f:
        json.dump(bbox_geojson, f)

    ref_path = output_dir / "reference.gpkg"
    if not ref_path.exists():
        county_gdf.to_file(ref_path, driver="GPKG", layer="fields")

    return str(study_area_path), county_gdf, str(ref_path)


def run_dinov3(source, year, study_area, gee_project, ref_path):
    """Run DINOv3 fine-tuning + inference for a single source/year."""
    output_path = OUTPUT_DIR / f"fields_dinov3_{source}_{year}.gpkg"

    if output_path.exists():
        return gpd.read_file(output_path), output_path

    kwargs = dict(
        study_area=study_area,
        source=source,
        year=year,
        engine="dinov3",
        output_path=str(output_path),
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

    gdf = agribound.delineate(**kwargs)

    # Reproject to match NMOSE reference CRS
    if gdf.crs is not None and str(gdf.crs) != OUTPUT_CRS:
        gdf = gdf.to_crs(OUTPUT_CRS)
        gdf.to_file(output_path, driver="GPKG", layer="fields")

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
    # Phase 2: Grand ensemble + SAM2 refinement
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: Grand ensemble (vote threshold={VOTE_THRESHOLD}) + SAM2")
    print(f"{'=' * 70}")

    from agribound.engines.ensemble import EnsembleEngine
    from agribound.postprocess import filter_polygons

    ensemble_results = {}

    for year in YEARS:
        year_results = all_results.get(year, {})
        if len(year_results) < 2:
            print(f"\n  {year}: only {len(year_results)} source(s), skipping ensemble.")
            if year_results:
                # Use single source result directly
                ensemble_results[year] = next(iter(year_results.values()))
            continue

        output_path = OUTPUT_DIR / f"fields_ensemble_dinov3_{year}.gpkg"

        if output_path.exists():
            print(f"\n  {year}: already exists, loading.")
            ensemble_results[year] = gpd.read_file(output_path)
            continue

        print(f"\n  {year}: merging {len(year_results)} sources...", end=" ")
        try:
            gdf = EnsembleEngine._merge_vote(year_results, VOTE_THRESHOLD)
            gdf = filter_polygons(gdf, min_area_m2=2500)
            print(f"{len(gdf)} fields")

            # SAM2 boundary refinement
            if SAM_REFINE:
                print(f"  {year}: refining {len(gdf)} boundaries with SAM2...", flush=True)
                try:
                    from agribound.config import AgriboundConfig
                    from agribound.engines.samgeo_engine import refine_boundaries

                    raster_cache = OUTPUT_DIR / ".agribound_cache"
                    raster_candidates = sorted(raster_cache.glob(f"*sentinel2*{year}*.tif"))
                    if not raster_candidates:
                        raster_candidates = sorted(raster_cache.glob(f"*{year}*.tif"))
                    if raster_candidates:
                        sam_config = AgriboundConfig(
                            source="sentinel2",
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
                        print(f"  {year}: SAM2 refined {len(gdf)} boundaries")
                    else:
                        print(f"  {year}: no raster found for SAM2, skipping")
                except Exception as exc:
                    print(f"  {year}: SAM2 failed: {exc}")

            # Reproject and save
            if gdf.crs is not None and str(gdf.crs) != OUTPUT_CRS:
                gdf = gdf.to_crs(OUTPUT_CRS)
            gdf.to_file(output_path, driver="GPKG", layer="fields")
            ensemble_results[year] = gdf
        except Exception as exc:
            print(f"FAILED — {exc}")

    # ================================================================
    # Phase 3: Evaluation
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Evaluation against NMOSE reference")
    print(f"{'=' * 70}")

    header = f"  {'Year':<6} {'Source':<20} {'Fields':>6} {'F1':>6} {'IoU':>6} {'P':>6} {'R':>6}"
    print(f"\n{header}")
    print(f"  {'-' * 6} {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")

    for year in YEARS:
        # Individual sources
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

        # Ensemble
        gdf = ensemble_results.get(year)
        if gdf is not None:
            try:
                m = evaluate(gdf, ref_gdf)
                print(
                    f"  {year:<6} {'** ENSEMBLE **':<20} {len(gdf):>6} "
                    f"{m['f1']:.3f} {m['iou_mean']:.3f} "
                    f"{m['precision']:.3f} {m['recall']:.3f}"
                )
            except Exception:
                pass

    # ================================================================
    # Phase 4: Visualization
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Generating maps...")
    print(f"{'=' * 70}")

    from agribound.visualize import show_comparison

    if ensemble_results:
        latest_year = max(ensemble_results.keys())

        # Ensemble vs reference
        show_comparison(
            [ensemble_results[latest_year], ref_gdf],
            labels=[f"DINOv3+SAM2 Ensemble ({latest_year})", "NMOSE Reference"],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_ensemble_vs_reference.html"),
        )
        print(f"  Ensemble vs Reference: {OUTPUT_DIR / 'map_ensemble_vs_reference.html'}")

        # Per-source comparison
        latest_sources = all_results.get(latest_year, {})
        if latest_sources:
            comp_gdfs = list(latest_sources.values()) + [ensemble_results[latest_year]]
            comp_labels = list(latest_sources.keys()) + ["Ensemble"]
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
