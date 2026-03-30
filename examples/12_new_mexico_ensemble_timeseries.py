"""
12 — Eastern Lea County, NM: Multi-Model Per-Source Ensemble (2024)

Comprehensive field boundary delineation using all available satellite
sources and engines with **per-source** vote-based ensemble merging.
Ensembles are computed within each sensor (multiple models on same data),
not across sensors — merging across different resolutions produces noisy
results.

Sources: Sentinel-2, Landsat, HLS, NAIP, SPOT, Google & TESSERA embeddings
Engines: delineate-anything (2 variants), FTW (3 models: B3/B5/B7),
         GeoAI, DINOv3, Prithvi, embedding

For each source, multiple engines are run and merged via majority vote.
Each engine is independently fine-tuned on NMOSE reference boundaries.
SAM2 refines each per-source ensemble using that source's native raster.

Study area: Eastern Lea County (County 25), a ~20×22 km bbox over the
center pivot irrigation area.

Estimated runtime: ~3–6 hours (up to 20+ source–engine–model combos +
per-model fine-tuning, GPU recommended).  Best run on HPC/cloud with GPU.

Prerequisites:
    pip install agribound[gee,delineate-anything,ftw,geoai,prithvi,samgeo]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*organizePolygons.*")
warnings.filterwarnings("ignore", message=".*STAC entry.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import logging

import agribound
from agribound.evaluate import evaluate

# Enable agribound logging so download/processing progress is visible
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
logging.getLogger("methods").setLevel(logging.WARNING)

# --- Configuration ---
NMOSE_SHAPEFILE = "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
OUTPUT_DIR = Path("outputs/lea_county_ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CRS = "EPSG:26913"  # Match NMOSE reference CRS (NAD83 / UTM zone 13N)

COUNTY_CODE = "25"  # Lea County
FINE_TUNE = True  # Fine-tune engines on NMOSE reference boundaries
FINE_TUNE_EPOCHS = 10  # Set to 20 for production runs
FINE_TUNE_ENGINES = {
    "delineate-anything",
    "geoai",
    "prithvi",
    "dinov3",
}  # FTW uses pre-trained weights
BATCH_SIZE = 8  # Increase for more RAM (e.g. 16 for 128 GB)
SAM_REFINE = True  # Refine boundaries with SAM2 (requires pip install agribound[samgeo])
SAM_MODEL = "tiny"  # SAM2 variant: "tiny", "small", "base_plus", "large"
SAM_BATCH_SIZE = 50  # Number of field boxes per SAM2 batch
YEARS = [2024]
VOTE_THRESHOLD = 0.3  # Fraction of source–engine combos that must agree

# Source → compatible engines
# For "ftw" entries, each FTW model is run separately via FTW_MODELS below.
SOURCE_ENGINE_MAP = {
    "sentinel2": ["ftw", "geoai", "dinov3", "prithvi", "delineate-anything"],
    "landsat": ["ftw", "dinov3", "prithvi", "delineate-anything"],
    "hls": ["ftw", "dinov3", "prithvi", "delineate-anything"],
    "naip": ["geoai", "dinov3", "delineate-anything"],
    "spot": ["dinov3", "delineate-anything"],
    "google-embedding": ["embedding"],
    "tessera-embedding": ["embedding"],
}

# FTW v3 EfficientNet models (current, best performance).
# When "ftw" appears in SOURCE_ENGINE_MAP, each of these models is run.
FTW_MODELS = [
    "FTW_PRUE_EFNET_B3",
    "FTW_PRUE_EFNET_B5",
    "FTW_PRUE_EFNET_B7",
]

# Delineate-Anything model variants (instance segmentation).
# When "delineate-anything" appears in SOURCE_ENGINE_MAP, both are run.
DA_MODELS = [
    "DelineateAnything",  # Full model (more accurate, slower)
    "DelineateAnything-S",  # Small model (faster, less accurate)
]

# Year availability constraints (failures outside range are expected)
SOURCE_YEAR_RANGE = {
    "sentinel2": (2017, 2025),
    "landsat": (1985, 2025),
    "hls": (2013, 2025),
    "naip": (2003, 2025),  # Periodic acquisition; may not cover every year
    "spot": (2012, 2023),  # Restricted access; 2012-10-17 to 2023-11-15
    "google-embedding": (2018, 2024),
    "tessera-embedding": (2017, 2024),
}


def create_county_study_area(shapefile_path, county_code):
    """Extract eastern Lea County study area and reference boundaries.

    Uses a ~20×22 km bbox over eastern Lea County where center pivots are
    dense.  This keeps NAIP (1 m) and SPOT (6 m) runtimes practical.
    """
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.read_file(shapefile_path)

    # Filter to target county
    county_gdf = gdf[gdf["County"] == county_code].copy()
    if len(county_gdf) == 0:
        raise ValueError(
            f"No records found for County {county_code}. "
            f"Available counties: {sorted(gdf['County'].unique())}"
        )

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
    out_path = OUTPUT_DIR / "lea_county_study_area.geojson"
    with open(out_path, "w") as f:
        json.dump(bbox_geojson, f)

    # Clip reference boundaries to the study area bbox
    bbox_geom = box(-103.25, 32.75, -103.05, 32.95)
    county_4326 = county_gdf.to_crs(epsg=4326)
    ref_clipped = county_4326[county_4326.intersects(bbox_geom)].copy()
    ref_clipped = ref_clipped.to_crs(county_gdf.crs)

    # Save reference boundaries for fine-tuning
    ref_path = OUTPUT_DIR / "lea_county_reference.gpkg"
    if not ref_path.exists():
        ref_clipped.to_file(ref_path, driver="GPKG")

    return str(out_path), ref_clipped, str(ref_path)


def run_delineation(source, engine, year, study_area, gee_project, model=None, ref_path=None):
    """Run a single source–engine delineation, returning (GeoDataFrame, path).

    Parameters
    ----------
    model : str or None
        Model name override for FTW (e.g. ``"FTW_PRUE_EFNET_B7"``) or
        Delineate-Anything (e.g. ``"DelineateAnything-S"``).
    ref_path : str or None
        Path to reference boundaries for fine-tuning.
    """
    import geopandas as gpd

    suffix = f"_{model}" if model else ""
    output_path = OUTPUT_DIR / f"fields_{source}_{engine}{suffix}_{year}.gpkg"

    if output_path.exists():
        return gpd.read_file(output_path), output_path

    kwargs = dict(
        study_area=study_area,
        source=source,
        year=year,
        engine=engine,
        output_path=str(output_path),
        gee_project=gee_project,
        min_area=2500,
        simplify=2.0,
        device="auto",
        engine_params={"batch_size": BATCH_SIZE},
    )

    # Fine-tune on NMOSE reference boundaries if supported
    if FINE_TUNE and ref_path and engine in FINE_TUNE_ENGINES:
        kwargs["reference_boundaries"] = ref_path
        kwargs["fine_tune"] = True
        kwargs["fine_tune_epochs"] = FINE_TUNE_EPOCHS

    # Source-specific composite parameters
    # Use October (harvest season) composite for non-FTW engines.
    # FTW builds its own bi-temporal input (Apr + Oct) internally.
    if source in ("sentinel2", "landsat", "hls"):
        kwargs["composite_method"] = "median"
        kwargs["cloud_cover_max"] = 20
        if engine != "ftw":
            kwargs["date_range"] = (f"{year}-10-01", f"{year}-10-31")
    elif source == "spot":
        kwargs["composite_method"] = "median"
        kwargs["cloud_cover_max"] = 15
    elif source == "naip":
        kwargs["min_area"] = 5000  # 1 m resolution → larger minimum
    elif source in ("google-embedding", "tessera-embedding"):
        kwargs["device"] = "cpu"
        kwargs["min_area"] = 5000
        kwargs["engine_params"].update(
            {
                "use_pca": True,
                "pca_components": 16,
                "n_clusters": "auto",
            }
        )

    # Model override for FTW or Delineate-Anything
    if model and engine == "ftw":
        kwargs.setdefault("engine_params", {})
        kwargs["engine_params"]["model"] = model
    elif model and engine == "delineate-anything":
        kwargs.setdefault("engine_params", {})
        kwargs["engine_params"]["da_model"] = model

    gdf = agribound.delineate(**kwargs)
    # Reproject to match NMOSE reference CRS
    if gdf.crs is not None and str(gdf.crs) != OUTPUT_CRS:
        gdf = gdf.to_crs(OUTPUT_CRS)
        gdf.to_file(output_path, driver="GPKG", layer="fields")
    return gdf, output_path


def grand_ensemble_vote(results, threshold):
    """Merge results from multiple source–engine combos via majority vote."""
    from agribound.engines.ensemble import EnsembleEngine

    return EnsembleEngine._merge_vote(results, threshold)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Lea County multi-source ensemble time series.")
    parser.add_argument("--gee-project", default=None, help="GEE project ID.")
    return parser.parse_args()


def main():
    """Run multi-source ensemble field boundary delineation for Lea County."""
    args = parse_args()
    gee_project = args.gee_project

    import geopandas as gpd

    # --- Derive study area from Lea County subset ---
    study_area, ref_gdf, ref_path = create_county_study_area(NMOSE_SHAPEFILE, COUNTY_CODE)
    print(f"Study area: Lea County ({len(ref_gdf)} reference polygons)")
    print(f"Study area GeoJSON: {study_area}")
    if FINE_TUNE:
        print(f"Fine-tuning enabled ({FINE_TUNE_EPOCHS} epochs) using: {ref_path}")

    # ================================================================
    # Phase 1: Individual source–engine delineation (2020–2022)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 1: Per-source, per-engine delineation")
    print(f"  Sources: {', '.join(SOURCE_ENGINE_MAP)}")
    print(f"  Years:   {min(YEARS)}–{max(YEARS)}")
    print(f"{'=' * 70}")

    all_results = {}  # {year: {"source/engine": gdf}}

    for year in YEARS:
        print(f"\n--- Year {year} ---")
        all_results[year] = {}

        # Skip Phase 1 entirely if the grand ensemble for this year exists
        ensemble_path = OUTPUT_DIR / f"fields_grand_ensemble_{year}.gpkg"
        if ensemble_path.exists():
            print(f"  Grand ensemble already exists: {ensemble_path}, skipping individual runs.")
            # Still load individual results for evaluation (Phase 3)
            for source, engines in SOURCE_ENGINE_MAP.items():
                yr_min, yr_max = SOURCE_YEAR_RANGE[source]
                if year < yr_min or year > yr_max:
                    continue
                for engine in engines:
                    if engine == "ftw":
                        for ftw_model in FTW_MODELS:
                            tag = f"{source}/ftw/{ftw_model}"
                            p = OUTPUT_DIR / f"fields_{source}_{engine}_{ftw_model}_{year}.gpkg"
                            if p.exists():
                                all_results[year][tag] = gpd.read_file(p)
                    elif engine == "delineate-anything":
                        for da_model in DA_MODELS:
                            tag = f"{source}/da/{da_model}"
                            p = OUTPUT_DIR / f"fields_{source}_{engine}_{da_model}_{year}.gpkg"
                            if p.exists():
                                all_results[year][tag] = gpd.read_file(p)
                    else:
                        tag = f"{source}/{engine}"
                        p = OUTPUT_DIR / f"fields_{source}_{engine}_{year}.gpkg"
                        if p.exists():
                            all_results[year][tag] = gpd.read_file(p)
            continue

        for source, engines in SOURCE_ENGINE_MAP.items():
            yr_min, yr_max = SOURCE_YEAR_RANGE[source]
            if year < yr_min or year > yr_max:
                continue

            for engine in engines:
                if engine == "ftw":
                    # Run every FTW model separately for ensemble diversity
                    for ftw_model in FTW_MODELS:
                        tag = f"{source}/ftw/{ftw_model}"
                        print(f"  {tag}: starting...", flush=True)
                        try:
                            gdf, _ = run_delineation(
                                source,
                                engine,
                                year,
                                study_area,
                                gee_project,
                                model=ftw_model,
                                ref_path=ref_path,
                            )
                            all_results[year][tag] = gdf
                            print(f"  {tag}: {len(gdf)} fields")
                        except Exception as exc:
                            print(f"  {tag}: FAILED — {exc}")
                elif engine == "delineate-anything":
                    # Run both DA model variants
                    for da_model in DA_MODELS:
                        tag = f"{source}/da/{da_model}"
                        print(f"  {tag}: starting...", flush=True)
                        try:
                            gdf, _ = run_delineation(
                                source,
                                engine,
                                year,
                                study_area,
                                gee_project,
                                model=da_model,
                                ref_path=ref_path,
                            )
                            all_results[year][tag] = gdf
                            print(f"  {tag}: {len(gdf)} fields")
                        except Exception as exc:
                            print(f"  {tag}: FAILED — {exc}")
                else:
                    tag = f"{source}/{engine}"
                    print(f"  {tag}: starting...", flush=True)
                    try:
                        gdf, _ = run_delineation(
                            source,
                            engine,
                            year,
                            study_area,
                            gee_project,
                            ref_path=ref_path,
                        )
                        all_results[year][tag] = gdf
                        print(f"  {tag}: {len(gdf)} fields")
                    except Exception as exc:
                        print(f"  {tag}: FAILED — {exc}")

    # ================================================================
    # Phase 2: Per-source ensemble (vote merge across engines, not sensors)
    # ================================================================
    # Ensembles work best when multiple models run on the same sensor.
    # Merging across different sensors (1m NAIP + 30m Landsat) produces
    # noisy vote overlap due to resolution/temporal mismatches.
    print(f"\n{'=' * 70}")
    print(f"Phase 2: Per-source ensemble (vote threshold={VOTE_THRESHOLD})")
    print(f"{'=' * 70}")

    from agribound.postprocess import filter_polygons

    ensemble_results = {}  # {year: {source: gdf}}

    for year in YEARS:
        year_results = all_results.get(year, {})
        if not year_results:
            continue

        ensemble_results[year] = {}

        # Group results by source
        source_groups = {}  # {source: {tag: gdf}}
        for tag, gdf in year_results.items():
            source = tag.split("/")[0]
            source_groups.setdefault(source, {})[tag] = gdf

        for source, engine_results in sorted(source_groups.items()):
            if len(engine_results) < 2:
                # Single engine — use directly, no vote needed
                only_gdf = next(iter(engine_results.values()))
                ensemble_results[year][source] = only_gdf
                print(f"  {year}/{source}: single engine, {len(only_gdf)} fields")
                continue

            output_path = OUTPUT_DIR / f"fields_ensemble_{source}_{year}.gpkg"

            if output_path.exists():
                ensemble_results[year][source] = gpd.read_file(output_path)
                print(f"  {year}/{source}: loaded cached ensemble")
                continue

            print(
                f"  {year}/{source}: merging {len(engine_results)} engines...",
                end=" ",
            )
            try:
                gdf = grand_ensemble_vote(engine_results, VOTE_THRESHOLD)
                gdf = filter_polygons(gdf, min_area_m2=2500)
                print(f"{len(gdf)} fields")

                # SAM2 refinement using this source's raster
                if SAM_REFINE:
                    try:
                        from agribound.config import AgriboundConfig
                        from agribound.engines.samgeo_engine import refine_boundaries

                        raster_cache = OUTPUT_DIR / ".agribound_cache"
                        raster_candidates = sorted(raster_cache.glob(f"*{source}*{year}*.tif"))
                        if raster_candidates:
                            sam_config = AgriboundConfig(
                                source=source,
                                engine="ensemble",
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
                            print(f"    SAM2 refined → {len(gdf)} fields")
                    except Exception as exc:
                        print(f"    SAM2 failed: {exc}")

                if gdf.crs is not None and str(gdf.crs) != OUTPUT_CRS:
                    gdf = gdf.to_crs(OUTPUT_CRS)
                gdf.to_file(output_path, driver="GPKG", layer="fields")
                ensemble_results[year][source] = gdf
            except Exception as exc:
                print(f"FAILED — {exc}")

    # ================================================================
    # Phase 3: Evaluation against NMOSE reference
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Evaluation against NMOSE reference")
    print(f"{'=' * 70}")

    header = f"  {'Source/Engine':<40} {'Fields':>6} {'F1':>6} {'IoU':>6} {'P':>6} {'R':>6}"
    print(f"\n{header}")
    print(f"  {'-' * 40} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")

    for year in YEARS:
        # Individual runs
        for tag, gdf in sorted(all_results.get(year, {}).items()):
            try:
                m = evaluate(gdf, ref_gdf)
                print(
                    f"  {tag:<40} {len(gdf):>6} "
                    f"{m['f1']:.3f} {m['iou_mean']:.3f} "
                    f"{m['precision']:.3f} {m['recall']:.3f}"
                )
            except Exception:
                pass

        # Per-source ensembles
        year_ensembles = ensemble_results.get(year, {})
        for source, gdf in sorted(year_ensembles.items()):
            try:
                m = evaluate(gdf, ref_gdf)
                print(
                    f"  {'** ' + source + ' ENSEMBLE **':<40} {len(gdf):>6} "
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

    for year in YEARS:
        year_ensembles = ensemble_results.get(year, {})
        if not year_ensembles:
            continue

        # Per-source ensemble comparison + reference
        comp_gdfs = list(year_ensembles.values()) + [ref_gdf]
        comp_labels = [f"{s} ensemble" for s in year_ensembles] + ["NMOSE Reference"]
        show_comparison(
            comp_gdfs,
            labels=comp_labels,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / f"map_ensemble_comparison_{year}.html"),
        )
        print(f"  Ensemble comparison: {OUTPUT_DIR / f'map_ensemble_comparison_{year}.html'}")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # Force exit — geedim\'s async runner hangs on cleanup
