"""
12 — Lea County, NM: Multi-Source, Multi-Model Ensemble Time Series (2020–2022)

Comprehensive field boundary delineation using ALL available satellite
products, delineation engines, and model variants with vote-based ensemble
merging and per-model fine-tuning on NMOSE reference boundaries.

Sources: Sentinel-2, Landsat, HLS, NAIP, SPOT, Google & TESSERA embeddings
Engines: delineate-anything (2 variants), FTW (3 models: B3/B5/B7),
         GeoAI, Prithvi, embedding

For each source, FTW is expanded into 3 EfficientNet models (B3, B5, B7)
and Delineate-Anything into both model variants (full + small). Each model
is independently fine-tuned on NMOSE reference boundaries before inference.
For Sentinel-2, DA automatically routes through FTW's instance segmentation
with proper S2 preprocessing and native MPS support.

Study area: Lea County (County 25) from NMOSE WUCB agricultural polygon
boundaries.  The ensemble runs every compatible source–engine–model
combination per year and merges via majority-vote overlap, producing
higher-confidence boundaries than any single run alone.

Estimated runtime: ~3–6 hours (3 years × up to 20+ source–engine–model
combos + per-model fine-tuning, GPU recommended).  Best run on HPC/cloud
with GPU.

Prerequisites:
    pip install agribound[gee,delineate-anything,ftw,geoai,prithvi]
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

COUNTY_CODE = "25"  # Lea County
FINE_TUNE = True  # Fine-tune engines on NMOSE reference boundaries
FINE_TUNE_EPOCHS = 3  # Set to 20 for production runs
FINE_TUNE_ENGINES = {"delineate-anything", "geoai", "prithvi"}  # FTW uses pre-trained weights
YEARS = range(2020, 2023)
VOTE_THRESHOLD = 0.3  # Fraction of source–engine combos that must agree

# Source → compatible engines
# For "ftw" entries, each FTW model is run separately via FTW_MODELS below.
SOURCE_ENGINE_MAP = {
    "sentinel2": ["ftw", "geoai", "prithvi", "delineate-anything"],
    "landsat": ["ftw", "prithvi", "delineate-anything"],
    "hls": ["ftw", "prithvi", "delineate-anything"],
    "naip": ["geoai", "delineate-anything"],
    "spot": ["delineate-anything"],
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
    """Extract study area GeoJSON and reference boundaries for a county."""
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path)

    # Filter to target county
    county_gdf = gdf[gdf["County"] == county_code].copy()
    if len(county_gdf) == 0:
        raise ValueError(
            f"No records found for County {county_code}. "
            f"Available counties: {sorted(gdf['County'].unique())}"
        )

    # Reproject to WGS84 for GeoJSON (bounds must be in lon/lat)
    county_4326 = county_gdf.to_crs(epsg=4326)
    bounds = county_4326.total_bounds  # [minx, miny, maxx, maxy]
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
    out_path = OUTPUT_DIR / "lea_county_study_area.geojson"
    with open(out_path, "w") as f:
        json.dump(bbox_geojson, f)

    # Save reference boundaries for fine-tuning
    ref_path = OUTPUT_DIR / "lea_county_reference.gpkg"
    if not ref_path.exists():
        county_gdf.to_file(ref_path, driver="GPKG")

    return str(out_path), county_gdf, str(ref_path)


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
        kwargs["engine_params"] = {
            "use_pca": True,
            "pca_components": 16,
            "n_clusters": "auto",
        }

    # Model override for FTW or Delineate-Anything
    if model and engine == "ftw":
        kwargs.setdefault("engine_params", {})
        kwargs["engine_params"]["model"] = model
    elif model and engine == "delineate-anything":
        kwargs.setdefault("engine_params", {})
        kwargs["engine_params"]["da_model"] = model

    gdf = agribound.delineate(**kwargs)
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
                elif engine == "delineate-anything":
                    # Run both DA model variants
                    for da_model in DA_MODELS:
                        tag = f"{source}/da/{da_model}"
                        print(f"  {tag}: starting...", flush=True)
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
                else:
                    tag = f"{source}/{engine}"
                    print(f"  {tag}: starting...", flush=True)
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

    # ================================================================
    # Phase 2: Grand ensemble per year (vote merge)
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"Phase 2: Grand ensemble (vote threshold={VOTE_THRESHOLD})")
    print(f"{'=' * 70}")

    ensemble_results = {}

    for year in YEARS:
        year_results = all_results.get(year, {})
        if len(year_results) < 2:
            print(f"\n  {year}: only {len(year_results)} result(s), skipping ensemble.")
            continue

        output_path = OUTPUT_DIR / f"fields_grand_ensemble_{year}.gpkg"

        if output_path.exists():
            print(f"\n  {year}: already exists, loading.")
            ensemble_results[year] = gpd.read_file(output_path)
            continue

        print(f"\n  {year}: merging {len(year_results)} source–engine results...", end=" ")
        try:
            gdf = grand_ensemble_vote(year_results, VOTE_THRESHOLD)

            # Post-process: filter small polygons
            from agribound.postprocess import filter_polygons

            gdf = filter_polygons(gdf, min_area_m2=2500)
            gdf.to_file(output_path, driver="GPKG")

            ensemble_results[year] = gdf
            print(f"{len(gdf)} fields")
        except Exception as exc:
            print(f"FAILED — {exc}")

    # ================================================================
    # Phase 3: Evaluation against NMOSE reference (Lea County)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Evaluation against NMOSE reference (Lea County)")
    print(f"{'=' * 70}")

    header = (
        f"  {'Year':<6} {'Source/Engine':<40} {'Fields':>6} {'F1':>6} {'IoU':>6} {'P':>6} {'R':>6}"
    )
    print(f"\n{header}")
    print(f"  {'-' * 6} {'-' * 40} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")

    for year in YEARS:
        # Evaluate individual runs
        for tag, gdf in sorted(all_results.get(year, {}).items()):
            try:
                m = evaluate(gdf, ref_gdf)
                print(
                    f"  {year:<6} {tag:<40} {len(gdf):>6} "
                    f"{m['f1']:.3f} {m['iou_mean']:.3f} "
                    f"{m['precision']:.3f} {m['recall']:.3f}"
                )
            except Exception:
                pass

        # Evaluate grand ensemble
        gdf = ensemble_results.get(year)
        if gdf is not None:
            try:
                m = evaluate(gdf, ref_gdf)
                print(
                    f"  {year:<6} {'** GRAND ENSEMBLE **':<40} {len(gdf):>6} "
                    f"{m['f1']:.3f} {m['iou_mean']:.3f} "
                    f"{m['precision']:.3f} {m['recall']:.3f}"
                )
            except Exception:
                pass

    # ================================================================
    # Phase 4: Ensemble time series summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Grand Ensemble Time Series Summary")
    print(f"{'=' * 70}")
    print(f"  {'Year':<6} {'Fields':>6} {'Area (ha)':>12} {'Sources':>8} {'F1':>6} {'IoU':>6}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 12} {'-' * 8} {'-' * 6} {'-' * 6}")

    for year in YEARS:
        gdf = ensemble_results.get(year)
        if gdf is None:
            continue
        area_ha = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        n_sources = len(all_results.get(year, {}))
        try:
            m = evaluate(gdf, ref_gdf)
            print(
                f"  {year:<6} {len(gdf):>6} {area_ha:>12,.1f} "
                f"{n_sources:>8} {m['f1']:.3f} {m['iou_mean']:.3f}"
            )
        except Exception:
            print(f"  {year:<6} {len(gdf):>6} {area_ha:>12,.1f} {n_sources:>8}")

    # ================================================================
    # Phase 5: Visualization
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Generating maps...")
    print(f"{'=' * 70}")

    from agribound.visualize import show_comparison

    # Grand ensemble vs NMOSE reference (latest year)
    if ensemble_results:
        latest_year = max(ensemble_results.keys())
        show_comparison(
            [ensemble_results[latest_year], ref_gdf],
            labels=[f"Grand Ensemble ({latest_year})", "NMOSE Reference"],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_ensemble_vs_reference.html"),
        )
        print(f"  Ensemble vs Reference: {OUTPUT_DIR / 'map_ensemble_vs_reference.html'}")

    # Per-source–engine comparison for latest year
    if ensemble_results:
        latest = all_results.get(latest_year, {})
        if latest:
            comp_gdfs = list(latest.values())
            comp_labels = list(latest.keys())
            comp_gdfs.append(ensemble_results[latest_year])
            comp_labels.append("Grand Ensemble")

            show_comparison(
                comp_gdfs,
                labels=comp_labels,
                basemap="Esri.WorldImagery",
                output_html=str(OUTPUT_DIR / "map_source_engine_comparison.html"),
            )
            print(f"  Source–engine comparison: {OUTPUT_DIR / 'map_source_engine_comparison.html'}")

    # Ensemble time series (all 3 years)
    ts_gdfs = [ensemble_results[y] for y in YEARS if y in ensemble_results]
    ts_labels = [str(y) for y in YEARS if y in ensemble_results]

    if len(ts_gdfs) >= 2:
        show_comparison(
            ts_gdfs,
            labels=ts_labels,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_ensemble_timeseries.html"),
        )
        print(f"  Time series: {OUTPUT_DIR / 'map_ensemble_timeseries.html'}")

    # Latest year standalone map
    if ensemble_results:
        agribound.show_boundaries(
            ensemble_results[max(ensemble_results.keys())],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_latest.html"),
        )
        print(f"  Latest year map: {OUTPUT_DIR / 'map_latest.html'}")


if __name__ == "__main__":
    main()
