"""
12 — New Mexico Ensemble Time Series (2017–2025)

Generates annual field boundary polygons using Sentinel-2 imagery and the
ensemble engine (delineate-anything + FTW + geoai with vote-based merge).
Uses NMOSE WUCB agricultural polygon boundaries for evaluation.

The ensemble approach runs all three engines on each year's composite and
merges results via majority-vote overlap, producing higher-confidence
boundaries than any single engine alone.

Estimated runtime: ~4–8 hours (9 years × 3 engines, GPU recommended).
Best run on HPC/cloud with GPU.

Prerequisites:
    pip install agribound[gee,delineate-anything,ftw,geoai]
    agribound auth --project YOUR_GEE_PROJECT
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*organizePolygons.*")

import agribound
from agribound.evaluate import evaluate

# --- Configuration ---
NMOSE_SHAPEFILE = "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
OUTPUT_DIR = Path("outputs/new_mexico_ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2017, 2026)
SOURCE = "sentinel2"
ENGINES = ["delineate-anything", "ftw", "geoai"]


def create_study_area_from_shapefile(shapefile_path):
    """Derive study area GeoJSON from the bounding box of a shapefile."""
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
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
                "properties": {"name": "NMOSE WUCB Study Area"},
            }
        ],
    }
    out_path = OUTPUT_DIR / "nm_study_area.geojson"
    with open(out_path, "w") as f:
        json.dump(bbox_geojson, f)
    return str(out_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="New Mexico ensemble time series field boundary delineation."
    )
    parser.add_argument(
        "--gee-project", default=None, help="GEE project ID (auto-detected from gcloud config if not set)."
    )
    return parser.parse_args()


def main():
    """Run ensemble field boundary delineation for New Mexico (2017–2025)."""
    args = parse_args()
    gee_project = args.gee_project

    import geopandas as gpd

    # --- Derive study area from NMOSE shapefile bounds ---
    study_area = create_study_area_from_shapefile(NMOSE_SHAPEFILE)
    print(f"Study area derived from NMOSE shapefile bounds: {study_area}")

    # --- Load NMOSE reference boundaries ---
    print("Loading NMOSE reference boundaries...")
    ref_gdf = gpd.read_file(NMOSE_SHAPEFILE)
    print(f"  {len(ref_gdf)} reference field polygons loaded")

    # --- Phase 1: Per-engine results for each year ---
    # Run each engine individually so we can compare before ensembling
    print(f"\n{'='*60}")
    print("Phase 1: Per-engine delineation (2017–2025)")
    print(f"  Engines: {', '.join(ENGINES)}")
    print(f"{'='*60}")

    per_engine_results = {}  # {year: {engine_name: gdf}}

    for year in YEARS:
        print(f"\n--- Year {year} ---")
        per_engine_results[year] = {}

        for engine_name in ENGINES:
            output_path = OUTPUT_DIR / f"fields_{engine_name}_{year}.gpkg"

            if output_path.exists():
                print(f"  {engine_name}: already exists, skipping.")
                per_engine_results[year][engine_name] = gpd.read_file(output_path)
                continue

            try:
                gdf = agribound.delineate(
                    study_area=study_area,
                    source=SOURCE,
                    year=year,
                    engine=engine_name,
                    output_path=str(output_path),
                    gee_project=gee_project,
                    composite_method="median",
                    cloud_cover_max=20,
                    min_area=2500,
                    simplify=2.0,
                    device="auto",
                )
                per_engine_results[year][engine_name] = gdf
                print(f"  {engine_name}: {len(gdf)} fields")
            except Exception as exc:
                print(f"  {engine_name}: FAILED — {exc}")

    # --- Phase 2: Ensemble for each year ---
    print(f"\n{'='*60}")
    print("Phase 2: Ensemble delineation (vote strategy)")
    print(f"{'='*60}")

    ensemble_results = {}

    for year in YEARS:
        print(f"\nEnsemble for {year}...", end=" ")
        output_path = OUTPUT_DIR / f"fields_ensemble_{year}.gpkg"

        if output_path.exists():
            print("already exists, skipping.")
            ensemble_results[year] = gpd.read_file(output_path)
            continue

        try:
            gdf = agribound.delineate(
                study_area=study_area,
                source=SOURCE,
                year=year,
                engine="ensemble",
                output_path=str(output_path),
                gee_project=gee_project,
                composite_method="median",
                cloud_cover_max=20,
                min_area=2500,
                simplify=2.0,
                device="auto",
                engine_params={
                    "engines": ENGINES,
                    "merge_strategy": "vote",
                },
            )
            ensemble_results[year] = gdf
            print(f"{len(gdf)} fields")
        except Exception as exc:
            print(f"FAILED — {exc}")

    # --- Phase 3: Evaluate against NMOSE ---
    print(f"\n{'='*60}")
    print("Phase 3: Evaluation against NMOSE reference boundaries")
    print(f"{'='*60}")

    print(f"\n  {'Year':<6} {'Engine':<25} {'Fields':>6} {'F1':>6} {'IoU':>6} "
          f"{'P':>6} {'R':>6}")
    print(f"  {'-'*6} {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for year in YEARS:
        # Evaluate each individual engine
        for engine_name in ENGINES:
            gdf = per_engine_results.get(year, {}).get(engine_name)
            if gdf is not None:
                m = evaluate(gdf, ref_gdf)
                print(f"  {year:<6} {engine_name:<25} {len(gdf):>6} "
                      f"{m['f1']:.3f} {m['iou_mean']:.3f} "
                      f"{m['precision']:.3f} {m['recall']:.3f}")

        # Evaluate ensemble
        gdf = ensemble_results.get(year)
        if gdf is not None:
            m = evaluate(gdf, ref_gdf)
            print(f"  {year:<6} {'** ensemble **':<25} {len(gdf):>6} "
                  f"{m['f1']:.3f} {m['iou_mean']:.3f} "
                  f"{m['precision']:.3f} {m['recall']:.3f}")

    # --- Phase 4: Time series summary ---
    print(f"\n{'='*60}")
    print("Ensemble Time Series Summary")
    print(f"{'='*60}")
    print(f"  {'Year':<6} {'Fields':>6} {'Area (ha)':>12} {'F1':>6} {'IoU':>6}")
    print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*6} {'-'*6}")

    for year, gdf in sorted(ensemble_results.items()):
        area_ha = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        m = evaluate(gdf, ref_gdf)
        print(f"  {year:<6} {len(gdf):>6} {area_ha:>12,.1f} "
              f"{m['f1']:.3f} {m['iou_mean']:.3f}")

    # --- Phase 5: Visualization ---
    print(f"\n{'='*60}")
    print("Generating maps...")
    print(f"{'='*60}")

    from agribound.visualize import show_comparison

    # Ensemble vs NMOSE reference for the latest year
    if ensemble_results:
        latest_year = max(ensemble_results.keys())
        m_ref = show_comparison(
            [ensemble_results[latest_year], ref_gdf],
            labels=[f"Ensemble ({latest_year})", "NMOSE Reference"],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_ensemble_vs_reference.html"),
        )
        print(f"  Ensemble vs Reference: {OUTPUT_DIR / 'map_ensemble_vs_reference.html'}")

    # Per-engine comparison for the latest year
    if ensemble_results and per_engine_results.get(latest_year):
        engine_gdfs = list(per_engine_results[latest_year].values())
        engine_labels = list(per_engine_results[latest_year].keys())
        if latest_year in ensemble_results:
            engine_gdfs.append(ensemble_results[latest_year])
            engine_labels.append("ensemble")

        m_engines = show_comparison(
            engine_gdfs,
            labels=engine_labels,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_engine_comparison.html"),
        )
        print(f"  Engine comparison: {OUTPUT_DIR / 'map_engine_comparison.html'}")

    # Ensemble time series comparison
    selected_years = [2017, 2019, 2021, 2023, 2025]
    ts_gdfs = [ensemble_results[y] for y in selected_years if y in ensemble_results]
    ts_labels = [str(y) for y in selected_years if y in ensemble_results]

    if len(ts_gdfs) >= 2:
        m_ts = show_comparison(
            ts_gdfs,
            labels=ts_labels,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_ensemble_timeseries.html"),
        )
        print(f"  Time series: {OUTPUT_DIR / 'map_ensemble_timeseries.html'}")

    # Latest year standalone map
    if ensemble_results:
        m_latest = agribound.show_boundaries(
            ensemble_results[max(ensemble_results.keys())],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_latest.html"),
        )
        print(f"  Latest year map: {OUTPUT_DIR / 'map_latest.html'}")


if __name__ == "__main__":
    main()
