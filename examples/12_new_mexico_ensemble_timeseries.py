"""
12 — Lea County, NM: Multi-Source Ensemble Time Series (2020–2022)

Comprehensive field boundary delineation using ALL available satellite
products and delineation engines with vote-based ensemble merging.

Sources: Sentinel-2, Landsat, HLS, NAIP, SPOT, Google & TESSERA embeddings
Engines: delineate-anything, FTW, GeoAI, Prithvi, embedding

Study area: Lea County (County 25) from NMOSE WUCB agricultural polygon
boundaries.  The ensemble runs every compatible source–engine combination
per year and merges via majority-vote overlap, producing higher-confidence
boundaries than any single engine alone.

Estimated runtime: ~2–4 hours (3 years × up to 15 source–engine combos,
GPU recommended).  Best run on HPC/cloud with GPU.

Prerequisites:
    pip install agribound[gee,delineate-anything,ftw,geoai,prithvi]
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
OUTPUT_DIR = Path("outputs/lea_county_ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_CODE = 25  # Lea County
YEARS = range(2020, 2023)
VOTE_THRESHOLD = 0.3  # Fraction of source–engine combos that must agree

# Source → compatible engines
SOURCE_ENGINE_MAP = {
    "sentinel2": ["delineate-anything", "ftw", "geoai", "prithvi"],
    "landsat": ["delineate-anything", "ftw", "prithvi"],
    "hls": ["delineate-anything", "ftw", "prithvi"],
    "naip": ["delineate-anything", "geoai"],
    "spot": ["delineate-anything"],
    "google-embedding": ["embedding"],
    "tessera-embedding": ["embedding"],
}

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

    bounds = county_gdf.total_bounds  # [minx, miny, maxx, maxy]
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
    return str(out_path), county_gdf


def run_delineation(source, engine, year, study_area, gee_project):
    """Run a single source–engine delineation, returning (GeoDataFrame, path)."""
    import geopandas as gpd

    output_path = OUTPUT_DIR / f"fields_{source}_{engine}_{year}.gpkg"

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

    # Source-specific composite parameters
    if source in ("sentinel2", "landsat", "hls"):
        kwargs["composite_method"] = "median"
        kwargs["cloud_cover_max"] = 20
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

    gdf = agribound.delineate(**kwargs)
    return gdf, output_path


def grand_ensemble_vote(results, threshold):
    """Merge results from multiple source–engine combos via majority vote."""
    from agribound.engines.ensemble import EnsembleEngine

    return EnsembleEngine._merge_vote(results, threshold)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lea County multi-source ensemble time series."
    )
    parser.add_argument(
        "--gee-project", default=None, help="GEE project ID."
    )
    return parser.parse_args()


def main():
    """Run multi-source ensemble field boundary delineation for Lea County."""
    args = parse_args()
    gee_project = args.gee_project

    import geopandas as gpd

    # --- Derive study area from Lea County subset ---
    study_area, ref_gdf = create_county_study_area(NMOSE_SHAPEFILE, COUNTY_CODE)
    print(f"Study area: Lea County ({len(ref_gdf)} reference polygons)")
    print(f"Study area GeoJSON: {study_area}")

    # ================================================================
    # Phase 1: Individual source–engine delineation (2020–2022)
    # ================================================================
    print(f"\n{'='*70}")
    print("Phase 1: Per-source, per-engine delineation")
    print(f"  Sources: {', '.join(SOURCE_ENGINE_MAP)}")
    print(f"  Years:   {min(YEARS)}–{max(YEARS)}")
    print(f"{'='*70}")

    all_results = {}  # {year: {"source/engine": gdf}}

    for year in YEARS:
        print(f"\n--- Year {year} ---")
        all_results[year] = {}

        for source, engines in SOURCE_ENGINE_MAP.items():
            yr_min, yr_max = SOURCE_YEAR_RANGE[source]
            if year < yr_min or year > yr_max:
                continue

            for engine in engines:
                tag = f"{source}/{engine}"

                try:
                    gdf, _ = run_delineation(
                        source, engine, year, study_area, gee_project
                    )
                    all_results[year][tag] = gdf
                    print(f"  {tag}: {len(gdf)} fields")
                except Exception as exc:
                    print(f"  {tag}: FAILED — {exc}")

    # ================================================================
    # Phase 2: Grand ensemble per year (vote merge)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"Phase 2: Grand ensemble (vote threshold={VOTE_THRESHOLD})")
    print(f"{'='*70}")

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

        print(f"\n  {year}: merging {len(year_results)} source–engine results...",
              end=" ")
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
    print(f"\n{'='*70}")
    print("Phase 3: Evaluation against NMOSE reference (Lea County)")
    print(f"{'='*70}")

    header = (
        f"  {'Year':<6} {'Source/Engine':<40} {'Fields':>6} "
        f"{'F1':>6} {'IoU':>6} {'P':>6} {'R':>6}"
    )
    print(f"\n{header}")
    print(f"  {'-'*6} {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

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
    print(f"\n{'='*70}")
    print("Grand Ensemble Time Series Summary")
    print(f"{'='*70}")
    print(
        f"  {'Year':<6} {'Fields':>6} {'Area (ha)':>12} "
        f"{'Sources':>8} {'F1':>6} {'IoU':>6}"
    )
    print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*8} {'-'*6} {'-'*6}")

    for year in YEARS:
        gdf = ensemble_results.get(year)
        if gdf is None:
            continue
        area_ha = (
            gdf["metrics:area"].sum() / 10000
            if "metrics:area" in gdf.columns
            else 0
        )
        n_sources = len(all_results.get(year, {}))
        try:
            m = evaluate(gdf, ref_gdf)
            print(
                f"  {year:<6} {len(gdf):>6} {area_ha:>12,.1f} "
                f"{n_sources:>8} {m['f1']:.3f} {m['iou_mean']:.3f}"
            )
        except Exception:
            print(
                f"  {year:<6} {len(gdf):>6} {area_ha:>12,.1f} {n_sources:>8}"
            )

    # ================================================================
    # Phase 5: Visualization
    # ================================================================
    print(f"\n{'='*70}")
    print("Generating maps...")
    print(f"{'='*70}")

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
        print(
            f"  Ensemble vs Reference: "
            f"{OUTPUT_DIR / 'map_ensemble_vs_reference.html'}"
        )

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
                output_html=str(
                    OUTPUT_DIR / "map_source_engine_comparison.html"
                ),
            )
            print(
                f"  Source–engine comparison: "
                f"{OUTPUT_DIR / 'map_source_engine_comparison.html'}"
            )

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
        print(
            f"  Time series: {OUTPUT_DIR / 'map_ensemble_timeseries.html'}"
        )

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
