"""
01 — New Mexico Landsat Time Series (1985–2025)

Generates annual field boundary polygons over 40 years using Landsat imagery
and the Delineate-Anything engine. Uses NMOSE (New Mexico Office of the State
Engineer) WUCB agricultural polygon boundaries for fine-tuning and evaluation.

The script operates in two modes:
  1. Fine-tuned inference — fine-tunes the engine on the NMOSE reference
     boundaries, then runs inference for all 40 years with the improved model.
  2. Evaluation — compares predictions against NMOSE reference boundaries
     to compute field-level accuracy metrics (IoU, precision, recall, F1).

Estimated runtime: ~8–12 hours (40 years of GEE composite + GPU inference).
Fine-tuning adds ~30–60 minutes. Best run on HPC/cloud with GPU.

Prerequisites:
    pip install agribound[gee,delineate-anything]
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
OUTPUT_DIR = Path("outputs/new_mexico_timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(1985, 2026)
SOURCE = "landsat"
ENGINE = "delineate-anything"

# Set to True to fine-tune the engine on NMOSE boundaries before inference
FINE_TUNE = True


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
        description="New Mexico Landsat time series field boundary delineation."
    )
    parser.add_argument(
        "--gee-project", default=None, help="GEE project ID."
    )
    return parser.parse_args()


def main():
    """Run annual field boundary delineation for New Mexico (1985–2025)."""
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

    # --- Phase 1: Fine-tune on NMOSE boundaries (optional) ---
    # The fine-tuned checkpoint is saved to disk and reused for all
    # subsequent years, so fine-tuning only runs once.
    checkpoint_path = None

    if FINE_TUNE:
        print(f"\n{'='*60}")
        print("Phase 1: Fine-tuning engine on NMOSE reference boundaries")
        print(f"{'='*60}")

        # Use a recent year for fine-tuning composite — recent imagery
        # is higher quality and best matches current reference boundaries
        finetune_year = 2024
        finetune_output = OUTPUT_DIR / f"fields_finetuned_{finetune_year}.gpkg"
        checkpoint_dir = OUTPUT_DIR / "checkpoints"

        # Check if a checkpoint already exists from a previous run
        existing_ckpts = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
        if existing_ckpts:
            checkpoint_path = str(existing_ckpts[0])
            print(f"  Reusing existing checkpoint: {checkpoint_path}")
        else:
            gdf_ft = agribound.delineate(
                study_area=study_area,
                source=SOURCE,
                year=finetune_year,
                engine=ENGINE,
                output_path=str(finetune_output),
                gee_project=gee_project,
                composite_method="median",
                cloud_cover_max=20,
                min_area=2500,
                simplify=2.0,
                device="auto",
                reference_boundaries=NMOSE_SHAPEFILE,
                fine_tune=True,
            )

            print(f"\n  Fine-tuned model produced {len(gdf_ft)} fields for {finetune_year}")

            # Evaluate fine-tuned results against NMOSE
            metrics = evaluate(gdf_ft, ref_gdf)
            print("  Fine-tuned evaluation:")
            print(f"    IoU:       {metrics['iou_mean']:.3f}")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall:    {metrics['recall']:.3f}")
            print(f"    F1:        {metrics['f1']:.3f}")

            # Locate the saved checkpoint for reuse in Phase 2
            new_ckpts = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
            if new_ckpts:
                checkpoint_path = str(new_ckpts[0])
                print(f"  Checkpoint saved: {checkpoint_path}")

    # --- Phase 2: Run inference for all years ---
    # If fine-tuning was done, pass the checkpoint via engine_params
    # so every year uses the improved model weights.
    print(f"\n{'='*60}")
    print("Phase 2: Annual field boundary delineation (1985–2025)")
    if checkpoint_path:
        print(f"  Using fine-tuned checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    engine_params = {}
    if checkpoint_path:
        engine_params["checkpoint_path"] = checkpoint_path

    all_results = {}

    for year in YEARS:
        print(f"\nProcessing year {year}...")
        output_path = OUTPUT_DIR / f"fields_landsat_{year}.gpkg"

        # Skip if already processed
        if output_path.exists():
            print(f"  Already exists: {output_path}, skipping.")
            all_results[year] = gpd.read_file(output_path)
            continue

        try:
            gdf = agribound.delineate(
                study_area=study_area,
                source=SOURCE,
                year=year,
                engine=ENGINE,
                output_path=str(output_path),
                gee_project=gee_project,
                composite_method="median",
                cloud_cover_max=20,
                min_area=2500,
                simplify=2.0,
                device="auto",
                # Evaluate each year against NMOSE (no fine-tuning)
                reference_boundaries=NMOSE_SHAPEFILE,
                fine_tune=False,
                engine_params=engine_params,
            )
            all_results[year] = gdf
            print(f"  Delineated {len(gdf)} fields for {year}")

            # Print evaluation metrics if available
            if hasattr(gdf, "attrs") and "evaluation_metrics" in gdf.attrs:
                m = gdf.attrs["evaluation_metrics"]
                print(f"  Evaluation: F1={m['f1']:.3f} IoU={m['iou_mean']:.3f} "
                      f"P={m['precision']:.3f} R={m['recall']:.3f}")
        except Exception as exc:
            print(f"  Failed for {year}: {exc}")

    # --- Phase 3: Summary statistics ---
    print(f"\n{'='*60}")
    print("Time Series Summary")
    print(f"{'='*60}")
    print(f"  {'Year':<6} {'Fields':>6} {'Area (ha)':>12} {'F1':>6} {'IoU':>6}")
    print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*6} {'-'*6}")

    for year, gdf in sorted(all_results.items()):
        area_ha = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0
        f1 = ""
        iou = ""
        if hasattr(gdf, "attrs") and "evaluation_metrics" in gdf.attrs:
            m = gdf.attrs["evaluation_metrics"]
            f1 = f"{m['f1']:.3f}"
            iou = f"{m['iou_mean']:.3f}"
        print(f"  {year:<6} {len(gdf):>6} {area_ha:>12,.1f} {f1:>6} {iou:>6}")

    # --- Phase 4: Visualization ---
    print(f"\n{'='*60}")
    print("Generating maps...")
    print(f"{'='*60}")

    # Map of latest year with NMOSE reference overlay
    if all_results:
        latest_year = max(all_results.keys())
        latest_gdf = all_results[latest_year]

        # Show predicted vs reference boundaries
        from agribound.visualize import show_comparison

        m = show_comparison(
            [latest_gdf, ref_gdf],
            labels=[f"Predicted ({latest_year})", "NMOSE Reference"],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_predicted_vs_reference.html"),
        )
        print(f"  Predicted vs Reference: {OUTPUT_DIR / 'map_predicted_vs_reference.html'}")

    # Multi-year comparison
    selected_years = [1985, 1995, 2005, 2015, 2025]
    boundaries_list = []
    labels = []
    for year in selected_years:
        if year in all_results:
            boundaries_list.append(all_results[year])
            labels.append(str(year))

    if boundaries_list:
        show_comparison(
            boundaries_list,
            labels=labels,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_timeseries_comparison.html"),
        )
        print(f"  Time series comparison: {OUTPUT_DIR / 'map_timeseries_comparison.html'}")

    # Latest year standalone map
    if all_results:
        agribound.show_boundaries(
            all_results[max(all_results.keys())],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_latest.html"),
        )
        print(f"  Latest year map: {OUTPUT_DIR / 'map_latest.html'}")


if __name__ == "__main__":
    main()
