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

from pathlib import Path

import agribound
from agribound.evaluate import evaluate

# --- Configuration ---
STUDY_AREA = "examples/NMOSE Field Boundaries/nm_study_area.geojson"
NMOSE_SHAPEFILE = "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
GEE_PROJECT = "your-gee-project"  # Replace with your GEE project ID
OUTPUT_DIR = Path("outputs/new_mexico_timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(1985, 2026)
SOURCE = "landsat"
ENGINE = "delineate-anything"

# Set to True to fine-tune the engine on NMOSE boundaries before inference
FINE_TUNE = True


def main():
    """Run annual field boundary delineation for New Mexico (1985–2025)."""
    import geopandas as gpd

    # --- Load NMOSE reference boundaries ---
    print("Loading NMOSE reference boundaries...")
    ref_gdf = gpd.read_file(NMOSE_SHAPEFILE)
    print(f"  {len(ref_gdf)} reference field polygons loaded")

    # --- Phase 1: Fine-tune on NMOSE boundaries (optional) ---
    if FINE_TUNE:
        print(f"\n{'='*60}")
        print("Phase 1: Fine-tuning engine on NMOSE reference boundaries")
        print(f"{'='*60}")

        # Use a recent year for fine-tuning composite
        finetune_year = 2024
        finetune_output = OUTPUT_DIR / f"fields_finetuned_{finetune_year}.gpkg"

        gdf_ft = agribound.delineate(
            study_area=STUDY_AREA,
            source=SOURCE,
            year=finetune_year,
            engine=ENGINE,
            output_path=str(finetune_output),
            gee_project=GEE_PROJECT,
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
        print(f"  Fine-tuned evaluation:")
        print(f"    IoU:       {metrics['iou_mean']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1:        {metrics['f1']:.3f}")

    # --- Phase 2: Run inference for all years ---
    print(f"\n{'='*60}")
    print("Phase 2: Annual field boundary delineation (1985–2025)")
    print(f"{'='*60}")

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
                study_area=STUDY_AREA,
                source=SOURCE,
                year=year,
                engine=ENGINE,
                output_path=str(output_path),
                gee_project=GEE_PROJECT,
                composite_method="median",
                cloud_cover_max=20,
                min_area=2500,
                simplify=2.0,
                device="auto",
                # Evaluate each year against NMOSE (no fine-tuning)
                reference_boundaries=NMOSE_SHAPEFILE,
                fine_tune=False,
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
        m_compare = show_comparison(
            boundaries_list,
            labels=labels,
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_timeseries_comparison.html"),
        )
        print(f"  Time series comparison: {OUTPUT_DIR / 'map_timeseries_comparison.html'}")

    # Latest year standalone map
    if all_results:
        m_latest = agribound.show_boundaries(
            all_results[max(all_results.keys())],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / "map_latest.html"),
        )
        print(f"  Latest year map: {OUTPUT_DIR / 'map_latest.html'}")


if __name__ == "__main__":
    main()
