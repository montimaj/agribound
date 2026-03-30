"""
13 — SAM2 Boundary Refinement on DINOv3 Field Boundaries

Standalone script that takes pre-computed DINOv3 field boundaries and
refines them using SAM2 box-prompted segmentation.  Each polygon's
bounding box is fed to SAM2 as a prompt, producing pixel-accurate masks
that replace the original geometry.

This demonstrates SAM2 refinement as a separate post-processing step,
decoupled from the main delineation pipeline.

Input:  fields_presam_sentinel2_dinov3_2020.gpkg  (555 fields from DINOv3)
Raster: sentinel2_2020_composite.tif              (Sentinel-2 annual composite)
Output: fields_sam2_sentinel2_dinov3_2020.gpkg     (SAM2-refined boundaries)

Prerequisites:
    pip install agribound[samgeo]
"""

import argparse
import logging
import time
from pathlib import Path

import geopandas as gpd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy SAM2 per-field "Image embeddings computed" messages
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("agribound").setLevel(logging.INFO)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/lea_county_ensemble")
INPUT_GPKG = OUTPUT_DIR / "fields_sentinel2_dinov3_2020.gpkg"
RASTER_PATH = OUTPUT_DIR / ".agribound_cache" / "sentinel2_2020_composite.tif"
OUTPUT_GPKG = OUTPUT_DIR / "fields_sam2_sentinel2_dinov3_2020.gpkg"
OUTPUT_CRS = "EPSG:26913"  # Match NMOSE reference CRS (NAD83 / UTM zone 13N)

# SAM2 parameters
SAM_MODEL = "large"  # "tiny", "small", "base_plus", "large"
SAM_BATCH_SIZE = 100  # Number of field boxes per SAM2 batch


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 boundary refinement on DINOv3 output.")
    parser.add_argument("--input", default=str(INPUT_GPKG), help="Input field boundary GPKG.")
    parser.add_argument("--raster", default=str(RASTER_PATH), help="Input satellite raster.")
    parser.add_argument("--output", default=str(OUTPUT_GPKG), help="Output refined GPKG.")
    parser.add_argument(
        "--sam-model",
        default=SAM_MODEL,
        choices=["tiny", "small", "base_plus", "large"],
        help="SAM2 model variant.",
    )
    parser.add_argument("--batch-size", type=int, default=SAM_BATCH_SIZE, help="SAM2 batch size.")
    parser.add_argument(
        "--smooth-only",
        action="store_true",
        help="Skip SAM2, load existing output, and re-apply smoothing.",
    )
    parser.add_argument(
        "--smooth-iterations", type=int, default=3, help="Chaikin smoothing iterations (default 3)."
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=2.0,
        help="RDP simplification tolerance in meters (default 2.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    raster_path = Path(args.raster)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input GPKG not found: {input_path}")
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    # Load input boundaries
    gdf = gpd.read_file(input_path)
    print(f"Loaded {len(gdf)} field boundaries from {input_path}")
    print(f"CRS: {gdf.crs}")

    # --smooth-only: reload existing output and re-apply smoothing
    if args.smooth_only:
        if not output_path.exists():
            raise FileNotFoundError(f"--smooth-only requires existing output: {output_path}")
        print(f"\nLoading existing output for re-smoothing: {output_path}")
        refined = gpd.read_file(output_path)
        print(f"Loaded {len(refined)} boundaries")

        from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

        refined = smooth_polygons(refined, iterations=args.smooth_iterations)
        refined = simplify_polygons(refined, tolerance=args.simplify_tolerance)
        print(
            f"Re-smoothed: {len(refined)} fields "
            f"(iterations={args.smooth_iterations}, tolerance={args.simplify_tolerance})"
        )

        if refined.crs is not None and str(refined.crs) != OUTPUT_CRS:
            refined = refined.to_crs(OUTPUT_CRS)
        if output_path.exists():
            output_path.unlink()
        refined.to_file(output_path, driver="GPKG", layer="fields")
        print(f"Saved to {output_path}")

    elif output_path.exists():
        print(f"\nOutput already exists: {output_path}")
        refined = gpd.read_file(output_path)
        print(f"Loaded {len(refined)} refined boundaries")
    else:
        # Configure SAM2 refinement
        from agribound.config import AgriboundConfig
        from agribound.engines.samgeo_engine import refine_boundaries

        config = AgriboundConfig(
            source="sentinel2",
            engine="dinov3",
            year=2020,
            study_area="",
            output_path=str(output_path),
            engine_params={
                "sam_model": args.sam_model,
                "sam_batch_size": args.batch_size,
            },
            device="auto",
        )

        # Run SAM2 refinement
        print(f"\nRunning SAM2 refinement (model={args.sam_model}, batch_size={args.batch_size})")
        tic = time.time()
        refined = refine_boundaries(gdf, str(raster_path), config)
        elapsed = time.time() - tic
        print(f"SAM2 refined {len(refined)} boundaries in {elapsed:.1f}s")

        # Clean up invalid/empty geometries from SAM refinement
        refined = refined[~refined.geometry.is_empty].copy()
        refined = refined[refined.geometry.is_valid].copy()
        refined = refined[refined.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        refined = refined.reset_index(drop=True)
        print(f"{len(gdf) - len(refined)} fields removed due to invalid geometry")

        # Smooth and simplify final boundaries
        from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

        refined = smooth_polygons(refined, iterations=args.smooth_iterations)
        refined = simplify_polygons(refined, tolerance=args.simplify_tolerance)
        print(
            f"Smoothed and simplified to {len(refined)} fields "
            f"(iterations={args.smooth_iterations}, tolerance={args.simplify_tolerance})"
        )

        # Reproject to match NMOSE reference CRS
        if refined.crs is not None and str(refined.crs) != OUTPUT_CRS:
            refined = refined.to_crs(OUTPUT_CRS)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        refined.to_file(output_path, driver="GPKG", layer="fields")
        print(f"Saved to {output_path}")

    # Compare before/after
    print(f"\n{'Metric':<20} {'Before':>10} {'After':>10}")
    print(f"{'-' * 20} {'-' * 10} {'-' * 10}")

    # Project to metric CRS for area comparison
    gdf_m = gdf.to_crs(OUTPUT_CRS) if gdf.crs and gdf.crs.is_geographic else gdf
    ref_m = refined.to_crs(OUTPUT_CRS) if refined.crs and refined.crs.is_geographic else refined

    area_before = gdf_m.geometry.area.sum() / 10000
    area_after = ref_m.geometry.area.sum() / 10000
    print(f"{'Fields':<20} {len(gdf):>10} {len(refined):>10}")
    print(f"{'Total area (ha)':<20} {area_before:>10,.1f} {area_after:>10,.1f}")

    # Evaluate against NMOSE reference if available
    nmose_path = Path("examples/NMOSE Field Boundaries/WUCB ag polys.shp")
    if nmose_path.exists():
        from agribound.evaluate import evaluate

        ref_gdf = gpd.read_file(nmose_path)
        ref_county = ref_gdf[ref_gdf["County"] == "25"].copy()

        print(f"\nEvaluation against NMOSE reference ({len(ref_county)} polygons):")
        print(f"{'Metric':<20} {'Before':>10} {'After':>10}")
        print(f"{'-' * 20} {'-' * 10} {'-' * 10}")

        try:
            m_before = evaluate(gdf, ref_county)
            m_after = evaluate(refined, ref_county)
            for key in ["f1", "iou_mean", "precision", "recall"]:
                print(f"{key:<20} {m_before[key]:>10.3f} {m_after[key]:>10.3f}")
        except Exception as exc:
            print(f"  Evaluation failed: {exc}")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # Force exit — geedim\'s async runner hangs on cleanup
