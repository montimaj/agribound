"""
13 — SAM Boundary Refinement of Existing DINOv3 Field Boundaries

Refines a finished delineation with box-prompted SAM as a separate step:
each polygon's bounding box is given to SAM as a single-object box prompt,
and the polygon is replaced by SAM's mask, of which only the part inside the
box padded by ``sam_crop_padding`` is kept. Polygons whose padded box is
smaller than ``sam_min_crop_px`` pixels on either side are not prompted and
keep their geometry, and so does a polygon whose mask covers less than
``engine_params["sam_min_coverage"]`` (default 0.5, since agribound 1.0.1) of
it (``gdf.attrs["sam_stats"]`` counts both).

Input: the Sentinel-2 DINOv3 output of example 12
(``outputs/lea_county_ensemble/fields_sentinel2_dinov3_2022.gpkg``; run
example 12 first, or pass ``--input``). The raster the polygons were
delineated from and the run's configuration are read from the input's
provenance record (``<input>.provenance.json``), so SAM sees exactly that
composite.

Backends (``--sam-backend``): ``sam2`` (default; SAM 2.0 via segment-geospatial),
``sam2.1``, ``sam3`` (Meta SAM 3: CUDA GPU and triton required; Linux,
Windows only through the community triton-windows wheel, not macOS) and
``sam3-hf`` (Hugging Face transformers SAM 3, no triton). SAM 3 weights are
gated on Hugging Face (request access first). Pipeline runs can do the same
with ``sam_refine=True`` (example 14).

Evaluation: against the reference boundaries recorded in the input's
provenance record (``config.reference_boundaries``; for the default input,
example 12's NMOSE polygons) or given with ``--reference``, restricted to the
input's study area by the pipeline's rule (representative point inside). It
is labelled in-sample when the input was fine-tuned (``config.fine_tune``)
on that same reference file, as the default DINOv3 input was. Without a
reference the evaluation is skipped.

Estimated runtime (not measured for 1.0): a few minutes on a GPU for a few
hundred polygons.

Prerequisites:
    pip install "agribound[samgeo]"      (sam3: "agribound[sam3]")
    Run from the repository root: python examples/13_sam2_refine_dinov3.py
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import geopandas as gpd

from agribound.provenance import read_provenance

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/lea_county_ensemble")
INPUT_GPKG = OUTPUT_DIR / "fields_sentinel2_dinov3_2022.gpkg"


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="SAM refinement of an existing delineation.")
    parser.add_argument("--input", default=str(INPUT_GPKG), help="agribound output to refine.")
    parser.add_argument("--output", default=None, help="Refined output (default: derived).")
    parser.add_argument(
        "--sam-backend", default="sam2", choices=["sam2", "sam2.1", "sam3", "sam3-hf"]
    )
    parser.add_argument(
        "--sam-model",
        default=None,
        help="Model id or, for sam2/sam2.1, a size alias (tiny, small, base_plus, large). "
        "Default: the backend's default (sam2: facebook/sam2-hiera-large).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Boxes per SAM decoder call.")
    parser.add_argument("--min-crop-px", type=int, default=64, help="sam_min_crop_px.")
    parser.add_argument("--padding", type=float, default=0.15, help="sam_crop_padding.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute an existing output.")
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference boundaries for the evaluation (default: the input's recorded reference).",
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}. Run example 12 first or pass --input.")
        return
    record = read_provenance(input_path)
    if record is None or "raster_path" not in (record.get("facts") or {}):
        print(f"{input_path} has no provenance record with a raster path; cannot refine it.")
        return
    raster_path = record["facts"]["raster_path"]
    if not Path(raster_path).exists():
        print(f"Raster {raster_path} (from the provenance record) no longer exists.")
        return
    output_path = Path(
        args.output or input_path.with_name(f"{input_path.stem}_{args.sam_backend}.gpkg")
    )

    gdf = gpd.read_file(input_path)
    print(f"Loaded {len(gdf)} polygons from {input_path} ({gdf.crs})")
    print(f"Raster: {raster_path}")

    if output_path.exists() and not args.overwrite:
        refined = gpd.read_file(output_path)
        # Reused by file name only: an output written by agribound 1.0.0 (no coverage test)
        # is loaded too, so pass --overwrite after upgrading.
        print(f"Loaded existing refined output {output_path} (use --overwrite to recompute)")
    else:
        from agribound.config import AgriboundConfig
        from agribound.engines.samgeo_engine import refine_boundaries
        from agribound.postprocess import filter_polygons
        from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

        config = AgriboundConfig.from_dict(record["config"]).merged(
            sam_backend=args.sam_backend,
            sam_model=args.sam_model,
            sam_min_crop_px=args.min_crop_px,
            sam_crop_padding=args.padding,
        )
        tic = time.time()
        refined = refine_boundaries(gdf, raster_path, config, batch_size=args.batch_size)
        stats = refined.attrs.get("sam_stats", {})
        print(
            f"\nSAM ({stats.get('backend')}, {stats.get('model')}, {stats.get('device')}) in "
            f"{time.time() - tic:.1f} s: refined {stats.get('n_refined')} of "
            f"{stats.get('n_total')}; too small {stats.get('n_skipped_small')}, outside the "
            f"raster {stats.get('n_skipped_outside')}, failed {stats.get('n_failed')}, "
            f"covering too little {stats.get('n_low_coverage')}"
        )
        # Area filter, smoothing and simplification, as in the pipeline's post-processing.
        refined = filter_polygons(
            refined,
            min_area_m2=config.min_field_area_m2,
            remove_holes_below_m2=config.min_field_area_m2,
        )
        refined = smooth_polygons(refined, iterations=3)
        refined = simplify_polygons(refined, tolerance=config.simplify_tolerance)
        refined.to_file(output_path, driver="GPKG", layer="fields")
        print(f"Saved {len(refined)} polygons to {output_path}")

    from agribound.io.crs import get_equal_area_crs

    ea = get_equal_area_crs()
    print(f"\n{'Metric':<20} {'Before':>10} {'After':>10}")
    print(f"{'Polygons':<20} {len(gdf):>10} {len(refined):>10}")
    before_ha = gdf.to_crs(ea).area.sum() / 10000
    after_ha = refined.to_crs(ea).area.sum() / 10000
    print(f"{'Total area (ha)':<20} {before_ha:>10,.1f} {after_ha:>10,.1f}")

    layers, labels = [gdf, refined], ["Before SAM", f"After {args.sam_backend}"]
    input_config = record["config"]
    recorded_ref = input_config.get("reference_boundaries")
    ref_file = args.reference or recorded_ref
    if ref_file and Path(ref_file).exists():
        from agribound.config import AgriboundConfig
        from agribound.evaluate import evaluate
        from agribound.pipeline import select_in_study_area, study_area_in_crs

        ref = gpd.read_file(ref_file)
        n_total = len(ref)
        if input_config.get("study_area"):
            aoi = study_area_in_crs(AgriboundConfig.from_dict(input_config), ref.crs)
            ref = select_in_study_area(ref, aoi, "representative_point")[0]
        in_sample = bool(input_config.get("fine_tune")) and (
            recorded_ref is not None and Path(ref_file).resolve() == Path(recorded_ref).resolve()
        )
        label = "In-sample evaluation" if in_sample else "Evaluation"
        m_before, m_after = evaluate(gdf, ref), evaluate(refined, ref)
        print(f"\n{label} against {len(ref)} of {n_total} reference polygons ({ref_file}):")
        print(f"{'':<26} {'Before':>10} {'After':>10}")
        for key in ("f1", "precision", "recall", "iou_mean", "boundary_distance_mean_m"):
            print(f"{key:<26} {m_before[key]:>10.3f} {m_after[key]:>10.3f}")
        layers.append(ref)
        labels.append("Reference")
    else:
        print("\nNo reference boundaries recorded for the input; evaluation skipped (--reference).")

    from agribound.visualize import show_comparison

    map_path = output_path.with_suffix(".html")
    web_map = show_comparison(
        layers, labels=labels, basemap="Esri.WorldImagery", output_html=str(map_path)
    )
    show_in_notebook(web_map)
    print(f"\nMap: {map_path}")


if __name__ == "__main__":
    main()
