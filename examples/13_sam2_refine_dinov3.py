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
(``outputs/lea_county_ensemble/fields_sentinel2_dinov3_2024.gpkg``; run
example 12 first, or pass ``--input``). The raster the polygons were
delineated from and the run's configuration are read from the input's
provenance record (``<input>.provenance.json``), so SAM sees exactly that
composite. The refined output is named after the input
(``<input stem>_<backend>.gpkg``), so it carries the input's year.

Backends (``--sam-backend``): ``sam2`` (default; SAM 2.0 via segment-geospatial),
``sam2.1``, ``sam3`` (Meta SAM 3: CUDA GPU and triton required; Linux,
Windows only through the community triton-windows wheel, not macOS) and
``sam3-hf`` (Hugging Face transformers SAM 3, no triton). SAM 3 weights are
gated on Hugging Face (request access first). Pipeline runs can do the same
with ``sam_refine=True`` (example 14).

Evaluation: against the reference boundaries recorded in the input's
provenance record (``config.reference_boundaries``; for the default input,
example 12's DRI/NMOSE field boundaries (V18) of the study area) or given
with ``--reference``, restricted to the input's study area by the pipeline's
rule (representative point inside), without the polygons smaller than
2,500 m2, the predictions' minimum area (areas in EPSG:6933, as agribound's
area filter computes them; the script prints how many it drops: for the
default input 1 of the 389 V18 polygons in its study area). ``--reference``
also takes a file inside a zip archive. For example, to evaluate example
20's output, which was not fine-tuned, against the whole V18 layer (1,355 of
the 1,705 V18 polygons in its study area are at least 2,500 m2), pass

    --input outputs/stratified_evaluation/fields_sentinel2_delineate-anything_2024.gpkg
    --reference 'zip://examples/NM_Field_Boundaries.zip!Master_NM_Field_Boundaries_V18.shp'

When the input was fine-tuned (``config.fine_tune``), as the default DINOv3
input was, the evaluation is labelled in-sample if the reference is the
fine-tuning reference file. With another reference, the script counts the
evaluated polygons whose representative point lies in a polygon of the
fine-tuning reference and labels the evaluation in-sample (all of them),
partly in-sample (some) or not in-sample (none); "possibly in-sample" when the
fine-tuning reference file is missing. Without a reference the evaluation is
skipped.

Runtime: 16 s in the development run of agribound 1.1 (2026-10-09, Apple M2
Max with MPS, the SAM 2 weights already cached), with example 20's output as
input (357 polygons, 72 of them prompted; SAM 2 10 s), the evaluation
included.

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
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/lea_county_ensemble")
INPUT_GPKG = OUTPUT_DIR / "fields_sentinel2_dinov3_2024.gpkg"
MIN_AREA = 2500  # m^2; the predictions' minimum area (agribound's default), also the reference's


def reference_exists(path):
    """True if *path* exists; for ``zip://ARCHIVE!MEMBER``, if the archive exists."""
    if str(path).startswith("zip://"):
        return Path(str(path)[len("zip://") :].split("!", 1)[0]).exists()
    return Path(path).exists()


def n_in_training(ref, training_path):
    """Number of *ref* polygons whose representative point lies in a polygon of *training_path*."""
    import numpy as np

    training = gpd.read_file(training_path).to_crs(ref.crs)
    points = ref.representative_point().to_numpy()
    return len(np.unique(training.sindex.query(points, predicate="within")[0]))


def evaluation_label(input_config, ref_file, ref):
    """Label of the evaluation: in-sample, partly in-sample or neither.

    An input fine-tuned on the same reference file is evaluated in-sample. With
    another reference, the evaluated polygons that lie in the fine-tuning reference
    (by representative point) are counted.
    """
    recorded = input_config.get("reference_boundaries")
    if not input_config.get("fine_tune") or not recorded:
        return "Evaluation"
    if Path(ref_file).resolve() == Path(recorded).resolve():
        return "In-sample evaluation"
    if not reference_exists(recorded):
        return f"Evaluation (possibly in-sample: the input was fine-tuned on {recorded})"
    shared = n_in_training(ref, recorded)
    where = f"the fine-tuning reference {recorded}"
    if shared == len(ref) and shared:
        return f"In-sample evaluation (all {shared} polygons lie in {where})"
    if shared:
        return f"Partly in-sample evaluation ({shared} of {len(ref)} polygons lie in {where})"
    return f"Evaluation (no polygon lies in {where})"


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
        help="Reference boundaries for the evaluation (default: the input's recorded "
        "reference); a file in a zip archive as zip://ARCHIVE!MEMBER.",
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
    ref_file = args.reference or input_config.get("reference_boundaries")
    if ref_file and reference_exists(ref_file):
        from agribound.config import AgriboundConfig
        from agribound.evaluate import evaluate
        from agribound.pipeline import select_in_study_area, study_area_in_crs
        from agribound.postprocess import filter_polygons

        ref = gpd.read_file(ref_file)
        n_total = len(ref)
        if input_config.get("study_area"):
            aoi = study_area_in_crs(AgriboundConfig.from_dict(input_config), ref.crs)
            ref = select_in_study_area(ref, aoi, "representative_point")[0]
        n_area = len(ref)
        # The predictions' minimum-area filter (areas in EPSG:6933), applied to the reference.
        ref = filter_polygons(ref, min_area_m2=MIN_AREA)
        label = evaluation_label(input_config, ref_file, ref)
        m_before, m_after = evaluate(gdf, ref), evaluate(refined, ref)
        print(
            f"\n{label} against {len(ref)} of {n_total} reference polygons ({ref_file}; "
            f"{n_area - len(ref)} in the study area smaller than {MIN_AREA:,} m2 dropped):"
        )
        print(f"{'':<26} {'Before':>10} {'After':>10}")
        for key in ("f1", "precision", "recall", "iou_mean", "boundary_distance_mean_m"):
            print(f"{key:<26} {m_before[key]:>10.3f} {m_after[key]:>10.3f}")
        layers.append(ref)
        labels.append("Reference")
    elif ref_file:
        print(f"\nReference {ref_file} not found; evaluation skipped.")
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
