"""
01 — New Mexico Landsat Time Series with a Fine-Tuned Delineate-Anything Model

Fine-tunes Delineate-Anything on the DRI/NMOSE field boundaries (V18; Desert
Research Institute and New Mexico Office of the State Engineer), then
delineates field boundaries from annual Landsat composites and evaluates
every year against the same polygons.

What it shows:
    1. Fine-tuning (``fine_tune=True``) on a reference layer. The checkpoint
       path is read back from the run's provenance record
       (``<output>.provenance.json``, ``facts["fine_tuned_checkpoint"]``).
    2. Re-using that checkpoint for every year with
       ``engine_params={"checkpoint_path": ...}``.
    3. The pipeline's built-in evaluation (``reference_boundaries`` without
       fine-tuning) and a time-series summary.

Data and caveats:
    - Source: Landsat 5/7/8/9 Collection 2 Level-2 surface reflectance (x 10000),
      30 m. Engine: Delineate-Anything (default model ``large_v2``). The
      published models were trained on 0.25-10 m imagery; 30 m Landsat is
      outside that range, so agribound logs a WARNING and records
      ``engine_meta["gsd_outside_training_range"] = True``.
    - Years: 2025 only by default (``--years``), on purpose: a held-out-year
      check of the fine-tuning, which uses the 2024 composite (2024 is also
      the year the other New Mexico examples, 12, 14 and 20, evaluate). The
      model never sees 2025 imagery, but the V18 polygons it is evaluated
      against are its training labels, so the 2025 scores are held out in
      time, not in space (see the evaluation below). The default run builds
      two state-wide composites (2024 for fine-tuning, 2025 for delineation)
      and also evaluates the fine-tuning run's own 2024 output (in-sample).
      The full run is ``--years 1985-2025`` (41 years); applying the
      checkpoint to Landsat 5 TM / 7 ETM+ years is a transfer to other
      sensors.
    - Fine-tuning epochs: the default ``--fine-tune-epochs 1`` is a quick
      test of the workflow, not a usable model; pass more epochs for a real
      run (for example 10-20; the pipeline default is 20; not benchmarked
      here). ``--no-fine-tune`` uses the published weights instead.
    - Outputs: ``fields_landsat_da_finetune-<E>ep_2024.gpkg`` (fine-tuning
      run), then ``fields_landsat_da_ft<E>ep_<year>.gpkg`` per year, or
      ``fields_landsat_da_pretrained_<year>.gpkg`` with ``--no-fine-tune``
      (``<E>`` = epochs). An existing output made with the same settings is
      loaded instead of recomputed; ``--overwrite`` recomputes it (outputs
      of an earlier version of this example, made with the NMOSE WUCB
      polygons and their bounding box, need it).
    - Reference: the V18 polygons are read from the zip archive and copied
      once to ``nm_reference_v18.gpkg`` in the output directory (agribound
      reads ``reference_boundaries`` from a vector file, not from a zip
      archive); 37 of the 85,316 records have no geometry and are left out.
      Fine-tuning uses all 85,279 polygons, whatever their size. The
      evaluations use a second copy, ``nm_reference_v18_min2500.gpkg``,
      without the 16,922 polygons smaller than 2,500 m2, the predictions'
      minimum area (``min_area``; areas in EPSG:6933, as agribound's area
      filter computes them): 68,357 polygons. The script prints both counts.
    - Study area: the bounding box of the 85,279 V18 polygons (all of New
      Mexico). The 30 m composite of that box is about 19,200 x 20,700 pixels
      x 6 bands of float32 (~9.5 GB per year). For state-wide production runs,
      tile the area with ``agribound.hpc`` (example 19,
      ``examples/regions/new_mexico_statewide_us.yaml``).
    - Evaluation is in-sample in space: the fine-tuned model was trained on
      the same V18 polygons it is evaluated against (a spatial block split
      holds out validation chips during training, but the evaluation uses
      every polygon of at least 2,500 m2); in the default run only the year
      (2025) is held out. V18 records no imagery date per field, and it may
      not include every field in the box; predictions of fields it lacks
      count as false positives. See example 20 for a stratified evaluation.
    - The LULC crop filter is on (NLCD is selected for New Mexico) and needs
      Earth Engine, like the Landsat composites.

Estimated runtime (not measured: the example was not run end to end for 1.0
or 1.1): hours per year for the state-wide box (composite download plus GPU
inference); fine-tuning adds more. Best run on HPC/cloud with a GPU.

Prerequisites:
    pip install "agribound[gee,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    DRI/NMOSE field boundaries (V18) at "examples/NM_Field_Boundaries.zip" (local
        data, not distributed; read in place, no need to unzip)
    Run from the repository root: python examples/01_new_mexico_landsat_timeseries.py
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import agribound
from agribound.evaluate import evaluate
from agribound.provenance import read_provenance

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", message=".*organizePolygons.*")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
REFERENCE_ZIP = "examples/NM_Field_Boundaries.zip"
REFERENCE = f"zip://{REFERENCE_ZIP}!Master_NM_Field_Boundaries_V18.shp"
REFERENCE_LABEL = "DRI/NMOSE field boundaries (V18)"
OUTPUT_DIR = Path("outputs/new_mexico_timeseries")
SOURCE = "landsat"
ENGINE = "delineate-anything"
FINE_TUNE_YEAR = 2024  # Composite used for fine-tuning
FINE_TUNE_EPOCHS = 1  # quick workflow test; pass --fine-tune-epochs for a real run
# 2025: a year held out from the fine-tuning (FINE_TUNE_YEAR). Full run: "1985-2025".
DEFAULT_YEARS = "2025"
MIN_AREA = 2500  # m^2; the predictions' minimum area (min_area), also the evaluated reference's
EVAL_REFERENCE = f"nm_reference_v18_min{MIN_AREA}.gpkg"  # the polygons evaluated


def parse_years(text):
    """Parse "2025", "2020,2022" or "1985-2025" into a list of years."""
    years = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            first, last = (int(v) for v in part.split("-"))
            years.extend(range(first, last + 1))
        elif part:
            years.append(int(part))
    return sorted(set(years))


def copy_reference(reference, out_path):
    """Copy the reference polygons that have a geometry to a GeoPackage, once.

    agribound reads ``reference_boundaries`` from a vector file (GeoPackage,
    Shapefile, GeoJSON, ...), not from a zip archive. The copy is written only
    once because its modification time is part of the fine-tuning cache key.
    """
    import geopandas as gpd

    if not out_path.exists():
        ref = gpd.read_file(reference)
        ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
        ref.to_file(out_path, driver="GPKG", layer="fields")
    return str(out_path)


def evaluation_reference(ref_gdf, out_path):
    """The reference polygons evaluated: those of at least MIN_AREA, written once to *out_path*.

    The predictions' area filter (areas in EPSG:6933) drops the smaller polygons; the
    fine-tuning uses them all. The pipeline evaluates against the file.
    """
    import geopandas as gpd

    from agribound.postprocess import filter_polygons

    if not out_path.exists():
        kept = filter_polygons(ref_gdf, min_area_m2=MIN_AREA)
        kept.to_file(out_path, driver="GPKG", layer="fields")
    return gpd.read_file(out_path)


def create_study_area_from_vector(path, out_path):
    """Write the EPSG:4326 bounding box of a vector file as a GeoJSON study area."""
    import geopandas as gpd

    bounds = gpd.read_file(path).to_crs(epsg=4326).total_bounds
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    bbox_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
                    ],
                },
                "properties": {"name": "DRI/NMOSE V18 bounding box"},
            }
        ],
    }
    out_path.write_text(json.dumps(bbox_geojson))
    return str(out_path)


def print_metrics(label, m):
    """Print the main object-level metrics of agribound.evaluate.evaluate()."""
    print(
        f"  {label}: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
        f"IoU(matched)={m['iou_mean']:.3f} area-weighted R={m['area_weighted_recall']:.3f} "
        f"(TP={m['count_tp']} FP={m['count_fp']} FN={m['count_fn']}; "
        f"FP overlapping no reference field: {m['count_fp_unassigned']})"
    )


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(
        description="New Mexico Landsat time series with a fine-tuned Delineate-Anything model."
    )
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument(
        "--years",
        default=DEFAULT_YEARS,
        help='Years to delineate, e.g. "2025", "2020,2022" or "1985-2025".',
    )
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS, help="Fine-tuning epochs."
    )
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="Use the published Delineate-Anything weights (label-free) instead.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute outputs that already exist."
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def map_ready(gdf):
    """Copy of *gdf* with datetime columns as ISO-8601 text, for an HTML map.

    leafmap 0.63 cannot write pandas Timestamp values (such as agribound's
    ``determination:datetime`` column) into the map's HTML: it raises a JSON
    serialisation error.
    """
    import pandas as pd

    out = gdf.copy()
    for column in out.columns:
        if column != out.geometry.name and pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].map(lambda t: t.isoformat() if pd.notna(t) else None)
    return out


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main():
    """Fine-tune once, then delineate and evaluate every requested year."""
    args = parse_args()
    years = parse_years(args.years)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(REFERENCE_ZIP).exists():
        raise SystemExit(
            f"{REFERENCE_LABEL} not found: {REFERENCE_ZIP} (local data, not distributed)"
        )

    import geopandas as gpd

    ref_path = copy_reference(REFERENCE, OUTPUT_DIR / "nm_reference_v18.gpkg")
    study_area = create_study_area_from_vector(ref_path, OUTPUT_DIR / "nm_study_area.geojson")
    print(f"Study area (bounding box of the reference): {study_area}")
    ref_gdf = gpd.read_file(ref_path)
    print(f"{REFERENCE_LABEL}: {len(ref_gdf)} polygons ({ref_gdf.crs}; copy: {ref_path})")
    eval_path = OUTPUT_DIR / EVAL_REFERENCE
    eval_gdf = evaluation_reference(ref_gdf, eval_path)
    print(
        f"  Evaluated: {len(eval_gdf)} polygons; {len(ref_gdf) - len(eval_gdf)} smaller than "
        f"{MIN_AREA:,} m2 (the predictions' minimum area) left out ({eval_path}). Fine-tuning "
        "uses all."
    )

    common = dict(
        study_area=study_area,
        source=SOURCE,
        engine=ENGINE,
        gee_project=args.gee_project,
        composite_method="median",
        cloud_cover_max=20,
        min_area=MIN_AREA,
        simplify=2.0,
        overwrite=args.overwrite,
    )
    # The output names record the fine-tuning setting, so runs with different
    # settings do not collide with each other's outputs.
    model_tag = "pretrained" if args.no_fine_tune else f"ft{args.fine_tune_epochs}ep"

    # --- Phase 1: fine-tune on the reference polygons -----------------------
    # The checkpoint is cached under OUTPUT_DIR/.agribound_cache and keyed by
    # the study area, source, year, reference file, epochs, split and seed, so
    # re-running this script does not retrain.
    engine_params = {}
    if not args.no_fine_tune:
        print(f"\n{'=' * 60}\nPhase 1: fine-tuning on {FINE_TUNE_YEAR} Landsat\n{'=' * 60}")
        ft_output = (
            OUTPUT_DIR
            / f"fields_landsat_da_finetune-{args.fine_tune_epochs}ep_{FINE_TUNE_YEAR}.gpkg"
        )
        gdf_ft = agribound.delineate(
            **common,
            year=FINE_TUNE_YEAR,
            output_path=str(ft_output),
            reference_boundaries=ref_path,
            fine_tune=True,
            fine_tune_epochs=args.fine_tune_epochs,
        )
        record = read_provenance(ft_output) or {}
        checkpoint_path = (record.get("facts") or {}).get("fine_tuned_checkpoint")
        if not checkpoint_path:
            raise SystemExit(f"No fine_tuned_checkpoint in {ft_output}.provenance.json")
        engine_params["checkpoint_path"] = checkpoint_path
        print(f"  {len(gdf_ft)} fields in {FINE_TUNE_YEAR}; checkpoint: {checkpoint_path}")
        # The pipeline does not evaluate fine-tuning runs; this is in-sample.
        print_metrics("In-sample evaluation (training polygons)", evaluate(gdf_ft, eval_gdf))

    # --- Phase 2: one run per year -------------------------------------------
    print(f"\n{'=' * 60}\nPhase 2: annual delineation {years[0]}-{years[-1]}\n{'=' * 60}")
    all_results = {}
    for year in years:
        output_path = OUTPUT_DIR / f"fields_landsat_da_{model_tag}_{year}.gpkg"
        print(f"\nYear {year} -> {output_path}")
        try:
            # An existing output with a matching provenance record is loaded
            # instead of recomputed (unless --overwrite).
            gdf = agribound.delineate(
                **common,
                year=year,
                output_path=str(output_path),
                reference_boundaries=str(eval_path),  # evaluation only
                engine_params=dict(engine_params),
            )
        except Exception as exc:
            print(f"  Failed for {year}: {type(exc).__name__}: {exc}")
            continue
        all_results[year] = gdf
        print(f"  {len(gdf)} fields")
        if "evaluation_metrics" in gdf.attrs:
            label = "Evaluation"
            if engine_params and year == FINE_TUNE_YEAR:
                label += " (in-sample)"
            elif engine_params:  # the training polygons, a year the model never saw
                label += f" (training polygons; held-out year, fine-tuned on {FINE_TUNE_YEAR})"
            print_metrics(label, gdf.attrs["evaluation_metrics"])

    # --- Phase 3: summary -------------------------------------------------------
    print(f"\n{'=' * 60}\nTime series summary\n{'=' * 60}")
    print(f"  {'Year':<6} {'Fields':>7} {'Area (ha)':>12} {'F1':>6} {'IoU':>6}")
    for year, gdf in sorted(all_results.items()):
        area_ha = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0.0
        m = gdf.attrs.get("evaluation_metrics")
        f1 = f"{m['f1']:.3f}" if m else ""
        iou = f"{m['iou_mean']:.3f}" if m else ""
        print(f"  {year:<6} {len(gdf):>7} {area_ha:>12,.1f} {f1:>6} {iou:>6}")

    # --- Phase 4: maps ---------------------------------------------------------
    if not all_results:
        print("\nNo year succeeded; no maps written.")
        return
    from agribound.visualize import show_comparison

    latest_year = max(all_results)
    map_path = OUTPUT_DIR / f"map_predicted_vs_reference_{latest_year}.html"
    web_map = show_comparison(
        [all_results[latest_year], eval_gdf],
        labels=[f"Predicted ({latest_year})", f"{REFERENCE_LABEL}, evaluated"],
        basemap="Esri.WorldImagery",
        output_html=str(map_path),
    )
    show_in_notebook(web_map)
    print(f"\n  Predicted vs reference: {map_path}")

    selected = [y for y in (1985, 1995, 2005, 2015, 2025) if y in all_results]
    if len(selected) >= 2:
        map_path = OUTPUT_DIR / f"map_timeseries_comparison_{selected[0]}-{selected[-1]}.html"
        web_map = show_comparison(
            [all_results[y] for y in selected],
            labels=[str(y) for y in selected],
            basemap="Esri.WorldImagery",
            output_html=str(map_path),
        )
        show_in_notebook(web_map)
        print(f"  Time series: {map_path}")

    map_path = OUTPUT_DIR / f"map_latest_{latest_year}.html"
    web_map = agribound.show_boundaries(
        map_ready(all_results[latest_year]),
        basemap="Esri.WorldImagery",
        output_html=str(map_path),
    )
    show_in_notebook(web_map)
    print(f"  Latest year: {map_path}")


if __name__ == "__main__":
    main()
