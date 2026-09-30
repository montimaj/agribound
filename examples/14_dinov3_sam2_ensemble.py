"""
14 — DINOv3 with and without SAM 2 Refinement on Five Sources (Eastern Lea County, NM)

Fine-tunes DINOv3 (ViT-L/16 backbone with SAT-493M satellite pre-training,
DPT head, via geoai) on the NMOSE polygons of eastern Lea County, runs it on
five sources, and compares each run with and without the pipeline's SAM 2
refinement stage (``sam_refine=True``). DINOv3 has no published
field-boundary weights, so it always needs fine-tuning with reference
boundaries.

Per source and year the script makes two pipeline runs with separate
outputs:
    1. ``fields_<source>_dinov3_<year>.gpkg``       (no SAM)
    2. ``fields_<source>_dinov3-sam2_<year>.gpkg``  (``sam_refine=True``)
Both use the same fine-tuned checkpoint: the fine-tuning cache key does not
depend on the SAM settings, so the second run does not retrain. In the
pipeline, SAM refinement runs right after delineation, before
post-processing and the LULC crop filter (on; NLCD is selected in the
conterminous US; needs Earth Engine).

SAM on 30 m sources: SAM prompts a polygon only if its bounding box, padded
by 15 %, is at least 64 pixels on each side (``sam_min_crop_px`` and
``sam_crop_padding`` defaults), i.e. about 49 pixels unpadded: ~1.5 km at
30 m (Landsat, HLS), ~490 m at 10 m (Sentinel-2), ~295 m at 6 m (SPOT) and
~49 m at 1 m (NAIP). Of the 230 NMOSE polygons intersecting the box (median
bounding-box width ~780 m), none pass at 30 m, 151 at 10 m, 186 at 6 m and
226 at 1 m. Polygons that do not pass keep their geometry, so on Landsat and
HLS the SAM runs can differ from the runs without SAM only in polygons of
about 1.5 km or more on each side; the script prints how many polygons SAM
refined in each run.

Sources: Sentinel-2, Landsat and HLS (October composites), NAIP (1 m) and
SPOT 6/7 (6 m; restricted access, see below). Years: 2022 by default
(``--years 2020,2022``); every source has imagery over this box in 2020 and
2022, while in 2021 NAIP has none and SPOT scenes cover about 27 % of it
(queried 2026-09-27). Fine-tuning is full fine-tuning by default
(``use_lora=False``; set ``engine_params["use_lora"] = True`` for LoRA).

Evaluation is in-sample: DINOv3 is fine-tuned on the NMOSE polygons that
intersect the box (230) and evaluated against the 227 of them whose
representative point lies in the box, the rule the pipeline uses to keep
predictions (``aoi_selection="representative_point"``). NMOSE may not
include every field in the box; predictions of fields it lacks count as
false positives ("FP-noref" = false positives overlapping no reference
polygon).

SPOT 6/7 is restricted to select Earth Engine users (internal DRI use); the
SPOT runs fail without access and the script continues.

Estimated runtime (not measured for 1.0): ~1-2 hours per year (five fine-tuning
runs, GPU required in practice).

Prerequisites:
    pip install "agribound[gee,dinov3,samgeo]"
    agribound auth --project YOUR_GEE_PROJECT
    NMOSE shapefile at "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
    Run from the repository root: python examples/14_dinov3_sam2_ensemble.py
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import agribound
from agribound.evaluate import evaluate

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", message=".*organizePolygons.*")
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

# --- Configuration ---
NMOSE_SHAPEFILE = "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
OUTPUT_DIR = Path("outputs/lea_county_dinov3_sam2")
COUNTY_CODE = "25"  # Lea County
BBOX = (-103.25, 32.75, -103.05, 32.95)  # eastern Lea County (centre pivots)
DEFAULT_YEARS = "2022"
FINE_TUNE_EPOCHS = 30  # upper limit; early stopping on val_loss (patience 10)
BATCH_SIZE = 8
SAM_MODEL = "large"  # SAM 2 size alias (facebook/sam2-hiera-large)
SOURCES = ["sentinel2", "landsat", "hls", "naip", "spot"]


def create_study_area(shapefile_path, county_code):
    """Write the study-area GeoJSON and the NMOSE polygons that intersect it."""
    import geopandas as gpd
    from shapely.geometry import box

    minx, miny, maxx, maxy = BBOX
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}
    feature["properties"] = {"name": f"Eastern Lea County (County {county_code})"}
    aoi_path = OUTPUT_DIR / "study_area.geojson"
    aoi_path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))

    ref_path = OUTPUT_DIR / "reference.gpkg"
    # Written once: the file's modification time is part of the fine-tuning cache key.
    if not ref_path.exists():
        ref = gpd.read_file(shapefile_path)
        ref = ref[ref["County"] == county_code]
        ref = ref[ref.to_crs(epsg=4326).intersects(box(*BBOX))]
        ref.to_file(ref_path, driver="GPKG", layer="fields")
    return str(aoi_path), gpd.read_file(ref_path), str(ref_path)


def evaluation_reference(ref_gdf, study_area):
    """Reference polygons whose representative point lies in the study area.

    This is the rule the pipeline applies to the predictions
    (``aoi_selection="representative_point"``), computed with the pipeline's
    own functions.
    """
    from agribound.config import AgriboundConfig
    from agribound.pipeline import select_in_study_area, study_area_in_crs

    aoi = study_area_in_crs(AgriboundConfig(study_area=study_area), ref_gdf.crs)
    return select_in_study_area(ref_gdf, aoi, "representative_point")[0]


def dinov3_kwargs(source, year, study_area, gee_project, ref_path):
    """Keyword arguments of agribound.delineate() shared by the two runs."""
    kwargs = dict(
        study_area=study_area,
        source=source,
        year=year,
        engine="dinov3",
        gee_project=gee_project,
        min_area=2500,
        simplify=2.0,
        reference_boundaries=ref_path,
        fine_tune=True,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        engine_params={"batch_size": BATCH_SIZE},
    )
    if source in ("sentinel2", "landsat", "hls"):
        kwargs.update(
            composite_method="median",
            cloud_cover_max=20,
            date_range=(f"{year}-10-01", f"{year}-10-31"),
        )
    elif source == "spot":
        kwargs.update(composite_method="median", cloud_cover_max=15)
    elif source == "naip":
        kwargs["min_area"] = 5000
    return kwargs


def parse_years(text):
    """Parse "2022", "2020,2022" or "2020-2022" into a list of years."""
    years = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            first, last = (int(v) for v in part.split("-"))
            years.extend(range(first, last + 1))
        elif part:
            years.append(int(part))
    return sorted(set(years))


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="DINOv3 with/without SAM 2 on five sources.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument("--years", default=DEFAULT_YEARS, help='e.g. "2022" or "2020,2022".')
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
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(NMOSE_SHAPEFILE).exists():
        raise SystemExit(f"NMOSE reference not found: {NMOSE_SHAPEFILE}")
    study_area, ref_gdf, ref_path = create_study_area(NMOSE_SHAPEFILE, COUNTY_CODE)
    years = parse_years(args.years)
    eval_ref = evaluation_reference(ref_gdf, study_area)
    print(
        f"Study area: {study_area} ({len(ref_gdf)} NMOSE polygons intersect it (fine-tuning "
        f"reference); {len(eval_ref)} have their representative point in it (evaluation))"
    )
    print(f"Sources: {', '.join(SOURCES)}; years: {years}")

    results = {}  # {(year, source, variant): gdf}
    for year in years:
        for source in SOURCES:
            base = dinov3_kwargs(source, year, study_area, args.gee_project, ref_path)
            variants = {
                "dinov3": {},
                "dinov3-sam2": {"sam_refine": True, "sam_backend": "sam2", "sam_model": SAM_MODEL},
            }
            for variant, extra in variants.items():
                output_path = OUTPUT_DIR / f"fields_{source}_{variant}_{year}.gpkg"
                print(f"\n{year} {source} {variant} -> {output_path}", flush=True)
                try:
                    gdf = agribound.delineate(**base, **extra, output_path=str(output_path))
                except Exception as exc:
                    print(f"  FAILED ({type(exc).__name__}: {exc})")
                    break  # the SAM run needs the same composite and checkpoint
                results[(year, source, variant)] = gdf
                stats = gdf.attrs.get("sam_stats")
                if stats:
                    print(
                        f"  SAM 2: refined {stats['n_refined']} of {stats['n_total']} "
                        f"(too small: {stats['n_skipped_small']}, "
                        f"covering too little: {stats.get('n_low_coverage', 0)})"
                    )
                print(f"  {len(gdf)} fields")

    print(
        f"\n{'=' * 70}\nIn-sample evaluation against NMOSE ({len(eval_ref)} polygons)\n{'=' * 70}"
    )
    print(f"  {'Year':<5} {'Source':<10} {'Run':<12} {'Fields':>6} {'F1':>6} {'IoU':>6} {'P':>6}")
    for (year, source, variant), gdf in sorted(results.items()):
        m = evaluate(gdf, eval_ref)
        print(
            f"  {year:<5} {source:<10} {variant:<12} {len(gdf):>6} {m['f1']:.3f}  "
            f"{m['iou_mean']:.3f}  {m['precision']:.3f}  (R={m['recall']:.3f}, "
            f"FP-noref={m['count_fp_unassigned']})"
        )

    if results:
        from agribound.visualize import show_comparison

        latest = max(year for year, _, _ in results)
        layers = {
            f"{source} {variant}": gdf
            for (year, source, variant), gdf in sorted(results.items())
            if year == latest and variant == "dinov3-sam2"
        }
        web_map = show_comparison(
            [*layers.values(), eval_ref],
            labels=[*layers.keys(), "NMOSE reference"],
            basemap="Esri.WorldImagery",
            output_html=str(OUTPUT_DIR / f"map_source_comparison_{latest}.html"),
        )
        show_in_notebook(web_map)
        print(f"\n  Source comparison map: {OUTPUT_DIR / f'map_source_comparison_{latest}.html'}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
