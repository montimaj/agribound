"""
14 — DINOv3 with and without SAM 2 Refinement on Five Sources (Eastern Lea County, NM)

Fine-tunes DINOv3 (ViT-L/16 backbone with SAT-493M satellite pre-training,
DPT head, via geoai) on the DRI/NMOSE field boundaries (V18) of eastern Lea
County, runs it on five sources, and compares each run with and without the
pipeline's SAM 2 refinement stage (``sam_refine=True``). DINOv3 has no
published field-boundary weights, so it always needs fine-tuning with
reference boundaries.

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
~49 m at 1 m (NAIP). Of the 393 DRI/NMOSE polygons intersecting the box
(median bounding-box width ~740 m), none pass at 30 m, 208 at 10 m, 301 at
6 m and 390 at 1 m. Polygons that do not pass keep their geometry, so on
Landsat and HLS the SAM runs can differ from the runs without SAM only in
polygons of about 1.5 km or more on each side; the script prints how many
polygons SAM refined in each run.

Sources: Sentinel-2, Landsat and HLS (October composites), NAIP (1 m) and
SPOT 6/7 (6 m; restricted access, see below). Years (``SOURCE_YEARS``): 2024
for Sentinel-2, Landsat and HLS; NAIP 2022 and SPOT 6/7 2023, the latest
years with imagery over the box (Earth Engine NAIP covers it in 2020 and 2022
and has nothing later; SPOT 6/7 ends on 2023-11-15 and its 2023 scenes cover
the whole box; queried 2026-10-08). The NAIP and SPOT imagery is therefore
2 and 1 years older than the other sources, and fields that changed in
between count as errors in their scores. ``--years 2020,2022`` runs every
source in the years given instead (every source covers the box in 2020 and
2022; in 2021 NAIP has none and SPOT scenes cover about 27 % of it, queried
2026-09-27). Output names and the evaluation table carry the year.
Fine-tuning is full fine-tuning by default (``use_lora=False``; set
``engine_params["use_lora"] = True`` for LoRA).

Boundary band: DINOv3 learns from masks whose field-boundary class is
``boundary_erosion`` pixels wide (4 px by default, chosen on 1.5 m SPOT-Pan
in example 23). A polygon less than 2 x band + 1 pixels across has no
interior pixel and cannot be learned as a field, which at 30 m is a large
share of these polygons. The script therefore sets the band of each run with
a rule on the training labels alone, chosen without training or scoring a
model (``agribound.engines.finetune.widest_boundary_erosion``): the widest
band of at most 4 px for which at most 10 % of the 393 training polygons
have no interior pixel on the grid DINOv3 reads the run's composite on, the
share above which the training chips log a WARNING. DINOv3 reads Landsat
and HLS (30 m) on a bilinear copy with 10 m pixels
(``engine_params["processing_gsd_m"]``; see
``agribound.engines.finetune.dinov3_processing_gsd``) and the other sources
on their own grids. On this box the rule keeps 4 px everywhere: on NAIP (no
polygon without an interior pixel), SPOT 6/7 (1 %) and Sentinel-2 (6 %),
computed 2026-10-08, and on Landsat and HLS on the 10 m grid (6 %),
computed 2026-10-10. On their own 30 m grid it would give 1 px (3 %; 2 px
would leave 14 % and 4 px 41 %). The script prints the shares of every run.

Evaluation is in-sample: DINOv3 is fine-tuned on the DRI/NMOSE polygons of
Lea County (``FIPS_code`` 35025) that intersect the box (393, whatever their
size) and evaluated against those whose representative point lies in the box
(389), the rule the pipeline uses to keep predictions
(``aoi_selection="representative_point"``), without those smaller than
2,500 m2, the predictions' minimum area (``min_area``; 5,000 m2 for NAIP;
areas in EPSG:6933, as agribound's area filter computes them): one polygon
is dropped and 388 are evaluated (the script prints the counts). The
reference may not include every field in the box; predictions of fields it
lacks count as false positives ("FP-noref" = false positives overlapping no
reference polygon).

SPOT 6/7 is restricted to select Earth Engine users (internal DRI use); the
SPOT runs fail without access and the script continues.

An output of an earlier version of this example at the same path (the NAIP
2022 runs, fine-tuned on the older NMOSE WUCB polygons) was made with another
configuration, so that run stops with a FileExistsError: delete those files
or use a new OUTPUT_DIR.

Runtime on an Apple M2 Max (MPS; composites downloaded, the model weights
already cached), and as much again for each further year with ``--years``:
about 3.9 hours from scratch in the development run of agribound 1.1
(2026-10-09, with Landsat and HLS still read at 30 m), of which the NAIP
download took 26 minutes, the NAIP fine-tuning 2.1 hours (stopped early at
epoch 28), NAIP inference and SAM 2 28 minutes, SPOT 30 minutes and
Sentinel-2 17 minutes (each with its download, fine-tuning and both runs),
and Landsat and HLS 3.4 and 3.5 minutes. On the 10 m grid of agribound
1.1.0, Landsat and HLS took 19 minutes each (2026-10-11; the Landsat
composite came from the cache, the HLS composite was downloaded again), so a
run from scratch takes about 4.4 hours. GPU required in practice.

Prerequisites:
    pip install "agribound[gee,dinov3,samgeo]"
    agribound auth --project YOUR_GEE_PROJECT
    DRI/NMOSE field boundaries (V18) at "examples/NM_Field_Boundaries.zip" (local
        data, not distributed; read in place, no need to unzip)
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
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
REFERENCE_ZIP = "examples/NM_Field_Boundaries.zip"
REFERENCE = f"zip://{REFERENCE_ZIP}!Master_NM_Field_Boundaries_V18.shp"
REFERENCE_LABEL = "DRI/NMOSE field boundaries (V18)"
OUTPUT_DIR = Path("outputs/lea_county_dinov3_sam2")
COUNTY_FIPS = "35025"  # Lea County in the V18 "FIPS_code" column
BBOX = (-103.25, 32.75, -103.05, 32.95)  # eastern Lea County (centre pivots)
#: Imagery year of each source: 2024, or the latest year with imagery over the box.
SOURCE_YEARS = {
    "sentinel2": 2024,
    "landsat": 2024,
    "hls": 2024,
    "naip": 2022,  # Earth Engine NAIP: 2020 and 2022 over the box, nothing later
    "spot": 2023,  # SPOT 6/7 ends on 2023-11-15
}
FINE_TUNE_EPOCHS = 30  # upper limit; early stopping on val_miou (patience 10)
BATCH_SIZE = 8
MIN_AREA = 2500  # m^2; the predictions' minimum area (NAIP: 5,000), also the reference's
SAM_MODEL = "large"  # SAM 2 size alias (facebook/sam2-hiera-large)
SOURCES = list(SOURCE_YEARS)


def create_study_area(reference, county_fips):
    """Write the study-area GeoJSON and the county's reference polygons that intersect it."""
    import geopandas as gpd
    from shapely.geometry import box

    minx, miny, maxx, maxy = BBOX
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}
    feature["properties"] = {"name": f"Eastern Lea County (FIPS {county_fips})"}
    aoi_path = OUTPUT_DIR / "study_area.geojson"
    aoi_path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))

    # A new name for V18: an older reference.gpkg holds the WUCB polygons.
    ref_path = OUTPUT_DIR / "reference_v18.gpkg"
    # Written once: the file's modification time is part of the fine-tuning cache key.
    if not ref_path.exists():
        ref = gpd.read_file(reference)
        ref = ref[ref["FIPS_code"] == county_fips]
        ref = ref[ref.to_crs(epsg=4326).intersects(box(*BBOX))]  # also drops empty geometries
        ref.to_file(ref_path, driver="GPKG", layer="fields")
    return str(aoi_path), gpd.read_file(ref_path), str(ref_path)


def evaluation_reference(ref_gdf, study_area):
    """Reference polygons evaluated, and the number of smaller polygons dropped.

    The polygons whose representative point lies in the study area, the rule the
    pipeline applies to the predictions (``aoi_selection="representative_point"``),
    computed with the pipeline's own functions; of these, the polygons smaller than
    MIN_AREA are dropped with the predictions' area filter (areas in EPSG:6933).
    """
    from agribound.config import AgriboundConfig
    from agribound.pipeline import select_in_study_area, study_area_in_crs
    from agribound.postprocess import filter_polygons

    aoi = study_area_in_crs(AgriboundConfig(study_area=study_area), ref_gdf.crs)
    in_area = select_in_study_area(ref_gdf, aoi, "representative_point")[0]
    kept = filter_polygons(in_area, min_area_m2=MIN_AREA)
    return kept, len(in_area) - len(kept)


def dinov3_kwargs(source, year, study_area, gee_project, ref_path):
    """Keyword arguments of agribound.delineate() shared by the two runs."""
    kwargs = dict(
        study_area=study_area,
        source=source,
        year=year,
        engine="dinov3",
        gee_project=gee_project,
        min_area=MIN_AREA,
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


def dinov3_boundary_band(kwargs, output_path=None):
    """DINOv3's ``boundary_erosion`` for one run, by a rule on the training labels alone.

    The widest band of at most 4 px (DINOv3's default) for which at most 10 % of the
    training polygons (``reference_boundaries``) have no interior pixel on the grid
    DINOv3 reads the run's composite on: the composite's own grid, or for Landsat and
    HLS (30 m) the 10 m grid of the bilinear copy that DINOv3 reads instead
    (``dinov3_processing_gsd``). The composite is built here (cached; the run reuses
    it). The rule depends on the labels and the pixel size only, never on a score.
    """
    from agribound.config import AgriboundConfig
    from agribound.engines.finetune import dinov3_processing_gsd, widest_boundary_erosion

    aliases = {"min_area": "min_field_area_m2", "simplify": "simplify_tolerance"}
    values = {aliases.get(key, key): value for key, value in kwargs.items()}
    if output_path is not None:
        values["output_path"] = str(output_path)
    config = AgriboundConfig(**values)
    raster = agribound.build_composite(config)
    grid = dinov3_processing_gsd(raster, config.engine_params)
    band, shares = widest_boundary_erosion(config.reference_boundaries, raster, grid_gsd_m=grid)
    without = ", ".join(f"{px} px {share:.0%}" for px, share in shares.items())
    on_grid = f" on DINOv3's {grid:g} m grid" if grid else ""
    print(
        f"  DINOv3 boundary band {band} px{on_grid} (training polygons without an interior: "
        f"{without})"
    )
    return band


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
    parser.add_argument(
        "--years",
        default=None,
        help='Years for every source, e.g. "2022" or "2020,2022" (default: SOURCE_YEARS, '
        "2024 or the latest year with imagery over the box for each source).",
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
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(REFERENCE_ZIP).exists():
        raise SystemExit(
            f"{REFERENCE_LABEL} not found: {REFERENCE_ZIP} (local data, not distributed)"
        )
    study_area, ref_gdf, ref_path = create_study_area(REFERENCE, COUNTY_FIPS)
    years = parse_years(args.years) if args.years else None
    runs = [(source, year) for source in SOURCES for year in (years or [SOURCE_YEARS[source]])]
    eval_ref, n_small = evaluation_reference(ref_gdf, study_area)
    print(
        f"Study area: {study_area}. {REFERENCE_LABEL}: {len(ref_gdf)} polygons intersect it "
        f"(fine-tuning reference); {len(eval_ref) + n_small} have their representative point "
        f"in it, {n_small} of them smaller than {MIN_AREA:,} m2 (dropped): {len(eval_ref)} "
        "evaluated"
    )
    print("Runs: " + ", ".join(f"{source} {year}" for source, year in runs))

    results = {}  # {(year, source, variant): gdf}
    for source, year in runs:
        base = dinov3_kwargs(source, year, study_area, args.gee_project, ref_path)
        print(f"\n{year} {source}: composite and boundary band", flush=True)
        try:
            band = dinov3_boundary_band(base, OUTPUT_DIR / f"fields_{source}_dinov3_{year}.gpkg")
        except Exception as exc:
            print(f"  FAILED ({type(exc).__name__}: {exc})")
            continue
        base["engine_params"] = {**base["engine_params"], "boundary_erosion": band}
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
        f"\n{'=' * 70}\nIn-sample evaluation against the {REFERENCE_LABEL} "
        f"({len(eval_ref)} polygons)\n{'=' * 70}"
    )
    print(f"  {'Year':<5} {'Source':<10} {'Run':<12} {'Fields':>6} {'F1':>6} {'IoU':>6} {'P':>6}")
    for (year, source, variant), gdf in sorted(results.items()):
        m = evaluate(gdf, eval_ref)
        print(
            f"  {year:<5} {source:<10} {variant:<12} {len(gdf):>6} {m['f1']:.3f}  "
            f"{m['iou_mean']:.3f}  {m['precision']:.3f}  (R={m['recall']:.3f}, "
            f"FP-noref={m['count_fp_unassigned']})"
        )

    # Map: the SAM 2 run of the latest year of each source.
    latest = {}  # {source: (year, gdf)}
    for (year, source, variant), gdf in sorted(results.items()):
        if variant == "dinov3-sam2":
            latest[source] = (year, gdf)
    if latest:
        from agribound.visualize import show_comparison

        shown = sorted({year for year, _ in latest.values()})
        span = str(shown[0]) if len(shown) == 1 else f"{shown[0]}-{shown[-1]}"
        map_path = OUTPUT_DIR / f"map_source_comparison_{span}.html"
        web_map = show_comparison(
            [*(gdf for _, gdf in latest.values()), eval_ref],
            labels=[
                *(f"{source} {year} dinov3-sam2" for source, (year, _) in latest.items()),
                REFERENCE_LABEL,
            ],
            basemap="Esri.WorldImagery",
            output_html=str(map_path),
        )
        show_in_notebook(web_map)
        print(f"\n  Source comparison map: {map_path}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
