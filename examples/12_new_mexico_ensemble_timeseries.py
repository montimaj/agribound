"""
12 — Eastern Lea County, NM: Multi-Model, Per-Source Ensembles (2022-2024)

Runs every engine that supports each source on one study area, merges the
results of each source with a pixel vote, refines each per-source ensemble
with SAM 2, and evaluates everything against the DRI/NMOSE field boundaries
(V18). Ensembles are built within a source (several models on the same
imagery), not across sources, and only for sources with at least two
finished runs: google-embedding and tessera-embedding have a single run each
(the embedding engine), so they get no ensemble and no SAM step.

Sources and engines (each engine only on the sources it supports):
    sentinel2, landsat, hls : FTW (FTW_PRUE_EFNET_B3/B5/B7), DINOv3, Prithvi,
                              Delineate-Anything (large_v2, small); GeoAI on
                              Sentinel-2 only
    naip                    : GeoAI, DINOv3, Delineate-Anything (large_v2, small)
    spot                    : DINOv3, Delineate-Anything (large_v2, small)
    google-embedding,
    tessera-embedding (v1)  : embedding clustering

Years (``SOURCE_YEARS``): 2024 for Sentinel-2, Landsat, HLS and the two
embeddings; NAIP 2022 and SPOT 6/7 2023, the latest years with imagery over
the box. Earth Engine NAIP covers the box in 2020 and 2022 and has nothing
later; SPOT 6/7 ends on 2023-11-15, and its 2023 scenes cover the whole box
(queried 2026-10-08). The NAIP and SPOT imagery is therefore 2 and 1 years
older than the other sources, and fields that changed in between count as
errors in their scores. TESSERA v1 has every tile of the box in 2017-2025
(queried 2026-09-27). Output names, log lines and the evaluation table carry
the year of each source.

Fine-tuning: Delineate-Anything, GeoAI, DINOv3 and Prithvi are fine-tuned on
the DRI/NMOSE polygons of the study area (GeoAI and DINOv3 cannot run
without a fine-tuned checkpoint). FTW and the embedding engine cannot be
fine-tuned and run label-free. On Landsat and HLS, FTW is out of distribution
(it is calibrated on Sentinel-2) and Delineate-Anything is outside its
0.25-10 m training range; agribound logs WARNINGs and records both in
``engine_meta``. Non-FTW engines on Sentinel-2, Landsat and HLS use an
October composite; FTW builds its own two seasonal windows. The LULC crop
filter is on in every run (NLCD is selected in the conterminous US) and
needs Earth Engine.

DINOv3's boundary band: DINOv3 learns from masks whose field-boundary class
is ``boundary_erosion`` pixels wide (4 px by default, chosen on 1.5 m
SPOT-Pan in example 23). A polygon less than 2 x band + 1 pixels across has
no interior pixel and cannot be learned as a field, which at 30 m is a large
share of these polygons. The script therefore sets DINOv3's band per source
with a rule on the training labels alone, chosen without training or scoring
a model (``agribound.engines.finetune.widest_boundary_erosion``): the
widest band of at most 4 px for which at most 10 % of the 393 training
polygons have no interior pixel on the grid DINOv3 reads the source's
composite on, the share above which the training chips log a WARNING.
DINOv3 reads Landsat and HLS (30 m) on a bilinear copy with 10 m pixels
(``engine_params["processing_gsd_m"]``; see
``agribound.engines.finetune.dinov3_processing_gsd``) and the other sources
on their own grids. On this box the rule keeps 4 px everywhere: on NAIP (no
polygon without an interior pixel), SPOT 6/7 (1 %) and Sentinel-2 (6 %),
computed 2026-10-08, and on Landsat and HLS on the 10 m grid (6 %),
computed 2026-10-10. On their own 30 m grid it would give 1 px (3 %; 2 px
would leave 14 % and 4 px 41 %). The script prints the shares. Prithvi
reads the 30 m grid and keeps its default 2 px band, so its Landsat and HLS
chips log that WARNING (14 %).

SAM 2 on 30 m sources: SAM prompts a polygon only if its bounding box,
padded by 15 %, is at least 64 pixels on each side (``sam_min_crop_px`` and
``sam_crop_padding`` defaults), i.e. about 49 pixels unpadded: ~1.5 km at
30 m, ~490 m at 10 m, ~295 m at 6 m and ~49 m at 1 m. Of the 393 DRI/NMOSE
polygons intersecting the box (median bounding-box width ~740 m), none pass
at 30 m, 208 at 10 m, 301 at 6 m and 390 at 1 m. On the Landsat and HLS
ensembles SAM therefore changes only polygons whose bounding box is about
1.5 km or more on each side (e.g. several fields merged into one); the
script prints how many polygons it refined.

Environments: FTW needs ftw-tools (core environment) and Prithvi needs
terratorch (GFM environment); they cannot be installed together. The script
skips engines whose package is missing. Outputs are separate files per
source, engine, model and year, so running the script once in each
environment adds the missing engines; the ensembles and the evaluation use
every output present. An output of an earlier version of this example at
the same path (the NAIP 2022 runs, fine-tuned on the older NMOSE WUCB
polygons) was made with another configuration, so that run stops with a
FileExistsError: delete those files or use a new OUTPUT_DIR.

Evaluation: against the DRI/NMOSE polygons whose representative point lies
in the box (389), the rule the pipeline uses to keep predictions
(``aoi_selection="representative_point"``), so a field crossing the box edge
is left out of both layers, without those smaller than 2,500 m2, the
predictions' minimum area (``min_area``; 5,000 m2 for NAIP and the
embeddings; areas in EPSG:6933, as agribound's area filter computes them):
one polygon is dropped and 388 are evaluated (the script prints the counts).
The fine-tuning reference file holds all 393 polygons of Lea County
(``FIPS_code`` 35025) that intersect the box, whatever their size. For the
fine-tuned engines (and ensembles that contain them) the evaluation is
in-sample: they were trained on these polygons. The reference may not
include every field in the box; predictions of fields it lacks count as
false positives ("FP-noref" = false positives that overlap no reference
polygon).

Estimated runtime: several hours (31 source-engine-model runs, 20 of them
fine-tuned, in the two environments together; GPU required in practice). Best
run on HPC/cloud. The whole example has not been run for 1.0 or 1.1. Its four
NAIP 2022 runs (gallery entry 12n) were run by a separate script with this
example's settings in the development run of agribound 1.1 (2026-10-09, Apple
M2 Max; the NAIP composite downloaded, the model weights already cached):
they took about 3.9 hours while the computer was awake (about 14,080 s by the
runs' own clocks, of the 30,839 s they spanned; the laptop slept, lid closed
on battery, for the rest): Delineate-Anything as released 13 minutes,
Delineate-Anything fine-tuned 80 minutes (10 epochs on 2,882 chips in 73
minutes), GeoAI 75 minutes (on the CPU) and DINOv3 65 minutes.

Prerequisites:
    Core env:  pip install "agribound[gee,delineate-anything,ftw,geoai,samgeo,tessera]"
    GFM env:   conda env create -f environment-gfm.yml   (Prithvi)
    agribound auth --project YOUR_GEE_PROJECT
    DRI/NMOSE field boundaries (V18) at "examples/NM_Field_Boundaries.zip" (local
        data, not distributed; read in place, no need to unzip)
    SPOT 6/7 is restricted to select Earth Engine users (the SPOT runs fail otherwise)
    Run from the repository root: python examples/12_new_mexico_ensemble_timeseries.py
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import agribound
from agribound.evaluate import evaluate
from agribound.provenance import read_provenance
from agribound.registry import engine_supports_source, list_sources

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", message=".*organizePolygons.*")
warnings.filterwarnings("ignore", message=".*STAC entry.*", category=RuntimeWarning)
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
OUTPUT_DIR = Path("outputs/lea_county_ensemble")
COUNTY_FIPS = "35025"  # Lea County in the V18 "FIPS_code" column
BBOX = (-103.25, 32.75, -103.05, 32.95)  # eastern Lea County (centre pivots)
#: Imagery year of each source: 2024, or the latest year with imagery over the box.
SOURCE_YEARS = {
    "sentinel2": 2024,
    "landsat": 2024,
    "hls": 2024,
    "naip": 2022,  # Earth Engine NAIP: 2020 and 2022 over the box, nothing later
    "spot": 2023,  # SPOT 6/7 ends on 2023-11-15
    "google-embedding": 2024,
    "tessera-embedding": 2024,
}
FINE_TUNE_EPOCHS = 10  # 20 or more for production runs
FINE_TUNE_ENGINES = {"delineate-anything", "geoai", "dinov3", "prithvi"}
BATCH_SIZE = 8
MIN_AREA = 2500  # m^2; minimum area of the predictions (NAIP, embeddings: 5,000) and reference
SAM_REFINE = True  # SAM 2 on each per-source ensemble (needs agribound[samgeo])
SAM_MODEL = "tiny"  # SAM 2 size alias: tiny, small, base_plus, large
VOTE_THRESHOLD = 0.3

SOURCE_ENGINE_MAP = {
    "sentinel2": ["ftw", "geoai", "dinov3", "prithvi", "delineate-anything"],
    "landsat": ["ftw", "dinov3", "prithvi", "delineate-anything"],
    "hls": ["ftw", "dinov3", "prithvi", "delineate-anything"],
    "naip": ["geoai", "dinov3", "delineate-anything"],
    "spot": ["dinov3", "delineate-anything"],
    "google-embedding": ["embedding"],
    "tessera-embedding": ["embedding"],
}
FTW_MODELS = ["FTW_PRUE_EFNET_B3", "FTW_PRUE_EFNET_B5", "FTW_PRUE_EFNET_B7"]
DA_MODELS = ["large_v2", "small"]  # Delineate-Anything v2 (YOLO11x) and v1 small (YOLO11n)

#: Python package each engine needs.
ENGINE_PACKAGES = {
    "delineate-anything": "ultralytics",
    "ftw": "ftw_tools",
    "geoai": "geoai",
    "dinov3": "geoai",
    "prithvi": "terratorch",
    "embedding": "sklearn",
}


def engine_available(engine):
    """True if the engine's Python package can be imported in this environment."""
    return importlib.util.find_spec(ENGINE_PACKAGES[engine]) is not None


def create_study_area(reference, county_fips):
    """Write the study-area GeoJSON and the county's reference polygons that intersect it."""
    import geopandas as gpd
    from shapely.geometry import box

    minx, miny, maxx, maxy = BBOX
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}
    feature["properties"] = {"name": f"Eastern Lea County (FIPS {county_fips})"}
    aoi_path = OUTPUT_DIR / "lea_county_study_area.geojson"
    aoi_path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))

    # A new name for V18: an older lea_county_reference.gpkg holds the WUCB polygons.
    ref_path = OUTPUT_DIR / "lea_county_reference_v18.gpkg"
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


def output_path_for(source, engine, model=None):
    """Output file of one source-engine(-model) run, named with the source's year."""
    suffix = f"_{model}" if model else ""
    return OUTPUT_DIR / f"fields_{source}_{engine}{suffix}_{SOURCE_YEARS[source]}.gpkg"


def load_finished(path):
    """Load an output whose provenance record reports success (e.g. from the other env)."""
    import geopandas as gpd

    record = read_provenance(path)
    if path.exists() and record is not None and record.get("status") == "success":
        return gpd.read_file(path)
    return None


def run_delineation(source, engine, study_area, gee_project, ref_path, model=None):
    """Run one source-engine(-model) delineation; return (GeoDataFrame, output path)."""
    output_path = output_path_for(source, engine, model)
    year = SOURCE_YEARS[source]
    engine_params = {}
    kwargs = dict(
        study_area=study_area,
        source=source,
        year=year,
        engine=engine,
        output_path=str(output_path),
        gee_project=gee_project,
        min_area=MIN_AREA,
        simplify=2.0,
    )
    if engine != "embedding":
        engine_params["batch_size"] = BATCH_SIZE
    if engine in FINE_TUNE_ENGINES:
        kwargs.update(
            reference_boundaries=ref_path, fine_tune=True, fine_tune_epochs=FINE_TUNE_EPOCHS
        )
    if source in ("sentinel2", "landsat", "hls"):
        kwargs.update(composite_method="median", cloud_cover_max=20)
        if engine != "ftw":
            kwargs["date_range"] = (f"{year}-10-01", f"{year}-10-31")
    elif source == "spot":
        kwargs.update(composite_method="median", cloud_cover_max=15)
    elif source == "naip":
        kwargs["min_area"] = 5000
    elif source == "tessera-embedding":
        kwargs.update(tessera_version="v1", device="cpu", min_area=5000)
    elif source == "google-embedding":
        kwargs.update(device="cpu", min_area=5000)
    if model and engine == "ftw":
        engine_params["model"] = model
    elif model and engine == "delineate-anything":
        engine_params["da_model"] = model
    kwargs["engine_params"] = engine_params
    if engine == "dinov3":
        engine_params["boundary_erosion"] = dinov3_boundary_band(kwargs)
    # An existing output with a matching provenance record is loaded, not recomputed.
    return agribound.delineate(**kwargs), output_path


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


def raster_of(output_path):
    """Composite path recorded in an output's provenance record (or None)."""
    record = read_provenance(output_path) or {}
    return (record.get("facts") or {}).get("raster_path"), record.get("config")


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Lea County: per-source multi-model ensembles.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument("--no-sam", action="store_true", help="Skip the SAM 2 refinement.")
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(REFERENCE_ZIP).exists():
        raise SystemExit(
            f"{REFERENCE_LABEL} not found: {REFERENCE_ZIP} (local data, not distributed)"
        )
    study_area, ref_gdf, ref_path = create_study_area(REFERENCE, COUNTY_FIPS)
    eval_ref, n_small = evaluation_reference(ref_gdf, study_area)
    print(
        f"Study area: {study_area}. {REFERENCE_LABEL}: {len(ref_gdf)} polygons intersect it "
        f"(fine-tuning reference); {len(eval_ref) + n_small} have their representative point "
        f"in it, {n_small} of them smaller than {MIN_AREA:,} m2 (dropped): {len(eval_ref)} "
        "evaluated"
    )
    print("Years: " + ", ".join(f"{source} {year}" for source, year in SOURCE_YEARS.items()))

    for engine in sorted({e for engines in SOURCE_ENGINE_MAP.values() for e in engines}):
        if not engine_available(engine):
            print(
                f"  {engine}: package {ENGINE_PACKAGES[engine]!r} is not installed; only "
                "finished outputs from another environment are used"
            )

    # ================================================================
    # Phase 1: one run per source, engine and model
    # ================================================================
    print(f"\n{'=' * 70}\nPhase 1: per-source, per-engine delineation\n{'=' * 70}")
    results = {}  # {source: {tag: (gdf, path)}}
    for source, engines in SOURCE_ENGINE_MAP.items():
        year = SOURCE_YEARS[source]
        for engine in engines:
            if not engine_supports_source(engine, source):
                continue
            models = FTW_MODELS if engine == "ftw" else DA_MODELS
            if engine not in ("ftw", "delineate-anything"):
                models = [None]
            for model in models:
                tag = f"{source}/{engine}" + (f"/{model}" if model else "")
                if not engine_available(engine):
                    path = output_path_for(source, engine, model)
                    gdf = load_finished(path)
                    if gdf is not None:
                        results.setdefault(source, {})[tag] = (gdf, path)
                        print(f"  {tag} {year}: loaded {len(gdf)} fields from {path}")
                    continue
                print(f"  {tag} {year}: starting", flush=True)
                try:
                    gdf, path = run_delineation(
                        source, engine, study_area, args.gee_project, ref_path, model=model
                    )
                except Exception as exc:
                    print(f"  {tag} {year}: FAILED ({type(exc).__name__}: {exc})")
                    continue
                results.setdefault(source, {})[tag] = (gdf, path)
                print(f"  {tag} {year}: {len(gdf)} fields")

    # ================================================================
    # Phase 2: per-source vote ensembles (+ SAM 2)
    # ================================================================
    print(f"\n{'=' * 70}\nPhase 2: per-source vote ensembles\n{'=' * 70}")
    from agribound.config import AgriboundConfig
    from agribound.engines.ensemble import EnsembleEngine
    from agribound.postprocess import filter_polygons
    from agribound.postprocess.simplify import simplify_polygons, smooth_polygons

    resolution = {name: info["resolution_m"] for name, info in list_sources().items()}
    ensembles = {}  # {label: gdf}
    use_sam = SAM_REFINE and not args.no_sam and importlib.util.find_spec("samgeo") is not None
    for source, members in results.items():
        if len(members) < 2:
            continue
        year = SOURCE_YEARS[source]
        member_gdfs = {tag: gdf for tag, (gdf, _) in members.items()}
        vote = EnsembleEngine._merge_vote(
            member_gdfs, VOTE_THRESHOLD, resolution=resolution[source]
        )
        min_votes = (vote.attrs.get("vote_stats") or {}).get("min_votes")
        vote = filter_polygons(vote, min_area_m2=MIN_AREA)
        print(
            f"  {source} {year}: {len(members)} members, min_votes={min_votes} -> "
            f"{len(vote)} polygons"
        )
        if len(vote):
            vote.to_file(OUTPUT_DIR / f"fields_{source}_ensemble-vote_{year}.gpkg", layer="fields")
        ensembles[f"{source} ensemble"] = vote
        if not use_sam or len(vote) == 0:
            continue
        # Refine on the raster the non-FTW members used (recorded in their provenance).
        non_ftw = [p for tag, (_, p) in members.items() if "/ftw/" not in tag]
        raster_path, member_config = raster_of(non_ftw[0]) if non_ftw else (None, None)
        if not raster_path or not member_config:
            print(f"    no raster recorded for {source}; SAM skipped")
            continue
        sam_config = AgriboundConfig.from_dict(member_config).merged(
            sam_backend="sam2", sam_model=SAM_MODEL
        )
        from agribound.engines.samgeo_engine import refine_boundaries

        try:
            refined = refine_boundaries(vote, raster_path, sam_config)
        except Exception as exc:
            print(f"    SAM 2 failed for {source}: {type(exc).__name__}: {exc}")
            continue
        stats = refined.attrs.get("sam_stats", {})
        refined = filter_polygons(refined, min_area_m2=MIN_AREA)
        refined = simplify_polygons(smooth_polygons(refined, iterations=3), tolerance=2.0)
        refined.to_file(
            OUTPUT_DIR / f"fields_{source}_ensemble-vote-sam2_{year}.gpkg", layer="fields"
        )
        ensembles[f"{source} ensemble + SAM 2"] = refined
        print(
            f"    SAM 2 ({stats.get('model')}): refined {stats.get('n_refined')} of "
            f"{stats.get('n_total')} (too small: {stats.get('n_skipped_small')}, "
            f"covering too little: {stats.get('n_low_coverage')})"
        )

    # ================================================================
    # Phase 3: evaluation against the reference polygons
    # ================================================================
    print(
        f"\n{'=' * 70}\nPhase 3: evaluation against the {REFERENCE_LABEL} "
        f"({len(eval_ref)} polygons)\n{'=' * 70}"
    )
    print(
        f"  {'Run':<46} {'Year':>4} {'Fields':>6} {'F1':>6} {'IoU':>6} {'P':>6} {'R':>6} "
        f"{'FP-noref':>8}"
    )

    def report(label, year, gdf, in_sample):
        m = evaluate(gdf, eval_ref)
        note = " (in-sample)" if in_sample else ""
        print(
            f"  {label + note:<46} {year:>4} {len(gdf):>6} {m['f1']:.3f}  {m['iou_mean']:.3f}  "
            f"{m['precision']:.3f}  {m['recall']:.3f}  {m['count_fp_unassigned']:>8}"
        )

    for source, members in results.items():
        for tag, (gdf, _) in members.items():
            report(tag, SOURCE_YEARS[source], gdf, tag.split("/")[1] in FINE_TUNE_ENGINES)
    for label, gdf in ensembles.items():
        source = label.split()[0]
        fine_tuned = any(tag.split("/")[1] in FINE_TUNE_ENGINES for tag in results[source])
        report(label, SOURCE_YEARS[source], gdf, fine_tuned)

    # ================================================================
    # Phase 4: map
    # ================================================================
    if not ensembles:
        print("\nNo ensemble was built; no map written.")
        return
    from agribound.visualize import show_comparison

    years = sorted({SOURCE_YEARS[label.split()[0]] for label in ensembles})
    span = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
    map_path = OUTPUT_DIR / f"map_ensemble_comparison_{span}.html"
    web_map = show_comparison(
        [*ensembles.values(), eval_ref],
        labels=[
            *(f"{label} ({SOURCE_YEARS[label.split()[0]]})" for label in ensembles),
            REFERENCE_LABEL,
        ],
        basemap="Esri.WorldImagery",
        output_html=str(map_path),
    )
    show_in_notebook(web_map)
    print(f"\n  Ensemble comparison map: {map_path}")


if __name__ == "__main__":
    main()
