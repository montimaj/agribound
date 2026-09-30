"""
12 — Eastern Lea County, NM: Multi-Model, Per-Source Ensembles (2022)

Runs every engine that supports each source on one study area and year,
merges the results of each source with a pixel vote, refines each per-source
ensemble with SAM 2, and evaluates everything against the NMOSE polygons.
Ensembles are built within a source (several models on the same imagery),
not across sources, and only for sources with at least two finished runs:
google-embedding and tessera-embedding have a single run each (the
embedding engine), so they get no ensemble and no SAM step.

Sources and engines (each engine only on the sources it supports):
    sentinel2, landsat, hls : FTW (FTW_PRUE_EFNET_B3/B5/B7), DINOv3, Prithvi,
                              Delineate-Anything (large_v2, small); GeoAI on
                              Sentinel-2 only
    naip                    : GeoAI, DINOv3, Delineate-Anything (large_v2, small)
    spot                    : DINOv3, Delineate-Anything (large_v2, small)
    google-embedding,
    tessera-embedding (v1)  : embedding clustering

Fine-tuning: Delineate-Anything, GeoAI, DINOv3 and Prithvi are fine-tuned on
the NMOSE polygons of the study area (GeoAI and DINOv3 cannot run without a
fine-tuned checkpoint). FTW and the embedding engine cannot be fine-tuned and
run label-free. On Landsat and HLS, FTW is out of distribution (it is
calibrated on Sentinel-2) and Delineate-Anything is outside its 0.25-10 m
training range; agribound logs WARNINGs and records both in ``engine_meta``.
Non-FTW engines on Sentinel-2, Landsat and HLS use an October composite; FTW
builds its own two seasonal windows. The LULC crop filter is on in every
run (NLCD is selected in the conterminous US) and needs Earth Engine.

SAM 2 on 30 m sources: SAM prompts a polygon only if its bounding box,
padded by 15 %, is at least 64 pixels on each side (``sam_min_crop_px`` and
``sam_crop_padding`` defaults), i.e. about 49 pixels unpadded: ~1.5 km at
30 m, ~490 m at 10 m, ~295 m at 6 m and ~49 m at 1 m. Of the 230 NMOSE
polygons intersecting the box (median bounding-box width ~780 m), none pass
at 30 m, 151 at 10 m, 186 at 6 m and 226 at 1 m. On the Landsat and HLS
ensembles SAM therefore changes only polygons whose bounding box is about
1.5 km or more on each side (e.g. several fields merged into one); the
script prints how many polygons it refined.

Why 2022: it is the latest year with every source over this box. Earth Engine
NAIP ends in 2023 and eastern Lea County has NAIP for 2020 and 2022 but not
2021 or 2023-2025; SPOT 6/7 ends on 2023-11-15; TESSERA v1 has every tile of
the box in 2017-2025 (queried 2026-09-27).

Environments: FTW needs ftw-tools (core environment) and Prithvi needs
terratorch (GFM environment); they cannot be installed together. The script
skips engines whose package is missing. Outputs are separate files per
source, engine, model and year, so running the script once in each
environment adds the missing engines; the ensembles and the evaluation use
every output present.

Evaluation: against the 227 NMOSE polygons whose representative point lies
in the box, the rule the pipeline uses to keep predictions
(``aoi_selection="representative_point"``), so a field crossing the box edge
is left out of both layers. The fine-tuning reference file holds all 230
polygons that intersect the box. For the fine-tuned engines (and ensembles
that contain them) the evaluation is in-sample: they were trained on these
polygons. NMOSE may not include every field in the box; predictions of
fields it lacks count as false positives ("FP-noref" = false positives that
overlap no reference polygon).

Estimated runtime (not measured for 1.0): several hours (31 source-engine-model
runs, 20 of them fine-tuned, in the two environments together; GPU required in
practice). Best run on HPC/cloud.

Prerequisites:
    Core env:  pip install "agribound[gee,delineate-anything,ftw,geoai,samgeo,tessera]"
    GFM env:   conda env create -f environment-gfm.yml   (Prithvi)
    agribound auth --project YOUR_GEE_PROJECT
    NMOSE shapefile at "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
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
OUTPUT_DIR = Path("outputs/lea_county_ensemble")
COUNTY_CODE = "25"  # Lea County in the NMOSE "County" column
BBOX = (-103.25, 32.75, -103.05, 32.95)  # eastern Lea County (centre pivots)
YEAR = 2022
FINE_TUNE_EPOCHS = 10  # 20 or more for production runs
FINE_TUNE_ENGINES = {"delineate-anything", "geoai", "dinov3", "prithvi"}
BATCH_SIZE = 8
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


def create_study_area(shapefile_path, county_code):
    """Write the study-area GeoJSON and the NMOSE polygons that intersect it."""
    import geopandas as gpd
    from shapely.geometry import box

    minx, miny, maxx, maxy = BBOX
    ring = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
    feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}
    feature["properties"] = {"name": f"Eastern Lea County (County {county_code})"}
    aoi_path = OUTPUT_DIR / "lea_county_study_area.geojson"
    aoi_path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))

    ref_path = OUTPUT_DIR / "lea_county_reference.gpkg"
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


def output_path_for(source, engine, model=None):
    """Output file of one source-engine(-model) run."""
    suffix = f"_{model}" if model else ""
    return OUTPUT_DIR / f"fields_{source}_{engine}{suffix}_{YEAR}.gpkg"


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
    engine_params = {}
    kwargs = dict(
        study_area=study_area,
        source=source,
        year=YEAR,
        engine=engine,
        output_path=str(output_path),
        gee_project=gee_project,
        min_area=2500,
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
            kwargs["date_range"] = (f"{YEAR}-10-01", f"{YEAR}-10-31")
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
    # An existing output with a matching provenance record is loaded, not recomputed.
    return agribound.delineate(**kwargs), output_path


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
    if not Path(NMOSE_SHAPEFILE).exists():
        raise SystemExit(f"NMOSE reference not found: {NMOSE_SHAPEFILE}")
    study_area, ref_gdf, ref_path = create_study_area(NMOSE_SHAPEFILE, COUNTY_CODE)
    eval_ref = evaluation_reference(ref_gdf, study_area)
    print(
        f"Study area: {study_area} ({len(ref_gdf)} NMOSE polygons intersect it (fine-tuning "
        f"reference); {len(eval_ref)} have their representative point in it (evaluation))"
    )

    for engine in sorted({e for engines in SOURCE_ENGINE_MAP.values() for e in engines}):
        if not engine_available(engine):
            print(
                f"  {engine}: package {ENGINE_PACKAGES[engine]!r} is not installed; only "
                "finished outputs from another environment are used"
            )

    # ================================================================
    # Phase 1: one run per source, engine and model
    # ================================================================
    print(f"\n{'=' * 70}\nPhase 1: per-source, per-engine delineation ({YEAR})\n{'=' * 70}")
    results = {}  # {source: {tag: (gdf, path)}}
    for source, engines in SOURCE_ENGINE_MAP.items():
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
                        print(f"  {tag}: loaded {len(gdf)} fields from {path}")
                    continue
                print(f"  {tag}: starting", flush=True)
                try:
                    gdf, path = run_delineation(
                        source, engine, study_area, args.gee_project, ref_path, model=model
                    )
                except Exception as exc:
                    print(f"  {tag}: FAILED ({type(exc).__name__}: {exc})")
                    continue
                results.setdefault(source, {})[tag] = (gdf, path)
                print(f"  {tag}: {len(gdf)} fields")

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
        member_gdfs = {tag: gdf for tag, (gdf, _) in members.items()}
        vote = EnsembleEngine._merge_vote(
            member_gdfs, VOTE_THRESHOLD, resolution=resolution[source]
        )
        min_votes = (vote.attrs.get("vote_stats") or {}).get("min_votes")
        vote = filter_polygons(vote, min_area_m2=2500)
        print(f"  {source}: {len(members)} members, min_votes={min_votes} -> {len(vote)} polygons")
        if len(vote):
            vote.to_file(OUTPUT_DIR / f"fields_{source}_ensemble-vote_{YEAR}.gpkg", layer="fields")
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
        refined = filter_polygons(refined, min_area_m2=2500)
        refined = simplify_polygons(smooth_polygons(refined, iterations=3), tolerance=2.0)
        refined.to_file(
            OUTPUT_DIR / f"fields_{source}_ensemble-vote-sam2_{YEAR}.gpkg", layer="fields"
        )
        ensembles[f"{source} ensemble + SAM 2"] = refined
        print(
            f"    SAM 2 ({stats.get('model')}): refined {stats.get('n_refined')} of "
            f"{stats.get('n_total')} (too small: {stats.get('n_skipped_small')}, "
            f"covering too little: {stats.get('n_low_coverage')})"
        )

    # ================================================================
    # Phase 3: evaluation against the NMOSE polygons
    # ================================================================
    print(f"\n{'=' * 70}\nPhase 3: evaluation against NMOSE ({len(eval_ref)} polygons)\n{'=' * 70}")
    print(f"  {'Run':<46} {'Fields':>6} {'F1':>6} {'IoU':>6} {'P':>6} {'R':>6} {'FP-noref':>8}")

    def report(label, gdf, in_sample):
        m = evaluate(gdf, eval_ref)
        note = " (in-sample)" if in_sample else ""
        print(
            f"  {label + note:<46} {len(gdf):>6} {m['f1']:.3f}  {m['iou_mean']:.3f}  "
            f"{m['precision']:.3f}  {m['recall']:.3f}  {m['count_fp_unassigned']:>8}"
        )

    for members in results.values():
        for tag, (gdf, _) in members.items():
            report(tag, gdf, tag.split("/")[1] in FINE_TUNE_ENGINES)
    for label, gdf in ensembles.items():
        fine_tuned = any(
            tag.split("/")[1] in FINE_TUNE_ENGINES for tag in results[label.split()[0]]
        )
        report(label, gdf, fine_tuned)

    # ================================================================
    # Phase 4: map
    # ================================================================
    if not ensembles:
        print("\nNo ensemble was built; no map written.")
        return
    from agribound.visualize import show_comparison

    map_path = OUTPUT_DIR / f"map_ensemble_comparison_{YEAR}.html"
    web_map = show_comparison(
        [*ensembles.values(), eval_ref],
        labels=[*ensembles.keys(), "NMOSE reference"],
        basemap="Esri.WorldImagery",
        output_html=str(map_path),
    )
    show_in_notebook(web_map)
    print(f"\n  Ensemble comparison map: {map_path}")


if __name__ == "__main__":
    main()
