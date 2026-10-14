"""
20 — Stratified Evaluation against a Reference Registry (DRI/NMOSE V18, San Juan Basin)

Evaluates predicted field boundaries against the DRI/NMOSE field boundaries
(V18) with the full ``agribound.evaluate`` toolkit:

    - one-to-one IoU matching (IoU >= 0.5): precision, recall, F1, mean IoU
      of matches, area-weighted recall/precision;
    - per stratum (V18's ``Sub_Basin`` column, the name of the HUC-8
      subbasin) and per field-size class (``size_bins="auto"``: 1-2-5 series
      of hectares);
    - boundary metrics: Hausdorff and mean boundary distance of matched
      pairs, boundary precision/recall/F1 and coverage within a 10 m
      tolerance (one Sentinel-2 pixel);
    - 95 % percentile bootstrap intervals (200 resamples of reference fields,
      stratified by sub-basin). They treat fields as independent, so they are
      too narrow when errors are spatially clustered;
    - the per-field table (``evaluate_frame``) and the pixels per field at the
      Sentinel-2 GSD (``pixels_per_field``);
    - a resolution comparison: the same engine on Landsat (30 m), Sentinel-2
      (10 m), SPOT 6/7 (6 m) and NAIP (1 m) of one year, each evaluated with
      the overall object and boundary metrics only (no strata, size classes
      or bootstrap; ``--no-resolution`` or ``--predicted`` skips it);
    - a crop-filter comparison: the pre-trained FTW and Delineate-Anything
      engines on the Sentinel-2 year, each with and without the LULC crop
      filter (``lulc_filter``), evaluated with the same overall metrics
      (``--no-lulc-comparison`` or ``--predicted`` skips it).

Predictions: by default the script delineates the study area label-free
(Sentinel-2 2024; ``--year`` changes it; Delineate-Anything ``large_v2``;
LULC crop filter on, NLCD). ``--predicted PATH`` evaluates any existing
layer instead (no Earth Engine needed then). Both layers are restricted to
the study area by the same rule: a polygon is kept when its representative
point lies inside the box (the pipeline applies this rule,
``aoi_selection="representative_point"``, to its own output by default).
Reference fields smaller than 2,500 m2, the predictions' minimum area
(``min_area``, agribound's default), are dropped, so that both layers have
the same minimum area (areas in EPSG:6933, as agribound's area filter
computes them); the script prints how many. Output, metrics and table file
names carry the year (or, with ``--predicted``, the name of the predicted
file).

Resolution comparison: one year for all four sources (``--resolution-year``,
default 2022), so that its rows differ in resolution only, not in the fields,
crops and season they see. It is an exception to the year rule of the New
Mexico examples (2024, or the latest year of a source without 2024 imagery),
and its imagery is 2 years older than the main run's. NAIP is flown
every two years here (2016, 2018, 2020 and 2022 cover the box) and Earth
Engine has no NAIP after 2022, so 2022 is the year closest to 2024 that
Landsat, Sentinel-2, SPOT 6/7 and NAIP all cover (queried 2026-10-08). In
2022 SPOT 6/7 has a single scene over the box (2022-11-07), with valid data
over 76.7 % of it: 158 of the 1,355 evaluated reference fields (190 of all
1,705 V18 polygons in the box) lie entirely in the western strip without data
(west of about 108.19 W) and count as missed in the SPOT row, and 5 more lie
partly in it (2023 has no SPOT scene over the box; the footprints of the two
2021 scenes cover all of it, their valid data were not checked).
The 2022 Landsat composite uses Landsat 7, 8 and 9. Delineate-Anything was
trained on 0.25-10 m imagery, so the Landsat run is outside its range
(agribound logs a WARNING and records it in ``engine_meta``). SPOT 6/7 is
restricted to select Earth Engine users; without access that run fails and
the script continues.

Crop-filter comparison: FTW (``FTW_PRUE_EFNET_B5``, two season windows) and
Delineate-Anything (``large_v2``), both used as released, on Sentinel-2 of
``--year``. With the filter on, a polygon is kept when at least 30 % of its
area is Annual NLCD cultivated crops or pasture/hay (classes 82 and 81) in
that year; with it off, every polygon is kept. Delineate-Anything with the
filter is the main run above, so it is loaded rather than recomputed.

Study area: 108.25-108.00 W, 36.6-36.8 N (San Juan County); 1,705 V18
polygons have their representative point inside it. 350 of them are smaller
than 2,500 m2 and are dropped (136 under 1,250 m2, whose IoU with any
prediction of at least 2,500 m2 is at most 0.5; 12 under 0.01 ha), so 1,355
are evaluated, in three sub-basins (Upper San Juan 689, Animas 559, Middle
San Juan 107), and the size-class table starts at 0.2 ha. V18 records no
imagery date per field, so fields that changed between the reference's
(unrecorded) date and the imagery count as errors. Its ``CDL_Crop_T``
column holds a Cropland Data Layer class per field (the CDL year is not
recorded either); most of the 1,355 have a class other than a crop
(Grassland/Pasture 573, the Developed classes 296, Shrubland 170 and Woody
Wetlands 118), and the script prints the most common classes. The
reference may not include every field in the box; ``count_fp_unassigned``
counts predictions that overlap no reference polygon.

The DRI/NMOSE field boundaries (V18) must be at
"examples/NM_Field_Boundaries.zip" (local data, not distributed; read in
place, no need to unzip), or pass ``--reference`` (a vector file, or a file
in a zip archive as ``zip://ARCHIVE!MEMBER``, with a ``Sub_Basin`` column);
otherwise the script exits with a message.

Runtime: 70 minutes (4,194 s) for the defaults in the development run of
agribound 1.1 (2026-10-09, Apple M2 Max with MPS; every composite and FTW's
checkpoint downloaded, Delineate-Anything's weights already cached). The main
run took 21 minutes: 20.7 minutes to download the 2301 x 2287 px Sentinel-2
2024 composite from Earth Engine, 18 s of Delineate-Anything inference and
5 s of LULC filtering. The resolution comparison took 40 minutes (Sentinel-2
2022 23 minutes, NAIP 15 minutes, SPOT 1.7 minutes, Landsat 39 s); its NAIP
mosaic is 23,006 x 22,865 px at 1 m (2.1 GB uncompressed, 1.4 GB on disk).
The crop-filter comparison took 8.6 minutes, FTW 8.3 of them. With
``--predicted`` on an existing output the script took 4 s in a test on
2026-09-27.

Prerequisites:
    pip install "agribound[gee,delineate-anything,ftw]"   (ftw only for the crop-filter
        comparison; or use environment.yml)
    agribound auth --project YOUR_GEE_PROJECT   (not needed with --predicted)
    Run from the repository root: python examples/20_stratified_evaluation.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import agribound
from agribound.evaluate import evaluate, evaluate_frame, pixels_per_field
from agribound.postprocess import filter_polygons
from agribound.provenance import to_jsonable

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

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
STRATA_COLUMN = "Sub_Basin"  # HUC-8 subbasin name
OUTPUT_DIR = Path("outputs/stratified_evaluation")
BBOX = (-108.25, 36.6, -108.0, 36.8)  # San Juan County, NM
SOURCE = "sentinel2"
ENGINE = "delineate-anything"
YEAR = 2024
GSD_M = 10.0
# One year for all four sources, so that the rows differ in resolution only: 2022, the year
# closest to 2024 that all four cover (see above).
RESOLUTION_YEAR = 2022
RESOLUTION_SOURCES = (("landsat", 30), ("sentinel2", 10), ("spot", 6), ("naip", 1))
BOUNDARY_TOLERANCE_M = 10.0
N_BOOTSTRAP = 200
MIN_AREA = 2500  # m^2; the predictions' minimum area (agribound's default), also for the reference


def reference_exists(path):
    """True if *path* exists; for ``zip://ARCHIVE!MEMBER``, if the archive exists."""
    if str(path).startswith("zip://"):
        return Path(str(path)[len("zip://") :].split("!", 1)[0]).exists()
    return Path(path).exists()


def inside_bbox(gdf, bbox):
    """Rows whose representative point lies inside the EPSG:4326 *bbox*."""
    from shapely.geometry import box

    points = gdf.geometry.to_crs(epsg=4326).representative_point()
    return gdf[points.within(box(*bbox)).to_numpy()].copy()


def fmt(value, digits=3):
    """Format a metric (NaN-safe)."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(
        description=f"Stratified evaluation against the {REFERENCE_LABEL}."
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
        "--reference",
        default=REFERENCE,
        help="Reference polygons with a Sub_Basin column (a file in a zip archive as "
        "zip://ARCHIVE!MEMBER).",
    )
    parser.add_argument("--predicted", default=None, help="Existing predictions to evaluate.")
    parser.add_argument("--year", type=int, default=YEAR, help="Sentinel-2 year (delineation).")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP, help="Bootstrap resamples.")
    parser.add_argument(
        "--resolution-year",
        type=int,
        default=RESOLUTION_YEAR,
        help="Year of the Landsat/Sentinel-2/SPOT/NAIP comparison.",
    )
    parser.add_argument(
        "--no-resolution", action="store_true", help="Skip the resolution comparison."
    )
    parser.add_argument(
        "--no-lulc-comparison",
        action="store_true",
        help="Skip the FTW / Delineate-Anything comparison with and without the crop filter.",
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def resolution_comparison(args, reference):
    """The same engine on Landsat, Sentinel-2, SPOT 6/7 and NAIP of one year."""
    year = args.resolution_year
    minx, miny, maxx, maxy = BBOX
    print(
        f"\n{'=' * 70}\nResolution comparison: {ENGINE} on Landsat, Sentinel-2, SPOT 6/7 "
        f"and NAIP ({year})\n{'=' * 70}"
    )
    rows = []
    for source, gsd in RESOLUTION_SOURCES:
        output_path = OUTPUT_DIR / f"fields_{source}_{ENGINE}_{year}.gpkg"
        try:
            gdf = agribound.delineate(
                study_area=f"bbox:{minx},{miny},{maxx},{maxy}",
                source=source,
                year=year,
                engine=ENGINE,
                output_path=str(output_path),
                gee_project=args.gee_project,
            )
        except Exception as exc:  # e.g. no SPOT access
            print(f"  {source} {year} failed: {type(exc).__name__}: {exc}")
            continue
        gdf = inside_bbox(gdf, BBOX)
        m = evaluate(gdf, reference, iou_threshold=0.5, boundary_tolerance_m=BOUNDARY_TOLERANCE_M)
        (OUTPUT_DIR / f"metrics_{source}_{year}.json").write_text(
            json.dumps(to_jsonable(m), indent=2, allow_nan=False)
        )
        rows.append((source, gsd, len(gdf), m))
    print(
        f"  {'Source':<10} {'GSD':>5} {'Fields':>6} {'P':>6} {'R':>6} {'F1':>6} {'IoU':>6} "
        f"{'AW-R':>6} {'bF1':>6}"
    )
    for source, gsd, n, m in rows:
        print(
            f"  {source:<10} {gsd:>4}m {n:>6} {fmt(m['precision']):>6} {fmt(m['recall']):>6} "
            f"{fmt(m['f1']):>6} {fmt(m['iou_mean']):>6} {fmt(m['area_weighted_recall']):>6} "
            f"{fmt(m['boundary_f1']):>6}"
        )
    print(
        f"  (one-to-one matching at IoU >= 0.5 against {len(reference)} reference polygons; "
        f"bF1 = boundary F1 within {BOUNDARY_TOLERANCE_M:g} m)"
    )


def lulc_comparison(args, reference):
    """FTW and Delineate-Anything on Sentinel-2, each with and without the crop filter."""
    minx, miny, maxx, maxy = BBOX
    print(
        f"\n{'=' * 70}\nCrop-filter comparison: FTW and Delineate-Anything on Sentinel-2 "
        f"{args.year}, with and without the LULC filter\n{'=' * 70}"
    )
    rows = []
    for engine in ("ftw", "delineate-anything"):
        for lulc in (True, False):
            suffix = "" if lulc else "_nolulc"
            output_path = OUTPUT_DIR / f"fields_{SOURCE}_{engine}_{args.year}{suffix}.gpkg"
            try:
                gdf = agribound.delineate(
                    study_area=f"bbox:{minx},{miny},{maxx},{maxy}",
                    source=SOURCE,
                    year=args.year,
                    engine=engine,
                    output_path=str(output_path),
                    gee_project=args.gee_project,
                    lulc_filter=lulc,
                )
            except Exception as exc:
                print(f"  {engine} (crop filter {'on' if lulc else 'off'}) failed: {exc}")
                continue
            gdf = inside_bbox(gdf, BBOX)
            m = evaluate(
                gdf, reference, iou_threshold=0.5, boundary_tolerance_m=BOUNDARY_TOLERANCE_M
            )
            name = f"metrics_{SOURCE}_{engine}_{args.year}_lulc-{'on' if lulc else 'off'}.json"
            (OUTPUT_DIR / name).write_text(json.dumps(to_jsonable(m), indent=2, allow_nan=False))
            rows.append((engine, "on" if lulc else "off", len(gdf), m))
    print(
        f"  {'Engine':<20} {'Filter':>6} {'Fields':>6} {'P':>6} {'R':>6} {'F1':>6} {'IoU':>6} "
        f"{'AW-R':>6} {'FP-noref':>8}"
    )
    for engine, lulc, n, m in rows:
        print(
            f"  {engine:<20} {lulc:>6} {n:>6} {fmt(m['precision']):>6} {fmt(m['recall']):>6} "
            f"{fmt(m['f1']):>6} {fmt(m['iou_mean']):>6} {fmt(m['area_weighted_recall']):>6} "
            f"{m['count_fp_unassigned']:>8}"
        )
    print(
        f"  (one-to-one matching at IoU >= 0.5 against {len(reference)} reference polygons; "
        "FP-noref = predictions that overlap no reference polygon)"
    )


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main():
    args = parse_args()
    if not reference_exists(args.reference):
        print(
            f"Reference registry not found: {args.reference}\n"
            f"This example needs the {REFERENCE_LABEL} at {REFERENCE_ZIP} (local data, not "
            f"distributed) or another reference layer with a '{STRATA_COLUMN}' column, passed "
            "with --reference."
        )
        return
    import geopandas as gpd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_box = inside_bbox(gpd.read_file(args.reference), BBOX)
    # The predictions' minimum-area filter (areas in EPSG:6933), applied to the reference.
    reference = filter_polygons(in_box, min_area_m2=MIN_AREA)
    print(
        f"Reference: {args.reference}, {len(in_box)} polygons in the box ({in_box.crs}); "
        f"{len(in_box) - len(reference)} smaller than {MIN_AREA:,} m2 (the predictions' minimum "
        f"area) dropped, {len(reference)} evaluated"
    )
    print(f"  Sub-basins: {reference[STRATA_COLUMN].value_counts().to_dict()}")
    if "CDL_Crop_T" in reference.columns:  # V18: a CDL class per field (year not recorded)
        classes = reference["CDL_Crop_T"].fillna("(none)").value_counts().head(8).to_dict()
        print(f"  Most common 'CDL_Crop_T' classes: {classes}")

    # --- Predictions -------------------------------------------------------------------
    if args.predicted:
        predicted = gpd.read_file(args.predicted)
        label = Path(args.predicted).name
        tag = Path(args.predicted).stem.removeprefix("fields_")
    else:
        minx, miny, maxx, maxy = BBOX
        output_path = OUTPUT_DIR / f"fields_{SOURCE}_{ENGINE}_{args.year}.gpkg"
        predicted = agribound.delineate(
            study_area=f"bbox:{minx},{miny},{maxx},{maxy}",
            source=SOURCE,
            year=args.year,
            engine=ENGINE,
            output_path=str(output_path),
            gee_project=args.gee_project,
        )
        label = f"{SOURCE} + {ENGINE} {args.year}"
        tag = f"{SOURCE}_{ENGINE}_{args.year}"
    predicted = inside_bbox(predicted, BBOX)
    print(f"Predictions ({label}): {len(predicted)} polygons in the box")

    # --- Summary metrics ---------------------------------------------------------------
    m = evaluate(
        predicted,
        reference,
        iou_threshold=0.5,
        strata=STRATA_COLUMN,
        size_bins="auto",
        bootstrap=args.bootstrap,
        bootstrap_seed=42,
        boundary_tolerance_m=BOUNDARY_TOLERANCE_M,
    )
    ci = (m.get("bootstrap") or {}).get("ci", {})

    def with_ci(key):
        lo_hi = ci.get(key)
        interval = f" [{fmt(lo_hi[0])}, {fmt(lo_hi[1])}]" if lo_hi else ""
        return f"{fmt(m[key])}{interval}"

    print(f"\n{'=' * 70}\nOverall (one-to-one, IoU >= 0.5; 95 % bootstrap intervals)\n{'=' * 70}")
    for key in ("precision", "recall", "f1", "area_weighted_recall", "area_weighted_precision"):
        print(f"  {key:<26} {with_ci(key)}")
    print(f"  {'iou_mean (matched)':<26} {fmt(m['iou_mean'])}")
    print(f"  TP={m['count_tp']} FP={m['count_fp']} FN={m['count_fn']}")
    print(f"  FP overlapping no reference polygon: {m['count_fp_unassigned']}")
    print(
        f"  Boundary: Hausdorff median {fmt(m['hausdorff_median_m'], 1)} m, mean distance "
        f"median {fmt(m['boundary_distance_median_m'], 1)} m; within "
        f"{BOUNDARY_TOLERANCE_M:g} m: boundary F1 {fmt(m['boundary_f1'])}, coverage "
        f"{fmt(m['coverage_within_tolerance'])}"
    )

    print(f"\n{'=' * 70}\nPer sub-basin ({STRATA_COLUMN})\n{'=' * 70}")
    print(f"  {'Stratum':<42} {'n':>5} {'P':>6} {'R':>6} {'F1':>6} {'AW-R':>6}")
    for name, s in sorted(m["per_stratum"].items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"  {name[:42]:<42} {s['n']:>5} {fmt(s['precision']):>6} {fmt(s['recall']):>6} "
            f"{fmt(s['f1']):>6} {fmt(s['area_weighted_recall']):>6}"
        )

    print(f"\n{'=' * 70}\nPer field-size class (ha)\n{'=' * 70}")
    print(f"  {'Class (ha)':<14} {'n':>5} {'R':>6} {'F1':>6} {'recall 95 % CI':>18}")
    for name, s in m["per_size_class"].items():
        lo_hi = (s.get("ci") or {}).get("recall")
        interval = f"[{fmt(lo_hi[0])}, {fmt(lo_hi[1])}]" if lo_hi else ""
        print(f"  {name:<14} {s['n']:>5} {fmt(s['recall']):>6} {fmt(s['f1']):>6} {interval:>18}")

    # --- Per-field table and pixels per field --------------------------------------------
    frame = evaluate_frame(
        predicted, reference, strata=STRATA_COLUMN, size_bins="auto", boundary_tolerance_m=10.0
    )
    frame["pixels_per_field_s2"] = pixels_per_field(reference, GSD_M)
    table_path = OUTPUT_DIR / f"per_field_evaluation_{tag}.csv"
    frame.to_csv(table_path)
    matched = frame.groupby("matched")["pixels_per_field_s2"].median()
    print(f"\nMedian pixels per field at {GSD_M:g} m: {matched.to_dict()} (by matched flag)")

    metrics_path = OUTPUT_DIR / f"metrics_{tag}.json"
    metrics_path.write_text(json.dumps(to_jsonable(m), indent=2, allow_nan=False))
    print(f"Metrics: {metrics_path}; per-field table: {table_path}")

    if not args.predicted and not args.no_resolution:
        resolution_comparison(args, reference)
    if not args.predicted and not args.no_lulc_comparison:
        lulc_comparison(args, reference)

    from agribound.visualize import show_comparison

    map_path = OUTPUT_DIR / f"map_predicted_vs_reference_{tag}.html"
    web_map = show_comparison(
        [predicted, reference],
        labels=[label, REFERENCE_LABEL if args.reference == REFERENCE else "Reference"],
        basemap="Esri.WorldImagery",
        output_html=str(map_path),
    )
    show_in_notebook(web_map)
    print(f"Map: {map_path}")


if __name__ == "__main__":
    main()
