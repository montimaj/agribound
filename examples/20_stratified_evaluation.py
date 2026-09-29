"""
20 — Stratified Evaluation against a Reference Registry (NMOSE, San Juan Basin)

Evaluates predicted field boundaries against the NMOSE WUCB agricultural
polygons with the full ``agribound.evaluate`` toolkit:

    - one-to-one IoU matching (IoU >= 0.5): precision, recall, F1, mean IoU
      of matches, area-weighted recall/precision;
    - per stratum (the NMOSE sub-basin column ``WUCR_SBasi``) and per
      field-size class (``size_bins="auto"``: 1-2-5 series of hectares);
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
(Sentinel-2 2019, a year close to the registry's 2016 imagery; ``--year``
changes it; Delineate-Anything ``large_v2``; LULC crop filter on, NLCD).
``--predicted PATH`` evaluates any existing layer instead (no Earth Engine
needed then). Both layers are restricted to the study area by the same rule:
a polygon is kept when its representative point lies inside the box (the
pipeline applies this rule, ``aoi_selection="representative_point"``, to its
own output by default).

Resolution comparison: 2018 for all four sources (``--resolution-year``).
NAIP is flown every two years here (2016, 2018, 2020 and 2022 cover the box;
queried 2026-09-28), so 2019 has no NAIP; 2018 is the year closest to the
registry's 2016 imagery that Landsat, Sentinel-2, SPOT 6/7 and NAIP all
cover (the 2018 Landsat composite uses Landsat 7 and 8; Landsat 9 data start
on 2021-10-31). Delineate-Anything was trained on 0.25-10 m imagery, so the
Landsat run is outside its range (agribound logs a WARNING and records it in
``engine_meta``). SPOT 6/7 is restricted to select Earth Engine users; without
access that run fails and the script continues.

Crop-filter comparison: FTW (``FTW_PRUE_EFNET_B5``, two season windows) and
Delineate-Anything (``large_v2``), both used as released, on Sentinel-2 of
``--year``. With the filter on, a polygon is kept when at least 30 % of its
area is Annual NLCD cultivated crops or pasture/hay (classes 82 and 81) in
that year; with it off, every polygon is kept. Delineate-Anything with the
filter is the main run above, so it is loaded rather than recomputed.

Study area: 108.25-108.00 W, 36.6-36.8 N (San Juan County); 944 NMOSE
polygons have their representative point inside it, in six sub-basins. The
registry's ``Source`` attribute cites 2016 NAIP for these polygons; the
script prints the values. Predictions from other years count real changes
since then as errors. NMOSE may not include every field in the box;
``count_fp_unassigned`` counts predictions that overlap no reference polygon.

The NMOSE shapefile must be at "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
(or pass ``--reference``); otherwise the script exits with a message.

Runtime: about 18 minutes with delineation in a test on 2026-09-27 (17.5 min
to download the 2301 x 2287 px Sentinel-2 composite from Earth Engine, 19 s of
Delineate-Anything inference on Apple MPS, 10 s of LULC filtering); 4 s with
``--predicted`` on that output. The resolution comparison adds four
composites, the largest the ~22 x 22 km NAIP mosaic at 1 m (about 1.9 GB).

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

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("geedim").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
NMOSE_SHAPEFILE = "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
STRATA_COLUMN = "WUCR_SBasi"
OUTPUT_DIR = Path("outputs/stratified_evaluation")
BBOX = (-108.25, 36.6, -108.0, 36.8)  # San Juan County, NM
SOURCE = "sentinel2"
ENGINE = "delineate-anything"
YEAR = 2019
GSD_M = 10.0
RESOLUTION_YEAR = 2018  # the year closest to 2016 that all four sources cover (see above)
RESOLUTION_SOURCES = (("landsat", 30), ("sentinel2", 10), ("spot", 6), ("naip", 1))
BOUNDARY_TOLERANCE_M = 10.0
N_BOOTSTRAP = 200


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
    parser = argparse.ArgumentParser(description="Stratified evaluation against NMOSE.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument("--reference", default=NMOSE_SHAPEFILE, help="Reference polygons.")
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
            json.dumps(m, indent=2, default=str)
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
        f"  (one-to-one matching at IoU >= 0.5 against the {len(reference)} NMOSE polygons; "
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
            (OUTPUT_DIR / name).write_text(json.dumps(m, indent=2, default=str))
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
        f"  (one-to-one matching at IoU >= 0.5 against the {len(reference)} NMOSE polygons; "
        "FP-noref = predictions that overlap no reference polygon)"
    )


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main():
    args = parse_args()
    if not Path(args.reference).exists():
        print(
            f"Reference registry not found: {args.reference}\n"
            "This example needs the NMOSE WUCB polygons (or another reference layer with a "
            f"'{STRATA_COLUMN}' column, passed with --reference)."
        )
        return
    import geopandas as gpd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = inside_bbox(gpd.read_file(args.reference), BBOX)
    print(f"Reference: {len(reference)} polygons in the box ({reference.crs})")
    print(f"  Sub-basins: {reference[STRATA_COLUMN].value_counts().to_dict()}")
    print(f"  'Source' attribute: {reference['Source'].fillna('').value_counts().to_dict()}")

    # --- Predictions -------------------------------------------------------------------
    if args.predicted:
        predicted = gpd.read_file(args.predicted)
        label = Path(args.predicted).name
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
    frame.to_csv(OUTPUT_DIR / "per_field_evaluation.csv")
    matched = frame.groupby("matched")["pixels_per_field_s2"].median()
    print(f"\nMedian pixels per field at {GSD_M:g} m: {matched.to_dict()} (by matched flag)")

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(m, indent=2, default=str))
    print(f"Metrics: {metrics_path}; per-field table: {OUTPUT_DIR / 'per_field_evaluation.csv'}")

    if not args.predicted and not args.no_resolution:
        resolution_comparison(args, reference)
    if not args.predicted and not args.no_lulc_comparison:
        lulc_comparison(args, reference)

    from agribound.visualize import show_comparison

    web_map = show_comparison(
        [predicted, reference],
        labels=[label, "NMOSE reference"],
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_predicted_vs_reference.html"),
    )
    show_in_notebook(web_map)
    print(f"Map: {OUTPUT_DIR / 'map_predicted_vs_reference.html'}")


if __name__ == "__main__":
    main()
