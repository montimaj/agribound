"""
21 — Audit of Published FTW Polygons against a Reference Layer (small AOI)

Downloads the published Fields of The World (FTW) global prediction polygons
for a small study area with ``agribound.query_ftw`` and evaluates them
against reference boundaries, without running any model:

    1. ``query_ftw`` with the default layout ``"by-admin-conf"`` (Source
       Cooperative ``alpha/results-by-admin-conf``: 2024 and 2025
       predictions partitioned by country and subdivision, with a
       ``confidence`` column on a 0-100 scale), ``min_confidence=69`` (the
       dataset README's recommendation) and ``keep_null_confidence=True``.
    2. A report of how many polygons have a confidence value at all. A null
       confidence means the 500 m confidence raster has no data at the field,
       not a low score. On 2026-09-27 the confidence was null for every row
       of the New Mexico partition (US_NM), so in this default AOI
       ``min_confidence`` filters nothing while null values are kept, and
       dropping them (``keep_null_confidence=False``) would leave no polygons.
    3. ``agribound.evaluate`` (one-to-one IoU matching, size classes,
       bootstrap intervals, 10 m boundary tolerance) against the DRI/NMOSE
       field boundaries (V18), with both layers restricted to the box by the
       same rule (representative point inside the box) and filtered at the
       same minimum area.

Minimum area: the published polygons have no minimum area, while agribound
drops polygons smaller than 2,500 m2 (``min_area``, the default) from its own
outputs, and the other New Mexico examples drop the reference fields smaller
than that. So both layers here are filtered at 2,500 m2 (``--min-area``;
areas in EPSG:6933, as agribound's area filter computes them; 0 keeps every
polygon), and the script prints how many polygons of each layer it drops. In
the default AOI that is 152 of the 797 V18 polygons (83 of them under
1,250 m2) and 95 of the 322 published FTW 2024 polygons in the box (queried
2026-10-08).

Default AOI: 106.80-106.75 W, 34.60-34.65 N near Belen (Valencia County, New
Mexico), in the Middle Rio Grande Conservancy District; 797 V18 polygons
have their representative point inside it, 645 of them at least 2,500 m2.
Vintage: the FTW polygons are predictions for 2024 (``--year 2025`` for
2025), and V18 records no imagery date per field, so fields that changed
between the reference's (unrecorded) date and the FTW year count as errors:
the scores measure agreement with the registry, not only FTW's error. V18's
``CDL_Crop_T`` column holds a Cropland Data Layer class per field (year not
recorded); in this AOI most of the 645 evaluated fields have a class other
than a crop (Shrubland 213, Grassland/Pasture 115, Woody Wetlands 86 and the
Developed classes 67), and the script prints the most common classes. The
reference may not include every field in the box (``count_fp_unassigned`` =
FTW polygons overlapping no reference polygon).

Other regions: ``--bbox minx,miny,maxx,maxy --reference PATH`` (a file in a
zip archive as ``zip://ARCHIVE!MEMBER``). Without
``examples/NM_Field_Boundaries.zip`` and without ``--reference`` the script
exits with a message.

Runtime: 32 s for the default AOI in the development run of agribound 1.1
(2026-10-09; the query 22 s), about 20 s in a test on 2026-10-08 (network;
the partition index is cached under ~/.cache/agribound/ftw). In that run the
published FTW 2024 polygons scored precision 0.63, recall 0.22 and F1 0.33
(95 % interval 0.28-0.37) against the 645 evaluated V18 fields (median
0.88 ha), with recall rising from 0.03 for fields of 0.2-0.5 ha to 0.92 for
10-20 ha.

Prerequisites:
    pip install agribound
    DRI/NMOSE field boundaries (V18) at "examples/NM_Field_Boundaries.zip" (local
        data, not distributed; read in place, no need to unzip)
    Run from the repository root: python examples/21_published_ftw_audit.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import agribound
from agribound.evaluate import evaluate
from agribound.ftw_arrow import RECOMMENDED_MIN_CONFIDENCE
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
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
REFERENCE_ZIP = "examples/NM_Field_Boundaries.zip"
REFERENCE = f"zip://{REFERENCE_ZIP}!Master_NM_Field_Boundaries_V18.shp"
REFERENCE_LABEL = "DRI/NMOSE field boundaries (V18)"
OUTPUT_DIR = Path("outputs/published_ftw_audit")
DEFAULT_BBOX = "-106.80,34.60,-106.75,34.65"  # near Belen, NM
YEAR = 2024
BOUNDARY_TOLERANCE_M = 10.0
MIN_AREA = 2500  # m^2; agribound's default minimum field area, applied to both layers


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
    parser = argparse.ArgumentParser(description="Audit published FTW polygons.")
    parser.add_argument("--bbox", default=DEFAULT_BBOX, help="minx,miny,maxx,maxy (EPSG:4326).")
    parser.add_argument("--year", type=int, default=YEAR, choices=[2024, 2025])
    parser.add_argument(
        "--reference",
        default=REFERENCE,
        help="Reference polygons (a file in a zip archive as zip://ARCHIVE!MEMBER).",
    )
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples.")
    parser.add_argument(
        "--min-area",
        type=float,
        default=MIN_AREA,
        help="Minimum polygon area in m2 (EPSG:6933) of both layers; 0 keeps every polygon.",
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
    if not reference_exists(args.reference):
        print(
            f"Reference layer not found: {args.reference}. Pass --reference PATH (and --bbox "
            f"for another region), or place the {REFERENCE_LABEL} at {REFERENCE_ZIP} (local "
            "data, not distributed)."
        )
        return
    import geopandas as gpd

    bbox = tuple(float(v) for v in args.bbox.split(","))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.year}_" + "_".join(f"{v:g}" for v in bbox)

    # 1. Published FTW polygons ----------------------------------------------------------
    ftw = agribound.query_ftw(
        study_area=bbox,
        year=args.year,
        clip=False,  # whole polygons; selected by representative point below
        min_confidence=RECOMMENDED_MIN_CONFIDENCE,
        keep_null_confidence=True,
        output_path=OUTPUT_DIR / f"ftw_published_{tag}.parquet",
    )
    info = ftw.attrs.get("ftw_query", {})
    print(f"\nPublished FTW polygons ({args.year}): {len(ftw)}")
    print(json.dumps(info, indent=1, default=str)[:2000])

    # 2. Confidence coverage ------------------------------------------------------------------
    if "confidence" in ftw.columns:
        has_conf = ftw["confidence"].notna()
        n_confident = int((has_conf & (ftw["confidence"] >= RECOMMENDED_MIN_CONFIDENCE)).sum())
        print(
            f"\nConfidence present for {int(has_conf.sum())} of {len(ftw)} polygons; "
            f"min_confidence={RECOMMENDED_MIN_CONFIDENCE:g} with keep_null_confidence=False "
            f"would keep {n_confident}."
        )

    # 3. Evaluation against the reference ----------------------------------------------------
    reference_box = inside_bbox(gpd.read_file(args.reference), bbox)
    ftw_box = inside_bbox(ftw, bbox)
    # The same minimum area for both layers, computed as agribound's area filter does.
    reference = filter_polygons(reference_box, min_area_m2=args.min_area)
    ftw_in = filter_polygons(ftw_box, min_area_m2=args.min_area)
    print(
        f"\nIn the box: {len(ftw_box)} FTW polygons, {len(reference_box)} reference polygons; "
        f"smaller than {args.min_area:,.0f} m2 (dropped): {len(ftw_box) - len(ftw_in)} FTW, "
        f"{len(reference_box) - len(reference)} reference; evaluated: {len(ftw_in)} FTW, "
        f"{len(reference)} reference"
    )
    if "Source" in reference.columns:
        sources = reference["Source"].fillna("(empty)").value_counts().to_dict()
        print(f"  Reference 'Source' attribute: {sources}")
    if "CDL_Crop_T" in reference.columns:  # V18: a CDL class per field (year not recorded)
        classes = reference["CDL_Crop_T"].fillna("(none)").value_counts().head(8).to_dict()
        print(f"  Most common 'CDL_Crop_T' classes: {classes}")

    m = evaluate(
        ftw_in,
        reference,
        size_bins="auto",
        bootstrap=args.bootstrap,
        boundary_tolerance_m=BOUNDARY_TOLERANCE_M,
    )
    ci = (m.get("bootstrap") or {}).get("ci", {})
    print(
        f"\n{'=' * 70}\nPublished FTW {args.year} vs reference (one-to-one, IoU >= 0.5)\n{'=' * 70}"
    )
    for key in ("precision", "recall", "f1", "area_weighted_recall", "boundary_f1"):
        lo_hi = ci.get(key)
        interval = f" [{fmt(lo_hi[0])}, {fmt(lo_hi[1])}]" if lo_hi else ""
        print(f"  {key:<22} {fmt(m[key])}{interval}")
    print(f"  TP={m['count_tp']} FP={m['count_fp']} FN={m['count_fn']}")
    print(f"  FTW polygons overlapping no reference polygon: {m['count_fp_unassigned']}")
    print(f"  Median reference field: {fmt(m['median_area_ha'], 2)} ha")
    print(f"\n  {'Size class (ha)':<16} {'n':>5} {'recall':>7}")
    for name, s in m["per_size_class"].items():
        print(f"  {name:<16} {s['n']:>5} {fmt(s['recall']):>7}")
    (OUTPUT_DIR / f"metrics_{tag}.json").write_text(
        json.dumps(to_jsonable(m), indent=2, allow_nan=False)
    )

    from agribound.visualize import show_comparison

    web_map = show_comparison(
        [ftw_in, reference],
        labels=[f"Published FTW {args.year}", "Reference"],
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / f"map_ftw_vs_reference_{tag}.html"),
    )
    show_in_notebook(web_map)
    print(f"\nMap: {OUTPUT_DIR / f'map_ftw_vs_reference_{tag}.html'}")


if __name__ == "__main__":
    main()
