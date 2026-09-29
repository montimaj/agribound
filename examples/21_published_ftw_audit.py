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
       bootstrap intervals, 10 m boundary tolerance) against the NMOSE WUCB
       polygons, with both layers restricted to the box by the same rule
       (representative point inside the box).

Default AOI: 106.80-106.75 W, 34.60-34.65 N near Belen (Valencia County, New
Mexico), in the Middle Rio Grande Conservancy District. Vintage mismatch:
the FTW polygons are predictions for 2024 (``--year 2025`` for 2025), while
the registry is older: most filled ``Source`` values cite 2011-2016 NAIP
and/or CDL imagery (387 of the 15,075 filled values over the whole registry
cite only an OSE shapefile or "CW", without an imagery year); it is empty
for every polygon of the default AOI (the script prints it). Fields that
changed in between count as errors, so the scores measure agreement with an
older registry, not only FTW's error. NMOSE may not include every field in
the box (``count_fp_unassigned`` = FTW polygons overlapping no reference
polygon).

Other regions: ``--bbox minx,miny,maxx,maxy --reference PATH``. Without the
NMOSE shapefile and without ``--reference`` the script exits with a message.

Runtime: 17 s for the default AOI in a test on 2026-09-27 (network; the
partition index is cached under ~/.cache/agribound/ftw).

Prerequisites:
    pip install agribound
    NMOSE shapefile at "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
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
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
NMOSE_SHAPEFILE = "examples/NMOSE Field Boundaries/WUCB ag polys.shp"
OUTPUT_DIR = Path("outputs/published_ftw_audit")
DEFAULT_BBOX = "-106.80,34.60,-106.75,34.65"  # near Belen, NM
YEAR = 2024
BOUNDARY_TOLERANCE_M = 10.0


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
    parser.add_argument("--reference", default=NMOSE_SHAPEFILE, help="Reference polygons.")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples.")
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
    if not Path(args.reference).exists():
        print(
            f"Reference layer not found: {args.reference}. Pass --reference PATH (and --bbox "
            "for another region), or place the NMOSE shapefile at the default path."
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
    reference = inside_bbox(gpd.read_file(args.reference), bbox)
    ftw_in = inside_bbox(ftw, bbox)
    print(f"\nIn the box: {len(ftw_in)} FTW polygons, {len(reference)} reference polygons")
    if "Source" in reference.columns:
        sources = reference["Source"].fillna("(empty)").value_counts().to_dict()
        print(f"  Reference 'Source' attribute: {sources}")

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
    (OUTPUT_DIR / f"metrics_{tag}.json").write_text(json.dumps(m, indent=2, default=str))

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
