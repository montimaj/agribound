"""
22 — SPOT 6/7 Panchromatic across the Global South: Six Farming Landscapes

Runs Delineate-Anything v2 (``large_v2``, as released) on SPOT 6/7
panchromatic composites (1.5 m, ``source="spot-pan"``) of six small study
areas, one per farming landscape:

    1. Cauvery Delta, Tamil Nadu, India: smallholder paddies (2018).
    2. Hetao irrigation district, Inner Mongolia, China: fields in a canal
       grid (2021).
    3. Agrelo, Mendoza, Argentina: irrigated vineyards (2019).
    4. Mwea irrigation scheme, Kenya: rice basins split into tenant strips
       (2020).
    5. Nile Delta near Tanta, Egypt: narrow strip plots (2020).
    6. Luis Eduardo Magalhaes, western Bahia, Brazil: centre pivots (2018).

Each study area is a 3 km x 3 km square in its UTM zone (about 2000 x 2000
pixels at 1.5 m), except in western Bahia, where the pivots are about 1 km
across and the square is 6 km (about 4000 x 4000 pixels). Each composite is the
median of the SPOT scenes of one calendar year with at most 15 % cloud cover.
The years and centres were chosen on 2026-09-29 from scene lists of
AIRBUS/SPOT6_7 (the collection ends on 2023-11-15): in each year, 2 to 6
scenes cover the whole square and no scene covers only part of it, so the
median has no seams.

Delineation runs without the LULC crop filter, and the filter (Dynamic World
of the same year, the mean ``crops`` probability >= 0.3) is applied afterwards
as a separate step, and both layers are written. The filtered layer can miss
fields: on 2026-09-29 it kept 82 to 95 % of the polygons in five of the areas,
but only 47 of the 320 in western Bahia, where it dropped most centre pivots.
Nothing here is evaluated against reference boundaries: the table compares
polygon counts and areas only.

SPOT 6/7 (AIRBUS/SPOT6_7) is restricted to select Earth Engine users. Without
access every run fails and the script says so.

Estimated runtime: all six areas took 6.1 minutes on an Apple M2 Max (MPS),
downloads and the crop-filter step included, on 2026-09-29 (a fresh run with
agribound 1.0.1).

Prerequisites:
    pip install "agribound[gee,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/22_global_south_spot_pan.py
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

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/global_south_spot_pan")
SIDE_M = 3000  # side of each square study area, in metres (unless an area sets its own)
MIN_AREA = 100  # m^2; the minimum polygon area (as in example 02's SPOT-Pan run)
CLOUD_COVER_MAX = 15  # per-scene cloud cover, percent
CROP_THRESHOLD = 0.3  # LULC crop value threshold of the separate crop-filter step
AREAS = [
    # slug, name, centre longitude and latitude, year, side of the square (m)
    ("cauvery_delta", "Cauvery Delta, Tamil Nadu, India", 79.0726, 10.7864, 2018, SIDE_M),
    ("hetao", "Hetao irrigation district, Inner Mongolia, China", 107.3400, 40.8300, 2021, SIDE_M),
    ("mendoza", "Agrelo, Mendoza, Argentina", -68.9200, -33.1100, 2019, SIDE_M),
    ("mwea", "Mwea irrigation scheme, Kenya", 37.3750, -0.7350, 2020, SIDE_M),
    ("nile_delta", "Nile Delta near Tanta, Egypt", 30.9750, 30.8350, 2020, SIDE_M),
    # Pivots about 1 km across: a larger square (5 full scenes in 2018, none partial).
    ("western_bahia", "Luis Eduardo Magalhaes, Bahia, Brazil", -45.7909, -12.1407, 2018, 6000),
]


def square_study_area(lon, lat, side_m, name):
    """A GeoJSON FeatureCollection: a square of *side_m* metres in the local UTM zone."""
    import pyproj
    from shapely.geometry import box, mapping
    from shapely.ops import transform

    from agribound.io import get_utm_crs

    if abs(lon) + side_m / 111_000 >= 180:  # the corners are transformed one by one
        raise ValueError(f"{name}: a {side_m} m square at lon={lon} crosses the antimeridian")
    utm = get_utm_crs(lon, lat)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_ll = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    x, y = to_utm.transform(lon, lat)
    half = side_m / 2
    square = transform(to_ll.transform, box(x - half, y - half, x + half, y + half))
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(square), "properties": {"name": name}}
        ],
    }


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="SPOT-Pan across the Global South.")
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
        "--only",
        nargs="+",
        default=None,
        metavar="SLUG",
        choices=[a[0] for a in AREAS],
        help=f"Run only these study areas ({', '.join(a[0] for a in AREAS)}).",
    )
    parser.add_argument(
        "--skip-crop-filter", action="store_true", help="Skip the separate crop-filter step."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute outputs that already exist."
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def main():
    args = parse_args()
    start_time = time.time()
    from agribound.config import AgriboundConfig
    from agribound.postprocess.lulc_filter import filter_by_lulc

    rows = []
    for slug, name, lon, lat, year, side_m in AREAS:
        if args.only and slug not in args.only:
            continue
        print(f"\n{'=' * 70}\n{name} ({year})\n{'=' * 70}")
        out_dir = OUTPUT_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        study_area = out_dir / "study_area.geojson"
        study_area.write_text(json.dumps(square_study_area(lon, lat, side_m, name)))
        config = AgriboundConfig(
            study_area=str(study_area),
            source="spot-pan",
            year=year,
            engine="delineate-anything",
            output_path=str(out_dir / f"fields_spot-pan_delineate-anything_{year}.gpkg"),
            gee_project=args.gee_project,
            cloud_cover_max=CLOUD_COVER_MAX,
            min_field_area_m2=MIN_AREA,
            simplify_tolerance=1.0,
            lulc_filter=False,
            lulc_crop_threshold=CROP_THRESHOLD,
            overwrite=args.overwrite,
        )
        try:
            fields = agribound.delineate(config=config)
        except Exception as exc:  # e.g. no SPOT access
            print(f"  {name} failed: {type(exc).__name__}: {exc}")
            continue
        print(f"  {len(fields)} polygons -> {config.output_path}")
        n_crop = None
        if not args.skip_crop_filter and len(fields):
            crop = filter_by_lulc(fields, config)
            crop_path = out_dir / f"fields_spot-pan_delineate-anything_{year}_crop.gpkg"
            crop.to_file(crop_path, layer="fields")
            n_crop = len(crop)
            print(f"  crop filter kept {n_crop} of {len(fields)} -> {crop_path}")
        area_ha = fields["metrics:area"] / 10000 if "metrics:area" in fields.columns else None
        rows.append((name, year, len(fields), area_ha, n_crop))

    print(f"\n{'=' * 70}\nComparison (no reference data: counts and areas only)\n{'=' * 70}")
    print(f"  {'Study area':<50} {'Year':>5} {'Polygons':>9} {'Median ha':>10} {'Crop filter':>12}")
    for name, year, n, area_ha, n_crop in rows:
        median = f"{area_ha.median():.2f}" if area_ha is not None and n else "-"
        kept = "-" if n_crop is None else f"{n_crop}"
        print(f"  {name:<50} {year:>5} {n:>9} {median:>10} {kept:>12}")
    if not rows:
        print("\nNo run succeeded.")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
