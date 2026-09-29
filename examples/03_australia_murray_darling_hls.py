"""
03 — Australia, Murray-Darling Basin: HLS with the Prithvi Engine (label-free modes)

Compares the two label-free modes of the Prithvi engine on Harmonized
Landsat Sentinel-2 (HLS v2.0, 30 m) composites near Narrabri, New South Wales,
with a label-free field-boundary model on higher-resolution imagery:

    1. ``mode="embed"`` — Prithvi-EO-2.0 (default ``Prithvi-EO-2.0-300M-TL``)
       patch-token features of one encoder layer, clustered with K-means.
    2. ``mode="pca"`` — baseline without the ViT: K-means on the PCA of
       per-band z-scores of R, G, B and NIR.
    3. For comparison, Delineate-Anything v2 (``large_v2``) on SPOT 6/7 (6 m),
       with the same minimum area, simplification and LULC crop filter.

Both Prithvi modes cluster pixels into land-cover segments; they do not
delineate field instances, and neither is trained on field boundaries. For
field boundaries from Prithvi, fine-tune a segmentation model on reference
polygons (``fine_tune=True``, ``mode="segment"``). Delineate-Anything is a
field-instance model, used here as released.

SPOT 6/7 is restricted to select Earth Engine users (internal DRI use); without
access the SPOT run fails and the script continues. AIRBUS/SPOT6_7 has no
scene over this study area in 2022, so the SPOT run uses 2023 (3 scenes cover
the whole box; 2 in 2021; queried 2026-09-28).

Inputs are HLS surface reflectance x 10000 (the scale Prithvi-EO-2.0 was
pre-trained on); the six Prithvi bands are Blue, Green, Red, narrow NIR
(HLS B5), SWIR 1 and SWIR 2.

Environment: ``mode="embed"`` needs terratorch, which is installed in the
GFM environment (``environment-gfm.yml`` or ``pip install "agribound[all-gfm]"``;
terratorch needs lightning >= 2.6 and ftw-tools needs lightning < 2.6, so they
cannot share one environment). ``mode="pca"`` runs in either environment. In
an environment without terratorch the script skips the embed mode.
Delineate-Anything needs ultralytics (``agribound[delineate-anything]``; in both
environments); without it the SPOT run is skipped.

Years: 2022 by default (``--years 2022-2024`` for three years). The LULC crop
filter is on (Dynamic World is selected outside the US) and needs Earth
Engine, like the HLS composites.

Estimated runtime (not measured for 1.0): ~10-30 minutes per year (GPU
recommended for embed mode).

Prerequisites:
    GFM environment: conda env create -f environment-gfm.yml
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/03_australia_murray_darling_hls.py
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import agribound

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
OUTPUT_DIR = Path("outputs/australia_murray_darling")
SOURCE = "hls"
ENGINE = "prithvi"
DEFAULT_YEARS = "2022"  # e.g. "2022-2024"
SPOT_YEAR = 2023  # no SPOT 6/7 scene over this box in 2022 (see the docstring)

AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [149.70, -30.30],
                        [149.85, -30.30],
                        [149.85, -30.15],
                        [149.70, -30.15],
                        [149.70, -30.30],
                    ]
                ],
            },
            "properties": {"name": "Murray-Darling Basin AOI (near Narrabri, NSW)"},
        }
    ],
}


def parse_years(text):
    """Parse "2022", "2022,2024" or "2022-2024" into a list of years."""
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
    parser = argparse.ArgumentParser(description="Murray-Darling Basin: Prithvi embed vs PCA.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument("--years", default=DEFAULT_YEARS, help='e.g. "2022" or "2022-2024".')
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
    study_area = str(OUTPUT_DIR / "murray_darling_aoi.geojson")
    Path(study_area).write_text(json.dumps(AOI))

    modes = ["pca"]
    if importlib.util.find_spec("terratorch") is not None:
        modes.insert(0, "embed")
    else:
        print(
            "terratorch is not installed: skipping mode='embed' (use the GFM environment, "
            "environment-gfm.yml, to run it)."
        )

    results = {}
    for year in parse_years(args.years):
        print(f"\n{'=' * 60}\nYear {year}\n{'=' * 60}")
        for mode in modes:
            output_path = OUTPUT_DIR / f"fields_hls_prithvi-{mode}_{year}.gpkg"
            try:
                gdf = agribound.delineate(
                    study_area=study_area,
                    source=SOURCE,
                    year=year,
                    engine=ENGINE,
                    output_path=str(output_path),
                    gee_project=args.gee_project,
                    composite_method="median",
                    min_area=5000,
                    simplify=3.0,
                    engine_params={"mode": mode},
                )
            except Exception as exc:
                print(f"  mode={mode} failed: {type(exc).__name__}: {exc}")
                continue
            results[f"Prithvi {mode} {year}"] = gdf
            print(f"  mode={mode}: {len(gdf)} polygons -> {output_path}")

    print(f"\n{'=' * 60}\nDelineate-Anything v2 on SPOT 6/7 ({SPOT_YEAR})\n{'=' * 60}")
    if importlib.util.find_spec("ultralytics") is None:
        print(
            "  ultralytics is not installed: skipping (pip install 'agribound[delineate-anything]')"
        )
    else:
        output_path = OUTPUT_DIR / f"fields_spot_delineate-anything_{SPOT_YEAR}.gpkg"
        try:
            gdf = agribound.delineate(
                study_area=study_area,
                source="spot",
                year=SPOT_YEAR,
                engine="delineate-anything",
                output_path=str(output_path),
                gee_project=args.gee_project,
                composite_method="median",
                cloud_cover_max=15,
                min_area=5000,
                simplify=3.0,
            )
            results[f"Delineate-Anything SPOT {SPOT_YEAR}"] = gdf
            print(f"  {len(gdf)} polygons -> {output_path}")
        except Exception as exc:  # e.g. no SPOT access
            print(f"  SPOT {SPOT_YEAR} failed: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 60}\nComparison\n{'=' * 60}")
    print(f"  {'Run':<32} {'Polygons':>9} {'Area (ha)':>12}")
    for label, gdf in results.items():
        area = gdf["metrics:area"].sum() / 10000 if "metrics:area" in gdf.columns else 0.0
        print(f"  {label:<32} {len(gdf):>9} {area:>12,.1f}")

    if not results:
        print("\nNo run succeeded; no map written.")
        return
    from agribound.visualize import show_comparison

    web_map = show_comparison(
        list(results.values()),
        labels=list(results.keys()),
        basemap="Esri.WorldImagery",
        output_html=str(OUTPUT_DIR / "map_murray_darling.html"),
    )
    show_in_notebook(web_map)
    print(f"\n  Map: {OUTPUT_DIR / 'map_murray_darling.html'}")


if __name__ == "__main__":
    main()
