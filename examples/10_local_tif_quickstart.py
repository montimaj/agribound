"""
10 — Local GeoTIFF Quickstart

Minimal example: delineate field boundaries from a local GeoTIFF with the
Delineate-Anything engine (default model ``large_v2``, label-free) and show
them on a map. No Earth Engine account is needed.

Notes:
    - ``source="local"`` needs ``local_tif_path``; a study area is optional.
      Without one the whole raster is used; with one the raster is cropped
      to the study area's bounding box.
    - Band order: without ``bands``, bands 1, 2 and 3 are read as R, G, B.
      Pass ``--bands R=3,G=2,B=1`` (1-based) for other layouts.
    - Delineate-Anything stretches each band to 8 bits with scene-wide 1-99
      percentiles (8-bit rasters are used unchanged), so the file's value
      scale does not need to be declared. The published models were trained on
      0.25-10 m imagery; a pixel size more than 5 % outside that range logs
      a WARNING.
    - The LULC crop filter reads land-cover maps from Earth Engine, so it is
      disabled here (``lulc_filter=False``).
    - Output: ``fields_local_delineate-anything_<tif name>[_bands-R3G2B1]
      [_aoi-<study area name>].gpkg`` and ``map.html`` in
      ``outputs/local_quickstart/``. An existing output made with the same
      settings is loaded instead of recomputed; ``--overwrite`` recomputes it.

Estimated runtime (not measured for 1.0): ~2-5 minutes for a small raster (GPU
or CPU).

Prerequisites:
    pip install "agribound[delineate-anything]"
    python examples/10_local_tif_quickstart.py --tif path/to/image.tif
"""

import argparse
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
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
LOCAL_TIF = "path/to/your/satellite_image.tif"  # replace, or pass --tif
OUTPUT_DIR = Path("outputs/local_quickstart")


def parse_bands(text):
    """Parse "R=1,G=2,B=3" into {"R": 1, "G": 2, "B": 3}."""
    if not text:
        return None
    return {k.strip().upper(): int(v) for k, v in (item.split("=") for item in text.split(","))}


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Delineate fields from a local GeoTIFF.")
    parser.add_argument("--tif", default=LOCAL_TIF, help="Input GeoTIFF.")
    parser.add_argument("--study-area", default=None, help="Optional study-area vector file.")
    parser.add_argument("--bands", default=None, help='Band indices, e.g. "R=3,G=2,B=1".')
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute an output that already exists."
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def map_ready(gdf):
    """Copy of *gdf* with datetime columns as ISO-8601 text, for an HTML map.

    leafmap 0.63 cannot write pandas Timestamp values (such as agribound's
    ``determination:datetime`` column) into the map's HTML: it raises a JSON
    serialisation error.
    """
    import pandas as pd

    out = gdf.copy()
    for column in out.columns:
        if column != out.geometry.name and pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].map(lambda t: t.isoformat() if pd.notna(t) else None)
    return out


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main():
    args = parse_args()
    if not Path(args.tif).exists():
        print(f"GeoTIFF not found: {args.tif}. Set LOCAL_TIF or pass --tif path/to/image.tif.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The output name records the input, band order and study area, so runs with
    # different options keep separate outputs.
    bands = parse_bands(args.bands)
    name = Path(args.tif).stem
    if bands:
        name += "_bands-" + "".join(f"{key}{value}" for key, value in bands.items())
    if args.study_area:
        name += f"_aoi-{Path(args.study_area).stem}"
    output_path = OUTPUT_DIR / f"fields_local_delineate-anything_{name}.gpkg"

    gdf = agribound.delineate(
        study_area=args.study_area,
        source="local",
        engine="delineate-anything",
        local_tif_path=args.tif,
        bands=bands,
        output_path=str(output_path),
        lulc_filter=False,  # the LULC filter needs Earth Engine
        overwrite=args.overwrite,
    )

    print(f"Delineated {len(gdf)} field boundaries -> {output_path}")
    if len(gdf) == 0:
        return
    area = gdf["metrics:area"]
    print(f"Total area: {area.sum() / 10000:,.1f} ha")
    print(f"Mean field: {area.mean() / 10000:,.2f} ha")
    print(f"Smallest field: {area.min():,.0f} m^2; largest: {area.max() / 10000:,.1f} ha")

    web_map = agribound.show_boundaries(
        map_ready(gdf),
        satellite_tif=args.tif,
        output_html=str(OUTPUT_DIR / "map.html"),
    )
    show_in_notebook(web_map)
    print(f"\nMap saved to {OUTPUT_DIR / 'map.html'}")


if __name__ == "__main__":
    main()
