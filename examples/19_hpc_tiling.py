"""
19 — Tiled Runs with agribound.hpc (make, run, merge) on a Small AOI

Runs the HPC tiling workflow locally on a small study area in the Beauce
(France), so every step can be inspected before submitting a large region to
a Slurm cluster (``examples/hpc/``, ``examples/regions/``):

    1. ``make_tiles`` cuts the study area into square core cells on a UTM grid
       and adds a halo around each core.
    2. ``write_tile_manifest`` writes ``manifest.json``, ``tiles.gpkg``,
       ``tiles.txt`` and one configuration per tile
       (``tiles/<tile_id>/config.yaml``; study area = the halo's bounding box,
       own output, cache and provenance record).
    3. ``run_tile`` in two phases, as on a cluster: ``stage="composite"``
       (downloads only; a CPU job with internet access) for every tile, then
       ``stage="delineate"`` (reads the cache; a GPU job). With the LULC
       filter on, this script sets ``lulc_mode="raster"``, so the composite
       stage also downloads each tile's LULC raster and the delineate stage
       computes the crop statistics locally; the default
       ``lulc_mode="server"`` would query Earth Engine during delineation.
       The delineate stage then needs no Earth Engine access; model weights
       are downloaded on first use unless prefetched
       (``agribound tiles prefetch --manifest OUT``). Both stages are
       idempotent: done stages are skipped when the script is run again.
    4. ``tile_status`` reports every tile; ``merge_tiles`` keeps each polygon
       only in the tile that owns its representative point and writes one
       merged file with a provenance summary.

The same steps from the command line (after writing a base configuration,
e.g. ``agribound delineate --dry-run ... --lulc-mode raster > base.yaml``):

    agribound tiles make --config base.yaml --out-dir OUT --tile-size-m 2500 --halo-m 1000
    agribound tiles run --manifest OUT --index 0 --stage composite   # one per tile
    agribound tiles run --manifest OUT --index 0 --stage delineate
    agribound tiles status --manifest OUT
    agribound tiles merge --manifest OUT

Halo rule: a field is delineated whole only if it lies inside the halo of the
tile that owns it, so the halo must exceed the largest field dimension (the
default 1000 m is a minimum).

Data: Sentinel-2 L2A 2024, Delineate-Anything (``large_v2``). The LULC crop
filter (Dynamic World here, ``lulc_mode="raster"``) is on by default;
``--no-lulc-filter`` skips it.

Runtime (tests on 2026-09-27, Apple MPS): about 1 minute with
``--no-lulc-filter`` (4 tiles: ~9 s of composite download and ~5 s of
inference per tile) and about 80 s with the LULC filter (1,181 tile
polygons, 304 after the merge). A second run skips every finished stage
and takes about a second. In the LULC test the delineate stage was also
re-run with all network access blocked (weights already cached,
``HF_HUB_OFFLINE=1``) and gave the same tile outputs.

Prerequisites:
    pip install "agribound[gee,delineate-anything]"
    agribound auth --project YOUR_GEE_PROJECT
    Run from the repository root: python examples/19_hpc_tiling.py [--no-lulc-filter]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from agribound.config import AgriboundConfig
from agribound.hpc import make_tiles, merge_tiles, run_tile, tile_status, write_tile_manifest

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
OUTPUT_DIR = Path("outputs/hpc_tiling_demo")
STUDY_AREA = "bbox:1.44,48.12,1.50,48.16"  # about 4.5 x 4.5 km in the Beauce
TILE_SIZE_M = 2500.0
HALO_M = 1000.0
SOURCE = "sentinel2"
ENGINE = "delineate-anything"
YEAR = 2024


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="agribound.hpc tiling on a small AOI.")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument("--no-lulc-filter", action="store_true", help="Skip the LULC filter.")
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
    lulc_tag = "nolulc" if args.no_lulc_filter else "lulc"
    out_dir = OUTPUT_DIR / f"{SOURCE}_{ENGINE}_{YEAR}_{lulc_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tiles --------------------------------------------------------------------
    tiles = make_tiles(STUDY_AREA, tile_size_m=TILE_SIZE_M, halo_m=HALO_M, crs="utm")
    print(f"{len(tiles)} tiles:")
    print(tiles[["index", "tile_id", "utm_epsg", "core_area_km2", "study_area"]].to_string())

    # 2. Manifest and per-tile configurations ------------------------------------------
    base_config = AgriboundConfig(
        study_area=STUDY_AREA,  # replaced by each tile's halo box
        source=SOURCE,
        engine=ENGINE,
        year=YEAR,
        output_path=str(out_dir / "fields.gpkg"),  # replaced by tiles/<tile_id>/fields.gpkg
        gee_project=args.gee_project,
        lulc_filter=not args.no_lulc_filter,
        # With the filter on, download the LULC raster in the composite stage,
        # so the delineate stage needs no Earth Engine access.
        lulc_mode="server" if args.no_lulc_filter else "raster",
    )
    manifest = write_tile_manifest(tiles, base_config, out_dir)
    print(f"\nManifest: {manifest}")

    # 3. Two-phase execution: downloads for every tile, then delineation --------------------
    for stage in ("composite", "delineate"):
        for index in tiles["index"]:
            result = run_tile(manifest, int(index), stage=stage)
            print(f"  {stage:<9} tile {result['tile_id']}: {result['status']}")

    # 4. Status and merge -----------------------------------------------------------------
    status = tile_status(manifest)
    print(f"\n{status[['index', 'tile_id', 'composite', 'delineate', 'n_output']].to_string()}")
    merged = merge_tiles(manifest)
    summary = merged.attrs.get("merge_summary", {})
    print(f"\nMerged: {len(merged)} polygons")
    for key in (
        "n_polygons_in",
        "n_polygons_out",
        "n_reaching_halo_edge",
        "n_cross_tile_overlap_pairs",
    ):
        if key in summary:
            print(f"  {key}: {summary[key]}")

    from agribound.visualize import show_comparison

    web_map = show_comparison(
        [merged, tiles],
        labels=["Merged field boundaries", "Tile cores"],
        basemap="Esri.WorldImagery",
        output_html=str(out_dir / "map_merged_tiles.html"),
    )
    show_in_notebook(web_map)
    print(f"\nMap: {out_dir / 'map_merged_tiles.html'}")


if __name__ == "__main__":
    main()
