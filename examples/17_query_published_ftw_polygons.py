"""
17 — Query Published FTW Polygons (offline demo with a local tile store)

``agribound.query_ftw`` reads already-published Fields of The World (FTW)
prediction polygons for a study area; it does not run FTW inference, and the
polygons are model predictions, not ground truth.

This example is self-contained and needs no network: it writes a tiny
FTW-like GeoParquet tile store and a tile manifest to a temporary directory
and queries it with the ``manifest`` backend (clipped, unclipped and an empty
AOI). The synthetic tiles use the legacy ``raw`` layout columns (``label``,
``time``).

Live data: without ``manifest_path``/``tile_dir``, ``query_ftw`` reads the
public Source Cooperative GeoParquet with PyArrow. The default layout
``"by-admin-conf"`` (``alpha/results-by-admin-conf``, partitioned by country
and subdivision) holds 2024 and 2025 predictions with a ``confidence`` column
(0-100); the dataset README recommends ``min_confidence=69``. Null confidence
means the confidence raster has no data at the field, not a low score, and
it is common: on 2026-09-27 it was null for all rows of the New Mexico
partition and 99.7 % of New South Wales. See example 21 for a live query and
an evaluation against reference boundaries.

Runtime: about a second (tested 2026-09-27).

Prerequisites:
    pip install agribound
    Run from the repository root: python examples/17_query_published_ftw_polygons.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

import agribound as ab

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

OUTPUT_DIR = Path("outputs/ftw_query_demo")


def build_synthetic_ftw_store(base_dir: Path) -> tuple[Path, Path, Path]:
    """Create a tiny FTW-like tile store (two tiles) and its manifest."""
    tile_dir = base_dir / "ftw_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    tile_west = gpd.GeoDataFrame(
        {
            "field_id": ["west-1", "west-2", "old-west"],
            "geometry_hash": ["hash-west-1", "hash-west-2", "hash-old-west"],
            "label": ["field", "field", "field"],
            "time": ["2025-01-01", "2025-01-01", "2024-01-01"],
        },
        geometry=[
            box(-100.80, 40.10, -100.25, 40.65),
            box(-100.45, 40.55, -100.05, 40.90),
            box(-100.70, 40.20, -100.35, 40.45),
        ],
        crs="EPSG:4326",
    )
    tile_west_path = tile_dir / "tile_west.parquet"
    tile_west.to_parquet(tile_west_path, index=False)

    tile_east = gpd.GeoDataFrame(
        {
            "field_id": ["east-1", "east-2", "west-1"],
            "geometry_hash": ["hash-east-1", "hash-east-2", "hash-west-1-dup"],
            "label": ["field", "non_field_background", "field"],
            "time": ["2025-01-01", "2025-01-01", "2025-01-01"],
        },
        geometry=[
            box(-99.90, 40.15, -99.20, 40.80),
            box(-99.70, 40.25, -99.35, 40.55),
            box(-100.80, 40.10, -100.25, 40.65),
        ],
        crs="EPSG:4326",
    )
    tile_east_path = tile_dir / "tile_east.parquet"
    tile_east.to_parquet(tile_east_path, index=False)

    manifest = gpd.GeoDataFrame(
        {
            "tile_id": ["tile_west", "tile_east"],
            "out_path": [tile_west_path.name, tile_east_path.name],
            "status": ["ok", "ok"],
        },
        geometry=[box(-101.0, 40.0, -100.0, 41.0), box(-100.0, 40.0, -99.0, 41.0)],
        crs="EPSG:4326",
    )
    manifest_path = base_dir / "ftw_tile_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    aoi_path = base_dir / "demo_aoi.geojson"
    aoi = gpd.GeoDataFrame(
        {"name": ["cross_tile_aoi"]},
        geometry=[box(-100.55, 40.30, -99.55, 40.75)],
        crs="EPSG:4326",
    )
    aoi.to_file(aoi_path, driver="GeoJSON")

    return manifest_path, tile_dir, aoi_path


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agribound-ftw-query-") as tmp:
        workspace = Path(tmp)
        manifest_path, tile_dir, aoi_path = build_synthetic_ftw_store(workspace)
        backend = dict(source_backend="manifest", manifest_path=manifest_path, tile_dir=tile_dir)

        clipped = ab.query_ftw(
            study_area=aoi_path,
            year=2025,
            label="field",
            clip=True,
            output_path=OUTPUT_DIR / "ftw_demo_clipped.parquet",
            **backend,
        )
        print(f"Clipped AOI result: {len(clipped)} polygons")
        print(clipped[["field_id", "source_tile_id"]])

        full_polygons = ab.query_ftw(
            study_area=[-100.55, 40.30, -99.55, 40.75],
            year=2025,
            label="field",
            clip=False,
            **backend,
        )
        print(f"Unclipped AOI result: {len(full_polygons)} polygons")

        empty = ab.query_ftw(
            study_area=[-95.0, 35.0, -94.0, 36.0], year=2025, label="field", **backend
        )
        print(f"Empty AOI result: {len(empty)} polygons")

        summary = {
            "clipped_count": int(len(clipped)),
            "unclipped_count": int(len(full_polygons)),
            "empty_count": int(len(empty)),
        }
        print(json.dumps(summary, indent=2))

    # Live public FTW polygons (network; ~20-30 s for a small AOI). Uncomment:
    #
    # live = ab.query_ftw(
    #     study_area=[-106.80, 34.60, -106.75, 34.65],  # near Belen, New Mexico
    #     year=2024,
    #     min_confidence=69,  # recommended by the dataset README
    #     keep_null_confidence=True,  # the US_NM partition has no confidence values
    #     output_path=OUTPUT_DIR / "ftw_live_aoi.parquet",
    # )
    # print(live.attrs["ftw_query"])

    web_map = ab.show_boundaries(clipped, output_html=str(OUTPUT_DIR / "map_ftw_demo_clipped.html"))
    show_in_notebook(web_map)
    print(f"\nMap of the clipped synthetic polygons: {OUTPUT_DIR / 'map_ftw_demo_clipped.html'}")


if __name__ == "__main__":
    main()
