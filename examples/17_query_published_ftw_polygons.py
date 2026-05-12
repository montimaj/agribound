"""Query published FTW polygons for small AOIs.

This example is intentionally self-contained. It creates a tiny local
FTW-like GeoParquet tile store and manifest, then runs ``agribound.query_ftw``
against several AOIs. Replace ``manifest_path`` and ``tile_dir`` with a real
local FTW tile inventory to query published FTW polygons.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

import agribound as ab


def build_synthetic_ftw_store(base_dir: Path) -> tuple[Path, Path, Path]:
    """Create a tiny FTW-like tile store for a runnable local example."""
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agribound-ftw-query-") as tmp:
        workspace = Path(tmp)
        manifest_path, tile_dir, aoi_path = build_synthetic_ftw_store(workspace)

        clipped = ab.query_ftw(
            study_area=aoi_path,
            year=2025,
            label="field",
            clip=True,
            manifest_path=manifest_path,
            tile_dir=tile_dir,
            output_path=workspace / "ftw_demo_clipped.parquet",
        )
        print(f"Clipped AOI result: {len(clipped)} polygons")
        print(clipped[["field_id", "source_tile_id"]])

        full_polygons = ab.query_ftw(
            study_area=[-100.55, 40.30, -99.55, 40.75],
            year=2025,
            label="field",
            clip=False,
            manifest_path=manifest_path,
            tile_dir=tile_dir,
        )
        print(f"Unclipped AOI result: {len(full_polygons)} polygons")

        empty = ab.query_ftw(
            study_area=[-95.0, 35.0, -94.0, 36.0],
            year=2025,
            label="field",
            manifest_path=manifest_path,
            tile_dir=tile_dir,
        )
        print(f"Empty AOI result: {len(empty)} polygons")

        summary = {
            "manifest_path": str(manifest_path),
            "tile_dir": str(tile_dir),
            "aoi_path": str(aoi_path),
            "clipped_count": int(len(clipped)),
            "unclipped_count": int(len(full_polygons)),
            "empty_count": int(len(empty)),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
