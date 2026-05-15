"""Tests for querying published FTW polygon tiles."""

from __future__ import annotations

import json

import geopandas as gpd
import pytest
from click.testing import CliRunner
from shapely.geometry import box

pytest.importorskip("pyarrow")

from agribound.cli import main
from agribound.ftw_query import query_ftw


@pytest.fixture
def synthetic_ftw_store(tmp_path):
    """Create a tiny local FTW-like GeoParquet tile store and manifest."""
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()

    tile_a = gpd.GeoDataFrame(
        {
            "field_id": ["a1", "a2", "a3"],
            "geometry_hash": ["hash-a1", "hash-a2", "hash-a3"],
            "label": ["field", "non_field_background", "field"],
            "time": ["2025-01-01", "2025-01-01", "2024-01-01"],
        },
        geometry=[
            box(0.2, 0.2, 0.8, 0.8),
            box(0.3, 0.3, 0.7, 0.7),
            box(0.4, 0.4, 0.9, 0.9),
        ],
        crs="EPSG:4326",
    )
    tile_a_path = tile_dir / "tile_a.parquet"
    tile_a.to_parquet(tile_a_path, index=False)

    tile_b = gpd.GeoDataFrame(
        {
            "field_id": ["a1", "b1"],
            "geometry_hash": ["hash-a1-duplicate", "hash-b1"],
            "label": ["field", "field"],
            "time": ["2025-01-01", "2025-01-01"],
        },
        geometry=[
            box(0.2, 0.2, 0.8, 0.8),
            box(1.1, 0.1, 1.8, 0.8),
        ],
        crs="EPSG:4326",
    )
    tile_b_path = tile_dir / "tile_b.parquet"
    tile_b.to_parquet(tile_b_path, index=False)

    # This deliberately is not a valid parquet file. A bbox query outside it
    # should not attempt to read it.
    bad_tile_path = tile_dir / "bad_far_tile.parquet"
    bad_tile_path.write_text("not parquet")

    manifest = gpd.GeoDataFrame(
        {
            "tile_id": ["tile_a", "tile_b", "bad_far_tile"],
            # Match the local workflow notebooks, where the tile manifest stores
            # output Parquet locations in an out_path column.
            "out_path": [
                tile_a_path.name,
                tile_b_path.name,
                bad_tile_path.name,
            ],
            "status": ["ok", "ok", "skipped_no_candidates"],
        },
        geometry=[
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(10.0, 10.0, 11.0, 11.0),
        ],
        crs="EPSG:4326",
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    aoi_geojson = tmp_path / "aoi.geojson"
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]],
                },
            }
        ],
    }
    aoi_geojson.write_text(json.dumps(aoi))

    return {
        "tile_dir": tile_dir,
        "manifest_path": manifest_path,
        "aoi_geojson": aoi_geojson,
    }


def test_aoi_bbox_selects_only_intersecting_tiles(synthetic_ftw_store):
    out = query_ftw(
        study_area=[0.0, 0.0, 1.0, 1.0],
        year=2025,
        label="field",
        clip=False,
        manifest_path=synthetic_ftw_store["manifest_path"],
        tile_dir=synthetic_ftw_store["tile_dir"],
    )

    assert not out.empty
    assert "bad_far_tile" not in set(out.get("source_tile_id", []))
    assert set(out["source_tile_id"]) == {"tile_a"}


def test_label_filters_field_class(synthetic_ftw_store):
    out = query_ftw(
        study_area=[0.0, 0.0, 1.0, 1.0],
        label="field",
        clip=False,
        manifest_path=synthetic_ftw_store["manifest_path"],
        tile_dir=synthetic_ftw_store["tile_dir"],
    )

    assert set(out["label"]) == {"field"}
    assert "a2" not in set(out["field_id"])


def test_year_filters_time_column(synthetic_ftw_store):
    out = query_ftw(
        study_area=[0.0, 0.0, 1.0, 1.0],
        year=2025,
        label="field",
        clip=False,
        manifest_path=synthetic_ftw_store["manifest_path"],
        tile_dir=synthetic_ftw_store["tile_dir"],
    )

    assert set(out["field_id"]) == {"a1"}
    assert all(str(value).startswith("2025") for value in out["time"])


def test_duplicate_field_ids_removed(synthetic_ftw_store):
    out = query_ftw(
        study_area=[0.0, 0.0, 2.0, 1.0],
        year=2025,
        label="field",
        clip=False,
        manifest_path=synthetic_ftw_store["manifest_path"],
        tile_dir=synthetic_ftw_store["tile_dir"],
    )

    assert out["field_id"].tolist().count("a1") == 1
    assert set(out["field_id"]) == {"a1", "b1"}


def test_clip_true_clips_geometry(synthetic_ftw_store):
    out = query_ftw(
        study_area=[0.0, 0.0, 0.5, 0.5],
        year=2025,
        label="field",
        clip=True,
        manifest_path=synthetic_ftw_store["manifest_path"],
        tile_dir=synthetic_ftw_store["tile_dir"],
    )

    assert len(out) == 1
    assert out.geometry.iloc[0].area == pytest.approx(0.09)


def test_clip_false_returns_full_intersecting_polygon(synthetic_ftw_store):
    out = query_ftw(
        study_area=[0.0, 0.0, 0.5, 0.5],
        year=2025,
        label="field",
        clip=False,
        manifest_path=synthetic_ftw_store["manifest_path"],
        tile_dir=synthetic_ftw_store["tile_dir"],
    )

    assert len(out) == 1
    assert out.geometry.iloc[0].area == pytest.approx(0.36)


def test_output_path_writes_geoparquet(synthetic_ftw_store, tmp_path):
    output_path = tmp_path / "ftw_aoi.parquet"
    out = query_ftw(
        study_area=[0.0, 0.0, 1.0, 1.0],
        year=2025,
        label="field",
        clip=True,
        output_path=output_path,
        manifest_path=synthetic_ftw_store["manifest_path"],
        tile_dir=synthetic_ftw_store["tile_dir"],
    )

    assert output_path.exists()
    loaded = gpd.read_parquet(output_path)
    assert len(loaded) == len(out)
    assert loaded.crs is not None


def test_empty_aoi_result_has_expected_schema(synthetic_ftw_store):
    out = query_ftw(
        study_area=[20.0, 20.0, 21.0, 21.0],
        year=2025,
        label="field",
        manifest_path=synthetic_ftw_store["manifest_path"],
        tile_dir=synthetic_ftw_store["tile_dir"],
    )

    assert out.empty
    assert out.crs.to_epsg() == 4326
    assert "geometry" in out.columns
    assert {"field_id", "geometry_hash", "label", "time", "year", "source_tile_id"}.issubset(
        out.columns
    )


def test_query_ftw_cli_writes_output(synthetic_ftw_store, tmp_path):
    output_path = tmp_path / "ftw_cli.parquet"
    result = CliRunner().invoke(
        main,
        [
            "query-ftw",
            "--study-area",
            str(synthetic_ftw_store["aoi_geojson"]),
            "--year",
            "2025",
            "--label",
            "field",
            "--clip",
            "--manifest-path",
            str(synthetic_ftw_store["manifest_path"]),
            "--tile-dir",
            str(synthetic_ftw_store["tile_dir"]),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "published FTW polygons" in result.output


@pytest.fixture
def synthetic_ftw_pyarrow_dataset(tmp_path):
    """Create tiny local GeoParquet files for the PyArrow backend."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import to_wkb

    dataset_dir = tmp_path / "arrow_dataset"
    dataset_dir.mkdir()

    geometries = [
        box(0.2, 0.2, 0.8, 0.8),
        box(0.3, 0.3, 0.7, 0.7),
        box(0.4, 0.4, 0.9, 0.9),
        box(1.1, 0.1, 1.8, 0.8),
    ]
    bbox_type = pa.struct(
        [
            pa.field("xmin", pa.float64()),
            pa.field("ymin", pa.float64()),
            pa.field("xmax", pa.float64()),
            pa.field("ymax", pa.float64()),
        ]
    )
    table = pa.table(
        {
            "field_id": pa.array(["pa1", "pa2", "pa-old", "pb1"], type=pa.string()),
            "geometry_hash": pa.array(
                ["hash-pa1", "hash-pa2", "hash-pa-old", "hash-pb1"],
                type=pa.string(),
            ),
            "label": pa.array(
                ["field", "non_field_background", "field", "field"],
                type=pa.string(),
            ),
            "time": pa.array(
                ["2025-01-01", "2025-01-01", "2024-01-01", "2025-01-01"],
                type=pa.string(),
            ),
            "bbox": pa.array(
                [
                    {"xmin": 0.2, "ymin": 0.2, "xmax": 0.8, "ymax": 0.8},
                    {"xmin": 0.3, "ymin": 0.3, "xmax": 0.7, "ymax": 0.7},
                    {"xmin": 0.4, "ymin": 0.4, "xmax": 0.9, "ymax": 0.9},
                    {"xmin": 1.1, "ymin": 0.1, "xmax": 1.8, "ymax": 0.8},
                ],
                type=bbox_type,
            ),
            "geometry": pa.array([to_wkb(geom) for geom in geometries], type=pa.binary()),
        }
    )
    pq.write_table(table.slice(0, 3), dataset_dir / "part_a.parquet")
    pq.write_table(table.slice(3, 1), dataset_dir / "part_b.parquet")

    flat_dir = tmp_path / "arrow_flat_dataset"
    flat_dir.mkdir()
    flat = pa.table(
        {
            "field_id": pa.array(["flat1"], type=pa.string()),
            "label": pa.array(["field"], type=pa.string()),
            "time": pa.array(["2025-01-01"], type=pa.string()),
            "xmin": pa.array([2.2], type=pa.float64()),
            "ymin": pa.array([0.2], type=pa.float64()),
            "xmax": pa.array([2.8], type=pa.float64()),
            "ymax": pa.array([0.8], type=pa.float64()),
            "geometry": pa.array([to_wkb(box(2.2, 0.2, 2.8, 0.8))], type=pa.binary()),
        }
    )
    pq.write_table(flat, flat_dir / "flat.parquet")

    return {
        "source_glob": str(dataset_dir / "*.parquet"),
        "flat_glob": str(flat_dir / "*.parquet"),
    }


def test_pyarrow_backend_bbox_label_year_filter(synthetic_ftw_pyarrow_dataset):
    out = query_ftw(
        study_area=[0.0, 0.0, 1.0, 1.0],
        year=2025,
        label="field",
        clip=False,
        source_backend="pyarrow",
        source_url=synthetic_ftw_pyarrow_dataset["source_glob"],
    )

    assert set(out["field_id"]) == {"pa1"}


def test_pyarrow_backend_auto_for_parquet_glob(synthetic_ftw_pyarrow_dataset):
    out = query_ftw(
        study_area=[0.0, 0.0, 1.0, 1.0],
        year=2025,
        label="field",
        clip=False,
        source_url=synthetic_ftw_pyarrow_dataset["source_glob"],
    )

    assert set(out["field_id"]) == {"pa1"}


def test_pyarrow_backend_clip_true(synthetic_ftw_pyarrow_dataset):
    out = query_ftw(
        study_area=[0.0, 0.0, 0.5, 0.5],
        year=2025,
        label="field",
        clip=True,
        source_backend="pyarrow",
        source_url=synthetic_ftw_pyarrow_dataset["source_glob"],
    )

    assert len(out) == 1
    assert out.geometry.iloc[0].area == pytest.approx(0.09)


def test_pyarrow_backend_clip_false(synthetic_ftw_pyarrow_dataset):
    out = query_ftw(
        study_area=[0.0, 0.0, 0.5, 0.5],
        year=2025,
        label="field",
        clip=False,
        source_backend="pyarrow",
        source_url=synthetic_ftw_pyarrow_dataset["source_glob"],
    )

    assert len(out) == 1
    assert out.geometry.iloc[0].area == pytest.approx(0.36)


def test_pyarrow_backend_empty_result(synthetic_ftw_pyarrow_dataset):
    out = query_ftw(
        study_area=[10.0, 10.0, 11.0, 11.0],
        year=2025,
        label="field",
        source_backend="pyarrow",
        source_url=synthetic_ftw_pyarrow_dataset["source_glob"],
    )

    assert out.empty
    assert out.crs.to_epsg() == 4326


def test_pyarrow_backend_output_path(synthetic_ftw_pyarrow_dataset, tmp_path):
    output_path = tmp_path / "arrow_output.parquet"
    out = query_ftw(
        study_area=[0.0, 0.0, 1.0, 1.0],
        year=2025,
        label="field",
        clip=True,
        source_backend="pyarrow",
        source_url=synthetic_ftw_pyarrow_dataset["source_glob"],
        output_path=output_path,
    )

    assert output_path.exists()
    loaded = gpd.read_parquet(output_path)
    assert len(loaded) == len(out)


def test_pyarrow_backend_flat_bbox_columns(synthetic_ftw_pyarrow_dataset):
    out = query_ftw(
        study_area=[2.0, 0.0, 3.0, 1.0],
        year=2025,
        label="field",
        source_backend="pyarrow",
        source_url=synthetic_ftw_pyarrow_dataset["flat_glob"],
    )

    assert set(out["field_id"]) == {"flat1"}


def test_pyarrow_backend_max_features(synthetic_ftw_pyarrow_dataset):
    out = query_ftw(
        study_area=[0.0, 0.0, 2.0, 1.0],
        year=2025,
        label="field",
        source_backend="pyarrow",
        source_url=synthetic_ftw_pyarrow_dataset["source_glob"],
        max_features=1,
    )

    assert len(out) <= 1


@pytest.mark.skipif(
    __import__("os").environ.get("AGRIBOUND_RUN_LIVE_FTW_TESTS") != "1",
    reason="Live public FTW test is skipped unless AGRIBOUND_RUN_LIVE_FTW_TESTS=1.",
)
def test_live_public_ftw_pyarrow_smoke(tmp_path):
    out = query_ftw(
        study_area=[-93.55, 41.90, -93.50, 41.95],
        year=2025,
        label="field",
        clip=True,
        output_path=tmp_path / "ftw_live_smoke.parquet",
        max_features=1000,
    )

    assert out.crs.to_epsg() == 4326
