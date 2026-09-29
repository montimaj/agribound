"""Tests for querying published FTW polygon tiles."""

from __future__ import annotations

import json
import sys

import geopandas as gpd
import pytest
from click.testing import CliRunner
from shapely.geometry import box

pytest.importorskip("pyarrow")

from agribound.cli import main
from agribound.ftw_query import query_ftw


def _country_partition(root, code):
    """``root/admin:country_code=<code>``, the published store's hive layout.

    Windows paths cannot contain ':', so tests that need a local copy of the layout
    are skipped there (the S3 keys themselves are read fine on Windows).
    """
    if sys.platform == "win32":
        pytest.skip("hive partition directories named 'admin:country_code=...' need ':' in a path")
    part = root / f"admin:country_code={code}"
    part.mkdir(parents=True)
    return part


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


@pytest.mark.network
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
        cache_dir=tmp_path / "index",
    )

    assert out.crs.to_epsg() == 4326
    assert len(out) > 0  # Iowa cropland (US_IA partition)
    assert (out["determination:datetime"].dt.year == 2025).all()
    assert (tmp_path / "ftw_live_smoke.parquet").exists()


# ---------------------------------------------------------------------------
# by-admin-conf layout (agribound 1.0): determination:datetime, confidence, partitions
# ---------------------------------------------------------------------------


@pytest.fixture
def by_admin_conf_store(tmp_path):
    """Hive-partitioned GeoParquet mimicking alpha/results-by-admin-conf."""
    import datetime as dt

    import pandas as pd

    root = tmp_path / "results-by-admin-conf"
    y2024 = pd.Timestamp(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
    y2025 = pd.Timestamp(dt.datetime(2025, 1, 1, tzinfo=dt.UTC))
    au = gpd.GeoDataFrame(
        {
            "id": ["1", "1", "2", "3", "4", "4"],
            "determination:datetime": [y2024, y2024, y2024, y2024, y2024, y2025],
            "determination:method": ["auto-imagery"] * 6,
            "confidence": [80.0, 80.0, 50.0, None, 90.0, 70.0],
            "admin:country_code": ["AU"] * 6,
            "admin:subdivision_code": ["NSW"] * 6,
        },
        geometry=[
            box(149.700, -30.400, 149.705, -30.395),
            box(149.700, -30.400, 149.705, -30.395),  # exact duplicate row of id 1
            box(149.706, -30.400, 149.710, -30.395),
            box(149.711, -30.400, 149.715, -30.395),
            box(149.700, -30.390, 149.705, -30.385),
            box(149.700, -30.390, 149.705, -30.385),  # id 4 predicted again in 2025
        ],
        crs="EPSG:4326",
    )
    us = gpd.GeoDataFrame(
        {
            "id": ["10"],
            "determination:datetime": [y2024],
            "determination:method": ["auto-imagery"],
            "confidence": [95.0],
            "admin:country_code": ["US"],
            "admin:subdivision_code": ["NM"],
        },
        geometry=[box(-106.79, 34.61, -106.78, 34.62)],
        crs="EPSG:4326",
    )
    for code, name, gdf in (("AU", "AU_NSW", au), ("US", "US_NM", us)):
        part = _country_partition(root, code)
        gdf.to_parquet(part / f"{name}.parquet", write_covering_bbox=True)
    return root


def _q(store, **kwargs):
    defaults = {
        "study_area": [149.69, -30.41, 149.72, -30.38],
        "clip": False,
        "source_backend": "pyarrow",
        "source_url": str(store),
    }
    defaults.update(kwargs)
    return query_ftw(**defaults)


def test_query_ftw_signature_defaults():
    import inspect

    params = inspect.signature(query_ftw).parameters
    assert params["min_confidence"].default is None
    assert params["keep_null_confidence"].default is True
    assert params["layout"].default == "by-admin-conf"


def test_by_admin_year_filter_uses_determination_datetime(by_admin_conf_store):
    out_2024 = _q(by_admin_conf_store, year=2024)
    assert sorted(out_2024["id"]) == ["1", "2", "3", "4"]
    out_2025 = _q(by_admin_conf_store, year=2025)
    assert out_2025["id"].tolist() == ["4"]
    assert "bbox" not in out_2024.columns  # struct column dropped unless requested


def test_by_admin_dedup_on_geometry_and_year(by_admin_conf_store):
    everything = _q(by_admin_conf_store)
    assert everything["id"].tolist().count("1") == 1  # exact duplicate removed
    assert everything["id"].tolist().count("4") == 2  # 2024 and 2025 kept separately
    assert everything.attrs["ftw_query"]["n_duplicates_dropped"] == 1
    raw = _q(by_admin_conf_store, deduplicate=False)
    assert len(raw) == len(everything) + 1


@pytest.fixture
def shared_id_store(tmp_path):
    """by-admin-conf partition in which one ``id`` labels different polygons.

    Mirrors ``US_NM`` (2026-09-27): 190,940 ids in 2024 are each shared by 2-4
    different polygons, a median ~324 km apart, and no geometry repeats.
    """
    import datetime as dt

    import pandas as pd
    from shapely.geometry import Polygon

    y2024 = pd.Timestamp(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
    near = box(-106.79, 34.61, -106.78, 34.62)
    # The same ring as ``near``, starting at another vertex and wound the other way.
    near_reordered = Polygon(
        [(-106.78, 34.62), (-106.78, 34.61), (-106.79, 34.61), (-106.79, 34.62)]
    )
    gdf = gpd.GeoDataFrame(
        {
            "id": ["7", "7", "7", "8", "9"],
            "determination:datetime": [y2024] * 5,
            "confidence": [None] * 5,
            "admin:country_code": ["US"] * 5,
            "admin:subdivision_code": ["NM"] * 5,
        },
        geometry=[
            near,
            box(-106.77, 34.61, -106.76, 34.62),  # another polygon, same id
            box(-106.75, 34.63, -106.74, 34.64),  # a third polygon, same id
            near_reordered,  # same polygon as the first row, different id
            None,  # rows without geometry are always kept
        ],
        crs="EPSG:4326",
    )
    part = _country_partition(tmp_path / "results-by-admin-conf", "US")
    gdf.to_parquet(part / "US_NM.parquet", write_covering_bbox=True)
    return tmp_path / "results-by-admin-conf"


def test_dedup_keeps_distinct_polygons_sharing_an_id(shared_id_store, caplog):
    kwargs = {
        "study_area": [-106.80, 34.60, -106.73, 34.65],
        "clip": False,
        "source_backend": "pyarrow",
        "source_url": str(shared_id_store),
        "year": 2024,
    }
    raw = query_ftw(**kwargs, deduplicate=False)
    with caplog.at_level("WARNING", logger="agribound.ftw_query"):
        out = query_ftw(**kwargs)
    present = out[out.geometry.notna()]
    # The three polygons with id 7 are all kept; only the repeated geometry (id 8) goes.
    assert sorted(present["id"]) == ["7", "7", "7"]
    assert out.attrs["ftw_query"]["n_duplicates_dropped"] == len(raw) - len(out) == 1
    assert "occur with different geometries" not in caplog.text


def test_geometry_hashes_match_per_geometry_hash():
    from shapely.geometry import Polygon

    from agribound.ftw_query import _geometry_hashes, _stable_geometry_hash

    geoms = gpd.GeoSeries(
        [
            box(0, 0, 1, 1),
            Polygon([(1, 1), (1, 0), (0, 0), (0, 1)]),
            None,
            Polygon(),
            box(2, 2, 3, 3),
        ],
        index=[10, 11, 12, 13, 14],
    )
    hashes = _geometry_hashes(geoms)
    assert hashes.index.tolist() == [10, 11, 12, 13, 14]
    assert hashes.tolist() == [_stable_geometry_hash(g) for g in geoms]
    assert hashes[10] == hashes[11]  # normalization ignores vertex order and winding
    assert hashes[12] is None and hashes[13] is None
    assert hashes[14] != hashes[10]


def test_manifest_backend_keeps_different_geometries_with_same_field_id(tmp_path):
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    for name, geom in (("t1", box(0.1, 0.1, 0.4, 0.4)), ("t2", box(0.5, 0.5, 0.9, 0.9))):
        gpd.GeoDataFrame(
            {"field_id": ["same"], "label": ["field"], "time": ["2025-01-01"]},
            geometry=[geom],
            crs="EPSG:4326",
        ).to_parquet(tile_dir / f"{name}.parquet", index=False)
    out = query_ftw(study_area=[0.0, 0.0, 1.0, 1.0], clip=False, tile_dir=tile_dir)
    assert out["field_id"].tolist() == ["same", "same"]


def test_by_admin_confidence_filter_keeps_or_drops_null(by_admin_conf_store):
    kept = _q(by_admin_conf_store, year=2024, min_confidence=69)
    assert sorted(kept["id"]) == ["1", "3", "4"]  # 3 has null confidence
    dropped = _q(by_admin_conf_store, year=2024, min_confidence=69, keep_null_confidence=False)
    assert sorted(dropped["id"]) == ["1", "4"]
    no_null = _q(by_admin_conf_store, year=2024, keep_null_confidence=False)
    assert "3" not in set(no_null["id"])


def test_by_admin_partition_pruning(by_admin_conf_store):
    out = _q(by_admin_conf_store, year=2024)
    info = out.attrs["ftw_query"]
    assert info["n_files_listed"] == 2 and info["n_files_opened"] == 1


def test_hive_glob_source(by_admin_conf_store):
    glob = str(by_admin_conf_store / "admin:country_code=*" / "*.parquet")
    out = _q(by_admin_conf_store, source_url=glob, year=2024)
    assert sorted(out["id"]) == ["1", "2", "3", "4"]
    nm = _q(
        by_admin_conf_store,
        source_url=glob,
        study_area="bbox:-106.80,34.60,-106.75,34.65",
        year=2024,
    )
    assert nm["id"].tolist() == ["10"]


def test_confidence_filter_requires_confidence_column(synthetic_ftw_pyarrow_dataset):
    with pytest.raises(ValueError, match="confidence"):
        query_ftw(
            study_area=[0.0, 0.0, 1.0, 1.0],
            source_backend="pyarrow",
            source_url=synthetic_ftw_pyarrow_dataset["source_glob"],
            min_confidence=69,
        )


def test_min_confidence_scale_checks(caplog):
    from agribound.ftw_arrow import validate_min_confidence

    with pytest.raises(ValueError, match="0-100"):
        validate_min_confidence(150)
    with caplog.at_level("WARNING"):
        assert validate_min_confidence(0.4) == 0.4
    assert "0-100 scale" in caplog.text
    assert validate_min_confidence(None) is None


def test_manifest_backend_confidence_filter(tmp_path):
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    gpd.GeoDataFrame(
        {"id": ["a", "b", "c"], "confidence": [90.0, 10.0, None]},
        geometry=[box(0.1, 0.1, 0.2, 0.2), box(0.3, 0.3, 0.4, 0.4), box(0.5, 0.5, 0.6, 0.6)],
        crs="EPSG:4326",
    ).to_parquet(tile_dir / "t.parquet")
    out = query_ftw(
        study_area=[0.0, 0.0, 1.0, 1.0], clip=False, tile_dir=tile_dir, min_confidence=69
    )
    assert sorted(out["id"]) == ["a", "c"]


def test_default_layout_warns_for_unpublished_year(monkeypatch, caplog):
    import agribound.ftw_query as fq

    seen = {}

    def fake_arrow(**kwargs):
        seen.update(kwargs)
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    monkeypatch.setattr(fq, "query_ftw_arrow", fake_arrow)
    with caplog.at_level("WARNING"):
        out = fq.query_ftw(study_area=[0, 0, 1, 1], year=2023, cache_dir="/tmp/idx")
    assert "2024, 2025" in caplog.text
    assert seen["layout"] == "by-admin-conf" and seen["source_url"] is None
    assert seen["index_cache_dir"] == "/tmp/idx"
    assert {"id", "confidence", "determination:datetime"}.issubset(out.columns)


@pytest.mark.parametrize(
    ("source_url", "warns"),
    [
        (
            "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/predictions/vectors/"
            "alpha/results-by-admin-conf/admin:country_code=AU/AU_NSW.parquet",
            True,
        ),
        (
            "https://data.source.coop/ftw/global-data/predictions/vectors/alpha/"
            "results-by-admin-conf/",
            True,
        ),
        (
            "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/predictions/vectors/"
            "alpha/results/",
            False,
        ),
        ("/data/my_ftw/*.parquet", False),
    ],
)
def test_explicit_by_admin_conf_source_warns_for_unpublished_year(
    monkeypatch, caplog, source_url, warns
):
    import agribound.ftw_query as fq

    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    monkeypatch.setattr(fq, "query_ftw_arrow", lambda **kwargs: empty)
    with caplog.at_level("WARNING"):
        fq.query_ftw(study_area=[0, 0, 1, 1], year=2023, source_url=source_url)
    assert ("2024, 2025" in caplog.text) is warns


def test_failed_footer_read_is_retried_instead_of_cached(
    by_admin_conf_store, tmp_path, monkeypatch
):
    import json

    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    from agribound import ftw_arrow

    nsw = by_admin_conf_store / "admin:country_code=AU" / "AU_NSW.parquet"
    stat = nsw.stat()
    entry = ftw_arrow._FileEntry(str(nsw), stat.st_size, str(stat.st_mtime))
    filesystem = pafs.LocalFileSystem()  # any filesystem object marks the source as remote
    real = pq.read_metadata

    def timed_out(*args, **kwargs):
        raise OSError("AWS Error NETWORK_CONNECTION: timed out")

    monkeypatch.setattr(pq, "read_metadata", timed_out)
    bounds = ftw_arrow._file_bounds(filesystem, [entry], "s3://bucket/x/", tmp_path)
    assert bounds[str(nsw)] is None  # scanned this time, nothing lost
    (index,) = tmp_path.glob("partition_index_*.json")
    assert str(nsw) not in json.loads(index.read_text())  # not cached as "no bbox"
    monkeypatch.setattr(pq, "read_metadata", real)
    bounds = ftw_arrow._file_bounds(filesystem, [entry], "s3://bucket/x/", tmp_path)
    assert bounds[str(nsw)] == pytest.approx([149.700, -30.400, 149.715, -30.385])
    record = json.loads(index.read_text())[str(nsw)]
    assert record["bbox_source"] == "row_group_stats"


def test_source_coop_path_normalisation_and_globs():
    from agribound.ftw_arrow import (
        FTW_GLOBAL_DATA_S3,
        _glob_match,
        _normalize_source_coop_path,
        _strip_parquet_glob,
    )

    tail = "predictions/vectors/alpha/results-by-admin-conf/admin:country_code=*/*.parquet"
    assert _normalize_source_coop_path("https://data.source.coop/ftw/global-data/" + tail) == (
        FTW_GLOBAL_DATA_S3 + tail
    )
    assert _normalize_source_coop_path(
        "s3://us-west-2.opendata.source.coop/ftw/global-data/" + tail
    ) == (FTW_GLOBAL_DATA_S3 + tail)
    assert _normalize_source_coop_path("/local/x.parquet") == "/local/x.parquet"
    path = "bucket/a/results-by-admin-conf/admin:country_code=*/*.parquet"
    assert _strip_parquet_glob(path) == "bucket/a/results-by-admin-conf"
    assert _strip_parquet_glob("bucket/a/results/*.parquet") == "bucket/a/results"
    assert _strip_parquet_glob("bucket/a/results/") == "bucket/a/results"
    assert _glob_match("bucket/a/results-by-admin-conf/admin:country_code=AU/AU_NSW.parquet", path)
    assert not _glob_match("bucket/a/results-by-admin-conf/extra/x/AU.parquet", path)


def _rewrite_geo_bbox(path, bbox, write_statistics=True):
    """Rewrite a GeoParquet file with a different 'geo' metadata bbox (as published)."""
    import json

    import pyarrow.parquet as pq

    table = pq.ParquetFile(path).read()  # no hive partition inference from the directory
    meta = dict(table.schema.metadata)
    geo = json.loads(meta[b"geo"])
    geo["columns"]["geometry"]["bbox"] = list(bbox)
    meta[b"geo"] = json.dumps(geo).encode()
    pq.write_table(table.replace_schema_metadata(meta), path, write_statistics=write_statistics)


def test_pruning_uses_row_group_stats_not_wrong_geo_bbox(by_admin_conf_store, caplog):
    """Published subdivided countries share one (wrong) 'geo' bbox across their files."""
    import pyarrow.parquet as pq

    from agribound.ftw_arrow import footer_bbox

    act = [148.75, -35.94, 149.39, -35.16]  # the ACT extent, as in every AU file on S3
    nsw = by_admin_conf_store / "admin:country_code=AU" / "AU_NSW.parquet"
    _rewrite_geo_bbox(nsw, act)
    bbox, source, mismatch = footer_bbox(pq.read_metadata(nsw))
    assert source == "row_group_stats" and mismatch is True
    assert bbox == pytest.approx([149.700, -30.400, 149.715, -30.385])
    with caplog.at_level("WARNING"):
        out = _q(by_admin_conf_store, year=2024)
    assert sorted(out["id"]) == ["1", "2", "3", "4"]  # not pruned away
    assert out.attrs["ftw_query"]["n_files_opened"] == 1
    assert "does not contain their data" in caplog.text


def test_pruning_falls_back_to_geo_bbox_without_statistics(by_admin_conf_store):
    import pyarrow.parquet as pq

    from agribound.ftw_arrow import footer_bbox

    nsw = by_admin_conf_store / "admin:country_code=AU" / "AU_NSW.parquet"
    _rewrite_geo_bbox(nsw, [149.0, -31.0, 150.0, -30.0], write_statistics=False)
    assert footer_bbox(pq.read_metadata(nsw)) == (
        [149.0, -31.0, 150.0, -30.0],
        "geo_metadata",
        False,
    )
    far = [140.0, -40.0, 141.0, -39.0]
    _rewrite_geo_bbox(nsw, far, write_statistics=False)
    out = _q(by_admin_conf_store, year=2024)  # geo bbox is all there is: file pruned
    assert out.empty and out.attrs["ftw_query"]["n_files_opened"] == 0


def test_min_confidence_warns_when_null_confidence_is_kept(by_admin_conf_store, caplog):
    with caplog.at_level("WARNING"):
        out = _q(by_admin_conf_store, year=2024, min_confidence=69)
    assert out.attrs["ftw_query"]["n_null_confidence"] == 1  # id 3
    assert "keep_null_confidence=False" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        _q(by_admin_conf_store, year=2024, min_confidence=69, keep_null_confidence=False)
        _q(by_admin_conf_store, year=2024)  # no threshold: nothing to warn about
    assert "no confidence value" not in caplog.text


def test_present_handles_missing_and_empty_geometries_without_warnings():
    import warnings

    from shapely.geometry import Polygon

    from agribound.ftw_query import _present

    series = gpd.GeoSeries([None, Polygon(), box(0, 0, 1, 1)], crs="EPSG:4326")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _present(series).tolist() == [False, False, True]


@pytest.mark.network
@pytest.mark.parametrize(
    ("bounds", "partition"),
    [((149.70, -30.40, 149.72, -30.38), "AU_NSW"), ((-106.80, 34.60, -106.75, 34.65), "US_NM")],
    ids=["namoi", "new_mexico"],
)
def test_live_by_admin_conf_query(bounds, partition, tmp_path):
    out = query_ftw(study_area=list(bounds), year=2024, cache_dir=tmp_path)
    info = out.attrs["ftw_query"]
    assert info["n_files_opened"] < info["n_files_listed"]
    assert out.crs.to_epsg() == 4326
    assert len(out) > 0  # both AOIs are cropland covered by the 2024 predictions
    assert (out["determination:datetime"].dt.year == 2024).all()
    codes = set(out["admin:country_code"] + "_" + out["admin:subdivision_code"])
    assert codes == {partition}


# ---------------------------------------------------------------------------
# clip=True re-measures polygons cut by the AOI boundary
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics_store(tmp_path):
    """by-admin-conf partition with published metrics:area / metrics:perimeter.

    ``inside`` lies within the query AOI; ``crossing`` extends east beyond it,
    so clipping keeps only its western half.
    """
    import datetime as dt

    import pandas as pd

    from agribound.io.crs import geodesic_perimeter_m, get_equal_area_crs

    geoms = [
        box(149.700, -30.400, 149.705, -30.395),  # inside
        box(149.715, -30.400, 149.725, -30.395),  # crosses maxx = 149.72
    ]
    gs = gpd.GeoSeries(geoms, crs="EPSG:4326")
    published_area = gs.to_crs(get_equal_area_crs()).area.to_numpy()
    published_perimeter = [geodesic_perimeter_m(g) for g in geoms]
    gdf = gpd.GeoDataFrame(
        {
            "id": ["inside", "crossing"],
            "determination:datetime": [pd.Timestamp(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))] * 2,
            "metrics:area": published_area,
            "metrics:perimeter": published_perimeter,
            "admin:country_code": ["AU"] * 2,
            "admin:subdivision_code": ["NSW"] * 2,
        },
        geometry=geoms,
        crs="EPSG:4326",
    )
    part = _country_partition(tmp_path / "results-by-admin-conf", "AU")
    gdf.to_parquet(part / "AU_NSW.parquet", write_covering_bbox=True)
    return {
        "root": tmp_path / "results-by-admin-conf",
        "area": dict(zip(gdf["id"], published_area, strict=True)),
        "perimeter": dict(zip(gdf["id"], published_perimeter, strict=True)),
    }


def test_clip_recomputes_metrics_of_polygons_crossing_the_aoi(metrics_store):
    """Regression: clip=True kept the unclipped polygon's published metrics."""
    from agribound.io.crs import geodesic_perimeter_m, get_equal_area_crs

    out = _q(metrics_store["root"], clip=True).set_index("id")
    crossing = out.loc["crossing"]
    assert bool(crossing["agribound:clipped"]) is True
    assert crossing.geometry.bounds[2] == pytest.approx(149.72)
    expected_area = (
        gpd.GeoSeries([crossing.geometry], crs="EPSG:4326").to_crs(get_equal_area_crs()).area[0]
    )
    assert crossing["metrics:area"] == pytest.approx(expected_area, rel=1e-9)
    assert crossing["metrics:area"] == pytest.approx(
        metrics_store["area"]["crossing"] / 2, rel=1e-3
    )
    assert crossing["metrics:perimeter"] == pytest.approx(
        geodesic_perimeter_m(crossing.geometry), rel=1e-9
    )
    assert crossing["metrics:perimeter"] < metrics_store["perimeter"]["crossing"]

    inside = out.loc["inside"]
    assert bool(inside["agribound:clipped"]) is False
    assert inside["metrics:area"] == metrics_store["area"]["inside"]
    assert inside["metrics:perimeter"] == metrics_store["perimeter"]["inside"]


def test_clip_false_keeps_published_metrics(metrics_store):
    out = _q(metrics_store["root"], clip=False).set_index("id")
    assert "agribound:clipped" not in out.columns
    for fid in ("inside", "crossing"):
        assert out.loc[fid, "metrics:area"] == metrics_store["area"][fid]
        assert out.loc[fid, "metrics:perimeter"] == metrics_store["perimeter"][fid]


def test_clip_keeps_polygonal_part_of_geometry_collection():
    from shapely.geometry import GeometryCollection, LineString, MultiPolygon

    from agribound.ftw_query import _polygonal_part

    poly = box(0, 0, 1, 1)
    line = LineString([(1, 0), (2, 0)])
    assert _polygonal_part(GeometryCollection([poly, line])).equals(poly)
    two = _polygonal_part(GeometryCollection([poly, box(2, 2, 3, 3), line]))
    assert isinstance(two, MultiPolygon) and len(two.geoms) == 2
    assert _polygonal_part(GeometryCollection([line])) is None


# ---------------------------------------------------------------------------
# Provenance sidecar of query outputs
# ---------------------------------------------------------------------------


def test_query_output_gets_a_provenance_sidecar(metrics_store, tmp_path):
    """Regression: query-ftw outputs recorded nothing about the query."""
    from agribound.provenance import provenance_path, read_provenance

    out_path = tmp_path / "ftw.gpkg"
    out = _q(metrics_store["root"], clip=True, year=2024, output_path=out_path)
    record = read_provenance(out_path)
    assert record is not None and out.attrs["provenance_path"] == str(provenance_path(out_path))
    assert record["kind"] == "query_ftw" and record["agribound_version"]
    params = record["parameters"]
    assert params["year"] == 2024 and params["clip"] is True and params["deduplicate"] is True
    assert params["layout"] == "by-admin-conf" and params["source_url"] == str(
        metrics_store["root"]
    )
    assert params["study_area"] == [149.69, -30.41, 149.72, -30.38]
    query = record["query"]
    assert query["backend"] == "pyarrow"
    assert query["n_returned"] == 2 and query["n_clipped"] == 1
    assert query["n_duplicates_dropped"] == 0
    assert record["aoi_bounds_4326"] == pytest.approx([149.69, -30.41, 149.72, -30.38])
    assert "not reference boundaries" in record["data"]
    assert record["versions"]["agribound"] == record["agribound_version"]
    # No sidecar without an output path, or when disabled.
    none_path = tmp_path / "none.gpkg"
    _q(metrics_store["root"], output_path=none_path, provenance=False)
    assert read_provenance(none_path) is None


def test_query_ftw_cli_writes_provenance(synthetic_ftw_store, tmp_path):
    from agribound.provenance import read_provenance

    output_path = tmp_path / "ftw_cli.gpkg"
    result = CliRunner().invoke(
        main,
        [
            "query-ftw",
            "--study-area",
            "bbox:0,0,0.5,0.5",
            "--year",
            "2025",
            "--manifest-path",
            str(synthetic_ftw_store["manifest_path"]),
            "--tile-dir",
            str(synthetic_ftw_store["tile_dir"]),
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Provenance →" in result.output
    record = read_provenance(output_path)
    assert record["parameters"]["study_area"] == "bbox:0,0,0.5,0.5"
    assert record["query"]["backend"] == "manifest" and record["query"]["n_tiles_read"] >= 1
    assert record["query"]["n_returned"] == 1 and record["query"]["n_clipped"] == 1


def test_query_ftw_help_names_the_accepted_study_areas():
    result = CliRunner().invoke(main, ["query-ftw", "--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "bbox:minx,miny,maxx,maxy (EPSG:4326)" in text
    assert "[default: clip]" in text and "[default: deduplicate]" in text
