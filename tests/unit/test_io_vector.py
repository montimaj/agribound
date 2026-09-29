"""Tests for agribound.io.vector read/write round-trips."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from agribound.io.vector import read_vector, write_vector


class TestVectorRoundTripGpkg:
    """Test GeoPackage write then read."""

    def test_gpkg_round_trip(self, tmp_path, sample_geodataframe):
        path = tmp_path / "output.gpkg"
        write_vector(sample_geodataframe, path)
        loaded = read_vector(path)

        assert len(loaded) == len(sample_geodataframe)
        assert loaded.crs is not None

    def test_gpkg_geometry_preserved(self, tmp_path):
        poly = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame({"val": [42]}, geometry=[poly], crs="EPSG:4326")
        path = tmp_path / "test.gpkg"
        write_vector(gdf, path)
        loaded = read_vector(path)
        assert loaded.geometry.iloc[0].equals(poly)


class TestVectorRoundTripGeojson:
    """Test GeoJSON write then read."""

    def test_geojson_round_trip(self, tmp_path):
        poly = box(-1, -1, 1, 1)
        gdf = gpd.GeoDataFrame({"name": ["square"]}, geometry=[poly], crs="EPSG:4326")
        path = tmp_path / "test.geojson"
        write_vector(gdf, path)
        loaded = read_vector(path)

        assert len(loaded) == 1
        assert loaded.crs is not None

    def test_geojson_reprojects_to_4326(self, tmp_path, sample_geodataframe):
        """GeoJSON output must be EPSG:4326; write_vector should reproject."""
        path = tmp_path / "reprojected.geojson"
        write_vector(sample_geodataframe, path)
        loaded = read_vector(path)
        # GeoJSON is always 4326
        assert loaded.crs.to_epsg() == 4326


class TestReadVectorErrors:
    """Test error handling in read_vector."""

    def test_missing_file_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            read_vector(tmp_path / "nonexistent.gpkg")

    def test_unsupported_extension_raises(self, tmp_path):
        import pytest

        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("not a vector")
        with pytest.raises(ValueError, match="Unsupported vector format"):
            read_vector(bad_file)


class TestFiboaParquet:
    def test_parquet_round_trip_adds_required_columns(self, tmp_path, sample_geodataframe):
        path = tmp_path / "fields.parquet"
        write_vector(sample_geodataframe.drop(columns=["id"]), path)
        loaded = read_vector(path)
        assert loaded.crs.to_epsg() == 4326
        assert {"id", "determination:method"} <= set(loaded.columns)
        assert loaded["id"].is_unique


class TestReadStudyArea:
    def test_bbox(self):
        from agribound.io.vector import read_study_area

        gdf = read_study_area("bbox:149.1,-30.5,149.2,-30.4")
        assert gdf.crs.to_epsg() == 4326
        assert tuple(round(v, 6) for v in gdf.total_bounds) == (149.1, -30.5, 149.2, -30.4)

    def test_bbox_with_spaces_and_case(self):
        from agribound.io.vector import read_study_area

        gdf = read_study_area("BBOX: -117, 36, -116.9, 36.1")
        assert len(gdf) == 1

    def test_invalid_bbox(self):
        import pytest

        from agribound.io.vector import read_study_area

        for bad in ("bbox:1,2,3", "bbox:a,b,c,d", "bbox:10,0,5,1", "bbox:0,-95,1,0"):
            with pytest.raises(ValueError, match="Invalid bbox"):
                read_study_area(bad)

    def test_wkt(self):
        from agribound.io.vector import read_study_area

        gdf = read_study_area("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
        assert gdf.crs.to_epsg() == 4326
        assert gdf.geometry.iloc[0].equals(box(0, 0, 1, 1))
        multi = read_study_area("multipolygon (((0 0, 1 0, 1 1, 0 1, 0 0)))")
        assert multi.geometry.iloc[0].geom_type == "MultiPolygon"

    def test_ewkt_srid(self):
        from agribound.io.vector import read_study_area

        gdf = read_study_area(
            "SRID=32611;POLYGON ((500000 4000000, 500100 4000000, 500100 4000100, 500000 4000000))"
        )
        assert gdf.crs.to_epsg() == 32611

    def test_invalid_wkt(self):
        import pytest

        from agribound.io.vector import read_study_area

        with pytest.raises(ValueError, match="Invalid WKT"):
            read_study_area("POLYGON ((0 0, 1 0")

    def test_file_path_and_pathlib(self, sample_aoi_geojson):
        from pathlib import Path

        from agribound.io.vector import read_study_area

        assert len(read_study_area(sample_aoi_geojson)) == 1
        assert len(read_study_area(Path(sample_aoi_geojson))) == 1

    def test_gee_asset_dispatch(self, monkeypatch):
        from agribound.io import vector

        called = {}

        def fake(asset_id, config=None):
            called["id"] = asset_id
            called["config"] = config
            return gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

        monkeypatch.setattr(vector, "_read_gee_asset", fake)
        vector.read_study_area("projects/p/assets/aoi")
        assert called == {"id": "projects/p/assets/aoi", "config": None}
        marker = object()
        vector.read_study_area("users/me/aoi", config=marker)
        assert called == {"id": "users/me/aoi", "config": marker}

    @staticmethod
    def _fake_ee(monkeypatch, events):
        import sys
        import types

        from shapely.geometry import mapping

        class FakeFC:
            def __init__(self, asset_id):
                events.append(("read", asset_id))

            def getInfo(self):  # noqa: N802 - mirrors ee.FeatureCollection.getInfo
                return {
                    "features": [{"geometry": mapping(box(1, 1, 2, 2)), "properties": {"k": 7}}]
                }

        monkeypatch.setitem(sys.modules, "ee", types.SimpleNamespace(FeatureCollection=FakeFC))

    def test_gee_asset_uses_configured_credentials(self, monkeypatch):
        """With a config, Earth Engine is initialised via ensure_gee (project, key, tag)."""
        from agribound.config import AgriboundConfig
        from agribound.io import vector

        events = []
        self._fake_ee(monkeypatch, events)
        monkeypatch.setattr(
            "agribound.auth.ensure_gee",
            lambda config: events.append(("ensure", config.gee_project)),
        )

        def no_default_setup(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("setup_gee() defaults must not be used when a config is given")

        monkeypatch.setattr("agribound.auth.setup_gee", no_default_setup)
        cfg = AgriboundConfig(
            source="sentinel2", gee_project="proj-x", study_area="projects/p/assets/aoi"
        )
        gdf = vector.read_study_area("projects/p/assets/aoi", config=cfg)
        assert events == [("ensure", "proj-x"), ("read", "projects/p/assets/aoi")]
        assert gdf.crs == "EPSG:4326" and gdf["k"].tolist() == [7]

    def test_gee_asset_without_config_uses_setup_defaults(self, monkeypatch):
        from agribound.io import vector

        events = []
        self._fake_ee(monkeypatch, events)
        monkeypatch.setattr("agribound.auth.check_gee_initialized", lambda: False)
        monkeypatch.setattr("agribound.auth.setup_gee", lambda **kw: events.append(("setup", kw)))
        vector.read_study_area("users/me/aoi")
        assert events == [("setup", {}), ("read", "users/me/aoi")]


class TestConfigStudyAreaCopy:
    """read_config_study_area keeps a local copy of a GEE-asset study area."""

    ASSET = "projects/p/assets/aoi"

    @staticmethod
    def _config(tmp_path, **kw):
        from agribound.config import AgriboundConfig

        params = {
            "source": "sentinel2",
            "gee_project": "proj-x",
            "study_area": TestConfigStudyAreaCopy.ASSET,
            "cache_dir": str(tmp_path / "cache"),
        }
        params.update(kw)
        return AgriboundConfig(**params)

    def test_asset_is_read_once_then_from_the_copy(self, monkeypatch, tmp_path):
        import json
        import sys

        from agribound._cache import gee_asset_fingerprint
        from agribound.io import vector

        events = []
        TestReadStudyArea._fake_ee(monkeypatch, events)
        monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: events.append("ensure"))
        cfg = self._config(tmp_path)
        path = vector.study_area_cache_file(cfg)
        key = gee_asset_fingerprint(self.ASSET)
        assert path == tmp_path / "cache" / f"study_area_gee_{key}.geojson"
        first = vector.read_config_study_area(cfg)
        assert events == ["ensure", ("read", self.ASSET)]
        assert first["k"].tolist() == [7]  # read from Earth Engine: with properties
        saved = json.loads(path.read_text())
        assert saved["agribound:asset_id"] == self.ASSET and len(saved["features"]) == 1
        assert not path.with_name(path.name + ".partial").exists()

        # Offline: no Earth Engine module and ensure_gee failing; the copy is used.
        monkeypatch.setitem(sys.modules, "ee", None)

        def offline(config):  # pragma: no cover - must not be called
            raise AssertionError("Earth Engine must not be contacted when the copy exists")

        monkeypatch.setattr("agribound.auth.ensure_gee", offline)
        again = vector.read_config_study_area(cfg)
        assert again.crs == "EPSG:4326"
        assert again.geometry.iloc[0].equals(box(1, 1, 2, 2))
        assert list(again.columns) == ["geometry"]

    def test_unreadable_copy_is_replaced(self, monkeypatch, tmp_path):
        from agribound.io import vector

        events = []
        TestReadStudyArea._fake_ee(monkeypatch, events)
        monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
        cfg = self._config(tmp_path)
        path = vector.study_area_cache_file(cfg)
        path.write_text("{not json")
        gdf = vector.read_config_study_area(cfg)
        assert events == [("read", self.ASSET)] and gdf.geometry.iloc[0].equals(box(1, 1, 2, 2))
        events.clear()
        assert vector.read_config_study_area(cfg).geometry.iloc[0].equals(box(1, 1, 2, 2))
        assert events == []

    def test_other_study_areas_are_read_directly(self, tmp_path, sample_aoi_geojson):
        from agribound.io import vector

        for study_area in ("bbox:1,1,2,2", str(sample_aoi_geojson)):
            cfg = self._config(tmp_path, study_area=study_area)
            assert vector.study_area_cache_file(cfg) is None
            assert len(vector.read_config_study_area(cfg)) == 1
        assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())


class TestWriteVectorFormat:
    def test_format_contradicting_extension_raises(self, tmp_path, sample_geodataframe):
        import pytest

        with pytest.raises(ValueError, match="contradicts"):
            write_vector(sample_geodataframe, tmp_path / "x.parquet", format="gpkg")
        assert not (tmp_path / "x.parquet").exists()

    def test_explicit_matching_format(self, tmp_path, sample_geodataframe):
        out = write_vector(sample_geodataframe, tmp_path / "x.gpkg", format="GPKG")
        assert len(read_vector(out)) == len(sample_geodataframe)

    def test_unknown_extension_needs_format(self, tmp_path, sample_geodataframe):
        import pytest

        with pytest.raises(ValueError, match="Cannot infer"):
            write_vector(sample_geodataframe, tmp_path / "x.dat")


class TestStudyAreaDispatchEdgeCases:
    def test_file_named_like_wkt_keyword(self, tmp_path):
        from agribound.io.vector import read_study_area

        path = tmp_path / "polygons.geojson"
        gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_file(path)
        import os

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert len(read_study_area("polygons.geojson")) == 1
        finally:
            os.chdir(cwd)

    def test_wkt_with_z_and_space(self):
        from agribound.io.vector import read_study_area

        gdf = read_study_area("  POLYGON Z ((0 0 1, 1 0 1, 1 1 1, 0 0 1))")
        assert gdf.geometry.iloc[0].has_z
