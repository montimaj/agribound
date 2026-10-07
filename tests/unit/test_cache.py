"""Tests for agribound._cache (content-addressed cache keys)."""

from __future__ import annotations

import json
import os
import re

import geopandas as gpd
import pytest
from shapely.geometry import box

from agribound import _cache
from agribound._cache import aoi_fingerprint, cache_key, cache_path
from agribound.config import AgriboundConfig

HEX12 = re.compile(r"^[0-9a-f]{12}$")


@pytest.fixture(autouse=True)
def _clear_memo():
    _cache.clear_fingerprint_cache()
    yield
    _cache.clear_fingerprint_cache()


def _write_aoi(path, geoms, crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame({"n": list(range(len(geoms)))}, geometry=geoms, crs=crs)
    if str(path).endswith(".gpkg"):
        gdf.to_file(path, driver="GPKG")
    else:
        gdf.to_file(path, driver="GeoJSON")
    return str(path)


@pytest.fixture
def s2_config(sample_aoi_geojson, tmp_path):
    return AgriboundConfig(
        source="sentinel2",
        gee_project="p",
        study_area=sample_aoi_geojson,
        year=2022,
        output_path=str(tmp_path / "out" / "fields.gpkg"),
    )


class TestAoiFingerprint:
    def test_format_and_stability(self, s2_config):
        fp = aoi_fingerprint(s2_config)
        assert HEX12.match(fp)
        _cache.clear_fingerprint_cache()
        assert aoi_fingerprint(s2_config) == fp

    def test_same_geometry_different_files_and_order(self, tmp_path):
        a, b = box(-117.0, 36.0, -116.99, 36.01), box(-116.99, 36.0, -116.98, 36.01)
        f1 = _write_aoi(tmp_path / "a.geojson", [a, b])
        f2 = _write_aoi(tmp_path / "b.gpkg", [b, a])
        c1 = AgriboundConfig(source="sentinel2", gee_project="p", study_area=f1)
        c2 = c1.merged(study_area=f2)
        assert aoi_fingerprint(c1) == aoi_fingerprint(c2)

    def test_bbox_equals_equivalent_wkt(self):
        c1 = AgriboundConfig(
            source="sentinel2", gee_project="p", study_area="bbox:149.0,-30.5,149.1,-30.4"
        )
        wkt = "POLYGON ((149.0 -30.5, 149.1 -30.5, 149.1 -30.4, 149.0 -30.4, 149.0 -30.5))"
        c2 = c1.merged(study_area=wkt)
        assert aoi_fingerprint(c1) == aoi_fingerprint(c2)

    def test_different_geometry_differs(self, tmp_path):
        f1 = _write_aoi(tmp_path / "a.geojson", [box(0, 0, 1, 1)])
        f2 = _write_aoi(tmp_path / "b.geojson", [box(0, 0, 1, 1.001)])
        c1 = AgriboundConfig(source="sentinel2", gee_project="p", study_area=f1)
        assert aoi_fingerprint(c1) != aoi_fingerprint(c1.merged(study_area=f2))

    def test_reprojected_file_uses_4326(self, tmp_path):
        geom = box(-117.0, 36.0, -116.99, 36.01)
        f1 = _write_aoi(tmp_path / "a.geojson", [geom])
        utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:32611")
        f2 = _write_aoi(tmp_path / "b.gpkg", list(utm), crs="EPSG:32611")
        c1 = AgriboundConfig(source="sentinel2", gee_project="p", study_area=f1)
        # Round-tripping through UTM can move vertices by < 1e-9 deg; both snap to 1e-7.
        assert aoi_fingerprint(c1) == aoi_fingerprint(c1.merged(study_area=f2))

    def test_file_change_invalidates_memo(self, tmp_path):
        path = tmp_path / "aoi.geojson"
        _write_aoi(path, [box(0, 0, 1, 1)])
        cfg = AgriboundConfig(source="sentinel2", gee_project="p", study_area=str(path))
        before = aoi_fingerprint(cfg)
        _write_aoi(path, [box(0, 0, 2, 2)])
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
        assert aoi_fingerprint(cfg) != before

    def test_gee_asset_not_downloaded(self, monkeypatch):
        def boom(*a, **k):  # pragma: no cover - must not be called
            raise AssertionError("GEE asset must not be read")

        monkeypatch.setattr("agribound.io.vector._read_gee_asset", boom)
        c1 = AgriboundConfig(
            source="sentinel2", gee_project="p", study_area="projects/p/assets/aoi_a"
        )
        c2 = c1.merged(study_area="projects/p/assets/aoi_b")
        assert HEX12.match(aoi_fingerprint(c1))
        assert aoi_fingerprint(c1) != aoi_fingerprint(c2)

    def test_local_without_study_area_uses_raster_identity(self, sample_rgb_tif, sample_rgbn_tif):
        c1 = AgriboundConfig(source="local", local_tif_path=sample_rgb_tif)
        c2 = c1.merged(local_tif_path=sample_rgbn_tif)
        assert aoi_fingerprint(c1) != aoi_fingerprint(c2)


class TestCacheKey:
    def test_format_and_stability(self, s2_config):
        key = cache_key(s2_config)
        assert HEX12.match(key)
        assert cache_key(s2_config) == key
        assert cache_key(s2_config.merged()) == key

    @pytest.mark.parametrize(
        "override",
        [
            {"year": 2023},
            {"date_range": ("2022-04-01", "2022-04-30")},
            {"composite_method": "greenest"},
            {"cloud_cover_max": 50},
            {"export_crs": "EPSG:32611"},
            {"s2_cloud_mask": "cloud_score_plus"},
            {"naip_resolution_m": 0.6},
            {"study_area": "bbox:-117.0,36.0,-116.9,36.1"},
            {"source": "landsat"},
        ],
    )
    def test_key_changes(self, s2_config, override):
        assert cache_key(s2_config.merged(**override)) != cache_key(s2_config)

    @pytest.mark.parametrize(
        "override",
        [
            {"engine": "ftw"},
            {"min_field_area_m2": 10.0},
            {"simplify_tolerance": 5.0},
            {"output_path": "elsewhere/x.gpkg"},
            {"device": "cpu"},
            {"lulc_filter": False},
            {"seed": 1},
            {"tessera_version": "v1.1"},  # not an embedding source
            {"cloud_score_threshold": 0.5},  # SCL mask selected
            {"landsat_pan_missions": ("LE07", "LC08")},  # not the landsat-pan source
            {"lulc_tree_crops": True},  # LULC rasters add their own key part
        ],
    )
    def test_key_stable_for_unrelated_fields(self, s2_config, override):
        assert cache_key(s2_config.merged(**override)) == cache_key(s2_config)

    def test_landsat_pan_missions_in_key_for_landsat_pan_only(self, s2_config):
        auto = s2_config.merged(source="landsat-pan")
        assert ("landsat_pan_missions", "auto") in _cache._key_fields(auto, True)
        assert "landsat_pan_missions" not in dict(_cache._key_fields(s2_config, True))
        keys = {
            cache_key(auto.merged(landsat_pan_missions=value))
            for value in ("auto", "LE07", "LC08", "LC08,LC09", "LE07,LC08,LC09")
        }
        assert len(keys) == 5
        # The normalised setting is hashed: order and case do not matter.
        assert cache_key(auto.merged(landsat_pan_missions="lc09,LC08")) == cache_key(
            auto.merged(landsat_pan_missions=["LC08", "LC09"])
        )
        # Not in the other sources' keys (their cached composites stay valid).
        landsat = s2_config.merged(source="landsat")
        assert cache_key(landsat.merged(landsat_pan_missions="LE07")) == cache_key(landsat)

    def test_landsat_pan_key_differs_from_the_key_without_missions(self, s2_config):
        """Composites cached before the mission rule (Landsat 7 and 8/9 mixed) are not reused."""
        import hashlib

        auto = s2_config.merged(source="landsat-pan")
        fields = [
            [n, _cache._canonical(v)]
            for n, v in _cache._key_fields(auto, True)
            if n != "landsat_pan_missions"
        ]
        text = json.dumps({"fields": fields, "parts": []}, sort_keys=True, separators=(",", ":"))
        assert cache_key(auto) != hashlib.sha1(text.encode()).hexdigest()[:12]

    def test_cloud_score_threshold_matters_with_cloud_score_plus(self, s2_config):
        cs = s2_config.merged(s2_cloud_mask="cloud_score_plus")
        assert cache_key(cs.merged(cloud_score_threshold=0.5)) != cache_key(cs)

    def test_schema_version_bump_changes_key(self, s2_config, monkeypatch):
        before = cache_key(s2_config)
        monkeypatch.setattr(_cache, "CACHE_SCHEMA_VERSION", "999")
        assert cache_key(s2_config) != before

    def test_tessera_version_and_variant(self, tmp_path):
        cfg = AgriboundConfig(
            source="tessera-embedding",
            engine="embedding",
            study_area="bbox:149.0,-30.5,149.1,-30.4",
            year=2024,
        )
        assert cache_key(cfg.merged(tessera_version="v1.1")) != cache_key(cfg)
        assert cache_key(cfg.merged(tessera_variant="cambridge")) != cache_key(cfg)

    def test_parts_are_ordered_and_hashed(self, s2_config):
        base = cache_key(s2_config)
        ab = cache_key(s2_config, "a", "b")
        assert ab != base
        assert cache_key(s2_config, "b", "a") != ab
        assert cache_key(s2_config, "a", "b") == ab
        assert cache_key(s2_config, 1) == cache_key(s2_config, "1")

    def test_include_temporal_false(self, s2_config):
        k = cache_key(s2_config, include_temporal=False)
        assert cache_key(s2_config.merged(year=2023), include_temporal=False) == k
        other_window = s2_config.merged(date_range=("2022-04-01", "2022-04-30"))
        assert cache_key(other_window, include_temporal=False) == k
        assert k != cache_key(s2_config)

    def test_local_raster_identity_in_key(
        self, sample_rgb_tif, sample_rgbn_tif, sample_aoi_geojson
    ):
        cfg = AgriboundConfig(
            source="local", local_tif_path=sample_rgb_tif, study_area=sample_aoi_geojson
        )
        assert cache_key(cfg) != cache_key(cfg.merged(local_tif_path=sample_rgbn_tif))

    def test_usgs_state_in_key(self, sample_aoi_geojson):
        cfg = AgriboundConfig(source="usgs-naip-plus", study_area=sample_aoi_geojson, year=2022)
        assert cache_key(cfg) != cache_key(cfg.merged(usgs_state="MI"))

    def test_key_is_json_sha1_prefix(self, s2_config):
        """The key is a plain function of the documented fields (no hidden state)."""
        import hashlib

        fields = [[n, _cache._canonical(v)] for n, v in _cache._key_fields(s2_config, True)]
        text = json.dumps({"fields": fields, "parts": []}, sort_keys=True, separators=(",", ":"))
        assert cache_key(s2_config) == hashlib.sha1(text.encode()).hexdigest()[:12]


class TestCachePath:
    def test_path_in_working_dir(self, s2_config):
        path = cache_path(s2_config, "composite", ".tif")
        assert path.parent == s2_config.get_working_dir()
        assert path.name == f"composite_{cache_key(s2_config)}.tif"
        assert path.parent.is_dir()

    def test_honours_cache_dir(self, s2_config, tmp_path):
        cfg = s2_config.merged(cache_dir=str(tmp_path / "shared"))
        path = cache_path(cfg, "win_a", ".tif", "window-a")
        assert path.parent == tmp_path / "shared"
        assert path.name == f"win_a_{cache_key(cfg, 'window-a')}.tif"

    def test_nested_stem_creates_parent(self, s2_config):
        path = cache_path(s2_config, "checkpoints/yolo/run", "", "large")
        assert path.parent.is_dir()
        assert path.parent.name == "yolo"

    def test_include_temporal_forwarded(self, s2_config):
        a = cache_path(s2_config, "x", ".tif", include_temporal=False)
        b = cache_path(s2_config.merged(year=2020), "x", ".tif", include_temporal=False)
        assert a == b


def test_long_wkt_study_area():
    """A WKT longer than the OS file-name limit is parsed, not stat()-ed."""
    import numpy as np

    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    coords = ", ".join(f"{149 + 0.01 * np.cos(a):.9f} {-30 + 0.01 * np.sin(a):.9f}" for a in angles)
    first = coords.split(", ")[0]
    wkt = f"POLYGON (({coords}, {first}))"
    assert len(wkt) > 4096
    cfg = AgriboundConfig(source="sentinel2", gee_project="p", study_area=wkt)
    assert HEX12.match(aoi_fingerprint(cfg))
