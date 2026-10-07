"""Tests for the LULC crop filter (dataset routing, NaN policy, zonal statistics).

Earth Engine is never contacted: the NLCD coverage test and the reduceRegions
call are replaced by stand-ins, and raster mode runs on a synthetic LULC
raster placed at the cache path the filter expects.
"""

from __future__ import annotations

import datetime as dt
import sys
import types

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from agribound.config import AgriboundConfig
from agribound.postprocess import lulc_filter as lf


@pytest.fixture
def needs_ee():
    """Skip unless earthengine-api (the ``gee`` extra) is installed.

    These tests raise real ``ee.ee_exception.EEException`` errors; the CI core
    job does not install the extra.
    """
    return pytest.importorskip("ee")


NAMOI = (149.0, -30.5, 149.05, -30.45)
IOWA = (-93.60, 42.00, -93.55, 42.05)
CHIHUAHUA = (-106.10, 28.60, -106.05, 28.65)  # inside the CONUS envelope, no NLCD
ONTARIO = (-81.00, 43.00, -80.95, 43.05)  # inside the CONUS envelope, no NLCD


def _bbox(b):
    return "bbox:" + ",".join(str(v) for v in b)


def _config(tmp_path, bbox=NAMOI, year=2023, **kwargs):
    params = {
        "source": "local",
        "local_tif_path": str(tmp_path / "unused.tif"),
        "study_area": _bbox(bbox),
        "year": year,
        "output_path": str(tmp_path / "fields.gpkg"),
        "lulc_filter": True,
    }
    params.update(kwargs)
    return AgriboundConfig(**params)


@pytest.fixture(autouse=True)
def _fixed_today(monkeypatch):
    monkeypatch.setattr(lf, "_today", lambda: dt.date(2026, 9, 26))
    lf._NLCD_SOURCE_CACHE.clear()


class _Calls(list):
    """Recorded NLCD coverage queries; ``events`` also records ensure_gee calls in order."""

    events: list


@pytest.fixture
def nlcd_coverage(monkeypatch):
    """Fake NLCD valid fraction: 1.0 over Iowa, 0.0 elsewhere; records calls."""
    calls = _Calls()
    calls.events = []

    def fake(geom, year):
        calls.append((geom.centroid.x, geom.centroid.y, year))
        calls.events.append("nlcd_query")
        return 1.0 if box(*IOWA).contains(geom.centroid) else 0.0

    monkeypatch.setattr(lf, "_nlcd_valid_fraction", fake)
    monkeypatch.setattr(
        "agribound.auth.ensure_gee", lambda config: calls.events.append("ensure_gee")
    )
    return calls


class TestConstants:
    def test_crop_classes(self):
        assert lf.NLCD_CROP_CLASSES == (81, 82)
        assert lf.C3S_CROP_CLASSES == (10, 11, 12, 20, 30)
        assert lf.CDL_CULTIVATED_VALUE == 2

    def test_tree_crop_classes(self):
        assert lf.C3S_TREE_CLASSES == (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90)
        assert not set(lf.C3S_TREE_CLASSES) & set(lf.C3S_CROP_CLASSES)
        # not the tree and shrub mosaic (100) or flooded tree cover (160, 170)
        assert not {100, 160, 170} & set(lf.C3S_TREE_CLASSES)
        assert lf.TREE_CROP_DATASETS == ("dynamic_world", "c3s")
        assert {"C3S_TREE_CLASSES", "TREE_CROP_DATASETS"} <= set(lf.__all__)

    def test_year_ranges(self):
        assert lf.dataset_year_range("nlcd") == (1985, 2025)
        assert lf.dataset_year_range("c3s") == (2000, 2022)
        assert lf.dataset_year_range("cdl") == (2013, 2023)
        assert lf.dataset_year_range("dynamic_world") == (2016, 2025)  # last full year


class TestRouting:
    def test_namoi_skips_nlcd_query(self, tmp_path, nlcd_coverage):
        assert lf.select_lulc_dataset(_config(tmp_path, NAMOI)) == ("dynamic_world", 2023)
        assert nlcd_coverage == []

    def test_iowa_routes_to_nlcd(self, tmp_path, nlcd_coverage):
        assert lf.select_lulc_dataset(_config(tmp_path, IOWA, year=2024)) == ("nlcd", 2024)
        assert nlcd_coverage[0][2] == 2024

    @pytest.mark.parametrize("bbox", [CHIHUAHUA, ONTARIO])
    def test_mexico_and_canada_do_not_route_to_nlcd(self, tmp_path, nlcd_coverage, bbox):
        assert lf.select_lulc_dataset(_config(tmp_path, bbox)) == ("dynamic_world", 2023)
        assert len(nlcd_coverage) == 1  # the coverage test was run and failed

    def test_pre_2016_outside_us_uses_c3s(self, tmp_path, nlcd_coverage):
        assert lf.select_lulc_dataset(_config(tmp_path, CHIHUAHUA, year=2010)) == ("c3s", 2010)
        assert lf.select_lulc_dataset(_config(tmp_path, NAMOI, year=2015)) == ("c3s", 2015)
        assert lf.select_lulc_dataset(_config(tmp_path, NAMOI, year=1995)) == ("c3s", 2000)

    def test_current_year_uses_last_full_dynamic_world_year(self, tmp_path, nlcd_coverage):
        assert lf.select_lulc_dataset(_config(tmp_path, NAMOI, year=2026)) == (
            "dynamic_world",
            2025,
        )

    def test_nlcd_nearest_year(self, tmp_path, nlcd_coverage):
        assert lf.select_lulc_dataset(_config(tmp_path, IOWA, year=1980)) == ("nlcd", 1985)
        assert nlcd_coverage[0][2] == 1985

    def test_override(self, tmp_path, nlcd_coverage):
        cfg = _config(tmp_path, NAMOI, year=2025, lulc_dataset="cdl")
        assert lf.select_lulc_dataset(cfg) == ("cdl", 2023)
        cfg = _config(tmp_path, IOWA, year=1995, lulc_dataset="c3s")
        assert lf.select_lulc_dataset(cfg) == ("c3s", 2000)
        assert nlcd_coverage == []

    def test_earth_engine_initialised_before_nlcd_query(self, tmp_path, nlcd_coverage):
        lf.select_lulc_dataset(_config(tmp_path, IOWA))
        assert nlcd_coverage.events == ["ensure_gee", "nlcd_query"]
        # areas far from the US need no Earth Engine request for routing
        nlcd_coverage.events.clear()
        lf.select_lulc_dataset(_config(tmp_path, NAMOI))
        assert nlcd_coverage.events == []

    def test_selection_is_cached(self, tmp_path, nlcd_coverage):
        cfg = _config(tmp_path, IOWA)
        lf.select_lulc_dataset(cfg)
        lf.select_lulc_dataset(cfg)
        assert len(nlcd_coverage) == 1
        # a different year is a different decision
        lf.select_lulc_dataset(cfg.merged(year=2020))
        assert len(nlcd_coverage) == 2


# ---------------------------------------------------------------------------
# Zonal statistics on a synthetic raster
# ---------------------------------------------------------------------------


def _lulc_raster(path, data, x0=500000.0, y0=6620000.0, res=10.0, crs="EPSG:32755", **tags):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=1,
        height=data.shape[0],
        width=data.shape[1],
        dtype="float32",
        crs=crs,
        transform=from_origin(x0, y0, res, res),
        nodata=float("nan"),
    ) as dst:
        dst.write(data.astype(np.float32), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return str(path)


class TestZonalMean:
    def test_matches_brute_force(self, tmp_path):
        rng = np.random.default_rng(3)
        data = rng.random((200, 300)).astype(np.float32)
        data[50:60, 50:60] = np.nan
        path = _lulc_raster(tmp_path / "r.tif", data)
        polys = []
        for _ in range(40):
            c = rng.integers(0, 280)
            r = rng.integers(0, 180)
            w, h = rng.integers(2, 20, size=2)
            polys.append(
                box(
                    500000 + c * 10, 6620000 - (r + h) * 10, 500000 + (c + w) * 10, 6620000 - r * 10
                )
            )
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32755")
        got = lf.zonal_mean_from_raster(gdf, path, block_rows=37)
        expected = []
        for p in polys:
            c0, c1 = int((p.bounds[0] - 500000) / 10), int((p.bounds[2] - 500000) / 10)
            r0, r1 = int((6620000 - p.bounds[3]) / 10), int((6620000 - p.bounds[1]) / 10)
            expected.append(np.nanmean(data[r0:r1, c0:c1].astype(np.float64)))
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_subpixel_polygon_uses_all_touched(self, tmp_path):
        data = np.zeros((10, 10), np.float32)
        data[2, 3] = 0.8
        path = _lulc_raster(tmp_path / "r.tif", data)
        tiny = box(500000 + 32, 6620000 - 28, 500000 + 34, 6620000 - 26)  # inside pixel (2, 3)
        got = lf.zonal_mean_from_raster(gpd.GeoDataFrame(geometry=[tiny], crs="EPSG:32755"), path)
        assert got[0] == pytest.approx(0.8)

    def test_all_nan_and_outside_give_nan(self, tmp_path):
        data = np.full((10, 10), np.nan, np.float32)
        path = _lulc_raster(tmp_path / "r.tif", data)
        polys = [box(500010, 6619950, 500050, 6619990), box(0, 0, 10, 10)]
        got = lf.zonal_mean_from_raster(gpd.GeoDataFrame(geometry=polys, crs="EPSG:32755"), path)
        assert np.isnan(got).all()

    def test_polygons_are_reprojected(self, tmp_path):
        data = np.ones((100, 100), np.float32)
        path = _lulc_raster(tmp_path / "r.tif", data)
        poly = gpd.GeoSeries([box(500100, 6619100, 500500, 6619500)], crs="EPSG:32755")
        got = lf.zonal_mean_from_raster(gpd.GeoDataFrame(geometry=poly.to_crs(4326)), path)
        assert got[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# filter_by_lulc (raster mode, offline)
# ---------------------------------------------------------------------------


@pytest.fixture
def namoi_raster_mode(tmp_path, nlcd_coverage):
    """Raster-mode config over Namoi with a synthetic DW raster in the cache."""
    cfg = _config(tmp_path, NAMOI, lulc_mode="raster")
    selection = lf._select(cfg)
    path = lf._lulc_raster_path(cfg, selection)
    # 200 x 200 px at 10 m: left half crop 0.9, right half 0.1, NaN block at the top-right
    data = np.full((200, 200), 0.1, np.float32)
    data[:, :100] = 0.9
    data[:40, 160:] = np.nan
    _lulc_raster(
        path,
        data,
        AGRIBOUND_LULC_DATASET="dynamic_world",
        AGRIBOUND_LULC_ASSET="GOOGLE/DYNAMICWORLD/V1",
        AGRIBOUND_LULC_BAND="crops",
        AGRIBOUND_LULC_YEAR=selection.year_used,
        AGRIBOUND_LULC_VALUE="mean annual-median Dynamic World crop probability",
    )
    polys = gpd.GeoDataFrame(
        {"name": ["crop", "noncrop", "nan", "straddle"]},
        geometry=[
            box(500100, 6619000, 500500, 6619400),
            box(501200, 6619000, 501600, 6619400),
            box(501700, 6619700, 501900, 6619900),
            box(500800, 6618000, 501200, 6618400),
        ],
        crs="EPSG:32755",
    )
    polys.attrs["engine_meta"] = {"backend": "stub"}
    return cfg, polys


class TestFilterRasterMode:
    def test_threshold_nan_policy_and_columns(self, namoi_raster_mode):
        cfg, polys = namoi_raster_mode
        out = lf.filter_by_lulc(polys, cfg)
        assert list(out["name"]) == ["crop", "nan", "straddle"]
        frac = dict(zip(out["name"], out["lulc:crop_fraction"], strict=True))
        assert frac["crop"] == pytest.approx(0.9)
        assert np.isnan(frac["nan"])  # never 0
        assert frac["straddle"] == pytest.approx(0.5)
        assert out["lulc:valid"].tolist() == [True, False, True]
        assert set(out["lulc:dataset"]) == {"dynamic_world"}
        assert set(out["lulc:year"]) == {2023}
        stats = out.attrs["lulc_stats"]
        assert stats["n_in"] == 4 and stats["n_kept"] == 3 and stats["n_nan"] == 1
        assert stats["n_below_threshold"] == 1 and stats["mode"] == "raster"
        assert stats["dataset"] == "dynamic_world" and stats["year_used"] == 2023
        assert out.attrs["engine_meta"] == {"backend": "stub"}
        assert out.crs == polys.crs

    def test_drop_policy(self, namoi_raster_mode):
        cfg, polys = namoi_raster_mode
        out = lf.filter_by_lulc(polys, cfg.merged(lulc_nodata_policy="drop"))
        assert list(out["name"]) == ["crop", "straddle"]
        assert out.attrs["lulc_stats"]["n_nan_dropped"] == 1

    def test_threshold_zero_keeps_valid(self, namoi_raster_mode):
        cfg, polys = namoi_raster_mode
        # the threshold is not part of the cache key: the same raster is used
        cfg0 = cfg.merged(lulc_crop_threshold=0.0)
        assert len(lf.filter_by_lulc(polys, cfg0)) == 4
        assert lf.filter_by_lulc(polys, cfg0).attrs["lulc_stats"]["tree_crops"] is False

    def test_tree_crops_use_their_own_raster(self, namoi_raster_mode, monkeypatch):
        cfg, polys = namoi_raster_mode
        tree = cfg.merged(lulc_tree_crops=True)
        selection = lf._select(tree)
        path = lf._lulc_raster_path(tree, selection)
        assert path != lf._lulc_raster_path(cfg, selection)
        # crops + trees: the whole raster is 0.95, so nothing is below the threshold
        _lulc_raster(
            path,
            np.full((200, 200), 0.95, np.float32),
            AGRIBOUND_LULC_DATASET="dynamic_world",
            AGRIBOUND_LULC_ASSET="GOOGLE/DYNAMICWORLD/V1",
            AGRIBOUND_LULC_BAND="crops+trees",
            AGRIBOUND_LULC_YEAR=selection.year_used,
            AGRIBOUND_LULC_VALUE="mean annual-median Dynamic World crops+trees probability",
            AGRIBOUND_LULC_TREE_CROPS="True",
        )

        def no_gee(config):
            raise AssertionError("must not contact Earth Engine")

        monkeypatch.setattr("agribound.auth.ensure_gee", no_gee)
        out = lf.filter_by_lulc(polys, tree)
        assert list(out["name"]) == ["crop", "noncrop", "nan", "straddle"]
        assert out["lulc:crop_fraction"].tolist() == pytest.approx([0.95] * 4)
        stats = out.attrs["lulc_stats"]
        assert stats["tree_crops"] is True and stats["mode"] == "raster"
        assert stats["band"] == "crops+trees" and "crops+trees" in stats["value"]
        # the default rule still reads the crops-only raster
        default = lf.filter_by_lulc(polys, cfg)
        assert list(default["name"]) == ["crop", "nan", "straddle"]
        assert default.attrs["lulc_stats"]["tree_crops"] is False
        assert default.attrs["lulc_stats"]["band"] == "crops"


@pytest.mark.parametrize(
    ("dataset", "differs"),
    [("dynamic_world", True), ("c3s", True), ("nlcd", False), ("cdl", False)],
)
def test_lulc_raster_path_depends_on_tree_crops_for_dw_and_c3s(tmp_path, dataset, differs):
    cfg = _config(tmp_path, NAMOI, lulc_mode="raster")
    selection = lf.LulcSelection(dataset, 2020, 2020, "test")
    default = lf._lulc_raster_path(cfg, selection)
    tree = lf._lulc_raster_path(cfg.merged(lulc_tree_crops=True), selection)
    assert (tree != default) is differs
    assert tree.name.startswith(f"lulc_{dataset}_") and tree.parent == default.parent
    # the tree-crop raster is not shared with another dataset or year
    other = lf.LulcSelection(dataset, 2021, 2021, "test")
    assert lf._lulc_raster_path(cfg.merged(lulc_tree_crops=True), other) != tree


class TestFilterServerMode:
    def test_missing_means_are_nan(self, tmp_path, nlcd_coverage, monkeypatch):
        monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
        monkeypatch.setattr("agribound.composites.gee.ee_geometry", lambda g: "REGION")
        meta = {"asset": "A", "band": "b", "year_used": 2023, "value": "v", "tree_crops": False}
        monkeypatch.setattr(
            lf, "crop_image", lambda ds, year, region, tree_crops=False: ("IMG", meta)
        )
        monkeypatch.setattr(
            lf, "server_zonal_means", lambda gdf, image, scale, batch_size: np.array([0.8, np.nan])
        )
        polys = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs="EPSG:32755")
        out = lf.filter_by_lulc(polys, _config(tmp_path, NAMOI))
        assert len(out) == 2
        assert np.isnan(out["lulc:crop_fraction"].iloc[1])
        assert out.attrs["lulc_stats"]["mode"] == "server"

    @pytest.mark.parametrize("tree_crops", [False, True])
    def test_tree_crops_passed_and_recorded(self, tmp_path, nlcd_coverage, monkeypatch, tree_crops):
        monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
        monkeypatch.setattr("agribound.composites.gee.ee_geometry", lambda g: "REGION")
        seen = []

        def fake_crop_image(dataset, year, region, tree_crops=False):
            seen.append((dataset, year, tree_crops))
            band = "crops+trees" if tree_crops else "crops"
            meta = {"asset": "A", "band": band, "year_used": year, "value": "v"}
            return "IMG", {**meta, "tree_crops": tree_crops}

        monkeypatch.setattr(lf, "crop_image", fake_crop_image)
        monkeypatch.setattr(
            lf, "server_zonal_means", lambda gdf, image, scale, batch_size: np.array([0.8, 0.1])
        )
        polys = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs="EPSG:32755")
        out = lf.filter_by_lulc(polys, _config(tmp_path, NAMOI, lulc_tree_crops=tree_crops))
        assert seen == [("dynamic_world", 2023, tree_crops)]
        stats = out.attrs["lulc_stats"]
        assert stats["tree_crops"] is tree_crops
        assert stats["band"] == ("crops+trees" if tree_crops else "crops")
        assert stats["n_kept"] == 1 and stats["n_below_threshold"] == 1

    def test_failures_become_runtime_error(self, tmp_path, nlcd_coverage, monkeypatch):
        monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
        monkeypatch.setattr("agribound.composites.gee.ee_geometry", lambda g: "REGION")

        def boom(*a, **k):
            raise ConnectionError("network down")

        monkeypatch.setattr(lf, "crop_image", boom)
        polys = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:32755")
        with pytest.raises(RuntimeError, match="network down"):
            lf.filter_by_lulc(polys, _config(tmp_path, NAMOI))


class _FakeEEError(Exception):
    pass


def _fake_ee_module(missing_every_other=True):
    ee = types.ModuleType("ee")
    ee.ee_exception = types.SimpleNamespace(EEException=_FakeEEError)
    ee.Geometry = lambda geojson, proj=None, geodesic=None: geojson
    ee.Feature = lambda geom, props: {"geometry": geom, "properties": props}
    ee.FeatureCollection = lambda features: features
    ee.Reducer = types.SimpleNamespace(mean=lambda: "mean")
    return ee


class _FakeReduced:
    def __init__(self, features):
        self.features = features

    def select(self, props, new=None, retain=True):
        assert props == ["_idx", "mean"] and retain is False
        return self

    def getInfo(self):  # noqa: N802 - ee API name
        out = []
        for f in self.features:
            idx = f["properties"]["_idx"]
            props = {"_idx": idx}
            if idx % 2 == 0:
                props["mean"] = idx / 10
            out.append({"properties": props})
        return {"features": out}


class _FakeImage:
    def __init__(self):
        self.batches = []

    def reduceRegions(self, collection, reducer, scale):  # noqa: N802 - ee API name
        self.batches.append((len(collection), scale))
        return _FakeReduced(collection)


def test_server_zonal_means_batches_and_nan(monkeypatch):
    monkeypatch.setitem(sys.modules, "ee", _fake_ee_module())
    image = _FakeImage()
    polys = gpd.GeoDataFrame(geometry=[box(i, 0, i + 0.5, 0.5) for i in range(5)], crs=4326)
    values = lf.server_zonal_means(polys, image, 30.0, batch_size=2)
    assert image.batches == [(2, 30.0), (2, 30.0), (1, 30.0)]
    np.testing.assert_allclose(values[[0, 2, 4]], [0.0, 0.2, 0.4])
    assert np.isnan(values[[1, 3]]).all()


class TestMisc:
    def test_empty_gdf(self, tmp_path):
        cfg = AgriboundConfig(
            study_area="missing.geojson",
            source="local",
            local_tif_path="test.tif",
            output_path=str(tmp_path / "t.gpkg"),
        )
        out = lf.filter_by_lulc(gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), cfg)
        assert len(out) == 0
        assert {"lulc:crop_fraction", "lulc:dataset", "lulc:year", "lulc:valid"} <= set(out.columns)
        assert out.attrs["lulc_stats"]["n_in"] == 0
        assert out.attrs["lulc_stats"]["tree_crops"] is False
        tree = lf.filter_by_lulc(
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), cfg.merged(lulc_tree_crops=True)
        )
        assert tree.attrs["lulc_stats"]["tree_crops"] is True

    def test_prefetch_disabled_returns_none(self, tmp_path):
        assert lf.prefetch_lulc_raster(_config(tmp_path, lulc_filter=False)) is None

    def test_prefetch_passes_tree_crops_and_tags_the_raster(
        self, tmp_path, nlcd_coverage, monkeypatch
    ):
        exports = []
        monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
        monkeypatch.setattr("agribound.composites.gee.ee_geometry", lambda geom: "REGION")

        def fake_crop_image(dataset, year, region, tree_crops=False):
            band = "crops+trees" if tree_crops else "crops"
            meta = {"asset": "DW", "band": band, "year_used": year, "value": f"v-{band}"}
            return "IMG", {**meta, "tree_crops": tree_crops}

        def fake_export(image, out_path, *, grid, dtype, band_names, tags, **kwargs):
            exports.append({"path": str(out_path), "tags": tags})
            open(out_path, "wb").close()
            return str(out_path)

        monkeypatch.setattr(lf, "crop_image", fake_crop_image)
        monkeypatch.setattr("agribound.composites.gee.export_ee_image", fake_export)
        cfg = _config(tmp_path, NAMOI, lulc_mode="raster")
        tree_path = lf.prefetch_lulc_raster(cfg.merged(lulc_tree_crops=True))
        default_path = lf.prefetch_lulc_raster(cfg)
        assert tree_path != default_path
        assert [e["path"] for e in exports] == [tree_path, default_path]
        tree_tags, default_tags = (e["tags"] for e in exports)
        assert tree_tags["AGRIBOUND_LULC_TREE_CROPS"] == "True"
        assert tree_tags["AGRIBOUND_LULC_BAND"] == "crops+trees"
        assert tree_tags["AGRIBOUND_LULC_VALUE"] == "v-crops+trees"
        assert default_tags["AGRIBOUND_LULC_TREE_CROPS"] == "False"
        assert default_tags["AGRIBOUND_LULC_BAND"] == "crops"
        # each raster is reused by its own rule
        assert lf.prefetch_lulc_raster(cfg.merged(lulc_tree_crops=True)) == tree_path
        assert len(exports) == 2

    def test_nlcd_raster_is_shared_and_tagged_alike_with_tree_crops(
        self, tmp_path, nlcd_coverage, monkeypatch
    ):
        """NLCD does not change with lulc_tree_crops: one raster, whichever run made it."""
        exports = []
        monkeypatch.setattr("agribound.composites.gee.ee_geometry", lambda geom: "REGION")

        def fake_crop_image(dataset, year, region, tree_crops=False):
            meta = {"asset": "N", "band": "b1", "year_used": year, "value": "v"}
            return "IMG", {**meta, "tree_crops": tree_crops}

        monkeypatch.setattr(lf, "crop_image", fake_crop_image)

        def fake_export(image, out_path, *, grid, dtype, band_names, tags, **kwargs):
            exports.append(tags)
            open(out_path, "wb").close()
            return str(out_path)

        monkeypatch.setattr("agribound.composites.gee.export_ee_image", fake_export)
        cfg = _config(tmp_path, IOWA, lulc_mode="raster", lulc_tree_crops=True)
        assert lf._select(cfg).dataset == "nlcd"
        path = lf.prefetch_lulc_raster(cfg)
        assert lf.prefetch_lulc_raster(cfg.merged(lulc_tree_crops=False)) == path
        (tags,) = exports
        assert tags["AGRIBOUND_LULC_TREE_CROPS"] == "False"  # tree crops did not change it

    def test_prefetch_reuses_cached_raster(self, namoi_raster_mode, monkeypatch):
        cfg, _ = namoi_raster_mode

        def no_gee(config):
            raise AssertionError("must not contact Earth Engine")

        monkeypatch.setattr("agribound.auth.ensure_gee", no_gee)
        path = lf.prefetch_lulc_raster(cfg)
        assert path.endswith(".tif") and "lulc_dynamic_world" in path


# ---------------------------------------------------------------------------
# Crop-image expressions (numeric ee stand-in), NLCD fallback, regions
# ---------------------------------------------------------------------------


class _Img:
    """Tiny numeric stand-in for the ee.Image operations used by crop_image."""

    def __init__(self, bands):
        self.bands = {k: np.asarray(v, dtype=np.float64) for k, v in bands.items()}

    def _one(self):
        (arr,) = self.bands.values()
        return arr

    def select(self, band):
        return _Img({band: self.bands[band]})

    def eq(self, v):
        return _Img({"x": (self._one() == v).astype(float)})

    def Or(self, other):  # noqa: N802 - ee API name
        return _Img({"x": ((self._one() != 0) | (other._one() != 0)).astype(float)})

    def remap(self, src, dst, default):
        out = np.full_like(self._one(), float(default))
        for s, d in zip(src, dst, strict=True):
            out[self._one() == s] = d
        return _Img({"x": out})

    def rename(self, name):
        return _Img({name: self._one()})

    def toFloat(self):  # noqa: N802 - ee API name
        return self


class _Collection:
    def __init__(self, asset, log, images):
        self.asset, self.log, self.images = asset, log, images

    def filter(self, flt):
        self.log.append((self.asset, flt))
        return self

    def first(self):
        return self.images[self.asset]


@pytest.fixture
def crop_ee(monkeypatch):
    log = []
    images = {
        lf.NLCD_ASSET: _Img({"b1": [[11, 81, 82, 21, 71]]}),
        lf.NLCD_FALLBACK_ASSET: _Img({"landcover": [[82, 41]]}),
        lf.CDL_ASSET: _Img({"cultivated": [[1, 2, 0]], "cropland": [[1, 1, 1]]}),
        lf.C3S_ASSET: _Img({"b1": [[10, 11, 12, 20, 30, 40, 50, 210]]}),
    }
    module = types.ModuleType("ee")
    module.ImageCollection = lambda asset: _Collection(asset, log, images)
    module.Image = lambda x: x
    module.Filter = types.SimpleNamespace(
        calendarRange=lambda a, b, unit: ("calendarRange", a, b, unit)
    )
    monkeypatch.setitem(sys.modules, "ee", module)
    return log


class TestCropImage:
    def test_nlcd_classes_81_82(self, crop_ee, monkeypatch):
        monkeypatch.setattr(lf, "_nlcd_source", lambda year: (lf.NLCD_ASSET, "b1", int(year)))
        image, meta = lf.crop_image("nlcd", 2023, "REGION")
        assert image.bands["crop"].tolist() == [[0, 1, 1, 0, 0]]
        assert meta == {
            "asset": lf.NLCD_ASSET,
            "band": "b1",
            "year_used": 2023,
            "value": lf.LULC_DATASETS["nlcd"].value,
            "tree_crops": False,
        }
        assert crop_ee == [(lf.NLCD_ASSET, ("calendarRange", 2023, 2023, "year"))]

    def test_nlcd_and_cdl_unchanged_by_tree_crops(self, crop_ee, monkeypatch):
        """NLCD 82 already includes orchards and vineyards; CDL counts them as cultivated."""
        monkeypatch.setattr(lf, "_nlcd_source", lambda year: (lf.NLCD_ASSET, "b1", int(year)))
        for dataset, year in (("nlcd", 2023), ("cdl", 2020)):
            image, meta = lf.crop_image(dataset, year, "REGION")
            tree_image, tree_meta = lf.crop_image(dataset, year, "REGION", tree_crops=True)
            assert tree_image.bands["crop"].tolist() == image.bands["crop"].tolist()
            assert {k: v for k, v in tree_meta.items() if k != "tree_crops"} == {
                k: v for k, v in meta.items() if k != "tree_crops"
            }
            assert tree_meta["value"] == lf.LULC_DATASETS[dataset].value

    def test_nlcd_fallback_release(self, crop_ee, monkeypatch):
        monkeypatch.setattr(
            lf, "_nlcd_source", lambda year: (lf.NLCD_FALLBACK_ASSET, "landcover", 2021)
        )
        image, meta = lf.crop_image("nlcd", 2024, "REGION")
        assert image.bands["crop"].tolist() == [[1, 0]]
        assert meta["asset"] == lf.NLCD_FALLBACK_ASSET and meta["year_used"] == 2021
        assert meta["band"] == "landcover"
        assert crop_ee == []  # the single-image 2021 release is not filtered by year

    def test_cdl_cultivated_band(self, crop_ee):
        image, meta = lf.crop_image("cdl", 2020, "REGION")
        assert image.bands["crop"].tolist() == [[0, 1, 0]]
        assert meta["band"] == "cultivated" and meta["year_used"] == 2020
        assert crop_ee == [(lf.CDL_ASSET, ("calendarRange", 2020, 2020, "year"))]

    def test_c3s_cropland_classes_include_11_and_12(self, crop_ee):
        image, meta = lf.crop_image("c3s", 2010, "REGION")
        assert image.bands["crop"].tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]
        assert meta["value"] == lf.LULC_DATASETS["c3s"].value and meta["tree_crops"] is False

    def test_c3s_tree_crops_add_the_tree_cover_classes(self, crop_ee, monkeypatch):
        lccs = [10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100]
        lccs += [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
        images = {lf.C3S_ASSET: _Img({"b1": [lccs]})}
        monkeypatch.setattr(
            sys.modules["ee"], "ImageCollection", lambda asset: _Collection(asset, crop_ee, images)
        )
        default, _ = lf.crop_image("c3s", 2015, "REGION")
        image, meta = lf.crop_image("c3s", 2015, "REGION", tree_crops=True)
        crop = dict(zip(lccs, default.bands["crop"][0], strict=True))
        assert {c for c, v in crop.items() if v == 1} == set(lf.C3S_CROP_CLASSES)
        tree = dict(zip(lccs, image.bands["crop"][0], strict=True))
        assert {c for c, v in tree.items() if v == 1} == set(
            lf.C3S_CROP_CLASSES + lf.C3S_TREE_CLASSES
        )
        assert meta == {
            "asset": lf.C3S_ASSET,
            "band": "b1",
            "year_used": 2015,
            "value": "fraction of C3S cropland or tree-cover pixels",
            "tree_crops": True,
        }

    def test_dynamic_world_uses_annual_median(self, crop_ee, monkeypatch):
        seen = {}

        def fake_dw(region, year, classes=("crops",)):
            seen["args"] = (region, year, classes)
            return _Img({"crop": [[0.2, 0.8]]})

        monkeypatch.setattr(
            "agribound.composites.dynamic_world.dynamic_world_crop_probability", fake_dw
        )
        image, meta = lf.crop_image("dynamic_world", 2023, "REGION")
        assert seen["args"] == ("REGION", 2023, ("crops",))
        assert image.bands["crop"].tolist() == [[0.2, 0.8]]
        assert meta == {
            "asset": "GOOGLE/DYNAMICWORLD/V1",
            "band": "crops",
            "year_used": 2023,
            "value": lf.LULC_DATASETS["dynamic_world"].value,
            "tree_crops": False,
        }

    def test_dynamic_world_tree_crops_count_crops_and_trees(self, crop_ee, monkeypatch):
        seen = {}

        def fake_dw(region, year, classes=("crops",)):
            seen["classes"] = classes
            return _Img({"crop": [[0.9, 0.1]]})

        monkeypatch.setattr(
            "agribound.composites.dynamic_world.dynamic_world_crop_probability", fake_dw
        )
        image, meta = lf.crop_image("dynamic_world", 2020, "REGION", tree_crops=True)
        assert seen["classes"] == ("crops", "trees")
        assert image.bands["crop"].tolist() == [[0.9, 0.1]]
        assert meta == {
            "asset": "GOOGLE/DYNAMICWORLD/V1",
            "band": "crops+trees",
            "year_used": 2020,
            "value": "mean annual-median Dynamic World crops+trees probability",
            "tree_crops": True,
        }

    def test_unknown_dataset(self, crop_ee):
        with pytest.raises(ValueError, match="Unknown LULC dataset 'worldcover'"):
            lf.crop_image("worldcover", 2021, "REGION")


@pytest.mark.usefixtures("needs_ee")
class TestNlcdSource:
    def test_other_earth_engine_errors_do_not_fall_back(self, monkeypatch):
        import ee

        def fail(fn, context="", **kwargs):
            raise ee.ee_exception.EEException(
                "Earth Engine client library not initialized. See http://goo.gle/ee-auth."
            )

        monkeypatch.setattr(lf, "_gee_call_with_retry", fail)
        with pytest.raises(ee.ee_exception.EEException, match="not initialized"):
            lf._nlcd_source(2023)
        assert 2023 not in lf._NLCD_SOURCE_CACHE

    def test_unreadable_annual_asset_falls_back_with_warning(self, monkeypatch, caplog):
        import ee

        def fail(fn, context="", **kwargs):
            raise ee.ee_exception.EEException(
                f"ImageCollection.load: ImageCollection asset '{lf.NLCD_ASSET}' not found "
                "(does not exist or caller does not have access)."
            )

        monkeypatch.setattr(lf, "_gee_call_with_retry", fail)
        with caplog.at_level("WARNING"):
            assert lf._nlcd_source(2023) == (lf.NLCD_FALLBACK_ASSET, "landcover", 2021)
        assert any("unavailable" in r.message for r in caplog.records)

    def test_missing_year_raises(self, monkeypatch):
        monkeypatch.setattr(lf, "_gee_call_with_retry", lambda fn, context="", **k: 0)
        with pytest.raises(RuntimeError, match="no image for 2023"):
            lf._nlcd_source(2023)

    def test_available_year_is_used_and_memoised(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            lf, "_gee_call_with_retry", lambda fn, context="", **k: calls.append(1) or 1
        )
        assert lf._nlcd_source(2020) == (lf.NLCD_ASSET, "b1", 2020)
        assert lf._nlcd_source(2020) == (lf.NLCD_ASSET, "b1", 2020)
        assert len(calls) == 1


def test_server_region_covers_polygons_outside_study_area(tmp_path):
    cfg = _config(tmp_path, NAMOI)
    far = box(149.10, -30.60, 149.12, -30.58)  # east and south of the study area
    gdf = gpd.GeoDataFrame(geometry=[far], crs="EPSG:4326")
    region = lf._selection_region_4326(cfg, gdf)
    assert region.contains(box(*NAMOI)) and region.contains(far)


@pytest.mark.usefixtures("needs_ee")
class TestGeeCallWithRetry:
    """Restricted-mode warnings are logged even when the Earth Engine call raises."""

    QUOTA = (
        "Your project has exceeded its noncommercial compute quota and is now in restricted mode."
    )

    def test_warning_before_rate_limit_is_logged_and_call_retried(self, monkeypatch, caplog):
        import warnings

        import ee

        monkeypatch.setattr(lf.time, "sleep", lambda s: None)
        attempts = []

        def fn():
            attempts.append(1)
            if len(attempts) == 1:
                warnings.warn(self.QUOTA, stacklevel=1)
                raise ee.ee_exception.EEException("429 Too Many Requests")
            return 5

        with caplog.at_level("WARNING"):
            assert lf._gee_call_with_retry(fn, context="unit") == 5
        assert len(attempts) == 2
        assert any("restricted mode" in r.message for r in caplog.records)

    def test_warning_before_other_error_is_logged_and_error_raised(self, caplog):
        import warnings

        import ee

        def fn():
            warnings.warn(self.QUOTA, stacklevel=1)
            raise ee.ee_exception.EEException("Image.load: asset not found")

        with caplog.at_level("WARNING"), pytest.raises(ee.ee_exception.EEException):
            lf._gee_call_with_retry(fn, context="unit")
        assert any("restricted mode" in r.message and "unit" in r.message for r in caplog.records)
