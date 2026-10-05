"""Unit tests for the Earth Engine composite builder (no network, no GEE).

Per-image masking/scaling functions are evaluated numerically on a small
numpy-backed stand-in for the ``ee.Image`` API, so the tests check pixel
values, not just call sequences.
"""

from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.geometry import box

from agribound.composites import gee
from agribound.config import AgriboundConfig

# ---------------------------------------------------------------------------
# numpy stand-in for the parts of ee.Image / ee.ImageCollection used here
# ---------------------------------------------------------------------------


class FakeImage:
    """Bands of equal-shape arrays with per-band masks (True = valid)."""

    def __init__(self, bands, masks=None, props=None):
        self.bands = {k: np.asarray(v, dtype=np.float64) for k, v in bands.items()}
        self.masks = {
            k: (np.ones_like(v, dtype=bool) if masks is None or k not in masks else masks[k])
            for k, v in self.bands.items()
        }
        self.props = dict(props or {})

    # band selection -------------------------------------------------------
    def select(self, names, new_names=None):
        names = [names] if isinstance(names, str) else list(names)
        new_names = names if new_names is None else list(new_names)
        return FakeImage(
            {n: self.bands[o] for o, n in zip(names, new_names, strict=True)},
            {n: self.masks[o] for o, n in zip(names, new_names, strict=True)},
            self.props,
        )

    def rename(self, *names):
        names = list(names[0]) if len(names) == 1 and isinstance(names[0], list) else list(names)
        old = list(self.bands)
        return FakeImage(
            {n: self.bands[o] for o, n in zip(old, names, strict=True)},
            {n: self.masks[o] for o, n in zip(old, names, strict=True)},
            self.props,
        )

    # arithmetic ------------------------------------------------------------
    def _map(self, fn):
        return FakeImage({k: fn(v) for k, v in self.bands.items()}, self.masks, self.props)

    def multiply(self, v):
        return self._map(lambda a: a * v)

    def add(self, v):
        return self._map(lambda a: a + v)

    def max(self, v):
        return self._map(lambda a: np.maximum(a, v))

    def bitwiseAnd(self, v):  # noqa: N802 - ee API name
        return self._map(lambda a: (a.astype(np.int64) & int(v)).astype(np.float64))

    def eq(self, v):
        return self._map(lambda a: (a == v).astype(np.float64))

    def neq(self, v):
        return self._map(lambda a: (a != v).astype(np.float64))

    def gte(self, v):
        return self._map(lambda a: (a >= v).astype(np.float64))

    def And(self, other):  # noqa: N802 - ee API name
        (a,) = self.bands.values()
        (b,) = other.bands.values()
        name = next(iter(self.bands))
        return FakeImage({name: ((a != 0) & (b != 0)).astype(np.float64)})

    def updateMask(self, other):  # noqa: N802 - ee API name
        (m,) = other.bands.values()
        (om,) = other.masks.values()
        valid = (m != 0) & om
        return FakeImage(self.bands, {k: v & valid for k, v in self.masks.items()}, self.props)

    def copyProperties(self, source, names):  # noqa: N802 - ee API name
        props = dict(self.props)
        props.update({k: source.props[k] for k in names if k in source.props})
        return FakeImage(self.bands, self.masks, props)

    def addBands(self, other):  # noqa: N802 - ee API name
        bands, masks = dict(self.bands), dict(self.masks)
        bands.update(other.bands)
        masks.update(other.masks)
        return FakeImage(bands, masks, self.props)

    def normalizedDifference(self, names):  # noqa: N802 - ee API name
        a, b = self.bands[names[0]], self.bands[names[1]]
        return FakeImage({"nd": (a - b) / (a + b)}, {"nd": self.masks[names[0]]})

    def clip(self, _region):
        return self

    def value(self, band):
        """Values with masked pixels as NaN (test helper)."""
        return np.where(self.masks[band], self.bands[band], np.nan)


class FakeCollection:
    def __init__(self, images):
        self.images = list(images)

    def map(self, fn):
        return FakeCollection([fn(i) for i in self.images])

    def median(self):
        names = list(self.images[0].bands)
        out = {}
        for n in names:
            stack = np.stack([i.value(n) for i in self.images])
            out[n] = np.nanmedian(stack, axis=0)
        return FakeImage(out, {n: np.isfinite(v) for n, v in out.items()})

    def qualityMosaic(self, band):  # noqa: N802 - ee API name
        q = np.stack([np.where(i.masks[band], i.bands[band], -np.inf) for i in self.images])
        best = np.argmax(q, axis=0)
        names = list(self.images[0].bands)
        out = {n: np.choose(best, [i.bands[n] for i in self.images]) for n in names}
        return FakeImage(out)


@pytest.fixture
def fake_ee(monkeypatch):
    module = types.ModuleType("ee")
    module.Image = lambda x: x
    monkeypatch.setitem(sys.modules, "ee", module)
    return module


# ---------------------------------------------------------------------------
# Masks and radiometry
# ---------------------------------------------------------------------------


class TestLandsat:
    def test_scaling_and_qa_mask(self, fake_ee):
        # 4 pixels: clear, cloud (bit 3), shadow (bit 4), dilated cloud (bit 1)
        qa = np.array([[0, 1 << 3, 1 << 4, 1 << 1]])
        dn = np.array([[10000.0, 20000.0, 30000.0, 40000.0]])
        bands = {b: dn for b in gee.LANDSAT_BANDS}
        bands["QA_PIXEL"] = qa
        img = FakeImage(bands, props={"system:time_start": 1})
        out = gee.prepare_landsat_image(img, gee.LANDSAT_BANDS)
        red = out.value("SR_B4")
        # (10000 * 2.75e-5 - 0.2) * 10000 = 750
        assert red[0, 0] == pytest.approx(750.0)
        assert np.isnan(red[0, 1:]).all()
        assert out.props["system:time_start"] == 1
        assert list(out.bands) == gee.LANDSAT_BANDS

    def test_qa_bits_0_to_4_masked_higher_bits_kept(self, fake_ee):
        # bits 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 shadow -> masked;
        # bits 5 snow, 6 clear, 7 water -> kept (not part of the mask)
        masked = [1 << b for b in range(5)]
        kept = [0, 1 << 5, 1 << 6, 1 << 7, (1 << 6) | (1 << 7)]
        qa = np.array([masked + kept], dtype=float)
        dn = np.full(qa.shape, 20000.0)
        bands = {b: dn for b in gee.LANDSAT_BANDS}
        bands["QA_PIXEL"] = qa
        red = gee.prepare_landsat_image(FakeImage(bands), gee.LANDSAT_BANDS).value("SR_B4")
        assert np.isnan(red[0, : len(masked)]).all()
        np.testing.assert_allclose(red[0, len(masked) :], (20000 * 2.75e-5 - 0.2) * 10000)

    def test_negative_reflectance_clipped_to_zero(self, fake_ee):
        dn = np.array([[5000.0, 7273.0]])  # 5000*2.75e-5-0.2 = -0.0625 -> 0
        bands = {b: dn for b in gee.LANDSAT_BANDS}
        bands["QA_PIXEL"] = np.zeros((1, 2))
        out = gee.prepare_landsat_image(FakeImage(bands), gee.LANDSAT_BANDS)
        assert out.value("SR_B2")[0, 0] == 0.0
        assert out.value("SR_B2")[0, 1] == pytest.approx((7273 * 2.75e-5 - 0.2) * 10000)

    def test_l57_bands_renamed_to_l89(self, fake_ee):
        bands = {
            b: np.full((1, 1), float(i + 1) * 10000) for i, b in enumerate(gee.LANDSAT_L57_SR_BANDS)
        }
        bands["QA_PIXEL"] = np.zeros((1, 1))
        out = gee.prepare_landsat_image(FakeImage(bands), gee.LANDSAT_L57_SR_BANDS)
        # L5/7 SR_B3 (red, 3rd in the list) must end up in L8/9 SR_B4
        expected_red = (3 * 10000 * 2.75e-5 - 0.2) * 10000
        assert out.value("SR_B4")[0, 0] == pytest.approx(expected_red)
        # L5/7 SR_B7 (SWIR2, last) -> SR_B7
        assert out.value("SR_B7")[0, 0] == pytest.approx((6 * 10000 * 2.75e-5 - 0.2) * 10000)


class TestHLS:
    def test_s30_mapping_uses_b8a_b11_b12(self):
        assert gee.HLSS30_SOURCE_BANDS == ["B1", "B2", "B3", "B4", "B8A", "B11", "B12"]
        assert gee.HLS_BANDS == ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]

    def test_s30_swir_not_red_edge(self, fake_ee):
        src = {
            b: np.full((1, 1), 0.01 * (i + 1))
            for i, b in enumerate(
                ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
            )
        }
        src["Fmask"] = np.zeros((1, 1))
        out = gee.prepare_hls_image(FakeImage(src), gee.HLSS30_SOURCE_BANDS)
        assert out.value("B6")[0, 0] == pytest.approx(src["B11"][0, 0] * 10000)
        assert out.value("B7")[0, 0] == pytest.approx(src["B12"][0, 0] * 10000)
        assert out.value("B5")[0, 0] == pytest.approx(src["B8A"][0, 0] * 10000)

    def test_fmask_bits_1_to_3_masked_only(self, fake_ee):
        # bit0 cirrus kept, bit1 cloud, bit2 adjacent, bit3 shadow masked, bit4 snow kept
        fmask = np.array([[1, 2, 4, 8, 16, 0]])
        src = {b: np.full((1, 6), 0.1) for b in gee.HLS_BANDS}
        src["Fmask"] = fmask
        out = gee.prepare_hls_image(FakeImage(src), gee.HLS_BANDS)
        valid = np.isfinite(out.value("B4"))[0]
        assert valid.tolist() == [True, False, False, False, True, True]
        assert out.value("B4")[0, 0] == pytest.approx(1000.0)


class TestSentinel2:
    def test_scl_classes(self, fake_ee):
        scl = np.array([[3, 4, 5, 8, 9, 10, 11, 7]])
        bands = {b: np.full((1, 8), 1500.0) for b in gee.S2_BANDS}
        bands["SCL"] = scl
        out = gee.mask_s2_scl(FakeImage(bands))
        valid = np.isfinite(out.value("B4"))[0]
        assert valid.tolist() == [False, True, True, False, False, False, True, True]
        assert out.value("B4")[0, 1] == 1500.0  # unchanged scale
        assert list(out.bands) == gee.S2_BANDS

    def test_cloud_score_threshold(self, fake_ee):
        bands = {b: np.full((1, 3), 900.0) for b in gee.S2_BANDS}
        bands["cs_cdf"] = np.array([[0.59, 0.60, 0.9]])
        out = gee.mask_s2_cloud_score(FakeImage(bands), 0.60)
        assert np.isfinite(out.value("B2"))[0].tolist() == [False, True, True]
        assert "cs_cdf" not in out.bands


class TestCompositeMethod:
    def _collection(self):
        # image A: low NDVI, image B: high NDVI at pixel 0; reversed at pixel 1
        a = FakeImage({"B4": [[0.1, 0.1]], "B8": [[0.2, 0.9]], "B2": [[1.0, 1.0]]})
        b = FakeImage({"B4": [[0.1, 0.1]], "B8": [[0.9, 0.2]], "B2": [[2.0, 2.0]]})
        return FakeCollection([a, b])

    def test_greenest_picks_max_ndvi_image(self, monkeypatch):
        monkeypatch.setitem(
            gee.SOURCE_REGISTRY,
            "fake",
            {"canonical_bands": {"NIR": "B8", "R": "B4"}, "all_bands": ["B2", "B4", "B8"]},
        )
        out = gee.apply_composite_method(self._collection(), "greenest", "fake")
        assert out.value("B2")[0].tolist() == [2.0, 1.0]
        assert "NDVI" not in out.bands

    def test_max_ndvi_is_alias_of_greenest(self, monkeypatch):
        monkeypatch.setitem(
            gee.SOURCE_REGISTRY,
            "fake",
            {"canonical_bands": {"NIR": "B8", "R": "B4"}, "all_bands": ["B2", "B4", "B8"]},
        )
        g = gee.apply_composite_method(self._collection(), "greenest", "fake")
        m = gee.apply_composite_method(self._collection(), "max_ndvi", "fake")
        for band in g.bands:
            np.testing.assert_array_equal(g.value(band), m.value(band))

    def test_greenest_without_nir_raises(self):
        with pytest.raises(ValueError, match="needs NIR"):
            gee.apply_composite_method(self._collection(), "greenest", "spot-pan")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown composite_method"):
            gee.apply_composite_method(self._collection(), "mean", "sentinel2")


# ---------------------------------------------------------------------------
# Dates, CRS and grids
# ---------------------------------------------------------------------------


class TestDateWindow:
    def test_calendar_year_end_exclusive(self):
        cfg = AgriboundConfig(source="local", local_tif_path="x.tif", year=2023)
        assert gee.date_window(cfg) == ("2023-01-01", "2024-01-01")

    def test_date_range_end_inclusive(self):
        cfg = AgriboundConfig(
            source="local", local_tif_path="x.tif", date_range=("2023-04-01", "2023-06-30")
        )
        assert gee.date_window(cfg) == ("2023-04-01", "2023-07-01")


class TestExportGrid:
    def test_utm_resolution(self):
        # Namoi test AOI (southern hemisphere, zone 55/56 boundary region)
        geom = box(149.0, -30.5, 149.05, -30.45)
        assert gee.resolve_export_crs("utm", geom) == "EPSG:32755"
        assert gee.resolve_export_crs("EPSG:3857", geom) == "EPSG:3857"

    def test_grid_is_aligned_and_covers_geometry(self):
        geom = box(149.0, -30.5, 149.05, -30.45)
        grid = gee.compute_export_grid(geom, "EPSG:32755", 10.0)
        t = grid.transform
        assert t.a == 10.0 and t.e == -10.0
        assert t.c % 10 == pytest.approx(0) and t.f % 10 == pytest.approx(0)
        import geopandas as gpd

        proj = gpd.GeoSeries([geom], crs=4326).to_crs(32755).iloc[0]
        minx, miny, maxx, maxy = proj.bounds
        assert t.c <= minx and t.f >= maxy
        assert t.c + grid.width * 10 >= maxx and t.f - grid.height * 10 <= miny
        # a 5 km box at 10 m is ~ 480 x 555 px
        assert 450 < grid.width < 520 and 520 < grid.height < 580

    def test_geographic_crs_uses_degree_scale(self):
        grid = gee.compute_export_grid(box(10, 10, 10.01, 10.01), "EPSG:4326", 10.0)
        assert grid.transform.a == pytest.approx(10.0 / gee.METRES_PER_DEGREE)

    def test_empty_geometry_raises(self):
        from shapely.geometry import Polygon

        with pytest.raises(ValueError):
            gee.compute_export_grid(Polygon(), "EPSG:32611", 10)

    def test_split_grid_tiles_cover_grid_without_overlap(self):
        grid = gee.ExportGrid("EPSG:32611", Affine(10, 0, 0, 0, -10, 0), 2501, 1203)
        tiles = gee.split_grid(grid, 1000)
        assert len(tiles) == 3 * 2
        cover = np.zeros((grid.height, grid.width), dtype=int)
        for r, c, h, w in tiles:
            assert h <= 1000 and w <= 1000
            cover[r : r + h, c : c + w] += 1
        assert (cover == 1).all()

    def test_window_transform(self):
        grid = gee.ExportGrid("EPSG:32611", Affine(10, 0, 500000, 0, -10, 4000000), 100, 100)
        sub = grid.window(20, 30, 10, 10)
        assert sub.transform.c == 500300 and sub.transform.f == 3999800
        assert sub.crs_transform == [10.0, 0.0, 500300.0, 0.0, -10.0, 3999800.0]


# ---------------------------------------------------------------------------
# Tile assembly and warnings
# ---------------------------------------------------------------------------


def _write(path, data, transform, crs="EPSG:32611", nodata=None):
    count, h, w = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=count,
        height=h,
        width=w,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


class TestAssembleTiles:
    def test_neg_inf_becomes_nan_and_tiles_placed(self, tmp_path):
        grid = gee.ExportGrid("EPSG:32611", Affine(10, 0, 0, 0, -10, 40), 4, 4)
        tiles = []
        for i, (r, c, h, w) in enumerate(gee.split_grid(grid, 2)):
            data = np.full((2, h, w), float(i), dtype=np.float32)
            data[0, 0, 0] = -np.inf
            sub = grid.window(r, c, h, w)
            tiles.append((_write(tmp_path / f"t{i}.tif", data, sub.transform), (r, c, h, w)))
        out = gee.assemble_tiles(
            tiles,
            tmp_path / "out.tif",
            grid,
            dtype="float32",
            band_names=["a", "b"],
            tags={"AGRIBOUND_SOURCE": "x"},
        )
        with rasterio.open(out) as src:
            arr = src.read()
            assert np.isnan(src.nodata)
            assert src.descriptions == ("a", "b")
            assert src.tags()["AGRIBOUND_SOURCE"] == "x"
            assert src.transform == grid.transform
        assert np.isnan(arr[0, 0, 0]) and np.isnan(arr[0, 2, 2])
        assert arr[1, 0, 0] == 0 and arr[1, 3, 3] == 3
        assert not (tmp_path / "out.tif.part").exists()

    def test_shape_mismatch_raises(self, tmp_path):
        grid = gee.ExportGrid("EPSG:32611", Affine(10, 0, 0, 0, -10, 40), 4, 4)
        bad = _write(tmp_path / "bad.tif", np.zeros((1, 3, 3), np.float32), grid.transform)
        with pytest.raises(RuntimeError, match="expected"):
            gee.assemble_tiles(
                [(bad, (0, 0, 4, 4))], tmp_path / "o.tif", grid, dtype="float32", band_names=["a"]
            )
        assert not (tmp_path / "o.tif").exists()


class TestWarningMonitor:
    def test_restricted_mode_logged_and_flagged(self, caplog):
        with gee.ee_warning_monitor("unit") as state:
            warnings.warn(
                "Your project has exceeded its noncommercial compute quota and is now in "
                "restricted mode.",
                stacklevel=1,
            )
        assert state.restricted_mode
        assert any("restricted mode" in r.message for r in caplog.records)

    def test_other_warnings_re_emitted(self):
        with (
            pytest.warns(UserWarning, match="unrelated"),
            gee.ee_warning_monitor("unit") as state,
        ):
            warnings.warn("unrelated", UserWarning, stacklevel=1)
        assert not state.restricted_mode

    def test_geedim_no_stac_warning_for_computed_images_is_dropped(self, recwarn):
        with gee.ee_warning_monitor("unit"):
            # The exact text of geedim 2.x (geedim/stac.py) for an image without an ID.
            warnings.warn("Couldn't find STAC entry for: 'None'.", RuntimeWarning, stacklevel=1)
        assert not [w for w in recwarn if "STAC" in str(w.message)]
        # A real asset ID without a STAC entry is still reported, as is another category.
        with (
            pytest.warns(RuntimeWarning, match="USDA/NAIP/DOQQ"),
            gee.ee_warning_monitor("unit"),
        ):
            warnings.warn(
                "Couldn't find STAC entry for: 'USDA/NAIP/DOQQ'.", RuntimeWarning, stacklevel=1
            )
        with (
            pytest.warns(UserWarning, match="STAC"),
            gee.ee_warning_monitor("unit"),
        ):
            warnings.warn("Couldn't find STAC entry for: 'None'.", UserWarning, stacklevel=1)

    def test_warnings_processed_when_the_call_raises(self, caplog):
        # A restricted-mode warning right before a failed (e.g. throttled) request
        # must still be logged and flagged, and other warnings re-emitted.
        with (
            pytest.warns(UserWarning, match="unrelated"),
            pytest.raises(RuntimeError, match="429"),
            gee.ee_warning_monitor("unit") as state,
        ):
            warnings.warn(
                "Your project has exceeded its noncommercial compute quota and is now in "
                "restricted mode.",
                stacklevel=1,
            )
            warnings.warn("unrelated", UserWarning, stacklevel=1)
            raise RuntimeError("429 Too Many Requests")
        assert state.restricted_mode
        assert any(
            r.levelname == "WARNING" and "restricted mode" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Builder with the Earth Engine layer stubbed out
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_builder(monkeypatch):
    """Stub every Earth Engine call of GEECompositeBuilder.build; record exports."""
    calls = {
        "export": [],
        "summary": {"n_images": 3, "years": [2023]},
        "years": [2019, 2021],
        "nan_fraction": 0.0,  # share of grid columns the fake download leaves NaN
    }

    monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
    monkeypatch.setattr(gee, "ee_geometry", lambda geom: "REGION")

    def fake_spec(config, region):
        img = FakeImage({b: np.ones((1, 1)) for b in gee.S2_BANDS})
        return gee.CollectionSpec(
            raw="RAW",
            prepared=FakeCollection([img]),
            history="HISTORY",
            bands=list(gee.S2_BANDS),
            resolution_m=10.0,
            dtype="float32",
            collections=[gee.S2_COLLECTION],
        )

    monkeypatch.setitem(gee._COLLECTION_BUILDERS, "sentinel2", fake_spec)
    monkeypatch.setattr(gee, "collection_summary", lambda col, context="": calls["summary"])
    monkeypatch.setattr(gee, "available_years", lambda col, context="": calls["years"])

    def fake_export(image, out_path, *, grid, dtype, band_names, tags=None, **kwargs):
        calls["export"].append({"path": str(out_path), "grid": grid, "tags": tags})
        data = np.ones((len(band_names), grid.height, grid.width), dtype=np.float32)
        data[:, :, : int(round(grid.width * calls["nan_fraction"]))] = np.nan
        _write(out_path, data, grid.transform, crs=grid.crs, nodata=float("nan"))
        with rasterio.open(out_path, "r+") as dst:
            dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
        return str(out_path)

    monkeypatch.setattr(gee, "export_ee_image", fake_export)
    return calls


def _s2_config(tmp_path, aoi, **kwargs):
    return AgriboundConfig(
        source="sentinel2",
        year=2023,
        study_area=aoi,
        gee_project="test-project",
        output_path=str(tmp_path / "out.gpkg"),
        lulc_filter=False,
        **kwargs,
    )


def _patch_task_status(monkeypatch, fn):
    """Make ``ee.data.getTaskStatus`` call *fn*, with or without earthengine-api.

    The builder imports ``ee`` only to query a recorded batch task, so without
    the ``gee`` extra (the CI core job) a stand-in ``ee`` module is enough.
    """
    try:
        import ee
    except ImportError:
        ee = types.ModuleType("ee")
        ee.data = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.setattr(ee.data, "getTaskStatus", fn, raising=False)


class TestBuilder:
    def test_landsat_pan_export_and_provenance(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch
    ):
        cfg = _cfg("landsat-pan", year=2023).merged(
            study_area=sample_aoi_geojson, output_path=str(tmp_path / "out.gpkg")
        )
        spec = gee._build_landsat_pan(cfg, "REGION")
        spec.prepared = FakeCollection([FakeImage({"B8": [[0.25]]})])
        monkeypatch.setitem(gee._COLLECTION_BUILDERS, "landsat-pan", lambda cfg, region: spec)
        path = gee.GEECompositeBuilder().build(cfg)
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.res == (15, 15)
            tags = src.tags()
        assert tags["AGRIBOUND_SOURCE"] == "landsat-pan"
        assert tags["AGRIBOUND_VALUE_SCALE"] == "unit"
        assert tags["AGRIBOUND_SENSORS"] == "LE07,LC08,LC09"
        assert tags["AGRIBOUND_COLLECTIONS"] == ",".join(spec.collections)
        assert "no special gap filling" in tags["AGRIBOUND_SLC_OFF"]

    def test_build_exports_utm_grid_and_tags(self, tmp_path, sample_aoi_geojson, stub_builder):
        path = gee.GEECompositeBuilder().build(_s2_config(tmp_path, sample_aoi_geojson))
        (call,) = stub_builder["export"]
        assert call["path"] == path
        assert call["grid"].crs == "EPSG:32611"
        assert call["tags"]["AGRIBOUND_VALUE_SCALE"] == "reflectance_x10000"
        assert call["tags"]["AGRIBOUND_DATE_END_EXCLUSIVE"] == "2024-01-01"
        assert call["tags"]["AGRIBOUND_N_IMAGES"] == 3

    def test_cached_composite_is_reused(self, tmp_path, sample_aoi_geojson, stub_builder):
        cfg = _s2_config(tmp_path, sample_aoi_geojson)
        first = gee.GEECompositeBuilder().build(cfg)
        second = gee.GEECompositeBuilder().build(cfg)
        assert first == second
        assert len(stub_builder["export"]) == 1

    def test_date_windows_get_distinct_files(self, tmp_path, sample_aoi_geojson, stub_builder):
        a = _s2_config(tmp_path, sample_aoi_geojson, date_range=("2023-03-01", "2023-05-31"))
        b = _s2_config(tmp_path, sample_aoi_geojson, date_range=("2023-09-01", "2023-11-30"))
        pa = gee.GEECompositeBuilder().build(a)
        pb = gee.GEECompositeBuilder().build(b)
        assert pa != pb
        assert len(stub_builder["export"]) == 2
        assert stub_builder["export"][0]["tags"]["AGRIBOUND_DATE_START"] == "2023-03-01"
        assert stub_builder["export"][1]["tags"]["AGRIBOUND_DATE_START"] == "2023-09-01"

    def test_other_settings_change_the_cache_file(self, tmp_path, sample_aoi_geojson, stub_builder):
        base = _s2_config(tmp_path, sample_aoi_geojson)
        paths = {
            gee.GEECompositeBuilder().build(base),
            gee.GEECompositeBuilder().build(base.merged(cloud_cover_max=50)),
            gee.GEECompositeBuilder().build(base.merged(s2_cloud_mask="cloud_score_plus")),
            gee.GEECompositeBuilder().build(base.merged(export_crs="EPSG:3857")),
        }
        assert len(paths) == 4

    def test_empty_collection_lists_available_years(
        self, tmp_path, sample_aoi_geojson, stub_builder
    ):
        stub_builder["summary"] = {"n_images": 0, "years": []}
        with pytest.raises(
            ValueError, match=r"Years with images over the study-area extent: 2019, 2021"
        ):
            gee.GEECompositeBuilder().build(_s2_config(tmp_path, sample_aoi_geojson))
        assert stub_builder["export"] == []

    def test_naip_rejects_greenest(self, tmp_path, sample_aoi_geojson, stub_builder):
        cfg = AgriboundConfig(
            source="naip",
            year=2022,
            study_area=sample_aoi_geojson,
            gee_project="p",
            output_path=str(tmp_path / "o.gpkg"),
            composite_method="greenest",
        )
        with pytest.raises(ValueError, match="mosaicked"):
            gee.GEECompositeBuilder().build(cfg)

    def test_batch_export_raises_after_starting_task(
        self, tmp_path, sample_aoi_geojson, stub_builder, monkeypatch
    ):
        started = []

        def fake_start(image, **kwargs):
            started.append(kwargs)
            return {"id": "T1", "description": kwargs["description"], "destination": "gs://b/x"}

        monkeypatch.setattr(gee, "start_batch_export", fake_start)
        cfg = _s2_config(tmp_path, sample_aoi_geojson, export_method="gcs", gcs_bucket="b")
        with pytest.raises(gee.ExportTaskStartedError) as info:
            gee.GEECompositeBuilder().build(cfg)
        assert info.value.tasks[0]["id"] == "T1"
        assert started[0]["method"] == "gcs"
        assert stub_builder["export"] == []

    @pytest.mark.parametrize(
        ("state", "starts_new"),
        [
            ("READY", False),
            ("RUNNING", False),
            ("COMPLETED", False),
            ("FAILED", True),
            ("CANCELLED", True),
            ("UNKNOWN", True),
        ],
    )
    def test_rerun_reuses_recorded_batch_task(
        self, tmp_path, sample_aoi_geojson, stub_builder, monkeypatch, state, starts_new
    ):
        import json

        started = []

        def fake_start(image, **kwargs):
            started.append(kwargs)
            tid = f"T{len(started)}"
            return {"id": tid, "description": kwargs["description"], "destination": "gs://b/x"}

        status_calls = []

        def fake_status(task_id):
            status_calls.append(task_id)
            return [{"id": task_id, "state": state, "error_message": "boom"}]

        monkeypatch.setattr(gee, "start_batch_export", fake_start)
        _patch_task_status(monkeypatch, fake_status)
        cfg = _s2_config(tmp_path, sample_aoi_geojson, export_method="gcs", gcs_bucket="b")
        with pytest.raises(gee.ExportTaskStartedError):
            gee.GEECompositeBuilder().build(cfg)
        (marker,) = (tmp_path / ".agribound_cache").glob("sentinel2_composite_*_task.json")
        assert json.loads(marker.read_text())["id"] == "T1"
        assert status_calls == []  # no record yet on the first run

        with pytest.raises(gee.ExportTaskStartedError) as info:
            gee.GEECompositeBuilder().build(cfg)
        assert status_calls == ["T1"]
        assert len(started) == (2 if starts_new else 1)
        if starts_new:
            assert info.value.tasks[0]["id"] == "T2"
            assert json.loads(marker.read_text())["id"] == "T2"
        else:
            assert info.value.tasks[0]["id"] == "T1"
            assert info.value.tasks[0]["state"] == state
            assert "no new task" in str(info.value) or state == "COMPLETED"

    def test_unqueryable_recorded_task_raises_without_starting(
        self, tmp_path, sample_aoi_geojson, stub_builder, monkeypatch
    ):
        started = []
        monkeypatch.setattr(
            gee,
            "start_batch_export",
            lambda image, **kw: started.append(kw) or {"id": "T1", "destination": "d"},
        )
        cfg = _s2_config(tmp_path, sample_aoi_geojson, export_method="gcs", gcs_bucket="b")
        with pytest.raises(gee.ExportTaskStartedError):
            gee.GEECompositeBuilder().build(cfg)

        def broken(task_id):
            raise ConnectionError("offline")

        _patch_task_status(monkeypatch, broken)
        with pytest.raises(RuntimeError, match="Could not query the status"):
            gee.GEECompositeBuilder().build(cfg)
        assert len(started) == 1


class TestValidFraction:
    def test_all_any_nodata_and_geometry(self, tmp_path):
        # 4 x 4 grid, 10 m pixels, UTM 11N; left column NaN in band 1 only.
        t = Affine(10, 0, 500000, 0, -10, 4000040)
        data = np.ones((2, 4, 4), np.float32)
        data[0, :, 0] = np.nan
        path = _write(tmp_path / "f.tif", data, t, nodata=float("nan"))
        import pyproj
        from shapely.ops import transform as shp_transform

        to_4326 = pyproj.Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
        whole = shp_transform(to_4326.transform, box(500000, 4000000, 500040, 4000040))
        right = shp_transform(to_4326.transform, box(500010, 4000000, 500040, 4000040))
        assert gee.raster_valid_fraction(path, whole, require="all") == pytest.approx(0.75)
        assert gee.raster_valid_fraction(path, whole, require="any") == pytest.approx(1.0)
        assert gee.raster_valid_fraction(path, right, require="all") == pytest.approx(1.0)
        # blocks of one row give the same answer
        assert gee.raster_valid_fraction(path, whole, rows_per_block=1) == pytest.approx(0.75)
        # uint8 with nodata 0: a pixel is uncovered when every band is 0
        u = np.full((2, 4, 4), 7, np.uint8)
        u[:, 0, :] = 0
        u[0, 1, :] = 0  # dark in one band only: still covered with require="any"
        upath = _write(tmp_path / "u.tif", u, t, nodata=0)
        assert gee.raster_valid_fraction(upath, whole, require="any") == pytest.approx(0.75)
        assert gee.raster_valid_fraction(upath, whole, require="all") == pytest.approx(0.5)

    def test_block_size_is_capped_by_bytes(self, tmp_path, monkeypatch):
        t = Affine(10, 0, 500000, 0, -10, 4000040)
        path = _write(tmp_path / "f.tif", np.ones((3, 4, 4), np.float32), t, nodata=float("nan"))
        import rasterio as rio

        windows = []
        real_read = rio.io.DatasetReader.read

        def spy(self, *a, **k):
            windows.append(k.get("window"))
            return real_read(self, *a, **k)

        monkeypatch.setattr(rio.io.DatasetReader, "read", spy)
        # one row of 3 float32 bands x 4 columns is 48 bytes: 100 bytes -> 2 rows per block
        gee.raster_valid_fraction(path, box(-117, 36, -116, 37), max_block_bytes=100)
        assert [int(w.height) for w in windows] == [2, 2]

    def test_empty_composite_raises_and_is_removed(
        self, tmp_path, sample_aoi_geojson, stub_builder
    ):
        stub_builder["nan_fraction"] = 1.0
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError, match="no valid pixel inside the study area"):
            gee.GEECompositeBuilder().build(_s2_config(tmp_path, sample_aoi_geojson))
        assert not list((tmp_path / ".agribound_cache").glob("sentinel2_composite_*.tif"))

    def test_low_coverage_warns_and_is_tagged(
        self, tmp_path, sample_aoi_geojson, stub_builder, caplog
    ):
        stub_builder["nan_fraction"] = 0.5
        builder = gee.GEECompositeBuilder()
        cfg = _s2_config(tmp_path, sample_aoi_geojson)
        with caplog.at_level("WARNING"):
            path = builder.build(cfg)
        fraction = float(builder.last_metadata["AGRIBOUND_VALID_FRACTION"])
        assert 0.3 < fraction < 0.7
        with rasterio.open(path) as src:
            assert float(src.tags()["AGRIBOUND_VALID_FRACTION"]) == pytest.approx(fraction)
        assert any("valid data in only" in r.message for r in caplog.records)
        # the warning is repeated when the cached composite is reused
        caplog.clear()
        with caplog.at_level("WARNING"):
            assert gee.GEECompositeBuilder().build(cfg) == path
        assert any("cached sentinel2 composite" in r.message for r in caplog.records)

    def test_full_coverage_does_not_warn(self, tmp_path, sample_aoi_geojson, stub_builder, caplog):
        with caplog.at_level("WARNING"):
            builder = gee.GEECompositeBuilder()
            builder.build(_s2_config(tmp_path, sample_aoi_geojson))
        assert float(builder.last_metadata["AGRIBOUND_VALID_FRACTION"]) == 1.0
        assert not any("valid data in only" in r.message for r in caplog.records)


def test_read_composite_tags_filters_agribound_keys(tmp_path):
    path = _write(tmp_path / "t.tif", np.zeros((1, 2, 2), np.uint8), Affine(1, 0, 0, 0, -1, 2))
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(AGRIBOUND_SOURCE="naip", OTHER="x")
    assert gee.read_composite_tags(path) == {"AGRIBOUND_SOURCE": "naip"}
    assert gee.read_composite_tags(tmp_path / "missing.tif") == {}


# ---------------------------------------------------------------------------
# Collection construction (filters, bands, resolution) with a recording ee stub
# ---------------------------------------------------------------------------


class Chain:
    """Records every method call made on an Earth Engine object chain."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)

        def call(*args, **kwargs):
            self.log.append((self.name, attr, args, kwargs))
            return Chain(f"{self.name}.{attr}", self.log)

        return call


@pytest.fixture
def recording_ee(monkeypatch):
    log = []
    module = types.ModuleType("ee")
    module.ImageCollection = lambda cid: Chain(f"IC({cid})", log)
    module.Image = lambda x: x
    module.Filter = types.SimpleNamespace(
        lte=lambda prop, value: ("lte", prop, value),
        listContains=lambda prop, value: ("listContains", prop, value),
    )
    monkeypatch.setitem(sys.modules, "ee", module)
    return log


def _calls(log, prefix, attr):
    return [(name, args) for name, a, args, _ in log if name.startswith(prefix) and a == attr]


def _cfg(source, year=2022, **kwargs):
    return AgriboundConfig(
        source=source,
        year=year,
        study_area="bbox:-104.2,35.0,-104.19,35.01",
        gee_project="p",
        lulc_filter=False,
        **kwargs,
    )


@pytest.mark.parametrize("mission", ["LE07", "LC08", "LC09"])
def test_landsat_pan_mask_and_radiometry(mission):
    qa = np.array([[0, 1, 2, 4, 8, 16, 32, 128]])
    values = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.1]])
    image = FakeImage({"B8": values, "QA_PIXEL": qa}, props={"system:time_start": 123})
    out = gee.prepare_landsat_pan_image(image, mission)
    np.testing.assert_array_equal(out.bands["B8"], values)
    np.testing.assert_array_equal(
        out.masks["B8"], [[True, False, False, mission == "LE07", False, False, True, True]]
    )
    assert out.props["system:time_start"] == 123


@pytest.mark.parametrize("method", ["greenest", "max_ndvi"])
def test_landsat_pan_rejects_ndvi_composites(method):
    with pytest.raises(ValueError, match="NIR and red"):
        gee.apply_composite_method(None, method, "landsat-pan")


class TestCollectionSpecs:
    @pytest.mark.parametrize(
        ("year", "missions"),
        [
            (1999, ["LE07"]),
            (2015, ["LE07", "LC08"]),
            (2023, ["LE07", "LC08", "LC09"]),
            (2025, ["LC08", "LC09"]),
        ],
    )
    def test_landsat_pan_missions(self, recording_ee, year, missions):
        spec = gee._build_landsat_pan(_cfg("landsat-pan", year=year), "REGION")
        assert spec.collections == [gee.LANDSAT_PAN_COLLECTIONS[m][0] for m in missions]
        assert spec.bands == ["B8"]
        assert spec.resolution_m == 15 and spec.dtype == "float32"
        assert spec.notes["sensors"] == ",".join(missions)
        filters = _calls(recording_ee, "IC(LANDSAT", "filter")
        assert len(filters) == 3
        assert all(args == (("lte", "CLOUD_COVER", 20),) for _, args in filters)

    def test_landsat_pan_date_range(self, recording_ee):
        cfg = _cfg("landsat-pan", date_range=("2012-12-01", "2013-04-01"))
        spec = gee._build_landsat_pan(cfg, "REGION")
        assert len(spec.collections) == 2
        assert all(
            args == ("2012-12-01", "2013-04-02")
            for _, args in _calls(recording_ee, "IC(LANDSAT", "filterDate")
        )

    def test_naip_band_filter_year_window_and_order(self, recording_ee):
        spec = gee._build_naip(_cfg("naip", naip_resolution_m=0.6), "REGION")
        naip = f"IC({gee.NAIP_COLLECTION})"
        assert _calls(recording_ee, naip, "filterBounds") == [(naip, ("REGION",))]
        # only 4-band images, applied before the date filter
        assert _calls(recording_ee, naip, "filter") == [
            (f"{naip}.filterBounds", (("listContains", "system:band_names", "N"),))
        ]
        # exact year +- 1, end exclusive
        assert _calls(recording_ee, naip, "filterDate")[0][1] == ("2021-01-01", "2024-01-01")
        assert _calls(recording_ee, naip, "sort")[0][1] == (gee._NAIP_ORDER_PROPERTY,)
        assert _calls(recording_ee, naip, "select")[0][1] == (gee.NAIP_BANDS,)
        assert spec.bands == ["R", "G", "B", "N"]
        assert spec.dtype == "uint8" and spec.resolution_m == 0.6
        # history without the band filter is kept to report RGB-only years
        assert spec.history_all_bands.name == f"{naip}.filterBounds"

    def test_naip_date_range_sorts_by_time(self, recording_ee):
        gee._build_naip(_cfg("naip", date_range=("2022-05-01", "2022-09-30")), "R")
        naip = f"IC({gee.NAIP_COLLECTION})"
        assert _calls(recording_ee, naip, "filterDate")[0][1] == ("2022-05-01", "2022-10-01")
        assert _calls(recording_ee, naip, "sort")[0][1] == ("system:time_start",)

    @pytest.mark.parametrize(
        ("source", "bands", "res"), [("spot", ["R", "G", "B", "N"], 6.0), ("spot-pan", ["P"], 1.5)]
    )
    def test_spot_bands_and_cloud_filter(self, recording_ee, source, bands, res):
        spec = gee._build_spot(_cfg(source, cloud_cover_max=15), "REGION")
        spot = f"IC({gee.SPOT_COLLECTION})"
        assert _calls(recording_ee, spot, "filter")[0][1] == (
            ("lte", "cloud_coverage_percentage", 15),
        )
        assert _calls(recording_ee, spot, "select")[0][1] == (bands,)
        assert spec.bands == bands and spec.resolution_m == res
        assert spec.dtype == "float32"

    @pytest.mark.parametrize(
        ("year", "missions"),
        [
            (1990, ["LT05"]),
            (2005, ["LT05", "LE07"]),
            (2023, ["LE07", "LC08", "LC09"]),
            (2025, ["LC08", "LC09"]),
        ],
    )
    def test_landsat_missions_by_year(self, recording_ee, year, missions):
        spec = gee._build_landsat(_cfg("landsat", year=year), "REGION")
        assert spec.collections == [gee.LANDSAT_COLLECTIONS[m][0] for m in missions]
        dated = {name for name, _ in _calls(recording_ee, "IC(", "filterDate")}
        assert dated == {
            f"IC({gee.LANDSAT_COLLECTIONS[m][0]}).filterBounds.filter" for m in missions
        }
        # every mission filters on CLOUD_COVER (the history covers all missions)
        filters = _calls(recording_ee, "IC(LANDSAT", "filter")
        assert len(filters) == 4
        assert all(args == (("lte", "CLOUD_COVER", 20),) for _, args in filters)

    def test_landsat_map_uses_mission_band_names(self, recording_ee):
        gee._build_landsat(_cfg("landsat", year=2005), "REGION")
        maps = [
            (name, a[0])
            for name, attr, a, _ in recording_ee
            if attr == "map" and name.endswith("filterDate")
        ]
        by_mission = {name.split("/")[1]: fn for name, fn in maps}
        l5 = {b: np.full((1, 1), 10000.0 * (i + 1)) for i, b in enumerate(gee.LANDSAT_L57_SR_BANDS)}
        l5["QA_PIXEL"] = np.zeros((1, 1))
        out = by_mission["LT05"](FakeImage(l5))
        # L5 SR_B3 (red) -> SR_B4
        assert out.value("SR_B4")[0, 0] == pytest.approx((3 * 10000 * 2.75e-5 - 0.2) * 10000)
        l8 = {b: np.full((1, 1), 10000.0 * (i + 1)) for i, b in enumerate(gee.LANDSAT_BANDS)}
        l8["QA_PIXEL"] = np.zeros((1, 1))
        with pytest.raises(KeyError):  # L5/7 band names are not read from L8/9 images
            by_mission["LT05"](FakeImage({"SR_B7": l8["SR_B7"], "QA_PIXEL": l8["QA_PIXEL"]}))
        assert set(by_mission) == {"LT05", "LE07"}

    def test_hls_cloud_property_on_both_collections(self, recording_ee):
        spec = gee._build_hls(_cfg("hls", cloud_cover_max=30), "REGION")
        for cid in (gee.HLSL30_COLLECTION, gee.HLSS30_COLLECTION):
            assert _calls(recording_ee, f"IC({cid})", "filter")[0][1] == (
                ("lte", "CLOUD_COVERAGE", 30),
            )
        assert spec.bands == gee.HLS_BANDS and spec.resolution_m == 30.0

    def test_sentinel2_cloud_score_plus_link(self, recording_ee):
        cfg = _cfg("sentinel2", year=2023, s2_cloud_mask="cloud_score_plus")
        spec = gee._build_sentinel2(cfg, "REGION")
        s2 = f"IC({gee.S2_COLLECTION})"
        assert _calls(recording_ee, s2, "filter")[0][1] == (("lte", "CLOUDY_PIXEL_PERCENTAGE", 20),)
        ((_, link_args),) = _calls(recording_ee, s2, "linkCollection")
        assert link_args[0].name == f"IC({gee.CLOUD_SCORE_PLUS_COLLECTION})"
        assert link_args[1] == ["cs_cdf"]
        assert gee.CLOUD_SCORE_PLUS_COLLECTION in spec.collections


class TestNaipMosaicOrder:
    """naip_order_key + ascending sort must put exact-year, newest images last (on top)."""

    @pytest.fixture
    def numeric_ee(self, monkeypatch):
        import datetime as dt

        class Num:
            def __init__(self, v):
                self.v = float(v.v if isinstance(v, Num) else v)

            def eq(self, other):
                return Num(1.0 if self.v == float(other) else 0.0)

            def multiply(self, other):
                return Num(self.v * float(other))

            def add(self, other):
                return Num(self.v + (other.v if isinstance(other, Num) else float(other)))

        class Date:
            def __init__(self, t):
                self.t = t.v if isinstance(t, Num) else float(t)

            def get(self, unit):
                assert unit == "year"
                return Num(dt.datetime.fromtimestamp(self.t / 1000, tz=dt.UTC).year)

        module = types.ModuleType("ee")
        module.Number, module.Date, module.Image = Num, Date, lambda x: x
        monkeypatch.setitem(sys.modules, "ee", module)
        return module

    class Img:
        def __init__(self, props):
            self.props = props

        def get(self, key):
            return self.props[key]

        def set(self, key, value):
            return type(self)({**self.props, key: value.v})

    def test_exact_year_newest_on_top(self, numeric_ee):
        import datetime as dt

        def ms(*ymd):
            return dt.datetime(*ymd, tzinfo=dt.UTC).timestamp() * 1000

        images = {
            "2021-06": self.Img({"system:time_start": ms(2021, 6, 1)}),
            "2022-05": self.Img({"system:time_start": ms(2022, 5, 1)}),
            "2022-08": self.Img({"system:time_start": ms(2022, 8, 1)}),
            "2023-07": self.Img({"system:time_start": ms(2023, 7, 1)}),  # newest overall
        }
        keyed = {k: gee.naip_order_key(v, 2022) for k, v in images.items()}
        order = sorted(keyed, key=lambda k: keyed[k].props[gee._NAIP_ORDER_PROPERTY])
        # ee mosaic(): the last image is on top
        assert order == ["2021-06", "2023-07", "2022-05", "2022-08"]


# ---------------------------------------------------------------------------
# geedim download (accessor calls, retries, resume)
# ---------------------------------------------------------------------------


class FakeGD:
    """Stand-in for the ``ee.Image.gd`` accessor of geedim 2."""

    def __init__(self, image, prepared=None):
        self.image = image
        self.prepared = prepared

    def prepareForExport(self, **kwargs):  # noqa: N802 - geedim API name
        self.image.prepare_calls.append(kwargs)
        return FakePrepared(self.image, kwargs)

    def toGeoTIFF(self, file, overwrite=False, max_requests=32, **kwargs):  # noqa: N802
        img = self.image
        img.geotiff_calls.append({"file": str(file), "overwrite": overwrite, "mr": max_requests})
        if img.failures:
            if img.warn_before_failure:
                warnings.warn(img.warn_before_failure, stacklevel=1)
            raise RuntimeError(img.failures.pop(0))
        if img.warn:
            warnings.warn(img.warn, stacklevel=1)
        kw = self.prepared
        h, w = kw["shape"]
        data = np.full((len(kw["bands"]), h, w), 5.0, dtype=np.float32)
        data[:, 0, 0] = -np.inf  # geedim's float nodata
        _write(file, data, Affine(*kw["crs_transform"]), crs=kw["crs"], nodata=float("-inf"))


class FakePrepared:
    def __init__(self, image, kwargs):
        self.gd = FakeGD(image, kwargs)


class FakeExportImage:
    def __init__(self, failures=(), warn=None, warn_before_failure=None):
        self.prepare_calls = []
        self.geotiff_calls = []
        self.failures = list(failures)
        self.warn = warn
        self.warn_before_failure = warn_before_failure
        self.gd = FakeGD(self)


class TestDownload:
    GRID = gee.ExportGrid("EPSG:32611", Affine(10, 0, 500000, 0, -10, 4000000), 6, 4)

    @pytest.fixture(autouse=True)
    def _geedim_accessor(self, monkeypatch):
        """``_download_tile`` imports geedim only to register ``ee.Image.gd``.

        The fake images carry their own ``.gd``, so a stand-in module lets these
        tests run without the ``gee`` extra (the CI core job).
        """
        try:
            import geedim  # noqa: F401
        except ImportError:
            monkeypatch.setitem(sys.modules, "geedim", types.ModuleType("geedim"))

    def test_accessor_arguments(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gee.time, "sleep", lambda s: None)
        img = FakeExportImage()
        out = tmp_path / "t.tif"
        mr = gee._download_tile(
            img, self.GRID, out, dtype="float32", band_names=["a", "b"], max_requests=6, label="x"
        )
        assert mr == 6 and out.exists() and not (tmp_path / "t.tif.part").exists()
        (kw,) = img.prepare_calls
        assert kw == {
            "crs": "EPSG:32611",
            "crs_transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0],
            "shape": (4, 6),
            "dtype": "float32",
            "bands": ["a", "b"],
        }
        (call,) = img.geotiff_calls
        assert call["file"].endswith("t.tif.part") and call["overwrite"] and call["mr"] == 6

    def test_rate_limit_retry_halves_requests(self, tmp_path, monkeypatch):
        sleeps = []
        monkeypatch.setattr(gee.time, "sleep", sleeps.append)
        img = FakeExportImage(failures=["HttpError 429 Too Many Requests"])
        mr = gee._download_tile(
            img,
            self.GRID,
            tmp_path / "t.tif",
            dtype="float32",
            band_names=["a"],
            max_requests=8,
            label="x",
        )
        assert mr == 4
        assert [c["mr"] for c in img.geotiff_calls] == [8, 4]
        assert sleeps == [10]

    def test_persistent_failure_raises_and_cleans_up(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gee.time, "sleep", lambda s: None)
        img = FakeExportImage(failures=["boom"] * 3)
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            gee._download_tile(
                img,
                self.GRID,
                tmp_path / "t.tif",
                dtype="float32",
                band_names=["a"],
                max_requests=2,
                label="x",
            )
        assert not list(tmp_path.iterdir())

    def test_restricted_mode_lowers_concurrency(self, tmp_path, caplog):
        img = FakeExportImage(
            warn="Your project has exceeded its noncommercial compute quota and is now in "
            "restricted mode."
        )
        mr = gee._download_tile(
            img,
            self.GRID,
            tmp_path / "t.tif",
            dtype="float32",
            band_names=["a"],
            max_requests=8,
            label="x",
        )
        assert mr == 4
        assert any("restricted mode" in r.message for r in caplog.records)

    def test_restricted_mode_before_failed_attempt_lowers_concurrency(
        self, tmp_path, monkeypatch, caplog
    ):
        # The quota warning arrives just before a (non rate-limit) failure: it must be
        # logged and concurrency halved for the retry.
        monkeypatch.setattr(gee.time, "sleep", lambda s: None)
        img = FakeExportImage(
            failures=["connection reset"],
            warn_before_failure="Your project has exceeded its noncommercial compute quota "
            "and is now in restricted mode.",
        )
        with caplog.at_level("WARNING"):
            mr = gee._download_tile(
                img,
                self.GRID,
                tmp_path / "t.tif",
                dtype="float32",
                band_names=["a"],
                max_requests=8,
                label="x",
            )
        assert [c["mr"] for c in img.geotiff_calls] == [8, 4]
        assert mr == 4
        assert any("restricted mode" in r.message for r in caplog.records)

    def test_export_tiles_assemble_and_resume(self, tmp_path):
        grid = gee.ExportGrid("EPSG:32611", Affine(10, 0, 500000, 0, -10, 4000000), 5, 3)
        out = tmp_path / "comp.tif"
        # a complete tile from an interrupted earlier run is reused
        tile_dir = tmp_path / "comp_tiles"
        tile_dir.mkdir()
        (r, c, h, w) = gee.split_grid(grid, 2)[0]
        sub = grid.window(r, c, h, w)
        _write(tile_dir / "tile_0000.tif", np.full((1, h, w), 7.0, np.float32), sub.transform)
        img = FakeExportImage()
        gee.export_ee_image(
            img, out, grid=grid, dtype="float32", band_names=["a"], tile_size=2, tags={"K": 1}
        )
        n_tiles = len(gee.split_grid(grid, 2))
        assert len(img.prepare_calls) == n_tiles - 1  # tile 0 was not downloaded again
        with rasterio.open(out) as src:
            data = src.read(1)
            assert src.transform == grid.transform and np.isnan(src.nodata)
            assert src.tags()["K"] == "1"
        assert (data[:2, :2] == 7.0).all()  # the reused tile
        # the -inf corner of every downloaded tile became NaN
        assert np.isnan(data[0, 2]) and np.isnan(data[2, 0])
        assert not tile_dir.exists()


class TestExtent:
    def test_grid_footprint_contains_geometry(self):
        geom = box(149.0, -30.5, 149.05, -30.45)
        grid = gee.compute_export_grid(geom, "EPSG:32755", 10.0)
        footprint = gee.grid_footprint_4326(grid)
        assert footprint.contains(geom)
        # the footprint is the projected bounding box: only slightly larger than the box
        assert footprint.area < geom.area * 1.05

    def test_build_selects_over_grid_and_does_not_clip(
        self, tmp_path, sample_aoi_geojson, stub_builder, monkeypatch
    ):
        seen = []
        monkeypatch.setattr(gee, "ee_geometry", lambda geom: seen.append(geom) or "REGION")

        def no_clip(self, region):
            raise AssertionError("the composite must not be clipped to the study area")

        monkeypatch.setattr(FakeImage, "clip", no_clip)
        gee.GEECompositeBuilder().build(_s2_config(tmp_path, sample_aoi_geojson))
        (region,) = seen
        (call,) = stub_builder["export"]
        assert region.equals_exact(gee.grid_footprint_4326(call["grid"]), 1e-9)

    def test_date_windows_share_one_grid(self, tmp_path, sample_aoi_geojson, stub_builder):
        a = _s2_config(tmp_path, sample_aoi_geojson, date_range=("2023-03-01", "2023-05-31"))
        b = a.merged(date_range=("2023-09-01", "2023-11-30"))
        gee.GEECompositeBuilder().build(a)
        gee.GEECompositeBuilder().build(b)
        ga, gb = (c["grid"] for c in stub_builder["export"])
        assert ga == gb


def test_gee_asset_study_area_initialises_ee_with_config(monkeypatch, tmp_path):
    from shapely.geometry import mapping

    events = []
    monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: events.append("ensure"))

    class FakeFC:
        def __init__(self, asset_id):
            events.append(("read", asset_id))

        def getInfo(self):  # noqa: N802 - mirrors ee.FeatureCollection.getInfo
            return {"features": [{"geometry": mapping(box(1, 1, 2, 2)), "properties": {}}]}

    monkeypatch.setitem(sys.modules, "ee", types.SimpleNamespace(FeatureCollection=FakeFC))
    cfg = AgriboundConfig(
        source="sentinel2",
        study_area="projects/p/assets/aoi",
        gee_project="p",
        lulc_filter=False,
        cache_dir=str(tmp_path),
    )
    geom = gee.study_area_geometry_4326(cfg)
    assert events == ["ensure", ("read", "projects/p/assets/aoi")]
    assert geom.equals(box(1, 1, 2, 2))
    # The asset was saved to the working directory; later reads do not contact Earth Engine.
    events.clear()
    assert gee.study_area_geometry_4326(cfg).equals(box(1, 1, 2, 2))
    assert events == []
    geom = gee.study_area_geometry_4326(cfg.merged(study_area="bbox:1,1,2,2"))
    assert events == [] and geom.equals(box(1, 1, 2, 2))
