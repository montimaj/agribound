"""Unit tests for the Earth Engine composite builder (no network, no GEE).

Per-image masking/scaling functions are evaluated numerically on a small
numpy-backed stand-in for the ``ee.Image`` API, so the tests check pixel
values, not just call sequences.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.affinity import translate
from shapely.geometry import box, shape

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

    def subtract(self, other):
        """Single-band image minus a single-band image (the first band's name; masks ANDed)."""
        (b,) = other.bands.values()
        (bm,) = other.masks.values()
        return FakeImage(
            {k: v - b for k, v in self.bands.items()},
            {k: m & bm for k, m in self.masks.items()},
            self.props,
        )

    def unitScale(self, low, high):  # noqa: N802 - ee API name
        return self._map(lambda a: (a - low) / (high - low))

    def clamp(self, low, high):
        return self._map(lambda a: np.clip(a, low, high))

    def round(self):
        return self._map(lambda a: np.floor(a + 0.5))  # halves up (inputs here are >= 0)

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

    def test_four_band_uint8_composite_does_not_tag_nir_as_alpha(self, tmp_path):
        """NAIP composites (R, G, B, N uint8) were written with band 4 tagged alpha.

        GDAL tags the 4th band of a 4-band uint8 GeoTIFF as alpha by default; GDAL's
        warper and rio-tiler then treat NIR as transparency.
        """
        from rasterio.enums import ColorInterp

        grid = gee.ExportGrid("EPSG:32611", Affine(10, 0, 0, 0, -10, 40), 4, 4)
        data = np.arange(64, dtype=np.uint8).reshape(4, 4, 4) + 1
        data[3, :, :2] = 0  # NIR 0 next to valid R, G, B
        tile = _write(tmp_path / "t.tif", data, grid.transform, nodata=0)
        out = gee.assemble_tiles(
            [(tile, (0, 0, 4, 4))],
            tmp_path / "naip.tif",
            grid,
            dtype="uint8",
            band_names=["R", "G", "B", "N"],
        )
        with rasterio.open(out) as src:
            assert ColorInterp.alpha not in src.colorinterp
            assert src.descriptions == ("R", "G", "B", "N")
            assert (src.read() == data).all()

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
        "le07_images": 0,  # Landsat 7 count of a landsat-pan no-data message (or an exception)
        "le07_windows": [],
    }

    monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
    monkeypatch.setattr(gee, "ee_geometry", lambda geom: "REGION")

    def fake_le07_images(config, region, window):
        calls["le07_windows"].append((region, window))
        if isinstance(calls["le07_images"], Exception):
            raise calls["le07_images"]
        return calls["le07_images"]

    monkeypatch.setattr(gee, "_landsat7_pan_images", fake_le07_images)

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

    def fake_summary(
        col, context="", count_property=None, *, image_list=False, cloud_property=None
    ):
        calls["count_property"] = count_property
        calls["counted"] = col
        calls["summary_context"] = context
        calls["image_list"] = image_list
        calls["cloud_property"] = cloud_property
        return calls["summary"]

    monkeypatch.setattr(gee, "collection_summary", fake_summary)
    monkeypatch.setattr(gee, "available_years", lambda col, context="": calls["years"])

    def fake_export(image, out_path, *, grid, dtype, band_names, tags=None, **kwargs):
        calls["export"].append(
            {
                "path": str(out_path),
                "grid": grid,
                "tags": tags,
                "dtype": dtype,
                "band_names": list(band_names),
                "image": image,
            }
        )
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


def _landsat_pan_build(tmp_path, aoi, monkeypatch, summary, calls, year=2023, **kwargs):
    """Build a landsat-pan composite with the stubbed Earth Engine layer; return the tags.

    The collection spec is made by the real ``_build_landsat_pan`` (recording ee);
    *summary* is what the image-count request returns.
    """
    cfg = _cfg("landsat-pan", year=year, **kwargs).merged(
        study_area=aoi, output_path=str(tmp_path / "out.gpkg")
    )
    spec = gee._build_landsat_pan(cfg, "REGION")
    spec.prepared = FakeCollection([FakeImage({"B8": [[0.25]]})])
    monkeypatch.setitem(gee._COLLECTION_BUILDERS, "landsat-pan", lambda cfg, region: spec)
    calls["summary"] = summary
    builder = gee.GEECompositeBuilder()
    path = builder.build(cfg)
    with rasterio.open(path) as src:
        assert src.count == 1
        assert src.res == (15, 15)
        tags = src.tags()
    assert calls["count_property"] == "SPACECRAFT_ID"
    assert tags["AGRIBOUND_COLLECTIONS"] == ",".join(spec.collections)
    assert builder.last_metadata["AGRIBOUND_SENSORS"] == tags["AGRIBOUND_SENSORS"]
    return tags


def _landsat_pan_no_data(tmp_path, aoi, monkeypatch, calls, **kwargs):
    """Build a landsat-pan composite whose image count is 0; return the NoDataError message.

    The collection spec is made by the real ``_build_landsat_pan`` (recording ee);
    the stubbed history lists 2019 and 2021.
    """
    from agribound.composites import NoDataError

    kwargs.setdefault("year", 2023)
    cfg = _cfg("landsat-pan", **kwargs).merged(
        study_area=aoi, output_path=str(tmp_path / "out.gpkg")
    )
    spec = gee._build_landsat_pan(cfg, "REGION")
    monkeypatch.setitem(gee._COLLECTION_BUILDERS, "landsat-pan", lambda cfg, region: spec)
    calls["summary"] = {"n_images": 0, "years": [], "counts": {}}
    with pytest.raises(NoDataError) as info:
        gee.GEECompositeBuilder().build(cfg)
    assert calls["export"] == []
    return str(info.value)


def _pan_warnings(caplog):
    return [r for r in caplog.records if r.levelname == "WARNING" and "mixes" in r.getMessage()]


class TestBuilder:
    def test_landsat_pan_export_and_provenance(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch, caplog
    ):
        summary = {"n_images": 32, "years": [2023], "counts": {"LANDSAT_9": 16, "LANDSAT_8": 16}}
        with caplog.at_level("INFO", logger="agribound.composites.gee"):
            tags = _landsat_pan_build(
                tmp_path, sample_aoi_geojson, monkeypatch, summary, stub_builder
            )
        assert tags["AGRIBOUND_SOURCE"] == "landsat-pan"
        assert tags["AGRIBOUND_VALUE_SCALE"] == "unit"
        # 'auto' uses the OLI missions only when they overlap the window (no Landsat 7).
        assert tags["AGRIBOUND_COLLECTIONS"] == "LANDSAT/LC08/C02/T1_TOA,LANDSAT/LC09/C02/T1_TOA"
        assert tags["AGRIBOUND_LANDSAT_PAN_MISSIONS"] == "auto"
        assert tags["AGRIBOUND_MISSIONS_SELECTED"] == "LC08,LC09"
        assert tags["AGRIBOUND_SENSORS"] == "LC08,LC09"
        assert tags["AGRIBOUND_SENSOR_IMAGES"] == "LC08:16,LC09:16"
        assert tags["AGRIBOUND_SPECTRAL_RESPONSE"] == "L8/9 PAN 0.50-0.68 um"
        assert "AGRIBOUND_SLC_OFF" not in tags
        assert not _pan_warnings(caplog)
        assert any("images per sensor: LC08 16, LC09 16" in r.getMessage() for r in caplog.records)

    def test_landsat_pan_explicit_mix_warns_and_is_tagged(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch, caplog
    ):
        summary = {"n_images": 50, "years": [2015], "counts": {"LANDSAT_7": 20, "LANDSAT_8": 30}}
        with caplog.at_level("WARNING", logger="agribound.composites.gee"):
            tags = _landsat_pan_build(
                tmp_path,
                sample_aoi_geojson,
                monkeypatch,
                summary,
                stub_builder,
                year=2015,
                landsat_pan_missions="LC08,LE07",
            )
        assert tags["AGRIBOUND_LANDSAT_PAN_MISSIONS"] == "LE07,LC08"
        assert tags["AGRIBOUND_SENSORS"] == "LE07,LC08"
        assert tags["AGRIBOUND_SENSOR_IMAGES"] == "LE07:20,LC08:30"
        assert tags["AGRIBOUND_SPECTRAL_RESPONSE"] == "L7 PAN 0.52-0.90 um; L8/9 PAN 0.50-0.68 um"
        assert "no special gap filling" in tags["AGRIBOUND_SLC_OFF"]
        (warning,) = _pan_warnings(caplog)
        assert "landsat-pan 2015" in warning.getMessage()
        assert "LE07 20, LC08 30" in warning.getMessage()

    def test_landsat_pan_tags_name_only_sensors_with_images(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch, caplog
    ):
        # Both missions were selected but every Landsat 7 scene failed the filters.
        summary = {"n_images": 9, "years": [2015], "counts": {"LANDSAT_8": 9}}
        with caplog.at_level("WARNING", logger="agribound.composites.gee"):
            tags = _landsat_pan_build(
                tmp_path,
                sample_aoi_geojson,
                monkeypatch,
                summary,
                stub_builder,
                year=2015,
                landsat_pan_missions=["LE07", "LC08"],
            )
        assert tags["AGRIBOUND_MISSIONS_SELECTED"] == "LE07,LC08"
        assert tags["AGRIBOUND_SENSORS"] == "LC08"
        assert tags["AGRIBOUND_SENSOR_IMAGES"] == "LC08:9"
        assert tags["AGRIBOUND_SPECTRAL_RESPONSE"] == "L8/9 PAN 0.50-0.68 um"
        assert not _pan_warnings(caplog)

    def test_landsat_pan_before_landsat_8_uses_landsat_7(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch
    ):
        summary = {"n_images": 11, "years": [2012], "counts": {"LANDSAT_7": 11}}
        tags = _landsat_pan_build(
            tmp_path, sample_aoi_geojson, monkeypatch, summary, stub_builder, year=2012
        )
        assert tags["AGRIBOUND_COLLECTIONS"] == "LANDSAT/LE07/C02/T1_TOA"
        assert tags["AGRIBOUND_SENSORS"] == "LE07"
        assert tags["AGRIBOUND_SPECTRAL_RESPONSE"] == "L7 PAN 0.52-0.90 um"
        assert "AGRIBOUND_SLC_OFF" in tags

    @pytest.mark.parametrize(
        ("missions", "searched"), [("auto", "LC08,LC09"), ("LE07", "LE07"), ("LC08", "LC08")]
    )
    def test_landsat_pan_without_images_names_the_missions_searched(
        self,
        tmp_path,
        sample_aoi_geojson,
        stub_builder,
        recording_ee,
        monkeypatch,
        missions,
        searched,
    ):
        message = _landsat_pan_no_data(
            tmp_path, sample_aoi_geojson, monkeypatch, stub_builder, landsat_pan_missions=missions
        )
        assert "No landsat-pan images (scene cloud cover <= 20%)" in message
        # The history covers the missions searched, so the years are labelled with them.
        assert f"Years with images of {searched} over the study-area extent: 2019, 2021." in message
        assert f"Missions searched: {searched} (landsat_pan_missions={missions!r})." in message
        assert message.endswith(f"(landsat_pan_missions={missions!r}).")  # no Landsat 7 images
        assert stub_builder["export"] == []
        # Only 'auto' without Landsat 7 among the missions searched counts Landsat 7 images.
        assert len(stub_builder["le07_windows"]) == (missions == "auto")

    @pytest.mark.parametrize(
        ("kwargs", "le07_images", "searched", "window", "hint"),
        [
            # Landsat 8's record starts on 2013-03-18: 'auto' searches Landsat 8 only.
            (
                {"year": 2013, "date_range": ("2013-01-01", "2013-03-20")},
                3,
                "LC08",
                ("2013-01-01", "2013-03-21"),
                True,
            ),
            (
                {"year": 2013, "date_range": ("2013-01-01", "2013-03-20")},
                0,
                "LC08",
                ("2013-01-01", "2013-03-21"),
                False,
            ),
            ({"year": 2016}, 1, "LC08", ("2016-01-01", "2017-01-01"), True),
            ({"year": 2023}, 2, "LC08,LC09", ("2023-01-01", "2024-01-01"), True),
            # Landsat 7's record ended on 2024-01-19 / Landsat 7 was searched / explicit list.
            ({"year": 2025}, 5, "LC08,LC09", None, False),
            ({"year": 2005}, 5, "LE07", None, False),
            ({"year": 2015, "landsat_pan_missions": "LC08"}, 5, "LC08", None, False),
        ],
    )
    def test_landsat_pan_without_images_counts_landsat_7_for_auto(
        self,
        tmp_path,
        sample_aoi_geojson,
        stub_builder,
        recording_ee,
        monkeypatch,
        kwargs,
        le07_images,
        searched,
        window,
        hint,
    ):
        stub_builder["le07_images"] = le07_images
        message = _landsat_pan_no_data(
            tmp_path, sample_aoi_geojson, monkeypatch, stub_builder, **kwargs
        )
        # One request on the no-data path, with the composite's region and window.
        assert stub_builder["le07_windows"] == ([] if window is None else [("REGION", window)])
        setting = kwargs.get("landsat_pan_missions", "auto")
        searched_text = f"Missions searched: {searched} (landsat_pan_missions={setting!r})."
        if not hint:
            assert message.endswith(searched_text)
            assert "Landsat 7 (LE07) has" not in message
            return
        assert (
            f"{searched_text} Landsat 7 (LE07) has {le07_images} image(s) with scene cloud cover "
            "<= 20% over the study-area extent in this window, which 'auto' does not use because "
            "the window overlaps the Landsat 8/9 record and Landsat 7 PAN (0.52-0.90 um) "
            "includes the near infrared; landsat_pan_missions='LE07' uses them."
        ) in message
        assert message.endswith("landsat_pan_missions='LE07' uses them.")

    def test_landsat_pan_no_data_survives_a_failed_landsat_7_count(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch
    ):
        stub_builder["le07_images"] = RuntimeError("Earth Engine: too many concurrent requests")
        message = _landsat_pan_no_data(
            tmp_path, sample_aoi_geojson, monkeypatch, stub_builder, year=2016
        )
        assert len(stub_builder["le07_windows"]) == 1
        assert message.endswith("(landsat_pan_missions='auto').")  # no hint, still no-data

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


class _OrFilter(tuple):
    """``ee.Filter.Or(...)`` of the recording stub; ``Not()`` wraps it."""

    def Not(self):  # noqa: N802 - ee API name
        return ("not", tuple(self))


def _fake_filter_namespace():
    return types.SimpleNamespace(
        lte=lambda prop, value: ("lte", prop, value),
        eq=lambda prop, value: ("eq", prop, value),
        listContains=lambda prop, value: ("listContains", prop, value),
        bounds=lambda geometry: ("bounds", geometry),
        Or=lambda *filters: _OrFilter(("or", *filters)),
    )


@pytest.fixture
def recording_ee(monkeypatch):
    log = []
    module = types.ModuleType("ee")
    module.ImageCollection = lambda cid: Chain(f"IC({cid})", log)
    module.Image = lambda x: x
    module.Filter = _fake_filter_namespace()
    module.Geometry = types.SimpleNamespace(Point=lambda coords: ("point", tuple(coords)))
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


class FakeRegion:
    """An ``ee.Geometry`` made from coordinates: ``toGeoJSON()`` returns them, no request."""

    def __init__(self, geometry):
        self.geometry = geometry

    def toGeoJSON(self):  # noqa: N802 - ee API name
        from shapely.geometry import mapping

        return {**mapping(self.geometry), "geodesic": False}

    def getInfo(self):  # noqa: N802 - ee API name
        raise AssertionError("the region's coordinates must be read without a request")


#: Export grid outlines (EPSG:4326, rounded to boxes) of the areas of the HLS footprint
#: investigation (outputs/hls_antimeridian): example 23's Twifo Praso square (Ghana), example
#: 14's Lea County box, example 03's Murray-Darling box, and two 5 km squares inside MGRS tile
#: T01KAA (Fiji), west and east of 180 degrees.
TWIFO_BOX = box(-1.5407, 5.5156, -1.4951, 5.5610)
LEA_BOX = box(-103.254, 32.7468, -103.0454, 32.9532)
MURRAY_BOX = box(149.6958, -30.3033, 149.8546, -30.1467)
FIJI_WEST_BOX = box(179.6761, -17.6230, 179.7239, -17.5769)
FIJI_EAST_BOX = box(-179.8740, -17.6231, -179.8259, -17.5769)


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


# ---------------------------------------------------------------------------
# landsat-pan missions: one PAN bandpass per composite unless asked otherwise
# ---------------------------------------------------------------------------

_YEAR_2023 = ("2023-01-01", "2024-01-01")
_YEAR_2015 = ("2015-01-01", "2016-01-01")


class TestLandsatPanMissionsForWindow:
    @pytest.mark.parametrize(
        ("window", "missions"),
        [
            (("1999-01-01", "2000-01-01"), ("LE07",)),
            (("2012-01-01", "2013-01-01"), ("LE07",)),
            (("2013-01-01", "2014-01-01"), ("LC08",)),
            (("2013-01-01", "2013-03-18"), ("LE07",)),  # ends (exclusive) on LC08's first day
            (("2013-01-01", "2013-03-19"), ("LC08",)),  # includes it
            (("2021-06-01", "2021-10-31"), ("LC08",)),  # LC09 starts 2021-10-31
            (("2021-06-01", "2021-12-01"), ("LC08", "LC09")),
            (_YEAR_2023, ("LC08", "LC09")),
            (("2024-01-01", "2025-01-01"), ("LC08", "LC09")),
            (("1990-01-01", "1991-01-01"), ()),
            (("1999-01-01", "1999-05-28"), ()),  # LE07 starts 1999-05-28
        ],
    )
    def test_auto(self, window, missions):
        assert gee.landsat_pan_missions_for_window("auto", window) == missions

    @pytest.mark.parametrize(
        ("setting", "window", "missions"),
        [
            (("LE07",), _YEAR_2023, ("LE07",)),
            (("LE07", "LC08", "LC09"), _YEAR_2023, ("LE07", "LC08", "LC09")),
            (("LC09", "LE07"), _YEAR_2023, ("LE07", "LC09")),  # canonical order
            (("LE07", "LC08"), _YEAR_2015, ("LE07", "LC08")),
            (("LE07", "LC09"), _YEAR_2015, ("LE07",)),
            (("LC09",), _YEAR_2015, ()),
            (("LE07",), ("2025-01-01", "2026-01-01"), ()),
            (("LE07",), ("2024-01-19", "2024-02-01"), ("LE07",)),  # last day included
            (("LE07",), ("2024-01-20", "2024-02-01"), ()),
            (("LC08", "LC09"), ("2012-01-01", "2013-01-01"), ()),
        ],
    )
    def test_explicit_missions(self, setting, window, missions):
        assert gee.landsat_pan_missions_for_window(setting, window) == missions

    def test_auto_never_mixes_the_bandpasses(self):
        for year in range(1995, 2027):
            for window in ((f"{year}-01-01", f"{year + 1}-01-01"), (f"{year}-06-01", "2027-01-01")):
                missions = gee.landsat_pan_missions_for_window("auto", window)
                assert missions in ((), ("LE07",)) or "LE07" not in missions, window

    def test_constants(self):
        assert gee.LANDSAT_PAN_OLI_MISSIONS == ("LC08", "LC09")
        assert gee.LANDSAT_PAN_SPACECRAFT == {
            "LANDSAT_7": "LE07",
            "LANDSAT_8": "LC08",
            "LANDSAT_9": "LC09",
        }
        assert list(gee.LANDSAT_PAN_COLLECTIONS) == ["LE07", "LC08", "LC09"]
        assert all(
            cid.endswith("/C02/T1_TOA") for cid, _, _ in gee.LANDSAT_PAN_COLLECTIONS.values()
        )
        for name in (
            "landsat_pan_bandpass",
            "landsat_pan_missions_for_window",
            "prepare_landsat_pan_image",
        ):
            assert name in gee.__all__


@pytest.mark.parametrize(
    ("missions", "bandpass"),
    [
        (("LE07",), "L7 PAN 0.52-0.90 um"),
        (("LC08",), "L8/9 PAN 0.50-0.68 um"),
        (["LC09"], "L8/9 PAN 0.50-0.68 um"),
        (("LC08", "LC09"), "L8/9 PAN 0.50-0.68 um"),
        (("LE07", "LC09"), "L7 PAN 0.52-0.90 um; L8/9 PAN 0.50-0.68 um"),
        ((), ""),
    ],
)
def test_landsat_pan_bandpass(missions, bandpass):
    assert gee.landsat_pan_bandpass(missions) == bandpass


class TestSensorTags:
    LABELS = gee.LANDSAT_PAN_SPACECRAFT

    def _tags(self, counts, source="landsat-pan"):
        cfg = _cfg(source, year=2023)
        return gee._sensor_tags(cfg, f"{source} 2023", counts, self.LABELS)

    def test_labels_in_label_order(self, caplog):
        with caplog.at_level("INFO", logger="agribound.composites.gee"):
            tags = self._tags({"LANDSAT_9": 5, "LANDSAT_8": 7})
        assert tags == {
            "AGRIBOUND_SENSORS": "LC08,LC09",
            "AGRIBOUND_SENSOR_IMAGES": "LC08:7,LC09:5",
            "AGRIBOUND_SPECTRAL_RESPONSE": "L8/9 PAN 0.50-0.68 um",
        }
        assert any(
            r.levelname == "INFO" and "images per sensor: LC08 7, LC09 5" in r.getMessage()
            for r in caplog.records
        )
        assert not _pan_warnings(caplog)

    def test_zero_counts_are_dropped(self, caplog):
        with caplog.at_level("WARNING", logger="agribound.composites.gee"):
            tags = self._tags({"LANDSAT_7": 0, "LANDSAT_8": 4})
        assert tags["AGRIBOUND_SENSORS"] == "LC08"
        assert tags["AGRIBOUND_SENSOR_IMAGES"] == "LC08:4"
        assert tags["AGRIBOUND_SPECTRAL_RESPONSE"] == "L8/9 PAN 0.50-0.68 um"
        assert not _pan_warnings(caplog)

    def test_unlabelled_values_follow_sorted(self):
        tags = self._tags({"LANDSAT_X": 1, "LANDSAT_8": 2, "ALPHA": 3})
        assert tags["AGRIBOUND_SENSORS"] == "LC08,ALPHA,LANDSAT_X"
        assert tags["AGRIBOUND_SENSOR_IMAGES"] == "LC08:2,ALPHA:3,LANDSAT_X:1"

    def test_no_images(self, caplog):
        with caplog.at_level("INFO", logger="agribound.composites.gee"):
            tags = self._tags({})
        assert tags["AGRIBOUND_SENSORS"] == "" and tags["AGRIBOUND_SENSOR_IMAGES"] == ""
        assert any("images per sensor: none" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "counts",
        [
            {"LANDSAT_7": 20, "LANDSAT_8": 16, "LANDSAT_9": 16},
            {"LANDSAT_9": 3, "LANDSAT_7": 1},
        ],
    )
    def test_mixed_bandpasses_warn(self, caplog, counts):
        with caplog.at_level("WARNING", logger="agribound.composites.gee"):
            tags = self._tags(counts)
        assert tags["AGRIBOUND_SPECTRAL_RESPONSE"] == "L7 PAN 0.52-0.90 um; L8/9 PAN 0.50-0.68 um"
        (warning,) = _pan_warnings(caplog)
        message = warning.getMessage()
        assert message.startswith("landsat-pan 2023: the median mixes Landsat 7 ETM+ PAN")
        assert tags["AGRIBOUND_SENSORS"].split(",")[0] == "LE07"
        expected = ", ".join(
            f"{m} {counts[k]}" for k, m in self.LABELS.items() if counts.get(k, 0) > 0
        )
        assert f"(images: {expected})" in message
        assert "landsat_pan_missions='auto'" in message

    @pytest.mark.parametrize("counts", [{"LANDSAT_7": 9}, {"LANDSAT_8": 9, "LANDSAT_9": 2}])
    def test_one_bandpass_does_not_warn(self, caplog, counts):
        with caplog.at_level("WARNING", logger="agribound.composites.gee"):
            self._tags(counts)
        assert not _pan_warnings(caplog)

    def test_other_sources_get_no_spectral_response_or_warning(self, caplog):
        with caplog.at_level("WARNING", logger="agribound.composites.gee"):
            tags = self._tags({"LANDSAT_7": 2, "LANDSAT_8": 2}, source="landsat")
        assert tags == {
            "AGRIBOUND_SENSORS": "LE07,LC08",
            "AGRIBOUND_SENSOR_IMAGES": "LE07:2,LC08:2",
        }
        assert not _pan_warnings(caplog)


class _SummaryCollection:
    """Records the aggregations requested from a collection."""

    def __init__(self, log):
        self.log = log

    def size(self):
        self.log.append(("size",))
        return "SIZE"

    def aggregate_array(self, prop):
        self.log.append(("aggregate_array", prop))
        return "TIMES"

    def aggregate_histogram(self, prop):
        self.log.append(("aggregate_histogram", prop))
        return "HISTOGRAM"

    def sort(self, prop):
        self.log.append(("sort", prop))
        return Chain("SORTED", self.log)


@pytest.fixture
def summary_ee(monkeypatch):
    """Fake ``ee`` whose ``Dictionary(query).getInfo()`` returns ``state["info"]``."""
    state = {"queries": [], "log": [], "info": None}
    module = types.ModuleType("ee")
    module.List = lambda value: Chain("List", state["log"])

    class Dictionary:
        def __init__(self, query):
            state["queries"].append(dict(query))

        def getInfo(self):  # noqa: N802 - ee API name
            return state["info"]

    module.Dictionary = Dictionary
    monkeypatch.setitem(sys.modules, "ee", module)
    return state


class TestCollectionSummary:
    def test_without_count_property(self, summary_ee):
        summary_ee["info"] = {"n": 3, "years": [2022.0, 2023]}
        collection = _SummaryCollection(summary_ee["log"])
        assert gee.collection_summary(collection, "unit") == {"n_images": 3, "years": [2022, 2023]}
        (query,) = summary_ee["queries"]
        assert set(query) == {"n", "years"}
        assert ("aggregate_histogram", "SPACECRAFT_ID") not in summary_ee["log"]

    def test_counts_in_the_same_request(self, summary_ee):
        summary_ee["info"] = {
            "n": 32,
            "years": [2023],
            "counts": {"LANDSAT_8": 16.0, "LANDSAT_9": 16},
        }
        collection = _SummaryCollection(summary_ee["log"])
        summary = gee.collection_summary(collection, "unit", count_property="SPACECRAFT_ID")
        assert summary == {
            "n_images": 32,
            "years": [2023],
            "counts": {"LANDSAT_8": 16, "LANDSAT_9": 16},
        }
        assert all(isinstance(v, int) for v in summary["counts"].values())
        (query,) = summary_ee["queries"]  # one request
        assert query["counts"] == "HISTOGRAM"
        assert ("aggregate_histogram", "SPACECRAFT_ID") in summary_ee["log"]

    @pytest.mark.parametrize("counts", [None, {}])
    def test_missing_counts_give_an_empty_mapping(self, summary_ee, counts):
        summary_ee["info"] = {"n": 0, "years": [], "counts": counts}
        summary = gee.collection_summary(
            _SummaryCollection(summary_ee["log"]), "unit", count_property="SPACECRAFT_ID"
        )
        assert summary == {"n_images": 0, "years": [], "counts": {}}


class TestCollectionSpecs:
    @pytest.mark.parametrize(
        ("year", "missions"),
        [
            (1999, ["LE07"]),
            (2012, ["LE07"]),
            (2013, ["LC08"]),  # Landsat 7 still flies, but 'auto' never mixes the bandpasses
            (2015, ["LC08"]),
            (2023, ["LC08", "LC09"]),
            (2025, ["LC08", "LC09"]),
        ],
    )
    def test_landsat_pan_missions(self, recording_ee, year, missions):
        spec = gee._build_landsat_pan(_cfg("landsat-pan", year=year), "REGION")
        assert spec.collections == [gee.LANDSAT_PAN_COLLECTIONS[m][0] for m in missions]
        assert spec.bands == ["B8"]
        assert spec.resolution_m == 15 and spec.dtype == "float32"
        assert spec.notes["landsat_pan_missions"] == "auto"
        assert spec.notes["missions_selected"] == ",".join(missions)
        assert spec.notes["spectral_response"] == gee.landsat_pan_bandpass(missions)
        assert ("slc_off" in spec.notes) is (missions == ["LE07"])
        assert spec.count_property == "SPACECRAFT_ID"
        assert spec.count_labels == gee.LANDSAT_PAN_SPACECRAFT
        # Only the selected missions are queried, for the images and for the history.
        queried = {name[: name.index(")") + 1] for name, *_ in recording_ee}
        assert queried == {f"IC({c})" for c in spec.collections}
        filters = _calls(recording_ee, "IC(LANDSAT", "filter")
        assert len(filters) == len(missions)
        assert all(args == (("lte", "CLOUD_COVER", 20),) for _, args in filters)

    def test_landsat_pan_prepares_each_image_with_its_mission_mask(self, recording_ee):
        gee._build_landsat_pan(
            _cfg("landsat-pan", year=2015, landsat_pan_missions="LE07,LC08"), "R"
        )
        maps = {
            name.split("/")[1]: args[0]
            for name, attr, args, _ in recording_ee
            if attr == "map" and name.endswith("filterDate")
        }
        assert set(maps) == {"LE07", "LC08"}
        qa = np.array([[0, 4]])  # bit 2: cirrus, used on Landsat 8/9 only
        image = FakeImage({"B8": [[0.3, 0.4]], "QA_PIXEL": qa})
        assert maps["LE07"](image).masks["B8"].tolist() == [[True, True]]
        assert maps["LC08"](image).masks["B8"].tolist() == [[True, False]]

    def test_landsat_pan_date_range(self, recording_ee):
        cfg = _cfg("landsat-pan", date_range=("2012-12-01", "2013-04-01"))
        spec = gee._build_landsat_pan(cfg, "REGION")
        assert spec.collections == [gee.LANDSAT_PAN_COLLECTIONS["LC08"][0]]
        both = gee._build_landsat_pan(cfg.merged(landsat_pan_missions="LE07,LC08"), "REGION")
        assert both.collections == [gee.LANDSAT_PAN_COLLECTIONS[m][0] for m in ("LE07", "LC08")]
        dated = _calls(recording_ee, "IC(LANDSAT", "filterDate")
        assert len(dated) == 3
        assert all(args == ("2012-12-01", "2013-04-02") for _, args in dated)

    @pytest.mark.parametrize(
        ("setting", "year", "missions"),
        [
            ("LE07", 2023, ["LE07"]),
            ("LE07,LC08,LC09", 2023, ["LE07", "LC08", "LC09"]),
            (["LC09", "LE07"], 2015, ["LE07"]),  # Landsat 9 had not launched
            (["LC08"], 2024, ["LC08"]),
        ],
    )
    def test_landsat_pan_explicit_missions(self, recording_ee, setting, year, missions):
        cfg = _cfg("landsat-pan", year=year, landsat_pan_missions=setting)
        spec = gee._build_landsat_pan(cfg, "REGION")
        assert spec.collections == [gee.LANDSAT_PAN_COLLECTIONS[m][0] for m in missions]
        assert spec.notes["landsat_pan_missions"] == ",".join(cfg.landsat_pan_missions)
        assert spec.notes["missions_selected"] == ",".join(missions)
        assert ("slc_off" in spec.notes) is ("LE07" in missions)
        assert len(_calls(recording_ee, "IC(LANDSAT", "filter")) == len(missions)

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            (
                {"year": 2015, "landsat_pan_missions": ("LC09",)},
                r"None of landsat_pan_missions='LC09' acquired images between 2015-01-01 and "
                r"2016-01-01 \(Landsat 7 1999-05-28 to 2024-01-19",
            ),
            (
                {"year": 2025, "landsat_pan_missions": ("LE07",)},
                r"None of landsat_pan_missions='LE07' acquired images between 2025-01-01",
            ),
            (
                {"year": 1999, "date_range": ("1999-01-01", "1999-05-01")},
                r"No Landsat PAN mission overlaps the date window; available from 1999-05-28",
            ),
        ],
    )
    def test_landsat_pan_no_selected_mission_raises(self, recording_ee, changes, match):
        """A configuration changed after validation fails as a configuration error.

        ``AgriboundConfig`` rejects these windows; the builder's own check is the
        safety net. It raises a plain ValueError, not NoDataError, so an HPC tile
        is recorded as failed, not as a no-data tile.
        """
        from agribound.composites import NoDataError

        cfg = _cfg("landsat-pan", year=2023)
        for name, value in changes.items():
            setattr(cfg, name, value)
        with pytest.raises(ValueError, match=match) as info:
            gee._build_landsat_pan(cfg, "REGION")
        assert not isinstance(info.value, NoDataError)
        assert recording_ee == []  # raised before any Earth Engine request

    def test_landsat7_pan_image_count_uses_the_composite_filters(self, monkeypatch):
        log = []

        class Collection:
            def __init__(self, cid):
                log.append(("collection", cid))

            def filterBounds(self, region):  # noqa: N802 - Earth Engine API name
                log.append(("bounds", region))
                return self

            def filter(self, value):
                log.append(("filter", value))
                return self

            def filterDate(self, start, end):  # noqa: N802 - Earth Engine API name
                log.append(("date", start, end))
                return self

            def size(self):
                return types.SimpleNamespace(getInfo=lambda: 4.0)

        module = types.ModuleType("ee")
        module.ImageCollection = Collection
        module.Filter = types.SimpleNamespace(lte=lambda prop, value: ("lte", prop, value))
        monkeypatch.setitem(sys.modules, "ee", module)
        cfg = _cfg("landsat-pan", year=2016, cloud_cover_max=35)
        assert gee._landsat7_pan_images(cfg, "REGION", ("2016-01-01", "2017-01-01")) == 4
        assert log == [
            ("collection", "LANDSAT/LE07/C02/T1_TOA"),
            ("bounds", "REGION"),
            ("filter", ("lte", "CLOUD_COVER", 35)),
            ("date", "2016-01-01", "2017-01-01"),
        ]

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
        spec = gee._build_hls(_cfg("hls", cloud_cover_max=30), FakeRegion(TWIFO_BOX))
        for cid in (gee.HLSL30_COLLECTION, gee.HLSS30_COLLECTION):
            assert _calls(recording_ee, f"IC({cid})", "filter")[0][1] == (
                ("lte", "CLOUD_COVERAGE", 30),
            )
        assert spec.bands == gee.HLS_BANDS and spec.resolution_m == 30.0

    def test_hls_far_point_filter_on_raw_and_history_of_both_collections(self, recording_ee):
        region = FakeRegion(TWIFO_BOX)
        spec = gee._build_hls(_cfg("hls", cloud_cover_max=30), region)
        p1, p2 = gee.hls_far_points(TWIFO_BOX)
        far = ("not", ("or", ("bounds", ("point", p1)), ("bounds", ("point", p2))))
        for cid in (gee.HLSL30_COLLECTION, gee.HLSS30_COLLECTION):
            ic = f"IC({cid})"
            assert _calls(recording_ee, ic, "filterBounds") == [(ic, (region,))]
            # The cloud filter, then the far-point exclusion, on the base of raw and history.
            assert _calls(recording_ee, ic, "filter") == [
                (f"{ic}.filterBounds", (("lte", "CLOUD_COVERAGE", 30),)),
                (f"{ic}.filterBounds.filter", (far,)),
            ]
            assert _calls(recording_ee, ic, "filterDate") == [
                (f"{ic}.filterBounds.filter.filter", ("2022-01-01", "2023-01-01"))
            ]
        l30 = f"IC({gee.HLSL30_COLLECTION}).filterBounds.filter.filter"
        s30 = f"IC({gee.HLSS30_COLLECTION}).filterBounds.filter.filter"
        merges = {name: args[0].name for name, args in _calls(recording_ee, "IC(", "merge")}
        assert spec.raw.name == f"{l30}.filterDate.merge"  # counted: AGRIBOUND_N_IMAGES
        assert merges[f"{l30}.filterDate"] == f"{s30}.filterDate"
        assert spec.history.name == f"{l30}.merge"  # the years of the no-image error
        assert merges[l30] == s30
        # Away from 180 degrees: no request, no image loaded by ID.
        assert not any(attr == "getInfo" for _, attr, _, _ in recording_ee)
        assert spec.notes["hls_recipe_version"] == gee.HLS_RECIPE_VERSION
        assert "hlss30_images_by_id" not in spec.notes
        assert spec.collections == [gee.HLSL30_COLLECTION, gee.HLSS30_COLLECTION]
        assert all(isinstance(v, str | int | float) for v in spec.notes.values())  # tags

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


# ---------------------------------------------------------------------------
# Sentinel-1 (sentinel1)
# ---------------------------------------------------------------------------

#: One Sentinel-1 image in dB: VV below -30 dB (pixel 0) is an edge to mask; pixels
#: 2-5 cover the ends and the middle of the VV and VH ranges, and values beyond them.
S1_VV = np.array([[-35.0, -30.0, -25.0, -12.5, 0.0, 3.0]])
S1_VH = np.array([[-40.0, -36.0, -32.0, -20.0, -8.0, -1.0]])


def _s1_image():
    return FakeImage(
        {"VV": S1_VV, "VH": S1_VH, "angle": np.full(S1_VV.shape, 38.0)},
        props={"system:time_start": 7, "system:index": "S1A_X", "platform_number": "A"},
    )


class TestSentinel1Image:
    def test_constants(self):
        assert gee.S1_COLLECTION == "COPERNICUS/S1_GRD"
        assert gee.S1_BANDS == ["VV", "VH", "VV_VH"] == list(gee.S1_DB_RANGES)
        assert gee.S1_DB_RANGES == {
            "VV": (-25.0, 0.0),
            "VH": (-32.0, -8.0),
            "VV_VH": (0.0, 15.0),
        }
        assert gee.S1_EDGE_MASK_DB == -30.0
        assert (gee.S1_INSTRUMENT_MODE, gee.S1_RESOLUTION_METERS) == ("IW", 10)
        assert gee.S1_POLARISATIONS == ("VV", "VH")
        assert gee.S1_ORBIT_PASSES == ("ASCENDING", "DESCENDING")
        assert gee.S1_PLATFORMS == {"A": "S1A", "B": "S1B", "C": "S1C", "D": "S1D"}
        for name in (
            "prepare_sentinel1_image",
            "sentinel1_collection",
            "sentinel1_db_to_uint8",
            "sentinel1_orbit_pass",
            "sentinel1_pass_counts",
            "sentinel1_uint8_to_db",
        ):
            assert name in gee.__all__

    def test_edges_masked_ratio_added_and_bands_scaled(self, fake_ee):
        out = gee.prepare_sentinel1_image(_s1_image())
        assert list(out.bands) == ["VV", "VH", "VV_VH"]  # 'angle' is dropped
        # VV < -30 dB is masked in every band; -30 dB itself is kept.
        for band in out.bands:
            assert out.masks[band].tolist() == [[False, True, True, True, True, True]], band
        # 1 + round(254 * clamp((dB - lo) / (hi - lo), 0, 1)): clamped below and above.
        assert out.bands["VV"].tolist() == [[1, 1, 1, 128, 255, 255]]
        assert out.bands["VH"].tolist() == [[1, 1, 1, 128, 255, 255]]
        # VV_VH = VV - VH (5, 6, 7, 7.5, 8, 4 dB) of this image, on 0-15 dB.
        assert out.bands["VV_VH"].tolist() == [[86, 103, 120, 128, 136, 69]]
        for band, db in (("VV", S1_VV), ("VH", S1_VH), ("VV_VH", S1_VV - S1_VH)):
            np.testing.assert_array_equal(out.bands[band], gee.sentinel1_db_to_uint8(db, band))
        assert out.props["system:time_start"] == 7 and out.props["system:index"] == "S1A_X"

    def test_valid_pixels_are_1_to_255(self, fake_ee):
        rng = np.random.default_rng(0)
        vv = rng.uniform(-45, 10, (1, 500))
        vh = vv - rng.uniform(-5, 25, (1, 500))
        out = gee.prepare_sentinel1_image(FakeImage({"VV": vv, "VH": vh}))
        for band in out.bands:
            valid = out.bands[band][out.masks[band]]
            assert valid.min() >= 1 and valid.max() <= 255
            assert np.array_equal(valid, np.round(valid))
        assert out.masks["VV"].tolist() == (vv >= -30).tolist()

    def test_masked_input_stays_masked(self, fake_ee):
        image = FakeImage(
            {"VV": S1_VV, "VH": S1_VH},
            masks={"VH": np.array([[True, True, False, True, True, True]])},
        )
        out = gee.prepare_sentinel1_image(image)
        assert out.masks["VV"][0].tolist() == [False, True, True, True, True, True]
        assert out.masks["VH"][0].tolist() == [False, True, False, True, True, True]
        assert out.masks["VV_VH"][0].tolist() == [False, True, False, True, True, True]

    def test_median_of_scaled_images_is_the_scaled_median(self, fake_ee):
        """The scaling keeps the order of the values, so it commutes with an odd-sized median."""
        rng = np.random.default_rng(1)
        stack_vv = rng.uniform(-28, 2, (5, 1, 40))
        stack_vh = stack_vv - rng.uniform(0, 14, (5, 1, 40))
        images = [FakeImage({"VV": a, "VH": b}) for a, b in zip(stack_vv, stack_vh, strict=True)]
        composite = gee.apply_composite_method(
            FakeCollection(images).map(gee.prepare_sentinel1_image), "median", "sentinel1"
        )
        np.testing.assert_array_equal(
            composite.value("VV"), gee.sentinel1_db_to_uint8(np.median(stack_vv, axis=0), "VV")
        )
        ratio = np.median(stack_vv - stack_vh, axis=0)  # per-image ratio, then the median
        np.testing.assert_array_equal(
            composite.value("VV_VH"), gee.sentinel1_db_to_uint8(ratio, "VV_VH")
        )


class TestSentinel1Scaling:
    def test_round_trip_within_half_a_step(self):
        for band, (lo, hi) in gee.S1_DB_RANGES.items():
            db = np.linspace(lo, hi, 2001)
            values = gee.sentinel1_db_to_uint8(db, band)
            assert values.dtype == np.uint8
            assert values.min() == 1 and values.max() == 255
            back = gee.sentinel1_uint8_to_db(values, band)
            assert np.abs(back - db).max() <= (hi - lo) / 254 / 2 + 1e-9
            assert np.all(np.diff(values.astype(int)) >= 0)  # monotonic

    def test_clamping_and_nodata(self):
        assert gee.sentinel1_db_to_uint8([-40.0, -25.0, 0.0, 5.0], "VV").tolist() == [
            1,
            1,
            255,
            255,
        ]
        assert gee.sentinel1_db_to_uint8([np.nan, np.inf], "VH").tolist() == [0, 0]
        db = gee.sentinel1_uint8_to_db(np.array([0, 1, 128, 255], dtype=np.uint8), "VH")
        assert np.isnan(db[0])
        assert db[1:].tolist() == [-32.0, -20.0, -8.0]
        assert np.isnan(gee.sentinel1_uint8_to_db([np.nan], "VV")).all()

    def test_ranges_from_the_composite_tag(self):
        tag = json.dumps({"VV": [-20.0, 5.0], "VH": [-30.0, -6.0], "VV_VH": [0.0, 12.0]})
        assert gee.sentinel1_uint8_to_db([1, 255], "VV", tag).tolist() == [-20.0, 5.0]
        assert gee.sentinel1_uint8_to_db([1, 255], "VV_VH", json.loads(tag)).tolist() == [
            0.0,
            12.0,
        ]


@pytest.mark.parametrize(
    ("counts", "orbit_pass"),
    [
        ({"ASCENDING": 12, "DESCENDING": 30}, "DESCENDING"),
        ({"ASCENDING": 31, "DESCENDING": 30}, "ASCENDING"),
        ({"ASCENDING": 15, "DESCENDING": 15}, "DESCENDING"),  # tie
        ({"ASCENDING": 4}, "ASCENDING"),
        ({"DESCENDING": 4}, "DESCENDING"),
        ({}, "DESCENDING"),  # no acquisitions: the no-data error follows
    ],
)
def test_sentinel1_orbit_pass(counts, orbit_pass):
    assert gee.sentinel1_orbit_pass(counts) == orbit_pass


class _SliceCollection:
    """GRD slices as property dicts, with Earth Engine's ``distinct`` (first image of each key)."""

    def __init__(self, images, log):
        self.images, self.log = list(images), log

    def distinct(self, properties):
        self.log.append(("distinct", properties))
        seen, kept = set(), []
        for image in self.images:
            key = tuple(image[p] for p in properties)
            if key not in seen:
                seen.add(key)
                kept.append(image)
        return _SliceCollection(kept, self.log)

    def aggregate_histogram(self, prop):
        self.log.append(("aggregate_histogram", prop))
        counts = {}
        for image in self.images:
            counts[image[prop]] = counts.get(image[prop], 0) + 1.0  # Earth Engine gives floats
        return types.SimpleNamespace(getInfo=lambda: counts or None)


def _slice(platform, orbit, orbit_pass, slice_number):
    return {
        "platform_number": platform,
        "orbitNumber_start": orbit,
        "orbitProperties_pass": orbit_pass,
        "sliceNumber": slice_number,
    }


#: Three ascending acquisitions (one image each; S1A and S1C share an absolute orbit
#: number, which is per satellite) and two descending ones of two slices each.
S1_SLICES = [
    _slice("A", 100, "ASCENDING", 7),
    _slice("C", 100, "ASCENDING", 3),
    _slice("A", 112, "ASCENDING", 7),
    _slice("A", 105, "DESCENDING", 11),
    _slice("A", 105, "DESCENDING", 12),
    _slice("A", 117, "DESCENDING", 11),
    _slice("A", 117, "DESCENDING", 12),
]


def test_sentinel1_pass_counts_count_acquisitions_in_one_request():
    log = []
    counts = gee.sentinel1_pass_counts(_SliceCollection(S1_SLICES, log), "unit")
    assert counts == {"ASCENDING": 3, "DESCENDING": 2}
    assert all(isinstance(v, int) for v in counts.values())
    assert log == [
        ("distinct", ["platform_number", "orbitNumber_start"]),
        ("aggregate_histogram", "orbitProperties_pass"),
    ]
    # Counting images would pick the pass whose two acquisitions crossed a slice boundary.
    images = {"ASCENDING": 3, "DESCENDING": 4}
    assert gee.sentinel1_orbit_pass(images) == "DESCENDING"
    assert gee.sentinel1_orbit_pass(counts) == "ASCENDING"
    assert gee.sentinel1_pass_counts(_SliceCollection([], [])) == {}


def test_sentinel1_acquisitions_keep_one_slice_per_satellite_and_orbit():
    assert gee.S1_ACQUISITION_PROPERTIES == ("platform_number", "orbitNumber_start")
    kept = gee.sentinel1_acquisitions(_SliceCollection(S1_SLICES, [])).images
    assert [(i["platform_number"], i["orbitNumber_start"]) for i in kept] == [
        ("A", 100),
        ("C", 100),
        ("A", 112),
        ("A", 105),
        ("A", 117),
    ]
    assert "sentinel1_acquisitions" in gee.__all__


@pytest.mark.parametrize("method", ["greenest", "max_ndvi"])
def test_sentinel1_rejects_ndvi_composites(method):
    with pytest.raises(ValueError, match="needs NIR and red bands, which source 'sentinel1'"):
        gee.apply_composite_method(None, method, "sentinel1")


class TestSentinel1CoverageAdvice:
    TAGS = {
        "AGRIBOUND_ORBIT_PASS": "DESCENDING",
        "AGRIBOUND_ORBIT_PASS_ACQUISITIONS": "ASCENDING:25,DESCENDING:31",
    }

    def test_names_the_pass_and_the_acquisitions_of_both(self):
        advice = gee._coverage_advice(_cfg("sentinel1", year=2023), self.TAGS)
        assert advice.startswith(
            "The rest has no acquisition of the DESCENDING pass or is masked (VV "
        )
        assert "below -30 dB: swath edges, border noise" in advice
        assert "(acquisitions per pass: ASCENDING:25,DESCENDING:31)" in advice
        assert "a composite uses one pass" in advice and "HPC tiles" in advice
        # A calendar year already holds every relative orbit of the pass.
        assert "date" not in advice and "cloud" not in advice

    def test_wider_window_only_with_a_date_range(self):
        cfg = _cfg("sentinel1", year=2023, date_range=("2023-06-01", "2023-06-30"))
        advice = gee._coverage_advice(cfg, self.TAGS)
        assert advice.endswith(
            "A wider date_range can add acquisitions of this pass (and can change the pass)."
        )

    def test_without_tags(self):
        advice = gee._coverage_advice(_cfg("sentinel1", year=2023))
        assert "no acquisition of the orbit pass used" in advice
        assert "acquisitions per pass" not in advice

    def test_other_sources_unchanged(self):
        advice = gee._coverage_advice(_cfg("sentinel2", year=2023), self.TAGS)
        assert advice == (
            "The rest is masked (clouds, shadows) or has no observation; consider a wider date "
            "window or a higher cloud_cover_max."
        )


def test_sentinel1_no_data_error_names_the_scene_filters():
    error = gee._years_error(_cfg("sentinel1", year=2023), [2021], ("2023-01-01", "2024-01-01"))
    assert str(error).startswith(
        "No sentinel1 images (IW mode, 10 m, VV and VH polarisations) intersect the study-area "
        "extent for 2023-01-01 to 2024-01-01 (end exclusive)."
    )
    assert "cloud" not in str(error)


def test_sentinel1_recipe_version_is_in_its_cache_key_only(monkeypatch):
    from agribound._cache import cache_key

    configs = {s: _cfg(s, year=2023) for s in ("sentinel1", "sentinel2", "landsat-pan")}
    assert gee._recipe_parts("sentinel2") == (gee.COMPOSITE_RECIPE_VERSION,)
    assert gee._recipe_parts("sentinel1") == (
        gee.COMPOSITE_RECIPE_VERSION,
        f"sentinel1-recipe-{gee.S1_RECIPE_VERSION}",
    )

    def keys():
        return {s: cache_key(c, *gee._recipe_parts(s)) for s, c in configs.items()}

    before = keys()
    monkeypatch.setattr(gee, "S1_RECIPE_VERSION", "999")
    after = keys()
    assert after["sentinel1"] != before["sentinel1"]
    assert after["sentinel2"] == before["sentinel2"]
    assert after["landsat-pan"] == before["landsat-pan"]


def _s1_counts(monkeypatch, counts):
    """Stub the orbit-pass request of _build_sentinel1; return what it was asked to count."""
    seen = []

    def fake(collection, context="sentinel1"):
        seen.append((collection.name, context))
        return dict(counts)

    monkeypatch.setattr(gee, "sentinel1_pass_counts", fake)
    return seen


class TestSentinel1Collection:
    S1 = f"IC({gee.S1_COLLECTION})"
    BASE = f"IC({gee.S1_COLLECTION}).filterBounds.filter.filter.filter.filter"

    def test_scene_filters(self, recording_ee):
        collection = gee.sentinel1_collection("REGION")
        assert collection.name == self.BASE
        assert _calls(recording_ee, self.S1, "filterBounds") == [(self.S1, ("REGION",))]
        assert [args[0] for _, args in _calls(recording_ee, self.S1, "filter")] == [
            ("eq", "instrumentMode", "IW"),
            ("eq", "resolution_meters", 10),
            ("listContains", "transmitterReceiverPolarisation", "VV"),
            ("listContains", "transmitterReceiverPolarisation", "VH"),
        ]

    @pytest.mark.parametrize(
        ("counts", "orbit_pass"),
        [
            ({"ASCENDING": 12, "DESCENDING": 30}, "DESCENDING"),
            ({"ASCENDING": 40, "DESCENDING": 30}, "ASCENDING"),
            ({"ASCENDING": 9, "DESCENDING": 9}, "DESCENDING"),
        ],
    )
    def test_spec(self, recording_ee, monkeypatch, counts, orbit_pass):
        seen = _s1_counts(monkeypatch, counts)
        spec = gee._build_sentinel1(_cfg("sentinel1", year=2023), "REGION")
        # The passes are counted among the dated, scene-filtered images (one request) ...
        dated = f"{self.BASE}.filterDate"
        assert _calls(recording_ee, self.S1, "filterDate") == [
            (self.BASE, ("2023-01-01", "2024-01-01"))
        ]
        assert seen == [(dated, "sentinel1 2023")]
        # ... and the composite keeps the images of one pass.
        filters = [args[0] for _, args in _calls(recording_ee, self.S1, "filter")]
        assert filters[-1] == ("eq", "orbitProperties_pass", orbit_pass)
        assert not any(f[0] == "lte" for f in filters)  # no scene cloud filter
        assert spec.raw.name == f"{dated}.filter"
        ((name, (prepare,)),) = _calls(recording_ee, self.S1, "map")
        assert name == spec.raw.name and prepare is gee.prepare_sentinel1_image
        assert spec.history.name == self.BASE  # every date, both passes
        assert spec.bands == ["VV", "VH", "VV_VH"]
        assert spec.dtype == "uint8" and spec.resolution_m == 10.0
        assert spec.collections == ["COPERNICUS/S1_GRD"]
        assert spec.count_property == "platform_number"
        assert spec.count_labels == gee.S1_PLATFORMS
        # The composite uses every slice of the pass but counts its acquisitions.
        assert _calls(recording_ee, self.S1, "distinct") == [
            (spec.raw.name, (["platform_number", "orbitNumber_start"],))
        ]
        assert spec.counted.name == f"{spec.raw.name}.distinct"
        assert spec.count_unit == "acquisition"
        assert spec.round_composite is True

    def test_other_sources_count_raw_images_unrounded(self):
        spec = gee.CollectionSpec(
            raw="RAW",
            prepared=None,
            history=None,
            bands=[],
            resolution_m=10.0,
            dtype="float32",
            collections=[],
        )
        assert spec.counted is None and spec.count_unit == "image"
        assert spec.round_composite is False

    def test_notes_become_tags(self, recording_ee, monkeypatch):
        _s1_counts(monkeypatch, {"DESCENDING": 30, "ASCENDING": 12})
        notes = gee._build_sentinel1(_cfg("sentinel1", year=2023), "REGION").notes
        assert all(isinstance(v, str) for v in notes.values())  # each becomes AGRIBOUND_<KEY>
        assert notes["orbit_pass"] == "DESCENDING"
        assert notes["orbit_pass_acquisitions"] == "ASCENDING:12,DESCENDING:30"
        assert "orbit_pass_images" not in notes
        assert notes["image_counts"] == (
            "acquisitions: the GRD slices with the same platform_number and orbitNumber_start "
            "count once"
        )
        assert notes["instrument_mode"] == "IW" and notes["polarisation"] == "VV+VH"
        assert json.loads(notes["s1_db_ranges"]) == {
            "VV": [-25.0, 0.0],
            "VH": [-32.0, -8.0],
            "VV_VH": [0.0, 15.0],
        }
        assert notes["rgb_mapping"] == "R=VV, G=VH, B=VV_VH (VV - VH, dB)"
        assert "1 + round(254 * clamp((dB - lo) / (hi - lo), 0, 1))" in notes["scaling"]
        assert "then the median, rounded (halves up)" in notes["scaling"]
        assert "dB = lo + (value - 1) * (hi - lo) / 254" in notes["scaling"]
        assert notes["edge_mask"] == "VV < -30 dB masked in every band"
        assert "cloud_cover_max and s2_cloud_mask are ignored" in notes["cloud_filter"]
        assert notes["s1_recipe_version"] == gee.S1_RECIPE_VERSION

    def test_date_range_and_ignored_optical_settings(self, recording_ee, monkeypatch):
        _s1_counts(monkeypatch, {"ASCENDING": 3})
        cfg = _cfg(
            "sentinel1",
            year=2024,
            date_range=("2024-04-01", "2024-09-30"),
            cloud_cover_max=5,
            s2_cloud_mask="cloud_score_plus",
        )
        spec = gee._build_sentinel1(cfg, "REGION")
        assert _calls(recording_ee, self.S1, "filterDate")[0][1] == ("2024-04-01", "2024-10-01")
        assert spec.notes["orbit_pass"] == "ASCENDING"
        assert spec.notes["orbit_pass_acquisitions"] == "ASCENDING:3,DESCENDING:0"
        assert spec.collections == ["COPERNICUS/S1_GRD"]  # no Cloud Score+ link
        assert not any(attr == "linkCollection" for _, attr, _, _ in recording_ee)


def _sentinel1_build(tmp_path, aoi, monkeypatch, calls, counts, summary, values=(100.0,), **kwargs):
    """Build a sentinel1 composite with the stubbed Earth Engine layer; return builder, path, cfg.

    The collection spec is made by the real ``_build_sentinel1`` (recording ee, with
    the orbit-pass counts *counts*); *summary* is what the image-count request returns,
    and the prepared images hold one pixel each, of *values* in every band.
    """
    cfg = _cfg("sentinel1", year=2023, **kwargs).merged(
        study_area=aoi, output_path=str(tmp_path / "out.gpkg")
    )
    _s1_counts(monkeypatch, counts)
    spec = gee._build_sentinel1(cfg, "REGION")
    spec.prepared = FakeCollection([FakeImage({b: [[v]] for b in gee.S1_BANDS}) for v in values])
    monkeypatch.setitem(gee._COLLECTION_BUILDERS, "sentinel1", lambda cfg, region: spec)
    calls["summary"] = summary
    builder = gee.GEECompositeBuilder()
    return builder, builder.build(cfg), cfg


class TestSentinel1Build:
    def test_export_and_tags(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch, caplog
    ):
        from agribound._cache import cache_key

        summary = {"n_images": 30, "years": [2023], "counts": {"C": 14, "A": 16}}
        with caplog.at_level("INFO", logger="agribound.composites.gee"):
            builder, path, cfg = _sentinel1_build(
                tmp_path,
                sample_aoi_geojson,
                monkeypatch,
                stub_builder,
                {"ASCENDING": 12, "DESCENDING": 30},
                summary,
            )
        (call,) = stub_builder["export"]
        assert call["dtype"] == "uint8" and call["band_names"] == ["VV", "VH", "VV_VH"]
        # The image-count request counts acquisitions (one image per satellite and orbit).
        assert stub_builder["count_property"] == "platform_number"
        assert stub_builder["counted"].name.endswith(".filterDate.filter.distinct")
        assert stub_builder["summary_context"] == "sentinel1 2023 acquisition count"
        with rasterio.open(path) as src:
            assert src.count == 3 and src.res == (10, 10)
            tags = src.tags()
        assert tags["AGRIBOUND_SOURCE"] == "sentinel1"
        assert tags["AGRIBOUND_VALUE_SCALE"] == "uint8"
        assert tags["AGRIBOUND_COLLECTIONS"] == "COPERNICUS/S1_GRD"
        assert tags["AGRIBOUND_COMPOSITE_METHOD"] == "median"
        assert tags["AGRIBOUND_N_IMAGES"] == "30"
        assert tags["AGRIBOUND_ORBIT_PASS"] == "DESCENDING"
        assert tags["AGRIBOUND_ORBIT_PASS_ACQUISITIONS"] == "ASCENDING:12,DESCENDING:30"
        assert "AGRIBOUND_ORBIT_PASS_IMAGES" not in tags
        assert tags["AGRIBOUND_IMAGE_COUNTS"].startswith("acquisitions: the GRD slices")
        assert tags["AGRIBOUND_SENSORS"] == "S1A,S1C"
        assert tags["AGRIBOUND_SENSOR_IMAGES"] == "S1A:16,S1C:14"
        assert json.loads(tags["AGRIBOUND_S1_DB_RANGES"])["VH"] == [-32.0, -8.0]
        assert tags["AGRIBOUND_RGB_MAPPING"].startswith("R=VV, G=VH, B=VV_VH")
        assert tags["AGRIBOUND_S1_RECIPE_VERSION"] == gee.S1_RECIPE_VERSION
        assert "AGRIBOUND_SPECTRAL_RESPONSE" not in tags  # a landsat-pan tag
        # The Sentinel-1 recipe is part of the cache key (file name and tag).
        key = cache_key(cfg, *gee._recipe_parts("sentinel1"))
        assert tags["AGRIBOUND_CACHE_KEY"] == key
        assert path.endswith(f"sentinel1_composite_{key}.tif")
        assert cache_key(cfg, gee.COMPOSITE_RECIPE_VERSION) != key
        assert builder.last_metadata["AGRIBOUND_ORBIT_PASS"] == "DESCENDING"
        messages = [r.getMessage() for r in caplog.records]
        assert any("acquisitions per sensor: S1A 16, S1C 14" in m for m in messages)
        assert any(
            "acquisitions per orbit pass: ASCENDING 12, DESCENDING 30; the composite uses "
            "DESCENDING (more acquisitions; DESCENDING on a tie)" in m
            for m in messages
        )
        assert any(
            "sentinel1 2023: 30 acquisition(s) in 2023-01-01..2024-01-01" in m for m in messages
        )

    @pytest.mark.parametrize(
        ("values", "exported"),
        [
            ((100.0, 101.0), 101.0),  # median 100.5: rounded, not truncated by the uint8 export
            ((100.0, 102.0, 101.0, 103.0), 102.0),  # 101.5
            ((100.0, 101.0, 140.0), 101.0),  # an odd count: a value of the images
        ],
    )
    def test_median_is_rounded_before_the_uint8_export(
        self,
        tmp_path,
        sample_aoi_geojson,
        stub_builder,
        recording_ee,
        monkeypatch,
        values,
        exported,
    ):
        summary = {"n_images": len(values), "years": [2023], "counts": {"A": len(values)}}
        _sentinel1_build(
            tmp_path,
            sample_aoi_geojson,
            monkeypatch,
            stub_builder,
            {"DESCENDING": len(values)},
            summary,
            values=values,
        )
        (call,) = stub_builder["export"]
        for band in gee.S1_BANDS:
            assert call["image"].value(band).tolist() == [[exported]], band

    def test_other_sources_export_the_median_unrounded(
        self, tmp_path, sample_aoi_geojson, stub_builder, monkeypatch
    ):
        def spec(config, region):
            images = [FakeImage({b: [[v]] for b in gee.S2_BANDS}) for v in (100.0, 101.0)]
            return gee.CollectionSpec(
                raw="RAW",
                prepared=FakeCollection(images),
                history="HISTORY",
                bands=list(gee.S2_BANDS),
                resolution_m=10.0,
                dtype="float32",
                collections=[gee.S2_COLLECTION],
            )

        monkeypatch.setitem(gee._COLLECTION_BUILDERS, "sentinel2", spec)
        gee.GEECompositeBuilder().build(_s2_config(tmp_path, sample_aoi_geojson))
        (call,) = stub_builder["export"]
        assert call["image"].value("B4").tolist() == [[100.5]]
        assert stub_builder["counted"] == "RAW"
        assert stub_builder["summary_context"] == "sentinel2 2023 image count"

    def test_without_images_names_the_scene_filters(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch
    ):
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError) as info:
            _sentinel1_build(
                tmp_path,
                sample_aoi_geojson,
                monkeypatch,
                stub_builder,
                {},
                {"n_images": 0, "years": [], "counts": {}},
            )
        message = str(info.value)
        assert message.startswith("No sentinel1 images (IW mode, 10 m, VV and VH polarisations)")
        assert "Years with images over the study-area extent: 2019, 2021." in message
        assert stub_builder["export"] == []

    @pytest.mark.parametrize("method", ["greenest", "max_ndvi"])
    @pytest.mark.parametrize("source", ["sentinel1", "landsat-pan"])
    def test_ndvi_composites_refused_before_any_earth_engine_request(
        self, tmp_path, sample_aoi_geojson, monkeypatch, method, source
    ):
        def fail(*args, **kwargs):
            raise AssertionError("no Earth Engine call is expected")

        monkeypatch.setattr("agribound.auth.ensure_gee", fail)
        monkeypatch.setattr(gee, "sentinel1_pass_counts", fail)
        monkeypatch.setitem(gee._COLLECTION_BUILDERS, source, fail)
        match = f"needs NIR and red bands, which source '{source}'"
        with pytest.raises(ValueError, match=match):  # rejected by the configuration ...
            _cfg(source, year=2023, composite_method=method)
        cfg = _cfg(source, year=2023).merged(
            study_area=sample_aoi_geojson, output_path=str(tmp_path / "o.gpkg")
        )
        cfg.composite_method = method  # ... and by the builder, if changed after validation
        with pytest.raises(ValueError, match=match):
            gee.GEECompositeBuilder().build(cfg)

    def test_cached_composite_warns_with_the_sentinel1_advice(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch, caplog
    ):
        stub_builder["nan_fraction"] = 0.5
        summary = {"n_images": 8, "years": [2023], "counts": {"A": 8}}
        advice = (
            "The rest has no acquisition of the ASCENDING pass or is masked",
            "(acquisitions per pass: ASCENDING:8,DESCENDING:0)",
        )
        with caplog.at_level("WARNING", logger="agribound.composites.gee"):
            _, path, cfg = _sentinel1_build(
                tmp_path,
                sample_aoi_geojson,
                monkeypatch,
                stub_builder,
                {"ASCENDING": 8},
                summary,
            )
        assert any(all(a in r.getMessage() for a in advice) for r in caplog.records)
        caplog.clear()
        with caplog.at_level("WARNING", logger="agribound.composites.gee"):
            assert gee.GEECompositeBuilder().build(cfg) == path  # cached
        assert any(
            "cached sentinel1 composite" in r.getMessage()
            and all(a in r.getMessage() for a in advice)
            for r in caplog.records
        )

    def test_no_valid_pixel_counts_acquisitions(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch
    ):
        from agribound.composites import NoDataError

        stub_builder["nan_fraction"] = 1.0
        summary = {"n_images": 8, "years": [2023], "counts": {"A": 8}}
        with pytest.raises(NoDataError) as info:
            _sentinel1_build(
                tmp_path,
                sample_aoi_geojson,
                monkeypatch,
                stub_builder,
                {"DESCENDING": 8, "ASCENDING": 2},
                summary,
            )
        message = str(info.value)
        assert "although 8 acquisition(s) matched the filters" in message
        assert "The rest has no acquisition of the DESCENDING pass" in message
        assert "(acquisitions per pass: ASCENDING:2,DESCENDING:8)" in message


# ---------------------------------------------------------------------------
# HLS: reversed HLSS30 footprints at 180 degrees
# ---------------------------------------------------------------------------


def _gc_km(a, b, radius_km=6371.0):
    """Great-circle distance in km between two (lon, lat) points."""
    import math

    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    cos_d = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(
        lon2 - lon1
    )
    return radius_km * math.acos(max(-1.0, min(1.0, cos_d)))


class TestHlsFarPoints:
    @pytest.mark.parametrize(
        ("centroid", "p1", "p2"),
        [
            # The investigation's values (outputs/hls_antimeridian/investigate/NOTES.md).
            ((-1.518, 5.538), (178.482, -5.538), (-91.518, -5.538)),  # Twifo Praso
            ((179.7, -17.6), (-0.3, 17.6), (89.7, 17.6)),  # Fiji, west of 180 degrees
            ((-179.85, -17.6), (0.15, 17.6), (90.15, 17.6)),  # Fiji, east of 180 degrees
            ((149.775, -30.225), (-30.225, 30.225), (59.775, 30.225)),  # Murray-Darling
            ((0.0, 0.0), (-180.0, 0.0), (-90.0, 0.0)),
            ((180.0, 10.0), (0.0, -10.0), (90.0, -10.0)),
            ((-180.0, 10.0), (0.0, -10.0), (90.0, -10.0)),
        ],
    )
    def test_points(self, centroid, p1, p2):
        from shapely.geometry import Point

        got1, got2 = gee.hls_far_points(Point(*centroid))
        assert got1 == pytest.approx(p1, abs=1e-9)
        assert got2 == pytest.approx(p2, abs=1e-9)

    def test_points_of_a_grid_outline_come_from_its_centroid(self):
        p1, p2 = gee.hls_far_points(FIJI_WEST_BOX)
        c = FIJI_WEST_BOX.centroid
        assert p1 == pytest.approx((c.x - 180.0, -c.y))
        assert p2 == pytest.approx((c.x - 90.0, -c.y))

    def test_far_from_the_region_and_from_each_other(self):
        import math

        for lat in np.arange(-80.0, 80.1, 5.0):
            for lon in np.arange(-180.0, 180.0, 15.0):
                c = (float(lon), float(lat))
                p1, p2 = gee.hls_far_points(box(lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02))
                for p in (p1, p2):
                    assert -180.0 <= p[0] < 180.0 and p[1] == pytest.approx(-lat)
                # A real granule (about 110 km) that covers part of the region meets neither.
                assert _gc_km(c, p1) == pytest.approx(math.pi * 6371.0, abs=0.01)  # 20,015 km
                assert _gc_km(c, p2) >= 10_007.0
                # A reversed footprint's tile holds at most one of them.
                apart = _gc_km(p1, p2)
                assert apart == pytest.approx(
                    6371.0 * math.acos(math.sin(math.radians(lat)) ** 2), abs=0.01
                )
                assert apart > 1_000.0
        # 2,792 km at 72.1 degrees, the latitude reached by the most northerly affected tile.
        p1, p2 = gee.hls_far_points(box(179.0, 72.0, 179.2, 72.2))
        assert _gc_km(p1, p2) == pytest.approx(2_791.5, abs=0.5)

    @pytest.mark.parametrize(
        ("geometry", "near"),
        [
            (FIJI_WEST_BOX, True),
            (FIJI_EAST_BOX, True),
            (TWIFO_BOX, False),
            (LEA_BOX, False),
            (MURRAY_BOX, False),
            (box(175.0, 10.0, 175.1, 10.1), True),  # 5 degrees from 180
            (box(174.8, 10.0, 174.99, 10.1), False),
            (box(-175.0, 10.0, -174.9, 10.1), True),
            (box(-174.99, 10.0, -174.8, 10.1), False),
            # An outline whose longitudes wrap at 180 degrees spans -180 to 180.
            (box(-179.98, -17.0, 179.98, -16.9), True),
        ],
    )
    def test_near_antimeridian(self, geometry, near):
        assert gee.near_antimeridian(geometry) is near

    def test_margin(self):
        # Every affected MGRS tile reaches at most 3.0006 degrees from 180 (2026-10-09).
        assert gee.HLS_ANTIMERIDIAN_MARGIN_DEG == 5.0
        assert gee.near_antimeridian(box(170.0, 0.0, 170.5, 0.5), margin_deg=10.0)
        assert not gee.near_antimeridian(box(170.0, 0.0, 170.5, 0.5))
        for name in (
            "hls_antimeridian_images",
            "hls_far_points",
            "hls_footprint_filter",
            "near_antimeridian",
            "near_hls_reversed_footprint_tiles",
        ):
            assert name in gee.__all__

    @pytest.mark.parametrize(
        ("geometry", "near"),
        [
            # Inside the affected tiles (outputs/hls_antimeridian/verify): Koro (T01KAA),
            # Amchitka (T01UBT, T60UXC), Wrangel Island (T01WCV, T60WWE), the edge of T01KAA.
            (box(179.38, -17.36, 179.44, -17.30), True),
            (box(179.23, 51.48, 179.27, 51.52), True),
            (box(179.48, 71.28, 179.52, 71.32), True),
            (box(-179.76, -17.62, -179.71, -17.58), True),
            (FIJI_WEST_BOX, True),
            (box(178.8, -49.7, 178.85, -49.65), True),  # Antipodes Islands (T01FBE)
            (box(179.4, -9.5, 179.45, -9.45), True),  # Tuvalu (T01LAK)
            (box(179.95, -16.85, 180.0, -16.80), True),  # ends at 180 degrees
            (box(-180.0, 64.0, -179.0, 65.0), True),
            # Within 5 degrees of 180 but outside the latitudes of the affected tiles:
            # New Zealand's North Island, Kamchatka, Tonga.
            (box(175.0, -41.6, 178.6, -37.5), False),
            (box(175.0, 55.0, 178.0, 58.0), False),
            (box(-175.3, -21.3, -175.0, -21.0), False),
            # Away from 180 degrees.
            (TWIFO_BOX, False),
            (MURRAY_BOX, False),
            (box(170.0, -17.5, 174.9, -17.0), False),
        ],
    )
    def test_near_the_tiles_with_reversed_footprints(self, geometry, near):
        assert gee.near_hls_reversed_footprint_tiles(geometry) is near

    def test_latitudes_of_the_tiles_with_reversed_footprints(self):
        # outputs/hls_antimeridian/verify/results/v06_rings.json: the 31 tiles, merged into
        # ranges where they overlap or nearly do (the margin covers the gaps between them).
        assert gee.HLS_REVERSED_FOOTPRINT_LATITUDES == (
            (-50.6084, -46.8854),
            (-19.9628, -9.0282),
            (50.3874, 52.3414),
            (62.0859, 72.0996),
        )
        # 0.5 degrees of latitude beyond a range still counts; 0.6 does not.
        assert gee.near_hls_reversed_footprint_tiles(box(179.5, 52.7, 179.6, 52.8))
        assert not gee.near_hls_reversed_footprint_tiles(box(179.5, 52.95, 179.6, 53.0))

    def test_region_coordinates_need_a_request_only_for_a_computed_geometry(self):
        from shapely.geometry import mapping

        class Computed:
            def toGeoJSON(self):  # noqa: N802 - ee API name
                raise RuntimeError("Cannot convert a computed geometry to GeoJSON.")

            def getInfo(self):  # noqa: N802 - ee API name
                return mapping(FIJI_EAST_BOX)

        assert gee._region_geometry_4326(FakeRegion(TWIFO_BOX)).equals(TWIFO_BOX)
        assert gee._region_geometry_4326(Computed()).equals(FIJI_EAST_BOX)


def test_hls_antimeridian_images_keep_unbounded_images_whose_tile_meets_the_region(monkeypatch):
    """The per-image flag: an unbounded geometry and an even-odd ring that meets the region."""
    polygons, intersects = [], []

    class Coordinates:
        def __init__(self, meets):
            self.meets = meets

    class Ring:
        def __init__(self, meets):
            self.meets = meets

        def coordinates(self):
            return Coordinates(self.meets)

    class Tile:
        def __init__(self, rings, proj, geodesic, max_error, even_odd):
            polygons.append((proj, geodesic, max_error, even_odd))
            (self.coords,) = rings

        def intersects(self, region, max_error):
            intersects.append((region, max_error))
            return self.coords.meets

    class Image:
        def __init__(self, name, unbounded, meets):
            self.name, self.unbounded, self.meets = name, unbounded, meets

        def get(self, prop):
            assert prop == "system:footprint"
            return Ring(self.meets)

        def geometry(self):
            return types.SimpleNamespace(isUnbounded=lambda: self.unbounded)

        def set(self, prop, value):
            return {"name": self.name, prop: value}

    class Collection:
        def __init__(self, items, steps):
            self.items, self.steps = list(items), tuple(steps)

        def filterBounds(self, geometry):  # noqa: N802 - ee API name
            return Collection(self.items, (*self.steps, ("bounds", geometry)))

        def filter(self, value):
            items = self.items
            if value[0] == "eq":
                items = [i for i in items if i[value[1]] == value[2]]
            return Collection(items, (*self.steps, value))

        def filterDate(self, start, end):  # noqa: N802 - ee API name
            return Collection(self.items, (*self.steps, ("date", start, end)))

        def map(self, fn):
            return Collection([fn(i) for i in self.items], (*self.steps, ("map",)))

    images = [
        Image("reversed_here", True, True),
        Image("reversed_elsewhere", True, False),
        Image("bounded_here", False, True),
        Image("bounded_elsewhere", False, False),
    ]

    def geometry(value):
        return value

    geometry.Point = lambda coords: ("point", tuple(coords))
    geometry.Polygon = Tile
    module = types.ModuleType("ee")
    module.ImageCollection = lambda cid: Collection(images, [("collection", cid)])
    module.Geometry = geometry
    module.List = lambda items: items
    module.Algorithms = types.SimpleNamespace(If=lambda cond, a, b: a if cond else b)
    module.Filter = _fake_filter_namespace()
    monkeypatch.setitem(sys.modules, "ee", module)

    window = ("2022-01-01", "2023-01-01")
    out = gee.hls_antimeridian_images("REGION", FIJI_WEST_BOX, cloud_cover_max=80, window=window)
    assert [i["name"] for i in out.items] == ["reversed_here"]
    p1, _ = gee.hls_far_points(FIJI_WEST_BOX)
    assert out.steps == (
        ("collection", gee.HLSS30_COLLECTION),
        ("bounds", ("point", p1)),
        ("lte", "CLOUD_COVERAGE", 80),
        ("date", *window),
        ("map",),
        ("eq", "agribound_real_footprint", 1),
    )
    # The ring is read as a geodesic polygon with the even-odd rule; 1 m maximum error.
    assert set(polygons) == {(None, True, None, True)}
    assert set(intersects) == {("REGION", 1)}
    # Without a cloud filter or a window (the agent's live check, the history).
    out = gee.hls_antimeridian_images("REGION", FIJI_WEST_BOX)
    assert [s[0] for s in out.steps] == ["collection", "bounds", "map", "eq"]


@pytest.fixture
def answering_ee(monkeypatch):
    """Recording ``ee`` for the HLS builder near 180 degrees.

    The listing request, ``ee.Dictionary({"HLSL30": ..., "HLSS30": ...}).getInfo()``,
    returns ``state["ids"]`` (image IDs per collection name). The listed chains are kept in
    ``state["listings"]`` (their names) and each list of images loaded by ID in
    ``state["by_id"]``; a collection made from such a list is named
    ``IC(by-id <collection>)``.
    """
    state = {"log": [], "ids": {}, "by_id": [], "listings": []}
    module = types.ModuleType("ee")

    def image_collection(arg):
        if isinstance(arg, list):
            state["by_id"].append(list(arg))
            collection = arg[0][1].rsplit("/", 1)[0] if arg else "none"
            return Chain(f"IC(by-id {collection})", state["log"])
        return Chain(f"IC({arg})", state["log"])

    class Dictionary:
        def __init__(self, entries):
            state["listings"].append({key: value.name for key, value in entries.items()})
            self.keys = list(entries)

        def getInfo(self):  # noqa: N802 - ee API name
            state["log"].append(("Dictionary", "getInfo", (), {}))
            return {key: list(state["ids"].get(key, [])) for key in self.keys}

    module.ImageCollection = image_collection
    module.Dictionary = Dictionary
    module.Image = lambda asset_id: ("Image", asset_id)
    module.Filter = _fake_filter_namespace()
    module.Geometry = types.SimpleNamespace(Point=lambda coords: ("point", tuple(coords)))
    monkeypatch.setitem(sys.modules, "ee", module)
    return state


def _recipe_output(log, collection_name, source_bands, band):
    """Run the per-image recipe mapped over *collection_name* on a one-pixel image.

    Returns ``(value of band, value of the source band it comes from)``.
    """
    (fn,) = [args[0] for name, attr, args, _ in log if name == collection_name and attr == "map"]
    src = {b: np.full((1, 1), 0.01 * (i + 1)) for i, b in enumerate(source_bands)}
    src["Fmask"] = np.zeros((1, 1))
    _, out = fn(FakeImage(src))  # ("Image", ...) from _keep_time with the recording ee
    source_band = source_bands[gee.HLS_BANDS.index(band)]
    return out.value(band)[0, 0], src[source_band][0, 0]


class TestHlsNear180:
    S30 = f"IC({gee.HLSS30_COLLECTION})"
    L30 = f"IC({gee.HLSL30_COLLECTION})"
    IDS = ["T01KAA_20220917T221941", "T01KAA_20220523T221939"]
    #: The listing chain after ``IC(<collection>)``: bounded by p1, cloud and date filters,
    #: the per-image flag and its filter, the IDs.
    LISTED = ".filterBounds.filter.filterDate.map.filter.aggregate_array"

    @staticmethod
    def _merges(log):
        return {name: args[0].name for name, attr, args, _ in log if attr == "merge"}

    @pytest.mark.parametrize("region_box", [FIJI_WEST_BOX, FIJI_EAST_BOX])
    def test_listed_images_are_loaded_by_id(self, answering_ee, caplog, region_box):
        answering_ee["ids"] = {"HLSS30": list(self.IDS)}
        region = FakeRegion(region_box)
        with caplog.at_level("INFO", logger="agribound.composites.gee"):
            spec = gee._build_hls(_cfg("hls", cloud_cover_max=80), region)
        log = answering_ee["log"]
        p1, p2 = gee.hls_far_points(region_box)
        # One request lists the images of both collections at 180 degrees whose tile meets
        # the region (bounded by p1, cloud and date filters, flagged per image).
        assert [name for name, attr, _, _ in log if attr == "getInfo"] == ["Dictionary"]
        assert answering_ee["listings"] == [
            {"HLSL30": f"{self.L30}{self.LISTED}", "HLSS30": f"{self.S30}{self.LISTED}"}
        ]
        for ic in (self.L30, self.S30):
            listed = f"{ic}{self.LISTED}"
            assert (listed.rsplit(".", 1)[0], "aggregate_array", ("system:index",), {}) in log
            assert [args for _, args in _calls(log, ic, "filterBounds")] == [
                (("point", p1),),  # the listing
                (region,),  # the composite's selection
                (("point", p1),),  # the history
            ]
            assert (
                f"{ic}.filterBounds.filter",
                "filterDate",
                ("2022-01-01", "2023-01-01"),
                {},
            ) in log
        # Loaded by ID (sorted) and merged into the HLSS30 images of the window.
        assert answering_ee["by_id"] == [
            [("Image", f"{gee.HLSS30_COLLECTION}/{i}") for i in sorted(self.IDS)]
        ]
        merges = self._merges(log)
        s30_base = f"{self.S30}.filterBounds.filter.filter"
        s30_raw = f"{s30_base}.filterDate.merge"
        assert merges[f"{s30_base}.filterDate"] == f"IC(by-id {gee.HLSS30_COLLECTION})"
        # The history (years of the no-image error) takes them from the collection.
        assert merges[s30_base] == f"{self.S30}.filterBounds.filter.map.filter"
        l30_base = f"{self.L30}.filterBounds.filter.filter"
        l30_raw = f"{l30_base}.filterDate"  # no HLSL30 image listed
        assert merges[l30_raw] == s30_raw
        assert merges[l30_base] == f"{self.L30}.filterBounds.filter.map.filter"
        assert merges[f"{l30_base}.merge"] == f"{s30_base}.merge"
        assert spec.raw.name == f"{l30_raw}.merge"
        assert spec.history.name == f"{l30_base}.merge.merge"
        # The images loaded by ID go through the recipe with the others (Fmask, x10000,
        # B8A renamed to B5): the recipe is mapped over the merged HLSS30 images.
        assert spec.prepared.name == f"{l30_raw}.map.merge"
        assert merges[f"{l30_raw}.map"] == f"{s30_raw}.map"
        value, source = _recipe_output(log, s30_raw, gee.HLSS30_SOURCE_BANDS, "B5")
        assert value == pytest.approx(source * 10000)  # B8A
        # Both collections still drop the images that meet a far point.
        far = ("not", ("or", ("bounds", ("point", p1)), ("bounds", ("point", p2))))
        for ic in (self.L30, self.S30):
            assert (f"{ic}.filterBounds.filter", "filter", (far,), {}) in log
        assert spec.notes["hlss30_images_by_id"] == 2
        assert spec.notes["hlsl30_images_by_id"] == 0
        assert "0 HLSL30 and 2 HLSS30 image(s) of the MGRS tiles there" in caplog.text

    def test_listed_hlsl30_images_are_loaded_by_id_too(self, answering_ee):
        """HLSL30 had no reversed footprint on 2026-10-09; the far-point exclusion applies to
        it as well, so any it would drop at a tile at 180 degrees is added back."""
        l30_ids = ["T01KAA_20220602T221845"]
        answering_ee["ids"] = {"HLSL30": l30_ids, "HLSS30": list(self.IDS)}
        spec = gee._build_hls(_cfg("hls", cloud_cover_max=80), FakeRegion(FIJI_WEST_BOX))
        log = answering_ee["log"]
        assert [name for name, attr, _, _ in log if attr == "getInfo"] == ["Dictionary"]
        assert answering_ee["by_id"] == [
            [("Image", f"{gee.HLSL30_COLLECTION}/{l30_ids[0]}")],
            [("Image", f"{gee.HLSS30_COLLECTION}/{i}") for i in sorted(self.IDS)],
        ]
        merges = self._merges(log)
        l30_base = f"{self.L30}.filterBounds.filter.filter"
        s30_base = f"{self.S30}.filterBounds.filter.filter"
        l30_raw, s30_raw = f"{l30_base}.filterDate.merge", f"{s30_base}.filterDate.merge"
        assert merges[f"{l30_base}.filterDate"] == f"IC(by-id {gee.HLSL30_COLLECTION})"
        assert merges[f"{s30_base}.filterDate"] == f"IC(by-id {gee.HLSS30_COLLECTION})"
        assert merges[l30_raw] == s30_raw and spec.raw.name == f"{l30_raw}.merge"
        assert spec.prepared.name == f"{l30_raw}.map.merge"
        value, source = _recipe_output(log, l30_raw, gee.HLS_BANDS, "B5")
        assert value == pytest.approx(source * 10000)
        assert spec.notes["hlsl30_images_by_id"] == 1
        assert spec.notes["hlss30_images_by_id"] == 2

    def test_no_listed_image(self, answering_ee):
        answering_ee["ids"] = {}
        spec = gee._build_hls(_cfg("hls", cloud_cover_max=80), FakeRegion(FIJI_EAST_BOX))
        assert answering_ee["by_id"] == []
        merges = self._merges(answering_ee["log"])
        for ic in (self.L30, self.S30):
            base = f"{ic}.filterBounds.filter.filter"
            assert merges[base] == f"{ic}.filterBounds.filter.map.filter"
        s30_dated = f"{self.S30}.filterBounds.filter.filter.filterDate"
        assert s30_dated not in merges  # nothing to load by ID
        assert merges[f"{self.L30}.filterBounds.filter.filter.filterDate"] == s30_dated
        assert spec.notes["hlss30_images_by_id"] == 0
        assert spec.notes["hlsl30_images_by_id"] == 0

    @pytest.mark.parametrize("region_box", [TWIFO_BOX, LEA_BOX, MURRAY_BOX])
    def test_no_request_away_from_180(self, answering_ee, region_box):
        answering_ee["ids"] = {"HLSS30": list(self.IDS)}
        region = FakeRegion(region_box)
        spec = gee._build_hls(_cfg("hls"), region)
        log = answering_ee["log"]
        assert [name for name, attr, _, _ in log if attr == "getInfo"] == []
        assert answering_ee["listings"] == [] and answering_ee["by_id"] == []
        assert "hlss30_images_by_id" not in spec.notes
        assert "hlsl30_images_by_id" not in spec.notes
        # Only the composite's selection: no listing, no history of the images at 180 degrees.
        for ic in (self.L30, self.S30):
            assert _calls(log, ic, "filterBounds") == [(ic, (region,))]
            dated = f"{ic}.filterBounds.filter.filter.filterDate"
            assert [name for name, _ in _calls(log, ic, "map")] == [dated]  # the recipe
        assert self._merges(log)[f"{self.L30}.filterBounds.filter.filter"] == (
            f"{self.S30}.filterBounds.filter.filter"
        )

    def test_grid_across_180_uses_the_antipode_of_the_split_outline(self, answering_ee):
        """A study area that ends at 180 degrees: its grid reaches across 180 degrees, and the
        selection region is the outline split there (``ee_geometry``)."""
        area = box(179.95, -16.85, 180.0, -16.80)
        grid = gee.compute_export_grid(area, gee.resolve_export_crs("utm", area), 30.0)
        outline = gee.split_at_antimeridian(gee.grid_footprint_4326(grid))
        assert outline.geom_type == "MultiPolygon"
        spec = gee._build_hls(_cfg("hls", cloud_cover_max=80), FakeRegion(outline))
        p1, p2 = gee.hls_far_points(outline)
        assert p1 == pytest.approx((-0.0253, 16.8251), abs=1e-3)  # not near 180 degrees
        assert p2 == pytest.approx((89.9747, 16.8251), abs=1e-3)
        log = answering_ee["log"]
        assert [name for name, attr, _, _ in log if attr == "getInfo"] == ["Dictionary"]
        for ic in (self.L30, self.S30):
            assert [args for _, args in _calls(log, ic, "filterBounds")][0] == (("point", p1),)
        assert spec.notes["hlss30_images_by_id"] == 0


class TestAntimeridianOutline:
    """Export grids that reach across 180 degrees (their study area ends at or near it)."""

    #: Study areas whose UTM grid reaches across 180 degrees: east edge at 180 degrees, or
    #: up to 0.2 degrees short of it (outward rounding to whole pixels, and UTM grid
    #: convergence away from the equator); and one in zone 1 that starts at -180 degrees.
    ACROSS = [
        box(179.95, -16.85, 180.0, -16.80),  # 5 km, Fiji
        box(179.5, -17.0, 179.999, -16.9),  # 0.001 degrees short, 0.1 degrees tall
        box(179.5, -18.0, 179.97, -16.0),  # 0.03 degrees short, 2 degrees tall
        box(179.5, 51.8, 179.981, 52.1),  # Aleutian Islands
        box(179.0, 71.0, 179.95, 71.5),  # Chukotka, 1.8 km short
        box(179.5, 64.0, 179.8, 66.0),  # 0.2 degrees (about 9 km) short
        box(-180.0, -16.85, -179.95, -16.80),  # zone 1
    ]

    @staticmethod
    def _geojson_sent(monkeypatch, geometry):
        sent = []
        module = types.ModuleType("ee")
        module.Geometry = lambda geojson, proj, geodesic: sent.append((geojson, proj, geodesic))
        monkeypatch.setitem(sys.modules, "ee", module)
        gee.ee_geometry(geometry)
        ((geojson, proj, geodesic),) = sent
        assert proj is None and geodesic is False
        return shape(geojson)

    @pytest.mark.parametrize("area", ACROSS)
    def test_outline_is_split_at_180_for_earth_engine(self, monkeypatch, area):
        grid = gee.compute_export_grid(area, gee.resolve_export_crs("utm", area), 30.0)
        outline = gee.grid_footprint_4326(grid)
        # As one polygon, its longitudes run from about -180 to 180 (a band around the globe).
        assert outline.bounds[2] - outline.bounds[0] > 359.0
        region = self._geojson_sent(monkeypatch, outline)
        assert region.geom_type == "MultiPolygon" and region.is_valid
        west, east = sorted(region.geoms, key=lambda p: p.bounds[0], reverse=True)
        assert west.bounds[2] == 180.0 and east.bounds[0] == -180.0
        assert west.bounds[0] > 178.0 and east.bounds[2] < -179.0
        # The parts are the grid: no area lost or added (planar, longitudes unwrapped to
        # 0..360), and they cover the study area.
        unwrapped = gee._unwrap_antimeridian(region)
        assert unwrapped.bounds[2] - unwrapped.bounds[0] < 1.5
        assert unwrapped.area == pytest.approx(gee._unwrap_antimeridian(outline).area, rel=1e-9)
        assert unwrapped.covers(translate(area, xoff=360.0) if area.bounds[0] < 0 else area)
        # Far points from the centroid near 180 degrees, not from the planar one near 0.
        c = unwrapped.centroid
        p1, _ = gee.hls_far_points(region)
        assert p1 == pytest.approx((gee._wrap_longitude(c.x + 180.0), -c.y))
        assert abs(p1[0]) < 1.0
        for a, b in zip(gee.hls_far_points(outline), gee.hls_far_points(region), strict=True):
            assert a == pytest.approx(b, abs=1e-9)
        assert gee.near_antimeridian(region)

    @pytest.mark.parametrize(
        "area",
        [
            box(179.94, -16.85, 179.99, -16.80),  # 0.01 degrees short: the grid stays west
            box(179.38, -17.36, 179.44, -17.30),  # Koro, inside T01KAA
            box(-179.76, -17.62, -179.71, -17.58),  # the edge of T01KAA, zone 1
            box(-1.30, 5.55, -1.25, 5.60),  # Twifo Praso
        ],
    )
    def test_other_outlines_are_sent_unchanged(self, monkeypatch, area):
        grid = gee.compute_export_grid(area, gee.resolve_export_crs("utm", area), 30.0)
        outline = gee.grid_footprint_4326(grid)
        assert gee.split_at_antimeridian(outline) is outline
        assert self._geojson_sent(monkeypatch, outline).equals_exact(outline, 0.0)

    def test_geographic_grid_beyond_180(self, monkeypatch):
        area = box(179.95, -16.85, 180.0, -16.80)
        outline = gee.grid_footprint_4326(gee.compute_export_grid(area, "EPSG:4326", 30.0))
        assert outline.bounds[2] > 180.0  # whole pixels: about 1.6 m beyond 180 degrees
        region = self._geojson_sent(monkeypatch, outline)
        assert region.geom_type == "MultiPolygon"
        assert max(p.bounds[2] for p in region.geoms) == 180.0
        assert min(p.bounds[0] for p in region.geoms) == -180.0

    def test_split_at_antimeridian(self):
        from shapely import wkt
        from shapely.geometry import MultiPolygon, Point

        east, west = box(179.9, -17.0, 180.0, -16.9), box(-180.0, -17.0, -179.9, -16.9)
        split = MultiPolygon([east, west])  # as GeoJSON (RFC 7946) writes it
        out = gee.split_at_antimeridian(split)
        assert out.equals(split)
        assert gee.hls_far_points(split)[0] == pytest.approx((0.0, 16.95))
        # Written across 180 degrees with longitudes beyond it, or with a jump of nearly 360
        # degrees between two vertices: split there.
        for geometry in (
            box(179.9, -17.0, 180.1, -16.9),
            box(-180.1, -17.0, -179.9, -16.9),
            wkt.loads("POLYGON ((179.9 -17, -179.9 -17, -179.9 -16.9, 179.9 -16.9, 179.9 -17))"),
        ):
            assert gee.split_at_antimeridian(geometry).equals(split)
            assert gee.hls_far_points(geometry)[0] == pytest.approx((0.0, 16.95))
        # Left as they are: planar geometries wider than 180 degrees, a band around the
        # globe, and anything that is not a polygon.
        for geometry in (
            box(-100.0, 0.0, 100.0, 1.0),
            box(-180.0, -17.0, 180.0, -16.9),
            Point(181.0, 0.0),
            box(10.0, 10.0, 11.0, 11.0),
        ):
            assert gee.split_at_antimeridian(geometry) is geometry
        assert "split_at_antimeridian" in gee.__all__


#: Composite cache keys and configuration hashes of ``_cfg(source)`` (2022, a bbox study
#: area), computed with the code before the HLS footprint fix (2026-10-09): only the
#: ``hls`` composite key changes.
PRE_HLS_FIX_KEYS = {
    "landsat": ("2a9b6073c31a", "b04f4c8132cf4aec917f072e77adff3bcd341f94"),
    "landsat-pan": ("b77cfc775aed", "470ff7675494dcbb9377db9e169af2980874fbba"),
    "sentinel2": ("5de6dabfae84", "3ff932f35f5239bddcba7115a0384a9a19c1beb3"),
    "hls": ("2e95b83d1e69", "df35395f0fa347f7da5911974ecae2d4c9c8d415"),
    "naip": ("ee2ad206e3b6", "e6ff2493105823ec3a26bc08ccbf78a77ddf7fa9"),
    "spot": ("c44b59c7c9c0", "72ab8e510025f6bb48ff3f1ab20b604dee9c40f1"),
    "spot-pan": ("51deebe0339d", "e8ad4970e34226a9b9d4394564c914b880220c42"),
    "sentinel1": ("749ad92e743e", "7cdeaa8cde38d9a937565077a02e80a71a66dfa2"),
}


@pytest.mark.parametrize("source", list(PRE_HLS_FIX_KEYS))
def test_hls_fix_changes_only_the_hls_composite_key(source):
    from agribound._cache import cache_key
    from agribound._results import RESULTS_VERSIONS, results_versions
    from agribound.provenance import config_hash, reuse_mismatch

    assert set(PRE_HLS_FIX_KEYS) == set(gee._COLLECTION_BUILDERS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SPOT is a restricted source
        config = _cfg(source)
    old_key, old_hash = PRE_HLS_FIX_KEYS[source]
    key = cache_key(config, *gee._recipe_parts(source))
    assert config_hash(config) == old_hash
    if source == "hls":
        # Re-keyed once: the recipe before had no HLS part.
        assert key != old_key == cache_key(config, gee.COMPOSITE_RECIPE_VERSION)
        assert key == cache_key(config, gee.COMPOSITE_RECIPE_VERSION, "hls-recipe-2")
    else:
        assert key == old_key
    # No results-version component: outputs made before the fix are reused, away from the
    # MGRS tiles at 180 degrees (this study area is in New Mexico).
    assert not any("hls" in name for name in RESULTS_VERSIONS)
    default_engine = {"delineate_anything": RESULTS_VERSIONS["delineate_anything"]}
    assert results_versions(config) == default_engine
    record = {
        "status": "success",
        "config_hash": old_hash,
        "facts": {"results_versions": results_versions(config), "composite": {}},
    }
    assert reuse_mismatch(record, config) is None


def test_hls_recipe_version_is_in_its_cache_key_only(monkeypatch):
    from agribound._cache import cache_key
    from agribound.provenance import config_hash

    configs = {s: _cfg(s, year=2023) for s in ("hls", "sentinel1", "sentinel2", "landsat")}
    assert gee.HLS_RECIPE_VERSION == "2"
    assert gee._recipe_parts("hls") == (gee.COMPOSITE_RECIPE_VERSION, "hls-recipe-2")
    assert gee._recipe_parts("landsat") == (gee.COMPOSITE_RECIPE_VERSION,)

    def keys():
        return {s: cache_key(c, *gee._recipe_parts(s)) for s, c in configs.items()}

    def hashes():
        return {s: config_hash(c) for s, c in configs.items()}

    before, before_hashes = keys(), hashes()
    monkeypatch.setattr(gee, "HLS_RECIPE_VERSION", "999")
    after = keys()
    assert after["hls"] != before["hls"]
    assert {s: k for s, k in after.items() if s != "hls"} == {
        s: k for s, k in before.items() if s != "hls"
    }
    assert hashes() == before_hashes


def _hls_spec(tmp_path, aoi, monkeypatch, calls, summary, **kwargs):
    """An hls configuration and its spec (real ``_build_hls``, recording ee, Twifo's outline),
    served to the stubbed builder; *summary* is what the image-count request returns."""
    cfg = _cfg("hls", **kwargs).merged(study_area=aoi, output_path=str(tmp_path / "out.gpkg"))
    spec = gee._build_hls(cfg, FakeRegion(TWIFO_BOX))
    spec.prepared = FakeCollection([FakeImage({b: [[400.0]] for b in gee.HLS_BANDS})])
    monkeypatch.setitem(gee._COLLECTION_BUILDERS, "hls", lambda cfg, region: spec)
    calls["summary"] = summary
    return cfg, spec


class TestHlsBuild:
    def test_tags_and_the_new_cache_key(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch
    ):
        from pathlib import Path

        from agribound._cache import cache_key

        summary = {"n_images": 45, "years": [2022]}
        cfg, spec = _hls_spec(tmp_path, sample_aoi_geojson, monkeypatch, stub_builder, summary)
        # A composite cached by the code before the fix is not reused.
        old = Path(cfg.get_working_dir()) / (
            f"hls_composite_{cache_key(cfg, gee.COMPOSITE_RECIPE_VERSION)}.tif"
        )
        _write(old, np.ones((7, 2, 2), np.float32), Affine(30, 0, 500000, 0, -30, 4000060))
        builder = gee.GEECompositeBuilder()
        path = builder.build(cfg)
        (call,) = stub_builder["export"]
        key = cache_key(cfg, gee.COMPOSITE_RECIPE_VERSION, "hls-recipe-2")
        assert path == call["path"] != str(old)
        assert path.endswith(f"hls_composite_{key}.tif")
        tags = call["tags"]
        assert tags["AGRIBOUND_CACHE_KEY"] == key
        assert tags["AGRIBOUND_RECIPE_VERSION"] == gee.COMPOSITE_RECIPE_VERSION
        assert tags["AGRIBOUND_HLS_RECIPE_VERSION"] == "2"
        assert tags["AGRIBOUND_FOOTPRINT_FILTER"].startswith(
            "images whose footprint meets the antipode of the grid outline's centroid"
        )
        assert "AGRIBOUND_HLSS30_IMAGES_BY_ID" not in tags
        assert tags["AGRIBOUND_COLLECTIONS"] == "NASA/HLS/HLSL30/v002,NASA/HLS/HLSS30/v002"
        # AGRIBOUND_N_IMAGES counts the selection without the far-point images.
        assert stub_builder["counted"] is spec.raw
        assert stub_builder["summary_context"] == "hls 2022 image count"
        assert tags["AGRIBOUND_N_IMAGES"] == 45
        assert gee.GEECompositeBuilder().build(cfg) == path  # cached under the new key
        assert len(stub_builder["export"]) == 1

    def test_no_valid_pixel_reports_the_count_without_the_far_point_images(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch
    ):
        from agribound.composites import NoDataError

        stub_builder["nan_fraction"] = 1.0
        summary = {"n_images": 10, "years": [2024]}
        cfg, spec = _hls_spec(
            tmp_path,
            sample_aoi_geojson,
            monkeypatch,
            stub_builder,
            summary,
            year=2024,
            date_range=("2024-10-01", "2024-10-31"),
        )
        with pytest.raises(NoDataError) as info:
            gee.GEECompositeBuilder().build(cfg)
        assert stub_builder["counted"] is spec.raw
        assert "The hls 2024 composite has no valid pixel inside the study area although 10 " in (
            str(info.value)
        )

    def test_a_window_of_reversed_footprints_only_raises_the_no_image_error(
        self, tmp_path, sample_aoi_geojson, stub_builder, recording_ee, monkeypatch
    ):
        """Lea County, 2022-10-13 to 2022-10-31: 7 reversed-footprint images and no other.

        Counted without them, the window has no image: the no-image error with the years
        of the history (also without them), and no download.
        """
        from agribound.composites import NoDataError

        histories = []
        monkeypatch.setattr(
            gee, "available_years", lambda col, context="": histories.append(col) or [2022, 2023]
        )
        cfg, spec = _hls_spec(
            tmp_path,
            sample_aoi_geojson,
            monkeypatch,
            stub_builder,
            {"n_images": 0, "years": []},
            date_range=("2022-10-13", "2022-10-31"),
        )
        with pytest.raises(NoDataError) as info:
            gee.GEECompositeBuilder().build(cfg)
        assert str(info.value) == (
            "No hls images (scene cloud cover <= 20%) intersect the study-area extent for "
            "2022-10-13 to 2022-11-01 (end exclusive). Years with images over the study-area "
            "extent: 2022, 2023."
        )
        assert histories == [spec.history]
        assert stub_builder["export"] == []


# ---------------------------------------------------------------------------
# Image dates: the image list in the composite's tags
# ---------------------------------------------------------------------------

#: ``system:time_start`` (ms) of 2020-01-02 10:39:23 UTC and 2020-08-04 10:39:20 UTC.
T_0102 = 1_577_961_563_770
T_0804 = 1_596_537_560_000


class TestImageListRequest:
    def test_image_list_in_the_same_request(self, summary_ee):
        summary_ee["info"] = {
            "n": 2,
            "years": [2020],
            "counts": {"Sentinel-2A": 2},
            "images": [["S2/b", T_0804, 10.6], ["S2/a", T_0102, -1]],
            "dates": ["2020-08-04", "2020-01-02"],
        }
        collection = _SummaryCollection(summary_ee["log"])
        summary = gee.collection_summary(
            collection,
            "unit",
            count_property="SPACECRAFT_NAME",
            image_list=True,
            cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        )
        (query,) = summary_ee["queries"]  # one request
        assert set(query) == {"n", "years", "counts", "images", "dates"}
        assert summary["images"] == [["S2/b", T_0804, 10.6], ["S2/a", T_0102, -1]]
        assert summary["dates"] == ["2020-01-02", "2020-08-04"]
        assert summary["n_images"] == 2 and summary["counts"] == {"Sentinel-2A": 2}
        # The first IMAGE_LIST_MAX images by time, one row each.
        assert ("sort", "system:time_start") in summary_ee["log"]
        assert ("SORTED", "toList", (gee.IMAGE_LIST_MAX,), {}) in summary_ee["log"]
        calls = [entry for entry in summary_ee["log"] if len(entry) == 4]
        (mapped,) = [args[0] for name, _, args, _ in calls if name == "SORTED.toList"]
        assert callable(mapped)

    def test_request_unchanged_without_the_image_list(self, summary_ee):
        summary_ee["info"] = {"n": 3, "years": [2023], "images": [["x", T_0102, 1]]}
        summary = gee.collection_summary(_SummaryCollection(summary_ee["log"]), "unit")
        (query,) = summary_ee["queries"]
        assert set(query) == {"n", "years"}
        assert summary == {"n_images": 3, "years": [2023]}
        assert ("sort", "system:time_start") not in summary_ee["log"]

    def test_missing_lists_are_empty(self, summary_ee):
        summary_ee["info"] = {"n": 0, "years": [], "images": None}
        summary = gee.collection_summary(
            _SummaryCollection(summary_ee["log"]), "unit", image_list=True
        )
        assert summary["images"] == [] and summary["dates"] == []


class _RowImage:
    """An image for the image-list row function: properties only."""

    def __init__(self, props):
        self.props = props

    def get(self, name):
        return self.props.get(name)

    def toDictionary(self, names):  # noqa: N802 - ee API name
        return {n: self.props[n] for n in names if n in self.props}


@pytest.fixture
def row_ee(monkeypatch):
    module = types.ModuleType("ee")
    module.Image = lambda x: x

    class Dictionary(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    module.Dictionary = Dictionary
    module.List = list
    module.Algorithms = types.SimpleNamespace(If=lambda c, a, b: a if c else b)
    monkeypatch.setitem(sys.modules, "ee", module)
    return module


class TestImageRow:
    def test_asset_id_time_and_cloud(self, row_ee):
        row = gee._image_row("CLOUDY_PIXEL_PERCENTAGE")
        image = _RowImage(
            {
                "system:id": "COPERNICUS/S2_SR_HARMONIZED/x",
                "system:index": "1_x",
                "system:time_start": T_0804,
                "CLOUDY_PIXEL_PERCENTAGE": 10.626962,
            }
        )
        assert row(image) == ["COPERNICUS/S2_SR_HARMONIZED/x", T_0804, 10.626962]

    def test_index_without_an_asset_id_and_missing_cloud(self, row_ee):
        image = _RowImage({"system:index": "2_LC09_x", "system:time_start": T_0102})
        assert gee._image_row("CLOUD_COVER")(image) == ["2_LC09_x", T_0102, -1]
        assert gee._image_row(None)(image) == ["2_LC09_x", T_0102, -1]


class TestImageListTags:
    def test_rows_sorted_by_date_with_rounded_cloud(self):
        images = [
            ["COPERNICUS/S2_SR_HARMONIZED/b", T_0804, 10.626962],
            ["COPERNICUS/S2_SR_HARMONIZED/a", T_0102, 6.164276],
            ["COPERNICUS/S2_SR_HARMONIZED/c", T_0102 + 1000, -1],
        ]
        tags = gee._image_list_tags(images, 3, ["2020-01-02", "2020-08-04"])
        assert json.loads(tags["AGRIBOUND_IMAGE_LIST"]) == [
            ["COPERNICUS/S2_SR_HARMONIZED/a", "2020-01-02", 6.16],
            ["COPERNICUS/S2_SR_HARMONIZED/c", "2020-01-02", None],
            ["COPERNICUS/S2_SR_HARMONIZED/b", "2020-08-04", 10.63],
        ]
        assert " " not in tags["AGRIBOUND_IMAGE_LIST"]  # compact JSON
        assert tags["AGRIBOUND_IMAGE_DATES"] == 2
        assert tags["AGRIBOUND_IMAGE_DATE_FIRST"] == "2020-01-02"
        assert tags["AGRIBOUND_IMAGE_DATE_LAST"] == "2020-08-04"
        assert "AGRIBOUND_IMAGE_LIST_TRUNCATED" not in tags

    def test_utc_dates(self):
        # 2020-12-31 23:30 UTC is 2021-01-01 in UTC+1, but the list gives UTC dates.
        late = int(_dt_utc(2020, 12, 31, 23, 30).timestamp() * 1000)
        tags = gee._image_list_tags([["x", late, 0]], 1)
        assert json.loads(tags["AGRIBOUND_IMAGE_LIST"]) == [["x", "2020-12-31", 0.0]]
        assert tags["AGRIBOUND_IMAGE_DATES"] == 1  # from the rows when no dates are given

    @pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")
    @pytest.mark.parametrize("zone", ["America/Los_Angeles", "Asia/Tokyo", "UTC"])
    def test_utc_dates_in_any_local_time_zone(self, local_zone, zone):
        # 23:30 UTC is the next day east of UTC, and 00:30 UTC the day before west of it
        # (Oro's image of 2021-08-21 was acquired at 00:27 UTC): a date in local time
        # would differ from the UTC date in one of the two zones.
        late = int(_dt_utc(2020, 12, 31, 23, 30).timestamp() * 1000)
        early = int(_dt_utc(2021, 8, 21, 0, 30).timestamp() * 1000)
        local_zone(zone)
        tags = gee._image_list_tags([["a", late, 1], ["b", early, 2]], 2)
        assert json.loads(tags["AGRIBOUND_IMAGE_LIST"]) == [
            ["a", "2020-12-31", 1.0],
            ["b", "2021-08-21", 2.0],
        ]
        assert tags["AGRIBOUND_IMAGE_DATE_FIRST"] == "2020-12-31"
        assert tags["AGRIBOUND_IMAGE_DATE_LAST"] == "2021-08-21"

    def test_truncated_list_keeps_the_dates_of_every_image(self):
        images = [[f"id{i}", T_0102 + i * 86_400_000, 1] for i in range(3)]
        dates = ["2020-01-02", "2020-01-03", "2020-01-04", "2020-06-30", "2020-12-27"]
        tags = gee._image_list_tags(images, 1500, dates)
        assert len(json.loads(tags["AGRIBOUND_IMAGE_LIST"])) == 3
        assert tags["AGRIBOUND_IMAGE_LIST_TRUNCATED"] == "first 3 of 1500 by date"
        assert tags["AGRIBOUND_IMAGE_DATES"] == 5
        assert tags["AGRIBOUND_IMAGE_DATE_LAST"] == "2020-12-27"

    def test_no_image(self):
        tags = gee._image_list_tags([], 0, [])
        assert tags == {"AGRIBOUND_IMAGE_LIST": "[]", "AGRIBOUND_IMAGE_DATES": 0}

    def test_cap(self):
        assert gee.IMAGE_LIST_MAX == 1000 and "IMAGE_LIST_MAX" in gee.__all__


def _dt_utc(*args):
    import datetime

    return datetime.datetime(*args, tzinfo=datetime.UTC)


@pytest.fixture
def local_zone():
    """Set the process's local time zone (``TZ`` and ``time.tzset``); restored afterwards."""
    old = os.environ.get("TZ")

    def set_zone(zone):
        os.environ["TZ"] = zone
        time.tzset()

    yield set_zone
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()


@pytest.mark.parametrize(
    ("source", "prop"),
    [
        ("landsat", "CLOUD_COVER"),
        ("landsat-pan", "CLOUD_COVER"),
        ("sentinel2", "CLOUDY_PIXEL_PERCENTAGE"),
        ("spot", "cloud_coverage_percentage"),
        ("spot-pan", "cloud_coverage_percentage"),
        ("naip", None),
    ],
)
def test_specs_name_their_scene_cloud_property(recording_ee, source, prop):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SPOT is a restricted source
        cfg = _cfg(source)
    spec = gee._COLLECTION_BUILDERS[source](cfg, "REGION")
    assert spec.scene_cloud_property == prop
    filters = [args for _, attr, args, _ in recording_ee if attr == "filter"]
    if prop:  # the property the scene filter uses
        assert (("lte", prop, cfg.cloud_cover_max),) in filters


def test_hls_and_sentinel1_specs_name_their_scene_cloud_property(recording_ee, monkeypatch):
    hls = gee._build_hls(_cfg("hls"), FakeRegion(TWIFO_BOX))
    assert hls.scene_cloud_property == "CLOUD_COVERAGE"
    _s1_counts(monkeypatch, {"DESCENDING": 3})
    assert gee._build_sentinel1(_cfg("sentinel1"), "REGION").scene_cloud_property is None


#: Rows and dates of an image-count request (Sentinel-2 at Twifo Praso, 2020, abridged).
SUMMARY_WITH_IMAGES = {
    "n_images": 3,
    "years": [2023],
    "images": [
        ["COPERNICUS/S2_SR_HARMONIZED/b", T_0804, 10.626962],
        ["COPERNICUS/S2_SR_HARMONIZED/a", T_0102, 6.164276],
        ["COPERNICUS/S2_SR_HARMONIZED/c", T_0102, 0.5],
    ],
    "dates": ["2020-01-02", "2020-08-04"],
}

#: The tags of a Sentinel-2 composite of the stubbed builder before the image list
#: (``AGRIBOUND_VALID_FRACTION`` is added to the same mapping after the download).
TAGS_BEFORE_THE_IMAGE_LIST = {
    "AGRIBOUND_VERSION",
    "AGRIBOUND_SOURCE",
    "AGRIBOUND_VALUE_SCALE",
    "AGRIBOUND_YEAR",
    "AGRIBOUND_DATE_START",
    "AGRIBOUND_DATE_END_EXCLUSIVE",
    "AGRIBOUND_COMPOSITE_METHOD",
    "AGRIBOUND_N_IMAGES",
    "AGRIBOUND_IMAGE_YEARS",
    "AGRIBOUND_COLLECTIONS",
    "AGRIBOUND_CLOUD_COVER_MAX",
    "AGRIBOUND_EXPORT_CRS",
    "AGRIBOUND_RESOLUTION_M",
    "AGRIBOUND_CACHE_KEY",
    "AGRIBOUND_RECIPE_VERSION",
    "AGRIBOUND_VALID_FRACTION",
}


class TestBuilderImageList:
    def test_tags_list_the_images_and_add_only_new_names(
        self, tmp_path, sample_aoi_geojson, stub_builder
    ):
        from agribound._cache import cache_key

        stub_builder["summary"] = SUMMARY_WITH_IMAGES
        cfg = _s2_config(tmp_path, sample_aoi_geojson)
        builder = gee.GEECompositeBuilder()
        path = builder.build(cfg)
        (call,) = stub_builder["export"]
        tags = call["tags"]
        new = {
            "AGRIBOUND_IMAGE_LIST",
            "AGRIBOUND_IMAGE_DATES",
            "AGRIBOUND_IMAGE_DATE_FIRST",
            "AGRIBOUND_IMAGE_DATE_LAST",
        }
        assert set(tags) == TAGS_BEFORE_THE_IMAGE_LIST | new
        # The image-count request asked for the list (one request).
        assert stub_builder["image_list"] is True
        assert stub_builder["summary_context"] == "sentinel2 2023 image count"
        # The existing tags keep their values; the cache key is unchanged.
        assert tags["AGRIBOUND_N_IMAGES"] == 3 and tags["AGRIBOUND_IMAGE_YEARS"] == "2023"
        assert tags["AGRIBOUND_CACHE_KEY"] == cache_key(cfg, gee.COMPOSITE_RECIPE_VERSION)
        assert path.endswith(f"sentinel2_composite_{tags['AGRIBOUND_CACHE_KEY']}.tif")
        assert json.loads(tags["AGRIBOUND_IMAGE_LIST"]) == [
            ["COPERNICUS/S2_SR_HARMONIZED/a", "2020-01-02", 6.16],
            ["COPERNICUS/S2_SR_HARMONIZED/c", "2020-01-02", 0.5],
            ["COPERNICUS/S2_SR_HARMONIZED/b", "2020-08-04", 10.63],
        ]
        assert tags["AGRIBOUND_IMAGE_DATES"] == 2
        assert (tags["AGRIBOUND_IMAGE_DATE_FIRST"], tags["AGRIBOUND_IMAGE_DATE_LAST"]) == (
            "2020-01-02",
            "2020-08-04",
        )
        # Written to the GeoTIFF (and so copied into the provenance record).
        with rasterio.open(path) as src:
            written = src.tags()
        assert written["AGRIBOUND_IMAGE_LIST"] == tags["AGRIBOUND_IMAGE_LIST"]
        assert written["AGRIBOUND_IMAGE_DATES"] == "2"
        assert builder.last_metadata["AGRIBOUND_IMAGE_DATE_LAST"] == "2020-08-04"
        # The pipeline copies the composite's AGRIBOUND_* tags into facts["composite"].
        from agribound.pipeline import _composite_tags

        facts = _composite_tags(path)
        assert facts["AGRIBOUND_IMAGE_LIST"] == tags["AGRIBOUND_IMAGE_LIST"]
        assert facts["AGRIBOUND_IMAGE_DATE_FIRST"] == "2020-01-02"

    def test_the_spec_names_the_cloud_property_of_the_list(
        self, tmp_path, sample_aoi_geojson, stub_builder, monkeypatch
    ):
        def spec_with_cloud(config, region):
            img = FakeImage({b: np.ones((1, 1)) for b in gee.S2_BANDS})
            return gee.CollectionSpec(
                raw="RAW",
                prepared=FakeCollection([img]),
                history="HISTORY",
                bands=list(gee.S2_BANDS),
                resolution_m=10.0,
                dtype="float32",
                collections=[gee.S2_COLLECTION],
                scene_cloud_property="CLOUDY_PIXEL_PERCENTAGE",
            )

        monkeypatch.setitem(gee._COLLECTION_BUILDERS, "sentinel2", spec_with_cloud)
        stub_builder["summary"] = SUMMARY_WITH_IMAGES
        gee.GEECompositeBuilder().build(_s2_config(tmp_path, sample_aoi_geojson))
        assert stub_builder["cloud_property"] == "CLOUDY_PIXEL_PERCENTAGE"
        assert stub_builder["counted"] == "RAW"

    def test_more_images_than_the_cap(self, tmp_path, sample_aoi_geojson, stub_builder):
        stub_builder["summary"] = {**SUMMARY_WITH_IMAGES, "n_images": 1500}
        gee.GEECompositeBuilder().build(_s2_config(tmp_path, sample_aoi_geojson))
        tags = stub_builder["export"][0]["tags"]
        assert tags["AGRIBOUND_IMAGE_LIST_TRUNCATED"] == "first 3 of 1500 by date"
        assert tags["AGRIBOUND_N_IMAGES"] == 1500

    def test_a_cached_composite_keeps_its_old_tags(
        self, tmp_path, sample_aoi_geojson, stub_builder
    ):
        """A composite cached before the image list is reused as it is (same key, no list)."""
        from agribound._cache import cache_path

        cfg = _s2_config(tmp_path, sample_aoi_geojson)
        old = cache_path(cfg, "sentinel2_composite", ".tif", gee.COMPOSITE_RECIPE_VERSION)
        _write(old, np.ones((12, 2, 2), np.float32), Affine(10, 0, 500000, 0, -10, 4000020))
        with rasterio.open(old, "r+") as dst:
            dst.update_tags(AGRIBOUND_N_IMAGES="3", AGRIBOUND_VALID_FRACTION="1.0")
        builder = gee.GEECompositeBuilder()
        assert builder.build(cfg) == str(old)
        assert stub_builder["export"] == []
        assert "AGRIBOUND_IMAGE_LIST" not in builder.last_metadata
        assert builder.last_metadata["AGRIBOUND_N_IMAGES"] == "3"


# ---------------------------------------------------------------------------
# NAIP: which neighbouring years show in the mosaic
# ---------------------------------------------------------------------------


class _ShapelyGeometry:
    """``ee.Geometry`` stand-in backed by shapely (planar areas)."""

    def __init__(self, geometry):
        self.geometry = geometry

    def difference(self, other, max_error):
        return _ShapelyGeometry(self.geometry.difference(other.geometry))

    def intersection(self, other, max_error):
        return _ShapelyGeometry(self.geometry.intersection(other.geometry))

    def area(self, max_error):
        return self.geometry.area


class _Footprints:
    """NAIP collection stand-in: ``(date, footprint)`` pairs; ``geometry()`` is their union."""

    def __init__(self, images):
        self.images = list(images)

    def filterDate(self, start, end):  # noqa: N802 - ee API name
        return _Footprints([(d, g) for d, g in self.images if start <= d < end])

    def geometry(self, max_error):
        from shapely.ops import unary_union

        return _ShapelyGeometry(unary_union([g for _, g in self.images]))


class TestNaipYearsShown:
    """The 1.1 run of example 25 warned "images from [...] fill gaps" in 7 squares, and
    listed those years in ``AGRIBOUND_IMAGE_YEARS``, although the run year's images
    covered every square: the other years' images lay under them."""

    def test_year_areas_follow_the_mosaic_order(self, monkeypatch):
        module = types.ModuleType("ee")
        module.Dictionary = lambda query: types.SimpleNamespace(getInfo=lambda: dict(query))
        monkeypatch.setitem(sys.modules, "ee", module)
        raw = _Footprints(
            [
                ("2022-06-01", box(0, 0, 10, 6)),  # the run year: the lower 60 %
                ("2023-07-01", box(0, 5, 10, 8)),  # newer, so above 2021
                ("2021-05-01", box(0, 0, 10, 10)),
            ]
        )
        areas = gee.naip_year_areas(raw, _ShapelyGeometry(box(0, 0, 10, 10)), 2022, [2021, 2023])
        assert areas == {"area_m2": 100.0, "uncovered_m2": 40.0, "2023": 20.0, "2021": 20.0}
        covered = gee.naip_year_areas(
            _Footprints([("2022-06-01", box(-1, -1, 11, 11)), ("2021-05-01", box(0, 0, 10, 10))]),
            _ShapelyGeometry(box(0, 0, 10, 10)),
            2022,
            [2021],
        )
        assert covered == {"area_m2": 100.0, "uncovered_m2": 0.0, "2021": 0.0}

    @pytest.fixture
    def naip_build(self, tmp_path, sample_aoi_geojson, stub_builder, monkeypatch):
        """Build a NAIP 2022 composite; *areas* is what naip_year_areas returns (or raises)."""

        def fake_spec(config, region):
            return gee.CollectionSpec(
                raw="RAW",
                prepared=types.SimpleNamespace(mosaic=lambda: "MOSAIC"),
                history="HISTORY",
                bands=list(gee.NAIP_BANDS),
                resolution_m=1.0,
                dtype="uint8",
                collections=[gee.NAIP_COLLECTION],
            )

        monkeypatch.setitem(gee._COLLECTION_BUILDERS, "naip", fake_spec)
        requests = []

        def build(areas, years=(2021, 2022, 2023), **kwargs):
            def fake_areas(raw, region, year, other):
                requests.append((raw, region, year, list(other)))
                if isinstance(areas, Exception):
                    raise areas
                return areas

            monkeypatch.setattr(gee, "naip_year_areas", fake_areas)
            stub_builder["summary"] = {"n_images": 6, "years": list(years)}
            config = AgriboundConfig(
                source="naip",
                year=2022,
                study_area=sample_aoi_geojson,
                gee_project="test-project",
                output_path=str(tmp_path / "out.gpkg"),
                lulc_filter=False,
                **kwargs,
            )
            gee.GEECompositeBuilder().build(config)
            return stub_builder["export"][-1]["tags"]

        build.requests = requests
        return build

    @staticmethod
    def _messages(caplog, level):
        return [r.getMessage() for r in caplog.records if r.levelno == level]

    def test_years_under_the_run_year_are_not_listed(self, naip_build, caplog):
        import logging

        areas = {"area_m2": 4e6, "uncovered_m2": 0.0, "2021": 0.0, "2023": 1.0}  # <= 1 pixel
        with caplog.at_level(logging.INFO, logger="agribound.composites.gee"):
            tags = naip_build(areas)
        assert naip_build.requests == [("RAW", "REGION", 2022, [2021, 2023])]
        assert tags["AGRIBOUND_IMAGE_YEARS"] == "2022"
        assert tags["AGRIBOUND_IMAGE_YEARS_UNDERNEATH"] == "2021,2023"
        assert not any("NAIP" in m for m in self._messages(caplog, logging.WARNING))
        assert (
            "NAIP mosaic for 2022: images from [2021, 2023] intersect the study-area extent but "
            "lie under the 2022 images." in self._messages(caplog, logging.INFO)
        )

    def test_a_year_that_fills_part_of_the_extent_is_listed_with_its_share(
        self, naip_build, caplog
    ):
        import logging

        areas = {"area_m2": 4e6, "uncovered_m2": 2e5, "2021": 0.5, "2023": 1.2e5}
        with caplog.at_level(logging.INFO, logger="agribound.composites.gee"):
            tags = naip_build(areas)
        assert tags["AGRIBOUND_IMAGE_YEARS"] == "2022,2023"
        assert tags["AGRIBOUND_IMAGE_YEARS_UNDERNEATH"] == "2021"
        assert self._messages(caplog, logging.WARNING) == [
            "NAIP mosaic for 2022: 2022 images are on top; images from [2023] fill 3 % of the "
            "study-area extent; images from [2021] lie under them."
        ]

    def test_without_the_measurement_every_year_is_listed(self, naip_build, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="agribound.composites.gee"):
            tags = naip_build(RuntimeError("429 Too Many Requests"))
        assert tags["AGRIBOUND_IMAGE_YEARS"] == "2021,2022,2023"
        assert "AGRIBOUND_IMAGE_YEARS_UNDERNEATH" not in tags
        (warning,) = self._messages(caplog, logging.WARNING)
        assert "images from [2021, 2023] may fill gaps" in warning
        assert "not measured: 429 Too Many Requests" in warning

    @pytest.mark.parametrize(
        ("years", "kwargs"),
        [
            ((2022,), {}),  # only the run year
            ((2021, 2022), {"date_range": ("2021-06-01", "2022-09-30")}),  # newest on top
            ((2021, 2023), {}),  # no run-year image: the existing WARNING, all years
        ],
    )
    def test_no_request_without_other_years_under_the_run_year(self, naip_build, years, kwargs):
        tags = naip_build({}, years=years, **kwargs)
        assert naip_build.requests == []
        assert tags["AGRIBOUND_IMAGE_YEARS"] == ",".join(map(str, years))
        assert "AGRIBOUND_IMAGE_YEARS_UNDERNEATH" not in tags


#: The four agribound 1.0.1 runs of example 23 recorded on 2026-09-30 (their provenance
#: records): the configuration as recorded, its configuration hash and the composite's
#: cache key. The study areas are example 23's squares (centre and side in SITES), here as
#: the same polygons in WKT: the cache key depends on the geometry, not on the file name.
#: The recorded gee_project is left out: neither the hash nor the key uses it.
RELEASE_1_0_1_BASE = {
    "aoi_selection": "representative_point",
    "bands": None,
    "cache_dir": None,
    "cloud_cover_max": 80,
    "cloud_score_threshold": 0.6,
    "composite_method": "median",
    "date_range": None,
    "device": "auto",
    "embedding_cache_dir": None,
    "engine": "delineate-anything",
    "engine_params": {"max_detections": 1000},
    "export_crs": "utm",
    "export_method": "local",
    "fine_tune": False,
    "fine_tune_block_size_m": 5000.0,
    "fine_tune_epochs": 20,
    "fine_tune_split": "block",
    "fine_tune_split_column": None,
    "fine_tune_val_split": 0.2,
    "gcs_bucket": None,
    "gee_high_volume": False,
    "gee_max_requests": 8,
    "gee_project": None,
    "gee_service_account_key": None,
    "gee_workload_tag": None,
    "google_embedding_backend": "gee",
    "landsat_pan_missions": "auto",
    "local_tif_path": None,
    "lulc_batch_size": 200,
    "lulc_crop_threshold": 0.3,
    "lulc_dataset": "auto",
    "lulc_filter": False,
    "lulc_mode": "server",
    "lulc_nodata_policy": "keep",
    "lulc_on_error": "raise",
    "lulc_tree_crops": False,
    "min_field_area_m2": 2500.0,
    "n_workers": 4,
    "naip_resolution_m": 1.0,
    "output_format": "gpkg",
    "output_path": "outputs/tree_crops/twifo/fields_sentinel2_da_2020.gpkg",
    "overwrite": False,
    "provenance": True,
    "reference_boundaries": None,
    "s2_cloud_mask": "cloud_score_plus",
    "sam_backend": "sam2",
    "sam_crop_padding": 0.15,
    "sam_min_crop_px": 64,
    "sam_model": None,
    "sam_refine": False,
    "seed": 42,
    "simplify_tolerance": 2.0,
    "source": "sentinel2",
    "study_area": "outputs/tree_crops/twifo/study_area.geojson",
    "tessera_variant": None,
    "tessera_version": "v1",
    "tile_size": 10000,
    "usgs_allow_year_fallback": False,
    "usgs_retries": 3,
    "usgs_service_url": (
        "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
    ),
    "usgs_state": None,
    "usgs_timeout_s": 120,
    "year": 2020,
}
TWIFO_WKT = (
    "POLYGON ((-1.4955032599320377 5.515674790288927, -1.495388865913433 5.560891430400164, "
    "-1.5405184541386092 5.56100479685173, -1.540629420016046 5.515787229421631, "
    "-1.4955032599320377 5.515674790288927))"
)
ORO_WKT = (
    "POLYGON ((148.1387484537715 -8.813674076127333, 148.13866551636252 -8.786543907157466, "
    "148.11139253897002 -8.786625292600124, 148.1114734908959 -8.813755716709187, "
    "148.1387484537715 -8.813674076127333))"
)
RELEASE_1_0_1_RUNS = {
    "twifo sentinel2 da 2020": (
        {},
        TWIFO_WKT,
        "f78e64763c12ede972d3111e8183a99e51e4d0f1",
        "e2c329c1855c",
    ),
    "oro sentinel2 da 2021": (
        {
            "output_path": "outputs/tree_crops/oro/fields_sentinel2_da_2021.gpkg",
            "study_area": "outputs/tree_crops/oro/study_area.geojson",
            "year": 2021,
        },
        ORO_WKT,
        "d767a270d6580cc35c8bf48092bedb7f67de9345",
        "24daf529b64a",
    ),
    "twifo sentinel2 ftw 2020": (
        {
            "engine": "ftw",
            "engine_params": {"window_dates": ["2020-01-15", "2020-08-15"]},
            "output_path": "outputs/tree_crops/twifo/fields_sentinel2_ftw_2020.gpkg",
        },
        TWIFO_WKT,
        "3b689611b3eace06682ebe1f49705bcded9acd62",
        "e2c329c1855c",
    ),
    "twifo landsat-pan da 2020": (
        {
            "cloud_cover_max": 20,
            "output_path": "outputs/tree_crops/twifo/fields_landsat-pan_da_2020.gpkg",
            "s2_cloud_mask": "scl",
            "source": "landsat-pan",
        },
        TWIFO_WKT,
        "54fe63ae1a496a959e4e4840fb8c8320bf0f5985",
        "cb79d2e12c3e",
    ),
}


@pytest.mark.parametrize("run", list(RELEASE_1_0_1_RUNS))
def test_release_1_0_1_hashes_and_composite_keys_are_unchanged(run, monkeypatch):
    """The image list adds tags only: 1.0.1's configuration hashes and composite keys hold."""
    from agribound._cache import cache_key
    from agribound.provenance import config_hash

    # The configuration needs an Earth Engine project, which a CI runner does not have
    # ($GEE_PROJECT, gcloud or a key); the project is not part of the hash or the key.
    monkeypatch.setenv("GEE_PROJECT", "test-project")
    changes, wkt, expected_hash, expected_key = RELEASE_1_0_1_RUNS[run]
    config = AgriboundConfig.from_dict({**RELEASE_1_0_1_BASE, **changes})
    assert config_hash(config) == expected_hash
    keyed = config.merged(study_area=wkt)
    assert cache_key(keyed, *gee._recipe_parts(keyed.source)) == expected_key


def test_image_list_tags_survive_odd_values():
    """A missing time or an odd cloud value never breaks a composite's tags."""
    images = [
        ["a", T_0102, "NaN"],
        ["b", None, 12.345],
        ["c", T_0804, "not a number"],
        ["d"],
    ]
    tags = gee._image_list_tags(images, 4)
    rows = json.loads(tags["AGRIBOUND_IMAGE_LIST"])  # strict JSON: no NaN
    assert rows == [
        ["b", None, 12.35],
        ["d", None, None],
        ["a", "2020-01-02", None],
        ["c", "2020-08-04", None],
    ]
    assert tags["AGRIBOUND_IMAGE_DATES"] == 2
    assert "NaN" not in tags["AGRIBOUND_IMAGE_LIST"]
