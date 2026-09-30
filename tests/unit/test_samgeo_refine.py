"""Tests for box-prompted SAM refinement (agribound.engines.samgeo_engine)."""

from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from agribound.config import AgriboundConfig
from agribound.engines import samgeo_engine as se
from agribound.engines.samgeo_engine import (
    _normalise_sam_output,
    _plan_windows,
    crop_window_px,
    is_refinable,
    refine_boundaries,
    resolve_sam_model,
)

UTM = "EPSG:32611"
X0, Y1 = 500000.0, 4003000.0  # raster top-left corner
RES = 10.0


def _raster(tmp_path, count=12, size=300, dtype="float32", fill=None, name="img.tif", seed=0):
    rng = np.random.default_rng(seed)
    if fill is not None:
        data = fill
        count = data.shape[0]
    elif dtype == "uint8":
        data = rng.integers(1, 255, (count, size, size), dtype=np.uint8)
    else:
        data = rng.uniform(100, 4000, (count, size, size)).astype(dtype)
    path = tmp_path / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=count,
        dtype=str(data.dtype),
        crs=UTM,
        transform=from_origin(X0, Y1, RES, RES),
        nodata=float("nan") if np.issubdtype(data.dtype, np.floating) else None,
    ) as dst:
        dst.write(data)
    return str(path)


def _config(tmp_path, source="sentinel2", **kw):
    base = {
        "source": source,
        "engine": "delineate-anything",
        "year": 2023,
        "output_path": str(tmp_path / "out.gpkg"),
        "lulc_filter": False,
        "sam_refine": True,
        "device": "cpu",
    }
    if source in ("sentinel2", "naip", "landsat", "hls"):
        base["gee_project"] = "test-project"
    if source == "local":
        base["local_tif_path"] = kw.pop("local_tif_path")
    if source in ("google-embedding", "tessera-embedding"):
        base["engine"] = "embedding"
    base.update(kw)
    return AgriboundConfig(**base)


class FakePredictor:
    """Returns the box interior (inset by *inset* px) as the mask; records calls."""

    backend = "fake"
    model_id = "fake/box"
    device = "cpu"

    def __init__(self, inset=1, empty_for=(), raise_on_image=None):
        self.inset = inset
        self.empty_for = set(empty_for)
        self.raise_on_image = raise_on_image
        self.images = []
        self.batches = []
        self._n = -1

    def set_image(self, image):
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        assert image.flags["C_CONTIGUOUS"]
        self._n += 1
        if self.raise_on_image is not None and self._n in self.raise_on_image:
            raise RuntimeError(f"boom {self._n}")
        self.images.append(image.copy())
        self.shape = image.shape[:2]

    def predict_boxes(self, boxes):
        self.batches.append(np.array(boxes))
        h, w = self.shape
        masks = np.zeros((len(boxes), h, w), dtype=bool)
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
            if (round(x0), round(y0)) in self.empty_for:
                continue
            i = self.inset
            masks[k, round(y0) + i : round(y1) - i, round(x0) + i : round(x1) - i] = True
        return masks, np.linspace(0.5, 0.9, len(boxes))


def _field(col, row, w_px, h_px):
    """Axis-aligned box given in raster pixel units (top-left col/row)."""
    return box(X0 + col * RES, Y1 - (row + h_px) * RES, X0 + (col + w_px) * RES, Y1 - row * RES)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


class TestGating:
    def test_crop_window_formula(self):
        # 49.3 px wide -> 49.3 * 1.3 = 64.09 -> 64 ; 49 px -> 63.7 -> 63
        assert crop_window_px((0, 0, 493, 490), (10, 10), 0.15) == (64, 63)
        assert crop_window_px((0, 0, 100, 50), (10, -10), 0.0) == (10, 5)  # sign ignored

    def test_exact_threshold_counts_as_refinable(self):
        side = 64 / 1.3 * 10  # exactly 64 px after padding
        assert is_refinable((0, 0, side, side), (10, 10), 64, 0.15)
        assert not is_refinable((0, 0, side - 0.01, side), (10, 10), 64, 0.15)

    def test_threshold_follows_parameters(self):
        b = (0, 0, 400, 400)  # 40 px
        assert not is_refinable(b, (10, 10))  # defaults 64 px / 15 %: 52 px
        assert is_refinable(b, (10, 10), min_crop_px=52)
        assert is_refinable(b, (10, 10), padding=0.3)  # 40 * 1.6 = 64
        assert is_refinable(b, (5, 5))  # 80 px * 1.3

    def test_either_side_too_small_skips(self):
        assert not is_refinable((0, 0, 2000, 300), (10, 10))  # 260 x 39 px

    def test_invalid_inputs(self):
        assert crop_window_px((np.nan,) * 4, (10, 10), 0.15) == (0, 0)
        assert not is_refinable((np.nan,) * 4, (10, 10))
        with pytest.raises(ValueError):
            crop_window_px((0, 0, 1, 1), (0, 10), 0.15)
        with pytest.raises(ValueError):
            crop_window_px((0, 0, 1, 1), (10, 10), -0.1)

    def test_raster_bounds_excludes_outside_and_straddling_boxes(self):
        raster = (0.0, 0.0, 2000.0, 2000.0)  # 200 x 200 px at 10 m
        inside = (500, 500, 1500, 1500)
        assert is_refinable(inside, (10, 10), raster_bounds=raster)
        # Entirely right of the raster; only its padded box reaches back inside.
        beyond = (2020, 1000, 3020, 1900)
        assert is_refinable(beyond, (10, 10))  # the size test alone accepts it
        assert not is_refinable(beyond, (10, 10), raster_bounds=raster)
        # 100 px field with only 1 px inside the raster.
        assert not is_refinable((1990, 500, 2990, 1500), (10, 10), raster_bounds=raster)
        # Up to half a pixel beyond the edge is tolerated.
        assert is_refinable((1000, 500, 2005, 1500), (10, 10), raster_bounds=raster)
        assert not is_refinable((1000, 500, 2005.1, 1500), (10, 10), raster_bounds=raster)
        # Order of the raster's y values does not matter (south-up rasters).
        assert is_refinable(inside, (10, 10), raster_bounds=(0.0, 2000.0, 2000.0, 0.0))

    @pytest.mark.parametrize(("min_crop", "padding"), [(64, 0.15), (32, 0.0), (48, 0.4)])
    def test_gating_reproduces_refine_boundaries(self, tmp_path, min_crop, padding):
        """is_refinable(raster_bounds=...) predicts exactly which polygons are prompted."""
        raster = _raster(tmp_path, count=4, dtype="uint8", size=400)
        rng = np.random.default_rng(1)
        geoms = []
        for _ in range(60):
            w, h = rng.uniform(20, 90, 2)
            c, r = rng.uniform(50, 400 - 50 - 90, 2)
            geoms.append(_field(c, r, w, h))
        for _ in range(20):  # near, across or beyond the raster edge
            w, h = rng.uniform(40, 120, 2)
            c, r = rng.uniform(-130, 20, 2) if rng.random() < 0.5 else rng.uniform(300, 420, 2)
            geoms.append(_field(c, r, w, h))
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=UTM)
        cfg = _config(
            tmp_path, source="naip", year=2020, sam_min_crop_px=min_crop, sam_crop_padding=padding
        )
        fake = FakePredictor()
        out = refine_boundaries(gdf, raster, cfg, predictor=fake)
        extent = (X0, Y1 - 400 * RES, X0 + 400 * RES, Y1)
        expected = [
            is_refinable(g.bounds, (RES, RES), min_crop, padding, raster_bounds=extent)
            for g in geoms
        ]
        assert out["agribound:sam_refined"].tolist() == expected
        assert sum(len(b) for b in fake.batches) == sum(expected)
        assert 0 < sum(expected) < len(expected)
        stats = out.attrs["sam_stats"]
        outside = [not se._inside_raster(g.bounds, extent, (RES, RES)) for g in geoms]
        assert 0 < sum(outside) < 20
        assert stats["n_skipped_outside"] == sum(outside)
        assert stats["n_skipped_small"] == len(expected) - sum(expected) - sum(outside)
        assert stats["min_crop_px"] == min_crop and stats["padding"] == padding


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------


class TestPlanWindows:
    def test_every_padded_box_inside_its_window(self):
        rng = np.random.default_rng(3)
        width, height, window = 3000, 2200, 1024
        w = rng.uniform(50, 2600, 300)
        h = rng.uniform(50, 2100, 300)
        c0 = rng.uniform(-0.5, width - w + 0.5)  # inside up to the half-pixel tolerance
        r0 = rng.uniform(-0.5, height - h + 0.5)
        bounds_px = np.column_stack([c0, r0, c0 + w, r0 + h])
        windows = _plan_windows(bounds_px, np.ones(300, bool), 0.15, width, height, window)
        seen = set()
        n_decimated = 0
        for key, entry in windows.items():
            col, row, ww, wh = entry["window"]
            out_h, out_w = entry["out_shape"]
            assert col >= 0 and row >= 0 and col + ww <= width and row + wh <= height
            if key[0] == "grid":
                assert (ww, wh) == (window, window)  # shifted inside, full size
                assert (out_h, out_w) == (wh, ww)
            else:
                assert ww == wh or ww == width or wh == height  # square unless clipped
                if max(ww, wh) > 2048:  # max(2 * window, 2048)
                    n_decimated += 1
                    assert max(out_h, out_w) == 2048
                    assert out_w / out_h == pytest.approx(ww / wh, rel=0.01)
                else:
                    assert (out_h, out_w) == (wh, ww)
            sx, sy = ww / out_w, wh / out_h
            for pos, (bx0, by0, bx1, by1), (cx0, cy0, cx1, cy1) in entry["items"]:
                seen.add(pos)
                cc0, rr0, cc1, rr1 = bounds_px[pos]
                pc, pr = (cc1 - cc0) * 0.15, (rr1 - rr0) * 0.15
                assert col <= max(cc0 - pc, 0) + 1e-9 and min(cc1 + pc, width) <= col + ww + 1e-9
                assert row <= max(rr0 - pr, 0) + 1e-9 and min(rr1 + pr, height) <= row + wh + 1e-9
                # Prompt box: the bounding box clipped to the raster, in output pixels.
                assert 0 <= bx0 < bx1 <= out_w and 0 <= by0 < by1 <= out_h
                assert bx0 == pytest.approx((max(cc0, 0) - col) / sx)
                assert by1 == pytest.approx((min(rr1, height) - row) / sy)
                # Mask clip: integer padded box containing the prompt box.
                assert 0 <= cx0 <= bx0 and bx1 <= cx1 <= out_w
                assert 0 <= cy0 <= by0 and by1 <= cy1 <= out_h
                assert all(isinstance(v, int) for v in (cx0, cy0, cx1, cy1))
        assert seen == set(range(300))
        assert n_decimated > 0

    def test_nearby_small_fields_share_one_window(self):
        bounds_px = np.array([[10, 10, 80, 80], [100, 10, 180, 90], [200, 200, 260, 280]], float)
        windows = _plan_windows(bounds_px, np.ones(3, bool), 0.15, 2048, 2048, 1024)
        assert len(windows) == 1
        assert len(next(iter(windows.values()))["items"]) == 3

    def test_decimation_limit_follows_window_px(self):
        bounds_px = np.array([[1000.0, 1000.0, 3900.0, 3900.0]])  # padded side 3770 px
        (entry,) = _plan_windows(bounds_px, np.ones(1, bool), 0.15, 5000, 5000, 1024).values()
        assert entry["window"][2:] == (3770, 3770) and entry["out_shape"] == (2048, 2048)
        (entry,) = _plan_windows(bounds_px, np.ones(1, bool), 0.15, 5000, 5000, 1536).values()
        assert entry["out_shape"] == (3072, 3072)  # 2 * window_px
        (entry,) = _plan_windows(bounds_px, np.ones(1, bool), 0.15, 5000, 5000, 256).values()
        assert entry["out_shape"] == (2048, 2048)  # never below 2048


# ---------------------------------------------------------------------------
# refine_boundaries
# ---------------------------------------------------------------------------


class TestRefineBoundaries:
    def _fields(self):
        return [
            _field(10, 10, 80, 70),  # refinable
            _field(150, 20, 30, 30),  # too small (39 px padded)
            Point(X0 + 1500, Y1 - 1500).buffer(600),  # 120 px circle -> box-like mask
            None,
            _field(5000, 5000, 80, 80),  # outside the raster
        ]

    def test_outputs_columns_stats_and_unchanged_rows(self, tmp_path):
        raster = _raster(tmp_path)
        gdf = gpd.GeoDataFrame({"k": [1, 2, 3, 4, 5]}, geometry=self._fields(), crs=UTM)
        gdf.attrs["engine_meta"] = {"backend": "x"}
        fake = FakePredictor()
        out = refine_boundaries(gdf, raster, _config(tmp_path), predictor=fake)

        assert out["k"].tolist() == [1, 2, 3, 4, 5]
        assert out["agribound:sam_refined"].tolist() == [True, False, True, False, False]
        assert out["agribound:sam_score"].isna().tolist() == [False, True, False, True, True]
        # Skipped / missing / outside rows are untouched.
        assert out.geometry.iloc[1].equals(gdf.geometry.iloc[1])
        assert out.geometry.iloc[3] is None
        assert out.geometry.iloc[4].equals(gdf.geometry.iloc[4])
        # Refined geometry = the fake mask (box inset by 1 px = 10 m).
        minx, miny, maxx, maxy = out.geometry.iloc[0].bounds
        assert (minx, miny, maxx, maxy) == pytest.approx((X0 + 110, Y1 - 790, X0 + 890, Y1 - 110))
        assert out.geometry.iloc[2].geom_type == "Polygon"
        assert not out.geometry.iloc[2].equals(gdf.geometry.iloc[2])

        stats = out.attrs["sam_stats"]
        assert out.attrs["engine_meta"] == {"backend": "x"}
        required = {
            "backend", "model", "n_total", "n_refined", "n_skipped_small", "n_failed",
            "min_crop_px", "padding",
        }  # fmt: skip
        assert required <= set(stats)
        assert (stats["n_total"], stats["n_refined"], stats["n_skipped_small"]) == (5, 2, 1)
        assert (stats["n_skipped_outside"], stats["n_failed"]) == (2, 0)
        assert stats["n_total"] == (
            stats["n_refined"]
            + stats["n_skipped_small"]
            + stats["n_skipped_outside"]
            + stats["n_failed"]
        )
        assert stats["backend"] == "fake" and stats["model"] == "fake/box"
        assert stats["rgb_bands"] == [4, 3, 2]  # Sentinel-2 R, G, B
        assert len(fake.images) == stats["n_windows"] == 1  # one encode for both fields

    def test_crs_roundtrip(self, tmp_path):
        raster = _raster(tmp_path)
        gdf = gpd.GeoDataFrame(geometry=[_field(10, 10, 80, 70)], crs=UTM).to_crs("EPSG:4326")
        out = refine_boundaries(gdf, raster, _config(tmp_path), predictor=FakePredictor())
        assert out.crs == gdf.crs
        back = out.to_crs(UTM).geometry.iloc[0].bounds
        assert back == pytest.approx((X0 + 110, Y1 - 790, X0 + 890, Y1 - 110), abs=0.01)

    def test_batches_respect_batch_size(self, tmp_path):
        raster = _raster(tmp_path, size=400)
        geoms = [_field(10 + 70 * i, 10, 60, 60) for i in range(5)]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=UTM)
        cfg = _config(tmp_path, engine_params={"sam_batch_size": 2})
        fake = FakePredictor()
        out = refine_boundaries(gdf, raster, cfg, predictor=fake)
        assert [len(b) for b in fake.batches] == [2, 2, 1]
        assert len(fake.images) == 1
        assert out.attrs["sam_stats"]["batch_size"] == 2
        # Boxes are passed as continuous window pixel coordinates.
        np.testing.assert_allclose(fake.batches[0][0], [10, 10, 70, 70])

    def test_empty_mask_counts_as_failed_and_keeps_geometry(self, tmp_path):
        raster = _raster(tmp_path)
        geoms = [_field(10, 10, 80, 70), _field(120, 10, 80, 70)]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=UTM)
        out = refine_boundaries(
            gdf, raster, _config(tmp_path), predictor=FakePredictor(empty_for={(120, 10)})
        )
        assert out["agribound:sam_refined"].tolist() == [True, False]
        assert out.geometry.iloc[1].equals(geoms[1])
        assert out.attrs["sam_stats"]["n_failed"] == 1

    def test_window_error_keeps_geometry_and_is_recorded(self, tmp_path):
        raster = _raster(tmp_path, size=1400)
        # Two far-apart fields -> two grid windows; the first encode raises.
        geoms = [_field(10, 10, 80, 70), _field(1250, 1250, 80, 70)]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=UTM)
        out = refine_boundaries(
            gdf, raster, _config(tmp_path), predictor=FakePredictor(raise_on_image={0})
        )
        stats = out.attrs["sam_stats"]
        assert stats["n_windows"] == 2
        assert out["agribound:sam_refined"].tolist() == [False, True]
        assert out.geometry.iloc[0].equals(geoms[0])
        assert stats["n_failed"] == 1 and "boom 0" in stats["errors"][0]

    def test_all_windows_failing_raises(self, tmp_path):
        raster = _raster(tmp_path)
        gdf = gpd.GeoDataFrame(geometry=[_field(10, 10, 80, 70)], crs=UTM)
        with pytest.raises(RuntimeError, match="failed in all 1 windows"):
            refine_boundaries(
                gdf, raster, _config(tmp_path), predictor=FakePredictor(raise_on_image={0})
            )

    def test_nothing_refinable_loads_no_model(self, tmp_path, monkeypatch):
        raster = _raster(tmp_path)
        monkeypatch.setattr(se, "_load_predictor", lambda *a, **k: pytest.fail("model loaded"))
        gdf = gpd.GeoDataFrame(geometry=[_field(10, 10, 20, 20)], crs=UTM)
        out = refine_boundaries(gdf, raster, _config(tmp_path, sam_model="tiny"))
        assert out["agribound:sam_refined"].tolist() == [False]
        stats = out.attrs["sam_stats"]
        assert stats["n_windows"] == 0
        # The configured model and device are recorded although nothing was prompted.
        assert (stats["backend"], stats["model"], stats["device"]) == (
            "sam2",
            "facebook/sam2-hiera-tiny",
            "cpu",
        )

    def test_polygon_beyond_raster_is_not_prompted(self, tmp_path):
        """A polygon right of the raster whose padded box reaches back inside keeps its shape."""
        raster = _raster(tmp_path, count=3, dtype="uint8", size=200)
        geom = _field(202, 10, 100, 90)
        fake = FakePredictor()
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=[geom], crs=UTM),
            raster,
            _config(tmp_path, source="naip", year=2020),
            predictor=fake,
        )
        assert fake.batches == [] and fake.images == []
        assert out["agribound:sam_refined"].tolist() == [False]
        assert out.geometry.iloc[0].equals(geom)
        stats = out.attrs["sam_stats"]
        assert (stats["n_skipped_outside"], stats["n_refined"], stats["n_windows"]) == (1, 0, 0)

    def test_polygon_straddling_the_edge_keeps_geometry(self, tmp_path):
        """SAM would see 1 px of this 100 px field; it must not be replaced by a sliver."""
        raster = _raster(tmp_path, count=3, dtype="uint8", size=200)
        geoms = [_field(199, 50, 100, 100), _field(-60, 20, 100, 100), _field(20, 20, 80, 80)]
        fake = FakePredictor()
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=geoms, crs=UTM),
            raster,
            _config(tmp_path, source="naip", year=2020),
            predictor=fake,
        )
        assert out["agribound:sam_refined"].tolist() == [False, False, True]
        assert out.geometry.iloc[0].equals(geoms[0]) and out.geometry.iloc[1].equals(geoms[1])
        assert out.attrs["sam_stats"]["n_skipped_outside"] == 2
        assert sum(len(b) for b in fake.batches) == 1

    def test_half_pixel_overhang_is_refined_with_clipped_box(self, tmp_path):
        raster = _raster(tmp_path, count=3, dtype="uint8", size=200)
        geom = _field(100, 50, 100.4, 100)  # 0.4 px beyond the right edge
        fake = FakePredictor()
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=[geom], crs=UTM),
            raster,
            _config(tmp_path, source="naip", year=2020),
            predictor=fake,
        )
        assert out["agribound:sam_refined"].tolist() == [True]
        np.testing.assert_allclose(fake.batches[0][0], [100, 50, 200, 150])

    def test_mask_is_confined_to_the_padded_box(self, tmp_path):
        """A mask covering the whole window is cut to the field's padded box, as in 0.1.x."""

        class Flood(FakePredictor):
            def predict_boxes(self, boxes):
                self.batches.append(np.array(boxes))
                return np.ones((len(boxes), *self.shape), dtype=bool), np.ones(len(boxes))

        raster = _raster(tmp_path, count=3, dtype="uint8", size=300)
        geom = _field(100, 100, 80, 70)  # padded: cols 88-192, rows 89.5-180.5
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=[geom], crs=UTM),
            raster,
            _config(tmp_path, source="naip", year=2020),
            predictor=Flood(),
        )
        assert out["agribound:sam_refined"].tolist() == [True]
        assert out.geometry.iloc[0].bounds == pytest.approx(
            (X0 + 880, Y1 - 1810, X0 + 1920, Y1 - 890)  # cols 88-192, rows 89-181
        )
        assert "padded box" in out.attrs["sam_stats"]["mask_selection"]

    def test_oversized_field_window_is_read_decimated(self, tmp_path):
        size = 2100
        raster = _raster(tmp_path, count=3, dtype="uint8", size=size)
        geom = _field(50, 50, 2000, 2000)  # padded box clipped to the raster: 2100 px
        fake = FakePredictor()
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=[geom], crs=UTM),
            raster,
            _config(tmp_path, source="naip", year=2020),
            predictor=fake,
        )
        stats = out.attrs["sam_stats"]
        assert (stats["n_windows"], stats["n_windows_decimated"]) == (1, 1)
        assert stats["max_window_px"] == 2048
        assert fake.images[0].shape == (2048, 2048, 3)
        with rasterio.open(raster) as src:
            expected = src.read(
                [1, 2, 3],
                out_shape=(3, 2048, 2048),
                resampling=rasterio.enums.Resampling.nearest,
            ).transpose(1, 2, 0)
        np.testing.assert_array_equal(fake.images[0], expected)
        scale = size / 2048
        np.testing.assert_allclose(fake.batches[0][0], np.array([50, 50, 2050, 2050]) / scale)
        # The fake mask (box inset by 1 output px) maps back to within ~2 raster px of the box.
        assert out["agribound:sam_refined"].tolist() == [True]
        assert out.geometry.iloc[0].bounds == pytest.approx(geom.bounds, abs=2 * scale * RES)

    def test_non_square_window_is_padded_to_square(self, tmp_path):
        path = tmp_path / "wide.tif"
        data = np.random.default_rng(2).integers(1, 255, (3, 120, 300), dtype=np.uint8)
        with rasterio.open(
            path, "w", driver="GTiff", height=120, width=300, count=3, dtype="uint8",
            crs=UTM, transform=from_origin(X0, Y1, RES, RES),
        ) as dst:  # fmt: skip
            dst.write(data)
        geom = _field(100, 20, 80, 70)
        fake = FakePredictor()
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=[geom], crs=UTM),
            str(path),
            _config(tmp_path, source="local", local_tif_path=str(path)),
            predictor=fake,
        )
        image = fake.images[0]
        assert image.shape == (300, 300, 3)
        np.testing.assert_array_equal(image[:120], data.transpose(1, 2, 0))
        assert not image[120:].any()  # black padding below the raster
        np.testing.assert_allclose(fake.batches[0][0], [100, 20, 180, 90])
        assert out.geometry.iloc[0].bounds == pytest.approx(
            (X0 + 1010, Y1 - 890, X0 + 1790, Y1 - 210)
        )

    def test_empty_frame(self, tmp_path):
        raster = _raster(tmp_path)
        gdf = gpd.GeoDataFrame(geometry=[], crs=UTM)
        out = refine_boundaries(gdf, raster, _config(tmp_path), predictor=FakePredictor())
        assert len(out) == 0 and out.attrs["sam_stats"]["n_total"] == 0

    def test_uint8_passthrough(self, tmp_path):
        raster = _raster(tmp_path, count=4, dtype="uint8")
        with rasterio.open(raster) as src:
            expected = src.read([1, 2, 3]).transpose(1, 2, 0)
        gdf = gpd.GeoDataFrame(geometry=[_field(10, 10, 80, 70)], crs=UTM)
        fake = FakePredictor()
        out = refine_boundaries(
            gdf, raster, _config(tmp_path, source="naip", year=2020), predictor=fake
        )
        np.testing.assert_array_equal(fake.images[0], expected)
        assert out.attrs["sam_stats"]["stretch"]["method"] == "none (uint8)"

    def test_scene_level_stretch_is_consistent_across_windows(self, tmp_path):
        """The same pixel value maps to the same uint8 in every window."""
        data = np.random.default_rng(5).uniform(100, 4000, (12, 1400, 1400)).astype("float32")
        data[:, 50, 50] = 2000.0
        data[:, 1300, 1300] = 2000.0
        raster = _raster(tmp_path, fill=data)
        geoms = [_field(10, 10, 80, 70), _field(1250, 1250, 80, 70)]
        fake = FakePredictor()
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=geoms, crs=UTM), raster, _config(tmp_path), predictor=fake
        )
        stats = out.attrs["sam_stats"]
        assert stats["stretch"]["percentiles"] == [1.0, 99.0]
        (w0, w1) = [e for e in _windows_for(geoms, 1400)]
        v0 = fake.images[0][50 - w0[1], 50 - w0[0]]
        v1 = fake.images[1][1300 - w1[1], 1300 - w1[0]]
        np.testing.assert_array_equal(v0, v1)
        lo, hi = stats["stretch"]["lows"][0], stats["stretch"]["highs"][0]
        assert v0[0] == int(np.clip(255 * (2000.0 - lo) / (hi - lo), 0, 255))

    def test_band_override_and_local_source(self, tmp_path):
        raster = _raster(tmp_path, count=5)
        cfg = _config(
            tmp_path, source="local", local_tif_path=raster, bands={"R": 5, "G": 4, "B": 3}
        )
        gdf = gpd.GeoDataFrame(geometry=[_field(10, 10, 80, 70)], crs=UTM)
        out = refine_boundaries(gdf, raster, cfg, predictor=FakePredictor())
        assert out.attrs["sam_stats"]["rgb_bands"] == [5, 4, 3]

    def test_embedding_raster_requires_explicit_rgb_bands(self, tmp_path):
        raster = _raster(tmp_path, count=64)
        gdf = gpd.GeoDataFrame(geometry=[_field(10, 10, 80, 70)], crs=UTM)
        cfg = _config(tmp_path, source="tessera-embedding", year=2024)
        with pytest.raises(ValueError, match="sam_rgb_bands"):
            refine_boundaries(gdf, raster, cfg, predictor=FakePredictor())
        cfg = _config(
            tmp_path,
            source="tessera-embedding",
            year=2024,
            engine_params={"sam_rgb_bands": [1, 2, 3]},
        )
        out = refine_boundaries(gdf, raster, cfg, predictor=FakePredictor())
        stats = out.attrs["sam_stats"]
        assert stats["rgb_bands"] == [1, 2, 3]
        assert stats["rgb_source"] == "embedding dimensions (pseudo-RGB)"

    def test_bad_rgb_bands(self, tmp_path):
        raster = _raster(tmp_path, count=3)
        gdf = gpd.GeoDataFrame(geometry=[_field(10, 10, 80, 70)], crs=UTM)
        cfg = _config(tmp_path, source="local", local_tif_path=raster)
        with pytest.raises(ValueError, match="out of range"):
            refine_boundaries(gdf, raster, cfg, rgb_bands=[1, 2, 4], predictor=FakePredictor())
        with pytest.raises(ValueError, match="3 band indices"):
            refine_boundaries(gdf, raster, cfg, rgb_bands=[1, 2], predictor=FakePredictor())

    def test_south_up_raster(self, tmp_path):
        """Pixel boxes are right for a south-up grid (positive row step)."""
        path = tmp_path / "south_up.tif"
        y0 = Y1 - 3000.0  # bottom edge; rows increase northwards
        with rasterio.open(
            path, "w", driver="GTiff", height=300, width=300, count=3, dtype="uint8",
            crs=UTM, transform=rasterio.Affine(RES, 0, X0, 0, RES, y0),
        ) as dst:  # fmt: skip
            dst.write(np.random.default_rng(0).integers(1, 255, (3, 300, 300), dtype=np.uint8))
        field = box(X0 + 100, y0 + 200, X0 + 900, y0 + 900)  # 80 x 70 px
        gdf = gpd.GeoDataFrame(geometry=[field], crs=UTM)
        cfg = _config(tmp_path, source="local", local_tif_path=str(path))
        fake = FakePredictor()
        out = refine_boundaries(gdf, str(path), cfg, predictor=fake)
        assert out["agribound:sam_refined"].tolist() == [True]
        np.testing.assert_allclose(fake.batches[0][0], [10, 20, 90, 90])  # cols 10-90, rows 20-90
        # The fake mask is the box inset by 1 px (10 m) on every side.
        assert out.geometry.iloc[0].bounds == pytest.approx(
            (X0 + 110, y0 + 210, X0 + 890, y0 + 890)
        )

    def test_rotated_raster_rejected(self, tmp_path):
        path = tmp_path / "rot.tif"
        transform = rasterio.Affine(10, 1, X0, 0, -10, Y1)
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=3, dtype="uint8",
            crs=UTM, transform=transform,
        ) as dst:  # fmt: skip
            dst.write(np.ones((3, 10, 10), dtype=np.uint8))
        gdf = gpd.GeoDataFrame(geometry=[_field(1, 1, 5, 5)], crs=UTM)
        cfg = _config(tmp_path, source="local", local_tif_path=str(path))
        with pytest.raises(ValueError, match="rotated"):
            refine_boundaries(gdf, str(path), cfg, predictor=FakePredictor())


def _windows_for(geoms, size, window_px=1024, padding=0.15):
    """Window origins (col, row) refine_boundaries would use for *geoms* (test helper)."""
    bounds_px = np.array(
        [
            [
                (g.bounds[0] - X0) / RES,
                (Y1 - g.bounds[3]) / RES,
                (g.bounds[2] - X0) / RES,
                (Y1 - g.bounds[1]) / RES,
            ]
            for g in geoms
        ]
    )
    windows = _plan_windows(bounds_px, np.ones(len(geoms), bool), padding, size, size, window_px)
    return [e["window"] for e in windows.values()]


# ---------------------------------------------------------------------------
# Models, backends, prefetch
# ---------------------------------------------------------------------------


class TestStretchBounds:
    def test_bounds_come_from_full_resolution_data(self, tmp_path, monkeypatch):
        """A decimated read of a raster with overviews must not sample the overviews."""
        from rasterio.enums import Resampling

        import agribound.io.raster as io_raster

        monkeypatch.setattr(io_raster, "MAX_SAMPLE_SIDE", 64)
        data = np.ones((3, 256, 256), dtype=np.float32)
        data[:, ::2, ::2] = 1000.0
        data[:, 1::2, 1::2] = 1000.0
        path = tmp_path / "ovr.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=3,
            dtype="float32",
            crs="EPSG:32611",
            transform=from_origin(500000, 4000000, 10, 10),
        ) as dst:
            dst.write(data)
        with rasterio.open(path, "r+") as dst:
            dst.build_overviews([2, 4], Resampling.average)
        with rasterio.open(path) as src:
            lows, highs, passthrough = se._stretch_bounds(src, [1, 2, 3], embedding=False)
        assert passthrough is False
        assert set(lows) | set(highs) <= {1.0, 1000.0}  # overviews would give 500.5


class TestModelSelection:
    def test_defaults(self):
        assert resolve_sam_model("sam2") == "facebook/sam2-hiera-large"
        assert resolve_sam_model("sam2.1") == "facebook/sam2.1-hiera-large"
        assert resolve_sam_model("sam3") == "facebook/sam3"
        assert resolve_sam_model("sam3-hf") == "facebook/sam3"

    def test_aliases_and_prefix(self):
        assert resolve_sam_model("sam2", "tiny") == "facebook/sam2-hiera-tiny"
        assert resolve_sam_model("sam2", "base_plus") == "facebook/sam2-hiera-base-plus"
        assert resolve_sam_model("sam2.1", "base-plus") == "facebook/sam2.1-hiera-base-plus"
        assert resolve_sam_model("sam2", "sam2-hiera-small") == "facebook/sam2-hiera-small"
        assert resolve_sam_model("sam3", "sam3.1") == "facebook/sam3.1"

    def test_incompatible_models_raise(self):
        with pytest.raises(ValueError, match="cannot load"):
            resolve_sam_model("sam2", "facebook/sam2.1-hiera-large")  # SamGeo2 is SAM 2.0 only
        with pytest.raises(ValueError, match="cannot load"):
            resolve_sam_model("sam3-hf", "facebook/sam3.1")  # no transformers weights
        with pytest.raises(ValueError, match="Unknown SAM backend"):
            resolve_sam_model("sam4")

    def test_legacy_engine_params_model(self, tmp_path):
        cfg = _config(tmp_path, engine_params={"sam_model": "tiny"})
        assert se._configured_model(cfg) == "tiny"
        cfg = _config(tmp_path, sam_model="small", engine_params={"sam_model": "tiny"})
        assert se._configured_model(cfg) == "small"


class TestNormaliseOutput:
    def test_single_box_squeezed_output(self):
        masks = np.zeros((1, 4, 5), dtype=np.float32)  # (C, H, W) for one box
        masks[0, 1, 1] = 1.0
        m, s = _normalise_sam_output(masks, np.array([0.7]), 1)
        assert m.shape == (1, 4, 5) and m.dtype == bool and m[0, 1, 1]
        assert s.tolist() == [pytest.approx(0.7)]

    def test_batched_output(self):
        masks = np.ones((3, 1, 4, 5), dtype=np.float32)
        m, s = _normalise_sam_output(masks, np.array([[0.1], [0.2], [0.3]]), 3)
        assert m.shape == (3, 4, 5) and s.tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_shape_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected SAM mask shape"):
            _normalise_sam_output(np.ones((2, 1, 4, 5)), np.ones((2, 1)), 3)


class TestBackendLoading:
    def test_sam2_uses_samgeo2_predictor(self, monkeypatch):
        samgeo2 = pytest.importorskip("samgeo.samgeo2")
        calls = {}

        class StubSamGeo2:
            def __init__(self, **kwargs):
                calls.update(kwargs)
                self.predictor = "PREDICTOR"

        monkeypatch.setattr(samgeo2, "SamGeo2", StubSamGeo2)
        adapter = se._load_predictor("sam2", "facebook/sam2-hiera-tiny", "cpu")
        assert calls == {
            "model_id": "facebook/sam2-hiera-tiny",
            "device": "cpu",
            "automatic": False,
        }
        assert adapter.backend == "sam2" and adapter._predictor == "PREDICTOR"

    def test_sam21_uses_sam2_predictor_without_postprocessing(self, monkeypatch):
        mod = pytest.importorskip("sam2.sam2_image_predictor")
        calls = {}

        def fake_from_pretrained(model_id, **kwargs):
            calls["model_id"] = model_id
            calls.update(kwargs)
            return "PRED"

        monkeypatch.setattr(
            mod.SAM2ImagePredictor, "from_pretrained", staticmethod(fake_from_pretrained)
        )
        adapter = se._load_predictor("sam2.1", "facebook/sam2.1-hiera-large", "mps")
        assert calls == {
            "model_id": "facebook/sam2.1-hiera-large",
            "device": "mps",
            "mode": "eval",
            "apply_postprocessing": False,
        }
        assert adapter.backend == "sam2.1"

    def test_sam3_requires_cuda_and_triton(self, monkeypatch):
        # Full gate matrix (Windows + triton-windows etc.) lives in test_sam3_platform.py.
        monkeypatch.setattr(sys, "platform", "linux")
        with pytest.raises(RuntimeError, match="needs a CUDA GPU"):
            se._load_predictor("sam3", "facebook/sam3", "cpu")
        monkeypatch.setitem(sys.modules, "triton", None)  # `import triton` -> ImportError
        with pytest.raises(RuntimeError, match="needs triton"):
            se._load_predictor("sam3", "facebook/sam3", "cuda")

    def test_sam3_hf_gated_repo_error_is_actionable(self, monkeypatch):
        transformers = pytest.importorskip("transformers")

        def refuse(*a, **k):
            raise _gated_error()

        monkeypatch.setattr(transformers.Sam3TrackerProcessor, "from_pretrained", refuse)
        with pytest.raises(
            RuntimeError, match="Request access at https://huggingface.co/facebook/sam3"
        ):
            se._load_predictor("sam3-hf", "facebook/sam3", "cpu")

    def test_sam3_hf_gated_error_wrapped_by_transformers(self, monkeypatch):
        """transformers raises OSError(...) from GatedRepoError; it must still be recognised."""
        transformers = pytest.importorskip("transformers")

        def refuse(*a, **k):
            try:
                raise _gated_error()
            except Exception as exc:  # what transformers.utils.hub.cached_file does
                raise OSError("You are trying to access a gated repo.") from exc

        monkeypatch.setattr(transformers.Sam3TrackerProcessor, "from_pretrained", refuse)
        with pytest.raises(RuntimeError, match="Request access at"):
            se._load_predictor("sam3-hf", "facebook/sam3", "cpu")

        def broken(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(transformers.Sam3TrackerProcessor, "from_pretrained", broken)
        with pytest.raises(OSError, match="disk full"):
            se._load_predictor("sam3-hf", "facebook/sam3", "cpu")

    def test_sam3_hf_adapter_shapes(self):
        torch = pytest.importorskip("torch")

        class Proc:
            def __call__(
                self, images=None, original_sizes=None, input_boxes=None, return_tensors=None
            ):
                if images is not None:
                    return {
                        "pixel_values": torch.zeros(1, 3, 8, 8),
                        "original_sizes": torch.tensor([[6, 7]]),
                    }
                return {"input_boxes": torch.tensor(input_boxes, dtype=torch.float32)}

            def post_process_masks(self, masks, sizes):
                b = masks.shape[1]
                return [torch.ones(b, 1, 6, 7, dtype=torch.bool)]

        class Out:
            def __init__(self, b):
                self.pred_masks = torch.zeros(1, b, 1, 4, 4)
                self.iou_scores = torch.full((1, b, 1), 0.8)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))
                self.seen = None

            def get_image_embeddings(self, pv):
                return [torch.zeros(1, 2, 2, 2)]

            def forward(self, image_embeddings=None, input_boxes=None, multimask_output=True):
                assert multimask_output is False
                self.seen = input_boxes
                return Out(input_boxes.shape[1])

        model = Model()
        adapter = se._SAM3HFAdapter(model, Proc(), "facebook/sam3", "cpu")
        adapter.set_image(np.zeros((6, 7, 3), dtype=np.uint8))
        masks, scores = adapter.predict_boxes(np.array([[0, 0, 3, 3], [1, 1, 5, 5]], float))
        assert masks.shape == (2, 6, 7) and masks.dtype == bool
        assert scores.tolist() == pytest.approx([0.8, 0.8])
        assert tuple(model.seen.shape) == (1, 2, 4)


class TestPrefetch:
    def test_sam2_downloads_checkpoint(self, tmp_path, monkeypatch):
        pytest.importorskip("sam2.build_sam")
        import huggingface_hub

        calls = []
        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda repo, fn, **k: calls.append((repo, fn)) or f"/c/{fn}",
        )
        cfg = _config(tmp_path, sam_model="tiny")
        assert se.prefetch(cfg) == ["/c/sam2_hiera_tiny.pt"]
        assert calls == [("facebook/sam2-hiera-tiny", "sam2_hiera_tiny.pt")]
        cfg = _config(tmp_path, sam_backend="sam2.1")
        assert se.prefetch(cfg) == ["/c/sam2.1_hiera_large.pt"]

    def test_sam3_downloads_checkpoint_and_vocab(self, tmp_path, monkeypatch):
        import huggingface_hub

        calls = []
        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda repo, fn, **k: calls.append((repo, fn, k)) or fn,
        )
        cfg = _config(tmp_path, sam_backend="sam3")
        assert se.prefetch(cfg) == ["config.json", "sam3.pt", "bpe_simple_vocab_16e6.txt.gz"]
        assert calls[2] == (
            "giswqs/geospatial",
            "bpe_simple_vocab_16e6.txt.gz",
            {"repo_type": "dataset"},
        )
        cfg = _config(tmp_path, sam_backend="sam3", sam_model="facebook/sam3.1")
        assert se.prefetch(cfg)[1] == "sam3.1_multiplex.pt"

    def test_sam3_hf_snapshot(self, tmp_path, monkeypatch):
        import huggingface_hub

        seen = {}

        def snap(repo, **kwargs):
            seen["repo"], seen["kwargs"] = repo, kwargs
            return "/snap"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", snap)
        assert se.prefetch(_config(tmp_path, sam_backend="sam3-hf")) == ["/snap"]
        assert seen == {
            "repo": "facebook/sam3",
            "kwargs": {"allow_patterns": ["*.json", "*.safetensors"]},
        }

    def test_gated_prefetch_is_actionable(self, tmp_path, monkeypatch):
        import huggingface_hub

        def refuse(*a, **k):
            raise _gated_error()

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
        with pytest.raises(RuntimeError, match="SAM3_CHECKPOINT_PATH"):
            se.prefetch(_config(tmp_path, sam_backend="sam3"))


def _gated_error():
    import httpx
    from huggingface_hub.errors import GatedRepoError

    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
    return GatedRepoError("403 gated", response=response)


def _hf_cached(repo, filename):
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo, filename), str)
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(
    not _hf_cached("facebook/sam2-hiera-tiny", "sam2_hiera_tiny.pt"),
    reason="facebook/sam2-hiera-tiny not in the Hugging Face cache",
)
def test_real_sam2_tiny_on_cpu(tmp_path, monkeypatch):
    """End-to-end with real SAM 2 weights: a bright square on a dark background."""
    pytest.importorskip("samgeo.samgeo2")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    img = np.full((3, 300, 300), 20, dtype=np.uint8)
    img[:, 100:200, 90:210] = 230
    raster = _raster(tmp_path, fill=img)
    rough = _field(85, 95, 130, 110)  # slightly larger than the square
    gdf = gpd.GeoDataFrame(geometry=[rough], crs=UTM)
    cfg = _config(tmp_path, source="naip", year=2020, sam_model="tiny")
    out = refine_boundaries(gdf, raster, cfg)
    assert out["agribound:sam_refined"].tolist() == [True]
    truth = _field(90, 100, 120, 100)
    refined = out.geometry.iloc[0]
    assert refined.intersection(truth).area / refined.union(truth).area > 0.9


# ---------------------------------------------------------------------------
# Overlaps created by refinement (module docstring, step 6)
# ---------------------------------------------------------------------------


class TestRefinementOverlaps:
    def test_trim_keeps_masks_off_other_polygons(self):
        from agribound.engines.samgeo_engine import trim_refinement_overlaps

        a, b = box(0, 0, 10, 10), box(10, 0, 14, 10)  # b: small unrefined neighbour of a
        c, d = box(20, 0, 30, 10), box(34, 0, 44, 10)  # two refined fields, 4 m apart
        e = box(50, 0, 60, 10)
        originals = [a, b, c, d, e]
        refined = {
            0: box(0, 0, 13, 10),  # grows 3 m into b
            2: box(20, 0, 33, 10),  # grows into the gap ...
            3: box(31, 0, 44, 10),  # ... and so does d (higher score): d keeps 31-33
            4: box(10.5, 1, 13.5, 9),  # lies entirely on b: nothing left
        }
        scores = np.array([0.9, np.nan, 0.5, 0.8, 0.7])
        kept, trimmed, removed = trim_refinement_overlaps(originals, refined, scores)
        assert kept[0].equals(a)
        assert kept[3].equals(box(31, 0, 44, 10))
        assert kept[2].equals(box(20, 0, 31, 10))
        assert 4 not in kept
        assert sorted(trimmed) == [0, 2, 4]
        assert removed == pytest.approx(30 + 20 + 24)
        for i, gi in kept.items():  # no refined polygon overlaps any other polygon
            others = [kept.get(j, originals[j]) for j in range(5) if j != i]
            assert all(gi.intersection(o).area == 0 for o in others)

    def test_existing_overlaps_are_kept_not_grown(self):
        from agribound.engines.samgeo_engine import trim_refinement_overlaps

        a, b = box(0, 0, 10, 10), box(8, 0, 20, 10)  # the inputs already overlap by 2 m
        kept, trimmed, _ = trim_refinement_overlaps(
            [a, b], {0: box(0, 0, 12, 10)}, np.array([0.9, np.nan])
        )
        assert kept[0].equals(a) and trimmed == [0]  # keeps its own 8-10, loses 10-12

    def test_refine_boundaries_does_not_cover_a_small_neighbour(self, tmp_path):
        """Regression: a refined mask covered most of an unrefined neighbour (both kept)."""

        class Flood(FakePredictor):
            def predict_boxes(self, boxes):
                self.batches.append(np.array(boxes))
                return np.ones((len(boxes), *self.shape), dtype=bool), np.full(len(boxes), 0.9)

        raster = _raster(tmp_path, count=3, dtype="uint8", size=300)
        big = _field(100, 100, 80, 70)  # padded box: cols 88-192
        small = _field(181, 120, 8, 8)  # below the refinable size, inside the padded box
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=[big, small], crs=UTM),
            raster,
            _config(tmp_path, source="naip", year=2020),
            predictor=Flood(),
        )
        stats = out.attrs["sam_stats"]
        assert out["agribound:sam_refined"].tolist() == [True, False]
        assert out.geometry.iloc[1].equals(small)
        assert out.geometry.iloc[0].intersection(small).area == 0
        assert out.geometry.iloc[0].area > big.area  # it still grew elsewhere
        assert stats["n_overlap_trimmed"] == 1 and 0 < stats["overlap_trimmed_fraction"] < 0.05
        assert stats["n_total"] == (
            stats["n_refined"]
            + stats["n_skipped_small"]
            + stats["n_skipped_outside"]
            + stats["n_failed"]
        )

    def test_mask_entirely_on_a_neighbour_keeps_the_input_geometry(self, tmp_path):
        class OnNeighbour(FakePredictor):
            """Every mask is the neighbour's footprint (cols 200-250, rows 100-170)."""

            def predict_boxes(self, boxes):
                self.batches.append(np.array(boxes))
                masks = np.zeros((len(boxes), *self.shape), dtype=bool)
                ox, oy = self.origin  # window-relative position of the neighbour
                for k in range(len(boxes)):
                    masks[k, 100 - oy : 170 - oy, 200 - ox : 250 - ox] = True
                return masks, np.full(len(boxes), 0.9)

        raster = _raster(tmp_path, count=3, dtype="uint8", size=300)
        field = _field(150, 100, 50, 70)
        neighbour = _field(200, 100, 50, 70)
        predictor = OnNeighbour()
        predictor.origin = (0, 0)
        out = refine_boundaries(
            gpd.GeoDataFrame(geometry=[field, neighbour], crs=UTM),
            raster,
            _config(tmp_path, source="naip", year=2020),
            predictor=predictor,
            window_px=300,
        )
        # The field's mask lies on the neighbour: nothing left, so it keeps its input
        # geometry and counts as failed; the neighbour's own mask is its footprint.
        assert out.geometry.iloc[0].equals(field)
        assert not out["agribound:sam_refined"].iloc[0]
        assert np.isnan(out["agribound:sam_score"].iloc[0])
        assert out.attrs["sam_stats"]["n_failed"] >= 1

    def test_keep_mode_and_invalid_mode(self, tmp_path):
        class Flood(FakePredictor):
            def predict_boxes(self, boxes):
                self.batches.append(np.array(boxes))
                return np.ones((len(boxes), *self.shape), dtype=bool), np.full(len(boxes), 0.9)

        raster = _raster(tmp_path, count=3, dtype="uint8", size=300)
        big, small = _field(100, 100, 80, 70), _field(181, 120, 8, 8)
        gdf = gpd.GeoDataFrame(geometry=[big, small], crs=UTM)
        keep = _config(tmp_path, source="naip", year=2020, engine_params={"sam_overlaps": "keep"})
        out = refine_boundaries(gdf, raster, keep, predictor=Flood())
        assert out.attrs["sam_stats"]["overlaps"] == "keep"
        assert out.attrs["sam_stats"]["n_overlap_trimmed"] == 0
        assert out.geometry.iloc[0].intersection(small).area == pytest.approx(small.area)
        bad = _config(tmp_path, source="naip", year=2020, engine_params={"sam_overlaps": "x"})
        with pytest.raises(ValueError, match="sam_overlaps must be one of"):
            refine_boundaries(gdf, raster, bad, predictor=Flood())


# ---------------------------------------------------------------------------
# Masks that cover too little of their input polygon (module docstring, step 7)
# ---------------------------------------------------------------------------


class Partial(FakePredictor):
    """Masks the left *fraction* of the boxes whose x0 is in *partial_for* (None: all boxes).

    The other boxes get the FakePredictor mask (the box inset by 1 px).
    """

    def __init__(self, fraction=0.2, partial_for=None, **kw):
        super().__init__(**kw)
        self.fraction = fraction
        self.partial_for = partial_for

    def predict_boxes(self, boxes):
        masks, scores = super().predict_boxes(boxes)
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            if self.partial_for is None or round(x0) in self.partial_for:
                masks[k] = False
                x_end = round(x0 + self.fraction * (x1 - x0))
                masks[k, round(y0) : round(y1), round(x0) : x_end] = True
        return masks, scores


def _counts_add_up(stats):
    return stats["n_total"] == (
        stats["n_refined"]
        + stats["n_skipped_small"]
        + stats["n_skipped_outside"]
        + stats["n_failed"]
        + stats["n_low_coverage"]
    )


class TestMinCoverage:
    def _refine(self, tmp_path, geoms, predictor, **engine_params):
        raster = _raster(tmp_path, count=3, dtype="uint8", size=300)
        cfg = _config(tmp_path, source="naip", year=2020, engine_params=engine_params)
        gdf = gpd.GeoDataFrame({"k": list(range(len(geoms)))}, geometry=geoms, crs=UTM)
        return refine_boundaries(gdf, raster, cfg, predictor=predictor)

    @pytest.mark.parametrize("overlaps", ["trim", "keep"])
    def test_mask_covering_a_fifth_is_reverted_at_half(self, tmp_path, caplog, overlaps):
        field = _field(10, 10, 80, 70)
        with caplog.at_level("INFO", logger="agribound.engines.samgeo_engine"):
            out = self._refine(
                tmp_path, [field], Partial(), sam_min_coverage=0.5, sam_overlaps=overlaps
            )
        assert out.geometry.iloc[0].equals(field)
        assert out["agribound:sam_refined"].tolist() == [False]
        assert np.isnan(out["agribound:sam_score"].iloc[0])
        stats = out.attrs["sam_stats"]
        assert (stats["min_coverage"], stats["n_low_coverage"]) == (0.5, 1)
        assert (stats["n_refined"], stats["n_failed"]) == (0, 0)
        assert _counts_add_up(stats)
        assert "1 masks covered less than 50 % of their input polygon" in caplog.text

    def test_mask_covering_a_fifth_is_kept_at_zero(self, tmp_path):
        field = _field(10, 10, 80, 70)
        out = self._refine(tmp_path, [field], Partial(), sam_min_coverage=0)
        refined = out.geometry.iloc[0]
        assert out["agribound:sam_refined"].tolist() == [True]
        assert refined.intersection(field).area / field.area == pytest.approx(0.2)
        assert refined.bounds == pytest.approx((X0 + 100, Y1 - 800, X0 + 260, Y1 - 100))
        stats = out.attrs["sam_stats"]
        assert (stats["min_coverage"], stats["n_low_coverage"], stats["n_refined"]) == (0.0, 0, 1)
        assert _counts_add_up(stats)

    def test_default_threshold_and_counts(self, tmp_path):
        """Only the field whose mask covers a fifth of it keeps its input geometry."""
        low, ok = _field(10, 10, 80, 70), _field(120, 10, 80, 70)
        out = self._refine(tmp_path, [low, ok], Partial(partial_for={10}))
        stats = out.attrs["sam_stats"]
        assert stats["min_coverage"] == se.DEFAULT_MIN_COVERAGE
        assert 0.2 < se.DEFAULT_MIN_COVERAGE < 0.9  # the inset-box mask covers 95 %
        assert out["agribound:sam_refined"].tolist() == [False, True]
        assert out["agribound:sam_score"].isna().tolist() == [True, False]
        assert out.geometry.iloc[0].equals(low)
        assert out.geometry.iloc[1].bounds == pytest.approx(
            (X0 + 1210, Y1 - 790, X0 + 1990, Y1 - 110)  # the inset-box mask
        )
        assert (stats["n_refined"], stats["n_low_coverage"], stats["n_failed"]) == (1, 1, 0)
        assert _counts_add_up(stats)
        assert out["k"].tolist() == [0, 1]

    def test_reverted_mask_claims_no_area(self):
        """A reverted higher-scoring mask does not trim a lower-scoring one."""
        a, b = box(0, 0, 10, 10), box(20, 0, 30, 10)
        refined = {0: box(9, 0, 18, 10), 1: box(15, 0, 30, 10)}  # a's mask covers 10 % of a
        scores = np.array([0.9, 0.5])
        kept, trimmed, removed, low = se._trim_overlaps([a, b], refined, scores, 0.5)
        assert low == [0] and list(kept) == [1]
        assert kept[1].equals(refined[1]) and trimmed == [] and removed == 0.0
        # Without the test a's mask is kept and takes 15-18 from b's mask.
        kept, trimmed, removed, low = se._trim_overlaps([a, b], refined, scores, 0.0)
        assert low == [] and kept[0].equals(refined[0])
        assert kept[1].equals(box(18, 0, 30, 10)) and trimmed == [1]
        assert removed == pytest.approx(30.0)
        # The public function keeps its three return values and tests no coverage.
        kept, trimmed, removed = se.trim_refinement_overlaps([a, b], refined, scores)
        assert set(kept) == {0, 1} and trimmed == [1]

    def test_coverage_is_measured_after_the_trim(self):
        """A trim that keeps only the part outside the input polygon makes the mask low."""
        own, strip = box(0, 0, 10, 10), box(10, 0, 12, 10)
        mask = box(4, 0, 30, 10)  # covers 60 % of own before the trim, 0 % after
        scores = np.array([0.9, np.nan])
        kept, trimmed, removed, low = se._trim_overlaps([own, strip], {0: mask}, scores, 0.5)
        assert low == [0] and kept == {} and trimmed == [] and removed == 0.0
        kept, _, _, low = se._trim_overlaps([own, strip], {0: mask}, scores, 0.0)
        assert low == [] and kept[0].equals(box(12, 0, 30, 10))

    def test_input_without_area_is_never_low(self):
        mask = box(0, 0, 1, 1)
        assert not se._covers_too_little(mask, box(5, 5, 5, 6), 0.5)  # zero area
        assert not se._covers_too_little(mask, None, 0.5)
        assert not se._covers_too_little(mask, box(5, 5, 6, 6), 0.0)
        assert se._covers_too_little(mask, box(5, 5, 6, 6), 0.5)

    @pytest.mark.parametrize("value", [0, 1, 0.25, np.float32(0.3)])
    def test_valid_values_are_recorded(self, tmp_path, value):
        field = _field(10, 10, 80, 70)
        out = self._refine(tmp_path, [field], FakePredictor(), sam_min_coverage=value)
        assert out.attrs["sam_stats"]["min_coverage"] == float(value)

    @pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), "0.5", True, None, [0.5]])
    def test_invalid_values_raise(self, tmp_path, value):
        field = _field(10, 10, 80, 70)
        with pytest.raises(ValueError, match="sam_min_coverage must be"):
            self._refine(tmp_path, [field], FakePredictor(), sam_min_coverage=value)
