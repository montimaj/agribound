"""Tests for agribound.io.raster read and info utilities."""

from __future__ import annotations

from agribound.io.raster import get_raster_info, read_raster


class TestGetRasterInfo:
    """Test raster metadata inspection."""

    def test_band_count_rgb(self, sample_rgb_tif):
        info = get_raster_info(sample_rgb_tif)
        assert info.count == 3

    def test_band_count_rgbn(self, sample_rgbn_tif):
        info = get_raster_info(sample_rgbn_tif)
        assert info.count == 4

    def test_shape(self, sample_rgb_tif):
        info = get_raster_info(sample_rgb_tif)
        assert info.width == 64
        assert info.height == 64

    def test_crs(self, sample_rgb_tif):
        info = get_raster_info(sample_rgb_tif)
        assert info.crs is not None
        assert info.crs.to_epsg() == 32611

    def test_dtype(self, sample_rgb_tif):
        info = get_raster_info(sample_rgb_tif)
        assert info.dtype == "uint16"

    def test_missing_file(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            get_raster_info(str(tmp_path / "missing.tif"))


class TestReadRaster:
    """Test raster pixel data reading."""

    def test_read_all_bands(self, sample_rgb_tif):
        data, meta = read_raster(sample_rgb_tif)
        assert data.shape == (3, 64, 64)
        assert meta["count"] == 3

    def test_read_single_band(self, sample_rgb_tif):
        data, meta = read_raster(sample_rgb_tif, bands=[1])
        assert data.shape == (1, 64, 64)
        assert meta["count"] == 1

    def test_read_subset_bands(self, sample_rgbn_tif):
        data, meta = read_raster(sample_rgbn_tif, bands=[1, 4])
        assert data.shape == (2, 64, 64)
        assert meta["count"] == 2

    def test_meta_has_crs(self, sample_rgb_tif):
        _, meta = read_raster(sample_rgb_tif)
        assert meta["crs"] is not None


class TestValueScaleConversions:
    """to_unit_reflectance / to_s2_dn / infer_value_scale."""

    def test_reflectance_x10000(self):
        import numpy as np

        from agribound.io.raster import to_s2_dn, to_unit_reflectance

        arr = np.array([[0.0, 5000.0], [10000.0, np.nan]])
        unit = to_unit_reflectance(arr, "sentinel2")
        assert unit.dtype == np.float32
        np.testing.assert_allclose(unit[0], [0.0, 0.5])
        assert unit[1, 0] == 1.0 and np.isnan(unit[1, 1])
        dn = to_s2_dn(arr, "landsat")
        assert dn.dtype == np.float32
        np.testing.assert_allclose(dn[0], [0.0, 5000.0])

    def test_uint8(self):
        import numpy as np

        from agribound.io.raster import to_s2_dn, to_unit_reflectance

        arr = np.array([0, 51, 255], dtype=np.uint8)
        np.testing.assert_allclose(to_unit_reflectance(arr, "naip"), [0.0, 0.2, 1.0], rtol=1e-6)
        np.testing.assert_allclose(
            to_s2_dn(arr, "usgs-naip-plus"), [0.0, 2000.0, 10000.0], rtol=1e-6
        )

    def test_unknown_scale_requires_opt_in(self, caplog):
        import numpy as np
        import pytest

        from agribound.io.raster import to_unit_reflectance

        arr = np.array([100, 2000, 4000], dtype=np.uint16)
        with pytest.raises(ValueError, match="allow_unknown"):
            to_unit_reflectance(arr, "spot")
        with pytest.raises(ValueError, match="allow_unknown"):
            to_unit_reflectance(arr, "local")
        with caplog.at_level("WARNING", logger="agribound.io.raster"):
            out = to_unit_reflectance(arr, "local", allow_unknown=True)
        assert "inferred" in caplog.text
        np.testing.assert_allclose(out, [0.01, 0.2, 0.4], rtol=1e-6)

    def test_explicit_value_scale_override(self):
        import numpy as np

        from agribound.io.raster import to_unit_reflectance

        out = to_unit_reflectance(np.array([255.0]), "local", value_scale="uint8")
        assert out[0] == 1.0

    def test_embedding_rejected(self):
        import numpy as np
        import pytest

        from agribound.io.raster import to_unit_reflectance

        with pytest.raises(ValueError, match="not optical"):
            to_unit_reflectance(np.zeros(3), "google-embedding")

    def test_infer_value_scale(self):
        import numpy as np
        import pytest

        from agribound.io.raster import infer_value_scale

        assert infer_value_scale(np.array([0.1, 0.9])) == "unit"
        assert infer_value_scale(np.array([0, 200], dtype=np.uint8)) == "uint8"
        assert infer_value_scale(np.array([0.0, 200.0])) == "uint8"
        assert infer_value_scale(np.array([0.0, 3000.0, np.nan])) == "reflectance_x10000"
        with pytest.raises(ValueError):
            infer_value_scale(np.array([np.nan]))


class TestPercentileStretch:
    """percentile_stretch_uint8 mirrors Delineate-Anything's DataAnalyser."""

    def test_uint8_passthrough(self):
        import numpy as np

        from agribound.io.raster import percentile_stretch_uint8

        arr = np.random.default_rng(0).integers(0, 256, (3, 8, 8), dtype=np.uint8)
        out = percentile_stretch_uint8(arr)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, arr)

    def test_matches_da_formula(self):
        import numpy as np

        from agribound.io.raster import percentile_stretch_uint8

        rng = np.random.default_rng(1)
        arr = rng.uniform(1, 5000, (2, 32, 32)).astype(np.float32)
        out, lows, highs = percentile_stretch_uint8(arr, return_bounds=True)
        for i in range(2):
            lo, hi = np.percentile(arr[i][arr[i] > 0].astype(np.float64), (1, 99))
            assert lows[i] == lo and highs[i] == hi
            expected = np.clip(255 * ((arr[i].astype(np.float64) - lo) / (hi - lo)), 0, 255)
            np.testing.assert_array_equal(out[i], expected.astype(np.uint8))

    def test_nodata_nan_and_nonpositive_excluded(self):
        import numpy as np

        from agribound.io.raster import percentile_stretch_uint8

        arr = np.full((1, 10, 10), 100.0, dtype=np.float32)
        arr[0, :, 5:] = 200.0
        arr[0, 0, 0] = np.nan
        arr[0, 1, 0] = -9999.0
        arr[0, 2, 0] = 0.0
        out, lows, highs = percentile_stretch_uint8(
            arr, nodata=-9999.0, low=0, high=100, return_bounds=True
        )
        assert lows == [100.0] and highs == [200.0]
        assert out[0, 0, 0] == 0 and out[0, 1, 0] == 0
        assert out[0, 5, 9] == 255

    def test_2d_input_and_pooled(self):
        import numpy as np

        from agribound.io.raster import percentile_stretch_uint8

        arr = np.linspace(1, 100, 100, dtype=np.float32).reshape(10, 10)
        out = percentile_stretch_uint8(arr)
        assert out.shape == (10, 10) and out.dtype == np.uint8
        stack = np.stack([arr, arr * 2])
        _, lows, highs = percentile_stretch_uint8(stack, per_band=False, return_bounds=True)
        assert lows[0] == lows[1] and highs[0] == highs[1]

    def test_sampling_grid_limits_size(self):
        import numpy as np

        from agribound.io.raster import _sample_grid

        idx = _sample_grid(10000, 4096)
        assert len(idx) <= 4096 and idx.max() < 10000 and idx.min() >= 0
        np.testing.assert_array_equal(_sample_grid(100, 4096), np.arange(100))

    def test_bounds_only_equals_full_call(self):
        import numpy as np

        from agribound.io.raster import percentile_stretch_uint8

        rng = np.random.default_rng(3)
        arr = rng.normal(1500, 600, (3, 90, 70)).astype(np.float32)
        arr[:, :4] = np.nan
        arr[1, 10:14, 10:14] = -3.0
        arr[:, 20:25, :] = -9999.0
        for per_band in (True, False):
            for side in (4096, 16):  # whole array, then a decimated grid
                _, lows, highs = percentile_stretch_uint8(
                    arr,
                    nodata=-9999.0,
                    per_band=per_band,
                    return_bounds=True,
                    max_sample_side=side,
                )
                assert percentile_stretch_uint8(
                    arr,
                    nodata=-9999.0,
                    per_band=per_band,
                    return_bounds_only=True,
                    max_sample_side=side,
                ) == (lows, highs)
        u8 = np.zeros((2, 5, 5), dtype=np.uint8)
        assert percentile_stretch_uint8(u8, return_bounds_only=True) == ([0.0] * 2, [255.0] * 2)

    def test_bounds_only_raises_for_empty_band(self):
        import numpy as np
        import pytest

        from agribound.io.raster import percentile_stretch_uint8

        arr = np.ones((2, 8, 8), dtype=np.float32)
        arr[1] = -1.0
        with pytest.raises(ValueError, match="band 2"):
            percentile_stretch_uint8(arr, return_bounds_only=True)

    def test_read_stretch_sample_ignores_overviews(self, tmp_path):
        """Averaged overviews would narrow the percentile range: read full-resolution data."""
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_origin

        from agribound.io.raster import read_stretch_sample

        data = np.ones((2, 256, 256), dtype=np.float32)
        data[:, ::2, ::2] = 1000.0
        data[:, 1::2, 1::2] = 1000.0  # checkerboard of 1 and 1000
        path = tmp_path / "ovr.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=2,
            dtype="float32",
            crs="EPSG:32611",
            transform=from_origin(500000, 4000000, 10, 10),
        ) as dst:
            dst.write(data)
        with rasterio.open(path, "r+") as dst:
            dst.build_overviews([2, 4], Resampling.average)
        with rasterio.open(path) as src:
            averaged = src.read(1, out_shape=(64, 64), resampling=Resampling.nearest)
            assert np.allclose(averaged, 500.5)  # what a plain decimated read returns
            sample = read_stretch_sample(src, [1, 2], max_sample_side=64)
            assert sample.shape == (2, 64, 64)
            assert set(np.unique(sample)) <= {1.0, 1000.0}
            full = read_stretch_sample(src, [2], max_sample_side=4096)
            np.testing.assert_array_equal(full[0], data[1])

    def test_no_valid_pixels_raises(self):
        import numpy as np
        import pytest

        from agribound.io.raster import percentile_stretch_uint8

        with pytest.raises(ValueError, match="No valid positive pixels"):
            percentile_stretch_uint8(np.zeros((1, 4, 4), dtype=np.float32))


class TestClipRaster:
    def test_float_raster_outside_is_nan(self, tmp_path):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from shapely.geometry import box

        from agribound.io.raster import clip_raster_to_geometry

        src = tmp_path / "f.tif"
        with rasterio.open(
            src,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="float32",
            crs="EPSG:32611",
            transform=from_bounds(0, 0, 100, 100, 10, 10),
        ) as dst:
            dst.write(np.ones((1, 10, 10), dtype=np.float32))
        tri = box(0, 0, 100, 100).difference(box(0, 0, 50, 50))
        out = clip_raster_to_geometry(src, tmp_path / "c.tif", tri, crs="EPSG:32611")
        with rasterio.open(out) as ds:
            data = ds.read(1)
            assert np.isnan(ds.nodata)
        assert np.isnan(data[-1, 0]) and data[0, 0] == 1.0


class TestRasterFixtures:
    def test_s2_fixture_round_trip(self, s2_reflectance_tif):
        import numpy as np

        from agribound.engines.base import get_canonical_band_indices
        from agribound.io.raster import to_unit_reflectance

        idx = get_canonical_band_indices("sentinel2", ["R", "G", "B", "NIR"])
        data, meta = read_raster(s2_reflectance_tif, bands=idx)
        assert data.dtype == np.float32 and np.isnan(meta["nodata"])
        unit = to_unit_reflectance(data, "sentinel2")
        assert np.isnan(unit[:, 0, 0]).all()
        assert np.nanmax(unit) <= 0.4 + 1e-6

    def test_naip_fixture_stretch_identity(self, naip_uint8_tif):
        import numpy as np

        from agribound.io.raster import percentile_stretch_uint8

        data, _ = read_raster(naip_uint8_tif, bands=[1, 2, 3])
        np.testing.assert_array_equal(percentile_stretch_uint8(data), data)
