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
