"""Tests for the local GeoTIFF builder (no network)."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from agribound.composites.base import get_composite_builder
from agribound.composites.local import LocalCompositeBuilder, validate_local_raster
from agribound.config import AgriboundConfig


def _tif(path, count=3, crs="EPSG:32611", nodata=0, dtype="uint16"):
    data = np.arange(count * 64 * 64, dtype=dtype).reshape(count, 64, 64) % 5000 + 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=count,
        height=64,
        width=64,
        dtype=dtype,
        crs=crs,
        transform=from_bounds(500000, 4000000, 500640, 4000640, 64, 64),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


def _aoi(tmp_path, name, lon0, lat0, lon1, lat1):
    path = tmp_path / f"{name}.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [lon0, lat0],
                                    [lon1, lat0],
                                    [lon1, lat1],
                                    [lon0, lat1],
                                    [lon0, lat0],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )
    return str(path)


def _config(tmp_path, tif, study_area="", **kwargs):
    return AgriboundConfig(
        source="local",
        local_tif_path=tif,
        study_area=study_area,
        output_path=str(tmp_path / "out.gpkg"),
        lulc_filter=False,
        **kwargs,
    )


class TestValidation:
    def test_ok(self, tmp_path):
        info = validate_local_raster(_tif(tmp_path / "a.tif"))
        assert info["count"] == 3 and info["crs"] == "EPSG:32611"

    def test_missing_crs_raises(self, tmp_path):
        path = tmp_path / "nocrs.tif"
        with rasterio.open(
            path, "w", driver="GTiff", count=3, height=8, width=8, dtype="uint8"
        ) as dst:
            dst.write(np.ones((3, 8, 8), np.uint8))
        with pytest.raises(ValueError, match="no CRS"):
            validate_local_raster(path)

    def test_band_mapping_beyond_count_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"'NIR': 4.*must be in 1\.\.3"):
            validate_local_raster(_tif(tmp_path / "a.tif"), {"R": 1, "NIR": 4})

    @pytest.mark.parametrize("index", [0, -1])
    def test_band_mapping_below_one_raises(self, tmp_path, index):
        # indices are 1-based: 0 or negative values must fail at validation
        with pytest.raises(ValueError, match="1-based"):
            validate_local_raster(_tif(tmp_path / "a.tif"), {"R": index, "G": 2, "B": 3})

    def test_no_nodata_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            validate_local_raster(_tif(tmp_path / "a.tif", nodata=None))
        assert any("no nodata value" in r.message for r in caplog.records)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_local_raster(tmp_path / "missing.tif")


class TestBuild:
    def test_without_study_area_returns_source(self, tmp_path):
        tif = _tif(tmp_path / "a.tif")
        builder = get_composite_builder("local")
        assert builder.build(_config(tmp_path, tif)) == tif
        assert builder.last_metadata["AGRIBOUND_VALUE_SCALE"] == "unknown"

    def test_crop_to_study_area_is_cached_and_keyed(self, tmp_path):
        tif = _tif(tmp_path / "a.tif")
        # UTM 11N (500000, 4000000) is lon -117.0, lat ~36.14
        aoi_a = _aoi(tmp_path, "a", -117.0, 36.1437, -116.998, 36.1455)
        aoi_b = _aoi(tmp_path, "b", -116.998, 36.1437, -116.996, 36.1455)
        builder = LocalCompositeBuilder()
        pa = builder.build(_config(tmp_path, tif, aoi_a))
        pb = builder.build(_config(tmp_path, tif, aoi_b))
        assert pa != pb and pa != tif
        with rasterio.open(pa) as src:
            assert src.width < 64 and src.height < 64
            assert src.tags()["AGRIBOUND_VALUE_SCALE"] == "unknown"
        # cached: the second call returns the same file without rewriting it
        mtime = (tmp_path / pa).stat().st_mtime_ns
        assert builder.build(_config(tmp_path, tif, aoi_a)) == pa
        assert (tmp_path / pa).stat().st_mtime_ns == mtime

    def test_non_overlapping_study_area_raises(self, tmp_path):
        tif = _tif(tmp_path / "a.tif")
        far = _aoi(tmp_path, "far", 10.0, 10.0, 10.01, 10.01)
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError, match="Could not crop") as info:
            LocalCompositeBuilder().build(_config(tmp_path, tif, far))
        assert isinstance(info.value.__cause__, NoDataError)  # crop_to_extent's own error
        assert not list((tmp_path / ".agribound_cache").glob("local_crop_*"))

    def test_crop_keeps_values_and_does_not_mask_polygons(self, tmp_path):
        # Triangle study area over the raster: the output is the triangle's bounding
        # box, with the source pixel values unchanged (including outside the triangle).
        tif = _tif(tmp_path / "a.tif")
        wkt = (
            "POLYGON ((-116.9995 36.1440, -116.9975 36.1440, -116.9995 36.1455, -116.9995 36.1440))"
        )
        path = LocalCompositeBuilder().build(_config(tmp_path, tif, wkt))
        with rasterio.open(tif) as full, rasterio.open(path) as crop:
            assert crop.crs == full.crs and crop.res == full.res
            assert crop.dtypes == full.dtypes and crop.nodata == full.nodata
            col_off = round((crop.transform.c - full.transform.c) / full.res[0])
            row_off = round((full.transform.f - crop.transform.f) / full.res[1])
            expected = full.read(
                window=rasterio.windows.Window(col_off, row_off, crop.width, crop.height)
            )
            np.testing.assert_array_equal(crop.read(), expected)
            assert (crop.read() > 0).all()  # nothing set to nodata

    def test_extent_containing_raster_returns_source(self, tmp_path):
        tif = _tif(tmp_path / "a.tif")
        big = _aoi(tmp_path, "big", -117.1, 36.0, -116.9, 36.3)
        assert LocalCompositeBuilder().build(_config(tmp_path, tif, big)) == tif
        assert not list((tmp_path / ".agribound_cache").glob("local_crop_*"))
