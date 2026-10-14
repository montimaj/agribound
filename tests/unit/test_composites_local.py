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


class TestCropSequenceMargin:
    """For engine crop-sequence the crop keeps 10 px around the study area's bounding box."""

    ZONES = {
        "engine": "crop-sequence",
        "engine_params": {"value_scale": "categorical", "dataset": "cdl"},
    }

    def test_margin_window_key_and_tags(self, tmp_path):
        from agribound._cache import cache_path

        tif = _tif(tmp_path / "a.tif")  # 64 x 64 px of 10 m
        aoi = _aoi(tmp_path, "inner", -116.99833, 36.14653, -116.99500, 36.14877)
        plain_cfg = _config(tmp_path, tif, aoi)
        builder = LocalCompositeBuilder()
        plain = builder.build(plain_cfg)
        assert "AGRIBOUND_LOCAL_CROP_MARGIN_PX" not in builder.last_metadata
        # Other engines keep the crop and the cache key they had (no margin part).
        assert plain == str(
            cache_path(plain_cfg, "local_crop_a", ".tif", LocalCompositeBuilder.CROP_VERSION)
        )
        zones_cfg = _config(tmp_path, tif, aoi, **self.ZONES)
        assert LocalCompositeBuilder.crop_margin_px(zones_cfg) == 10
        assert LocalCompositeBuilder.crop_margin_px(plain_cfg) == 0
        padded = builder.build(zones_cfg)
        assert padded != plain
        assert builder.last_metadata["AGRIBOUND_LOCAL_CROP_MARGIN_PX"] == 10
        with rasterio.open(plain) as p, rasterio.open(padded) as q:
            assert (q.width, q.height) == (p.width + 20, p.height + 20)
            assert q.transform.c == pytest.approx(p.transform.c - 10 * p.res[0])
            assert q.transform.f == pytest.approx(p.transform.f + 10 * p.res[1])
            np.testing.assert_array_equal(q.read()[:, 10:-10, 10:-10], p.read())
            assert q.tags()["AGRIBOUND_LOCAL_CROP_MARGIN_PX"] == "10"
            assert "AGRIBOUND_LOCAL_CROP_MARGIN_PX" not in p.tags()

    def test_margin_stops_at_the_raster_edge(self, tmp_path):
        tif = _tif(tmp_path / "a.tif")
        # 3 px from the west edge: the margin is cut there, and kept on the other sides.
        west = _aoi(tmp_path, "west", -116.99967, 36.14653, -116.99833, 36.14877)
        builder = LocalCompositeBuilder()
        plain = builder.build(_config(tmp_path, tif, west))
        padded = builder.build(_config(tmp_path, tif, west, **self.ZONES))
        with rasterio.open(tif) as full, rasterio.open(plain) as p, rasterio.open(padded) as q:
            assert q.transform.c == full.transform.c  # column 0
            assert p.transform.c > full.transform.c
            left = round((p.transform.c - full.transform.c) / full.res[0])
            assert left < 10 and q.width == p.width + left + 10
            assert q.height == p.height + 20
        # A study area whose box plus the margin covers the raster: the file itself.
        most = _aoi(tmp_path, "most", -116.99940, 36.14560, -116.99350, 36.14960)
        assert builder.build(_config(tmp_path, tif, most)) != tif
        assert builder.build(_config(tmp_path, tif, most, **self.ZONES)) == tif


class TestCropColourInterpretation:
    """The crop keeps the source's colour interpretation (GDAL's default is RGBA for 4 uint8 bands).

    Band 4 tagged as alpha is transparency to GDAL's warper: read through a ``WarpedVRT``
    (as DINOv3's processing copy reads the crop), R, G and B became 0 wherever band 4 was 0.
    """

    @staticmethod
    def _rgbn(path, **creation):
        data = np.full((4, 64, 64), 120, np.uint8)
        data[1] = 90
        data[2] = 60
        data[3, 25:35, 25:35] = 0  # NIR 0 (dark water) under valid R, G, B
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            count=4,
            height=64,
            width=64,
            dtype="uint8",
            crs="EPSG:32611",
            transform=from_bounds(500000, 4000000, 500640, 4000640, 64, 64),
            nodata=None,
            **creation,
        ) as dst:
            dst.write(data)
        return str(path)

    @pytest.mark.parametrize(
        ("creation", "expected"),
        [
            ({"ALPHA": "UNSPECIFIED"}, ["red", "green", "blue", "undefined"]),  # R, G, B, NIR
            ({"photometric": "MINISBLACK"}, ["gray", "undefined", "undefined", "undefined"]),
            ({}, ["red", "green", "blue", "alpha"]),  # a source tagged RGBA stays RGBA
        ],
    )
    def test_crop_keeps_the_tags_and_the_pixels(self, tmp_path, creation, expected):
        from rasterio.vrt import WarpedVRT

        tif = self._rgbn(tmp_path / "rgbn.tif", **creation)
        aoi = _aoi(tmp_path, "inner", -116.99833, 36.14653, -116.99500, 36.14877)
        crop = LocalCompositeBuilder().build(_config(tmp_path, tif, aoi))
        assert crop != tif
        with rasterio.open(crop) as src:
            assert [c.name for c in src.colorinterp] == expected
            data = src.read()
            nir0 = data[3] == 0
            assert nir0.sum() == 100  # the whole NIR-0 block is in the crop
            if expected[3] != "alpha":
                with WarpedVRT(src) as vrt:  # GDAL's warper, as DINOv3's grid copy
                    warped = vrt.read()
                np.testing.assert_array_equal(warped[:3, nir0], data[:3, nir0])
                assert (warped[:3, nir0] > 0).all()

    def test_palette_band_keeps_its_colour_table(self, tmp_path):
        path = tmp_path / "classes.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            count=1,
            height=64,
            width=64,
            dtype="uint8",
            crs="EPSG:32611",
            transform=from_bounds(500000, 4000000, 500640, 4000640, 64, 64),
            nodata=0,
        ) as dst:
            dst.write(np.ones((1, 64, 64), np.uint8))
            dst.write_colormap(1, {0: (0, 0, 0, 0), 1: (255, 211, 0, 255)})
        aoi = _aoi(tmp_path, "inner", -116.99833, 36.14653, -116.99500, 36.14877)
        crop = LocalCompositeBuilder().build(_config(tmp_path, str(path), aoi))
        with rasterio.open(crop) as src:
            assert [c.name for c in src.colorinterp] == ["palette"]
            assert src.colormap(1)[1] == (255, 211, 0, 255)
