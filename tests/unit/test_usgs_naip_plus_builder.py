from __future__ import annotations

import pytest
from shapely.geometry import box

from agribound.composites.base import get_composite_builder
from agribound.config import AgriboundConfig


class TestUSGSNAIPPlusBuilder:
    def test_config_accepts_usgs_source(self, tmp_path):
        cfg = AgriboundConfig(
            source="usgs-naip-plus",
            engine="delineate-anything",
            year=2023,
            study_area=str(tmp_path / "aoi.geojson"),
            output_path=str(tmp_path / "out.gpkg"),
        )
        assert cfg.source == "usgs-naip-plus"
        assert cfg.is_gee_source() is False

    def test_factory_returns_usgs_builder(self):
        builder = get_composite_builder("usgs-naip-plus")
        assert builder.__class__.__name__ == "USGSNAIPPlusCompositeBuilder"

    def test_select_lock_raster_ids_returns_nonempty_list(
        self,
        sample_usgs_query_features,
    ):
        from agribound.clients.usgs_naip_plus import USGSNAIPPlusClient
        from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder

        client = USGSNAIPPlusClient("https://example.com/ImageServer")
        builder = USGSNAIPPlusCompositeBuilder()

        candidates = [
            client._feature_to_candidate(feature)
            for feature in sample_usgs_query_features["features"]
        ]
        aoi = box(-13024380.0, 5265000.0, -13022380.0, 5266000.0)

        selected_ids, ranked = builder._select_lock_raster_ids(
            candidates,
            aoi,
            target_year=2023,
            max_ids=50,
        )

        assert selected_ids
        assert selected_ids == sorted(selected_ids)
        assert len(ranked) == 2


class TestUSGSNAIPPlusBuilderBehaviour:
    def _candidate(self, oid, value, units, year=2022):
        from agribound.clients.usgs_naip_plus import USGSRasterCandidate

        return USGSRasterCandidate(
            object_id=oid,
            year=year,
            state="NM",
            acquisition_date=None,
            resolution_value=value,
            resolution_units=units,
            band_count=4,
            category=1,
            name=None,
            download_url=None,
            geometry=box(0, 0, 10, 10),
            attributes={},
        )

    def test_resolution_uses_units(self):
        from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder

        b = USGSNAIPPlusCompositeBuilder()
        assert b._estimate_resolution_m([self._candidate(1, 0.6, "METER")]) == 0.6
        feet = b._estimate_resolution_m([self._candidate(1, 2.0, "Feet")])
        assert feet == pytest.approx(0.6096)
        # unknown units are ignored; the finest remaining value wins
        mixed = [self._candidate(1, 0.3, "furlongs"), self._candidate(2, 0.6, "Meters")]
        assert b._estimate_resolution_m(mixed) == 0.6
        assert b._estimate_resolution_m([self._candidate(1, None, None)]) == 1.0

    def test_no_imagery_error_lists_years(self, monkeypatch, tmp_path, sample_usgs_aoi_geojson):
        from agribound.clients.usgs_naip_plus import USGSNAIPPlusClient
        from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder

        seen = {}
        monkeypatch.setattr(
            USGSNAIPPlusClient, "get_service_metadata", lambda self: {"maxImageWidth": 4000}
        )
        monkeypatch.setattr(
            USGSNAIPPlusClient, "query_candidates", lambda self, b, where, out_sr=3857: []
        )

        def year_counts(self, bounds, where):
            seen["where"] = where
            return {2019: 3, 2022: 12}

        monkeypatch.setattr(USGSNAIPPlusClient, "query_year_counts", year_counts)
        cfg = AgriboundConfig(
            source="usgs-naip-plus",
            year=2016,
            study_area=sample_usgs_aoi_geojson,
            output_path=str(tmp_path / "o.gpkg"),
            usgs_state="id",
        )
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError, match=r"2019 \(3 items\), 2022 \(12 items\)"):
            USGSNAIPPlusCompositeBuilder().build(cfg)
        assert "Year" not in seen["where"] and "State = 'ID'" in seen["where"]

    def test_final_raster_covers_extent_without_polygon_mask(self, tmp_path, sample_export_tif):
        import geopandas as gpd
        import rasterio
        from shapely.geometry import Polygon, box

        from agribound.composites.gee import compute_export_grid
        from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder

        # triangle inside the 3857 test tile (lon ~ -117.0, lat ~ 42.0)
        tri = Polygon([(-116.999, 42.0005), (-116.990, 42.0005), (-116.999, 42.004)])
        grid = compute_export_grid(tri, "EPSG:32611", 1.0)
        out = tmp_path / "final.tif"
        n = USGSNAIPPlusCompositeBuilder._write_final([sample_export_tif], out, grid)
        assert n == 4
        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 32611 and src.res == (1.0, 1.0)
            assert src.nodata == 0 and src.dtypes[0] == "uint8"
            assert src.descriptions == ("R", "G", "B", "N")
            assert (src.width, src.height) == (grid.width, grid.height)
            data = src.read(1)
            aoi = gpd.GeoSeries([tri], crs=4326).to_crs(32611).iloc[0]
            # NE corner of the bounding box lies outside the triangle but inside the
            # tile: it keeps the tile data (random 0-254, mostly > 0), not 0.
            r, c = src.index(aoi.bounds[2] - 3, aoi.bounds[3] - 3)
            assert (data[max(0, r - 3) : r + 3, max(0, c - 3) : c + 3] > 0).any()

        # A study area reaching west of the tile: pixels no tile covers are 0.
        wide = box(-117.002, 42.001, -116.995, 42.004)
        grid2 = compute_export_grid(wide, "EPSG:32611", 1.0)
        out2 = tmp_path / "final2.tif"
        USGSNAIPPlusCompositeBuilder._write_final([sample_export_tif], out2, grid2)
        with rasterio.open(out2) as src:
            data = src.read()
            west = gpd.GeoSeries([box(-117.0019, 42.002, -117.0012, 42.003)], crs=4326)
            wx, wy = west.to_crs(32611).iloc[0].centroid.coords[0]
            r, c = src.index(wx, wy)
            assert (data[:, r, c] == 0).all()
            east = gpd.GeoSeries([box(-116.9965, 42.002, -116.9960, 42.003)], crs=4326)
            ex, ey = east.to_crs(32611).iloc[0].centroid.coords[0]
            r, c = src.index(ex, ey)
            assert (data[:, max(0, r - 3) : r + 3, max(0, c - 3) : c + 3] > 0).any()


def _usgs_candidate(oid, geometry):
    from agribound.clients.usgs_naip_plus import USGSRasterCandidate

    return USGSRasterCandidate(
        object_id=oid,
        year=2023,
        state="ID",
        acquisition_date=None,
        resolution_value=1.0,
        resolution_units="Meters",
        band_count=4,
        category=1,
        name=None,
        download_url=None,
        geometry=geometry,
        attributes={},
    )


def _write_3857_tile(path, bbox, width, height, covered_east_x):
    """uint8 4-band tile; 0 (no locked raster) east of *covered_east_x*."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    data = np.full((4, height, width), 120, np.uint8)
    xs = bbox[0] + (np.arange(width) + 0.5) * (bbox[2] - bbox[0]) / width
    data[:, :, xs > covered_east_x] = 0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=4,
        width=width,
        height=height,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_bounds(*bbox, width, height),
        nodata=0,
    ) as dst:
        dst.write(data)


class TestUSGSCoverage:
    """Coverage of the selected footprints and of the written raster (WARNINGs, tags)."""

    def test_footprint_coverage(self):
        from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder

        aoi = box(0, 0, 10, 10)
        cands = [_usgs_candidate(1, box(0, 0, 5, 10)), _usgs_candidate(2, box(4, 0, 6, 10))]
        assert USGSNAIPPlusCompositeBuilder._footprint_coverage(cands, aoi) == pytest.approx(0.6)
        assert USGSNAIPPlusCompositeBuilder._footprint_coverage([], aoi) == 0.0

    def _build(self, monkeypatch, tmp_path, aoi_geojson, covered_share):
        """Two half-extent candidates, maxMosaicImageCount=1: only the west one is used."""
        from agribound.clients.usgs_naip_plus import USGSNAIPPlusClient
        from agribound.composites.gee import (
            compute_export_grid,
            grid_footprint_4326,
            study_area_geometry_4326,
        )
        from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder, _to_3857

        cfg = AgriboundConfig(
            source="usgs-naip-plus",
            year=2023,
            study_area=aoi_geojson,
            output_path=str(tmp_path / "o.gpkg"),
            lulc_filter=False,
        )
        grid = compute_export_grid(study_area_geometry_4326(cfg), "EPSG:32611", 1.0)
        x0, y0, x1, y1 = _to_3857(grid_footprint_4326(grid)).bounds
        split = x0 + (x1 - x0) * 0.55
        west = box(x0 - 50, y0 - 50, split, y1 + 50)
        east = box(split, y0 - 50, x1 + 50, y1 + 50)
        covered_east_x = x0 + (x1 - x0) * covered_share
        monkeypatch.setattr(
            USGSNAIPPlusClient,
            "get_service_metadata",
            lambda self: {"maxImageWidth": 4000, "maxImageHeight": 4000, "maxMosaicImageCount": 1},
        )
        monkeypatch.setattr(
            USGSNAIPPlusClient,
            "query_candidates",
            lambda self, b, where, out_sr=3857: [
                _usgs_candidate(7, west),
                _usgs_candidate(8, east),
            ],
        )

        def fake_export(self, *, bbox_3857, width, height, lock_raster_ids, output_path, **kw):
            assert lock_raster_ids == [7]
            _write_3857_tile(output_path, bbox_3857, width, height, covered_east_x)
            return {"href": "mock://tile.tif"}

        monkeypatch.setattr(USGSNAIPPlusClient, "export_image", fake_export)
        builder = USGSNAIPPlusCompositeBuilder()
        return builder, builder.build(cfg)

    def test_partial_coverage_warns_and_is_recorded(
        self, monkeypatch, tmp_path, sample_usgs_aoi_geojson, caplog
    ):
        import json
        from pathlib import Path

        import rasterio

        with caplog.at_level("WARNING"):
            builder, path = self._build(monkeypatch, tmp_path, sample_usgs_aoi_geojson, 0.55)
        warnings_ = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("maxMosaicImageCount (1)" in m for m in warnings_)
        assert any("imagery covers" in m for m in warnings_)
        share = float(builder.last_metadata["AGRIBOUND_FOOTPRINT_COVERAGE"])
        valid = float(builder.last_metadata["AGRIBOUND_VALID_FRACTION"])
        assert 0.5 < share < 0.65
        assert 0.4 < valid < 0.7
        with rasterio.open(path) as src:
            assert float(src.tags()["AGRIBOUND_VALID_FRACTION"]) == pytest.approx(valid)
        manifest = json.loads(Path(path).with_suffix(".json").read_text())
        assert manifest["coverage"]["max_mosaic_image_count_reached"] is True
        assert manifest["coverage"]["valid_fraction_in_study_area"] == pytest.approx(valid)

    def test_no_imagery_in_study_area_raises(self, monkeypatch, tmp_path, sample_usgs_aoi_geojson):
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError, match="no imagery inside the study area"):
            self._build(monkeypatch, tmp_path, sample_usgs_aoi_geojson, 0.0)
        assert not list((tmp_path / ".agribound_cache").glob("usgs_naip_plus_*.tif"))
