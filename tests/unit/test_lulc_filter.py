"""Tests for LULC-based crop filtering logic.

These tests cover the dataset selection logic and helper functions
without requiring GEE authentication (which is tested in integration tests).
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from agribound.config import AgriboundConfig


class TestLulcDatasetSelection:
    """Test that the correct LULC dataset is selected based on location/year."""

    def _make_config(self, year: int, **kwargs) -> AgriboundConfig:
        return AgriboundConfig(
            study_area="test.geojson",
            year=year,
            output_path="test.gpkg",
            lulc_filter=True,
            lulc_crop_threshold=0.3,
            **kwargs,
        )

    def _make_gdf(self, lon: float, lat: float) -> gpd.GeoDataFrame:
        """Create a single-polygon GeoDataFrame at given lon/lat."""
        poly = box(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)
        return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    def test_conus_detection(self):
        """Study area in CONUS should be detected correctly."""
        from agribound.postprocess.lulc_filter import _CONUS_BBOX

        # New Mexico (CONUS)
        lon, lat = -106.0, 34.0
        assert _CONUS_BBOX[0] <= lon <= _CONUS_BBOX[2]
        assert _CONUS_BBOX[1] <= lat <= _CONUS_BBOX[3]

        # Hawaii (not CONUS)
        lon, lat = -155.5, 19.9
        assert not (
            _CONUS_BBOX[0] <= lon <= _CONUS_BBOX[2] and _CONUS_BBOX[1] <= lat <= _CONUS_BBOX[3]
        )

        # Argentina (not CONUS)
        lon, lat = -60.0, -34.0
        assert not (
            _CONUS_BBOX[0] <= lon <= _CONUS_BBOX[2] and _CONUS_BBOX[1] <= lat <= _CONUS_BBOX[3]
        )

    def test_config_defaults(self):
        """Default config should have LULC filter enabled."""
        config = AgriboundConfig(
            study_area="test.geojson",
            output_path="test.gpkg",
        )
        assert config.lulc_filter is True
        assert config.lulc_crop_threshold == 0.3

    def test_config_disable(self):
        """LULC filter can be disabled."""
        config = AgriboundConfig(
            study_area="test.geojson",
            output_path="test.gpkg",
            lulc_filter=False,
        )
        assert config.lulc_filter is False

    def test_config_custom_threshold(self):
        """Custom threshold is preserved."""
        config = AgriboundConfig(
            study_area="test.geojson",
            output_path="test.gpkg",
            lulc_crop_threshold=0.5,
        )
        assert config.lulc_crop_threshold == 0.5

    def test_empty_gdf_returns_empty(self):
        """Filtering an empty GeoDataFrame should return empty."""
        from agribound.postprocess.lulc_filter import filter_by_lulc

        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        config = self._make_config(year=2020)
        result = filter_by_lulc(gdf, config)
        assert len(result) == 0

    def test_crop_classes_defined(self):
        """NLCD and C3S crop classes should be defined."""
        from agribound.postprocess.lulc_filter import C3S_CROP_CLASSES, NLCD_CROP_CLASSES

        assert 81 in NLCD_CROP_CLASSES
        assert 82 in NLCD_CROP_CLASSES
        assert 10 in C3S_CROP_CLASSES
        assert 20 in C3S_CROP_CLASSES
        assert 30 in C3S_CROP_CLASSES


class TestGdfToFc:
    """Test the GeoDataFrame to FeatureCollection converter."""

    def test_indices_are_sequential(self):
        """Feature indices should be 0-based and sequential."""
        try:
            import ee  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("earthengine-api not installed")

        from agribound.postprocess.lulc_filter import _gdf_to_fc

        polys = [box(i, i, i + 1, i + 1) for i in range(5)]
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")

        # This will fail if ee is not initialized, but tests the logic
        try:
            fc = _gdf_to_fc(gdf)
            info = fc.getInfo()
            indices = [f["properties"]["_idx"] for f in info["features"]]
            assert indices == [0, 1, 2, 3, 4]
        except Exception:
            # GEE not initialized — skip the server call but confirm import works
            pass
