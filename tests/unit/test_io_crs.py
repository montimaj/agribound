"""Tests for agribound.io.crs CRS utilities."""

from __future__ import annotations

import pyproj

from agribound.io.crs import get_equal_area_crs, get_utm_crs


class TestGetUtmCrs:
    """Test UTM CRS determination from lon/lat."""

    def test_northern_hemisphere(self):
        # Los Angeles area: lon=-118.25, lat=34.05 -> UTM zone 11N
        crs = get_utm_crs(-118.25, 34.05)
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 32611

    def test_southern_hemisphere(self):
        # Sydney area: lon=151.2, lat=-33.87 -> UTM zone 56S
        crs = get_utm_crs(151.2, -33.87)
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 32756

    def test_greenwich_meridian_north(self):
        # London area: lon=-0.12, lat=51.5 -> UTM zone 30N
        crs = get_utm_crs(-0.12, 51.5)
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 32630

    def test_equator(self):
        # Equator at lon=37 (Kenya) -> UTM zone 37N (lat >= 0)
        crs = get_utm_crs(37.0, 0.0)
        assert isinstance(crs, pyproj.CRS)
        epsg = crs.to_epsg()
        # Zone 37 north: 32637
        assert epsg == 32637

    def test_negative_longitude(self):
        # Brazil: lon=-47.9, lat=-15.8 -> UTM zone 23S
        crs = get_utm_crs(-47.9, -15.8)
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 32723


class TestGetEqualAreaCrs:
    """Test equal-area CRS."""

    def test_returns_epsg_6933(self):
        crs = get_equal_area_crs()
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 6933

    def test_is_projected(self):
        crs = get_equal_area_crs()
        assert crs.is_projected
