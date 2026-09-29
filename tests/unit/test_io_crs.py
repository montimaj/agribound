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


class TestUtmHelpers:
    """UTM zone helpers used for export_crs='utm'."""

    def test_zone_edges(self):
        from agribound.io.crs import utm_zone_for_lon

        assert utm_zone_for_lon(-180.0) == 1
        assert utm_zone_for_lon(-177.0) == 1
        assert utm_zone_for_lon(179.99) == 60
        assert utm_zone_for_lon(180.0) == 1  # antimeridian wraps; never zone 61
        assert utm_zone_for_lon(150.0) == 56

    def test_get_utm_crs_at_antimeridian_is_valid(self):
        crs = get_utm_crs(180.0, 10.0)
        assert 32601 <= crs.to_epsg() <= 32660

    def test_utm_crs_for_shapely_geometry(self):
        from shapely.geometry import box

        from agribound.io.crs import utm_crs_for_geometry

        # Namoi test AOI centroid ~149.8E, 30.3S -> zone 55 south
        assert utm_crs_for_geometry(box(149.7, -30.4, 149.9, -30.2)).to_epsg() == 32755

    def test_utm_crs_for_geodataframe_in_other_crs(self):
        import geopandas as gpd
        from shapely.geometry import box

        from agribound.io.crs import utm_crs_for_geometry

        gdf = gpd.GeoDataFrame(geometry=[box(-117.0, 36.0, -116.9, 36.1)], crs="EPSG:4326")
        assert utm_crs_for_geometry(gdf.to_crs("EPSG:3857")).to_epsg() == 32611

    def test_empty_geometry_raises(self):
        import pytest
        from shapely.geometry import Polygon

        from agribound.io.crs import utm_crs_for_geometry

        with pytest.raises(ValueError, match="empty"):
            utm_crs_for_geometry(Polygon())

    def test_zones_for_bounds(self):
        from agribound.io.crs import utm_zones_for_bounds

        assert utm_zones_for_bounds((147.6, -31.5, 151.03, -29.8)) == [55, 56]
        assert utm_zones_for_bounds((144.0, 0.0, 150.0, 1.0)) == [55]  # east edge on boundary
        assert utm_zones_for_bounds((-120.5, 30.0, -120.4, 31.0)) == [10]
        assert utm_zones_for_bounds((179.0, 0.0, -179.0, 1.0)) == [60, 1]
        assert utm_zones_for_bounds((174.0, 0.0, 180.0, 1.0)) == [60]

    def test_utm_epsg(self):
        import pytest

        from agribound.io.crs import utm_epsg

        assert utm_epsg(55, south=True) == 32755
        assert utm_epsg(11, south=False) == 32611
        with pytest.raises(ValueError):
            utm_epsg(61, south=False)
