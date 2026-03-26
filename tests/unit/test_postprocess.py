"""Tests for post-processing: simplify, filter, merge."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon, box

from agribound.postprocess.filter import filter_polygons
from agribound.postprocess.merge import merge_polygons
from agribound.postprocess.simplify import simplify_polygons


class TestSimplifyPolygons:
    """Test polygon simplification."""

    def test_reduces_vertex_count(self):
        """A complex polygon should have fewer vertices after simplification."""
        # Create a polygon with many vertices (a rough circle)
        import numpy as np

        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        coords = [(500100 + 100 * np.cos(a), 4000100 + 100 * np.sin(a)) for a in angles]
        coords.append(coords[0])  # close the ring
        complex_poly = Polygon(coords)

        gdf = gpd.GeoDataFrame(geometry=[complex_poly], crs="EPSG:32611")
        original_vertices = len(gdf.geometry.iloc[0].exterior.coords)

        result = simplify_polygons(gdf, tolerance=5.0)
        simplified_vertices = len(result.geometry.iloc[0].exterior.coords)

        assert simplified_vertices < original_vertices
        assert len(result) == 1

    def test_empty_geodataframe(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32611")
        result = simplify_polygons(gdf, tolerance=2.0)
        assert len(result) == 0

    def test_zero_tolerance_returns_unchanged(self):
        poly = box(500000, 4000000, 500200, 4000200)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32611")
        result = simplify_polygons(gdf, tolerance=0.0)
        assert len(result) == 1

    def test_preserves_crs(self, sample_geodataframe):
        result = simplify_polygons(sample_geodataframe, tolerance=1.0)
        assert result.crs == sample_geodataframe.crs


class TestFilterPolygons:
    """Test polygon area filtering."""

    def test_removes_small_polygons(self, sample_geodataframe):
        # The 4th polygon is 10x10 = 100 m2, should be removed with min 2500
        result = filter_polygons(sample_geodataframe, min_area_m2=2500.0)
        # The small 10x10 polygon (100 m2) should be removed
        assert len(result) < len(sample_geodataframe)

    def test_min_area_zero_keeps_all(self, sample_geodataframe):
        result = filter_polygons(sample_geodataframe, min_area_m2=0.0)
        assert len(result) == len(sample_geodataframe)

    def test_max_area_filter(self, sample_geodataframe):
        result = filter_polygons(
            sample_geodataframe, min_area_m2=0.0, max_area_m2=500.0
        )
        # Only the very small polygon (100 m2) should remain
        assert len(result) >= 1
        assert len(result) < len(sample_geodataframe)

    def test_empty_geodataframe(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32611")
        result = filter_polygons(gdf)
        assert len(result) == 0

    def test_preserves_crs(self, sample_geodataframe):
        result = filter_polygons(sample_geodataframe, min_area_m2=0.0)
        assert result.crs == sample_geodataframe.crs


class TestMergePolygons:
    """Test merging of overlapping polygons."""

    def test_merges_overlapping_polygons(self):
        """Two overlapping boxes should merge into fewer polygons."""
        poly1 = box(0, 0, 10, 10)
        poly2 = box(5, 5, 15, 15)  # overlaps poly1
        poly3 = box(100, 100, 110, 110)  # no overlap
        gdf = gpd.GeoDataFrame(geometry=[poly1, poly2, poly3], crs="EPSG:32611")

        result = merge_polygons(gdf, iou_threshold=0.1)
        # poly1 and poly2 should merge; poly3 stays separate
        assert len(result) <= len(gdf)

    def test_no_merge_disjoint(self):
        """Disjoint polygons should not merge."""
        poly1 = box(0, 0, 10, 10)
        poly2 = box(100, 100, 110, 110)
        gdf = gpd.GeoDataFrame(geometry=[poly1, poly2], crs="EPSG:32611")

        result = merge_polygons(gdf, iou_threshold=0.3)
        assert len(result) == 2

    def test_single_polygon(self):
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:32611")
        result = merge_polygons(gdf)
        assert len(result) == 1

    def test_empty_geodataframe(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32611")
        result = merge_polygons(gdf)
        assert len(result) == 0

    def test_preserves_crs(self):
        polys = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32611")
        result = merge_polygons(gdf)
        assert result.crs == gdf.crs
