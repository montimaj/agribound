"""Tests for post-processing: merge, filter, simplify, smooth, regularize, polygonize."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Point, Polygon, box

from agribound.postprocess.filter import filter_polygons
from agribound.postprocess.merge import merge_polygons
from agribound.postprocess.polygonize import polygonize_mask
from agribound.postprocess.regularize import regularize_polygons
from agribound.postprocess.simplify import (
    make_polygonal,
    simplify_polygons,
    smooth_polygons,
    to_metric_crs,
)

UTM = "EPSG:32611"


def _write_mask(path, data, nodata=None, dtype=None):
    data = np.asarray(data)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype or str(data.dtype),
        crs=UTM,
        transform=from_origin(500000, 4000000, 10, 10),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype or data.dtype), 1)
    return str(path)


def _noisy_rotated_rectangle(seed=0, noise=1.5):
    rng = np.random.default_rng(seed)
    rect = rotate(box(500000, 4000000, 500200, 4000120), 30, origin="centroid")
    corners = np.array(rect.exterior.coords)
    pts = []
    for a, b in zip(corners[:-1], corners[1:], strict=True):
        for t in np.linspace(0, 1, 20, endpoint=False):
            pts.append(tuple(a + t * (b - a) + rng.normal(0, noise, 2)))
    return Polygon(pts)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_to_metric_crs_keeps_utm(self):
        gdf = gpd.GeoDataFrame(geometry=[box(500000, 4000000, 500010, 4000010)], crs=UTM)
        out, original = to_metric_crs(gdf)
        assert out is gdf
        assert original == gdf.crs

    @pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857", "EPSG:6933", "EPSG:2227"])
    def test_to_metric_crs_reprojects_non_utm(self, crs):
        """Geographic, Web Mercator, equal-area and State Plane (feet) all go to UTM."""
        gdf = gpd.GeoDataFrame(geometry=[box(500000, 4000000, 500100, 4000100)], crs=UTM).to_crs(
            crs
        )
        out, original = to_metric_crs(gdf)
        assert out.crs.utm_zone is not None
        assert original.equals(gdf.crs)

    def test_make_polygonal_repairs_bowtie(self):
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        assert not bowtie.is_valid
        fixed = make_polygonal(bowtie)
        assert fixed.is_valid
        assert fixed.geom_type in ("Polygon", "MultiPolygon")
        assert fixed.area == pytest.approx(2.0)  # both triangles are kept

    def test_make_polygonal_none_and_empty(self):
        assert make_polygonal(None) is None
        assert make_polygonal(Polygon()).is_empty


# ---------------------------------------------------------------------------
# Simplify / smooth
# ---------------------------------------------------------------------------


class TestSimplifyPolygons:
    def test_reduces_vertex_count(self):
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        coords = [(500100 + 100 * np.cos(a), 4000100 + 100 * np.sin(a)) for a in angles]
        gdf = gpd.GeoDataFrame(geometry=[Polygon(coords)], crs=UTM)
        result = simplify_polygons(gdf, tolerance=5.0)
        assert len(result.geometry.iloc[0].exterior.coords) < 101
        assert len(result) == 1

    def test_tolerance_is_in_metres_for_geographic_input(self):
        """A 5 m wiggle is removed by a 10 m tolerance but kept by a 1 m one, in EPSG:4326."""
        pts = [(500000, 4000000), (500100, 4000005), (500200, 4000000), (500200, 4000200)]
        gdf = gpd.GeoDataFrame(geometry=[Polygon(pts)], crs=UTM).to_crs("EPSG:4326")
        coarse = simplify_polygons(gdf, tolerance=10.0)
        fine = simplify_polygons(gdf, tolerance=1.0)
        assert coarse.crs == gdf.crs
        assert len(coarse.geometry.iloc[0].exterior.coords) == 4  # middle vertex removed
        assert len(fine.geometry.iloc[0].exterior.coords) == 5

    def test_tolerance_is_in_metres_for_feet_crs(self):
        """In a State Plane (US feet) CRS the tolerance is still metres, not feet."""
        pts = [(500000, 4000000), (500100, 4000002), (500200, 4000000), (500200, 4000200)]
        gdf = gpd.GeoDataFrame(geometry=[Polygon(pts)], crs=UTM).to_crs("EPSG:2227")
        # 2 m deviation: a 3 m tolerance removes it; 3 *feet* (0.91 m) would not.
        out = simplify_polygons(gdf, tolerance=3.0)
        assert len(out.geometry.iloc[0].exterior.coords) == 4

    def test_keeps_attributes(self, sample_geodataframe):
        result = simplify_polygons(sample_geodataframe, tolerance=1.0)
        assert list(result["id"]) == [1, 2, 3, 4]

    def test_empty_geodataframe(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs=UTM)
        assert len(simplify_polygons(gdf, tolerance=2.0)) == 0

    def test_zero_tolerance_returns_input(self):
        gdf = gpd.GeoDataFrame(geometry=[box(500000, 4000000, 500200, 4000200)], crs=UTM)
        assert simplify_polygons(gdf, tolerance=0.0) is gdf

    def test_preserves_crs(self, sample_geodataframe):
        result = simplify_polygons(sample_geodataframe, tolerance=1.0)
        assert result.crs == sample_geodataframe.crs

    @pytest.mark.filterwarnings("error:GeoSeries.notna:UserWarning")
    def test_only_missing_or_empty_geometries(self):
        """No extent to estimate a UTM zone from: must not raise."""
        gdf = gpd.GeoDataFrame({"k": [1, 2]}, geometry=[None, Polygon()], crs="EPSG:4326")
        out = simplify_polygons(gdf, tolerance=2.0)
        assert out["k"].tolist() == [1]  # empty removed, missing kept (as for collapsed rows)
        assert out.geometry.iloc[0] is None

    @pytest.mark.filterwarnings("error:GeoSeries.notna:UserWarning")
    def test_mixed_missing_and_empty_geometries_do_not_warn(self):
        """geopandas warns on GeoSeries.notna() when empty geometries are present."""
        gdf = gpd.GeoDataFrame(
            {"k": [1, 2, 3]}, geometry=[box(0, 0, 100, 100), Polygon(), None], crs=UTM
        )
        out = simplify_polygons(gdf, tolerance=2.0)
        assert out["k"].tolist() == [1, 3]


class TestSmoothPolygons:
    def test_rectangle_corners_shrink_as_documented(self):
        """Corner-only rectangle loses about 16 % of its area after 3 Chaikin iterations."""
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 50)], crs=UTM)
        out = smooth_polygons(gdf, iterations=3)
        ratio = out.geometry.iloc[0].area / 5000.0
        assert 0.83 < ratio < 0.85
        assert len(out.geometry.iloc[0].exterior.coords) == 4 * 2**3 + 1

    @staticmethod
    def _traced(mask):
        """Largest polygon traced by rasterio.features.shapes (10 m pixels)."""
        from rasterio.features import shapes
        from shapely.geometry import shape

        tr = from_origin(0, 1000, 10, 10)
        polys = [
            shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=tr) if v == 1
        ]
        return max(polys, key=lambda g: g.area)

    def test_traced_axis_aligned_rectangle_shrinks_16_percent(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:30, 20:40] = True
        poly = self._traced(mask)
        assert len(poly.exterior.coords) == 5  # shapes() keeps only the corners
        out = smooth_polygons(gpd.GeoDataFrame(geometry=[poly], crs=UTM), iterations=3)
        assert out.geometry.iloc[0].area / poly.area == pytest.approx(0.8359, abs=1e-3)

    def test_traced_oblique_staircase_loses_little_area(self):
        from rasterio.features import rasterize

        rect = rotate(box(200, 200, 800, 600), 30, origin="centroid")
        tr = from_origin(0, 1000, 10, 10)
        mask = rasterize([(rect, 1)], out_shape=(100, 100), transform=tr).astype(bool)
        poly = self._traced(mask)
        out = smooth_polygons(gpd.GeoDataFrame(geometry=[poly], crs=UTM), iterations=3)
        assert abs(out.geometry.iloc[0].area / poly.area - 1) < 0.001

    def test_densification_confines_rounding_to_corners(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:30, 20:40] = True
        poly = self._traced(mask)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs=UTM)
        out = smooth_polygons(gdf, iterations=3, max_segment_length=10.0)
        assert abs(out.geometry.iloc[0].area / poly.area - 1) < 0.002
        with pytest.raises(ValueError, match="max_segment_length"):
            smooth_polygons(gdf, iterations=3, max_segment_length=0)

    def test_zero_iterations_returns_input(self, sample_geodataframe):
        assert smooth_polygons(sample_geodataframe, iterations=0) is sample_geodataframe

    def test_multipolygon_and_attributes(self):
        mp = MultiPolygon([box(0, 0, 10, 10), box(20, 0, 30, 10)])
        gdf = gpd.GeoDataFrame({"k": [7]}, geometry=[mp], crs=UTM)
        out = smooth_polygons(gdf, iterations=2)
        assert out.geometry.iloc[0].geom_type == "MultiPolygon"
        assert out["k"].tolist() == [7]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class TestFilterPolygons:
    def test_removes_exactly_the_small_polygon(self, sample_geodataframe):
        # Areas: 40 000, 40 000, 10 000 and 100 m² (the last is removed).
        result = filter_polygons(sample_geodataframe, min_area_m2=2500.0)
        assert result["id"].tolist() == [1, 2, 3]

    def test_min_area_zero_keeps_all(self, sample_geodataframe):
        assert len(filter_polygons(sample_geodataframe, min_area_m2=0.0)) == 4

    def test_max_area_filter(self, sample_geodataframe):
        result = filter_polygons(sample_geodataframe, min_area_m2=0.0, max_area_m2=500.0)
        assert result["id"].tolist() == [4]

    def test_area_threshold_is_m2_for_geographic_input(self, sample_geodataframe):
        gdf = sample_geodataframe.to_crs("EPSG:4326")
        result = filter_polygons(gdf, min_area_m2=20_000.0)
        assert result["id"].tolist() == [1, 2]
        assert result.crs == gdf.crs

    def test_no_stale_area_columns_added(self, sample_geodataframe):
        result = filter_polygons(sample_geodataframe, min_area_m2=0.0, remove_holes_below_m2=1.0)
        assert "area_m2" not in result.columns
        assert "perimeter_m" not in result.columns

    def test_removes_only_small_holes(self):
        outer = [(500000, 4000000), (500300, 4000000), (500300, 4000300), (500000, 4000300)]
        small_hole = [(500010, 4000010), (500020, 4000010), (500020, 4000020), (500010, 4000020)]
        big_hole = [(500100, 4000100), (500200, 4000100), (500200, 4000200), (500100, 4000200)]
        poly = Polygon(outer, [small_hole, big_hole])  # holes of 100 m² and 10 000 m²
        gdf = gpd.GeoDataFrame(geometry=[poly], crs=UTM).to_crs("EPSG:4326")
        result = filter_polygons(gdf, min_area_m2=0.0, remove_holes_below_m2=1000.0)
        assert len(result.geometry.iloc[0].interiors) == 1

    def test_no_crs_raises(self):
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)])
        with pytest.raises(ValueError, match="CRS"):
            filter_polygons(gdf)

    def test_empty_geodataframe(self):
        gdf = gpd.GeoDataFrame(geometry=[], crs=UTM)
        assert len(filter_polygons(gdf)) == 0

    def test_lulc_mask_uses_polygon_pixels_and_ignores_nodata(self, tmp_path):
        # 20 x 20 px (10 m) raster: left half class 1 (crop), right half class 5.
        lulc = np.full((20, 20), 5, dtype=np.uint8)
        lulc[:, :10] = 1
        lulc[:, 15:] = 255  # nodata, excluded from the fraction
        path = _write_mask(tmp_path / "lulc.tif", lulc, nodata=255)
        x0, y1 = 500000, 4000000
        in_crop = box(x0, y1 - 100, x0 + 80, y1)  # entirely class 1
        straddle = box(x0 + 60, y1 - 100, x0 + 140, y1)  # 4 px crop / 4 px other
        crop_and_nodata = box(x0 + 80, y1 - 100, x0 + 200, y1)  # 2 crop, 5 other, 5 nodata
        gdf = gpd.GeoDataFrame(
            {"k": [1, 2, 3]}, geometry=[in_crop, straddle, crop_and_nodata], crs=UTM
        )
        result = filter_polygons(
            gdf, min_area_m2=0.0, lulc_mask_path=path, lulc_agricultural_classes=[1]
        )
        assert result["k"].tolist() == [1]
        relaxed = filter_polygons(
            gdf,
            min_area_m2=0.0,
            lulc_mask_path=path,
            lulc_agricultural_classes=[1],
            lulc_min_fraction=0.25,
        )
        assert relaxed["k"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMergePolygons:
    def test_merges_overlapping_pair(self):
        gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10), box(1, 1, 11, 11), box(100, 100, 110, 110)], crs=UTM
        )
        result = merge_polygons(gdf)
        assert len(result) == 2
        assert result.geometry.iloc[0].equals(box(0, 0, 10, 10).union(box(1, 1, 11, 11)))
        assert result.geometry.iloc[1].equals(box(100, 100, 110, 110))

    def test_non_range_index(self):
        """Regression: positional ids must be used even with a custom index."""
        gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10), box(1, 1, 11, 11), box(100, 100, 110, 110)],
            crs=UTM,
            index=[10, 20, 30],
        )
        assert len(merge_polygons(gdf)) == 2

    def test_touching_fields_stay_separate(self):
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=UTM)
        assert len(merge_polygons(gdf)) == 2

    def test_low_iou_not_merged_but_containment_is(self):
        big = box(0, 0, 100, 100)
        small_inside = box(10, 10, 20, 20)  # IoU 0.01, containment 1.0
        partial = box(95, 0, 195, 100)  # IoU 0.026, containment 0.05
        gdf = gpd.GeoDataFrame(geometry=[big, small_inside, partial], crs=UTM)
        result = merge_polygons(gdf)
        assert len(result) == 2
        assert result.geometry.iloc[0].equals(big)

    def test_attributes_come_from_largest_member(self):
        gdf = gpd.GeoDataFrame(
            {
                "score": [0.2, 0.9, 0.5],
                "agribound:sam_refined": [False, True, False],
            },
            geometry=[box(0, 0, 10, 10), box(0, 0, 11, 11), box(50, 50, 60, 60)],
            crs=UTM,
        )
        gdf.attrs["engine_meta"] = {"backend": "x"}
        result = merge_polygons(gdf)
        assert result["score"].tolist() == [0.9, 0.5]
        assert result["agribound:sam_refined"].tolist() == [True, False]
        assert result.attrs["engine_meta"] == {"backend": "x"}
        assert result.attrs["merge_stats"]["n_groups_merged"] == 1

    def test_transitive_chain_is_merged(self):
        a, b, c = box(0, 0, 10, 10), box(5, 0, 15, 10), box(10, 0, 20, 10)
        gdf = gpd.GeoDataFrame(geometry=[a, b, c], crs=UTM)  # a~b and b~c, a and c touch
        result = merge_polygons(gdf)
        assert len(result) == 1
        assert result.geometry.iloc[0].area == pytest.approx(200.0)

    def test_return_groups(self):
        gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10), box(100, 0, 110, 10), box(1, 1, 11, 11)], crs=UTM
        )
        result, groups = merge_polygons(gdf, return_groups=True)
        assert groups == [[0, 2], [1]]
        assert len(result) == 2

    def test_invalid_geometry_is_repaired_in_merge(self):
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
        gdf = gpd.GeoDataFrame(geometry=[bowtie, box(0, 0, 10, 10)], crs=UTM)
        result = merge_polygons(gdf)
        assert len(result) == 1
        assert result.geometry.iloc[0].is_valid

    def test_single_and_empty(self):
        one = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=UTM)
        assert len(merge_polygons(one)) == 1
        assert len(merge_polygons(gpd.GeoDataFrame(geometry=[], crs=UTM))) == 0

    def test_preserves_crs(self):
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(100, 100, 110, 110)], crs=UTM)
        assert merge_polygons(gdf).crs == gdf.crs


# ---------------------------------------------------------------------------
# Regularize
# ---------------------------------------------------------------------------


class TestRegularizePolygons:
    def test_none_returns_input(self, sample_geodataframe):
        assert regularize_polygons(sample_geodataframe, method="none") is sample_geodataframe

    def test_unknown_method_raises(self, sample_geodataframe):
        with pytest.raises(ValueError, match="Unknown regularization method"):
            regularize_polygons(sample_geodataframe, method="orthogonalise")

    def test_adaptive_changes_geometry(self):
        """Regression: 0.1.x was a silent no-op; the adaptive method must change geometries."""
        pytest.importorskip("geoai.utils.geometry")
        clean = rotate(box(500400, 4000000, 500600, 4000100), 20, origin="centroid")
        # A near-rectangle with an extra near-collinear vertex on one side.
        coords = list(clean.exterior.coords)
        mid = ((coords[0][0] + coords[1][0]) / 2 + 0.3, (coords[0][1] + coords[1][1]) / 2)
        near_rect = Polygon([coords[0], mid, *coords[1:]])
        gdf = gpd.GeoDataFrame(
            {"k": [1, 2]}, geometry=[_noisy_rotated_rectangle(), near_rect], crs=UTM
        ).to_crs("EPSG:4326")
        out = regularize_polygons(gdf, method="adaptive")
        assert out.crs == gdf.crs
        assert out["k"].tolist() == [1, 2]
        for before, after in zip(gdf.geometry, out.geometry, strict=True):
            assert not before.equals(after)
            assert after.is_valid
        # The noisy outline is simplified: fewer vertices, same shape.
        assert len(out.geometry.iloc[0].exterior.coords) < len(gdf.geometry.iloc[0].exterior.coords)
        g0 = gdf.to_crs(UTM).geometry.iloc[0]
        g1 = out.to_crs(UTM).geometry.iloc[0]
        assert g0.intersection(g1).area / g0.union(g1).area > 0.95

    def test_orthogonal_changes_geometry_and_keeps_rows(self):
        pytest.importorskip("geoai.utils.geometry")
        pytest.importorskip("buildingregulariser")
        noisy = _noisy_rotated_rectangle(seed=1)
        pivot = Point(501000, 4000500).buffer(200, quad_segs=64)
        rng = np.random.default_rng(2)
        pivot = Polygon(
            [
                (x + rng.normal(0, 0.5), y + rng.normal(0, 0.5))
                for x, y in pivot.exterior.coords[:-1]
            ]
        )
        assert pivot.is_valid and noisy.is_valid
        gdf = gpd.GeoDataFrame({"k": [1, 2, 3]}, geometry=[noisy, pivot, None], crs=UTM)
        out = regularize_polygons(gdf, method="orthogonal", simplify_tolerance_m=2.0)
        assert out["k"].tolist() == [1, 2, 3]
        assert out.geometry.iloc[2] is None
        for i in (0, 1):
            before, after = gdf.geometry.iloc[i], out.geometry.iloc[i]
            assert after.is_valid and not before.equals(after)
            assert before.intersection(after).area / before.union(after).area > 0.9

    @pytest.mark.filterwarnings("error:GeoSeries.notna:UserWarning")
    def test_adaptive_with_empty_geometry_keeps_rows(self):
        pytest.importorskip("geoai.utils.geometry")
        gdf = gpd.GeoDataFrame(
            {"k": [1, 2]}, geometry=[_noisy_rotated_rectangle(), Polygon()], crs=UTM
        )
        out = regularize_polygons(gdf, method="adaptive")
        assert out["k"].tolist() == [1, 2]
        assert out.geometry.iloc[1].is_empty and not out.geometry.iloc[0].is_empty

    @pytest.mark.filterwarnings("error:GeoSeries.notna:UserWarning")
    @pytest.mark.parametrize("method", ["adaptive", "orthogonal"])
    def test_only_missing_geometries_returned_unchanged(self, method):
        gdf = gpd.GeoDataFrame({"k": [1, 2]}, geometry=[None, Polygon()], crs=UTM)
        assert regularize_polygons(gdf, method=method) is gdf

    def test_orthogonal_with_renamed_geometry_column(self):
        """buildingregulariser addresses the column 'geometry'; other names must still work."""
        pytest.importorskip("geoai.utils.geometry")
        pytest.importorskip("buildingregulariser")
        noisy = _noisy_rotated_rectangle(seed=3)
        gdf = gpd.GeoDataFrame({"k": [5]}, geometry=[noisy], crs=UTM).rename_geometry("geom")
        out = regularize_polygons(gdf, method="orthogonal", simplify_tolerance_m=2.0)
        assert out.geometry.name == "geom" and list(out.columns) == ["k", "geom"]
        after = out.geometry.iloc[0]
        assert not after.equals(noisy) and after.is_valid
        assert noisy.intersection(after).area / noisy.union(after).area > 0.9

    def test_orthogonal_without_buildingregulariser_raises(self, monkeypatch, sample_geodataframe):
        pytest.importorskip("geoai.utils.geometry")
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: None if name == "buildingregulariser" else real(name, *a, **k),
        )
        with pytest.raises(ImportError, match="buildingregulariser"):
            regularize_polygons(sample_geodataframe, method="orthogonal")

    def test_missing_geoai_raises(self, monkeypatch, sample_geodataframe):
        import sys

        monkeypatch.setitem(sys.modules, "geoai.utils", None)
        with pytest.raises(ImportError, match="geoai"):
            regularize_polygons(sample_geodataframe, method="adaptive")


# ---------------------------------------------------------------------------
# Polygonize
# ---------------------------------------------------------------------------


class TestPolygonizeMask:
    def test_regions_and_class_values(self, tmp_path):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:8, 2:8] = 1  # 36 px = 3600 m²
        mask[10:18, 10:18] = 2  # 64 px = 6400 m²
        mask[0, 19] = 1  # 1 px = 100 m²
        path = _write_mask(tmp_path / "m.tif", mask)
        gdf = polygonize_mask(path, min_area_m2=0)
        assert sorted(gdf.class_value.tolist()) == [1, 1, 2]
        assert sorted(gdf.area.round(3).tolist()) == [100.0, 3600.0, 6400.0]
        assert gdf.crs == UTM
        filtered = polygonize_mask(path, min_area_m2=2500)
        assert sorted(filtered.area.round(3).tolist()) == [3600.0, 6400.0]
        only2 = polygonize_mask(path, min_area_m2=0, field_value=2)
        assert only2.class_value.tolist() == [2]

    def test_connectivity(self, tmp_path):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1, 1] = mask[2, 2] = 1  # diagonal neighbours
        path = _write_mask(tmp_path / "d.tif", mask)
        assert len(polygonize_mask(path, min_area_m2=0, connectivity=4)) == 2
        assert len(polygonize_mask(path, min_area_m2=0, connectivity=8)) == 1
        with pytest.raises(ValueError, match="connectivity"):
            polygonize_mask(path, connectivity=6)

    def test_nodata_value_is_background(self, tmp_path):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :5] = 255  # nodata
        mask[6:, 6:] = 1
        path = _write_mask(tmp_path / "n.tif", mask, nodata=255)
        gdf = polygonize_mask(path, min_area_m2=0)
        assert gdf.class_value.tolist() == [1]

    def test_float_labels_accepted_probabilities_rejected(self, tmp_path):
        labels = np.zeros((10, 10), dtype=np.float32)
        labels[2:6, 2:6] = 3.0
        labels[0, 0] = np.nan
        ok = _write_mask(tmp_path / "f.tif", labels)
        assert polygonize_mask(ok, min_area_m2=0).class_value.tolist() == [3]
        probs = np.full((10, 10), 0.7, dtype=np.float32)
        bad = _write_mask(tmp_path / "p.tif", probs)
        with pytest.raises(ValueError, match="non-integer"):
            polygonize_mask(bad, min_area_m2=0)

    def test_empty_mask(self, tmp_path):
        path = _write_mask(tmp_path / "e.tif", np.zeros((5, 5), dtype=np.uint8))
        gdf = polygonize_mask(path)
        assert len(gdf) == 0
        assert "class_value" in gdf.columns
