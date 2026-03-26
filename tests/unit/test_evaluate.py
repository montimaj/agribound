"""Tests for agribound.evaluate accuracy metrics."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from agribound.evaluate import evaluate


class TestEvaluatePerfectMatch:
    """Test evaluation when prediction matches reference exactly."""

    def test_perfect_iou_and_f1(self):
        polys = [
            box(500000, 4000000, 500200, 4000200),
            box(500500, 4000500, 500700, 4000700),
        ]
        predicted = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32611")
        reference = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32611")

        metrics = evaluate(predicted, reference)
        assert metrics["iou_mean"] == pytest.approx(1.0, abs=0.01)
        assert metrics["f1"] == pytest.approx(1.0, abs=0.01)
        assert metrics["precision"] == pytest.approx(1.0, abs=0.01)
        assert metrics["recall"] == pytest.approx(1.0, abs=0.01)
        assert metrics["count_tp"] == 2
        assert metrics["count_fp"] == 0
        assert metrics["count_fn"] == 0


class TestEvaluateEmptyPredictions:
    """Test evaluation with no predictions."""

    def test_zero_metrics(self, sample_reference_gdf):
        empty_pred = gpd.GeoDataFrame(geometry=[], crs="EPSG:32611")
        metrics = evaluate(empty_pred, sample_reference_gdf)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["iou_mean"] == 0.0
        assert metrics["count_predicted"] == 0
        assert metrics["count_fn"] == len(sample_reference_gdf)


class TestEvaluateBothEmpty:
    """Test evaluation when both predicted and reference are empty."""

    def test_perfect_trivial(self):
        empty1 = gpd.GeoDataFrame(geometry=[], crs="EPSG:32611")
        empty2 = gpd.GeoDataFrame(geometry=[], crs="EPSG:32611")
        metrics = evaluate(empty1, empty2)
        assert metrics["f1"] == 1.0
        assert metrics["iou_mean"] == 1.0


class TestEvaluatePartialOverlap:
    """Test evaluation with partial overlap between predicted and reference."""

    def test_partial_metrics(self):
        ref_polys = [
            box(500000, 4000000, 500200, 4000200),
            box(500500, 4000500, 500700, 4000700),
        ]
        # First prediction overlaps first reference; second is a false positive
        pred_polys = [
            box(500050, 4000050, 500250, 4000250),  # partial overlap with ref[0]
            box(501000, 4001000, 501200, 4001200),  # no overlap (FP)
        ]
        reference = gpd.GeoDataFrame(geometry=ref_polys, crs="EPSG:32611")
        predicted = gpd.GeoDataFrame(geometry=pred_polys, crs="EPSG:32611")

        metrics = evaluate(predicted, reference)

        # Should have some matches but not perfect
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert metrics["count_predicted"] == 2
        assert metrics["count_reference"] == 2


class TestEvaluateNoReference:
    """Test evaluation with predictions but no reference."""

    def test_all_false_positives(self):
        pred = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:32611")
        ref = gpd.GeoDataFrame(geometry=[], crs="EPSG:32611")
        metrics = evaluate(pred, ref)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["count_fp"] == 1
