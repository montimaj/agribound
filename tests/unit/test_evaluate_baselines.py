"""Tests for agribound.evaluate.chance_baselines and evaluate(..., baselines=...).

Layouts are axis-aligned rectangles in UTM zone 13N on its central meridian
(as in test_evaluate.py), so the blind grid, which is built in the WGS 84 / UTM
zone of its extent's centre, is built in the layers' own CRS and its cells can
be written down by hand. Areas are computed in EPSG:6933: a UTM square of side
w has the area (w / k0)², k0 = 0.9996, hence the ``_K0`` factor.
"""

from __future__ import annotations

import importlib
import math

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from agribound.evaluate import (
    BLIND_GRID_MAX_CELLS,
    BLIND_GRID_OFFSET_STEPS,
    chance_baselines,
    evaluate,
)

# ``agribound.evaluate`` (package attribute) is the re-exported function.
ev = importlib.import_module("agribound.evaluate")

CRS = "EPSG:32613"
X0, Y0 = 500_000.0, 3_800_000.0
_K0 = 0.9996  # UTM central scale factor
METRIC_KEYS = (
    "precision",
    "recall",
    "f1",
    "iou_mean",
    "best_iou_mean",
    "area_weighted_precision",
    "area_weighted_recall",
    "area_weighted_f1",
    "oversegmentation_mean",
    "undersegmentation_mean",
    "count_predicted",
    "count_tp",
    "count_fp",
    "count_fn",
    "count_fp_unassigned",
    "predicted_area_ha",
)
TOLERANCE_KEYS = (
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "coverage_within_tolerance",
    "coverage_within_tolerance_area",
)


def sq(x: float, y: float, w: float = 100.0, h: float | None = None):
    """Axis-aligned rectangle with lower-left corner (X0 + x, Y0 + y)."""
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + (w if h is None else h))


def gdf(geoms, crs=CRS, **columns) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns, geometry=list(geoms), crs=crs)


def tiling(nx: int, ny: int, w: float = 100.0) -> gpd.GeoDataFrame:
    """*nx* x *ny* squares of side *w* from (X0, Y0), sharing their edges."""
    return gdf([sq(i * w, j * w, w) for i in range(nx) for j in range(ny)])


def same(a, b) -> bool:
    """Equality of nested results, with NaN equal to NaN."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# ---------------------------------------------------------------------------
# The blind grid
# ---------------------------------------------------------------------------


class TestBlindGrid:
    def test_grid_equal_to_the_reference_scores_one(self):
        ref = tiling(4, 4)
        out = chance_baselines(
            ref, ref, cell_side_m=100.0, offset_steps=1, boundary_tolerance_m=5.0
        )
        grid = out["blind_grid"]
        assert (out["grid_crs"], out["extent"]) == ("EPSG:32613", "predictions_bbox")
        assert grid["n_grids"] == 1 and grid["count_predicted"] == 16
        for key in ("precision", "recall", "f1", "boundary_f1", "area_weighted_f1"):
            assert grid[key] == pytest.approx(1.0)
        assert "shifted_reference" not in out

    def test_scores_equal_evaluate_on_the_same_cells(self):
        # The grid's cells written down by hand (anchored at the predictions'
        # bounding box, 70 m cells, clipped to it), scored by evaluate() with the
        # same non-default rules, give the baseline's numbers.
        ref = gdf([sq(0, 0, 130, 90), sq(140, 0, 80, 200), sq(0, 100, 120, 120), sq(230, 30, 60)])
        pred = gdf([sq(5, 5, 120, 80), sq(150, 10, 60, 180), sq(10, 110, 100, 100)])
        rules = {
            "iou_threshold": 0.3,
            "matching": "many_to_one",
            "boundary_tolerance_m": 10.0,
            "boundary_sample_spacing_m": None,
        }
        out = chance_baselines(
            pred, ref, offset_steps=1, cell_side_m=70.0, min_area_m2=0.0, **rules
        )
        x0, y0, x1, y1 = pred.total_bounds
        cells = [
            box(x, y, x + 70.0, y + 70.0).intersection(box(x0, y0, x1, y1))
            for y in np.arange(y0, y1, 70.0)
            for x in np.arange(x0, x1, 70.0)
        ]
        expected = evaluate(gdf(cells), ref, **rules)
        grid = out["blind_grid"]
        assert grid["count_predicted"] == len(cells)
        for key in METRIC_KEYS + TOLERANCE_KEYS:
            assert grid[key] == pytest.approx(expected[key], nan_ok=True), key
        assert (out["iou_threshold"], out["matching"], out["boundary_tolerance_m"]) == (
            0.3,
            "many_to_one",
            10.0,
        )

    def test_cell_side_and_minimum_area_come_from_the_predictions(self):
        pred = gdf([sq(0, 0, 100), sq(200, 0, 50), sq(400, 0, 200)])  # median 100 x 100
        out = chance_baselines(pred, tiling(2, 2), offset_steps=1)
        grid = out["blind_grid"]
        assert grid["cell_side_m"] == pytest.approx(100.0 / _K0, rel=1e-5)
        assert grid["median_predicted_area_ha"] == pytest.approx(1.0 / _K0**2, rel=1e-5)
        assert out["min_area_m2"] == pytest.approx(2500.0 / _K0**2, rel=1e-5)

    def test_offsets_are_scored_and_averaged(self):
        ref = tiling(4, 4)
        grid = chance_baselines(ref, ref, cell_side_m=100.0, offset_steps=2)["blind_grid"]
        assert grid["n_grids"] == 4 and grid["offset_steps"] == 2
        rows = grid["grids"]
        assert [(r["offset_x_m"], r["offset_y_m"]) for r in rows] == [
            (0.0, 0.0),
            (0.0, 50.0),
            (50.0, 0.0),
            (50.0, 50.0),
        ]
        # Shifted by half a cell, every cell overlaps two or four fields by a half or a
        # quarter: IoU 1/3 or 1/7, no match.
        assert [r["f1"] for r in rows] == pytest.approx([1.0, 0.0, 0.0, 0.0])
        # The half cells at the edges are smaller than the smallest prediction: dropped.
        assert [r["count_predicted"] for r in rows] == [16, 12, 12, 9]
        assert grid["f1"] == pytest.approx(0.25)
        assert (grid["f1_min"], grid["f1_max"]) == pytest.approx((0.0, 1.0))
        assert grid["count_predicted"] == pytest.approx(49 / 4)
        assert "boundary_f1_min" not in grid  # no tolerance

    def test_default_scores_36_grids(self):
        ref = tiling(3, 3)
        grid = chance_baselines(ref, ref, cell_side_m=100.0)["blind_grid"]
        assert BLIND_GRID_OFFSET_STEPS == 6
        assert grid["n_grids"] == 36 and grid["offset_steps"] == 6
        offsets = {(round(r["offset_x_m"], 6), round(r["offset_y_m"], 6)) for r in grid["grids"]}
        steps = {round(k * 100.0 / 6, 6) for k in range(6)}
        assert offsets == {(x, y) for x in steps for y in steps}
        assert grid["f1"] == pytest.approx(np.mean([r["f1"] for r in grid["grids"]]))

    @pytest.mark.parametrize(("east_m", "whole_cells"), [(0.0, 9), (250_000.0, 16)])
    def test_cells_as_large_as_every_prediction_are_kept(self, east_m, whole_cells):
        # Predictions of one size: the cells have the smallest prediction's true
        # area, the default minimum, but the UTM scale factor makes the true areas
        # of equal cells differ along the grid (by about 3e-9 over 300 m on the
        # central meridian); no whole cell may be dropped for that. The cell side,
        # 100 m divided by the scale factor, is 100.04 m on the central meridian
        # (3 x 3 whole cells in the 400 m square) and 99.96 m 250 km east (4 x 4).
        ref = gdf([sq(east_m + i * 100, j * 100) for i in range(4) for j in range(4)])
        aoi = gdf([sq(east_m, 0, 400)])
        for kwargs in ({}, {"study_area": aoi}, {"study_area": aoi, "footprint": True}):
            grid = chance_baselines(ref, ref, offset_steps=1, **kwargs)["blind_grid"]
            assert grid["count_predicted"] == whole_cells, kwargs

    def test_min_area_zero_keeps_the_edge_parts(self):
        ref = tiling(4, 4)
        grid = chance_baselines(ref, ref, cell_side_m=100.0, offset_steps=2, min_area_m2=0)[
            "blind_grid"
        ]
        assert [r["count_predicted"] for r in grid["grids"]] == [16, 20, 20, 25]

    def test_tolerance_metrics_are_reported(self):
        ref = tiling(3, 3)
        grid = chance_baselines(
            ref, ref, cell_side_m=100.0, offset_steps=2, boundary_tolerance_m=10.0
        )["blind_grid"]
        for key in TOLERANCE_KEYS:
            assert key in grid and all(key in r for r in grid["grids"])
        values = [r["boundary_f1"] for r in grid["grids"]]
        assert grid["boundary_f1_min"] == pytest.approx(min(values))
        assert grid["boundary_f1_max"] == pytest.approx(max(values))
        assert grid["grids"][0]["boundary_f1"] == pytest.approx(1.0)

    def test_study_area_sets_the_extent(self):
        ref = tiling(4, 4)
        left = tiling(2, 4)  # predictions on the left half only
        aoi = gdf([sq(0, 0, 400)])
        out = chance_baselines(left, ref, cell_side_m=100.0, offset_steps=1, study_area=aoi)
        assert out["extent"] == "study_area"
        assert out["blind_grid"]["count_predicted"] == 16
        assert out["blind_grid"]["f1"] == pytest.approx(1.0)
        bbox = chance_baselines(left, ref, cell_side_m=100.0, offset_steps=1)
        assert bbox["extent"] == "predictions_bbox"
        assert bbox["blind_grid"]["count_predicted"] == 8
        assert bbox["blind_grid"]["recall"] == pytest.approx(0.5)

    def test_study_area_in_another_crs(self):
        ref = tiling(4, 4)
        aoi = gdf([sq(-0.5, -0.5, 401.0)]).to_crs("EPSG:4326")
        out = chance_baselines(
            ref, ref, cell_side_m=100.0, offset_steps=1, study_area=aoi, min_area_m2=5000.0
        )
        assert out["grid_crs"] == "EPSG:32613" and out["min_area_m2"] == 5000.0
        # Cells 0.5 m off the fields still match them; the 1 m slivers are dropped.
        assert out["blind_grid"]["count_predicted"] == 16
        assert out["blind_grid"]["f1"] == pytest.approx(1.0)

    def test_geographic_predictions(self):
        ref = tiling(4, 4)
        utm = chance_baselines(ref, ref, offset_steps=2)
        geo = chance_baselines(ref.to_crs("EPSG:4326"), ref, offset_steps=2)
        assert geo["grid_crs"] == "EPSG:32613"
        assert geo["blind_grid"]["cell_side_m"] == pytest.approx(
            utm["blind_grid"]["cell_side_m"], rel=1e-6
        )
        assert [r["count_predicted"] for r in geo["blind_grid"]["grids"]] == [
            r["count_predicted"] for r in utm["blind_grid"]["grids"]
        ]

    def test_footprint_cuts_the_cells_to_the_predictions(self):
        ref = tiling(4, 4)
        left = tiling(2, 4)
        aoi = gdf([sq(0, 0, 400)])
        out = chance_baselines(
            left, ref, cell_side_m=100.0, offset_steps=1, study_area=aoi, footprint=True
        )
        grid = out["blind_grid"]
        assert grid["footprint"] is True
        assert grid["count_predicted"] == 8 and grid["count_tp"] == 8
        assert grid["recall"] == pytest.approx(0.5)
        # A narrower footprint: 4 whole cells and 4 strips 30 m wide (IoU 0.3).
        strip = gdf([sq(0, 0, 130, 400)])
        out = chance_baselines(
            strip,
            ref,
            cell_side_m=100.0,
            offset_steps=1,
            study_area=aoi,
            footprint=True,
            min_area_m2=0,
        )
        assert out["blind_grid"]["count_predicted"] == 8
        assert out["blind_grid"]["count_tp"] == 4

    def test_footprint_closes_cracks_between_predictions(self):
        # Two predictions 0.01 mm apart, as reprojection leaves neighbours: the
        # footprint is snapped to a 1 mm grid, so the 60 m cells across the crack
        # are not split in two (4 x 4 pieces, not 4 x 4 + 4).
        pred = gdf([sq(0, 0, 200, 100), sq(0, 100.00001, 200, 99.99999)])
        out = chance_baselines(
            pred, tiling(2, 2), offset_steps=1, cell_side_m=60.0, footprint=True, min_area_m2=0
        )
        assert out["blind_grid"]["count_predicted"] == 16

    def test_footprint_cut_equals_the_intersection_with_the_union(self):
        rng = np.random.default_rng(3)
        rects = [
            sq(*rng.uniform(0, 400, 2), w=rng.uniform(20, 150), h=rng.uniform(20, 150))
            for _ in range(40)
        ]
        union = shapely.union_all(rects)
        pieces = shapely.get_parts(union)
        cells = shapely.box(
            *np.meshgrid(np.arange(X0, X0 + 500, 37.0), np.arange(Y0, Y0 + 500, 37.0)),
            *np.meshgrid(np.arange(X0 + 37, X0 + 537, 37.0), np.arange(Y0 + 37, Y0 + 537, 37.0)),
        ).ravel()
        fast, fast_source = ev._cut_to(cells, pieces, shapely.STRtree(pieces))
        slow, slow_source = ev._clip_to(cells, union)
        assert len(fast) == len(slow)
        assert sorted(fast_source) == sorted(slow_source)  # the cell of each part
        assert sorted(shapely.area(fast)) == pytest.approx(sorted(shapely.area(slow)))
        assert shapely.union_all(fast).symmetric_difference(shapely.union_all(slow)).area < 1e-6

    def test_results_are_deterministic(self):
        ref = gdf([sq(0, 0, 130, 90), sq(140, 0, 80, 200), sq(0, 100, 120, 120)])
        pred = gdf([sq(5, 5, 120, 80), sq(150, 10, 60, 180), sq(10, 110, 100, 100)])
        kwargs = {"offset_steps": 3, "boundary_tolerance_m": 10.0, "shift_m": "auto"}
        assert same(chance_baselines(pred, ref, **kwargs), chance_baselines(pred, ref, **kwargs))

    def test_cell_limit(self, monkeypatch):
        ref = tiling(4, 4)
        # 100 m cells over a 400 m extent: 5 x 5 cells before clipping.
        monkeypatch.setattr(ev, "BLIND_GRID_MAX_CELLS", 25)
        chance_baselines(ref, ref, cell_side_m=100.0, offset_steps=1)
        monkeypatch.setattr(ev, "BLIND_GRID_MAX_CELLS", 24)
        with pytest.raises(ValueError, match="BLIND_GRID_MAX_CELLS"):
            chance_baselines(ref, ref, cell_side_m=100.0, offset_steps=1)
        assert BLIND_GRID_MAX_CELLS == 1_000_000


# ---------------------------------------------------------------------------
# The shifted reference
# ---------------------------------------------------------------------------


class TestShiftedReference:
    def test_auto_shift_is_half_the_median_width(self):
        ref = gdf([sq(0, 0, 100, 40), sq(200, 0, 60, 300), sq(400, 0, 80, 80)])
        out = chance_baselines(ref, ref, shift_m="auto", offset_steps=1, min_area_m2=0)
        assert out["shifted_reference"]["shift_m"] == pytest.approx(30.0)  # widths 40, 60, 80

    def test_small_and_large_shifts(self):
        ref = tiling(4, 4)
        # 10 m east and north: every field still overlaps its copy at IoU >= 0.68.
        near = chance_baselines(ref, ref, shift_m=10.0, offset_steps=1, min_area_m2=0)
        assert near["shifted_reference"]["shift_m"] == 10.0
        assert near["shifted_reference"]["f1"] == pytest.approx(1.0)
        # Half a field: IoU 1/7, no match; the copies clipped to 50 m are dropped.
        far = chance_baselines(ref, ref, shift_m=50.0, offset_steps=1)["shifted_reference"]
        assert far["f1"] == 0.0 and far["count_predicted"] == 9
        assert far["best_iou_mean"] == pytest.approx(1 / 7, rel=1e-6)

    def test_shifted_polygons_stay_one_feature(self):
        # A field recorded as two parts stays one shifted feature, and so does a U-shaped
        # field whose shifted copy the extent cuts into two parts.
        from shapely.geometry import MultiPolygon

        two_parts = MultiPolygon([sq(0, 0, 100), sq(150, 0, 100)])
        u_shape = sq(0, 200, 300, 100).difference(sq(100, 200, 100, 80))
        ref = gdf([two_parts, u_shape])
        aoi = gdf([sq(0, 0, 300, 285)])  # cuts the U's arms off its top
        out = chance_baselines(ref, ref, shift_m=3.0, offset_steps=1, study_area=aoi, min_area_m2=0)
        shifted = out["shifted_reference"]
        assert shifted["count_predicted"] == 2
        assert shifted["f1"] == pytest.approx(1.0)

    def test_shift_with_tolerance(self):
        # 3 m east and north: every boundary point is within 3 * sqrt(2) < 5 m of the copy's.
        ref = tiling(3, 3)
        out = chance_baselines(
            ref, ref, shift_m=3.0, offset_steps=1, boundary_tolerance_m=5.0, min_area_m2=0
        )
        assert out["shifted_reference"]["boundary_f1"] == pytest.approx(1.0)
        assert out["shifted_reference"]["f1"] == pytest.approx(1.0)

    def test_shift_moves_east_and_north(self):
        # The study area reaches 100 m beyond the fields to the west and south only.
        # Moved east and north by half a field, the copies of the 7 fields along the
        # east and north edges are cut by the study area to half or quarter fields,
        # which are dropped; moved west and south, all 16 copies would stay whole.
        ref = tiling(4, 4)
        aoi = gdf([sq(-100, -100, 500)])
        out = chance_baselines(ref, ref, shift_m=50.0, offset_steps=1, study_area=aoi)
        shifted = out["shifted_reference"]
        assert shifted["shift_m"] == 50.0
        assert shifted["count_predicted"] == 9
        # Each kept copy covers a quarter of four fields (IoU 1/7), and every field
        # meets one of them.
        assert shifted["best_iou_mean"] == pytest.approx(1 / 7, rel=1e-6)

    def test_auto_shift_needs_reference_polygons(self):
        with pytest.raises(ValueError, match="needs reference polygons"):
            chance_baselines(tiling(2, 2), gdf([]), shift_m="auto", offset_steps=1)

    def test_shift_needs_reference_polygons(self):
        # An empty shifted copy scored against an empty reference would score F1 1.0
        # (the convention for two empty layers): refused, as for "auto".
        with pytest.raises(ValueError, match="needs reference polygons to shift"):
            chance_baselines(tiling(2, 2), gdf([]), shift_m=10.0, offset_steps=1)
        out = chance_baselines(tiling(2, 2), gdf([]), offset_steps=1)  # no shift: scored
        assert out["blind_grid"]["f1"] == 0.0 and "shifted_reference" not in out

    @pytest.mark.parametrize("bad", [0, -5.0, float("nan"), "half", True])
    def test_bad_shift(self, bad):
        with pytest.raises(ValueError, match="shift_m"):
            chance_baselines(tiling(2, 2), tiling(2, 2), shift_m=bad)


# ---------------------------------------------------------------------------
# evaluate(..., baselines=...)
# ---------------------------------------------------------------------------


class TestEvaluateOptIn:
    @pytest.fixture
    def layers(self):
        ref = gdf([sq(0, 0, 130, 90), sq(140, 0, 80, 200), sq(0, 100, 120, 120)])
        pred = gdf([sq(5, 5, 120, 80), sq(150, 10, 60, 180), sq(10, 110, 100, 100)])
        return pred, ref

    @pytest.mark.parametrize("off", [False, None])
    def test_off_by_default(self, layers, off):
        pred, ref = layers
        base = evaluate(pred, ref, boundary_tolerance_m=5.0)
        assert "baselines" not in base
        assert same(evaluate(pred, ref, boundary_tolerance_m=5.0, baselines=off), base)

    def test_other_keys_unchanged(self, layers):
        pred, ref = layers
        base = evaluate(pred, ref, boundary_tolerance_m=5.0, bootstrap=20, strata=[1, 1, 2])
        with_baselines = evaluate(
            pred, ref, boundary_tolerance_m=5.0, bootstrap=20, strata=[1, 1, 2], baselines=True
        )
        assert same({k: v for k, v in with_baselines.items() if k != "baselines"}, base)

    def test_same_rules_as_the_evaluation(self, layers):
        pred, ref = layers
        rules = {"iou_threshold": 0.3, "matching": "many_to_one", "boundary_tolerance_m": 5.0}
        metrics = evaluate(pred, ref, **rules, baselines={"offset_steps": 2, "shift_m": 20.0})
        direct = chance_baselines(pred, ref, **rules, offset_steps=2, shift_m=20.0)
        assert same(metrics["baselines"], direct)
        assert metrics["baselines"]["iou_threshold"] == 0.3
        assert metrics["baselines"]["matching"] == "many_to_one"
        assert metrics["baselines"]["boundary_tolerance_m"] == 5.0
        assert "bootstrap" not in metrics["baselines"]["blind_grid"]

    def test_true_uses_the_defaults(self, layers):
        pred, ref = layers
        out = evaluate(pred, ref, baselines=True)["baselines"]
        assert out["blind_grid"]["n_grids"] == 36 and "shifted_reference" not in out

    @pytest.mark.parametrize("rule", ["iou_threshold", "matching", "boundary_tolerance_m"])
    def test_rules_are_not_options(self, layers, rule):
        pred, ref = layers
        with pytest.raises(ValueError, match="evaluate\\(\\)'s own"):
            evaluate(pred, ref, baselines={rule: 0.3})

    def test_unknown_option(self, layers):
        pred, ref = layers
        with pytest.raises(ValueError, match="unknown baselines option"):
            evaluate(pred, ref, baselines={"cells": 4})

    @pytest.mark.parametrize("bad", ["yes", 1, [("offset_steps", 2)]])
    def test_bad_value(self, layers, bad):
        pred, ref = layers
        with pytest.raises(ValueError, match="baselines must be"):
            evaluate(pred, ref, baselines=bad)

    def test_options_are_checked_before_the_evaluation(self, layers, monkeypatch):
        pred, ref = layers

        def fail(*args, **kwargs):
            raise AssertionError("evaluated before the options were checked")

        monkeypatch.setattr(ev, "_run", fail)
        with pytest.raises(ValueError, match="offset_steps"):
            evaluate(pred, ref, baselines={"offset_steps": 0})

    def test_uses_the_evaluated_layers(self, layers):
        # A null geometry in the predictions is dropped by evaluate() and by the
        # baselines alike: the median area and minimum area are the same.
        pred, ref = layers
        with_null = gdf([*pred.geometry, None])
        a = evaluate(with_null, ref, baselines={"offset_steps": 1})["baselines"]
        b = chance_baselines(pred, ref, offset_steps=1)
        assert same(a, b)

    def test_no_prediction(self, caplog):
        # An engine that found no field: the evaluation stands, and the baselines that
        # the layers cannot give are reported as skipped, with a WARNING.
        ref = tiling(2, 2)
        base = evaluate(gdf([]), ref)
        for options, advice in (
            (True, "pass cell_side_m and study_area"),
            ({"cell_side_m": 100.0}, "pass study_area"),
            ({"study_area": gdf([sq(0, 0, 200)])}, "pass cell_side_m"),
        ):
            caplog.clear()
            with caplog.at_level("WARNING", logger="agribound.evaluate"):
                metrics = evaluate(gdf([]), ref, baselines=options)
            assert set(metrics["baselines"]) == {"skipped"}
            assert metrics["baselines"]["skipped"].endswith(advice)
            assert "no predicted polygon was evaluated" in metrics["baselines"]["skipped"]
            assert same({k: v for k, v in metrics.items() if k != "baselines"}, base)
            assert any("Chance baselines not scored" in r.getMessage() for r in caplog.records)
        options = {"cell_side_m": 100.0, "study_area": gdf([sq(0, 0, 200)]), "offset_steps": 1}
        metrics = evaluate(gdf([]), ref, baselines=options)
        assert metrics["f1"] == 0.0
        assert metrics["baselines"]["blind_grid"]["f1"] == pytest.approx(1.0)

    def test_no_reference_with_a_shift(self, layers, caplog):
        pred, _ = layers
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            metrics = evaluate(pred, gdf([]), baselines={"shift_m": 10.0, "offset_steps": 1})
        assert metrics["f1"] == 0.0
        assert metrics["baselines"] == {
            "skipped": "shift_m needs reference polygons to shift, and no reference polygon "
            "was evaluated"
        }
        metrics = evaluate(pred, gdf([]), baselines={"shift_m": "auto"})
        assert metrics["baselines"]["skipped"].startswith("shift_m='auto' needs reference")
        # Without a shift, the blind grid is scored against the empty reference.
        metrics = evaluate(pred, gdf([]), baselines={"offset_steps": 1})
        assert metrics["baselines"]["blind_grid"]["f1"] == 0.0

    @pytest.mark.parametrize(
        ("aoi", "message"),
        [
            (gpd.GeoDataFrame(geometry=[sq(0, 0, 200)]), "study_area has no CRS"),
            (gdf([]), "study_area has no polygon"),
            (gdf([sq(0, 0, 0.0)]), "study_area has no polygon"),
        ],
    )
    def test_study_area_is_checked_before_the_evaluation(self, layers, monkeypatch, aoi, message):
        pred, ref = layers

        def fail(*args, **kwargs):
            raise AssertionError("evaluated before the study area was checked")

        monkeypatch.setattr(ev, "_run", fail)
        with pytest.raises(ValueError, match=message):
            evaluate(pred, ref, baselines={"study_area": aoi})


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


class TestArguments:
    def test_no_prediction_needs_a_cell_side(self):
        with pytest.raises(ValueError, match="pass cell_side_m"):
            chance_baselines(gdf([]), tiling(2, 2))

    def test_no_prediction_needs_a_study_area(self):
        with pytest.raises(ValueError, match="pass study_area"):
            chance_baselines(gdf([]), tiling(2, 2), cell_side_m=100.0)

    def test_no_prediction_with_cell_side_and_study_area(self):
        out = chance_baselines(
            gdf([]),
            tiling(2, 2),
            cell_side_m=100.0,
            offset_steps=1,
            study_area=gdf([sq(0, 0, 200)]),
        )
        assert out["min_area_m2"] == 0.0
        assert math.isnan(out["blind_grid"]["median_predicted_area_ha"])
        assert out["blind_grid"]["f1"] == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "6"])
    def test_offset_steps(self, bad):
        with pytest.raises(ValueError, match="offset_steps"):
            chance_baselines(tiling(2, 2), tiling(2, 2), offset_steps=bad)

    @pytest.mark.parametrize("bad", [0, -5.0, float("nan"), float("inf"), "big"])
    def test_cell_side(self, bad):
        with pytest.raises(ValueError, match="cell_side_m"):
            chance_baselines(tiling(2, 2), tiling(2, 2), cell_side_m=bad)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), "none"])
    def test_min_area(self, bad):
        with pytest.raises(ValueError, match="min_area_m2"):
            chance_baselines(tiling(2, 2), tiling(2, 2), min_area_m2=bad)

    def test_footprint_must_be_a_bool(self):
        with pytest.raises(ValueError, match="footprint"):
            chance_baselines(tiling(2, 2), tiling(2, 2), footprint="yes")

    def test_study_area_type(self):
        with pytest.raises(TypeError, match="study_area"):
            chance_baselines(tiling(2, 2), tiling(2, 2), study_area=sq(0, 0, 200))

    def test_study_area_without_polygon(self):
        with pytest.raises(ValueError, match="study_area has no polygon"):
            chance_baselines(tiling(2, 2), tiling(2, 2), study_area=gdf([]))

    def test_study_area_without_crs(self):
        aoi = gpd.GeoDataFrame(geometry=[sq(0, 0, 200)])
        with pytest.raises(ValueError, match="no CRS"):
            chance_baselines(tiling(2, 2), tiling(2, 2), study_area=aoi)

    def test_scoring_rules_are_validated(self):
        with pytest.raises(ValueError, match="iou_threshold"):
            chance_baselines(tiling(2, 2), tiling(2, 2), iou_threshold=0)
        with pytest.raises(ValueError, match="matching"):
            chance_baselines(tiling(2, 2), tiling(2, 2), matching="best")

    def test_inputs_must_be_geodataframes(self):
        with pytest.raises(TypeError):
            chance_baselines([sq(0, 0)], tiling(2, 2))

    def test_public_api(self):
        for name in ("chance_baselines", "BLIND_GRID_OFFSET_STEPS", "BLIND_GRID_MAX_CELLS"):
            assert name in ev.__all__


# ---------------------------------------------------------------------------
# ignore
# ---------------------------------------------------------------------------


class TestIgnore:
    @pytest.fixture
    def layers(self):
        # A 4 x 4 tiling: the two western columns are scored, the two eastern ones are
        # left out of the score.
        tiles = tiling(4, 4)
        west = tiles.iloc[:8].reset_index(drop=True)
        east = tiles.iloc[8:].reset_index(drop=True)
        return tiles, west, east, gdf([sq(0, 0, 400)])

    def test_cells_in_ignored_fields_are_not_scored(self, layers):
        tiles, west, east, aoi = layers
        rules = {"cell_side_m": 100.0, "offset_steps": 1, "study_area": aoi}
        grid = chance_baselines(tiles, west, ignore=east, **rules)["blind_grid"]
        assert (grid["count_predicted"], grid["count_predicted_ignored"]) == (8, 8)
        assert grid["f1"] == pytest.approx(1.0)
        without = chance_baselines(tiles, west, **rules)["blind_grid"]
        assert without["count_predicted"] == 16 and without["precision"] == pytest.approx(0.5)
        assert "count_predicted_ignored" not in without

    def test_evaluate_scores_the_baselines_with_its_ignore(self, layers):
        tiles, west, east, aoi = layers
        options = {"offset_steps": 2, "study_area": aoi, "shift_m": 60.0}
        m = evaluate(tiles, west, boundary_tolerance_m=5.0, ignore=east, baselines=options)
        direct = chance_baselines(tiles, west, boundary_tolerance_m=5.0, ignore=east, **options)
        assert same(m["baselines"], direct)
        assert (m["count_predicted_ignored"], m["f1"]) == (8, pytest.approx(1.0))
        # Shifted 60 m east and north and clipped to the square, six shifted fields are
        # kept; the three of the second column now lie (by representative point) in the
        # ignored third column.
        shifted = direct["shifted_reference"]
        assert (shifted["count_predicted"], shifted["count_predicted_ignored"]) == (3, 3)

    def test_ignore_is_a_rule_not_an_option(self, layers):
        tiles, west, east, _ = layers
        with pytest.raises(ValueError, match="evaluate\\(\\)'s own ignore"):
            evaluate(tiles, west, baselines={"ignore": east})

    def test_every_prediction_ignored(self, layers, caplog):
        tiles, west, _, aoi = layers
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            m = evaluate(tiles, west, ignore=aoi, baselines=True)
        assert (m["count_predicted"], m["count_predicted_ignored"]) == (0, 16)
        assert "no predicted polygon was evaluated" in m["baselines"]["skipped"]
        with pytest.raises(ValueError, match="no predicted polygon was evaluated"):
            chance_baselines(tiles, west, ignore=aoi)
