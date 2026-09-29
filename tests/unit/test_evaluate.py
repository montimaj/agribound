"""Tests for agribound.evaluate (object-level accuracy assessment).

Layouts are axis-aligned rectangles in UTM zone 13N on its central meridian.
Distances are computed in the UTM zone of each field, which for most layouts
here is the input CRS, so hand-computed distances and lengths hold to
rounding. IoU and areas are computed in EPSG:6933 after splitting edges into
pieces of at most 50 m and reprojecting vertex by vertex; for these layouts
that changes area ratios by less than about 1e-6 (relative), and ratios are
compared with ``rel=1e-5``. Absolute EPSG:6933 areas are true areas, i.e. UTM
planar areas divided by k0² = 0.9996² on the central meridian, hence the
``_K0`` factor.
"""

from __future__ import annotations

import importlib
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from agribound.evaluate import (
    BOUNDARY_MAX_SAMPLES,
    evaluate,
    evaluate_frame,
    pixels_per_field,
)
from agribound.io.crs import utm_epsg, utm_zone_for_lon

# ``agribound.evaluate`` (package attribute) is the re-exported function.
ev = importlib.import_module("agribound.evaluate")

CRS = "EPSG:32613"
X0, Y0 = 500_000.0, 3_800_000.0
_K0 = 0.9996  # UTM central scale factor


def sq(x: float, y: float, w: float = 100.0, h: float | None = None):
    """Axis-aligned rectangle with lower-left corner (X0 + x, Y0 + y)."""
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + (w if h is None else h))


def gdf(geoms, crs=CRS, **columns) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns, geometry=list(geoms), crs=crs)


# ---------------------------------------------------------------------------
# 0.1.x behaviour kept
# ---------------------------------------------------------------------------


class TestV1Compatibility:
    def test_perfect_match(self):
        polys = [sq(0, 0, 200), sq(500, 500, 200)]
        m = evaluate(gdf(polys), gdf(polys))
        assert m["iou_mean"] == pytest.approx(1.0)
        assert (m["precision"], m["recall"], m["f1"]) == pytest.approx((1.0, 1.0, 1.0))
        assert (m["count_tp"], m["count_fp"], m["count_fn"]) == (2, 0, 0)
        assert m["area_error_mean_m2"] == pytest.approx(0.0, abs=1e-6)
        for key in ("over_segmentation", "under_segmentation"):
            assert m[key] == 0.0

    def test_empty_predictions(self, sample_reference_gdf):
        m = evaluate(gdf([], crs="EPSG:32611"), sample_reference_gdf)
        assert (m["precision"], m["recall"], m["f1"], m["iou_mean"]) == (0.0, 0.0, 0.0, 0.0)
        assert m["count_predicted"] == 0
        assert m["count_fn"] == len(sample_reference_gdf)
        assert m["oversegmentation_mean"] == pytest.approx(1.0)  # nothing overlaps
        assert math.isnan(m["undersegmentation_mean"])
        assert math.isnan(m["boundary_distance_mean_m"])

    def test_both_empty_rates_are_one(self):
        m = evaluate(gdf([]), gdf([]), boundary_tolerance_m=5)
        for key in ("precision", "recall", "f1", "iou_mean", "area_weighted_recall"):
            assert m[key] == 1.0
        assert m["boundary_f1"] == 1.0
        assert m["count_reference"] == m["count_predicted"] == 0
        assert math.isnan(m["hausdorff_mean_m"])

    def test_no_reference_all_false_positives(self):
        m = evaluate(gdf([sq(0, 0, 10)]), gdf([]))
        assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)
        assert m["count_fp"] == 1
        assert m["count_fp_unassigned"] == 1

    @pytest.mark.parametrize("which", ["predicted", "reference", "both"])
    def test_empty_inputs_report_requested_threshold(self, which):
        # 0.1.x hard-coded 0.5 in the empty-input branches.
        full = gdf([sq(0, 0)])
        pred = gdf([]) if which in ("predicted", "both") else full
        ref = gdf([]) if which in ("reference", "both") else full
        assert evaluate(pred, ref, iou_threshold=0.7)["iou_threshold"] == 0.7

    def test_partial_overlap_exact_values(self):
        ref = gdf([sq(0, 0, 200), sq(500, 500, 200)])
        # 150x150 overlap of two 200x200 squares: IoU = 22500 / 57500.
        pred = gdf([sq(50, 50, 200), sq(1000, 1000, 200)])
        m = evaluate(pred, ref, iou_threshold=0.3)
        assert m["count_tp"] == 1 and m["count_fp"] == 1 and m["count_fn"] == 1
        assert m["iou_mean"] == pytest.approx(22500 / 57500, rel=1e-5)
        assert (m["precision"], m["recall"], m["f1"]) == pytest.approx((0.5, 0.5, 0.5))
        # At the default threshold the pair does not match (IoU 0.391).
        m50 = evaluate(pred, ref)
        assert m50["count_tp"] == 0 and m50["iou_mean"] == 0.0
        assert m50["best_iou_mean"] == pytest.approx(22500 / 57500 / 2, rel=1e-5)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class TestMatching:
    def test_split_reference(self):
        # One 100x100 field predicted as a 60 m and a 40 m wide strip.
        ref = gdf([sq(0, 0)])
        pred = gdf([sq(0, 0, 60, 100), sq(60, 0, 40, 100)])
        m = evaluate(pred, ref)
        assert m["count_tp"] == 1 and m["count_fp"] == 1 and m["count_fn"] == 0
        assert m["iou_mean"] == pytest.approx(0.6, rel=1e-5)
        assert m["precision"] == pytest.approx(0.5) and m["recall"] == 1.0
        # Clinton / Persello OS & US against the max-overlap (60 m) strip.
        assert m["oversegmentation_mean"] == pytest.approx(0.4, rel=1e-5)
        assert m["undersegmentation_mean"] == pytest.approx(0.0, abs=1e-6)
        # 0.1.x definition only sees the split when both strips pass the threshold.
        assert m["over_segmentation"] == 0.0
        assert evaluate(pred, ref, iou_threshold=0.3)["over_segmentation"] == 1.0

    def test_merged_references_one_to_one_vs_many_to_one(self):
        # Two fields (100 m and 60 m wide) predicted as one 160 m wide polygon.
        ref = gdf([sq(0, 0), sq(100, 0, 60, 100)])
        pred = gdf([sq(0, 0, 160, 100)])
        one = evaluate(pred, ref, iou_threshold=0.3)
        assert (one["count_tp"], one["count_fp"], one["count_fn"]) == (1, 0, 1)
        assert (one["precision"], one["recall"]) == pytest.approx((1.0, 0.5))
        assert one["iou_mean"] == pytest.approx(100 / 160, rel=1e-5)
        many = evaluate(pred, ref, iou_threshold=0.3, matching="many_to_one")
        # 0.1.x behaviour: both references count as TP although there is one prediction.
        assert (many["count_tp"], many["count_fp"], many["count_fn"]) == (2, 0, 0)
        assert (many["precision"], many["recall"]) == pytest.approx((1.0, 1.0))
        assert many["iou_mean"] == pytest.approx((100 / 160 + 60 / 160) / 2, rel=1e-5)
        assert many["under_segmentation"] == 1.0  # 0.1.x: prediction matches >1 reference
        # US against the merged prediction: 1 - 10000/16000 and 1 - 6000/16000.
        assert one["undersegmentation_mean"] == pytest.approx((0.375 + 0.625) / 2, rel=1e-5)
        assert one["oversegmentation_mean"] == pytest.approx(0.0, abs=1e-5)

    def test_duplicate_references(self):
        ref = gdf([sq(0, 0), sq(0, 0)])
        pred = gdf([sq(0, 0)])
        one = evaluate(pred, ref)
        assert (one["count_tp"], one["count_fn"], one["recall"]) == (1, 1, 0.5)
        many = evaluate(pred, ref, matching="many_to_one")
        assert (many["count_tp"], many["count_fn"], many["recall"]) == (2, 0, 1.0)

    def test_duplicate_predictions_are_false_positives(self):
        m = evaluate(gdf([sq(0, 0), sq(0, 0)]), gdf([sq(0, 0)]))
        assert (m["count_tp"], m["count_fp"], m["precision"]) == (1, 1, 0.5)
        assert m["count_fp_unassigned"] == 0  # the duplicate overlaps the reference

    def test_missing_field(self):
        ref = gdf([sq(0, 0), sq(1000, 0)])
        frame = evaluate_frame(gdf([sq(0, 0)]), ref)
        missed = frame.iloc[1]
        assert not missed["matched"] and missed["pred_index"] is None
        assert missed["best_iou"] == 0.0 and missed["n_overlapping_pred"] == 0
        assert missed["oversegmentation"] == 1.0 and math.isnan(missed["undersegmentation"])
        assert math.isnan(missed["iou"]) and math.isnan(missed["hausdorff_m"])

    def test_greedy_assignment_order(self):
        # Pairs: (ref0, pred0) 0.9, (ref0, pred1) 0.8, (ref1, pred0) 0.7.
        # Greedy by descending IoU takes (ref0, pred0) first, leaving ref1 and
        # pred1 unmatched (an optimal assignment would match both references).
        pairs = ev._Pairs(
            ref=np.array([0, 0, 1]),
            pred=np.array([0, 1, 0]),
            inter=np.ones(3),
            iou=np.array([0.9, 0.8, 0.7]),
        )
        ref_match, pred_matched, _ = ev._match(pairs, 2, 2, 0.5, "one_to_one")
        assert ref_match.tolist() == [0, -1]
        assert pred_matched.tolist() == [True, False]

    def test_ties_break_by_position(self):
        pairs = ev._Pairs(
            ref=np.array([0, 0]), pred=np.array([1, 0]), inter=np.ones(2), iou=np.array([0.6, 0.6])
        )
        ref_match, _, _ = ev._match(pairs, 1, 2, 0.5, "one_to_one")
        assert ref_match.tolist() == [0]

    def test_threshold_is_inclusive_up_to_rounding(self):
        pairs = ev._Pairs(
            ref=np.array([0]), pred=np.array([0]), inter=np.ones(1), iou=np.array([0.5 - 1e-12])
        )
        assert ev._match(pairs, 1, 1, 0.5, "one_to_one")[0].tolist() == [0]
        pairs.iou[:] = 0.5 - 1e-6
        assert ev._match(pairs, 1, 1, 0.5, "one_to_one")[0].tolist() == [-1]

    @pytest.mark.parametrize(
        ("strata", "home"),
        [(["a", "b"], "a"), (["b", "a"], "b")],
    )
    def test_many_to_one_home_is_the_highest_iou_match(self, strata, home):
        # One 160 m prediction matches both a 100 m field (IoU 0.625) and a
        # 60 m field (IoU 0.375) at threshold 0.3. In a stratum breakdown it
        # counts in the stratum of its highest-IoU match, the 100 m field.
        ref = gdf([sq(0, 0), sq(100, 0, 60, 100)], s=strata)
        pred = gdf([sq(0, 0, 160, 100)])
        m = evaluate(pred, ref, iou_threshold=0.3, matching="many_to_one", strata="s")
        other = "b" if home == "a" else "a"
        assert m["per_stratum"][home]["count_predicted"] == 1
        assert m["per_stratum"][other]["count_predicted"] == 0
        for key in (home, other):
            assert m["per_stratum"][key]["count_tp"] == 1
            assert m["per_stratum"][key]["count_fp"] == 0
        frame = evaluate_frame(pred, ref, iou_threshold=0.3, matching="many_to_one")
        assert frame["pred_index"].tolist() == [0, 0]

    def test_many_to_one_home_ties_break_by_reference_position(self):
        pairs = ev._Pairs(
            ref=np.array([0, 1]), pred=np.array([0, 0]), inter=np.ones(2), iou=np.array([0.6, 0.6])
        )
        ref_match, pred_matched, home = ev._match(pairs, 2, 1, 0.5, "many_to_one")
        assert ref_match.tolist() == [0, 0] and pred_matched.tolist() == [True]
        assert home.tolist() == [0]
        pairs.iou[:] = [0.6, 0.7]
        assert ev._match(pairs, 2, 1, 0.5, "many_to_one")[2].tolist() == [1]

    def test_touching_polygons_do_not_match(self):
        m = evaluate(gdf([sq(100, 0)]), gdf([sq(0, 0)]), iou_threshold=1e-6)
        assert m["count_tp"] == 0 and m["best_iou_mean"] == 0.0


# ---------------------------------------------------------------------------
# Area weighting
# ---------------------------------------------------------------------------


class TestAreaWeighted:
    def test_area_weighted_recall_and_precision(self):
        ref = gdf([sq(0, 0, 200), sq(1000, 0, 50)])  # 4 ha matched, 0.25 ha missed
        pred = gdf([sq(0, 0, 200), sq(3000, 0, 50)])  # plus a 0.25 ha false positive
        m = evaluate(pred, ref)
        assert m["recall"] == 0.5 and m["precision"] == 0.5
        assert m["area_weighted_recall"] == pytest.approx(40000 / 42500, rel=1e-5)
        assert m["area_weighted_precision"] == pytest.approx(40000 / 42500, rel=1e-5)
        assert m["area_weighted_f1"] == pytest.approx(40000 / 42500, rel=1e-5)
        assert m["reference_area_ha"] == pytest.approx(4.25 / _K0**2, rel=1e-5)
        assert m["count_fp_unassigned"] == 1

    def test_signed_and_absolute_area_error(self):
        ref = gdf([sq(0, 0), sq(1000, 0)])
        pred = gdf([sq(0, 0, 110, 100), sq(1000, 0, 90, 100)])  # +1000 and -1000 m²
        m = evaluate(pred, ref)
        assert m["area_error_mean_m2"] == pytest.approx(1000 / _K0**2, rel=1e-5)
        assert m["area_error_signed_mean_m2"] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Boundary metrics
# ---------------------------------------------------------------------------


class TestBoundaryMetrics:
    @pytest.mark.parametrize("d", [1.0, 5.0, 12.5])
    def test_shifted_square_hausdorff_and_mean_distance(self, d):
        # Shifting a square of side s by d < s/2 gives Hausdorff d and a
        # symmetric mean boundary distance of exactly d / 2.
        frame = evaluate_frame(gdf([sq(d, 0)]), gdf([sq(0, 0)]))
        assert frame["hausdorff_m"].iloc[0] == pytest.approx(d, rel=1e-9)
        assert frame["boundary_mean_distance_m"].iloc[0] == pytest.approx(d / 2, rel=1e-9)
        assert frame["iou"].iloc[0] == pytest.approx((100 - d) / (100 + d), rel=1e-5)

    def test_diagonal_shift_hausdorff_is_euclidean(self):
        frame = evaluate_frame(gdf([sq(3, 4)]), gdf([sq(0, 0)]))
        assert frame["hausdorff_m"].iloc[0] == pytest.approx(5.0, rel=1e-9)

    def test_boundary_precision_recall_at_tolerance(self):
        # 100 m square shifted by 5 m, tolerance 4 m: per square, 2 x 99 m of the
        # top/bottom edges and 2 x 4 m of the trailing edge are within 4 m.
        m = evaluate(gdf([sq(5, 0)]), gdf([sq(0, 0)]), boundary_tolerance_m=4.0)
        assert m["boundary_recall"] == pytest.approx(206 / 400, rel=1e-12)
        assert m["boundary_precision"] == pytest.approx(206 / 400, rel=1e-12)
        assert m["boundary_f1"] == pytest.approx(206 / 400, rel=1e-12)
        assert m["boundary_tolerance_m"] == 4.0

    def test_coverage_within_tolerance(self):
        ref = gdf([sq(0, 0), sq(1000, 0, 200)])
        pred = gdf([sq(5, 0), sq(1000, 0, 200)])  # mean distances 2.5 m and 0 m
        loose = evaluate(pred, ref, boundary_tolerance_m=3.0)
        assert loose["coverage_within_tolerance"] == 1.0
        tight = evaluate(pred, ref, boundary_tolerance_m=2.0)
        assert tight["coverage_within_tolerance"] == 0.5
        assert tight["coverage_within_tolerance_area"] == pytest.approx(0.8, rel=1e-5)
        frame = evaluate_frame(pred, ref, boundary_tolerance_m=2.0)
        assert frame["within_tolerance"].tolist() == [False, True]

    def test_unmatched_fields_are_not_within_tolerance(self):
        m = evaluate(gdf([]), gdf([sq(0, 0)]), boundary_tolerance_m=10.0)
        assert m["coverage_within_tolerance"] == 0.0 and m["boundary_recall"] == 0.0

    def test_overlapping_buffers_are_not_double_counted(self):
        frame = evaluate_frame(gdf([sq(0, 0), sq(0, 0)]), gdf([sq(0, 0)]), boundary_tolerance_m=1)
        assert frame["boundary_within_tolerance_m"].iloc[0] == pytest.approx(400.0, rel=1e-9)
        assert frame["boundary_recall"].iloc[0] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("spacing_arg", "spacing"),
        [
            (ev.BOUNDARY_SAMPLE_SPACING_M, 1.0),  # the default: 1 m for every boundary
            # Length-dependent: perimeter 6 km > BOUNDARY_MAX_SAMPLES m, so 6 m.
            (None, 6000 / BOUNDARY_MAX_SAMPLES),
            (0.5, 0.5),
        ],
    )
    def test_long_boundaries_are_sampled_with_bounded_error(self, spacing_arg, spacing):
        # W x H = 2 km x 1 km field shifted by d = 10 m along x: Hausdorff d
        # and a symmetric mean boundary distance of d * H / (W + H).
        pred, ref = gdf([sq(10, 0, 2000, 1000)]), gdf([sq(0, 0, 2000, 1000)])
        frame = evaluate_frame(pred, ref, boundary_sample_spacing_m=spacing_arg)
        haus = frame["hausdorff_m"].iloc[0]
        assert 10 - spacing / 2 <= haus <= 10 + 1e-9
        assert frame["boundary_mean_distance_m"].iloc[0] == pytest.approx(
            10 * 1000 / 3000, abs=spacing / 4
        )
        assert frame["boundary_sample_spacing_m"].iloc[0] == pytest.approx(spacing, rel=1e-9)
        m = evaluate(pred, ref, boundary_sample_spacing_m=spacing_arg)
        assert m["boundary_sample_spacing_m"] == spacing_arg
        assert m["boundary_sample_spacing_max_m"] == pytest.approx(spacing, rel=1e-9)

    @staticmethod
    def _notched_strip():
        # Reference: 2000 x 10 m strip. Prediction: the same strip minus a
        # 15 m wide notch cut from the top edge down to 0.5 m above the bottom
        # edge, over x in [a, b] = [1000.5, 1015.5].
        a, b = 1000.5, 1015.5
        ref = sq(0, 0, 2000, 10)
        pred = Polygon(
            [
                (X0, Y0),
                (X0 + 2000, Y0),
                (X0 + 2000, Y0 + 10),
                (X0 + b, Y0 + 10),
                (X0 + b, Y0 + 0.5),
                (X0 + a, Y0 + 0.5),
                (X0 + a, Y0 + 10),
                (X0, Y0 + 10),
            ]
        )
        return gdf([pred]), gdf([ref])

    @pytest.mark.parametrize("spacing_arg", [ev.BOUNDARY_SAMPLE_SPACING_M, None, 0.25])
    def test_notched_strip_hand_computed(self, spacing_arg):
        # The top edge of the reference over the notch is min(x - a, b - x)
        # from the prediction: its maximum, 7.5 m at the notch centre (not a
        # vertex), is the Hausdorff distance (the notch walls are at most 5 m
        # from the reference, its floor 0.5 m). Mean distance: the reference
        # contributes 2 * 7.5² / 2 = 56.25 m², each wall
        # ∫_0.5^10 min(y, 10 - y) dy = 24.875 m² and the floor 15 * 0.5 m²,
        # over |∂r| + |∂p| = 4020 + 4039 m.
        pred, ref = self._notched_strip()
        row = evaluate_frame(pred, ref, boundary_sample_spacing_m=spacing_arg).iloc[0]
        delta = row["boundary_sample_spacing_m"]
        expected_delta = 4039 / BOUNDARY_MAX_SAMPLES if spacing_arg is None else spacing_arg
        assert delta == pytest.approx(expected_delta, rel=1e-9)
        assert 7.5 - delta / 2 <= row["hausdorff_m"] <= 7.5 + 1e-9
        mean = (56.25 + 2 * 24.875 + 7.5) / (4020 + 4039)
        assert row["boundary_mean_distance_m"] == pytest.approx(mean, abs=delta / 4)
        # IoU: the prediction lies inside the reference. The long top edge of
        # the reference (2 vertices) and the collinear top edges of the
        # prediction (vertices at the notch) must still coincide in EPSG:6933.
        # Without splitting edges before reprojection the reference's single
        # 2 km chord misses the curved image of the line by ~5 cm, which gave
        # 0.990285 instead of 0.992875 (relative error 2.6e-3). With 50 m
        # pieces the residual chord mismatch is ~1e-6.
        assert row["iou"] == pytest.approx((20000 - 15 * 9.5) / 20000, rel=5e-6)

    def test_distances_are_nan_without_a_match(self):
        frame = evaluate_frame(gdf([]), gdf([sq(0, 0)]))
        assert math.isnan(frame["boundary_sample_spacing_m"].iloc[0])
        assert math.isnan(evaluate(gdf([]), gdf([sq(0, 0)]))["boundary_sample_spacing_max_m"])

    def test_boundary_precision_uses_the_prediction_zone(self, monkeypatch):
        # A reference field whose centroid is just east of the zone 12/13
        # border (108° W) and a prediction whose centroid is just west of it,
        # both built in zone 13 coordinates and given in EPSG:4326. Boundary
        # recall is computed in the reference's zone (13), boundary precision
        # in the prediction's (12), so each layer's boundaries are
        # reprojected into the other's zone.
        import pyproj

        xb, yb = pyproj.Transformer.from_crs(4326, 32613, always_xy=True).transform(-108.0, 34.0)
        ref = gpd.GeoSeries([box(xb - 20, yb, xb + 80, yb + 100)], crs=32613).to_crs(4326)
        pred = gpd.GeoSeries([box(xb - 70, yb, xb + 30, yb + 100)], crs=32613).to_crs(4326)
        calls = []
        real_in_zone = ev._in_zone

        def spy(layer, idx, epsg):
            calls.append((layer.name, int(epsg)))
            return real_in_zone(layer, idx, epsg)

        monkeypatch.setattr(ev, "_in_zone", spy)
        m = evaluate(pred, ref, boundary_tolerance_m=1.0)
        assert m["utm_epsg_codes"] == [32612, 32613]
        assert ("predicted", 32613) in calls  # predicted boundaries in the reference's zone
        assert ("reference", 32612) in calls  # reference boundaries in the prediction's zone
        # Per layer: 51 m of each horizontal edge and 2 m of one vertical edge
        # lie within 1 m of the other boundary (to the ~0.05 % scale error of
        # zone-13 coordinates 3° from the central meridian).
        assert m["boundary_precision"] == pytest.approx(104 / 400, rel=1e-3)
        assert m["boundary_recall"] == pytest.approx(104 / 400, rel=1e-3)

    def test_interior_rings_are_boundary(self):
        # Reference: 100 m square with a central 20 m hole; prediction: the
        # same square without the hole. Every point of the hole ring is 40 m
        # from the outer ring, and the outer rings coincide.
        outer = [(X0, Y0), (X0 + 100, Y0), (X0 + 100, Y0 + 100), (X0, Y0 + 100)]
        hole = [(X0 + 40, Y0 + 40), (X0 + 60, Y0 + 40), (X0 + 60, Y0 + 60), (X0 + 40, Y0 + 60)]
        ref = gdf([Polygon(outer, [hole])])
        frame = evaluate_frame(gdf([sq(0, 0)]), ref, boundary_tolerance_m=1.0)
        row = frame.iloc[0]
        assert row["iou"] == pytest.approx(9600 / 10000, rel=1e-5)
        assert row["perimeter_m"] == pytest.approx(480.0, rel=1e-12)
        assert row["hausdorff_m"] == pytest.approx(40.0, rel=1e-12)
        assert row["boundary_mean_distance_m"] == pytest.approx(80 * 40 / 880, rel=1e-12)
        assert row["boundary_within_tolerance_m"] == pytest.approx(400.0, rel=1e-12)
        assert row["boundary_recall"] == pytest.approx(400 / 480, rel=1e-12)
        assert row["undersegmentation"] == pytest.approx(0.04, rel=1e-5)

    def test_tolerance_chunking_does_not_change_results(self, monkeypatch):
        ref = gdf([sq(0, 0), sq(100, 0), sq(0, 100, 50), sq(300, 0, 80)])
        pred = gdf([sq(3, 2), sq(104, 0, 90), sq(1, 101, 48), sq(290, 5, 70)])
        kwargs = {"boundary_tolerance_m": 4.0}
        whole = evaluate(pred, ref, **kwargs)
        frame = evaluate_frame(pred, ref, **kwargs)
        monkeypatch.setattr(ev, "_SEGMENT_CHUNK", 1)  # one polygon per chunk
        chunked = evaluate(pred, ref, **kwargs)
        assert chunked.keys() == whole.keys()
        for key, value in whole.items():  # equal up to floating-point rounding
            expected = pytest.approx(value, rel=1e-12) if isinstance(value, float) else value
            assert chunked[key] == expected, key
        pd.testing.assert_frame_equal(evaluate_frame(pred, ref, **kwargs), frame, rtol=1e-12)

    def test_large_candidate_shared_by_many_fields(self, monkeypatch):
        # One jagged prediction over a 6 x 6 grid of 90 m fields: every field
        # has it as a candidate, and its far parts are filtered out per chunk.
        # Chunked results must equal a direct computation per field, whatever
        # the chunk budget.
        rng = np.random.default_rng(1)
        side, step = 600.0, 2.0
        xs = np.arange(0, side + step, step)
        jag = rng.uniform(-1, 1, (4, len(xs)))
        ring = np.r_[
            np.c_[xs, jag[0]],
            np.c_[side + jag[1], xs],
            np.c_[xs[::-1], side + jag[2]],
            np.c_[jag[3], xs[::-1]],
        ] + [X0, Y0]
        import shapely

        blob = shapely.make_valid(Polygon(ring))
        refs = [sq(x, y, 90) for x in range(0, 600, 100) for y in range(0, 600, 100)]
        pred, ref = gdf([blob]), gdf(refs)
        tol = 3.0
        blob_line = np.array([blob.boundary], dtype=object)
        ref_lines = np.array([g.boundary for g in refs], dtype=object)
        direct_recall = [
            ev._length_within(ref_lines[k : k + 1], blob_line, tol)[0] for k in range(36)
        ]
        direct_precision = ev._length_within(blob_line, ref_lines, tol)[0]
        results = []
        for budget in (1, 50, ev._SEGMENT_CHUNK, 10**9):
            monkeypatch.setattr(ev, "_SEGMENT_CHUNK", budget)
            frame = evaluate_frame(pred, ref, boundary_tolerance_m=tol)
            assert frame["boundary_within_tolerance_m"].tolist() == pytest.approx(
                direct_recall, rel=1e-12, abs=1e-9
            )
            m = evaluate(pred, ref, boundary_tolerance_m=tol)
            results.append((m["boundary_recall"], m["boundary_precision"]))
        assert results[0][1] == pytest.approx(direct_precision / blob.length, rel=1e-12)
        assert all(r == pytest.approx(results[0], rel=1e-12) for r in results)
        assert 0 < results[0][0] < 1 and 0 < results[0][1] < 1

    def test_distances_use_each_fields_utm_zone(self):
        # Fields at the central meridians of zones 12 and 13, given in EPSG:4326.
        # A 5 m shift is 5 m in the field's own zone; in the other zone (6° away)
        # the UTM scale error would inflate it by ~0.4 %.
        parts_ref, parts_pred = [], []
        for epsg in (32612, 32613):
            parts_ref.append(gpd.GeoSeries([sq(0, 0)], crs=epsg).to_crs(4326))
            parts_pred.append(gpd.GeoSeries([sq(5, 0)], crs=epsg).to_crs(4326))
        ref = gpd.GeoDataFrame(geometry=pd.concat(parts_ref, ignore_index=True), crs=4326)
        pred = gpd.GeoDataFrame(geometry=pd.concat(parts_pred, ignore_index=True), crs=4326)
        m = evaluate(pred, ref)
        assert m["utm_epsg_codes"] == [32612, 32613]
        assert m["hausdorff_mean_m"] == pytest.approx(5.0, rel=1e-5)
        frame = evaluate_frame(pred, ref)
        assert frame["utm_epsg"].tolist() == [32612, 32613]
        assert frame["perimeter_m"].tolist() == pytest.approx([400.0, 400.0], rel=1e-5)


def _lines(*coords_list):
    from shapely.geometry import LineString

    return np.array([LineString(c) for c in coords_list], dtype=object)


class TestLengthWithinTolerance:
    """Exact length of a line within distance r of other lines (segment capsules)."""

    A = ((0.0, 0.0), (10.0, 0.0))

    @pytest.mark.parametrize(
        ("others", "r", "expected"),
        [
            # Nearest part is the end point (5, 3): |x - 5| <= sqrt(r² - 9).
            ([((5, 3), (5, 10))], 4.0, 2 * math.sqrt(7)),
            # Parallel segment 1 m away: its rectangle plus the two end discs.
            ([((2, 1), (6, 1))], 2.0, 4 + 2 * math.sqrt(3)),
            # Perpendicular crossing: |x - 5| <= r.
            ([((5, -5), (5, 5))], 1.0, 2.0),
            # 45° crossing at (5, 0): |x - 5| sin 45° <= r.
            ([((0, -5), (10, 5))], 1.0, 2 * math.sqrt(2)),
            # Collinear overlap (a shared edge): [0, 3] plus r beyond its end.
            ([((-5, 0), (3, 0))], 1.0, 4.0),
            # Two capsules whose parts overlap on the line: union, not sum.
            ([((2, 0.5), (3, 0.5)), ((3.5, 0.5), (8, 0.5))], 1.0, 6 + math.sqrt(3)),
            # Duplicated other line: counted once.
            ([((5, -5), (5, 5)), ((5, -5), (5, 5))], 1.0, 2.0),
            # Farther than r: nothing.
            ([((0, 3), (10, 3))], 2.0, 0.0),
            # A far segment (dropped by the bounding-box filter) changes nothing.
            ([((5, -5), (5, 5)), ((100, 100), (200, 200))], 1.0, 2.0),
            # Covers the whole line.
            ([((-1, 0.5), (11, 0.5))], 1.0, 10.0),
        ],
    )
    def test_hand_computed_lengths(self, others, r, expected):
        got = ev._length_within(_lines(self.A), _lines(*others), r)
        assert got.tolist() == pytest.approx([expected], abs=1e-12)

    def test_multiple_lines_and_multipart_lines(self):
        from shapely.geometry import MultiLineString

        lines = np.array(
            [MultiLineString([self.A, ((0, 20), (10, 20))]), _lines(((0, 10), (4, 10)))[0]],
            dtype=object,
        )
        others = _lines(((5, -5), (5, 5)), ((0, 11), (4, 11)))
        got = ev._length_within(lines, others, 1.0)
        # Line 0: 2 m near the crossing, its far part (y = 20) is 9 m away.
        # Line 1: the whole 4 m is 1 m from a parallel line of equal extent.
        assert got.tolist() == pytest.approx([2.0, 4.0], abs=1e-12)

    def test_ranges(self):
        got = ev._ranges(np.array([5, 0, 9]), np.array([2, 0, 3]))
        assert got.tolist() == [5, 6, 9, 10, 11]
        assert ev._ranges(np.array([3]), np.array([0])).tolist() == []

    def test_agrees_with_fine_geos_buffers(self):
        # Independent check against GEOS buffers with 256 segments per quarter
        # circle. They approximate the circular arcs (and GEOS may simplify
        # the input slightly), so agreement is to a small fraction of the line
        # length rather than to rounding.
        import shapely

        rng = np.random.default_rng(0)
        lines = [
            shapely.LineString(np.cumsum(rng.normal(size=(60, 2)) * 3.0, axis=0)) for _ in range(20)
        ]
        others = [shapely.affinity.translate(g, 1.5, -0.8) for g in lines]
        for r in (0.5, 2.0, 6.0):
            exact = [
                ev._length_within(_lines(g.coords), _lines(o.coords), r)[0]
                for g, o in zip(lines, others, strict=True)
            ]
            buffered = [
                g.intersection(o.buffer(r, quad_segs=256)).length
                for g, o in zip(lines, others, strict=True)
            ]
            lengths = [g.length for g in lines]
            for e, b, total in zip(exact, buffered, lengths, strict=True):
                assert 0.0 <= e <= total + 1e-9
                assert e == pytest.approx(b, abs=1e-4 * total)


# ---------------------------------------------------------------------------
# Strata and size classes
# ---------------------------------------------------------------------------


@pytest.fixture
def stratified_layout():
    # Stratum "a": 2 fields, both matched. Stratum "b": 2 fields, one matched,
    # one overlapped by an unmatched (too small) prediction. Plus one
    # prediction far from every reference field.
    ref = gdf(
        [sq(0, 0), sq(200, 0), sq(1000, 0), sq(1200, 0)],
        basin=["a", "a", "b", "b"],
    )
    pred = gdf([sq(0, 0), sq(200, 0), sq(1000, 0), sq(1200, 0, 30), sq(5000, 5000)])
    return pred, ref


class TestStrata:
    def test_per_stratum_counts_and_home_assignment(self, stratified_layout):
        pred, ref = stratified_layout
        m = evaluate(pred, ref, strata="basin")
        a, b = m["per_stratum"]["a"], m["per_stratum"]["b"]
        assert m["strata_column"] == "basin" and list(m["per_stratum"]) == ["a", "b"]
        assert (a["n"], a["count_tp"], a["count_fp"], a["count_fn"]) == (2, 2, 0, 0)
        assert (b["n"], b["count_tp"], b["count_fp"], b["count_fn"]) == (2, 1, 1, 1)
        assert b["precision"] == 0.5 and b["recall"] == 0.5
        assert m["count_fp"] == 2 and m["count_fp_unassigned"] == 1
        assert a["count_tp"] + b["count_tp"] == m["count_tp"]
        assert a["count_fp"] + b["count_fp"] + m["count_fp_unassigned"] == m["count_fp"]
        assert a["median_area_ha"] == pytest.approx(1 / _K0**2, rel=1e-5)

    def test_series_and_array_strata(self, stratified_layout):
        pred, ref = stratified_layout
        ref.index = ["f1", "f2", "f3", "f4"]
        by_column = evaluate(pred, ref, strata="basin")["per_stratum"]
        shuffled = ref["basin"].iloc[::-1]  # same labels, different order
        by_series = evaluate(pred, ref, strata=shuffled)["per_stratum"]
        by_array = evaluate(pred, ref, strata=np.array(["a", "a", "b", "b"]))["per_stratum"]
        assert by_series == by_column == by_array

    def test_missing_stratum_values(self, stratified_layout):
        pred, ref = stratified_layout
        m = evaluate(pred, ref, strata=["a", None, "b", np.nan])
        assert m["count_reference_no_stratum"] == 2
        assert m["per_stratum"]["a"]["n"] == 1 and m["per_stratum"]["b"]["n"] == 1
        assert m["count_reference"] == 4

    def test_bad_strata(self, stratified_layout):
        pred, ref = stratified_layout
        with pytest.raises(ValueError, match="not found"):
            evaluate(pred, ref, strata="nope")
        with pytest.raises(ValueError, match="one value per reference row"):
            evaluate(pred, ref, strata=["a", "b"])
        with pytest.raises(ValueError, match="indexed like reference"):
            evaluate(pred, ref, strata=pd.Series(["a"] * 4, index=[9, 8, 7, 6]))


class TestSizeClasses:
    def test_explicit_edges(self):
        ref = gdf([sq(0, 0, 50), sq(1000, 0), sq(2000, 0, 200)])  # 0.25, 1, 4 ha
        pred = gdf([sq(0, 0, 50)])
        m = evaluate(pred, ref, size_bins=[0, 0.5, 2])
        assert list(m["per_size_class"]) == ["0-0.5", "0.5-2"]
        small, mid = m["per_size_class"]["0-0.5"], m["per_size_class"]["0.5-2"]
        assert (small["n"], small["recall"]) == (1, 1.0)
        assert (mid["n"], mid["recall"]) == (1, 0.0)
        assert (mid["lower_ha"], mid["upper_ha"]) == (0.5, 2.0)
        assert m["count_reference_outside_size_classes"] == 1
        assert m["size_class_edges_ha"] == [0.0, 0.5, 2.0]

    def test_auto_edges_use_1_2_5_series(self):
        ref = gdf([sq(0, 0, 50), sq(1000, 0), sq(2000, 0, 200)])
        m = evaluate(gdf([]), ref, size_bins="auto")
        assert m["size_class_edges_ha"] == [0.2, 0.5, 1.0, 2.0, 5.0]
        assert m["count_reference_outside_size_classes"] == 0
        assert sum(v["n"] for v in m["per_size_class"].values()) == 3

    def test_empty_size_class_has_nan_rates(self):
        m = evaluate(gdf([sq(0, 0, 50)]), gdf([sq(0, 0, 50)]), size_bins=[0, 0.5, 1, 2])
        empty = m["per_size_class"]["1-2"]
        assert empty["n"] == 0 and empty["count_reference"] == 0
        assert math.isnan(empty["recall"]) and math.isnan(empty["precision"])
        assert m["per_size_class"]["0-0.5"]["recall"] == 1.0

    def test_last_edge_is_inclusive(self):
        groups = ev._resolve_size_bins([0.0, 1.0, 2.0], np.array([0.0, 1e4, 2e4, 2.5e4]))
        assert groups.codes.tolist() == [0, 1, 1, -1]
        groups = ev._resolve_size_bins([1.0, math.inf], np.array([5e3, 1e4, 1e9]))
        assert groups.codes.tolist() == [-1, 0, 0]

    def test_close_edges_get_distinct_keys(self):
        groups = ev._resolve_size_bins([1.0, 1.0000001, 1.0000002], np.array([1.00000015e4]))
        assert groups.keys == ["1.0-1.0000001", "1.0000001-1.0000002"]
        assert groups.codes.tolist() == [1]

    @pytest.mark.parametrize("bins", [[1.0], [2.0, 1.0], [-1.0, 1.0], [0.0, math.nan], "log"])
    def test_invalid_bins(self, bins):
        with pytest.raises(ValueError):
            evaluate(gdf([]), gdf([sq(0, 0)]), size_bins=bins)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_deterministic_and_chunk_invariant(self, stratified_layout, monkeypatch):
        pred, ref = stratified_layout
        kwargs = {"bootstrap": 200, "bootstrap_seed": 3, "boundary_tolerance_m": 2.0}
        first = evaluate(pred, ref, **kwargs)["bootstrap"]
        assert evaluate(pred, ref, **kwargs)["bootstrap"] == first
        monkeypatch.setattr(ev, "_BOOTSTRAP_CHUNK_CELLS", 7)  # 1 resample per chunk
        assert evaluate(pred, ref, **kwargs)["bootstrap"] == first
        other = evaluate(pred, ref, bootstrap=200, bootstrap_seed=4)["bootstrap"]
        assert other["ci"] != first["ci"]
        lo, hi = first["ci"]["recall"]
        assert 0.0 <= lo <= 0.75 <= hi <= 1.0
        assert first["n_resamples"] == 200 and first["method"] == "percentile"
        assert set(first["ci"]) >= {"coverage_within_tolerance", "boundary_f1", "f1"}

    def test_stratified_resampling_keeps_stratum_sizes(self):
        # Stratum a: 3 matched fields, stratum b: 1 missed field. Resampling
        # within strata keeps 3 + 1 fields, so recall is 0.75 in every resample.
        ref = gdf([sq(0, 0), sq(200, 0), sq(400, 0), sq(600, 0)], s=["a", "a", "a", "b"])
        pred = gdf([sq(0, 0), sq(200, 0), sq(400, 0)])
        stratified = evaluate(pred, ref, strata="s", bootstrap=100)
        assert stratified["bootstrap"]["ci"]["recall"] == pytest.approx([0.75, 0.75])
        assert stratified["per_stratum"]["a"]["ci"]["recall"] == pytest.approx([1.0, 1.0])
        assert "within strata" in stratified["bootstrap"]["resampling"]
        pooled = evaluate(pred, ref, bootstrap=100)["bootstrap"]["ci"]["recall"]
        assert pooled[0] < 0.75 < pooled[1]

    def test_undefined_metrics_are_counted(self):
        # No prediction ever matches, so the matched-pair distance means are
        # undefined in every resample.
        m = evaluate(gdf([]), gdf([sq(0, 0), sq(500, 0)]), bootstrap=20)
        assert m["bootstrap"]["n_undefined"]["hausdorff_mean_m"] == 20
        assert m["bootstrap"]["n_undefined"]["precision"] == 20  # no predictions: 0 / 0
        assert m["precision"] == 0.0  # the point estimate keeps the 0.1.x convention
        assert all(math.isnan(v) for v in m["bootstrap"]["ci"]["hausdorff_mean_m"])
        assert m["bootstrap"]["ci"]["recall"] == [0.0, 0.0]

    def test_intervals_match_direct_resampling_of_the_frame(self):
        # The documented procedure, written out directly: resample reference
        # fields with the child seeds of SeedSequence(seed) and recompute
        # recall and area-weighted recall from the per-field table.
        ref = gdf([sq(200 * k, 0, 60 + 20 * k) for k in range(8)])
        pred = gdf([sq(200 * k + (0 if k % 3 else 25), 0, 60 + 20 * k) for k in range(8)])
        n_boot, seed = 300, 11
        m = evaluate(pred, ref, bootstrap=n_boot, bootstrap_seed=seed)
        frame = evaluate_frame(pred, ref)
        matched = frame["matched"].to_numpy(dtype=float)
        area = frame["area_m2"].to_numpy()
        assert 0 < matched.sum() < len(matched)  # both outcomes occur
        recall, aw_recall = [], []
        for child in np.random.SeedSequence(seed).spawn(n_boot):
            draw = np.random.default_rng(child).integers(0, len(ref), size=len(ref))
            recall.append(matched[draw].mean())
            aw_recall.append((area * matched)[draw].sum() / area[draw].sum())
        ci = m["bootstrap"]["ci"]
        assert ci["recall"] == pytest.approx(np.percentile(recall, [2.5, 97.5]).tolist())
        assert ci["area_weighted_recall"] == pytest.approx(
            np.percentile(aw_recall, [2.5, 97.5]).tolist()
        )

    def test_perfect_layer_has_degenerate_intervals(self):
        # One 4 ha field among four: absent from ~(3/4)^4 of the pooled
        # resamples. Those resamples leave the class undefined (not 0), so its
        # interval stays [1, 1].
        polys = [sq(0, 0), sq(500, 0), sq(1000, 0), sq(2000, 0, 200)]
        m = evaluate(gdf(polys), gdf(polys), bootstrap=200, size_bins=[0, 2, 5])
        assert m["bootstrap"]["ci"]["f1"] == pytest.approx([1.0, 1.0])
        big = m["per_size_class"]["2-5"]
        assert big["n"] == 1
        assert big["ci"]["precision"] == pytest.approx([1.0, 1.0])
        assert big["ci"]["recall"] == pytest.approx([1.0, 1.0])
        assert 0 < big["ci_n_undefined"]["recall"] < 200


# ---------------------------------------------------------------------------
# Geometry and CRS handling
# ---------------------------------------------------------------------------


class TestGeometryAndCrs:
    def test_mixed_crs_inputs(self):
        ref = gdf([sq(0, 0), sq(1000, 0)])
        pred_utm = gdf([sq(5, 0), sq(1000, 0)])
        pred_ll = pred_utm.to_crs("EPSG:4326")
        a, b = evaluate(pred_utm, ref), evaluate(pred_ll, ref)
        for key in ("iou_mean", "precision", "recall", "hausdorff_mean_m"):
            assert a[key] == pytest.approx(b[key], rel=1e-5)

    def test_missing_crs_raises(self):
        with pytest.raises(ValueError, match="no CRS"):
            evaluate(gpd.GeoDataFrame(geometry=[sq(0, 0)]), gdf([sq(0, 0)]))

    def test_layer_without_geometries_needs_no_crs(self):
        # e.g. an engine that found nothing and returned a CRS-less empty frame
        ref = gdf([sq(0, 0), sq(500, 0)])
        for empty in (gpd.GeoDataFrame(geometry=[]), gpd.GeoDataFrame(geometry=[None])):
            m = evaluate(empty, ref)
            assert (m["count_predicted"], m["count_fn"], m["recall"]) == (0, 2, 0.0)
        assert evaluate(gpd.GeoDataFrame(geometry=[]), gpd.GeoDataFrame(geometry=[]))["f1"] == 1.0

    def test_equal_area_crs_option(self):
        m = evaluate(gdf([sq(5, 0)]), gdf([sq(0, 0)]), equal_area_crs="EPSG:5070")
        assert m["equal_area_crs"] == "EPSG:5070"
        assert m["iou_mean"] == pytest.approx(95 / 105, rel=1e-5)
        with pytest.raises(ValueError, match="projected"):
            evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0)]), equal_area_crs="EPSG:4326")
        # Areas are reported in m², so feet- or kilometre-based CRSs are refused.
        with pytest.raises(ValueError, match="metres"):
            evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0)]), equal_area_crs="EPSG:2263")
        with pytest.raises(ValueError, match="metres"):
            pixels_per_field(gdf([sq(0, 0)]), 10.0, equal_area_crs="+proj=cea +units=km")
        assert evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0)]))["equal_area_crs"] == "EPSG:6933"

    def test_invalid_and_null_geometries(self):
        bowtie = Polygon([(X0, Y0), (X0 + 100, Y0 + 100), (X0 + 100, Y0), (X0, Y0 + 100), (X0, Y0)])
        assert not bowtie.is_valid
        ref = gpd.GeoDataFrame(geometry=[bowtie, None, sq(1000, 0)], crs=CRS)
        m = evaluate(gdf([sq(1000, 0)]), ref)
        assert m["count_reference_repaired"] == 1
        assert m["count_reference_dropped"] == 1
        assert m["count_reference"] == 2 and m["count_tp"] == 1
        frame = evaluate_frame(gdf([sq(1000, 0)]), ref)
        assert frame.index.tolist() == [0, 2]
        assert frame["area_m2"].iloc[0] == pytest.approx(5000 / _K0**2, rel=1e-5)

    def test_invalid_predictions_are_repaired_and_counted(self):
        # The bowtie repairs (structure method) to two triangles of 2500 m²,
        # kept as one MultiPolygon prediction.
        bowtie = Polygon([(X0, Y0), (X0 + 100, Y0 + 100), (X0 + 100, Y0), (X0, Y0 + 100), (X0, Y0)])
        pred = gpd.GeoDataFrame(geometry=[bowtie, sq(1000, 0)], crs=CRS)
        m = evaluate(pred, gdf([sq(1000, 0)]))
        assert m["count_predicted_repaired"] == 1 and m["count_reference_repaired"] == 0
        assert m["count_predicted"] == 2 and m["count_predicted_dropped"] == 0
        assert m["predicted_area_ha"] == pytest.approx((0.5 + 1.0) / _K0**2, rel=1e-5)

    def test_old_shapely_raises_actionable_import_error(self, monkeypatch):
        import shapely

        def make_valid_2_0(geometry, **kwargs):  # shapely 2.0.x signature
            raise TypeError("make_valid() got an unexpected keyword argument 'method'")

        monkeypatch.setattr(shapely, "make_valid", make_valid_2_0)
        bowtie = Polygon([(X0, Y0), (X0 + 100, Y0 + 100), (X0 + 100, Y0), (X0, Y0 + 100), (X0, Y0)])
        ref = gpd.GeoDataFrame(geometry=[bowtie, sq(1000, 0)], crs=CRS)
        with pytest.raises(ImportError, match=r"reference layer contains 1 invalid.*shapely>=2\.1"):
            evaluate(gdf([sq(1000, 0)]), ref)
        with pytest.raises(ImportError, match=r"shapely>=2\.1"):
            pixels_per_field(ref, 10.0)
        # Valid layers need no repair and are evaluated as usual.
        assert evaluate(gdf([sq(1000, 0)]), gdf([sq(1000, 0)]))["f1"] == 1.0

    def test_old_geos_raises_actionable_import_error(self, monkeypatch):
        import shapely

        monkeypatch.setattr(shapely, "geos_version", (3, 9, 1))
        bowtie = Polygon([(X0, Y0), (X0 + 100, Y0 + 100), (X0 + 100, Y0), (X0, Y0 + 100), (X0, Y0)])
        pred = gpd.GeoDataFrame(geometry=[bowtie], crs=CRS)
        with pytest.raises(ImportError, match=r"predicted layer contains 1 invalid.*GEOS 3\.9\.1"):
            evaluate(pred, gdf([sq(0, 0)]))

    # UTM (conformal), Web Mercator, and Mollweide (spherical formulas applied
    # to WGS 84 coordinates: PROJ's own areal scale factor is exactly 1 there,
    # yet ellipsoidal areas change by ~0.25 % at 34° N).
    @pytest.mark.parametrize("crs", ["EPSG:32613", "EPSG:3857", "ESRI:54009"])
    def test_crs_that_does_not_preserve_area_warns(self, crs, caplog):
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            m = evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0)]), equal_area_crs=crs)
        assert f"equal_area_crs {crs} does not preserve area" in caplog.text
        assert "times their areas in EPSG:6933" in caplog.text
        assert m["equal_area_crs"] == crs
        caplog.clear()
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            pixels_per_field(gdf([sq(0, 0)]), 10.0, equal_area_crs=crs)
        assert "does not preserve area" in caplog.text

    def test_web_mercator_areas_are_scaled(self):
        # At ~34.3° N a Web Mercator area is ~1.47 times the true area, so the
        # 1 ha field is reported as ~1.47 ha and falls in another size class.
        field = gdf([sq(0, 0)])
        m = evaluate(field, field, equal_area_crs="EPSG:3857", size_bins=[0, 1.2, 2])
        assert m["reference_area_ha"] == pytest.approx(
            field.to_crs("EPSG:3857").area.iloc[0] / 1e4, rel=1e-5
        )
        assert m["reference_area_ha"] > 1.4
        assert m["per_size_class"]["1.2-2"]["n"] == 1

    @pytest.mark.parametrize(
        "crs",
        [None, "EPSG:6933", "EPSG:5070", "ESRI:102003", "EPSG:8857", "ESRI:54034", "EPSG:3035"],
    )
    def test_equal_area_crs_does_not_warn(self, crs, caplog):
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            evaluate(gdf([sq(5, 0)]), gdf([sq(0, 0)]), equal_area_crs=crs)
            pixels_per_field(gdf([sq(0, 0)]), 10.0, equal_area_crs=crs)
        assert "does not preserve area" not in caplog.text

    def test_crs_label_requires_an_exact_authority_match(self):
        # pyproj's default to_epsg() confidence (70) would report this GRS80
        # cylindrical equal-area CRS as EPSG:6933, which is on WGS 84.
        proj = "+proj=cea +lat_ts=30 +units=m"
        label = evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0)]), equal_area_crs=proj)["equal_area_crs"]
        assert label.startswith("+proj=cea") and "6933" not in label
        assert (
            evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0)]), equal_area_crs="ESRI:102003")[
                "equal_area_crs"
            ]
            == "ESRI:102003"
        )

    def test_densify_step_units(self):
        import pyproj

        assert ev._densify_step(pyproj.CRS("EPSG:26913")) == pytest.approx(50.0)
        assert ev._densify_step(pyproj.CRS("EPSG:2263")) == pytest.approx(50.0 / 0.3048006096)
        assert ev._densify_step(pyproj.CRS("EPSG:4326")) == pytest.approx(
            50.0 / (6_378_137.0 * math.pi / 180.0)
        )
        engineering = pyproj.CRS(
            'LOCAL_CS["grid",LOCAL_DATUM["d",0],UNIT["metre",1],AXIS["X",EAST],AXIS["Y",NORTH]]'
        )
        assert ev._densify_step(engineering) is None

    def test_densification_keeps_source_geometry(self):
        # Splitting edges adds collinear vertices only: areas in an equal-area
        # source CRS are unchanged.
        ref = gdf([sq(0, 0, 2000, 10)]).to_crs("EPSG:6933")
        area = ref.geometry.area.iloc[0]
        frame = evaluate_frame(ref, ref)
        assert frame["area_m2"].iloc[0] == pytest.approx(area, rel=1e-12)
        assert frame["iou"].iloc[0] == pytest.approx(1.0, rel=1e-12)

    def test_non_polygonal_predictions_are_dropped(self):
        from shapely.geometry import LineString, Point

        pred = gpd.GeoDataFrame(
            geometry=[sq(0, 0), Point(X0, Y0), LineString([(X0, Y0), (X0 + 1, Y0)])], crs=CRS
        )
        m = evaluate(pred, gdf([sq(0, 0)]))
        assert m["count_predicted"] == 1 and m["count_predicted_dropped"] == 2
        assert m["precision"] == 1.0

    def test_geoseries_inputs(self):
        s = gpd.GeoSeries([sq(0, 0)], crs=CRS)
        assert evaluate(s, s)["f1"] == 1.0

    def test_utm_zone_formula_matches_io_crs(self):
        lon = np.array([-180.0, -179.99, -105.0, -108.0, -0.1, 0.0, 5.99, 179.99, 180.0])
        lat = np.array([10.0, -10.0, 34.0, 34.0, -1.0, 1.0, 45.0, -45.0, 1.0])
        expected = [utm_epsg(utm_zone_for_lon(x), y < 0) for x, y in zip(lon, lat, strict=True)]
        assert ev._utm_epsg_array(lon, lat).tolist() == expected


# ---------------------------------------------------------------------------
# Argument validation, output types
# ---------------------------------------------------------------------------


class TestArguments:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"iou_threshold": 0.0},
            {"iou_threshold": 1.5},
            {"matching": "hungarian"},
            {"bootstrap": -1},
            {"bootstrap": 2.5},
            {"boundary_tolerance_m": 0.0},
            {"boundary_tolerance_m": float("nan")},
            # float(True) == 1.0: booleans are refused rather than read as 1.
            {"boundary_tolerance_m": True},
            {"iou_threshold": True},
            {"iou_threshold": np.True_},
            {"boundary_sample_spacing_m": 0.0},
            {"boundary_sample_spacing_m": -1.0},
            {"boundary_sample_spacing_m": float("nan")},
            {"boundary_sample_spacing_m": True},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0)]), **kwargs)

    def test_type_error_for_non_geodataframes(self):
        with pytest.raises(TypeError):
            evaluate([sq(0, 0)], gdf([sq(0, 0)]))

    def test_output_is_json_serialisable_builtins(self, stratified_layout):
        pred, ref = stratified_layout
        m = evaluate(
            pred, ref, strata="basin", size_bins="auto", bootstrap=5, boundary_tolerance_m=1
        )

        def check(value):
            if isinstance(value, dict):
                assert all(isinstance(k, str) for k in value)
                for v in value.values():
                    check(v)
            elif isinstance(value, list):
                for v in value:
                    check(v)
            else:
                assert value is None or type(value) in (bool, int, float, str), type(value)

        check(m)
        json.dumps(m)
        assert isinstance(m["count_tp"], int) and isinstance(m["precision"], float)


# ---------------------------------------------------------------------------
# evaluate_frame / pixels_per_field
# ---------------------------------------------------------------------------


class TestFrame:
    def test_columns_index_and_consistency(self, stratified_layout):
        pred, ref = stratified_layout
        ref.index = ["f1", "f2", "f3", "f4"]
        pred.index = [10, 11, 12, 13, 14]
        frame = evaluate_frame(pred, ref, strata="basin", size_bins=[0, 2])
        assert frame.index.tolist() == ["f1", "f2", "f3", "f4"]
        assert frame["pred_index"].tolist() == [10, 11, 12, None]
        assert frame["max_overlap_pred_index"].tolist() == [10, 11, 12, 13]
        assert frame["stratum"].tolist() == ["a", "a", "b", "b"]
        assert frame["size_class"].tolist() == ["0-2"] * 4
        assert frame["undersegmentation"].iloc[3] == pytest.approx(0.0, abs=1e-6)
        assert frame["oversegmentation"].iloc[3] == pytest.approx(0.91, rel=1e-5)  # 30x30 m
        assert frame.attrs["matching"] == "one_to_one"
        m = evaluate(pred, ref)
        assert frame["matched"].sum() == m["count_tp"]
        assert frame.loc[frame["matched"], "iou"].mean() == pytest.approx(m["iou_mean"])
        # Joins back onto the reference.
        assert ref.join(frame)["area_ha"].notna().all()


class TestPixelsPerField:
    def test_values_and_nan_for_missing(self):
        ref = gpd.GeoDataFrame(geometry=[sq(0, 0), None, sq(1000, 0, 30)], crs=CRS)
        p = pixels_per_field(ref, 10.0)
        assert p.name == "pixels_per_field" and p.index.tolist() == [0, 1, 2]
        assert p.iloc[0] == pytest.approx(100 / _K0**2, rel=1e-5)
        assert math.isnan(p.iloc[1])
        assert p.iloc[2] == pytest.approx(9 / _K0**2, rel=1e-5)

    @pytest.mark.parametrize("gsd", [0.0, -1.0, float("inf"), "x", True])
    def test_invalid_gsd(self, gsd):
        with pytest.raises(ValueError):
            pixels_per_field(gdf([sq(0, 0)]), gsd)

    def test_layer_without_geometries_needs_no_crs(self):
        p = pixels_per_field(gpd.GeoSeries([None, None]), 10.0)
        assert p.index.tolist() == [0, 1] and p.isna().all()
        with pytest.raises(ValueError, match="no CRS"):
            pixels_per_field(gpd.GeoSeries([sq(0, 0)]), 10.0)


# ---------------------------------------------------------------------------
# Diagnostics and the CLI contract
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_geos_errors_fall_back_to_pairwise_then_a_1_mm_grid(self, monkeypatch, caplog):
        # Simulate GEOS topology errors: the vectorised call fails, and one of
        # the pairs also fails one by one, so it is recomputed on a 1 mm grid.
        import shapely

        real_intersection = shapely.intersection

        def flaky(a, b, grid_size=None, **kwargs):
            if isinstance(a, np.ndarray):
                raise shapely.errors.GEOSException("TopologyException: simulated")
            if grid_size is None and a.area > 30_000:  # the 200 m reference field
                raise shapely.errors.GEOSException("TopologyException: simulated")
            return real_intersection(a, b, grid_size=grid_size, **kwargs)

        ref = gdf([sq(0, 0, 200), sq(1000, 0)])
        pred = gdf([sq(50, 50, 200), sq(1000, 0)])
        expected = evaluate(pred, ref, iou_threshold=0.3)
        monkeypatch.setattr(shapely, "intersection", flaky)
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            m = evaluate(pred, ref, iou_threshold=0.3)
        assert "Vectorised intersection failed; recomputing 2 intersections one by one" in (
            caplog.text
        )
        assert "1 intersections raised a GEOS topology error" in caplog.text
        assert m["count_tp"] == expected["count_tp"] == 2
        # Snapping to a 1 mm grid moves each vertex by < 1 mm, which changes the
        # 150 x 150 m overlap (perimeter 600 m) by < 0.5 m², i.e. < 1e-4.
        assert m["iou_mean"] == pytest.approx(expected["iou_mean"], rel=1e-4)
        assert expected["iou_mean"] == pytest.approx((22500 / 57500 + 1) / 2, rel=1e-5)

    def test_warns_when_layers_do_not_overlap(self, caplog):
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0)], crs="EPSG:32612"))
        assert "No predicted polygon overlaps any reference polygon" in caplog.text

    def test_strata_keys_must_be_distinct_as_strings(self):
        with pytest.raises(ValueError, match="distinct after conversion"):
            evaluate(gdf([sq(0, 0)]), gdf([sq(0, 0), sq(500, 0)]), strata=[1, "1"])


def test_cli_forwards_all_options_to_evaluate(tmp_path):
    from click.testing import CliRunner

    from agribound.cli import main

    ref = gdf([sq(0, 0), sq(1000, 0, 200)], basin=["a", "b"]).to_crs("EPSG:4326")
    pred = gdf([sq(5, 0)]).to_crs("EPSG:4326")
    ref_path, pred_path = tmp_path / "ref.gpkg", tmp_path / "pred.gpkg"
    ref.to_file(ref_path)
    pred.to_file(pred_path)
    out = tmp_path / "metrics.json"
    args = [
        "evaluate",
        "-p",
        str(pred_path),
        "-r",
        str(ref_path),
        "--iou-threshold",
        "0.6",
        "--strata-column",
        "basin",
        "--size-bins",
        "0,2,10",
        "--bootstrap",
        "20",
        "--bootstrap-seed",
        "1",
        "--boundary-tolerance-m",
        "3",
        "--equal-area-crs",
        "EPSG:5070",
        "-o",
        str(out),
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    m = json.loads(out.read_text())
    assert m["iou_threshold"] == 0.6 and m["equal_area_crs"] == "EPSG:5070"
    assert m["recall"] == 0.5 and m["coverage_within_tolerance"] == 0.5
    assert m["hausdorff_mean_m"] == pytest.approx(5.0, rel=1e-5)
    assert list(m["per_stratum"]) == ["a", "b"] and list(m["per_size_class"]) == ["0-2", "2-10"]
    assert m["bootstrap"]["n_resamples"] == 20 and m["bootstrap"]["seed"] == 1
    assert m["bootstrap"]["ci"]["hausdorff_mean_m"] is not None


def test_single_polygon_layers_raise_no_numpy_deprecation_warning():
    """pyproj's scalar path turned a one-element centroid array into a DeprecationWarning."""
    import warnings

    field = gpd.GeoDataFrame(geometry=[box(0, 0, 0.0009, 0.0009)], crs="EPSG:4326")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        metrics = evaluate(field, field, boundary_tolerance_m=1.0)
    assert metrics["recall"] == 1.0 and metrics["precision"] == 1.0


def test_centroids_match_geopandas_reprojection():
    ref = gpd.GeoDataFrame(
        geometry=[box(-117.0, 36.0, -116.99, 36.01), box(149.0, -30.0, 149.01, -29.99)],
        crs="EPSG:4326",
    )
    frame = ev.evaluate_frame(ref, ref)
    assert frame["utm_epsg"].tolist() == [32611, 32755]  # 11N (Nevada), 55S (NSW)
    layer_centroids = gpd.GeoSeries(ref.geometry.to_crs("EPSG:6933").centroid).to_crs(4326)
    for lon, lat, epsg in zip(layer_centroids.x, layer_centroids.y, frame["utm_epsg"], strict=True):
        assert utm_epsg(utm_zone_for_lon(lon), lat < 0) == epsg
