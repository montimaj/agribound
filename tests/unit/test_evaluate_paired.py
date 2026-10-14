"""Tests for agribound.evaluate.paired_bootstrap (two prediction layers, one reference).

Layouts are axis-aligned squares in UTM zone 13N (EPSG:32613) on its central
meridian, as in test_evaluate.py. Its origin (500,000 m, 3,800,000 m) is a
multiple of every block side used here, and the representative point of a
square is its centre, so the block of every polygon is known. The expected
intervals are computed here from the documented procedure: the child seeds of
``numpy.random.SeedSequence(seed)``, the per-unit counts derived with
``GeoSeries.representative_point()`` and :func:`evaluate_frame`, and the rates
precision = TP / (TP + FP) and recall = TP / n_reference.
"""

from __future__ import annotations

import importlib
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from agribound.evaluate import evaluate, evaluate_frame, paired_bootstrap

ev = importlib.import_module("agribound.evaluate")

CRS = "EPSG:32613"
X0, Y0 = 500_000.0, 3_800_000.0
METRICS = ("f1", "precision", "recall")


def sq(x: float, y: float, w: float = 100.0, h: float | None = None):
    """Axis-aligned rectangle with lower-left corner (X0 + x, Y0 + y)."""
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + (w if h is None else h))


def gdf(geoms, crs=CRS, **columns) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns, geometry=list(geoms), crs=crs)


def _same(a, b) -> bool:
    """Equality of nested results (keys in the same order), with NaN equal to NaN."""
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _approx_same(a, b, where="") -> None:
    """Assert that nested results agree, floats up to 1e-12 (BLAS rounding)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), where
        for key in a:
            _approx_same(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for k, (x, y) in enumerate(zip(a, b, strict=True)):
            _approx_same(x, y, f"{where}[{k}]")
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), where
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=1e-12, abs=1e-15), where
    else:
        assert a == b, where


# ---------------------------------------------------------------------------
# The documented procedure, written out
# ---------------------------------------------------------------------------


def _unit_counts(pred, ref):
    """Per reference field (in row order), the sums (TP, reference fields, FP) of *pred*.

    A false positive counts with the reference field it overlaps (the layouts give
    each at most one); the false positives that overlap none are returned apart, as
    the sums fixed in every resample.
    """
    frame = evaluate_frame(pred, ref)
    matched_pred = set(frame.loc[frame["matched"], "pred_index"])
    unmatched = pred[~pred.index.isin(matched_pred)]
    tp = frame["matched"].to_numpy(dtype=float)
    fp_home = np.zeros(len(ref))
    fixed_fp = 0.0
    for geom in unmatched.geometry:
        overlap = np.flatnonzero(ref.geometry.intersection(geom).area.to_numpy() > 0)
        assert len(overlap) <= 1, "each false positive overlaps at most one field here"
        if len(overlap):
            fp_home[overlap[0]] += 1
        else:
            fixed_fp += 1
    return np.column_stack([tp, np.ones(len(ref)), fp_home]), np.array([0.0, 0.0, fixed_fp])


def _cells(layer, side):
    points = layer.representative_point()
    rows, cols = np.floor(points.y / side).astype(int), np.floor(points.x / side).astype(int)
    return list(zip(rows, cols, strict=True))


def _block_units(pred_a, pred_b, ref, side):
    """Per-block (TP, reference fields, FP) of each layer, blocks by row then column."""
    blocks = sorted(set(_cells(ref, side)) | set(_cells(pred_a, side)) | set(_cells(pred_b, side)))
    position = {cell: k for k, cell in enumerate(blocks)}
    ref_block = np.array([position[c] for c in _cells(ref, side)], dtype=int)
    out = []
    for pred in (pred_a, pred_b):
        frame = evaluate_frame(pred, ref)
        matched_pred = set(frame.loc[frame["matched"], "pred_index"])
        unmatched = ~pred.index.isin(matched_pred)
        pred_block = np.array([position[c] for c in _cells(pred, side)], dtype=int)
        units = np.zeros((len(blocks), 3))
        np.add.at(units[:, 0], ref_block, frame["matched"].to_numpy(dtype=float))
        np.add.at(units[:, 1], ref_block, 1.0)
        np.add.at(units[:, 2], pred_block[unmatched], 1.0)
        out.append(units)
    return out, len(blocks)


def _rates(sums):
    tp, n_ref, fp = sums
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / n_ref if n_ref else math.nan
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def _direct(units, fixed, pools, n_boot, seed):
    """Intervals of each layer's rates and of their differences A - B, resampled as documented."""
    n = len(units[0])
    values = {side: {m: [] for m in METRICS} for side in ("a", "b", "difference")}
    for child in np.random.SeedSequence(seed).spawn(n_boot):
        rng = np.random.default_rng(child)
        weights = np.zeros(n)
        for pool in pools:
            draw = rng.integers(0, len(pool), size=len(pool))
            weights[pool] = np.bincount(draw, minlength=len(pool))
        rates = [_rates(weights @ u + f) for u, f in zip(units, fixed, strict=True)]
        for m in METRICS:
            values["a"][m].append(rates[0][m])
            values["b"][m].append(rates[1][m])
            values["difference"][m].append(rates[0][m] - rates[1][m])
    out = {}
    for side, metrics in values.items():
        out[side] = {}
        for m, v in metrics.items():
            v = np.asarray(v)
            v = v[np.isfinite(v)]
            out[side][m] = np.percentile(v, [2.5, 97.5]).tolist()
    return out


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


@pytest.fixture
def two_layers():
    """Ten fields in two rows of 1 km blocks; A finds 7, B 5, each with false positives.

    A: fields 0-3, 5, 7, 8 exactly; a false positive on field 4 (IoU 0.25) and one on
    unmapped ground in block (row 1, column 5). B: fields 0, 2, 4, 6, 9; a false
    positive on field 1 (IoU 0.25) and three on unmapped ground in block (0, 1).
    """
    xs = [100, 500, 900, 1300, 1700, 2100, 2500, 2900, 3300, 3700]
    ys = [50, 50, 50, 50, 50, 1500, 1500, 1500, 1500, 1500]
    fields = [sq(x, y) for x, y in zip(xs, ys, strict=True)]
    ref = gdf(fields, basin=["n", "n", "n", "s", "s", "s", "n", "s", "n", "s"])
    a = gdf(
        [fields[k] for k in (0, 1, 2, 3, 5, 7, 8)]
        + [sq(xs[4] + 60, ys[4]), sq(5200, 1200)]  # IoU 40/160 on field 4; unmapped
    )
    b = gdf(
        [fields[k] for k in (0, 2, 4, 6, 9)]
        + [sq(xs[1] + 60, ys[1])]  # IoU 0.25 on field 1
        + [sq(1100, 400), sq(1300, 400), sq(1500, 400)]  # unmapped, block (0, 1)
    )
    return a, b, ref


# ---------------------------------------------------------------------------
# Known differences
# ---------------------------------------------------------------------------


class TestKnownDifferences:
    def test_counts_and_point_differences(self, two_layers):
        a, b, ref = two_layers
        out = paired_bootstrap(a, b, ref, n_resamples=50)
        # A: TP 7 of 10 fields, 9 predictions; B: TP 5, 9 predictions.
        counts = [(out[s]["count_tp"], out[s]["count_predicted"], out[s]["count_fp"]) for s in "ab"]
        assert counts == [(7, 9, 2), (5, 9, 4)]
        assert (out["a"]["count_fp_unassigned"], out["b"]["count_fp_unassigned"]) == (1, 3)
        assert out["difference"]["recall"] == pytest.approx(0.7 - 0.5)
        assert out["difference"]["precision"] == pytest.approx(7 / 9 - 5 / 9)
        assert out["difference"]["f1"] == pytest.approx(14 / 19 - 10 / 19)
        for m in METRICS:
            assert out["a"][m] - out["b"][m] == out["difference"][m]

    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"matching": "many_to_one", "iou_threshold": 0.2},
            {"boundary_tolerance_m": 5.0, "strata": "basin"},
            {"block_size_m": 1000.0, "boundary_tolerance_m": 5.0, "strata": "basin"},
        ],
    )
    def test_point_estimates_are_those_of_evaluate(self, two_layers, options):
        a, b, ref = two_layers
        out = paired_bootstrap(a, b, ref, n_resamples=20, **options)
        rules = {k: v for k, v in options.items() if k != "block_size_m"}
        ea, eb = evaluate(a, ref, **rules), evaluate(b, ref, **rules)
        names = METRICS + (("boundary_f1",) if "boundary_tolerance_m" in options else ())
        assert list(out["difference"]) == list(names)
        for m in names:
            assert out["a"][m] == ea[m] and out["b"][m] == eb[m]
            assert out["difference"][m] == ea[m] - eb[m]
        for key in ("count_reference", "count_predicted", "count_tp", "count_fp", "count_fn"):
            assert out["a"][key] == ea[key] and out["b"][key] == eb[key]
        assert out["iou_threshold"] == ea["iou_threshold"] and out["matching"] == ea["matching"]
        assert out["equal_area_crs"] == ea["equal_area_crs"] == "EPSG:6933"
        if "strata" in options:
            assert out["strata_column"] == "basin" and list(out["per_stratum"]) == ["n", "s"]
            for key, group in out["per_stratum"].items():
                assert group["n"] == ea["per_stratum"][key]["n"] == 5
                for m in names:
                    assert group["a"][m] == ea["per_stratum"][key][m]
                    expected = ea["per_stratum"][key][m] - eb["per_stratum"][key][m]
                    assert group["difference"][m] == expected
                assert "count_fp_unassigned" not in group["a"]

    def test_field_intervals_match_the_documented_procedure(self, two_layers):
        a, b, ref = two_layers
        n_boot, seed = 400, 5
        out = paired_bootstrap(a, b, ref, n_resamples=n_boot, seed=seed)
        units_a, fixed_a = _unit_counts(a, ref)
        units_b, fixed_b = _unit_counts(b, ref)
        assert fixed_a.tolist() == [0, 0, 1] and fixed_b.tolist() == [0, 0, 3]
        expected = _direct(
            [units_a, units_b], [fixed_a, fixed_b], [np.arange(len(ref))], n_boot, seed
        )
        for m in METRICS:
            assert out["ci"][m] == pytest.approx(expected["difference"][m], rel=1e-12)
            assert out["a"]["ci"][m] == pytest.approx(expected["a"][m], rel=1e-12)
            assert out["b"]["ci"][m] == pytest.approx(expected["b"][m], rel=1e-12)
        assert out["resampling"] == "reference fields with replacement"
        assert (out["block_size_m"], out["block_crs"], out["n_blocks"]) == (None, None, None)

    def test_block_intervals_match_the_documented_procedure(self, two_layers):
        a, b, ref = two_layers
        n_boot, seed = 400, 5
        out = paired_bootstrap(a, b, ref, n_resamples=n_boot, seed=seed, block_size_m=1000.0)
        units, n_blocks = _block_units(a, b, ref, 1000.0)
        # Field centres x = 150-1750 (row 0) and 2150-3750 (row 1): blocks (0, 0) x3,
        # (0, 1) x2, (1, 2) x3, (1, 3) x2. A's unmapped false positive adds (1, 5); B's
        # three lie in (0, 1), and its false positive on field 1 (x = 610) in (0, 0).
        assert n_blocks == out["n_blocks"] == 5
        assert units[0][:, 1].tolist() == [3, 2, 3, 2, 0]  # reference fields per block
        assert units[1][:, 2].tolist() == [1, 3, 0, 0, 0]  # B's false positives
        assert units[0][:, 2].tolist() == [0, 1, 0, 0, 1]  # A's: on field 4, unmapped
        zeros = np.zeros(3)
        expected = _direct(units, [zeros, zeros], [np.arange(n_blocks)], n_boot, seed)
        for m in METRICS:
            assert out["ci"][m] == pytest.approx(expected["difference"][m], rel=1e-12)
            assert out["a"]["ci"][m] == pytest.approx(expected["a"][m], rel=1e-12)
            assert out["b"]["ci"][m] == pytest.approx(expected["b"][m], rel=1e-12)
        assert out["resampling"] == "1000 m blocks with replacement"
        assert out["block_size_m"] == 1000.0 and out["block_crs"] == "EPSG:32613"

    def test_matching_rules_change_the_difference(self):
        # Two fields (100 m and 60 m wide); A merges them, B draws both.
        ref = gdf([sq(0, 0), sq(100, 0, 60, 100)])
        a = gdf([sq(0, 0, 160, 100)])
        b = gdf([sq(0, 0), sq(100, 0, 60, 100)])
        one = paired_bootstrap(a, b, ref, iou_threshold=0.3, n_resamples=20)
        many = paired_bootstrap(
            a, b, ref, iou_threshold=0.3, matching="many_to_one", n_resamples=20
        )
        assert one["difference"]["recall"] == pytest.approx(-0.5)
        assert many["difference"]["recall"] == 0.0  # the merged polygon matches both fields
        assert many["matching"] == "many_to_one" and one["matching"] == "one_to_one"
        # At IoU >= 0.7 the merged polygon matches neither field.
        strict = paired_bootstrap(a, b, ref, iou_threshold=0.7, n_resamples=20)
        assert strict["difference"]["recall"] == -1.0 and strict["iou_threshold"] == 0.7

    def test_boundary_f1_needs_a_tolerance(self, two_layers):
        a, b, ref = two_layers
        assert "boundary_f1" not in paired_bootstrap(a, b, ref, n_resamples=10)["ci"]
        out = paired_bootstrap(a, b, ref, n_resamples=50, boundary_tolerance_m=5.0)
        assert out["boundary_tolerance_m"] == 5.0
        lower, upper = out["ci"]["boundary_f1"]
        assert lower <= out["difference"]["boundary_f1"] <= upper


# ---------------------------------------------------------------------------
# Identical layers
# ---------------------------------------------------------------------------


class TestIdenticalLayers:
    @pytest.mark.parametrize("block_size_m", [None, 1000.0])
    def test_zero_difference_and_zero_width_intervals(self, two_layers, block_size_m):
        a, _, ref = two_layers
        out = paired_bootstrap(
            a,
            a.copy(),
            ref,
            n_resamples=300,
            block_size_m=block_size_m,
            strata="basin",
            boundary_tolerance_m=5.0,
        )
        groups = [out] + list(out["per_stratum"].values())
        for group in groups:
            for m in METRICS + ("boundary_f1",):
                assert group["difference"][m] == 0.0
                assert group["ci"][m] == [0.0, 0.0]
                assert group["a"]["ci"][m] == group["b"]["ci"][m]
        # The layer's own intervals are not degenerate.
        assert out["a"]["ci"]["f1"][0] < out["a"]["ci"]["f1"][1]


# ---------------------------------------------------------------------------
# Reference fields: the resamples of evaluate()
# ---------------------------------------------------------------------------


class TestFieldResampling:
    @pytest.mark.parametrize("strata", [None, "basin"])
    def test_each_layer_gets_evaluates_intervals(self, two_layers, strata):
        a, b, ref = two_layers
        options = {"strata": strata, "boundary_tolerance_m": 5.0}
        out = paired_bootstrap(a, b, ref, n_resamples=250, seed=9, **options)
        for side, layer in (("a", a), ("b", b)):
            m = evaluate(layer, ref, bootstrap=250, bootstrap_seed=9, **options)
            for name, bounds in out[side]["ci"].items():
                assert bounds == m["bootstrap"]["ci"][name], (side, name)
            if strata is not None:
                for key, group in out["per_stratum"].items():
                    for name, bounds in group[side]["ci"].items():
                        assert bounds == m["per_stratum"][key]["ci"][name], (side, key, name)
        expected = "reference fields with replacement" + (" within strata" if strata else "")
        assert out["resampling"] == expected

    def test_strata_are_resampled_at_their_own_size(self):
        # Stratum a: 3 fields both layers find; stratum b: 1 field only A finds. Resampled
        # within strata, recall is 1.0 and 0.75 in every resample.
        ref = gdf([sq(0, 0), sq(200, 0), sq(400, 0), sq(600, 0)], s=["a", "a", "a", "b"])
        a = gdf([sq(0, 0), sq(200, 0), sq(400, 0), sq(600, 0)])
        b = gdf([sq(0, 0), sq(200, 0), sq(400, 0)])
        stratified = paired_bootstrap(a, b, ref, strata="s", n_resamples=200)
        assert stratified["ci"]["recall"] == pytest.approx([0.25, 0.25])
        assert stratified["per_stratum"]["b"]["ci"]["recall"] == pytest.approx([1.0, 1.0])
        pooled = paired_bootstrap(a, b, ref, n_resamples=200)["ci"]["recall"]
        assert pooled[0] < 0.25 < pooled[1]

    def test_false_positives_on_unmapped_ground(self):
        """Critique F19: reference-field resamples keep them fixed; block resamples do not.

        Six fields, two in each of three 1 km blocks. A finds all six and draws four
        polygons in a fourth block that no field reaches; B finds the six. Resampling
        fields, A's precision is 6 / 10 in every resample. Resampling the four blocks,
        the empty block is drawn 0 to 4 times.
        """
        ref = gdf([sq(1000 * k + dx, 0) for k in range(3) for dx in (100, 600)])
        unmapped = [sq(5000 + 200 * j, 0) for j in range(4)]
        a = gdf(list(ref.geometry) + unmapped)
        b = ref.copy()
        fields = paired_bootstrap(a, b, ref, n_resamples=400)
        assert fields["a"]["count_fp_unassigned"] == 4
        assert fields["ci"]["precision"] == pytest.approx([-0.4, -0.4])
        blocks = paired_bootstrap(a, b, ref, n_resamples=400, block_size_m=1000.0)
        assert blocks["n_blocks"] == 4
        lower, upper = blocks["ci"]["precision"]
        # No draw of the empty block: difference 0 (in (3/4)^4 of resamples); once: -0.4;
        # twice: 4 / 12 - 1; three times: 2 / 14 - 1. Four times: B undefined.
        assert upper == 0.0 and lower < -0.4
        assert lower >= 2 / 14 - 1 - 1e-12
        assert blocks["n_undefined"].get("precision", 0) < 400 * 0.02  # (1/4)^4 expected


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class TestBlocks:
    def test_points_on_block_edges_go_to_the_next_block(self):
        # Centres on x = 1000 and y = 1000 belong to the blocks east and north of them.
        ref = gdf([sq(450, 450), sq(950, 450), sq(450, 950), sq(950, 950)])
        out = paired_bootstrap(ref, ref.copy(), ref, n_resamples=10, block_size_m=1000.0)
        assert out["n_blocks"] == 4
        shifted = gdf([sq(449, 449), sq(949, 449), sq(449, 949), sq(949, 949)])
        out = paired_bootstrap(
            shifted, shifted.copy(), shifted, n_resamples=10, block_size_m=1000.0
        )
        assert out["n_blocks"] == 1  # every centre at most 999 m from the origin

    def test_a_polygon_counts_in_the_block_of_its_representative_point(self):
        # An L-shaped false positive: a square and a long thin arm. Its centroid
        # (x = 804) lies in block 0, its representative point (x = 1300) in block 1.
        arm = sq(500, 0, 400, 400).union(sq(900, 190, 1200, 20))
        # A strip that overlaps the field (so the field is its home) but whose
        # representative point (x = 1100) lies in block 1.
        strip = sq(150, 620, 1900, 60)
        a = gdf([sq(100, 600), arm, strip])
        points = a.representative_point()
        assert (points.x - X0).tolist() == pytest.approx([150.0, 1300.0, 1100.0])
        assert a.centroid.iloc[1].x - X0 == pytest.approx(804.35, abs=0.01)
        ref = gdf([sq(100, 600)])
        out = paired_bootstrap(a, ref.copy(), ref, n_resamples=200, block_size_m=1000.0)
        assert out["n_blocks"] == 2 and out["a"]["count_fp_unassigned"] == 1
        # Block 0 holds the field (found by both), block 1 A's two false positives:
        # each counts in its own block, not in its home field's.
        units, _ = _block_units(a, ref, ref, 1000.0)
        assert units[0].tolist() == [[1, 1, 0], [0, 0, 2]]
        expected = _direct(units, [np.zeros(3)] * 2, [np.arange(2)], 200, 42)
        for m in METRICS:
            assert out["ci"][m] == pytest.approx(expected["difference"][m], rel=1e-12)

    def test_a_matched_prediction_counts_with_its_reference_field(self):
        # Field 0's centre is at x = 990 (block 0). A's prediction of it, 20 m east, has
        # its centre in block 1; A2's, 20 m west, in block 0; both match at IoU 2/3.
        # Precision is TP / (TP + FP), TP counted with the field, so the two give the
        # same intervals (block 1 is occupied by field 1 either way).
        ref = gdf([sq(940, 0), sq(1500, 0), sq(100, 0)])
        a = gdf([sq(960, 0), sq(1500, 0), sq(3000, 0)])
        a2 = gdf([sq(920, 0), sq(1500, 0), sq(3000, 0)])
        b = gdf([sq(100, 0), sq(3100, 400)])
        east = paired_bootstrap(a, b, ref, n_resamples=300, block_size_m=1000.0)
        west = paired_bootstrap(a2, b, ref, n_resamples=300, block_size_m=1000.0)
        assert east["n_blocks"] == west["n_blocks"] == 3
        assert east["a"]["count_tp"] == west["a"]["count_tp"] == 2
        assert east["ci"] == west["ci"] and east["a"]["ci"] == west["a"]["ci"]

    def test_blocks_are_numbered_by_row_then_column(self):
        # Blocks (0, 2), (0, 3), (1, 0), (1, 1): numbered by row from the south, (1, 0)
        # is the third; by column it would be the first, and the draws would differ.
        ref = gdf(
            [sq(2100, 100), sq(2600, 100), sq(3100, 100), sq(100, 1100), sq(600, 1100)]
            + [sq(1100, 1100)]
        )
        a = gdf([sq(2100, 100), sq(3100, 100), sq(100, 1100), sq(1100, 1100), sq(1300, 1300)])
        b = gdf([sq(2600, 100), sq(100, 1100), sq(600, 1100)])
        out = paired_bootstrap(a, b, ref, n_resamples=300, block_size_m=1000.0)
        units, n_blocks = _block_units(a, b, ref, 1000.0)
        assert n_blocks == out["n_blocks"] == 4
        assert units[0][:, 1].tolist() == [2, 1, 2, 1]  # (0, 2), (0, 3), (1, 0), (1, 1)
        expected = _direct(units, [np.zeros(3)] * 2, [np.arange(n_blocks)], 300, 42)
        for m in METRICS:
            assert out["ci"][m] == pytest.approx(expected["difference"][m], rel=1e-12)

    def test_only_occupied_blocks_are_resampled(self, two_layers):
        a, b, ref = two_layers
        # 500 m blocks: field centres in columns 0, 1, 1, 2, 3 (row 0) and 4, 5, 5, 6, 7
        # (row 3), 8 blocks; A's unmapped false positive adds (2, 10). B's three (centres
        # at y = 450) fall in the blocks of fields 3 and 4.
        out = paired_bootstrap(a, b, ref, n_resamples=20, block_size_m=500.0)
        _, n_blocks = _block_units(a, b, ref, 500.0)
        assert out["n_blocks"] == n_blocks == 9
        # A polygon 90 km away adds its own block and none of the empty ones between.
        far = gdf(list(a.geometry) + [sq(90_000, 90_000)])
        assert paired_bootstrap(far, b, ref, n_resamples=20, block_size_m=500.0)["n_blocks"] == 10

    def test_block_crs(self, two_layers):
        a, b, ref = two_layers
        base = paired_bootstrap(a, b, ref, n_resamples=100, block_size_m=1000.0)
        assert base["block_crs"] == "EPSG:32613"  # the UTM zone of the layers' centre
        same = paired_bootstrap(a, b, ref, n_resamples=100, block_size_m=1000.0, block_crs=32613)
        assert _same(same, base)
        # The representative points are found in each layer's own CRS and projected.
        geographic = paired_bootstrap(
            a.to_crs(4326), b, ref.to_crs(4326), n_resamples=100, block_size_m=1000.0
        )
        assert geographic["n_blocks"] == base["n_blocks"]
        _approx_same(geographic["ci"], base["ci"])
        # Another lattice: EPSG:5070 (Albers) cuts the plane elsewhere.
        albers = paired_bootstrap(a, b, ref, n_resamples=100, block_size_m=1000.0, block_crs=5070)
        assert albers["block_crs"] == "EPSG:5070"
        assert albers["difference"] == base["difference"]
        assert albers["ci"] != base["ci"]

    def test_row_order_does_not_matter(self, two_layers):
        a, b, ref = two_layers
        out = paired_bootstrap(a, b, ref, n_resamples=200, block_size_m=1000.0, strata="basin")
        rng = np.random.default_rng(0)
        shuffled = [layer.iloc[rng.permutation(len(layer))] for layer in (a, b, ref)]
        again = paired_bootstrap(*shuffled, n_resamples=200, block_size_m=1000.0, strata="basin")
        _approx_same(again, out)

    def test_one_block_gives_the_point_estimates(self, two_layers):
        a, b, ref = two_layers
        out = paired_bootstrap(
            a, b, ref, n_resamples=50, block_size_m=1e6, boundary_tolerance_m=5.0
        )
        assert out["n_blocks"] == 1
        for m in METRICS + ("boundary_f1",):
            assert out["ci"][m] == pytest.approx([out["difference"][m]] * 2, rel=1e-12)
            assert out["a"]["ci"][m] == pytest.approx([out["a"][m]] * 2, rel=1e-12)

    def test_strata_group_but_do_not_split_the_blocks(self, two_layers):
        a, b, ref = two_layers
        plain = paired_bootstrap(a, b, ref, n_resamples=200, block_size_m=1000.0)
        grouped = paired_bootstrap(a, b, ref, n_resamples=200, block_size_m=1000.0, strata="basin")
        assert grouped["resampling"] == plain["resampling"] == "1000 m blocks with replacement"
        for key in ("difference", "ci", "n_undefined", "a", "b"):
            assert _same(grouped[key], plain[key])
        assert grouped["count_reference_no_stratum"] == 0
        north = grouped["per_stratum"]["n"]
        lower, upper = north["ci"]["f1"]
        assert lower <= north["difference"]["f1"] <= upper


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


class TestSeeds:
    @pytest.mark.parametrize("block_size_m", [None, 1000.0])
    def test_deterministic_and_batch_invariant(self, two_layers, monkeypatch, block_size_m):
        a, b, ref = two_layers
        kwargs = {"n_resamples": 300, "seed": 3, "block_size_m": block_size_m, "strata": "basin"}
        first = paired_bootstrap(a, b, ref, **kwargs)
        assert _same(paired_bootstrap(a, b, ref, **kwargs), first)
        monkeypatch.setattr(ev, "_BOOTSTRAP_CHUNK_CELLS", 1)  # one resample per batch
        _approx_same(paired_bootstrap(a, b, ref, **kwargs), first)
        monkeypatch.undo()
        other = paired_bootstrap(a, b, ref, **{**kwargs, "seed": 4})
        assert other["seed"] == 4 and other["ci"] != first["ci"]
        assert other["difference"] == first["difference"]

    def test_defaults(self, two_layers):
        a, b, ref = two_layers
        out = paired_bootstrap(a, b, ref)
        assert (out["n_resamples"], out["seed"], out["method"]) == (2000, 42, "percentile")
        assert out["confidence_level"] == ev.BOOTSTRAP_CONFIDENCE_LEVEL == 0.95
        assert out["proj_network"] is ev.proj_network_enabled()


# ---------------------------------------------------------------------------
# ignore
# ---------------------------------------------------------------------------


class TestIgnore:
    @pytest.mark.parametrize("block_size_m", [None, 1000.0])
    def test_same_as_without_the_ignored_predictions(self, two_layers, block_size_m):
        a, b, ref = two_layers
        # Ground left out of the score: A's unmapped false positive and one of B's.
        ignored = gdf([sq(5150, 1150, 200), sq(1050, 350, 200)])
        options = {"n_resamples": 200, "block_size_m": block_size_m, "strata": "basin"}
        out = paired_bootstrap(a, b, ref, ignore=ignored, **options)
        assert (out["a"]["count_predicted_ignored"], out["b"]["count_predicted_ignored"]) == (1, 1)
        keep = [~layer.representative_point().intersects(ignored.union_all()) for layer in (a, b)]
        expected = paired_bootstrap(a[keep[0]], b[keep[1]], ref, **options)
        for side in ("a", "b"):
            out[side].pop("count_predicted_ignored")
        assert _same(out, expected)
        assert "count_predicted_ignored" not in expected["a"]


# ---------------------------------------------------------------------------
# Empty layers and arguments
# ---------------------------------------------------------------------------


class TestEmptyAndArguments:
    def test_no_reference_fields_to_resample(self, caplog):
        a = gdf([sq(0, 0)])
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            out = paired_bootstrap(a, gdf([]), gdf([]), n_resamples=20)
        assert "No reference fields to resample" in caplog.text
        assert all(math.isnan(v) for m in METRICS for v in out["ci"][m])
        assert out["n_undefined"] == {m: 20 for m in METRICS}
        # Blocks hold A's polygon: precision is defined for A only.
        blocks = paired_bootstrap(a, gdf([]), gdf([]), n_resamples=20, block_size_m=1000.0)
        assert blocks["n_blocks"] == 1
        assert blocks["a"]["ci"]["precision"] == [0.0, 0.0]
        assert blocks["n_undefined"]["precision"] == 20

    def test_an_empty_layer(self, two_layers):
        a, _, ref = two_layers
        out = paired_bootstrap(a, gdf([]), ref, n_resamples=30, block_size_m=1000.0)
        assert out["b"]["count_predicted"] == 0 and out["b"]["precision"] == 0.0
        assert out["difference"]["precision"] == out["a"]["precision"]
        assert out["n_undefined"]["precision"] == 30  # B has no prediction to rate
        assert out["b"]["ci"]["recall"] == [0.0, 0.0]

    def test_nothing_at_all(self, caplog):
        with caplog.at_level("WARNING", logger="agribound.evaluate"):
            out = paired_bootstrap(gdf([]), gdf([]), gdf([]), n_resamples=5, block_size_m=1000.0)
        assert "No blocks to resample" in caplog.text
        assert out["n_blocks"] == 0 and out["block_crs"] is None
        assert out["difference"]["f1"] == 0.0  # 1.0 - 1.0, evaluate()'s convention

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_resamples": 0}, "n_resamples must be at least 1"),
            ({"n_resamples": -1}, "n_resamples must be a non-negative integer"),
            ({"n_resamples": 2.5}, "n_resamples must be a non-negative integer"),
            ({"n_resamples": True}, "n_resamples must be a non-negative integer"),
            ({"seed": -1}, "seed must be a non-negative integer"),
            ({"seed": 1.0}, "seed must be a non-negative integer"),
            ({"block_size_m": 0}, "block_size_m must be a positive finite number"),
            ({"block_size_m": float("nan")}, "block_size_m must be a positive finite number"),
            ({"block_size_m": "1 km"}, "block_size_m must be a positive number"),
            ({"block_size_m": True}, "block_size_m must be a positive number"),
            ({"block_crs": 32613}, "pass block_size_m with it"),
            ({"block_size_m": 1000.0, "block_crs": 4326}, "block_crs must be a projected CRS"),
            ({"block_size_m": 1000.0, "block_crs": "EPSG:2263"}, "block_crs must have axes in"),
            ({"block_size_m": 1000.0, "block_crs": "no such crs"}, "is not a valid CRS"),
            ({"iou_threshold": 0.0}, "iou_threshold must be in"),
            ({"matching": "best"}, "matching must be one of"),
            ({"boundary_tolerance_m": -1.0}, "boundary_tolerance_m must be"),
            ({"equal_area_crs": 4326}, "equal_area_crs must be a projected CRS"),
        ],
    )
    def test_bad_arguments_are_refused_before_scoring(self, two_layers, monkeypatch, kwargs, match):
        a, b, ref = two_layers

        def fail(*args, **kw):
            raise AssertionError("scored before the arguments were checked")

        monkeypatch.setattr(ev, "_run", fail)
        monkeypatch.setattr(ev, "_prepare_layer", fail)
        with pytest.raises(ValueError, match=match):
            paired_bootstrap(a, b, ref, **kwargs)

    def test_bad_layers(self, two_layers):
        a, b, ref = two_layers
        with pytest.raises(TypeError, match="predicted_b must be a GeoDataFrame or GeoSeries"):
            paired_bootstrap(a, list(b.geometry), ref)
        with pytest.raises(ValueError, match="predicted_a has no CRS"):
            paired_bootstrap(gpd.GeoSeries([sq(0, 0)]), b, ref)
        with pytest.raises(ValueError, match="strata column 'nope' not found"):
            paired_bootstrap(a, b, ref, strata="nope")
        with pytest.raises(ValueError, match="ignore has no CRS"):
            paired_bootstrap(a, b, ref, ignore=gpd.GeoSeries([sq(0, 0)]))

    def test_a_block_crs_that_cannot_hold_the_points(self, two_layers):
        a, b, ref = two_layers
        # An orthographic view of the northern hemisphere has no coordinates for
        # points south of the equator (PROJ returns inf).
        north_view = "+proj=ortho +lat_0=90 +lon_0=0 +datum=WGS84 +units=m +no_defs"
        south = [layer.to_crs(4326).translate(yoff=-70.0) for layer in (a, b, ref)]
        with pytest.raises(ValueError, match="gives no finite coordinates.*reference polygons"):
            paired_bootstrap(*south, n_resamples=5, block_size_m=1000.0, block_crs=north_view)


def test_the_api_page_lists_every_public_name():
    page = Path(__file__).resolve().parents[2] / "docs" / "api" / "evaluate.md"
    members = re.findall(r"^\s+- (\w+)$", page.read_text(encoding="utf-8"), flags=re.M)
    assert sorted(members) == sorted(ev.__all__)
    assert "paired_bootstrap" in members
