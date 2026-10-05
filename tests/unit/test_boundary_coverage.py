"""Coverage must restrict real lines, without adding polygon clipping edges."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from agribound.evaluate import evaluate, evaluate_frame


def polygons(*geometries):
    return gpd.GeoDataFrame(geometry=list(geometries), crs=32612)


def test_mask_does_not_manufacture_edges_or_change_iou():
    ref = polygons(box(500000, 4500000, 500100, 4500100))
    pred = polygons(box(500000, 4500000, 500100, 4500200))
    mask = polygons(box(500020, 4500020, 500080, 4500080))
    original = evaluate(pred, ref, boundary_tolerance_m=10)
    restricted = evaluate(pred, ref, boundary_tolerance_m=10, boundary_mask=mask)
    assert restricted["iou_mean"] == pytest.approx(original["iou_mean"])
    assert restricted["f1"] == original["f1"]
    # Both boundaries are outside the mask. Polygon clipping would spuriously
    # create identical rectangular boundaries and report perfect agreement.
    assert restricted["boundary_precision"] == 0
    assert restricted["boundary_recall"] == 0
    frame = evaluate_frame(pred, ref, boundary_tolerance_m=10, boundary_mask=mask)
    assert frame.perimeter_m.iloc[0] == 0


def test_unknown_edges_are_excluded_and_crs_is_respected():
    ref = polygons(box(500000, 4500000, 500100, 4500100))
    pred = polygons(box(500000, 4500000, 500200, 4500100))
    mask = polygons(box(499990, 4499990, 500050, 4500110)).to_crs(4326)
    result = evaluate(pred, ref, boundary_tolerance_m=10, boundary_mask=mask, size_bins=[0, 2, 10])
    assert result["boundary_precision"] == pytest.approx(1, abs=1e-6)
    assert result["boundary_recall"] == pytest.approx(1, abs=1e-6)
    assert result["per_size_class"]["0-2"]["boundary_f1"] == pytest.approx(1, abs=1e-6)
    assert np.isnan(result["per_size_class"]["2-10"]["boundary_f1"])
    frame = evaluate_frame(pred, ref, boundary_tolerance_m=10, boundary_mask=mask)
    assert frame.perimeter_m.iloc[0] == pytest.approx(200, abs=0.1)


def test_mask_holes_exclude_real_segments():
    ref = polygons(box(500000, 4500000, 500100, 4500100))
    mask = polygons(
        Polygon(
            box(499990, 4499990, 500110, 4500110).exterior,
            [box(499995, 4500040, 500005, 4500060).exterior],
        )
    )
    frame = evaluate_frame(ref, ref, boundary_tolerance_m=10, boundary_mask=mask)
    assert frame.perimeter_m.iloc[0] == pytest.approx(380, abs=0.1)
    assert frame.boundary_recall.iloc[0] == pytest.approx(1)


def test_coincident_mask_border_preserves_perimeter_after_reprojection():
    # Original and mask take different projection paths internally. Exact
    # line intersection without a numerical guard loses coincident segments.
    ref = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (-122.04, 39.08),
                    (-122.035, 39.085),
                    (-122.031, 39.085),
                    (-122.03, 39.081),
                    (-122.036, 39.078),
                    (-122.04, 39.08),
                ]
            )
        ],
        crs=4326,
    ).to_crs(3310)
    original = evaluate_frame(ref, ref, boundary_tolerance_m=10)
    masked = evaluate_frame(ref, ref, boundary_tolerance_m=10, boundary_mask=ref)
    assert masked.perimeter_m.iloc[0] == pytest.approx(original.perimeter_m.iloc[0], abs=0.01)
    assert masked.boundary_recall.iloc[0] == pytest.approx(1)


def test_long_geographic_mask_edges_use_original_boundary_projection():
    # Densifying the coverage through an equal-area CRS bends long geographic
    # edges relative to the evaluator's undensified UTM boundary segments.
    ref = gpd.GeoDataFrame(
        geometry=[box(-112.62, 45.20, -112.56, 45.24)],
        crs=4326,
    )
    original = evaluate_frame(ref, ref, boundary_tolerance_m=10)
    masked = evaluate_frame(ref, ref, boundary_tolerance_m=10, boundary_mask=ref)
    assert masked.perimeter_m.iloc[0] == pytest.approx(original.perimeter_m.iloc[0], abs=0.01)


def test_component_coverage_preserves_overlapping_and_shared_borders():
    ref = gpd.GeoDataFrame(
        geometry=[box(-72.96, 45.74, -72.93, 45.78), box(-72.94, 45.74, -72.91, 45.78)],
        crs=4269,
    )
    original = evaluate_frame(ref, ref, boundary_tolerance_m=10)
    masked = evaluate_frame(ref, ref, boundary_tolerance_m=10, boundary_mask=ref)
    assert masked.perimeter_m.to_numpy() == pytest.approx(original.perimeter_m.to_numpy(), abs=0.01)


@pytest.mark.parametrize(
    "mask", [polygons(), polygons(LineString([(0, 0), (1, 1)])), polygons(Polygon())]
)
def test_bad_coverage_is_rejected(mask):
    ref = polygons(box(500000, 4500000, 500100, 4500100))
    with pytest.raises(ValueError, match="boundary_mask"):
        evaluate(ref, ref, boundary_tolerance_m=10, boundary_mask=mask)


def test_mask_requires_crs_and_tolerance():
    ref = polygons(box(500000, 4500000, 500100, 4500100))
    with pytest.raises(ValueError, match="requires boundary_tolerance"):
        evaluate(ref, ref, boundary_mask=ref)
    with pytest.raises(ValueError, match="no CRS"):
        evaluate(
            ref, ref, boundary_tolerance_m=10, boundary_mask=ref.set_crs(None, allow_override=True)
        )
