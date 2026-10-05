"""Coverage-aware controls for comparisons with published FTW predictions.

Accuracy requires reference labels. Prediction-to-prediction agreement has
separate names and denominators, even though both reuse the same evaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import box

from agribound.evaluate import evaluate
from agribound.ftw_arrow import PUBLISHED_YEARS, row_years


def inside_aoi(frame, bbox):
    if frame.crs is None:
        raise ValueError("Polygon CRS is required")
    points = frame.to_crs(4326).geometry.representative_point()
    return frame.loc[points.within(box(*bbox)).to_numpy()].copy()


def validate_ftw_snapshot(frame, year, info):
    """Do not confuse missing partitions or unsupported years with zero fields."""
    if year not in PUBLISHED_YEARS:
        raise ValueError("Unsupported published FTW year")
    if frame.crs is None:
        raise ValueError("FTW snapshot lacks CRS")
    opened = info.get("n_files_opened", info.get("n_tiles_read", 0))
    if not opened:
        raise ValueError("No published source coverage opened; unmeasured, not empty predictions")
    if info.get("max_features") is not None:
        raise ValueError("Truncated FTW queries cannot be evaluated")
    if not frame.empty:
        years = row_years(frame)
        if years is None or years.isna().any() or not (years.astype(int) == year).all():
            raise ValueError("FTW snapshot has missing or mixed prediction years")
        if not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError("FTW requires polygon geometries")


def ftw_variant(frame, name, *, threshold=69, min_area_m2=1000):
    """Derive sensitivities without altering the cached product geometry."""
    keep = pd.Series(True, index=frame.index)
    if name in ("ftw_conf69", "ftw_conf69_known"):
        if "confidence" not in frame:
            raise ValueError("Confidence sensitivity unavailable without confidence column")
        confidence = pd.to_numeric(frame.confidence, errors="raise")
        if not confidence.dropna().between(0, 100).all():
            raise ValueError("FTW confidence must be on the 0-100 scale")
        keep = confidence.ge(threshold)
        if name == "ftw_conf69":
            keep |= confidence.isna()
    elif name == "ftw_area1000":
        keep = frame.to_crs(6933).area.ge(min_area_m2)
    elif name != "ftw":
        raise ValueError("Unknown FTW sensitivity")
    output = frame.loc[keep].copy()
    input_area = float(frame.to_crs(6933).area.sum() / 10000)
    retained_area = float(output.to_crs(6933).area.sum() / 10000)
    return output, {
        "variant": name,
        "n_input": len(frame),
        "n_retained": len(output),
        "n_excluded": len(frame) - len(output),
        "polygon_area_ha_input": input_area,
        "polygon_area_ha_retained": retained_area,
        "polygon_area_ha_excluded": input_area - retained_area,
        "n_null_confidence_input": int(frame.confidence.isna().sum())
        if "confidence" in frame
        else len(frame),
        "n_null_confidence_retained": int(output.confidence.isna().sum())
        if "confidence" in output
        else len(output),
        "geometry_modified": False,
    }


def select_product_coverage(frame, coverage, selector):
    """Repair working masks after reprojection, preserving raw vector geometry.

    A valid projected polygon can acquire tiny self-intersections on geographic
    reprojection. Selection still delegates to the established whole-field rule.
    """
    if frame.empty:
        if frame.crs is None or coverage.crs is None:
            raise ValueError("Predictions and coverage must have a CRS")
        return frame.copy(), {
            "criterion": "Whole representative points in fixed mask",
            "n_predictions_in_aoi": 0,
            "n_predictions_evaluated": 0,
            "n_predictions_unknown_coverage": 0,
            "coverage_repaired_after_projection": 0,
        }
    projected = coverage.to_crs(frame.crs)
    repaired = int((~projected.is_valid).sum())
    projected.geometry = projected.geometry.make_valid(method="structure")
    selected, diagnostic = selector(frame, projected)
    diagnostic["coverage_repaired_after_projection"] = repaired
    return selected, diagnostic


def product_agreement(left, ftw, *, tolerance_m, size_bins, boundary_mask=None):
    """FTW is a peer prediction product here, never truth.

    Forward polygon correspondence uses FTW as the matching anchor. Reverse
    statistics describe fragmentation with the roles exchanged. The symmetric
    boundary score is the mean of directional F1 computations, including any
    numerical differences in sampling and projection.
    """
    kwargs = dict(
        size_bins=size_bins,
        boundary_tolerance_m=tolerance_m,
        boundary_mask=boundary_mask,
        bootstrap=0,
    )
    forward = evaluate(left, ftw, **kwargs)
    reverse = evaluate(ftw, left, **kwargs)
    scores = [forward["boundary_f1"], reverse["boundary_f1"]]
    finite = [v for v in scores if np.isfinite(v)]
    # Descriptive many-to-many adjacency, separate from one-to-one detection.
    # Require intersection >=10% of the smaller polygon to suppress tiny slivers.
    left_m, ftw_m = left.to_crs(6933), ftw.to_crs(6933)
    invalid_left, invalid_ftw = int((~left_m.is_valid).sum()), int((~ftw_m.is_valid).sum())
    left_m.geometry = left_m.geometry.make_valid(method="structure")
    ftw_m.geometry = ftw_m.geometry.make_valid(method="structure")
    left_degree = np.zeros(len(left_m), dtype=int)
    ftw_degree = np.zeros(len(ftw_m), dtype=int)
    for i, geom in enumerate(left_m.geometry):
        for j in ftw_m.sindex.query(geom, predicate="intersects"):
            other = ftw_m.geometry.iloc[j]
            denominator = min(geom.area, other.area)
            if denominator > 0 and geom.intersection(other).area / denominator >= 0.1:
                left_degree[i] += 1
                ftw_degree[j] += 1
    return {
        "track": "prediction_agreement",
        "boundary_tolerance_m": tolerance_m,
        "n_left": forward["count_predicted"],
        "n_ftw": forward["count_reference"],
        "n_corresponding": forward["count_tp"],
        "ftw_polygons_split_across_left": int((ftw_degree >= 2).sum()),
        "left_polygons_merging_ftw": int((left_degree >= 2).sum()),
        "split_merge_overlap_fraction_min": 0.1,
        "left_invalid_geometries_repaired_in_memory": invalid_left,
        "ftw_invalid_geometries_repaired_in_memory": invalid_ftw,
        "left_correspondence_fraction": forward["precision"],
        "ftw_correspondence_fraction": forward["recall"],
        "correspondence_f1": forward["f1"],
        "matched_iou": forward["iou_mean"],
        "left_boundary_near_ftw": forward["boundary_precision"],
        "ftw_boundary_near_left": forward["boundary_recall"],
        "symmetric_boundary_agreement_f1": float(np.mean(finite)) if finite else np.nan,
        "left_fragmentation_relative_to_ftw": forward["oversegmentation_mean"],
        "left_merging_relative_to_ftw": forward["undersegmentation_mean"],
        "ftw_fragmentation_relative_to_left": reverse["oversegmentation_mean"],
        "ftw_merging_relative_to_left": reverse["undersegmentation_mean"],
        "count_difference_left_minus_ftw": len(left) - len(ftw),
        "area_difference_ha_left_minus_ftw": (
            left.to_crs(6933).area.sum() - ftw.to_crs(6933).area.sum()
        )
        / 10000,
        "interpretation": "Product correspondence, not reference-based accuracy",
    }
