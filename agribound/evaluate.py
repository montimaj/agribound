"""
Accuracy evaluation metrics for field boundary delineation.

Computes field-level metrics (IoU, precision, recall, F1) by matching
predicted field polygons to reference boundaries using spatial indexing.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.io.crs import get_equal_area_crs

logger = logging.getLogger(__name__)


def evaluate(
    predicted: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute field-level accuracy metrics.

    Matches predicted polygons to reference polygons using IoU (Intersection
    over Union) and computes standard detection metrics.

    Parameters
    ----------
    predicted : geopandas.GeoDataFrame
        Predicted field boundaries.
    reference : geopandas.GeoDataFrame
        Ground-truth reference field boundaries.
    iou_threshold : float
        Minimum IoU for a match to count as a true positive (default 0.5).

    Returns
    -------
    dict[str, Any]
        Dictionary of metrics:

        - ``iou_mean``: Mean IoU across matched fields.
        - ``precision``: TP / (TP + FP) at field level.
        - ``recall``: TP / (TP + FN) at field level.
        - ``f1``: Harmonic mean of precision and recall.
        - ``over_segmentation``: Fraction of reference fields matched by >1 prediction.
        - ``under_segmentation``: Fraction of predictions matching >1 reference.
        - ``area_error_mean_m2``: Mean absolute area difference for matched fields.
        - ``count_predicted``: Number of predicted fields.
        - ``count_reference``: Number of reference fields.
        - ``count_tp``: True positives.
        - ``count_fp``: False positives.
        - ``count_fn``: False negatives.

    Examples
    --------
    >>> from agribound.evaluate import evaluate
    >>> metrics = evaluate(predicted_gdf, reference_gdf)
    >>> print(f"F1: {metrics['f1']:.3f}")
    """
    if len(predicted) == 0 and len(reference) == 0:
        return _empty_metrics()
    if len(predicted) == 0:
        return _zero_precision_metrics(len(reference))
    if len(reference) == 0:
        return _zero_recall_metrics(len(predicted))

    # Ensure same CRS (equal-area for accurate IoU)
    ea_crs = get_equal_area_crs()
    pred = predicted.to_crs(ea_crs).copy().reset_index(drop=True)
    ref = reference.to_crs(ea_crs).copy().reset_index(drop=True)

    # Build spatial index on predictions
    pred_sindex = pred.sindex

    # Match reference → predicted
    ref_matched = {}  # ref_idx → list of (pred_idx, iou)
    pred_matched = {}  # pred_idx → list of (ref_idx, iou)

    for ref_idx, ref_row in ref.iterrows():
        ref_geom = ref_row.geometry
        if ref_geom is None or ref_geom.is_empty or not ref_geom.is_valid:
            continue

        candidates = list(pred_sindex.intersection(ref_geom.bounds))
        best_iou = 0

        for pred_idx in candidates:
            pred_geom = pred.iloc[pred_idx].geometry
            if pred_geom is None or pred_geom.is_empty or not pred_geom.is_valid:
                continue

            try:
                intersection = ref_geom.intersection(pred_geom)
                union_area = ref_geom.area + pred_geom.area - intersection.area
                iou = intersection.area / union_area if union_area > 0 else 0
            except Exception:
                iou = 0

            if iou > best_iou:
                best_iou = iou

            # Track all matches above threshold for over/under segmentation
            if iou >= iou_threshold:
                ref_matched.setdefault(ref_idx, []).append((pred_idx, iou))
                pred_matched.setdefault(pred_idx, []).append((ref_idx, iou))

    # Compute metrics
    tp = 0
    ious = []
    area_errors = []
    matched_refs = set()
    matched_preds = set()

    for ref_idx, matches in ref_matched.items():
        if matches:
            best_pred_idx, best_iou = max(matches, key=lambda x: x[1])
            tp += 1
            ious.append(best_iou)
            matched_refs.add(ref_idx)
            matched_preds.add(best_pred_idx)

            # Area error
            ref_area = (
                ref.iloc[ref_idx].geometry.area if hasattr(ref.iloc[ref_idx], "geometry") else 0
            )
            pred_area = (
                pred.iloc[best_pred_idx].geometry.area
                if hasattr(pred.iloc[best_pred_idx], "geometry")
                else 0
            )
            area_errors.append(abs(ref_area - pred_area))

    fp = len(pred) - len(matched_preds)
    fn = len(ref) - len(matched_refs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Over-segmentation: reference fields matched by >1 prediction
    over_seg_count = sum(1 for matches in ref_matched.values() if len(matches) > 1)
    over_seg = over_seg_count / len(ref) if len(ref) > 0 else 0

    # Under-segmentation: predictions matching >1 reference
    under_seg_count = sum(1 for matches in pred_matched.values() if len(matches) > 1)
    under_seg = under_seg_count / len(pred) if len(pred) > 0 else 0

    metrics = {
        "iou_mean": float(np.mean(ious)) if ious else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "over_segmentation": float(over_seg),
        "under_segmentation": float(under_seg),
        "area_error_mean_m2": float(np.mean(area_errors)) if area_errors else 0.0,
        "count_predicted": len(pred),
        "count_reference": len(ref),
        "count_tp": tp,
        "count_fp": fp,
        "count_fn": fn,
        "iou_threshold": iou_threshold,
    }

    logger.info(
        "Evaluation: P=%.3f R=%.3f F1=%.3f IoU=%.3f (TP=%d FP=%d FN=%d)",
        precision,
        recall,
        f1,
        metrics["iou_mean"],
        tp,
        fp,
        fn,
    )

    return metrics


def _empty_metrics() -> dict[str, Any]:
    return {
        "iou_mean": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "over_segmentation": 0.0,
        "under_segmentation": 0.0,
        "area_error_mean_m2": 0.0,
        "count_predicted": 0,
        "count_reference": 0,
        "count_tp": 0,
        "count_fp": 0,
        "count_fn": 0,
        "iou_threshold": 0.5,
    }


def _zero_precision_metrics(n_ref: int) -> dict[str, Any]:
    return {
        "iou_mean": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "over_segmentation": 0.0,
        "under_segmentation": 0.0,
        "area_error_mean_m2": 0.0,
        "count_predicted": 0,
        "count_reference": n_ref,
        "count_tp": 0,
        "count_fp": 0,
        "count_fn": n_ref,
        "iou_threshold": 0.5,
    }


def _zero_recall_metrics(n_pred: int) -> dict[str, Any]:
    return {
        "iou_mean": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "over_segmentation": 0.0,
        "under_segmentation": 0.0,
        "area_error_mean_m2": 0.0,
        "count_predicted": n_pred,
        "count_reference": 0,
        "count_tp": 0,
        "count_fp": n_pred,
        "count_fn": 0,
        "iou_threshold": 0.5,
    }
