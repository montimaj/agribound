"""
Object-level accuracy assessment of field-boundary polygons.

Predicted field polygons are compared with reference field polygons. The
functions here report, for a prediction layer and a reference layer:

- **object detection metrics** — precision, recall and F1 of polygon matches
  at an IoU threshold (one-to-one or legacy many-to-one matching);
- **area-weighted metrics** — the fraction of reference (predicted) area that
  belongs to matched fields;
- **segmentation errors** — over- and under-segmentation of each reference
  field against the prediction it overlaps most (Persello & Bruzzone, 2010);
- **boundary metrics** — Hausdorff and symmetric mean boundary distance for
  matched pairs, boundary precision/recall/F1 within a distance tolerance
  (buffered-line method) and the fraction of reference fields that are
  matched with a mean boundary distance within that tolerance;
- the same metrics **per stratum** (a reference attribute) and **per field-size
  class**, with **percentile bootstrap confidence intervals** obtained by
  resampling reference fields.

:func:`evaluate` returns the summary dictionary, :func:`evaluate_frame` the
per-reference-field table the summary is built from, and
:func:`pixels_per_field` the number of pixels a field spans at a given ground
sample distance.

References
----------
Clinton, N., Holt, A., Scarborough, J., Yan, L., Gong, P., 2010. Accuracy
assessment measures for object-based image segmentation goodness.
Photogrammetric Engineering & Remote Sensing 76 (3), 289-299.
https://doi.org/10.14358/PERS.76.3.289

Persello, C., Bruzzone, L., 2010. A novel protocol for accuracy assessment in
classification of very high resolution images. IEEE Transactions on Geoscience
and Remote Sensing 48 (3), 1232-1244. https://doi.org/10.1109/TGRS.2009.2029570

Stehman, S.V., Foody, G.M., 2019. Key issues in rigorous accuracy assessment
of land cover products. Remote Sensing of Environment 231, 111199.
https://doi.org/10.1016/j.rse.2019.05.018
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely

from agribound.io.crs import get_equal_area_crs

logger = logging.getLogger(__name__)

__all__ = [
    "BOUNDARY_MAX_SAMPLES",
    "BOUNDARY_SAMPLE_SPACING_M",
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "MATCHING_METHODS",
    "evaluate",
    "evaluate_frame",
    "pixels_per_field",
]

#: Supported values of the ``matching`` argument.
MATCHING_METHODS: tuple[str, ...] = ("one_to_one", "many_to_one")

#: Default maximum spacing (m) between boundary sample points for the distance
#: metrics (the default of ``boundary_sample_spacing_m``). Original vertices
#: are always kept, so samples can be closer than this. See :func:`evaluate`,
#: Notes.
BOUNDARY_SAMPLE_SPACING_M: float = 1.0

#: With ``boundary_sample_spacing_m=None``, boundaries longer than
#: ``BOUNDARY_SAMPLE_SPACING_M * BOUNDARY_MAX_SAMPLES`` are sampled at a
#: maximum spacing of ``perimeter / BOUNDARY_MAX_SAMPLES`` instead.
BOUNDARY_MAX_SAMPLES: int = 1000

#: Confidence level of the percentile bootstrap intervals.
BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95

# Relative tolerance used when comparing IoU with the threshold, so that an
# IoU equal to the threshold in exact arithmetic is not lost to rounding.
_IOU_RTOL = 1e-9

# An equal_area_crs is reported as not preserving area when the area of any
# evaluated polygon in it differs from its area in EPSG:6933 (exact on the
# WGS 84 ellipsoid) by more than this (relative). On the 50,603 NMOSE fields
# (New Mexico), equal-area CRSs (EPSG:5070, 8857, 3035, ESRI:102003, 54034,
# 54008) gave ratios within 1.1e-5 of 1, the residual coming from edges being
# chords in each CRS; UTM zone 13N gave 0.9992-1.0028, ESRI:54009 (Mollweide,
# a spherical projection applied to ellipsoidal coordinates) 1.0019-1.0031
# and Web Mercator 1.38-1.57.
_AREA_RATIO_RTOL = 1e-4

# Before reprojection to equal_area_crs, edges are split into pieces of at
# most this many metres (converted to source-CRS units), so an edge that is
# straight in the source CRS is not replaced by one long chord in the
# equal-area CRS (see evaluate(), Notes). The pieces are collinear in the
# source CRS, so the polygons themselves are unchanged there.
_DENSIFY_M = 50.0

# Points processed per shapely.distance call when sampling boundaries.
_DISTANCE_CHUNK = 1_000_000

# Budget of boundary vertices per chunk of the tolerance computation: the
# chunk's own vertices plus, for each candidate polygon of the other layer,
# its vertex count divided by the number of polygons it is a candidate of.
# Results depend on it only through floating-point rounding.
_SEGMENT_CHUNK = 5_000

# Bootstrap resamples are processed in chunks of about this many
# (resample x reference field) weights. The resamples do not depend on it; the
# metric sums (a BLAS matrix product) can differ in the last bit between chunk
# sizes on some platforms (seen on Windows), and the chunk size is fixed for a
# given number of reference fields, so a run is reproducible on one platform.
_BOOTSTRAP_CHUNK_CELLS = 2_000_000

# Shapely geometry type ids.
_POLYGON = 3
_MULTIPOLYGON = 6
_COLLECTION = 7

_AUTO_BIN_MANTISSAS = (1.0, 2.0, 5.0)

# Per-reference-field quantities whose sums define every metric. Quantities of
# predictions ("*_pred*", "legacy_underseg") are attributed to the prediction's
# home reference field (see _Run.pred_home).
_QUANTITIES: tuple[str, ...] = (
    "n_ref",
    "area_ref",
    "tp",
    "iou_tp",
    "best_iou",
    "abs_area_err_tp",
    "signed_area_err_tp",
    "area_ref_tp",
    "os",
    "us",
    "us_defined",
    "legacy_overseg",
    "n_dist",
    "hausdorff",
    "mean_dist",
    "perim_ref",
    "within_len_ref",
    "within_tol",
    "area_within_tol",
    "n_pred",
    "n_pred_matched",
    "area_pred",
    "area_pred_matched",
    "legacy_underseg",
    "perim_pred",
    "within_len_pred",
)
_QI = {name: k for k, name in enumerate(_QUANTITIES)}

_CI_METRICS: tuple[str, ...] = (
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
    "hausdorff_mean_m",
    "boundary_distance_mean_m",
)
_CI_METRICS_TOLERANCE: tuple[str, ...] = (
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "coverage_within_tolerance",
    "coverage_within_tolerance_area",
)
_RATE_KEYS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "iou_mean",
    "area_weighted_precision",
    "area_weighted_recall",
    "area_weighted_f1",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "coverage_within_tolerance",
    "coverage_within_tolerance_area",
)
_COUNT_KEYS: tuple[str, ...] = (
    "count_reference",
    "count_predicted",
    "count_tp",
    "count_fp",
    "count_fn",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    predicted: gpd.GeoDataFrame | gpd.GeoSeries,
    reference: gpd.GeoDataFrame | gpd.GeoSeries,
    iou_threshold: float = 0.5,
    *,
    matching: str = "one_to_one",
    strata: str | pd.Series | Sequence[Any] | np.ndarray | None = None,
    size_bins: str | Sequence[float] | np.ndarray | None = None,
    bootstrap: int = 0,
    bootstrap_seed: int = 42,
    boundary_tolerance_m: float | None = None,
    equal_area_crs: Any = None,
    boundary_sample_spacing_m: float | None = BOUNDARY_SAMPLE_SPACING_M,
    boundary_mask: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
) -> dict[str, Any]:
    """Compute object-level accuracy metrics of predicted field polygons.

    Parameters
    ----------
    predicted : geopandas.GeoDataFrame or geopandas.GeoSeries
        Predicted field polygons. Must have a CRS unless it contains no
        geometries.
    reference : geopandas.GeoDataFrame or geopandas.GeoSeries
        Reference field polygons. Must have a CRS unless it contains no
        geometries (it may differ from the prediction's CRS).
    boundary_mask : geopandas.GeoDataFrame or geopandas.GeoSeries, optional
        Known reference coverage for boundary tolerance metrics. Intersect
        original boundary lines with this polygon mask; never use the boundary
        of clipped polygons, which would introduce artificial mask edges.
        Requires ``boundary_tolerance_m``. Does not select objects or change
        their IoU, area, matched-pair distances or size classes. Select whole
        objects consistently before calling when coverage is incomplete.
        A 1 mm outward guard in the metric CRS prevents floating-point
        reprojection from discarding lines exactly on the mask boundary.
    iou_threshold : float
        Minimum IoU (intersection over union) for a prediction and a reference
        field to match, in ``(0, 1]`` (default 0.5). A pair matches when
        ``IoU >= iou_threshold`` (compared with a relative tolerance of 1e-9 to
        absorb floating-point rounding) and the intersection has positive area.
    matching : {"one_to_one", "many_to_one"}
        ``"one_to_one"`` (default): greedy one-to-one assignment. All pairs
        with IoU at or above the threshold are sorted by descending IoU (ties
        by reference then prediction position) and a pair is accepted when
        neither polygon has been matched yet. This is a greedy assignment in
        the spirit of COCO-style detection evaluation (which processes
        detections in order of confidence); without confidence scores, pairs
        are processed in order of IoU. It is not an optimal (maximum-cardinality)
        assignment. When neither layer contains overlapping polygons and
        ``iou_threshold > 0.5``, every polygon has at most one partner above
        the threshold, so the matching is unique.
        ``"many_to_one"``: the 0.1.x behaviour. Each reference field is matched
        to its highest-IoU prediction at or above the threshold, and one
        prediction may match several reference fields; ``count_tp`` counts
        matched reference fields and ``count_fp`` counts predictions that
        matched none, so ``precision = TP / (TP + FP)`` can exceed the
        fraction of predictions that are matched.
    strata : str, pandas.Series or array-like, optional
        Reference stratum of each reference field: the name of a column of
        *reference*, a Series aligned with ``reference.index``, or an array
        with one value per reference row. Metrics are also reported per
        stratum (``per_stratum``). Rows with a missing value contribute to the
        overall metrics only.
    size_bins : "auto" or sequence of float, optional
        Field-size classes, in **hectares** (1 ha = 10,000 m²), of the
        reference field area in *equal_area_crs*. Either ``"auto"`` —
        approximately log-spaced edges from the 1-2-5 series (..., 0.1, 0.2,
        0.5, 1, 2, 5, 10, ... ha) spanning the reference areas — or at least
        two strictly increasing, non-negative edges (the last may be
        ``inf``). Classes are ``[lower, upper)`` except the last, which is
        ``[lower, upper]`` (the :func:`numpy.histogram` convention). Metrics
        are also reported per class (``per_size_class``); reference fields
        outside the edges contribute to the overall metrics only.
    bootstrap : int
        Number of bootstrap resamples for confidence intervals (0 disables).
    bootstrap_seed : int
        Seed (a non-negative integer) of the bootstrap random number
        generator.
    boundary_tolerance_m : float, optional
        Distance tolerance in metres. When given, boundary precision/recall/F1
        and coverage within tolerance are computed.
    equal_area_crs : CRS-like, optional
        Projected CRS with axes in metres used for areas and IoU (anything
        :meth:`pyproj.CRS.from_user_input` accepts). Defaults to
        :func:`agribound.io.crs.get_equal_area_crs` (EPSG:6933). Areas are
        preserved only by an equal-area projection, and only on the ellipsoid
        it is defined for. Spherical projections applied to WGS 84
        coordinates, such as ESRI:54009 (Mollweide), change ellipsoidal areas
        by up to about 0.7 %. Another projected CRS is accepted and used as
        given, but a WARNING is logged when the area of any evaluated polygon
        in it differs from its area in EPSG:6933 by more than 0.01 %
        (relative). The reported areas, area errors, area-weighted metrics and
        size classes then use the areas in the given CRS.
    boundary_sample_spacing_m : float or None
        Maximum spacing in metres between the sample points of each boundary
        for the Hausdorff and mean boundary distances. The default,
        ``BOUNDARY_SAMPLE_SPACING_M`` (1 m), applies to every boundary, so the
        sampling error bound (see Notes) is the same for every field
        whatever its size. ``None`` selects a length-dependent spacing,
        ``max(BOUNDARY_SAMPLE_SPACING_M, perimeter / BOUNDARY_MAX_SAMPLES)``:
        1 m, or perimeter / 1000 for boundaries longer than 1 km. It is
        faster on layers with many large fields: in our benchmark, the
        50,603 NMOSE fields took 33 s instead of 56 s. But its error bound
        grows with field size, which biases comparisons of boundary distances
        between size classes. Run time grows with the number of samples.

    Returns
    -------
    dict[str, Any]
        Keys kept from 0.1.x (``over_segmentation``/``under_segmentation`` keep
        their 0.1.x definitions; the default matching changed to one-to-one):

        - ``iou_mean``: mean IoU of matched pairs (0.0 when nothing matched).
        - ``precision``: TP / (TP + FP).
        - ``recall``: TP / (TP + FN), i.e. matched / evaluated reference fields.
        - ``f1``: harmonic mean of precision and recall.
        - ``over_segmentation``: fraction of reference fields with more than
          one prediction at IoU >= threshold (0.1.x definition).
        - ``under_segmentation``: fraction of predictions with more than one
          reference field at IoU >= threshold (0.1.x definition). With
          non-overlapping polygons and a threshold above 0.5 both are always
          0; use ``oversegmentation_mean`` and ``undersegmentation_mean`` to
          measure splits and merges.
        - ``area_error_mean_m2``: mean absolute area difference of matched
          pairs (0.0 when nothing matched).
        - ``count_predicted``, ``count_reference``: evaluated polygons.
        - ``count_tp``, ``count_fp``, ``count_fn``: true positives (matched
          reference fields), false positives (unmatched predictions) and
          false negatives (unmatched reference fields).
        - ``iou_threshold``.

        Added in 1.0:

        - ``matching``, ``equal_area_crs`` (e.g. ``"EPSG:6933"``),
          ``distance_crs`` (description) and ``utm_epsg_codes`` (UTM zones
          used for distances).
        - ``best_iou_mean``: mean over all reference fields of the highest IoU
          with any prediction (0 for fields no prediction overlaps); unlike
          ``iou_mean`` it is not conditional on a match.
        - ``area_weighted_recall``: area of matched reference fields / area of
          all reference fields. ``area_weighted_precision``: area of matched
          predictions / area of all predictions. ``area_weighted_f1``.
        - ``reference_area_ha``, ``predicted_area_ha``: total areas.
        - ``median_area_ha``: median reference field area.
        - ``area_error_signed_mean_m2``: mean of (predicted - reference) area
          over matched pairs (NaN when nothing matched).
        - ``oversegmentation_mean``, ``undersegmentation_mean``: means over
          reference fields of ``OS = 1 - |r ∩ p'| / |r|`` and
          ``US = 1 - |r ∩ p'| / |p'|``, where ``p'`` is the prediction with
          the largest intersection area with reference field ``r`` (ties:
          the first in prediction order; the maximum-overlap pairing of
          Persello & Bruzzone, 2010; the measures
          have the form of Clinton et al.'s (2010) OverSegmentation and
          UnderSegmentation, which pair each reference object with a
          different subset of segments). A reference field that no
          prediction overlaps has OS = 1 and no US (excluded from the US
          mean). Both are 0 for a perfect delineation.
        - ``hausdorff_mean_m``, ``hausdorff_median_m``: Hausdorff distance
          between the boundaries of matched pairs.
        - ``boundary_distance_mean_m``, ``boundary_distance_median_m``:
          symmetric mean boundary distance of matched pairs,
          ``(∫_∂r d(x, ∂p) dx + ∫_∂p d(x, ∂r) dx) / (|∂r| + |∂p|)``.
        - ``boundary_sample_spacing_m``: the argument (None for the
          length-dependent spacing). ``boundary_sample_spacing_max_m``: the
          largest spacing ``Δ`` used for any matched pair (NaN when nothing
          matched), which bounds the sampling error of the two distances
          above (see Notes).
        - ``count_fp_unassigned``: false positives that overlap no reference
          field (they cannot be attributed to a stratum or size class). When
          the reference layer does not contain every field in the area (a
          registry of some fields only), these predictions may be real fields
          missing from it; ``count_tp / (count_tp + count_fp -
          count_fp_unassigned)`` is the precision among predictions that
          overlap a reference field.
        - ``count_reference_dropped``, ``count_predicted_dropped``: rows
          excluded because their geometry is null, empty, non-polygonal or
          has zero area. ``count_reference_repaired``,
          ``count_predicted_repaired``: evaluated geometries that were invalid
          and were repaired with ``shapely.make_valid(method="structure")``.

        With ``boundary_tolerance_m``:

        - ``boundary_tolerance_m``.
        - ``boundary_recall``: length of reference boundaries within the
          tolerance of any predicted boundary / total reference boundary
          length. ``boundary_precision``: the same for predicted boundaries
          against reference boundaries. ``boundary_f1``. Lengths are summed
          per polygon, so an edge shared by two adjacent fields counts once
          for each (buffered-line method: the length of one layer's boundary
          inside the union of the tolerance buffers of the other layer's
          boundaries, computed exactly; see Notes).
        - ``coverage_within_tolerance``: fraction of reference fields that are
          matched **and** whose symmetric mean boundary distance is at most the
          tolerance; ``coverage_within_tolerance_area``: the same weighted by
          reference field area.

        With ``strata``: ``strata_column`` (name or None),
        ``count_reference_no_stratum`` and ``per_stratum`` — a dict keyed by
        ``str(stratum value)`` of the core metrics above plus ``n`` (reference
        fields in the stratum). With ``size_bins``: ``size_class_edges_ha``,
        ``count_reference_outside_size_classes`` and ``per_size_class`` —
        keyed ``"<lower>-<upper>"`` (ha, formatted with ``:g``, or in full
        precision if ``:g`` would make two keys equal), each with
        ``lower_ha``/``upper_ha``.
        In a group, a prediction counts in the group of its *home* reference
        field: its matched reference field if matched (for many-to-one, the
        matched field with the highest IoU), else the reference field it
        overlaps most. Predictions overlapping no reference field count only
        in the overall metrics. A size class without reference fields is
        reported with ``n = 0`` and NaN rates.

        With ``bootstrap > 0``: ``bootstrap`` — ``{"n_resamples", "seed",
        "confidence_level", "method", "resampling", "ci", "n_undefined"}``,
        where ``ci`` maps metric name to ``[lower, upper]``; each
        ``per_stratum``/``per_size_class`` entry also gains ``ci`` (and
        ``ci_n_undefined`` when a metric was undefined in some resamples).

        Conventions for empty denominators: the rates (keys ending in
        ``precision``, ``recall`` or ``f1``, ``iou_mean`` and the coverage
        keys) and the 0.1.x keys ``over_segmentation``, ``under_segmentation``
        and ``area_error_mean_m2`` are 0.0; all other means and distances are
        NaN. When both layers are empty the rates are 1.0 (0.1.x behaviour).

    Raises
    ------
    TypeError
        If an input is not a GeoDataFrame/GeoSeries.
    ValueError
        If an input that contains geometries has no CRS, or an argument is out
        of range.
    ImportError
        If a layer contains invalid geometries and the installed shapely does
        not provide ``make_valid(method="structure")`` (shapely >= 2.1 built
        with GEOS >= 3.10).

    Notes
    -----
    **Geometry handling.** Invalid geometries are repaired with
    ``shapely.make_valid(method="structure", keep_collapsed=False)`` and only
    their polygonal parts are kept; null, empty, non-polygonal and zero-area
    geometries are excluded (counted in ``count_*_dropped``). The repair
    needs shapely >= 2.1 built with GEOS >= 3.10. With older versions an
    ImportError is raised when a repair is needed, rather than switching to
    the ``"linework"`` method, which repairs some geometries differently. A layer
    without a CRS is accepted only when it has no geometries (for example an
    empty prediction). Layers that cross the antimeridian are not supported.

    **Coordinate systems.** Areas and IoU are computed in *equal_area_crs*.
    Geometries are reprojected vertex by vertex, so each edge becomes a
    straight chord in the target CRS. Before reprojection to *equal_area_crs*,
    edges are split into pieces of at most 50 m in the source CRS. The pieces
    are collinear there, so the polygons are unchanged in their own CRS.
    Without this split, a long edge with few vertices would become a single
    chord, while a coincident edge of the other layer with more vertices
    follows the curved image of the line, and the two would no longer
    coincide. For a 2 km × 10 m field that changed the IoU by up to 0.4 % at
    34° latitude and 1.5 % at 70°.

    With the split, we compared against the planar IoU in the source UTM
    CRS (inputs in UTM zone 13N). For 100 m and 1 km shifted squares and a
    notched 2 km × 10 m field, at latitudes from 0.5° to 70°, on and 2.7° off
    the central meridian, the relative difference was at most 5.4e-6. For
    1,500 fields polygonised at 1 m, compared with their own ~20-vertex
    polygons at 34° N, it was at most 2.5e-7.

    Distances (Hausdorff, boundary distances and the tolerance metrics) are
    computed in the WGS 84 / UTM zone of each polygon's centroid: reference
    fields and their matched predictions in the reference field's zone, and
    predicted boundaries in the prediction's zone for boundary precision.
    Inside a zone the linear scale error is below about 0.1 %. Geometries are
    not split before reprojection to UTM. From a source CRS in the same zone
    (for example NAD83 / UTM 13N into WGS 84 / UTM 13N) edges stay straight to
    within 1e-8 m. A 1 km edge along a parallel that is straight in EPSG:4326
    is replaced by a chord up to 1.3 cm from its image at 34° latitude, and
    3.4 cm at 60°. The deviation grows with the square of the edge length.

    **Candidate pairs** come from a shapely ``STRtree`` query with the
    ``"intersects"`` predicate; intersection areas are computed with
    vectorised shapely operations.

    **Boundary distances** are computed by sampling each boundary (interior
    rings included) at points at most ``Δ`` apart, keeping every original
    vertex, and taking the exact distance from each sample to the other
    boundary. ``Δ`` is *boundary_sample_spacing_m* (1 m by default), or,
    when that is None, ``max(BOUNDARY_SAMPLE_SPACING_M, perimeter /
    BOUNDARY_MAX_SAMPLES)``: 1 m, or perimeter / 1000 for boundaries longer
    than 1 km. The distance to a boundary is 1-Lipschitz. So, with ``Δ`` the
    larger spacing of the two boundaries of a pair
    (``boundary_sample_spacing_m`` in :func:`evaluate_frame`), the Hausdorff
    distance is underestimated by at most ``Δ / 2``, and the symmetric mean
    distance (trapezoidal rule) differs from the exact line-integral mean by
    at most ``Δ / 4``.

    **Tolerance metrics** are exact up to floating-point rounding. Boundaries
    are split into their straight segments. For each pair of nearby segments,
    one from each layer, the part of the first within the tolerance of the
    second — its intersection with the second's tolerance region, a
    rectangle capped by two discs — is computed in closed form, and the
    union of these parts along each segment is measured. This is the length
    inside the circular-arc buffers of the other layer's boundaries, with no
    polygonal approximation of the arcs. Run time grows with the number of
    boundary segments within the tolerance of each other.

    **Bootstrap.** Each resample draws reference fields with replacement (the
    same number as evaluated); with *strata*, every stratum — and the set of
    fields with a missing stratum — is resampled separately at its own size.
    A resampled field brings its per-field results and the predictions whose
    home it is; predictions overlapping no reference field are kept unchanged
    in every resample, so the intervals reflect sampling variability of the
    reference fields only. Intervals are percentile intervals at
    ``BOOTSTRAP_CONFIDENCE_LEVEL`` (95 %) over the resamples in which the
    metric is defined; resamples are generated from child seeds of
    ``numpy.random.SeedSequence(bootstrap_seed)``, so the resamples do not
    depend on how they are batched. Within a resample a rate whose
    denominator is 0 — for example any metric of a size class none of whose
    fields was drawn — is undefined rather than 0, and ``n_undefined`` counts
    such resamples. Medians have no intervals. The intervals treat the
    evaluated reference fields as an independent random sample (a stratified
    random sample when *strata* is given). They do not model spatial
    autocorrelation between neighbouring fields, so they are too narrow when
    errors are spatially clustered, and they are not design-based estimates
    for a probability sample of reference fields (see Stehman & Foody, 2019,
    on sampling design and analysis).

    Examples
    --------
    >>> from agribound.evaluate import evaluate
    >>> metrics = evaluate(predicted_gdf, reference_gdf, boundary_tolerance_m=5.0)
    >>> print(f"F1: {metrics['f1']:.3f}, area-weighted recall: "
    ...       f"{metrics['area_weighted_recall']:.3f}")
    """
    params = _Params.from_args(
        iou_threshold=iou_threshold,
        matching=matching,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
        boundary_tolerance_m=boundary_tolerance_m,
        equal_area_crs=equal_area_crs,
        boundary_sample_spacing_m=boundary_sample_spacing_m,
    )
    run = _run(
        predicted,
        reference,
        params,
        strata=strata,
        size_bins=size_bins,
        boundary_mask=boundary_mask,
    )
    metrics = _summarise(run)
    if boundary_mask is not None:
        metrics["boundary_mask_applied"] = True
    if params.bootstrap > 0:
        _add_bootstrap(metrics, run)

    logger.info(
        "Evaluation (%s, IoU>=%.2f): P=%.3f R=%.3f F1=%.3f IoU=%.3f "
        "area-weighted R=%.3f (TP=%d FP=%d FN=%d)",
        params.matching,
        params.iou_threshold,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["iou_mean"],
        metrics["area_weighted_recall"],
        metrics["count_tp"],
        metrics["count_fp"],
        metrics["count_fn"],
    )
    return metrics


def evaluate_frame(
    predicted: gpd.GeoDataFrame | gpd.GeoSeries,
    reference: gpd.GeoDataFrame | gpd.GeoSeries,
    *,
    iou_threshold: float = 0.5,
    matching: str = "one_to_one",
    strata: str | pd.Series | Sequence[Any] | np.ndarray | None = None,
    equal_area_crs: Any = None,
    boundary_tolerance_m: float | None = None,
    size_bins: str | Sequence[float] | np.ndarray | None = None,
    boundary_sample_spacing_m: float | None = BOUNDARY_SAMPLE_SPACING_M,
    boundary_mask: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
) -> pd.DataFrame:
    """Return the per-reference-field evaluation table.

    One row per evaluated reference field (null, empty, non-polygonal and
    zero-area reference geometries are excluded), indexed by the reference
    index, so the table can be joined back to *reference*. The arguments and
    methods are those of :func:`evaluate`; the summary returned by
    :func:`evaluate` is computed from the same per-field results.

    Parameters
    ----------
    predicted, reference, iou_threshold, matching, strata, equal_area_crs
        See :func:`evaluate`.
    boundary_tolerance_m : float, optional
        See :func:`evaluate`; adds the tolerance columns.
    size_bins : "auto" or sequence of float, optional
        See :func:`evaluate` (hectares); adds a ``size_class`` column.
    boundary_sample_spacing_m : float or None
        See :func:`evaluate` (default 1 m).
    boundary_mask : geopandas.GeoDataFrame or geopandas.GeoSeries, optional
        See :func:`evaluate`; ``perimeter_m`` then measures only original
        boundary lines inside the mask, while matched-pair distances stay whole.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``area_m2``, ``area_ha``: reference field area in *equal_area_crs*.
        - ``perimeter_m``: reference boundary length (UTM zone of the field).
        - ``utm_epsg``: EPSG code of that UTM zone.
        - ``stratum`` (with *strata*), ``size_class`` (with *size_bins*;
          None outside the edges).
        - ``matched``: bool. ``pred_index``: index label of the matched
          prediction (None when unmatched). ``iou``: IoU with it (NaN when
          unmatched). ``pred_area_m2``: its area (NaN when unmatched).
        - ``best_iou``: highest IoU with any prediction (0 when none
          overlaps). ``n_overlapping_pred``: predictions with a positive-area
          intersection.
        - ``max_overlap_pred_index``: index label of the prediction with the
          largest intersection (ties: the first in prediction order; None
          when none overlaps);
          ``oversegmentation``, ``undersegmentation``: OS and US against it
          (see :func:`evaluate`; US is NaN when no prediction overlaps).
        - ``hausdorff_m``, ``boundary_mean_distance_m``: boundary distances to
          the matched prediction (NaN when unmatched).
          ``boundary_sample_spacing_m``: the sample spacing ``Δ`` behind them
          (the larger of the two boundaries' spacings; NaN when unmatched).
          The Hausdorff distance is underestimated by at most ``Δ / 2``, and
          the mean distance is within ``Δ / 4`` of its exact value (see
          :func:`evaluate`, Notes).
        - With *boundary_tolerance_m*: ``boundary_within_tolerance_m`` (length
          of the field boundary within the tolerance of any predicted
          boundary), ``boundary_recall`` (that length / ``perimeter_m``) and
          ``within_tolerance`` (matched and ``boundary_mean_distance_m`` at
          most the tolerance).

        ``DataFrame.attrs`` records ``iou_threshold``, ``matching``,
        ``equal_area_crs``, ``boundary_tolerance_m`` and
        ``boundary_sample_spacing_m``.
    """
    params = _Params.from_args(
        iou_threshold=iou_threshold,
        matching=matching,
        bootstrap=0,
        bootstrap_seed=0,
        boundary_tolerance_m=boundary_tolerance_m,
        equal_area_crs=equal_area_crs,
        boundary_sample_spacing_m=boundary_sample_spacing_m,
    )
    run = _run(
        predicted,
        reference,
        params,
        strata=strata,
        size_bins=size_bins,
        boundary_mask=boundary_mask,
    )
    frame = _frame(run)
    if boundary_mask is not None:
        frame.attrs["boundary_mask_applied"] = True
    return frame


def pixels_per_field(
    reference: gpd.GeoDataFrame | gpd.GeoSeries,
    gsd_m: float,
    equal_area_crs: Any = None,
) -> pd.Series:
    """Return the number of pixels each field spans at a ground sample distance.

    ``p = A / gsd_m²``, with ``A`` the field area in m² in *equal_area_crs*:
    the number of ``gsd_m × gsd_m`` pixels whose total area equals the field
    area (not a count of pixel centres inside the field).

    Parameters
    ----------
    reference : geopandas.GeoDataFrame or geopandas.GeoSeries
        Field polygons. Must have a CRS unless it contains no geometries
        (then every value is NaN).
    gsd_m : float
        Ground sample distance (pixel size) in metres, > 0.
    equal_area_crs : CRS-like, optional
        Projected CRS for areas (default EPSG:6933). As in :func:`evaluate`,
        a WARNING is logged when it does not preserve the fields' areas.

    Returns
    -------
    pandas.Series
        Float series named ``"pixels_per_field"`` with the index of
        *reference*. Invalid geometries are repaired as in :func:`evaluate`;
        null, empty, non-polygonal and zero-area geometries give NaN.

    Raises
    ------
    TypeError
        If *reference* is not a GeoDataFrame/GeoSeries.
    ValueError
        If *gsd_m* is not a positive finite number, *equal_area_crs* is not a
        projected CRS in metres, or *reference* contains geometries but has
        no CRS.
    ImportError
        If invalid geometries need repair and the installed shapely does not
        provide ``make_valid(method="structure")`` (see :func:`evaluate`).
    """
    gsd = _positive_float(gsd_m, "gsd_m")
    ea_crs = _resolve_equal_area_crs(equal_area_crs)
    layer = _prepare_layer(reference, "reference", ea_crs)
    _check_equal_area(ea_crs, (layer,))
    values = np.full(layer.n_input, np.nan)
    values[layer.keep] = layer.area / (gsd * gsd)
    return pd.Series(values, index=layer.index, name="pixels_per_field")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Params:
    iou_threshold: float
    matching: str
    bootstrap: int
    bootstrap_seed: int
    tolerance: float | None
    ea_crs: pyproj.CRS
    spacing: float | None  # boundary sample spacing (None: length-dependent)

    @classmethod
    def from_args(
        cls,
        *,
        iou_threshold: Any,
        matching: Any,
        bootstrap: Any,
        bootstrap_seed: Any,
        boundary_tolerance_m: Any,
        equal_area_crs: Any,
        boundary_sample_spacing_m: Any,
    ) -> _Params:
        if isinstance(iou_threshold, bool | np.bool_):
            raise ValueError(f"iou_threshold must be a number, got {iou_threshold!r}")
        try:
            tau = float(iou_threshold)
        except (TypeError, ValueError):
            raise ValueError(f"iou_threshold must be a number, got {iou_threshold!r}") from None
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold!r}")
        if matching not in MATCHING_METHODS:
            raise ValueError(f"matching must be one of {MATCHING_METHODS}, got {matching!r}")
        n_boot = _non_negative_int(bootstrap, "bootstrap")
        seed = _non_negative_int(bootstrap_seed, "bootstrap_seed")
        tol = None
        if boundary_tolerance_m is not None:
            tol = _positive_float(boundary_tolerance_m, "boundary_tolerance_m")
        spacing = None
        if boundary_sample_spacing_m is not None:
            spacing = _positive_float(boundary_sample_spacing_m, "boundary_sample_spacing_m")
        return cls(
            iou_threshold=tau,
            matching=str(matching),
            bootstrap=n_boot,
            bootstrap_seed=seed,
            tolerance=tol,
            ea_crs=_resolve_equal_area_crs(equal_area_crs),
            spacing=spacing,
        )


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool | np.bool_):  # float(True) == 1.0 would pass silently
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number, got {value!r}") from None
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return out


def _resolve_equal_area_crs(value: Any) -> pyproj.CRS:
    if value is None:
        return get_equal_area_crs()
    try:
        crs = pyproj.CRS.from_user_input(value)
    except pyproj.exceptions.CRSError as exc:
        raise ValueError(f"equal_area_crs {value!r} is not a valid CRS: {exc}") from None
    if not crs.is_projected:
        raise ValueError(
            f"equal_area_crs must be a projected CRS (areas in metres), got {crs.to_string()}"
        )
    units = [(axis.unit_name, axis.unit_conversion_factor) for axis in crs.axis_info]
    if not units or any(factor != 1.0 for _, factor in units):
        raise ValueError(
            f"equal_area_crs must have axes in metres, got {crs.to_string()} with axis units "
            f"{[name for name, _ in units]}"
        )
    return crs


def _crs_label(crs: pyproj.CRS) -> str:
    """``"AUTHORITY:CODE"`` when *crs* matches an authority code exactly, else its string.

    An inexact match is not reported: pyproj's default confidence (70) would
    label, for example, ``+proj=cea +lat_ts=30`` on the GRS80 ellipsoid as
    EPSG:6933, which is defined on WGS 84.
    """
    authority = crs.to_authority(min_confidence=100)
    return f"{authority[0]}:{authority[1]}" if authority is not None else crs.to_string()


def _check_equal_area(ea_crs: pyproj.CRS, layers: Sequence[_Layer]) -> None:
    """Log a WARNING when *ea_crs* does not preserve the areas of the evaluated polygons.

    Each polygon's area in *ea_crs* is compared with the area of the same
    vertices reprojected to EPSG:6933, an equal-area projection on the WGS 84
    ellipsoid. PROJ's own scale factors (:meth:`pyproj.Proj.get_factors`) are
    not used: they refer to the projection's model, so for example they
    report exactly 1 for ESRI:54009 (Mollweide), whose spherical formulas
    change the ellipsoidal areas of New Mexico fields by 0.2-0.3 %.
    """
    reference = get_equal_area_crs()
    if ea_crs == reference:
        return
    ratios = [
        layer.area / shapely.area(_to_crs(layer.ea, ea_crs, reference))
        for layer in layers
        if layer.n
    ]
    if not ratios:
        return
    ratio = np.concatenate(ratios)
    finite = ratio[np.isfinite(ratio)]
    if len(finite) == len(ratio) and np.all(np.abs(finite - 1.0) <= _AREA_RATIO_RTOL):
        return
    n_undefined = len(ratio) - len(finite)
    if len(finite):
        extent = f"{finite.min():.6g} to {finite.max():.6g} times"
        if n_undefined:
            extent += f" (undefined for {n_undefined})"
    else:
        extent = "not comparable with"
    logger.warning(
        "equal_area_crs %s does not preserve area at the evaluated polygons: their areas in "
        "it are %s their areas in %s (equal-area on the WGS 84 ellipsoid). Reported areas and "
        "area errors, area-weighted metrics and size classes use the areas in %s; use an "
        "equal-area CRS such as %s for true areas",
        _crs_label(ea_crs),
        extent,
        _crs_label(reference),
        _crs_label(ea_crs),
        _crs_label(reference),
    )


# ---------------------------------------------------------------------------
# Geometry preparation
# ---------------------------------------------------------------------------


@dataclass
class _Layer:
    """Evaluated polygons of one input layer (positions 0..n-1)."""

    name: str
    index: pd.Index  # labels of all input rows
    keep: np.ndarray  # input row positions of the evaluated polygons
    src: np.ndarray  # repaired polygonal geometries, source CRS
    crs: pyproj.CRS
    ea: np.ndarray  # in the equal-area CRS, edges split into <= 50 m pieces first
    area: np.ndarray  # m² in the equal-area CRS
    n_input: int
    n_dropped: int
    n_repaired: int
    epsg: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    lon: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lat: np.ndarray = field(default_factory=lambda: np.zeros(0))
    utm: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    _tree: shapely.STRtree | None = None

    @property
    def n(self) -> int:
        return len(self.keep)

    @property
    def labels(self) -> pd.Index:
        return self.index[self.keep]

    @property
    def tree(self) -> shapely.STRtree:
        if self._tree is None:
            self._tree = shapely.STRtree(self.ea)
        return self._tree


def _as_geoseries(obj: Any, name: str) -> gpd.GeoSeries:
    if isinstance(obj, gpd.GeoDataFrame):
        series = obj.geometry
    elif isinstance(obj, gpd.GeoSeries):
        series = obj
    else:
        raise TypeError(f"{name} must be a GeoDataFrame or GeoSeries, got {type(obj).__name__}")
    if series.crs is None:
        geoms = np.asarray(series.values, dtype=object)
        if not np.all(shapely.is_missing(geoms) | shapely.is_empty(geoms)):
            raise ValueError(
                f"{name} has no CRS; set it with .set_crs(...) so areas and distances are defined"
            )
    return series


def _densify_step(crs: pyproj.CRS) -> float | None:
    """``_DENSIFY_M`` in units of *crs*, or None when its units are unknown.

    For a geographic CRS the step is the angle subtending ``_DENSIFY_M`` on a
    sphere of the WGS 84 equatorial radius: about 50 m along a meridian and at
    most that along a parallel.
    """
    axes = crs.axis_info
    factor = axes[0].unit_conversion_factor if axes else float("nan")
    if not (math.isfinite(factor) and factor > 0):
        return None
    if crs.is_geographic:
        return _DENSIFY_M / (6_378_137.0 * factor)  # factor: radians per unit
    if crs.is_projected:
        return _DENSIFY_M / factor  # factor: metres per unit
    return None


def _to_crs(geoms: np.ndarray, src_crs: Any, dst_crs: Any) -> np.ndarray:
    if len(geoms) == 0:
        return np.asarray(geoms, dtype=object)
    out = gpd.GeoSeries(geoms, crs=src_crs).to_crs(dst_crs)
    return np.asarray(out.values, dtype=object)


def _require_structure_make_valid(n_invalid: int, where: str) -> None:
    """Raise ImportError unless ``shapely.make_valid(method="structure")`` exists.

    The keyword arrived in shapely 2.1 (2.0.x has ``make_valid(geometry,
    **kwargs)`` only) and the method needs GEOS >= 3.10. Falling back to the
    default "linework" method would repair some geometries differently.
    """
    try:
        has_method = "method" in inspect.signature(shapely.make_valid).parameters
    except (TypeError, ValueError):  # signature not introspectable
        has_method = False
    geos = tuple(shapely.geos_version)
    if has_method and geos >= (3, 10, 0):
        return
    raise ImportError(
        f"{where} contains {n_invalid} invalid geometr{'y' if n_invalid == 1 else 'ies'}. "
        "agribound.evaluate repairs "
        "invalid geometries with shapely.make_valid(method='structure'), which needs "
        f"shapely>=2.1 built with GEOS>=3.10 (found shapely {shapely.__version__}, GEOS "
        f"{'.'.join(str(v) for v in geos)}). Upgrade shapely (pip install 'shapely>=2.1'), "
        "or repair the geometries yourself before evaluating."
    )


def _repair(geoms: np.ndarray, where: str) -> tuple[np.ndarray, np.ndarray]:
    """Repair invalid geometries; return (geometries, repaired mask).

    *where* names the geometries in the error raised when shapely is too old.
    """
    invalid = ~shapely.is_missing(geoms) & ~shapely.is_valid(geoms)
    if invalid.any():
        _require_structure_make_valid(int(invalid.sum()), where)
        geoms = geoms.copy()
        geoms[invalid] = shapely.make_valid(
            geoms[invalid], method="structure", keep_collapsed=False
        )
    return geoms, invalid


def _polygonal(geoms: np.ndarray) -> np.ndarray:
    """Keep only polygonal parts; non-polygonal geometries become None."""
    out = np.array(geoms, dtype=object, copy=True)
    type_id = shapely.get_type_id(out)
    for k in np.flatnonzero(type_id == _COLLECTION):
        parts = shapely.get_parts(out[k])
        polys = parts[np.isin(shapely.get_type_id(parts), (_POLYGON, _MULTIPOLYGON))]
        out[k] = shapely.union_all(polys) if len(polys) else None
    out[~np.isin(type_id, (_POLYGON, _MULTIPOLYGON, _COLLECTION))] = None
    return out


def _prepare_layer(obj: Any, name: str, ea_crs: pyproj.CRS) -> _Layer:
    series = _as_geoseries(obj, name)
    geoms = np.asarray(series.values, dtype=object)
    n_input = len(geoms)
    # A layer without a CRS has no geometries (see _as_geoseries); any CRS will do.
    crs = pyproj.CRS.from_user_input(series.crs) if series.crs is not None else ea_crs
    src, repaired_src = _repair(geoms, f"The {name} layer")
    src = _polygonal(src)
    step = _densify_step(crs)
    if step is None and n_input and not np.all(shapely.is_missing(src)):
        logger.warning(
            "The %s CRS has no usable axis unit; edges are not split into pieces "
            "before reprojection to equal_area_crs",
            name,
        )
    dense = shapely.segmentize(src, step) if step is not None and n_input else src
    ea = _to_crs(dense, crs, ea_crs)
    ea, repaired_ea = _repair(ea, f"The {name} layer, reprojected to equal_area_crs,")
    ea = _polygonal(ea)
    area = shapely.area(ea) if n_input else np.zeros(0)
    ok = np.isfinite(area) & (area > 0)
    keep = np.flatnonzero(ok)
    n_dropped = n_input - len(keep)
    n_repaired = int(np.count_nonzero((repaired_src | repaired_ea) & ok))
    if n_dropped:
        logger.warning(
            "%d of %d %s geometries are null, empty, non-polygonal or have zero area "
            "and are excluded from the evaluation",
            n_dropped,
            n_input,
            name,
        )
    if n_repaired:
        logger.warning(
            "%d %s geometries were invalid and were repaired with "
            "shapely.make_valid(method='structure')",
            n_repaired,
            name,
        )
    return _Layer(
        name=name,
        index=series.index,
        keep=keep,
        src=src[keep],
        crs=crs,
        ea=ea[keep],
        area=area[keep].astype(float),
        n_input=n_input,
        n_dropped=n_dropped,
        n_repaired=n_repaired,
    )


def _utm_epsg_array(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Vectorised :func:`agribound.io.crs.utm_epsg` of :func:`utm_zone_for_lon`."""
    wrapped = np.mod(np.asarray(lon, dtype=float) + 180.0, 360.0) - 180.0
    zone = np.mod(np.floor((wrapped + 180.0) / 6.0).astype(np.int64), 60) + 1
    return np.where(np.asarray(lat) < 0, 32700, 32600).astype(np.int64) + zone


def _set_centroids(layer: _Layer, ea_crs: pyproj.CRS) -> None:
    """Set the longitude/latitude (EPSG:4326) of each polygon's centroid."""
    if layer.n == 0:
        return
    centroids = shapely.centroid(layer.ea)
    x = shapely.get_x(centroids)
    y = shapely.get_y(centroids)
    # The transformer GeoSeries.to_crs uses. Transformer.transform tries its
    # scalar path first, which converts a one-element array with float() (a
    # NumPy >= 1.25 DeprecationWarning, an error in later NumPy), so a single
    # point is passed as Python floats.
    transformer = pyproj.Transformer.from_crs(ea_crs, "EPSG:4326", always_xy=True)
    if layer.n == 1:
        lon, lat = transformer.transform(float(x[0]), float(y[0]))
        layer.lon = np.array([lon], dtype=float)
        layer.lat = np.array([lat], dtype=float)
    else:
        lon, lat = transformer.transform(x, y)
        layer.lon = np.asarray(lon, dtype=float)
        layer.lat = np.asarray(lat, dtype=float)


def _assign_zones(layer: _Layer) -> None:
    """Set the UTM zone of each polygon's centroid and the polygons in that zone.

    Needs :func:`_set_centroids` first.
    """
    if layer.n == 0:
        return
    layer.epsg = _utm_epsg_array(layer.lon, layer.lat)
    layer.utm = np.empty(layer.n, dtype=object)
    for epsg in np.unique(layer.epsg):
        sel = np.flatnonzero(layer.epsg == epsg)
        layer.utm[sel] = _to_crs(layer.src[sel], layer.crs, int(epsg))


def _in_zone(layer: _Layer, idx: np.ndarray, epsg: int) -> np.ndarray:
    """Geometries ``layer[idx]`` in UTM zone *epsg* (reusing those already in it)."""
    out = layer.utm[idx].copy()
    other = layer.epsg[idx] != epsg
    if other.any():
        out[other] = _to_crs(layer.src[idx[other]], layer.crs, int(epsg))
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _intersection_area(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return shapely.area(shapely.intersection(a, b))
    except shapely.errors.GEOSException:
        logger.warning(
            "Vectorised intersection failed; recomputing %d intersections one by one", len(a)
        )
    out = np.empty(len(a))
    n_snapped = 0
    for k in range(len(a)):
        try:
            out[k] = shapely.area(shapely.intersection(a[k], b[k]))
        except shapely.errors.GEOSException:
            # Snap both to a 1 mm grid (in the metric CRS) and retry.
            out[k] = shapely.area(shapely.intersection(a[k], b[k], grid_size=1e-3))
            n_snapped += 1
    if n_snapped:
        logger.warning(
            "%d intersections raised a GEOS topology error and were computed on a 1 mm "
            "precision grid",
            n_snapped,
        )
    return out


@dataclass
class _Pairs:
    ref: np.ndarray  # reference positions
    pred: np.ndarray  # prediction positions
    inter: np.ndarray  # intersection area (m²)
    iou: np.ndarray


def _candidate_pairs(ref: _Layer, pred: _Layer) -> _Pairs:
    empty = np.zeros(0, dtype=np.int64)
    if ref.n == 0 or pred.n == 0:
        return _Pairs(empty, empty, np.zeros(0), np.zeros(0))
    ri, pj = pred.tree.query(ref.ea, predicate="intersects")
    order = np.lexsort((pj, ri))
    ri, pj = ri[order].astype(np.int64), pj[order].astype(np.int64)
    inter = _intersection_area(ref.ea[ri], pred.ea[pj]) if len(ri) else np.zeros(0)
    positive = inter > 0
    ri, pj, inter = ri[positive], pj[positive], inter[positive]
    union = ref.area[ri] + pred.area[pj] - inter
    iou = np.clip(inter / union, 0.0, 1.0)
    return _Pairs(ri, pj, inter, iou)


def _first_per_group(keys: np.ndarray) -> np.ndarray:
    """Positions of the first element of each run of equal *keys* (sorted)."""
    if len(keys) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])


def _argmax_per_group(
    group: np.ndarray, value: np.ndarray, other: np.ndarray, n: int
) -> np.ndarray:
    """For each group id in ``0..n-1`` the *other* id of the maximum *value*.

    Ties are broken by the smallest *other* id; groups without elements get -1.
    """
    out = np.full(n, -1, dtype=np.int64)
    if len(group) == 0:
        return out
    order = np.lexsort((other, -value, group))
    first = order[_first_per_group(group[order])]
    out[group[first]] = other[first]
    return out


def _match(
    pairs: _Pairs, n_ref: int, n_pred: int, tau: float, matching: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ref -> matched pred, pred matched?, pred -> home ref among matches)."""
    above = pairs.iou >= tau * (1.0 - _IOU_RTOL)
    r, p, v = pairs.ref[above], pairs.pred[above], pairs.iou[above]
    ref_match = np.full(n_ref, -1, dtype=np.int64)
    pred_match = np.full(n_pred, -1, dtype=np.int64)
    if matching == "one_to_one":
        for k in np.lexsort((p, r, -v)):
            a, b = r[k], p[k]
            if ref_match[a] < 0 and pred_match[b] < 0:
                ref_match[a] = b
                pred_match[b] = a
    else:  # many_to_one
        ref_match = _argmax_per_group(r, v, p, n_ref)
        matched = ref_match >= 0
        mr = np.flatnonzero(matched)
        mp = ref_match[mr]
        # A prediction's home among the references it matched: the one with
        # the highest IoU (ties: lowest reference position).
        pair_iou = {(int(a), int(b)): float(x) for a, b, x in zip(r, p, v, strict=True)}
        iou_mr = np.array([pair_iou[(int(a), int(b))] for a, b in zip(mr, mp, strict=True)])
        pred_match = _argmax_per_group(mp, iou_mr, mr, n_pred)
    return ref_match, pred_match >= 0, pred_match


# ---------------------------------------------------------------------------
# Boundary distances
# ---------------------------------------------------------------------------


def _chunks_by_budget(cost: np.ndarray, budget: float) -> list[tuple[int, int]]:
    """Split ``range(len(cost))`` into consecutive chunks of total cost <= budget."""
    chunks: list[tuple[int, int]] = []
    start, acc = 0, 0.0
    for k, c in enumerate(cost):
        if k > start and acc + c > budget:
            chunks.append((start, k))
            start, acc = k, 0.0
        acc += c
    if start < len(cost):
        chunks.append((start, len(cost)))
    return chunks


def _sample_spacing(length: np.ndarray, fixed: float | None) -> np.ndarray:
    """Maximum sample spacing of boundaries of the given lengths (see :func:`evaluate`)."""
    if fixed is not None:
        return np.full(len(length), fixed)
    return np.maximum(BOUNDARY_SAMPLE_SPACING_M, length / BOUNDARY_MAX_SAMPLES)


def _directed_distance(
    lines: np.ndarray, targets: np.ndarray, spacing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Integral and maximum over each ``lines[i]`` of the distance to ``targets[i]``.

    ``lines`` are sampled at every vertex plus points spaced at most
    ``spacing[i]`` apart; the integral uses the trapezoidal rule.
    """
    n = len(lines)
    integral = np.zeros(n)
    dmax = np.zeros(n)
    if n == 0:
        return integral, dmax
    length = shapely.length(lines)
    cost = np.ceil(length / spacing) + shapely.get_num_coordinates(lines)
    for s, e in _chunks_by_budget(cost, _DISTANCE_CHUNK):
        parts, owner = shapely.get_parts(lines[s:e], return_index=True)
        dense = shapely.segmentize(parts, spacing[s:e][owner])
        coords, part = shapely.get_coordinates(dense, return_index=True)
        if len(coords) == 0:
            continue
        own = owner[part]
        seg = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
        seg[part[1:] != part[:-1]] = 0.0
        weight = np.zeros(len(coords))
        weight[:-1] += 0.5 * seg
        weight[1:] += 0.5 * seg
        dist = shapely.distance(shapely.points(coords), targets[s:e][own])
        integral[s:e] = np.bincount(own, weights=weight * dist, minlength=e - s)
        starts = _first_per_group(own)
        dmax[s + own[starts]] = np.maximum.reduceat(dist, starts)
    return integral, dmax


def _pair_boundary_distances(
    ref: _Layer,
    pred: _Layer,
    ref_idx: np.ndarray,
    pred_idx: np.ndarray,
    fixed_spacing: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hausdorff distance, symmetric mean boundary distance and sample spacing of each pair.

    The spacing returned is the larger of the two boundaries' spacings.
    """
    haus = np.full(len(ref_idx), np.nan)
    mean = np.full(len(ref_idx), np.nan)
    spacing = np.full(len(ref_idx), np.nan)
    if len(ref_idx) == 0:
        return haus, mean, spacing
    zones = ref.epsg[ref_idx]
    for epsg in np.unique(zones):
        sel = np.flatnonzero(zones == epsg)
        ref_lines = shapely.boundary(ref.utm[ref_idx[sel]])
        pred_lines = shapely.boundary(_in_zone(pred, pred_idx[sel], int(epsg)))
        len_r, len_p = shapely.length(ref_lines), shapely.length(pred_lines)
        spacing_r = _sample_spacing(len_r, fixed_spacing)
        spacing_p = _sample_spacing(len_p, fixed_spacing)
        int_r, max_r = _directed_distance(ref_lines, pred_lines, spacing_r)
        int_p, max_p = _directed_distance(pred_lines, ref_lines, spacing_p)
        haus[sel] = np.maximum(max_r, max_p)
        mean[sel] = (int_r + int_p) / (len_r + len_p)
        spacing[sel] = np.maximum(spacing_r, spacing_p)
    return haus, mean, spacing


def _segments(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Straight segments of *lines*: start points, end points, owning line, length.

    Segments of zero length are dropped (they contribute no length).
    """
    parts, owner = shapely.get_parts(lines, return_index=True)
    coords, part = shapely.get_coordinates(parts, return_index=True)
    if len(coords) < 2:
        empty = np.zeros((0, 2))
        return empty, empty, np.zeros(0, dtype=np.int64), np.zeros(0)
    same = part[1:] == part[:-1]
    start, end = coords[:-1][same], coords[1:][same]
    own = owner[part[:-1][same]]
    length = np.hypot(end[:, 0] - start[:, 0], end[:, 1] - start[:, 1])
    ok = length > 0
    return start[ok], end[ok], own[ok], length[ok]


def _slab_interval(
    alpha: np.ndarray, beta: np.ndarray, lo: Any, hi: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Interval of ``t`` with ``lo <= alpha + beta * t <= hi`` (empty: lower > upper)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - alpha) / beta
        t2 = (hi - alpha) / beta
    lower, upper = np.minimum(t1, t2), np.maximum(t1, t2)
    flat = beta == 0
    inside = (alpha >= lo) & (alpha <= hi)
    lower = np.where(flat, np.where(inside, -np.inf, np.inf), lower)
    upper = np.where(flat, np.where(inside, np.inf, -np.inf), upper)
    return lower, upper


def _capsule_interval(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, r: float
) -> tuple[np.ndarray, np.ndarray]:
    """Part of each segment ``a0-a1`` within distance *r* of segment ``b0-b1``.

    The points within *r* of a segment form a capsule, the union of a
    rectangle and two discs; it is convex, so its intersection with the line
    ``a0 + t (a1 - a0)`` is one interval of ``t``: the hull of the intervals
    cut by the rectangle (two slabs) and by each disc (a quadratic). Returns
    that interval clipped to ``[0, 1]``; it is empty (lower >= upper) when the
    segments are farther apart than *r*.
    """
    u = a1 - a0
    uu = np.einsum("ij,ij->i", u, u)
    lower = np.full(len(a0), np.inf)
    upper = np.full(len(a0), -np.inf)
    for centre in (b0, b1):
        d = a0 - centre
        half_b = np.einsum("ij,ij->i", u, d)
        c = np.einsum("ij,ij->i", d, d) - r * r
        disc = half_b * half_b - uu * c
        hit = disc >= 0
        root = np.sqrt(np.where(hit, disc, 0.0))
        lower = np.where(hit, np.minimum(lower, (-half_b - root) / uu), lower)
        upper = np.where(hit, np.maximum(upper, (-half_b + root) / uu), upper)
    w = b1 - b0
    length_b = np.hypot(w[:, 0], w[:, 1])
    has_length = length_b > 0
    w_hat = w / np.where(has_length, length_b, 1.0)[:, None]
    n_hat = np.column_stack([-w_hat[:, 1], w_hat[:, 0]])
    d = a0 - b0
    along_lo, along_hi = _slab_interval(
        np.einsum("ij,ij->i", d, w_hat), np.einsum("ij,ij->i", u, w_hat), 0.0, length_b
    )
    across_lo, across_hi = _slab_interval(
        np.einsum("ij,ij->i", d, n_hat), np.einsum("ij,ij->i", u, n_hat), -r, r
    )
    rect_lo = np.maximum(along_lo, across_lo)
    rect_hi = np.minimum(along_hi, across_hi)
    hit = has_length & (rect_lo <= rect_hi)
    lower = np.where(hit, np.minimum(lower, rect_lo), lower)
    upper = np.where(hit, np.maximum(upper, rect_hi), upper)
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


def _covered_fraction(seg: np.ndarray, lower: np.ndarray, upper: np.ndarray, n: int) -> np.ndarray:
    """Length of the union of the intervals ``[lower, upper] ⊂ [0, 1]`` of each segment id."""
    keep = upper > lower
    seg, lower, upper = seg[keep], lower[keep], upper[keep]
    if len(seg) == 0:
        return np.zeros(n)
    order = np.lexsort((lower, seg))
    seg, lower, upper = seg[order], lower[order], upper[order]
    # Sweep over intervals sorted by start: each adds the part beyond the
    # largest end so far. Offsetting segment k by 2k keeps segments apart.
    offset = 2.0 * seg
    start, end = offset + lower, offset + upper
    reach = np.r_[-np.inf, np.maximum.accumulate(end)[:-1]]
    added = np.maximum(0.0, end - np.maximum(start, reach))
    return np.bincount(seg, weights=added, minlength=n)


def _length_within(lines: np.ndarray, others: np.ndarray, tol: float) -> np.ndarray:
    """Length of each line within distance *tol* of any of *others* (exact)."""
    within = np.zeros(len(lines))
    a0, a1, owner, length = _segments(lines)
    b0, b1, _, _ = _segments(others)
    if len(a0) == 0 or len(b0) == 0:
        return within
    # Only segments of *others* whose bounding box meets the bounding box of
    # all of *lines* grown by tol can be within tol of them. This keeps just
    # the nearby part of a large polygon of *others*.
    lo = np.minimum(a0.min(axis=0), a1.min(axis=0)) - tol
    hi = np.maximum(a0.max(axis=0), a1.max(axis=0)) + tol
    near = np.all(np.minimum(b0, b1) <= hi, axis=1) & np.all(np.maximum(b0, b1) >= lo, axis=1)
    b0, b1 = b0[near], b1[near]
    if len(b0) == 0:
        return within
    tree = shapely.STRtree(shapely.linestrings(np.stack([b0, b1], axis=1)))
    # Envelope query with the segment boxes grown by tol: a superset of the
    # segment pairs within tol (farther pairs get empty intervals below).
    lo_xy, hi_xy = np.minimum(a0, a1) - tol, np.maximum(a0, a1) + tol
    ia, ib = tree.query(shapely.box(lo_xy[:, 0], lo_xy[:, 1], hi_xy[:, 0], hi_xy[:, 1]))
    if len(ia) == 0:
        return within
    lower, upper = _capsule_interval(a0[ia], a1[ia], b0[ib], b1[ib], tol)
    fraction = _covered_fraction(ia, lower, upper, len(a0))
    return np.bincount(owner, weights=fraction * length, minlength=len(lines))


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(starts[k], starts[k] + counts[k])`` over *k*."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = starts - np.r_[0, np.cumsum(counts)[:-1]]
    return np.arange(total, dtype=np.int64) + np.repeat(offsets, counts)


def _local_boundary_mask(mask: _Layer, epsg: int) -> Any:
    """Project original mask vertices just as boundary vertices are projected.

    Equal-area preparation densifies edges for area measurements. Reprojecting
    that densified mask would follow curves instead of the boundary evaluator's
    original straight segments, excluding parts of a coincident perimeter.
    """
    return shapely.union_all(_to_crs(mask.src, mask.crs, epsg)).buffer(0.001)


def _boundary_lengths(layer: _Layer, mask: Any) -> np.ndarray:
    """Original boundary length inside coverage, in each object's metric zone."""
    lines = shapely.boundary(layer.utm)
    if mask is None:
        return shapely.length(lines)
    lengths = np.zeros(layer.n)
    for epsg in np.unique(layer.epsg):
        selected = np.flatnonzero(layer.epsg == epsg)
        local_mask = _local_boundary_mask(mask, int(epsg))
        lengths[selected] = shapely.length(shapely.intersection(lines[selected], local_mask))
    return lengths


def _boundary_within_tolerance(
    layer: _Layer, other: _Layer, tol: float, ea_crs: pyproj.CRS, mask: Any = None
) -> np.ndarray:
    """Per polygon of *layer*: boundary length within *tol* of any *other* boundary."""
    within = np.zeros(layer.n)
    if layer.n == 0 or other.n == 0:
        return within
    # Candidate prefilter in the equal-area CRS: a true distance <= tol is at
    # most tol * (largest local linear scale) there; 10 % margin on top. Only
    # bounding boxes are compared (grown by that distance), which gives a
    # superset of the polygons within it without computing polygon distances,
    # whose cost grows with the vertex count of large polygons. The exact
    # test is the segment-level one in _length_within.
    scale = pyproj.Proj(ea_crs).get_factors(layer.lon, layer.lat).tissot_semimajor
    scale = np.nan_to_num(np.asarray(scale, dtype=float), nan=1e3, posinf=1e3)
    reach = tol * scale * 1.1
    bounds = shapely.bounds(layer.ea)
    grown = shapely.box(
        bounds[:, 0] - reach, bounds[:, 1] - reach, bounds[:, 2] + reach, bounds[:, 3] + reach
    )
    ia, ib = other.tree.query(grown)
    if len(ia) == 0:
        return within
    order = np.argsort(ia, kind="stable")
    ia, ib = ia[order].astype(np.int64), ib[order].astype(np.int64)
    n_cand = np.bincount(ia, minlength=layer.n)
    first_cand = np.r_[0, np.cumsum(n_cand)[:-1]]
    # Work in chunks whose vertex budget (see _SEGMENT_CHUNK) keeps the
    # segment pairs of a chunk in memory. A candidate shared by many polygons
    # (a large prediction overlapping many reference fields) is charged in
    # shares, so those polygons can share a chunk and its boundary is split
    # into segments once per chunk rather than once per polygon.
    other_coords = shapely.get_num_coordinates(other.utm).astype(float)
    degree = np.bincount(ib, minlength=other.n)
    cost = shapely.get_num_coordinates(layer.utm) + np.bincount(
        ia, weights=other_coords[ib] / degree[ib], minlength=layer.n
    )
    for epsg in np.unique(layer.epsg):
        sel = np.flatnonzero((layer.epsg == epsg) & (n_cand > 0))
        if len(sel) == 0:
            continue
        local_mask = _local_boundary_mask(mask, int(epsg)) if mask is not None else None
        # Hilbert-curve order keeps each chunk compact, so the bounding-box
        # filter in _length_within drops the far parts of large candidates.
        hilbert = gpd.GeoSeries(layer.utm[sel]).hilbert_distance().to_numpy()
        sel = sel[np.argsort(hilbert, kind="stable")]
        for s, e in _chunks_by_budget(cost[sel], _SEGMENT_CHUNK):
            chunk = sel[s:e]
            cand = np.unique(ib[_ranges(first_cand[chunk], n_cand[chunk])])
            lines = shapely.boundary(layer.utm[chunk])
            others = shapely.boundary(_in_zone(other, cand, int(epsg)))
            if local_mask is not None:
                lines = shapely.intersection(lines, local_mask)
                others = shapely.intersection(others, local_mask)
            within[chunk] = _length_within(lines, others, tol)
    return within


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


@dataclass
class _Groups:
    codes: np.ndarray  # per evaluated reference field; -1 = no group
    keys: list[str]
    values: list[Any] | None = None  # strata: original values per key
    per_ref_values: np.ndarray | None = None  # strata: value per reference field
    name: str | None = None
    edges: np.ndarray | None = None  # size classes (ha)
    n_missing: int = 0


def _resolve_strata(strata: Any, reference: Any, ref: _Layer) -> _Groups | None:
    if strata is None:
        return None
    if isinstance(strata, str):
        if not isinstance(reference, gpd.GeoDataFrame) or strata not in reference.columns:
            columns = list(reference.columns) if isinstance(reference, gpd.GeoDataFrame) else []
            raise ValueError(
                f"strata column {strata!r} not found in reference (columns: {columns})"
            )
        values = reference[strata].to_numpy(dtype=object)
        name: str | None = strata
    elif isinstance(strata, pd.Series):
        if strata.index.equals(ref.index):
            aligned = strata
        elif strata.index.is_unique and ref.index.isin(strata.index).all():
            aligned = strata.reindex(ref.index)
        else:
            raise ValueError(
                "strata Series must be indexed like reference (every reference index label "
                "present, labels unique)"
            )
        values = aligned.to_numpy(dtype=object)
        name = None if strata.name is None else str(strata.name)
    else:
        values = np.asarray(strata, dtype=object)
        if values.ndim != 1 or len(values) != ref.n_input:
            raise ValueError(
                f"strata must have one value per reference row ({ref.n_input}), "
                f"got shape {values.shape}"
            )
        name = None
    per_ref = values[ref.keep]
    missing = np.asarray(pd.isna(per_ref), dtype=bool)
    unique = list(pd.unique(pd.Series(per_ref[~missing], dtype=object)))
    try:
        unique = sorted(unique)
    except TypeError:
        unique = sorted(unique, key=str)
    keys = [str(v) for v in unique]
    if len(set(keys)) != len(keys):
        raise ValueError("strata values must be distinct after conversion to str")
    lookup = {v: k for k, v in enumerate(unique)}
    codes = np.full(len(per_ref), -1, dtype=np.int64)
    for k in np.flatnonzero(~missing):
        codes[k] = lookup[per_ref[k]]
    return _Groups(
        codes=codes,
        keys=keys,
        values=unique,
        per_ref_values=per_ref,
        name=name,
        n_missing=int(missing.sum()),
    )


def _auto_size_edges(area_ha: np.ndarray) -> np.ndarray:
    lo, hi = float(area_ha.min()), float(area_ha.max())
    k0 = math.floor(math.log10(lo)) - 1
    k1 = math.ceil(math.log10(hi)) + 1
    ladder = sorted(m * 10.0**k for k in range(k0, k1 + 1) for m in _AUTO_BIN_MANTISSAS)
    ladder = np.array([float(f"{e:.12g}") for e in ladder])
    first = ladder[ladder <= lo].max()
    last = ladder[ladder > hi].min()
    return ladder[(ladder >= first) & (ladder <= last)]


def _resolve_size_bins(size_bins: Any, area_m2: np.ndarray) -> _Groups | None:
    if size_bins is None:
        return None
    area_ha = area_m2 / 1e4
    if isinstance(size_bins, str):
        if size_bins != "auto":
            raise ValueError(f"size_bins must be 'auto' or a sequence of edges, got {size_bins!r}")
        edges = _auto_size_edges(area_ha) if len(area_ha) else np.zeros(0)
    else:
        try:
            edges = np.asarray(size_bins, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"size_bins edges must be numbers, got {size_bins!r}") from None
        if (
            edges.ndim != 1
            or len(edges) < 2
            or np.any(np.diff(edges) <= 0)
            or edges[0] < 0
            or not np.all(np.isfinite(edges[:-1]))
            or np.isnan(edges[-1])
        ):
            raise ValueError(
                "size_bins must be at least two strictly increasing, non-negative edges in "
                f"hectares (the last may be inf), got {size_bins!r}"
            )
    codes = np.full(len(area_ha), -1, dtype=np.int64)
    keys: list[str] = []
    if len(edges) >= 2:
        n_bins = len(edges) - 1
        codes = np.searchsorted(edges, area_ha, side="right").astype(np.int64) - 1
        codes[area_ha == edges[-1]] = n_bins - 1
        codes[(codes < 0) | (codes >= n_bins)] = -1
        keys = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
        if len(set(keys)) != len(keys):  # edges that differ beyond 6 significant digits
            keys = [
                f"{float(lo)!r}-{float(hi)!r}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)
            ]
    return _Groups(codes=codes, keys=keys, edges=edges, n_missing=int(np.sum(codes < 0)))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    params: _Params
    ref: _Layer
    pred: _Layer
    ref_match: np.ndarray
    pred_matched: np.ndarray
    pred_home: np.ndarray  # reference position or -1
    iou: np.ndarray  # per reference (NaN if unmatched)
    pred_area_matched: np.ndarray  # per reference (NaN if unmatched)
    best_iou: np.ndarray
    n_overlap: np.ndarray
    max_overlap_pred: np.ndarray
    os: np.ndarray
    us: np.ndarray
    hausdorff: np.ndarray
    mean_dist: np.ndarray
    spacing: np.ndarray  # boundary sample spacing of the matched pair (NaN if unmatched)
    perim_ref: np.ndarray
    within_ref: np.ndarray | None
    within_tol: np.ndarray | None
    Q: np.ndarray  # (n_ref, len(_QUANTITIES))
    unassigned: np.ndarray  # (len(_QUANTITIES),) sums of predictions without a home
    strata: _Groups | None
    size: _Groups | None


def _run(
    predicted: Any,
    reference: Any,
    params: _Params,
    *,
    strata: Any,
    size_bins: Any,
    boundary_mask: Any = None,
) -> _Run:
    ea_crs = params.ea_crs
    mask = None
    if boundary_mask is not None:
        if params.tolerance is None:
            raise ValueError("boundary_mask requires boundary_tolerance_m")
        series = _as_geoseries(boundary_mask, "boundary_mask")
        if series.empty or not series.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError("boundary_mask requires nonempty polygon geometries")
        layer = _prepare_layer(series, "boundary_mask", ea_crs)
        if layer.n != len(series):
            raise ValueError("boundary_mask contains missing, empty or collapsed geometries")
        mask = layer
    ref = _prepare_layer(reference, "reference", ea_crs)
    pred = _prepare_layer(predicted, "predicted", ea_crs)
    strata_groups = _resolve_strata(strata, reference, ref)
    size_groups = _resolve_size_bins(size_bins, ref.area)
    n_ref, n_pred = ref.n, pred.n

    pairs = _candidate_pairs(ref, pred)
    if n_ref and n_pred and len(pairs.ref) == 0:
        logger.warning(
            "No predicted polygon overlaps any reference polygon (%d predicted, %d reference); "
            "check that both layers carry the correct CRS and cover the same area",
            n_pred,
            n_ref,
        )
    ref_match, pred_matched, pred_home_matched = _match(
        pairs, n_ref, n_pred, params.iou_threshold, params.matching
    )
    matched = ref_match >= 0

    # Per-reference overlap statistics.
    best_iou = np.zeros(n_ref)
    np.maximum.at(best_iou, pairs.ref, pairs.iou)
    n_overlap = np.bincount(pairs.ref, minlength=n_ref).astype(np.int64)
    max_overlap_pred = _argmax_per_group(pairs.ref, pairs.inter, pairs.pred, n_ref)
    max_overlap_ref = _argmax_per_group(pairs.pred, pairs.inter, pairs.ref, n_pred)
    inter_max = np.zeros(n_ref)
    np.maximum.at(inter_max, pairs.ref, pairs.inter)
    has_overlap = max_overlap_pred >= 0
    os_ = 1.0 - inter_max / np.where(ref.area > 0, ref.area, 1.0)
    us_ = np.full(n_ref, np.nan)
    us_[has_overlap] = 1.0 - inter_max[has_overlap] / pred.area[max_overlap_pred[has_overlap]]
    os_ = np.clip(os_, 0.0, 1.0)
    us_ = np.clip(us_, 0.0, 1.0)

    above = pairs.iou >= params.iou_threshold * (1.0 - _IOU_RTOL)
    n_above_ref = np.bincount(pairs.ref[above], minlength=n_ref)
    n_above_pred = np.bincount(pairs.pred[above], minlength=n_pred)

    iou = np.full(n_ref, np.nan)
    pair_lookup = {
        (int(a), int(b)): float(x) for a, b, x in zip(pairs.ref, pairs.pred, pairs.iou, strict=True)
    }
    mr = np.flatnonzero(matched)
    iou[mr] = [pair_lookup[(int(a), int(ref_match[a]))] for a in mr]
    pred_area_matched = np.full(n_ref, np.nan)
    pred_area_matched[mr] = pred.area[ref_match[mr]]

    pred_home = np.where(pred_matched, pred_home_matched, max_overlap_ref)

    for layer in (ref, pred):
        _set_centroids(layer, ea_crs)
        _assign_zones(layer)
    _check_equal_area(ea_crs, (ref, pred))

    # Boundary distances of matched pairs (UTM zone of the reference field).
    hausdorff = np.full(n_ref, np.nan)
    mean_dist = np.full(n_ref, np.nan)
    spacing = np.full(n_ref, np.nan)
    hausdorff[mr], mean_dist[mr], spacing[mr] = _pair_boundary_distances(
        ref, pred, mr, ref_match[mr], params.spacing
    )
    perim_ref = _boundary_lengths(ref, mask)

    tol = params.tolerance
    within_ref = within_tol = None
    within_pred = np.zeros(n_pred)
    perim_pred = np.zeros(n_pred)
    if tol is not None:
        within_ref = _boundary_within_tolerance(ref, pred, tol, ea_crs, mask)
        within_pred = _boundary_within_tolerance(pred, ref, tol, ea_crs, mask)
        perim_pred = _boundary_lengths(pred, mask)
        within_tol = matched & (np.nan_to_num(mean_dist, nan=np.inf) <= tol)

    # Per-reference quantity matrix (see _QUANTITIES).
    q = np.zeros((n_ref, len(_QUANTITIES)))
    m = matched.astype(float)
    q[:, _QI["n_ref"]] = 1.0
    q[:, _QI["area_ref"]] = ref.area
    q[:, _QI["tp"]] = m
    q[:, _QI["iou_tp"]] = np.where(matched, iou, 0.0)
    q[:, _QI["best_iou"]] = best_iou
    q[:, _QI["abs_area_err_tp"]] = np.where(matched, np.abs(pred_area_matched - ref.area), 0.0)
    q[:, _QI["signed_area_err_tp"]] = np.where(matched, pred_area_matched - ref.area, 0.0)
    q[:, _QI["area_ref_tp"]] = ref.area * m
    q[:, _QI["os"]] = os_
    q[:, _QI["us"]] = np.nan_to_num(us_, nan=0.0)
    q[:, _QI["us_defined"]] = np.isfinite(us_)
    q[:, _QI["legacy_overseg"]] = n_above_ref > 1
    has_dist = matched & np.isfinite(mean_dist)
    q[:, _QI["n_dist"]] = has_dist
    q[:, _QI["hausdorff"]] = np.where(has_dist, hausdorff, 0.0)
    q[:, _QI["mean_dist"]] = np.where(has_dist, mean_dist, 0.0)
    if tol is not None:
        q[:, _QI["perim_ref"]] = perim_ref
        q[:, _QI["within_len_ref"]] = within_ref
        q[:, _QI["within_tol"]] = within_tol
        q[:, _QI["area_within_tol"]] = ref.area * within_tol

    pred_values = {
        "n_pred": np.ones(n_pred),
        "n_pred_matched": pred_matched.astype(float),
        "area_pred": pred.area,
        "area_pred_matched": pred.area * pred_matched,
        "legacy_underseg": (n_above_pred > 1).astype(float),
        "perim_pred": perim_pred,
        "within_len_pred": within_pred,
    }
    unassigned = np.zeros(len(_QUANTITIES))
    homed = pred_home >= 0
    for name, values in pred_values.items():
        q[:, _QI[name]] = np.bincount(pred_home[homed], weights=values[homed], minlength=n_ref)
        unassigned[_QI[name]] = float(values[~homed].sum())

    return _Run(
        params=params,
        ref=ref,
        pred=pred,
        ref_match=ref_match,
        pred_matched=pred_matched,
        pred_home=pred_home,
        iou=iou,
        pred_area_matched=pred_area_matched,
        best_iou=best_iou,
        n_overlap=n_overlap,
        max_overlap_pred=max_overlap_pred,
        os=os_,
        us=us_,
        hausdorff=hausdorff,
        mean_dist=mean_dist,
        spacing=spacing,
        perim_ref=perim_ref,
        within_ref=within_ref,
        within_tol=within_tol,
        Q=q,
        unassigned=unassigned,
        strata=strata_groups,
        size=size_groups,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _ratio(num: Any, den: Any, empty: float) -> np.ndarray:
    """``num / den`` with *empty* where ``den == 0``; NaN inputs propagate."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(den == 0, empty, out)


def _f1(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Harmonic mean; 0 when ``p + r == 0``, NaN when *p* or *r* is NaN."""
    return _ratio(2.0 * p * r, p + r, 0.0)


def _metrics_from_sums(
    s: np.ndarray, tolerance: bool, empty_rate: float = 0.0
) -> dict[str, np.ndarray]:
    """Metrics from quantity sums; *s* has shape ``(..., len(_QUANTITIES))``.

    Rates with a zero denominator take *empty_rate* (0.0 for reported point
    estimates, NaN inside bootstrap resamples); means with a zero denominator
    are NaN.
    """

    def rate(num: Any, den: Any) -> np.ndarray:
        return _ratio(num, den, empty_rate)

    def mean(num: Any, den: Any) -> np.ndarray:
        return _ratio(num, den, np.nan)

    def q(name: str) -> np.ndarray:
        return s[..., _QI[name]]

    tp = q("tp")
    n_ref = q("n_ref")
    n_pred = q("n_pred")
    fp = n_pred - q("n_pred_matched")
    precision = rate(tp, tp + fp)
    recall = rate(tp, n_ref)
    aw_precision = rate(q("area_pred_matched"), q("area_pred"))
    aw_recall = rate(q("area_ref_tp"), q("area_ref"))
    out = {
        "iou_mean": rate(q("iou_tp"), tp),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "over_segmentation": rate(q("legacy_overseg"), n_ref),
        "under_segmentation": rate(q("legacy_underseg"), n_pred),
        "area_error_mean_m2": rate(q("abs_area_err_tp"), tp),
        "count_predicted": n_pred,
        "count_reference": n_ref,
        "count_tp": tp,
        "count_fp": fp,
        "count_fn": n_ref - tp,
        "best_iou_mean": mean(q("best_iou"), n_ref),
        "area_weighted_precision": aw_precision,
        "area_weighted_recall": aw_recall,
        "area_weighted_f1": _f1(aw_precision, aw_recall),
        "reference_area_ha": q("area_ref") / 1e4,
        "predicted_area_ha": q("area_pred") / 1e4,
        "area_error_signed_mean_m2": mean(q("signed_area_err_tp"), tp),
        "oversegmentation_mean": mean(q("os"), n_ref),
        "undersegmentation_mean": mean(q("us"), q("us_defined")),
        "hausdorff_mean_m": mean(q("hausdorff"), q("n_dist")),
        "boundary_distance_mean_m": mean(q("mean_dist"), q("n_dist")),
    }
    if tolerance:
        b_precision = rate(q("within_len_pred"), q("perim_pred"))
        b_recall = rate(q("within_len_ref"), q("perim_ref"))
        out.update(
            {
                "boundary_precision": b_precision,
                "boundary_recall": b_recall,
                "boundary_f1": _f1(b_precision, b_recall),
                "coverage_within_tolerance": rate(q("within_tol"), n_ref),
                "coverage_within_tolerance_area": rate(q("area_within_tol"), q("area_ref")),
            }
        )
    return out


def _nanmedian(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def _to_builtin(metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        value = float(value)
        out[key] = int(round(value)) if key in _COUNT_KEYS else value
    return out


def _group_metrics(run: _Run, idx: np.ndarray | None) -> dict[str, Any]:
    """Metrics of the reference fields *idx* (all fields plus unassigned if None)."""
    tolerance = run.params.tolerance is not None
    if idx is None:
        sums = run.Q.sum(axis=0) + run.unassigned
        sel = np.arange(run.ref.n)
    else:
        sums = run.Q[idx].sum(axis=0)
        sel = idx
    # A group without reference fields (an empty size class) has no defined
    # rates; the 0.0 convention would misreport it as a total failure.
    empty_rate = np.nan if idx is not None and len(idx) == 0 else 0.0
    metrics = _to_builtin(_metrics_from_sums(sums, tolerance, empty_rate=empty_rate))
    matched = sel[run.ref_match[sel] >= 0]
    metrics["median_area_ha"] = _nanmedian(run.ref.area[sel] / 1e4)
    metrics["hausdorff_median_m"] = _nanmedian(run.hausdorff[matched])
    metrics["boundary_distance_median_m"] = _nanmedian(run.mean_dist[matched])
    return metrics


def _summarise(run: _Run) -> dict[str, Any]:
    params = run.params
    metrics = _group_metrics(run, None)
    both_empty = run.ref.n == 0 and run.pred.n == 0
    if both_empty:
        for key in _RATE_KEYS:
            if key in metrics:
                metrics[key] = 1.0
    metrics["iou_threshold"] = params.iou_threshold
    metrics["matching"] = params.matching
    metrics["equal_area_crs"] = _crs_label(params.ea_crs)
    metrics["distance_crs"] = "WGS 84 / UTM zone of each polygon's centroid"
    metrics["boundary_sample_spacing_m"] = params.spacing
    matched_spacing = run.spacing[np.isfinite(run.spacing)]
    metrics["boundary_sample_spacing_max_m"] = (
        float(matched_spacing.max()) if len(matched_spacing) else float("nan")
    )
    zones = np.unique(np.r_[run.ref.epsg, run.pred.epsg]).astype(int)
    metrics["utm_epsg_codes"] = [int(z) for z in zones]
    metrics["count_fp_unassigned"] = int(
        round(
            run.unassigned[_QI["n_pred"]] - run.unassigned[_QI["n_pred_matched"]],
        )
    )
    metrics["count_reference_dropped"] = run.ref.n_dropped
    metrics["count_predicted_dropped"] = run.pred.n_dropped
    metrics["count_reference_repaired"] = run.ref.n_repaired
    metrics["count_predicted_repaired"] = run.pred.n_repaired
    if params.tolerance is not None:
        metrics["boundary_tolerance_m"] = params.tolerance
    if run.strata is not None:
        metrics["strata_column"] = run.strata.name
        metrics["count_reference_no_stratum"] = run.strata.n_missing
        per_stratum: dict[str, Any] = {}
        for k, key in enumerate(run.strata.keys):
            members = _members(run.strata, k)
            per_stratum[key] = {"n": len(members), **_group_metrics(run, members)}
        metrics["per_stratum"] = per_stratum
    if run.size is not None:
        edges = run.size.edges if run.size.edges is not None else np.zeros(0)
        metrics["size_class_edges_ha"] = [float(e) for e in edges]
        metrics["count_reference_outside_size_classes"] = run.size.n_missing
        per_size: dict[str, Any] = {}
        for k, key in enumerate(run.size.keys):
            members = _members(run.size, k)
            per_size[key] = {
                "lower_ha": float(edges[k]),
                "upper_ha": float(edges[k + 1]),
                "n": len(members),
                **_group_metrics(run, members),
            }
        metrics["per_size_class"] = per_size
    return metrics


def _members(groups: _Groups, code: int) -> np.ndarray:
    return np.flatnonzero(groups.codes == code)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _percentile_ci(values: np.ndarray) -> tuple[list[float], int]:
    finite = values[np.isfinite(values)]
    n_undefined = int(len(values) - len(finite))
    if len(finite) == 0:
        return [float("nan"), float("nan")], n_undefined
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    lo, hi = np.percentile(finite, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return [float(lo), float(hi)], n_undefined


def _add_bootstrap(metrics: dict[str, Any], run: _Run) -> None:
    params = run.params
    n_boot = params.bootstrap
    tolerance = params.tolerance is not None
    names = _CI_METRICS + (_CI_METRICS_TOLERANCE if tolerance else ())
    resampling = (
        "reference fields with replacement within strata"
        if run.strata is not None
        else "reference fields with replacement"
    )
    info: dict[str, Any] = {
        "n_resamples": n_boot,
        "seed": params.bootstrap_seed,
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "method": "percentile",
        "resampling": resampling,
        "ci": {},
        "n_undefined": {},
    }
    metrics["bootstrap"] = info
    n = run.ref.n
    if n == 0:
        logger.warning("No reference fields to resample; bootstrap intervals are not computed")
        return

    # One resampling pool per stratum (missing strata form their own pool).
    pool_codes = run.strata.codes if run.strata is not None else np.zeros(n, dtype=np.int64)
    pools = [np.flatnonzero(pool_codes == c) for c in np.unique(pool_codes)]

    targets: list[tuple[str, str | None, np.ndarray | None]] = [("overall", None, None)]
    if run.strata is not None:
        targets += [
            ("per_stratum", key, _members(run.strata, k)) for k, key in enumerate(run.strata.keys)
        ]
    if run.size is not None:
        targets += [
            ("per_size_class", key, _members(run.size, k)) for k, key in enumerate(run.size.keys)
        ]
    replicates = {
        (kind, key): {name: np.empty(n_boot) for name in names} for kind, key, _ in targets
    }

    children = np.random.SeedSequence(params.bootstrap_seed).spawn(n_boot)
    chunk = max(1, min(n_boot, _BOOTSTRAP_CHUNK_CELLS // n))
    for c0 in range(0, n_boot, chunk):
        c1 = min(n_boot, c0 + chunk)
        weights = np.zeros((c1 - c0, n))
        for b in range(c0, c1):
            rng = np.random.default_rng(children[b])
            for pool in pools:
                draw = rng.integers(0, len(pool), size=len(pool))
                weights[b - c0, pool] = np.bincount(draw, minlength=len(pool))
        for kind, key, idx in targets:
            sums = weights @ run.Q + run.unassigned if idx is None else weights[:, idx] @ run.Q[idx]
            values = _metrics_from_sums(sums, tolerance, empty_rate=np.nan)
            store = replicates[(kind, key)]
            for name in names:
                store[name][c0:c1] = values[name]

    for kind, key, _ in targets:
        ci: dict[str, list[float]] = {}
        undefined: dict[str, int] = {}
        for name in names:
            ci[name], n_undef = _percentile_ci(replicates[(kind, key)][name])
            if n_undef:
                undefined[name] = n_undef
        if kind == "overall":
            info["ci"] = ci
            info["n_undefined"] = undefined
        else:
            metrics[kind][key]["ci"] = ci
            if undefined:
                metrics[kind][key]["ci_n_undefined"] = undefined


# ---------------------------------------------------------------------------
# Per-field table
# ---------------------------------------------------------------------------


def _labels_or_none(index: pd.Index, positions: np.ndarray) -> list[Any]:
    return [index[p] if p >= 0 else None for p in positions]


def _frame(run: _Run) -> pd.DataFrame:
    ref, pred = run.ref, run.pred
    labels = ref.labels
    matched = run.ref_match >= 0
    data: dict[str, Any] = {
        "area_m2": ref.area,
        "area_ha": ref.area / 1e4,
        "perimeter_m": run.perim_ref,
        "utm_epsg": ref.epsg if ref.n else np.zeros(0, dtype=np.int64),
    }
    if run.strata is not None:
        data["stratum"] = pd.array(run.strata.per_ref_values, dtype=object)
    if run.size is not None:
        data["size_class"] = [run.size.keys[c] if c >= 0 else None for c in run.size.codes]
    data.update(
        {
            "matched": matched,
            "pred_index": pd.array(_labels_or_none(pred.labels, run.ref_match), dtype=object),
            "iou": run.iou,
            "pred_area_m2": run.pred_area_matched,
            "best_iou": run.best_iou,
            "n_overlapping_pred": run.n_overlap,
            "max_overlap_pred_index": pd.array(
                _labels_or_none(pred.labels, run.max_overlap_pred), dtype=object
            ),
            "oversegmentation": run.os,
            "undersegmentation": run.us,
            "hausdorff_m": run.hausdorff,
            "boundary_mean_distance_m": run.mean_dist,
            "boundary_sample_spacing_m": run.spacing,
        }
    )
    if run.params.tolerance is not None:
        data["boundary_within_tolerance_m"] = run.within_ref
        data["boundary_recall"] = _ratio(run.within_ref, run.perim_ref, np.nan)
        data["within_tolerance"] = run.within_tol
    frame = pd.DataFrame(data, index=labels)
    frame.attrs.update(
        {
            "iou_threshold": run.params.iou_threshold,
            "matching": run.params.matching,
            "equal_area_crs": _crs_label(run.params.ea_crs),
            "boundary_tolerance_m": run.params.tolerance,
            "boundary_sample_spacing_m": run.params.spacing,
        }
    )
    return frame
