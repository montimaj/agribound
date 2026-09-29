# Evaluation

`agribound.evaluate.evaluate(predicted, reference, ...)` compares predicted
field polygons with reference field polygons at the object level and returns
a dictionary of metrics. `evaluate_frame` returns the per-reference-field table
the summary is computed from, and `pixels_per_field` the number of pixels a
field spans at a given ground sample distance. The same function backs the
`agribound evaluate` command and the pipeline's automatic evaluation.

```python
from agribound.evaluate import evaluate
from agribound.io import read_vector

metrics = evaluate(
    read_vector("fields.gpkg"),
    read_vector("reference.gpkg"),
    iou_threshold=0.5,
    strata="county",  # a reference column
    size_bins=[0, 0.5, 1, 2, 5, 10, float("inf")],  # hectares
    bootstrap=1000,
    boundary_tolerance_m=10.0,
)
print(metrics["f1"], metrics["area_weighted_recall"], metrics["bootstrap"]["ci"]["recall"])
```

```bash
agribound evaluate -p fields.gpkg -r reference.gpkg --strata-column county \
    --size-bins 0,0.5,1,2,5,10,inf --bootstrap 1000 --boundary-tolerance-m 10 -o metrics.json
```

Example 20
([`examples/20_stratified_evaluation.py`](https://github.com/montimaj/agribound/blob/main/examples/20_stratified_evaluation.py))
uses these functions against the NMOSE registry in the San Juan Basin: strata,
size classes, bootstrap intervals, the 10 m boundary tolerance, the per-field
table and pixels per field. It also compares Delineate-Anything v2 on Landsat,
Sentinel-2, SPOT 6/7 and NAIP of one year (`--resolution-year`, default 2018;
`--no-resolution` skips it) and the pre-trained FTW and Delineate-Anything
engines with and without the LULC crop filter (`--no-lulc-comparison` skips
it); `--predicted PATH` evaluates an existing layer and skips both
comparisons.

## In the pipeline

When `reference_boundaries` is set and `fine_tune` is False, `delineate()`
calls `evaluate(output, reference)` with the defaults (IoU 0.5, one-to-one
matching, no tolerance metrics) after the metadata step. With a study area,
the reference polygons are first selected with the same `aoi_selection` rule
as the predictions (with `"none"`: the references that intersect the study
area), so a field crossing the outline is kept or dropped on both sides alike.
The metrics are stored in `gdf.attrs["evaluation_metrics"]` and in the
provenance record (`facts.evaluation`, with the reference counts in
`facts.evaluation_reference`).

!!! warning "Incomplete reference layers"
    Predictions that overlap no reference field count as false positives. If
    the reference covers only some fields in the area (a registry of some
    fields), precision is not meaningful; use recall, or the precision among
    predictions that overlap a reference field,
    `count_tp / (count_tp + count_fp - count_fp_unassigned)`.

## Matching

A prediction `p` and a reference field `r` can match when
`IoU(p, r) = |p ∩ r| / |p ∪ r| >= iou_threshold` (compared with a relative
tolerance of 1e-9) and the intersection has positive area.

- `matching="one_to_one"` (default since 1.0): all candidate pairs at or above
  the threshold are sorted by descending IoU (ties by reference, then
  prediction position), and a pair is accepted when neither polygon has been
  matched yet. This is a greedy assignment in the spirit of COCO-style
  detection evaluation (processed in order of IoU, since there are no
  confidence scores), not an optimal assignment. With non-overlapping
  polygons and `iou_threshold > 0.5` the matching is unique.
- `matching="many_to_one"`: the 0.1.x rule. Each reference field is matched to
  its highest-IoU prediction at or above the threshold, and one prediction may
  match several reference fields, so precision can exceed the fraction of
  predictions that are matched.

TP = matched reference fields, FP = unmatched predictions, FN = unmatched
reference fields.

## Summary metrics

Kept from 0.1.x:

| Key | Definition |
|---|---|
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1` | 2 · precision · recall / (precision + recall) |
| `iou_mean` | mean IoU of matched pairs (0.0 when nothing matched) |
| `over_segmentation` | fraction of reference fields with more than one prediction at IoU ≥ threshold (0.1.x definition) |
| `under_segmentation` | fraction of predictions with more than one reference field at IoU ≥ threshold (0.1.x definition) |
| `area_error_mean_m2` | mean absolute area difference of matched pairs |
| `count_predicted`, `count_reference`, `count_tp`, `count_fp`, `count_fn`, `iou_threshold` | counts and the threshold |

With non-overlapping polygons and a threshold above 0.5, the two 0.1.x
segmentation keys are always 0; use `oversegmentation_mean` and
`undersegmentation_mean` below to measure splits and merges.

Added in 1.0:

| Key | Definition |
|---|---|
| `best_iou_mean` | mean over all reference fields of the highest IoU with any prediction (0 when none overlaps); not conditional on a match |
| `area_weighted_recall` | area of matched reference fields / area of all reference fields |
| `area_weighted_precision` | area of matched predictions / area of all predictions |
| `area_weighted_f1` | harmonic mean of the two |
| `reference_area_ha`, `predicted_area_ha`, `median_area_ha` | total areas and the median reference field area |
| `area_error_signed_mean_m2` | mean of (predicted − reference) area over matched pairs |
| `oversegmentation_mean`, `undersegmentation_mean` | means over reference fields of `OS = 1 − |r ∩ p′| / |r|` and `US = 1 − |r ∩ p′| / |p′|`, where `p′` is the prediction with the largest intersection with `r` (the maximum-overlap pairing of Persello & Bruzzone, 2010; the form of Clinton et al.'s, 2010, measures). A reference field that no prediction overlaps has OS = 1 and no US. |
| `hausdorff_mean_m`, `hausdorff_median_m` | Hausdorff distance between the boundaries of matched pairs |
| `boundary_distance_mean_m`, `boundary_distance_median_m` | symmetric mean boundary distance of matched pairs, `(∫∂r d(x, ∂p) dx + ∫∂p d(x, ∂r) dx) / (|∂r| + |∂p|)` |
| `boundary_sample_spacing_m`, `boundary_sample_spacing_max_m` | the sampling argument, and the largest spacing Δ used for any matched pair |
| `count_fp_unassigned` | false positives that overlap no reference field |
| `count_*_dropped`, `count_*_repaired` | rows excluded (null, empty, non-polygonal, zero area) and invalid geometries repaired |
| `matching`, `equal_area_crs`, `distance_crs`, `utm_epsg_codes` | how the metrics were computed |

With `boundary_tolerance_m`:

| Key | Definition |
|---|---|
| `boundary_recall` | length of reference boundaries within the tolerance of any predicted boundary / total reference boundary length |
| `boundary_precision` | the same for predicted boundaries against reference boundaries |
| `boundary_f1` | harmonic mean of the two |
| `coverage_within_tolerance` | fraction of reference fields that are matched **and** have a symmetric mean boundary distance ≤ tolerance |
| `coverage_within_tolerance_area` | the same, weighted by reference area |

Boundary lengths are summed per polygon (an edge shared by two fields counts
once for each). The within-tolerance lengths are computed exactly (the length
of one layer's boundary inside the union of circular-arc buffers of the other
layer's boundaries), without polygonal buffer approximation.

Empty denominators: rates (keys ending in `precision`, `recall`, `f1`,
`iou_mean`, the coverage keys) and the 0.1.x keys `over_segmentation`,
`under_segmentation`, `area_error_mean_m2` are 0.0; other means and distances
are NaN. When both layers are empty the rates are 1.0 (0.1.x behaviour).

## Strata, size classes and confidence intervals

- `strata`: a reference column name, an aligned Series or an array. Metrics
  are also reported per stratum (`per_stratum`); rows with a missing value
  count only in the overall metrics.
- `size_bins`: `"auto"` (1-2-5 series spanning the reference areas) or
  strictly increasing edges in **hectares** of the reference field area;
  classes are `[lower, upper)` except the last, `[lower, upper]`. Metrics are
  also reported per class (`per_size_class`).
- In a group, a prediction counts with its *home* reference field: the matched
  one, else the one it overlaps most. Predictions overlapping no reference
  field count only in the overall metrics.
- `bootstrap=N`: percentile bootstrap intervals at 95 %
  (`BOOTSTRAP_CONFIDENCE_LEVEL`), resampling reference fields with replacement
  (each stratum separately when `strata` is given), from child seeds of
  `numpy.random.SeedSequence(bootstrap_seed)` (default 42). The intervals treat
  the reference fields as an independent (stratified) random sample: they do
  not model spatial autocorrelation, so they are too narrow when errors are
  spatially clustered, and they are not design-based estimates for a
  probability sample (Stehman & Foody, 2019).

## Geometry and coordinate systems

- Invalid geometries are repaired with
  `shapely.make_valid(method="structure", keep_collapsed=False)` and only their
  polygonal parts are kept. This needs shapely >= 2.1 (GEOS >= 3.10); with an
  older shapely an `ImportError` is raised when a repair is needed (there is
  no fallback to another repair method).
- Areas and IoU are computed in `equal_area_crs` (default EPSG:6933). Edges are
  split into pieces of at most 50 m before reprojection, so a long straight
  edge is not replaced by one chord. A WARNING is logged when a given CRS
  changes any polygon's area by more than 0.01 % relative to EPSG:6933; for
  example ESRI:54009 (Mollweide, a spherical projection) is not equal-area on
  WGS 84 coordinates.
- Distances (Hausdorff, boundary distances, tolerance metrics) are computed in
  the WGS 84 / UTM zone of each reference field (predicted boundaries in their
  own zone for boundary precision).
- Boundaries are sampled at points at most Δ apart
  (`boundary_sample_spacing_m`, default 1 m; `None` selects
  `max(1 m, perimeter / 1000)`, which is faster on large fields). The
  Hausdorff distance is then underestimated by at most Δ/2 and the mean
  distance is within Δ/4 of its exact value.
- Layers that cross the antimeridian are not supported.

Public constants: `MATCHING_METHODS`, `BOUNDARY_SAMPLE_SPACING_M` (1.0, the
default spacing), `BOUNDARY_MAX_SAMPLES` (1000, used only with spacing `None`)
and `BOOTSTRAP_CONFIDENCE_LEVEL` (0.95).

Run time: on 50,603 NMOSE reference fields against 35,225 predictions,
`evaluate()` took about 56 s with the defaults and 33 s with
`boundary_sample_spacing_m=None` (measured 2026-09-27 on the development
machine).

## Per-field table

`evaluate_frame(...)` returns one row per evaluated reference field, indexed
like the reference layer: `area_m2`, `area_ha`, `perimeter_m`, `utm_epsg`,
`matched`, `pred_index`, `iou`, `pred_area_m2`, `best_iou`,
`n_overlapping_pred`, `max_overlap_pred_index`, `oversegmentation`,
`undersegmentation`, `hausdorff_m`, `boundary_mean_distance_m`,
`boundary_sample_spacing_m`, plus `stratum`/`size_class` and the tolerance
columns when requested.

## Pixels per field

`pixels_per_field(reference, gsd_m)` returns `A / gsd_m²` per field, the
number of `gsd_m × gsd_m` pixels whose total area equals the field area. It is
a quick measure of how well a sensor can resolve the fields (the agent's
`estimate_resolvability` tool uses it).

## References

- Clinton, N., Holt, A., Scarborough, J., Yan, L., & Gong, P. (2010). Accuracy
  assessment measures for object-based image segmentation goodness.
  *Photogrammetric Engineering & Remote Sensing* 76(3), 289-299.
  <https://doi.org/10.14358/PERS.76.3.289>
- Persello, C., & Bruzzone, L. (2010). A novel protocol for accuracy
  assessment in classification of very high resolution images. *IEEE
  Transactions on Geoscience and Remote Sensing* 48(3), 1232-1244.
  <https://doi.org/10.1109/TGRS.2009.2029570>
- Stehman, S. V., & Foody, G. M. (2019). Key issues in rigorous accuracy
  assessment of land cover products. *Remote Sensing of Environment* 231,
  111199. <https://doi.org/10.1016/j.rse.2019.05.018>
