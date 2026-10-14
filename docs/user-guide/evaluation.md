# Evaluation

`agribound.evaluate.evaluate(predicted, reference, ...)` compares predicted
field polygons with reference field polygons at the object level and returns
a dictionary of metrics. `evaluate_frame` returns the per-reference-field table
the summary is computed from, `pixels_per_field` the number of pixels a
field spans at a given ground sample distance, `chance_baselines` the
scores of a blind grid and a shifted reference ([Chance
baselines](#chance-baselines)), and `paired_bootstrap` the differences
between two layers scored against one reference ([Comparing two layers:
paired bootstrap](#comparing-two-layers-paired-bootstrap)). `evaluate()` also
backs the `agribound evaluate` command and the pipeline's automatic
evaluation.

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
uses these functions against the DRI/NMOSE field boundaries (V18) in the San
Juan Basin, without the reference polygons smaller than 2,500 m², the
predictions' minimum area (350 of the 1,705 in its box): strata (the HUC-8
subbasin column `Sub_Basin`), size classes, bootstrap intervals, the 10 m
boundary tolerance, the per-field table and pixels per field. It also
compares Delineate-Anything v2 on Landsat, Sentinel-2, SPOT 6/7 and NAIP of
one year, so that only the resolution differs (`--resolution-year`, default
2022, the year closest to 2024 that all four cover; `--no-resolution` skips
it), and the pre-trained FTW and Delineate-Anything engines with and without
the LULC crop filter (`--no-lulc-comparison` skips it); `--predicted PATH`
evaluates an existing layer and skips both comparisons. In the development run
of agribound 1.1, Delineate-Anything v2 on Sentinel-2 2024 scored precision
0.64, recall 0.17 and F1 0.27 against the 1,355 fields (95 % intervals: F1
0.24-0.29, recall 0.15-0.19); recall rose with field size, from 0.01 at
0.2-0.5 ha to 0.78 at 50-100 ha, and F1 was 0.42, 0.31 and 0.16 in the Middle
San Juan, Upper San Juan and Animas sub-basins (median fields 4.6, 1.1 and
0.9 ha). On the 2022 composites F1 was 0.13 (Landsat), 0.29 (Sentinel-2),
0.17 (SPOT 6/7, a single scene with valid data over 76.7 % of the box) and
0.39 (NAIP). See the [gallery](../gallery.md).

Example 21
([`examples/21_published_ftw_audit.py`](https://github.com/montimaj/agribound/blob/main/examples/21_published_ftw_audit.py))
evaluates the published FTW polygons of 2024 near Belen, New Mexico, against
V18, both layers without polygons under 2,500 m² (227 of the 322 FTW
polygons and 645 of the 797 V18 fields in its box are evaluated), with size
classes, bootstrap intervals and the 10 m boundary tolerance. In the
development run of 1.1 they scored precision 0.63, recall 0.22 and F1 0.33
(95 % interval 0.28-0.37), with recall from 0.03 for fields of 0.2-0.5 ha to
0.92 at 10-20 ha (median field 0.88 ha). V18 records no imagery date, so the
scores measure agreement with the registry, not only FTW's error.

Example 24
([`examples/24_nevada_field_boundaries.py`](https://github.com/montimaj/agribound/blob/main/examples/24_nevada_field_boundaries.py))
evaluates every run in two Nevada basins against the DRI field boundaries
(Nevada, 2026-08-20), with three kinds of strata taken from the reference:
the irrigation system (`ITYPE_2024`: centre pivot, sprinkler, controlled
flood, uncontrolled flood or no apparent irrigation equipment, micro-irrigation
or retired), fixed size classes (under 2 ha, 2-10 ha, 10-50 ha, 50 ha or
more), and whether IrrMapper finds the field irrigated in the year of the
imagery (one strata column per year). A field with no apparent irrigation
equipment need not be irrigated, so the recall of that stratum is read with
the irrigated / not irrigated strata. Every run is also scored against the
fields irrigated in the year of its imagery alone (IrrMapper share of the
field at least 0.5: in Diamond Valley 39 of the 74 fields in 2022, 40 in 2023
and 34 in 2024; in Mason Valley 148 of the 187, 173 and 159), with the other
fields of the square as `ignore` ([Ignoring predictions on excluded
ground](#ignoring-predictions-on-excluded-ground)). As in example 20,
reference fields smaller than 2,500 m² are dropped, but the two evaluation
squares have none (they hold 74 and 187 DRI fields). Its fine-tuned models
learn from a separate square of each basin, at least 5 km from the evaluated
one, so their scores are held out. The inventory is not in the repository
(the script reads a local copy in place).

In the development run of 1.1, against all the fields of the square,
Delineate-Anything v2 as released on NAIP 2022 scored F1 0.26 in Diamond
Valley and 0.57 in Mason Valley, and DINOv3 fine-tuned on NAIP of the
training square 0.38 and 0.74. In Diamond Valley the 39 fields irrigated in
2022 are its centre pivots, which both runs find (recall 0.90 and 0.95),
while they find only 2 and 4 of the other 35 fields. Scored again on
2026-10-11 against the irrigated fields only, with the predictions on the
other fields ignored (71 of 216 and 50 of 142), the same outputs reach F1
0.38 and 0.56; precision stays low (0.24 and 0.40), because 78 and 51
predictions on ground the inventory does not map are still false positives.
In Mason Valley few predictions are ignored, and F1 changes by +0.01 to +0.03
in the panels the gallery shows (0.58 and 0.77 for these two runs). See the
[gallery](../gallery.md) (entries 24 to 24d).

Two opt-in flags add runs; they ran with agribound 1.1.0 on 2026-10-13,
reusing the 36 outputs above unchanged. `--sam-auto` adds SAM 2.1 automatic
masks ([`sam-auto`](engines.md#sam-21-automatic-masks-sam-auto)) on NAIP,
SPOT 6/7 and Landsat 8/9. These runs repeat the SAM evaluation's Nevada runs
(identical layers, and the scores that evaluation gave them after its
amendment of 2026-10-13, which re-scored Mason Valley against its full
reference), whose settings were chosen on this example's two training
squares: held out from that choice, not from the basins. In
Diamond Valley they scored F1 0.387 on NAIP and 0.589 on SPOT 6/7, against
0.255 and 0.126 for Delineate-Anything v2 as released; in Mason Valley 0.618
and 0.596, against 0.567 and 0.513. `--options` adds four "-opt" runs per
basin: runs above made again with the opt-in options of 1.1.0 that passed
the pre-registered test of the Diamond Valley options (PREREG-DV3, run on
three pivot squares in other Nevada basins; amended on 2026-10-13 for a
defect of its check step that changed no layer or score): the 3 m grid for
Delineate-Anything on NAIP, the half-band growth for DINOv3, the pivot snap
for every run and, for the optical runs, the crop filter, whose layer is
then the scored result. `options.csv` gives each run's change against its
base run with paired 1 km-block intervals (`paired_bootstrap`, 2,000
resamples, seed 42; the irrigated-only change with the other fields as
`ignore`). In Diamond Valley, Delineate-Anything v2 on NAIP went from F1
0.255 to 0.422 (+0.167, interval +0.090 to +0.240), DINOv3 on NAIP from
0.380 to 0.481, and DINOv3 on SPOT 6/7 from 0.500 to 0.557 (+0.057,
interval −0.022 to +0.128) with boundary F1 0.073 to 0.402; Sentinel-1
stayed at 0.346. In Mason Valley the four runs changed
F1 by +0.110, +0.033, −0.004 and 0. These runs combine options that were tested
one at a time, and on these two squares only the pivot snap, and in Mason
Valley the 3 m grid, was tested blind; the other effects there were known
or partly known before the test. They are reported results, not tests
(gallery entry 24e).

Example 25
([`examples/25_multistate_openet_field_boundaries.py`](https://github.com/montimaj/agribound/blob/main/examples/25_multistate_openet_field_boundaries.py))
evaluates Delineate-Anything v2 as released in one 5 km square in each of
the 47 states of the OpenET agricultural field boundaries v2.1.1, against
every field of the square: USDA FSA Common Land Units in most states, the
fields of state and federal agencies elsewhere and DRI's field delineation
in Oregon, with the DRI field boundaries in New Mexico (V18) and Nevada
(2026-08-20). Each square was chosen by a fixed rule from the reference and
the 2024 Cropland Data Layer alone (the cell with the most farmed fields of
at least 2,500 m²: fields whose representative point CDL maps as a crop,
hay or pasture class), the reference is selected by representative point
and filtered at the predictions' 2,500 m² minimum area, and the matching is
one-to-one at IoU ≥ 0.5. The sources define a field differently (a CLU is a
farm-programme unit; several state sources map irrigated land only; the
Nevada DRI file maps irrigated fields and fields retired from irrigation),
so the per-state scores also measure how far each source's idea of a field
matches what the imagery shows. The OpenET data are not public yet: the script
reads a local copy in place. In the development run of 1.1, F1 on NAIP had a
median of 0.31 over the 47 squares, from 0.01 (Montana) to 0.70
(California); pooled over the squares, precision was 0.30, recall 0.32 and
F1 0.31 (Sentinel-2 2024: pooled F1 0.29, median 0.33). Pooled NAIP F1 was
0.27 in the 35 squares whose reference is all CLU and 0.42 in the 9 with
none, and per-square F1 followed the elongation of the reference fields
(Spearman ρ = −0.49) more than their size (ρ = 0.09). See the
[gallery](../gallery.md).

Example 28
([`examples/28_new_york_field_boundaries.py`](https://github.com/montimaj/agribound/blob/main/examples/28_new_york_field_boundaries.py))
evaluates every run in one 5 km square in each of three New York landscapes (the onion,
vegetable and sod strips of the Black Dirt in Orange County, example 25's
New York square; the dairy rotations of the Finger Lakes in Cayuga County;
the corn, hay and pasture of the North Country in Jefferson County) against
the OpenET agricultural field boundaries, version 2.1.1 (Desert Research
Institute for the OpenET Consortium, 2026; local data, not public yet),
whose New York fields are all Common Land Units (CLU) of their source
agency, the USDA Farm Service Agency. Its fine-tuned models learn from the NAIP of a
square of the same county 5-15 km away (held out); in the Black Dirt also
from a square 1 km away, flown on the same day, which is reported as
near-sample and never as held out. Recall is reported per land use of the
imagery year, strip shape (Polsby-Popper score below 0.3) and field size,
beside merge and split rates and both chance baselines. Every run is
compared with Delineate-Anything v2 as released on the same source with
`paired_bootstrap`: a field and a 1 km-block interval of the F1 difference
(2,000 resamples each). The claim and gallery-panel rules were fixed before
the run (PREREG-NY, with amendments A1-A3, all before any run: A1 the
implementation of the paired intervals, A2 the opt-in Sentinel-2 `sam-auto`
runs, A3 the approval to show the outlines): a run is "better" (or "worse")
than the released model only if both intervals lie above (below) 0, and
"above chance" only if the lower bound of its F1 interval exceeds the higher
of the two baselines' F1.

In its run of 2026-10-12/13 (agribound 1.1.0, with `--sam-auto-sentinel2`;
one recorded deviation, in the driver's check after the run, changed no
number), Delineate-Anything v2 as released on NAIP 2022 scored F1 0.147 in the Black
Dirt, 0.339 in the Finger Lakes and 0.373 in the North Country; the Black
Dirt's two released runs reproduce example 25's New York scores, which were
known before. Fine-tuned on the NAIP of a square 15 and 10 km away, DINOv3 in
the Finger Lakes (0.433) and both models in the North Country (0.433 and
0.432) were better by the rule; Delineate-Anything fine-tuned in the Finger
Lakes (0.346) showed no clear difference. In the Black Dirt, fine-tuned on
the pasture and forest of a square 5 km away, Delineate-Anything scored
worse (0.085) and DINOv3 no differently (0.143); fine-tuned 1 km away on the
same muck (near-sample, not held out), they scored 0.305 and 0.280.
`sam-auto` on NAIP showed no clear difference in the three squares
(differences of +0.023 to +0.035), and FTW and `sam-auto` on Sentinel-2
scored worse than Delineate-Anything there in all three. The released NAIP
model matched 21 of
the 439 strip-like CLUs of the three squares and 322 of the 992 compact
ones. No Black Dirt run is above chance by the rule: 305 of its 668 CLUs are
strips, and the reference moved 24.3 m east and north still matches 470 of
them (F1 0.704), because parallel strips of one width repeat side by side
(an explanation found after the scores were seen).

A CLU is a farm-programme unit (FSA: the smallest unit of land with a
permanent, contiguous boundary, a common land cover and land management, a
common owner and a common producer). Several visible fields can be one CLU,
a strip field can be several, CLUs also cover lawns, house lots and old
fields, and some farmland has none: in example 28's evaluation squares,
83.5-89.3 % of the farmed CDL 2024 pixels lie in a CLU. Scores against CLUs
therefore measure agreement with that partition, not with the fields an
image shows, and one square per region is not an accuracy estimate for its
county; read them with the merge rates, the strip stratum and the chance
baselines. See the [gallery](../gallery.md) (entries 28, 28b and 28c).

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
| `proj_network` | PROJ's network setting when the metrics were computed (`pyproj.network.is_network_enabled()`, set by `PROJ_NETWORK=ON`); reprojections between datums, such as a NAD83 reference to WGS 84, use the downloadable grids only with it on. It records the setting, not the use of the grids: a CRS pair that the process reprojected before the setting was changed keeps its transformation, so set `PROJ_NETWORK=ON` before the process starts (see [Reproducibility](reproducibility.md#seeds)) |

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
  probability sample (Stehman & Foody, 2019). To compare two layers, or to
  resample spatial blocks, see [Comparing two
  layers](#comparing-two-layers-paired-bootstrap).

## Comparing two layers: paired bootstrap

`paired_bootstrap(predicted_a, predicted_b, reference, ...)` scores two
prediction layers against one reference with the rules of `evaluate()` and
reports the differences A − B of F1, precision and recall (and boundary F1
with `boundary_tolerance_m`), each with a 95 % percentile interval. Every
resample scores both layers on the same draw, so the variation that comes
from which fields are drawn, which the two layers share, cancels in the
difference. Use it for a change of F1 between two runs scored against the
same reference: another engine, another source, an option turned on.

```python
from agribound.evaluate import paired_bootstrap
from agribound.io import read_vector

out = paired_bootstrap(
    read_vector("fields_new.gpkg"),
    read_vector("fields_released.gpkg"),
    read_vector("reference.gpkg"),
    block_size_m=1000.0,  # resample 1 km blocks
    block_crs="EPSG:32611",  # the CRS the study square was laid out in
    boundary_tolerance_m=10.0,
)
lower, upper = out["ci"]["f1"]
print(
    f"F1 {out['a']['f1']:.3f} - {out['b']['f1']:.3f} = {out['difference']['f1']:+.3f} "
    f"(95 % interval {lower:+.3f} to {upper:+.3f}, {out['n_blocks']} blocks)"
)
```

It takes `evaluate()`'s `iou_threshold`, `matching`, `boundary_tolerance_m`,
`equal_area_crs`, `strata` and `ignore` (applied to both layers), and
`n_resamples` (default 2,000) and `seed` (default 42). The result holds
`difference`, its intervals `ci` and `n_undefined` (resamples in which a
rate of A or B has a zero denominator); under `a` and `b`, each layer's
values, which are those of `evaluate()`, its counts (`count_fp_unassigned`
among them) and its own intervals over the same resamples; with `strata`,
the same per stratum; and `resampling`, `n_blocks`, `block_crs` and
`proj_network`.

What is resampled:

- **Reference fields** (the default). These are the resamples of
  `evaluate(..., bootstrap=n_resamples, bootstrap_seed=seed)`, so each
  layer's own intervals (`out["a"]["ci"]`, `out["b"]["ci"]`) are those
  `evaluate()` reports. A field brings each layer's predictions whose home
  it is (the field it matches, else the one it overlaps most); predictions
  that overlap no reference field are fixed in every resample. With
  `strata`, each stratum is resampled at its own size.
- **Square blocks** (`block_size_m`). The plane of `block_crs` (default: the
  WGS 84 / UTM zone of the centre of the three layers) is cut into squares
  with edges at multiples of `block_size_m`. The squares that hold the
  representative point of a reference field or of a prediction of either
  layer are resampled, each with every reference field and prediction whose
  representative point lies in it (`shapely.point_on_surface` in the
  polygon's own CRS, as for `ignore`). Predictions on ground that no
  reference field covers are resampled with their block. A true positive
  counts with its reference field, and precision is TP / (TP + FP), so a
  matched prediction acts on precision, recall and F1 only through its
  field. Strata only group the reported metrics here: a block holds fields
  of several strata.

Which to report. Resampling fields one by one treats them as independent
and holds the predictions that overlap no reference field fixed. Where many
predictions lie on ground the reference does not map (78 of
Delineate-Anything's 216 NAIP polygons in example 24's Diamond Valley square,
above), the intervals are then too narrow, and neighbouring fields often
share errors (one image, one farm layout). Blocks resample those predictions
and keep neighbours together, so report the block interval, with the field
interval beside it if useful, and fix the block size before looking at any
result. A 5 km square holds at most 25
blocks of 1 km, so the intervals are rough; blocks smaller than the distance
over which errors are correlated still give intervals that are too narrow.
The intervals describe the sampling of fields or blocks for the two outputs
as they are: they do not include the variation of a model between runs
(another seed, another device), and neither scheme is a design-based
estimate for a probability sample of fields (Stehman & Foody, 2019).

An end of an interval near 0 also depends on the random draws. In the
pre-registered SAM evaluation (see
[Engines](engines.md#pre-registered-evaluation)), Mason Valley's SPOT pair
had the 1 km-block interval −0.003 to +0.161 at the default seed 42; seeds
0-19 gave lower bounds of −0.003 to +0.004, above 0 for 11 of them. Fix the
seed with the rule before any result is seen, and read an end within a few
thousandths of 0 as being at the edge. Examples 24 and 28 and the SAM
evaluation of 1.1.0 report the block interval (2,000 resamples, seed 42,
1 km blocks in the UTM zone the squares were laid out in: EPSG:32611 in
Nevada, EPSG:32618 in New York).

The block assignment was checked against an independent scorer written for
the agribound 1.1 review (2026-10-12): on 11 pairs of layers in three 5 km
squares of the 1.1 development work, in Nevada and New Mexico, the counts
of reference fields, true positives and predictions in each 1 km block were
the same, and that scorer's bootstrap on agribound's counts gave its own
intervals. The New Mexico reference is in NAD83 and the predictions in
WGS 84; with PROJ's network grids on (`proj_network`), two of the layers had
one match fewer than with them off. On that square (461 reference fields,
426 and 518 predictions, 2,000 resamples, 10 m tolerance) a call took about
0.5 s on the development machine, about the time of two `evaluate()` calls.

## Ignoring predictions on excluded ground

`ignore` leaves out of the score the predictions that lie on ground excluded
from it, typically reference fields left out of the reference passed. A
prediction whose representative point lies in an `ignore` polygon or on its
boundary is removed before matching: it is neither a true nor a false
positive. The point is `shapely.point_on_surface` of the (repaired)
prediction in its own CRS (what `GeoSeries.representative_point()` gives for
the repaired polygon), tested in the CRS of `ignore`. The metrics gain
`count_predicted_ignored`, and the other keys are computed without the
removed predictions (`count_predicted` counts the scored ones), except
`count_predicted_dropped` and `count_predicted_repaired`, which describe the
layer passed, as without `ignore` (a repaired prediction is counted whether
it was scored or removed).

```python
from agribound.evaluate import evaluate
from agribound.io import read_vector

pred = read_vector("fields.gpkg")
reference = read_vector("reference.gpkg")
irrigated = reference["irrmapper_2022"] >= 0.5  # fields irrigated in the imagery year
metrics = evaluate(pred, reference[irrigated], ignore=reference[~irrigated])
print(metrics["count_predicted_ignored"], metrics["precision"], metrics["recall"])
```

What it does and does not do:

- The excluded fields are left out by passing a reference without them;
  `ignore` removes the predictions on them. It does not restrict the score
  to the reference fields: predictions on ground in neither layer (field
  corners, roads, unmapped land) are still false positives.
- A prediction counts by its representative point alone. One that covers a
  scored field and an ignored one is removed when the point lies in the
  ignored field, even if it would match the scored field, and scored
  otherwise. The `ignore` polygons should therefore not overlap the
  reference.
- It is not a stratum. A stratum's scores (`per_stratum`) come from one
  matching of all reference fields and all predictions; with `ignore`, the
  excluded fields and the predictions on them take no part in the matching,
  so the recall of the two can differ. In example 24's 28 runs they were
  equal.

Example 24 scores every run this way against the fields irrigated in the
year of its imagery, with the other fields of the square as `ignore` (see
above and the gallery entries 24c and 24d). In Diamond Valley, 4 to 158
predictions per run were ignored and F1 rose by 0.011 to 0.238
(Delineate-Anything on NAIP from 0.26 to 0.38); in Mason Valley 2 to 49 were
ignored and F1 changed by −0.001 to +0.053.

`evaluate_frame` and `chance_baselines` take the same argument, and
`evaluate(..., baselines=...)` scores the baseline layers with the call's
`ignore` (their cells and shifted polygons whose representative point lies
in an `ignore` polygon are left out). Without `ignore`, `evaluate()` returns
what it returned before.

## Chance baselines

`chance_baselines(predicted, reference, ...)` scores two layers that use no
image, with the rules of `evaluate()` (the same code, matching, IoU threshold,
areas and boundary tolerance; no bootstrap, strata or size classes):

- **A blind grid.** Square cells whose side is the square root of the median
  area of the evaluated predictions (`cell_side_m` sets it), laid out in the
  WGS 84 / UTM zone of the centre of the study area (`study_area`) or,
  without one, of the predictions' bounding box, anchored at the lower-left
  corner of its bounding box and clipped to it. The grid is shifted by k/6 of a cell along each
  axis (k = 0-5; `offset_steps=6`), and the metrics of the 36 grids are
  averaged; `f1_min`, `f1_max` and the metrics of each grid (`grids`) are
  reported too. With `footprint=True` the cells are also cut to the union of
  the predictions, so the grid covers only the ground they cover.
- **A shifted reference** (`shift_m`, optional): each reference polygon
  moved `shift_m` metres east and north and clipped to the same extent.
  `shift_m="auto"` moves it by half the median width of the reference
  polygons (the shorter side of each one's minimum rotated rectangle).

Clipping and cutting leave pieces of cells and of shifted polygons; a piece
is dropped when its area is below `min_area_m2` (default: the area of the
smallest prediction). Whole cells and whole shifted polygons are kept.

```python
from agribound.evaluate import chance_baselines, evaluate
from agribound.io import read_vector

pred = read_vector("fields.gpkg")
ref = read_vector("reference.gpkg")
aoi = read_vector("study_area.geojson")
metrics = evaluate(pred, ref, boundary_tolerance_m=20.0)
blind = chance_baselines(pred, ref, boundary_tolerance_m=20.0, study_area=aoi, shift_m="auto")
print(metrics["f1"], blind["blind_grid"]["f1"], blind["shifted_reference"]["f1"])
print(metrics["boundary_f1"], blind["blind_grid"]["boundary_f1"])
```

`evaluate(..., baselines=True)` adds the same dictionary as
`metrics["baselines"]`, scored with the call's `iou_threshold`, `matching`,
`boundary_tolerance_m`, `equal_area_crs`, `boundary_sample_spacing_m` and
`ignore`;
`baselines={"study_area": aoi, "shift_m": "auto"}` passes the other options,
which are checked before the evaluation runs (the study area's CRS and
polygons too). Some layers give no baselines: without an evaluated prediction
the grid needs `cell_side_m` and `study_area`, and a shifted reference needs
reference polygons. `chance_baselines()` then raises `ValueError`, while
`evaluate()` logs a WARNING, returns its metrics and sets
`metrics["baselines"]` to `{"skipped": <reason>}`, so a loop over sites keeps
the metrics of a site where the engine found no field.
Without `baselines`, `evaluate()` returns what it returned before.
`agribound evaluate --baselines` passes `baselines=True` (the blind grid over
the predictions' bounding box; no shifted reference); the pipeline's
evaluation computes no baselines.

**Why.** In two agribound pilots on small-field sites, boundary F at a loose
tolerance gave layers that use no image about the scores of real outputs:

- **Crop-sequence pilot** (2026-10-09; ten of example 25's 5 km US squares,
  with median reference fields of 0.9-5.2 ha). A blind grid with cells of
  the median area of the crop-history zones under test (116-178 m sides, one
  offset, clipped to the square) had a boundary F at 60 m of 0.65-0.90: at
  least that of the published USDA NASS Crop Sequence Boundaries on 6 of the
  10 squares, and at least that of Delineate-Anything on Sentinel-2 (2024)
  on 4. Its object F1 at IoU ≥ 0.5 was 0.001-0.048, against
  Delineate-Anything's 0.011-0.649.
- **Composites pilot** (2026-10-09; oil palm smallholder parcels at Oro,
  Papua New Guinea, median 1.53 ha, and Chenderiang, Malaysia, median
  1.20 ha). The boundary F at 20 m of each output (Delineate-Anything, which
  drew no line at Oro, and SNIC and SLIC segmentations) was within −0.019 to
  +0.030 of that of an image-blind square mesh with the same line length
  (mean over 36 offsets). 77-81 % of the scored area of the two squares lies
  within 20 m of a reference line.

A point placed at random among square fields of side *a* lies within *t* of
a field edge with probability 1 − (1 − min(1, 2*t*/*a*))², so this is the
boundary precision expected of lines drawn without regard to the fields.
With *a* = 4 × reference area / reference perimeter (both summed over the
fields; exact for squares, an approximation otherwise), it is 0.967 or more
at 60 m on 7 of the 10 US squares, and 0.41-1.0 at 30 m. So report object
metrics with boundary F, and read both against the baselines. The
superpixel test for agribound 1.1 used the same blind grid (36 offsets) and
the shifted reference in its pass rule (see [Superpixel segmentation (SNIC,
SLIC)](satellite-sources.md#superpixel-segmentation-snic-slic)).
`chance_baselines()` with the pilots' cell sides gives the counts and scores
of the crop-sequence pilot's blind grid on all ten squares and of that
test's 900 grids and 5 shifted references.

Example 27 scores both baselines with `evaluate(..., baselines=...)` beside
the crop-history zones of the `crop-sequence` engine and Delineate-Anything
on Sentinel-2 in two Quebec squares (agribound 1.1.0, 2026-10-11; declared
parcels of 2.4 ha median): the blind grids had an F1 of 0.007-0.018 and a
boundary F at 60 m of 0.85-0.89, and the reference shifted by half its median
width (54 m in Bas-Saint-Laurent, 40 m in Montérégie) 0.92 and 0.99, against
0.88-0.93 for the two outputs, whose F1 were 0.119-0.197 (zones) and
0.472-0.486 (Delineate-Anything). See
[Crop-sequence zones](engines.md#measured-limits).

Run time: one evaluation per grid (36 by default) and one for the shifted
reference. In a synthetic 5 km square with 2,156 reference fields of about
1 ha and a 20 m tolerance, `evaluate()` took about 1 s and
`chance_baselines(..., shift_m="auto")` 13 s (24 s with `footprint=True`) on
the development machine (2026-10-10). A grid may have at most
`BLIND_GRID_MAX_CELLS` (1,000,000) cells before clipping; a larger one
raises `ValueError`.

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
default spacing), `BOUNDARY_MAX_SAMPLES` (1000, used only with spacing `None`),
`BOOTSTRAP_CONFIDENCE_LEVEL` (0.95), and for the chance baselines
`BLIND_GRID_OFFSET_STEPS` (6) and `BLIND_GRID_MAX_CELLS` (1,000,000).

Run time: on 50,603 NMOSE WUCB reference fields against 35,225 predictions,
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
