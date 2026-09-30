# SAM Refinement

With `sam_refine=True` (CLI `--sam-refine`) the pipeline refines the
engine's polygons with a Segment Anything model: every polygon's bounding box
is given to SAM as a single-object box prompt, and the polygon is replaced by
the mask SAM returns, unless the mask covers too little of it (see
[Masks that cover too little of the polygon](#masks-that-cover-too-little-of-the-polygon)).
It is a post-processing stage
(`agribound.engines.samgeo_engine.refine_boundaries`), not an engine. It runs
after delineation and before the study-area selection and post-processing, for
every engine except `embedding`, which refines its own polygons (see
[Engines](engines.md#embedding-clustering-embedding)).
`engine_params["sam_refine"]` is still honoured as a legacy switch.

```python
gdf = agribound.delineate(
    study_area="area.geojson",
    source="naip",
    year=2022,
    engine="delineate-anything",
    gee_project="my-project",
    sam_refine=True,
    sam_backend="sam2",
)
```

## Which polygons are refined

A polygon is refined only if

1. its bounding box lies inside the raster (within half a pixel; polygons
   outside are counted in `n_skipped_outside`), **and**
2. both sides of its padded bounding box are at least `sam_min_crop_px`
   pixels (default 64). The padded side is
   `floor(side_px × (1 + 2 × sam_crop_padding))` with `sam_crop_padding`
   (default 0.15), so with the defaults a field's unpadded bounding box must
   be at least 64 / 1.3 ≈ 49.2 pixels on both sides.

Skipped polygons keep their geometry, and so do prompted polygons whose mask
failed or [covered too little of them](#masks-that-cover-too-little-of-the-polygon).
The boolean column
`agribound:sam_refined` marks refined polygons, the column
`agribound:sam_score` holds SAM's predicted IoU of each refined mask (NaN for
the other polygons), and `gdf.attrs["sam_stats"]`
(also in the provenance record) holds the backend, model, device, the
counts `n_total`, `n_refined`, `n_skipped_small`, `n_skipped_outside`,
`n_failed` and `n_low_coverage` (the last five add up to `n_total`), the
overlap mode with `n_overlap_trimmed` and `overlap_trimmed_fraction` (see
[Overlapping masks](#overlapping-masks)), and the coverage threshold
`min_coverage`.
`crop_window_px` and `is_refinable` reproduce the gating test
exactly (the agent's `estimate_resolvability` tool uses them).

The size gate matters at coarse resolution: a 49 px side is 490 m at 10 m
and about 1.5 km at 30 m. In a Kenya Sentinel-2 smoke run of the pipeline,
631 of 634 fields were skipped as too small.

## Polygons that cover several fields

SAM returns one object for each box prompt. When one input polygon covers
several fields, the mask usually follows one of them. In agribound 1.0.0 the
mask replaced the whole input polygon whatever share of it the mask covered,
so the rest of the input's area was left with no polygon. Embedding clusters
produce such polygons (neighbouring fields of the same land-cover class form
one connected region), and so can other engines where field edges are faint.
Since 1.0.1, by default, a polygon keeps its input geometry when its mask
covers less than half of it (see
[Masks that cover too little of the polygon](#masks-that-cover-too-little-of-the-polygon)).
This limits the area left uncovered, but it does not separate the fields:
they stay in the one input polygon.

In example 15 (Pampas, Argentina) as run with agribound 1.0.0 on 2026-09-29,
29 centre pivots inside the study area were located in the Sentinel-2
composite and checked by eye. Before SAM, 10 of them (TESSERA clusters) were
part of a cluster polygon more than twice their area. After SAM 2 on
Sentinel-2, 8 of the 29 were at most 7 % covered by any polygon, all of them
among those 10, and SAM removed 16 % of the area of the TESSERA crop-filter
polygons (24 % for the Google Satellite Embedding polygons, where 14 of the
29 pivots were left less than half covered, 10 of them because they had been
part of a larger polygon).

The example therefore also writes a split variant, which is the one the
[gallery](../gallery.md) shows: parts over 50 ha are kept unrefined
(multi-part polygons are split into parts first; the 1.0.0 and 1.0.1 crop
layers had none), and because `"trim"` sees only the polygons given to
`refine_boundaries`, the example trims the refined masks where they overlap
the kept parts. In the 1.0.0 run, this variant left no TESSERA pivot and 1
Google pivot (which the clustering had broken into smaller pieces; 49 %
covered) less than half covered, and the refined layers covered about the
crop-filter area (0.6 % and 1.7 % less); 10 and 13 of the pivots stayed
inside unrefined polygons more than twice their area. The rule is a
heuristic of the example, not a pipeline option: it also leaves large single
fields unrefined, and fields close to 50 ha (such as these pivots) fall on
either side of it. In 1.0.1 the example still writes both variants, with the
default `sam_min_coverage` in both; the coverage test runs inside
`refine_boundaries`, before the example's own trim.

In the 1.0.1 run of example 15 on 2026-09-29 (default `sam_min_coverage` of
0.5), 6 of the 29 pivots were part of a TESSERA cluster polygon more than
twice their area before SAM (10 with 1.0.0; the 1.0.1 k-means solution puts
18.2 % of the crop-filter area in polygons over 200 ha, against 38.4 %, see
[Engines](engines.md#embedding-clustering-embedding)). SAM 2 on
Sentinel-2 over every polygon refined 546 of the 2,170 TESSERA crop-filter
polygons; 37 masks covered less than half of their polygon and were not used,
and 1 failed. It left 2 of the 29 pivots less than half covered (2.4 % and
45.7 %; the first had been part of a polygon 2.7 times its area) and removed
6.6 % of the crop-filter area (48,186.2 to 45,013.6 ha). The 568.6 ha TESSERA
polygon that holds 4 pivots keeps its input geometry, so these stay 94-98 %
covered; with 1.0.0, SAM had left them at most 7.4 % covered. For the Google
Satellite Embedding polygons, SAM 2 refined 367 of 1,986 (70 masks covered
too little, 1 failed), left 4 of the 29 pivots less than half covered and
removed 4.9 % of the area (54,206.0 to 51,540.4 ha). Three of those 4 pivots
had been part of a polygon more than twice their area. The fourth had almost
no polygon already before SAM, because the crop filter had removed the
11,105 ha cluster polygon that held it.

In the 1.0.1 run of 2026-09-29, the split variant kept 283 TESSERA and 205
Google polygons over 50 ha unrefined. Of the other 1,887 and 1,781, SAM 2
refined 289 and 208; 13 and 28 masks covered too little, and none failed. It
left no TESSERA pivot and 1 Google pivot less than half covered: the Google
one (0.2 % covered) is the pivot that already had almost no polygon after the
crop filter. The refined layers covered 0.1 % (TESSERA) and 0.8 % (Google)
less than the crop-filter polygons, and 6 and 9 of the pivots stayed inside
unrefined polygons more than twice their area. On the example's other SAM
input, three TESSERA dimensions, SAM 2 refined 550 of the 2,170 TESSERA
polygons when given every polygon (33 masks covered too little, 1 failed) and
293 of the 1,887 parts of 50 ha or less in the split variant (9 covered too
little, none failed).

## How refinement works

- **Image.** The canonical R, G, B bands of the source (`bands` or
  `engine_params["sam_rgb_bands"]` override them) are stretched to uint8 with
  one scene-wide 1-99 percentile stretch; uint8 rasters are used as they are.
- **Windows.** Polygons that fit are assigned to a grid of
  `engine_params["sam_window_px"]` × `sam_window_px` windows (default 1024,
  stride half a window); each window is encoded once and its boxes are
  decoded in batches of `engine_params["sam_batch_size"]` (default 32).
  Larger fields get their own square window; windows longer than
  `max(2 × sam_window_px, 2048)` px are read decimated. Non-square windows are
  padded to a square with black pixels.
- **Masks.** One mask per box; only the part inside the field's padded box is
  kept, vectorised, and the largest polygon is used.
- **Overlaps.** With the default `engine_params["sam_overlaps"]="trim"`, a
  refined mask may not take area from another polygon; see
  [Overlapping masks](#overlapping-masks).
- **Coverage.** A mask that covers less than
  `engine_params["sam_min_coverage"]` (default 0.5) of its input polygon is
  not used; see
  [Masks that cover too little of the polygon](#masks-that-cover-too-little-of-the-polygon).

!!! note "Changed in 1.0.0"
    agribound 0.1.x encoded every field's padded crop on its own, so SAM
    upsampled a 64 px crop about 16-fold. 1.0.0 encodes fields at about native
    scale in shared windows. A smaller `sam_window_px` (at least
    `2 × sam_min_crop_px`) restores part of that zoom at the cost of more
    encoder passes. `sam_batch_size` is now a real decoder batch size; in 0.1.x
    it was only a logging interval.

Masks depend on the compute device: on a Sentinel-2 test crop, SAM 2
(`sam2-hiera-tiny`) masks computed on Apple MPS overlapped the CPU masks of
the same fields with IoU between 0.59 and 0.97. CPU results were
deterministic.

## Overlapping masks

A mask can grow over a neighbouring polygon. `engine_params["sam_overlaps"]`
decides what happens then:

- `"trim"` (default): a refined polygon never takes area that another input
  polygon covered and its own input polygon did not. Where two refined masks
  grew over the same new area, the mask with the higher SAM score keeps it.
  A trimmed mask keeps its largest part, which then goes through the
  [coverage test](#masks-that-cover-too-little-of-the-polygon). A mask with
  nothing left keeps the input geometry and counts in `n_failed`. So the
  refined output never overlaps more than the input polygons did, and SAM
  adds no overlap to an engine output that had none.
- `"keep"`: the masks are kept as SAM returned them (with
  `sam_min_coverage=0`, as in agribound 0.1.x), so refined polygons can
  overlap their neighbours.

Any other value raises `ValueError` when the refinement starts.
`sam_stats["overlaps"]` records the mode. `n_overlap_trimmed` counts the
trimmed masks, and `overlap_trimmed_fraction` is the share of the refined
mask area that was removed (masks rejected by the coverage test are left out
of both). The same parameter applies to the `embedding`
engine's own refinement and to `refine_boundaries` called directly (pass it
in `config.engine_params`).

!!! warning "Trade-off, measured once on a small area"
    Delineate-Anything (`large_v2`) + SAM 2 (`sam2-hiera-large`, Apple MPS),
    Sentinel-2 2023, Namoi test area (4 reference fields), 16 of 230 polygons
    refined, measured on 2026-09-28 with agribound 1.0.0 (before the coverage
    test):

    | | no SAM | `"keep"` | `"trim"` (default) |
    |---|---|---|---|
    | Overlap between polygons of the post-processed output | 0.18 ha | 3.59 ha (3.45 ha between a refined polygon and a neighbour) | 0.20 ha |
    | Best IoU of the reference field SAM changed most | 0.636 | 0.773 | 0.676 |
    | Mean IoU of the matched fields (2 of 4 at IoU ≥ 0.5) | 0.757 | 0.826 | 0.777 |

    With `"keep"`, one refined mask grew over the neighbours of that field and
    matched it better. The other three reference fields were unchanged or
    changed by at most 0.02. When a mask grows past an engine boundary, it
    may be correcting a field the engine split in two, or it may be leaking
    into a real neighbour. `"trim"` keeps the engine's boundaries between
    polygons. Use `engine_params={"sam_overlaps": "keep"}` if SAM should be
    allowed to override them. One small area does not show which setting is
    more accurate in general. Check both on your own reference data.

## Masks that cover too little of the polygon

`engine_params["sam_min_coverage"]` (default 0.5, added in 1.0.1) is the
smallest share of its input polygon that a mask must cover to replace it.
Coverage is `area(mask ∩ input) / area(input)`, measured on the mask after
the overlap trim; an invalid input polygon is repaired first. A polygon whose
mask covers less keeps its input geometry: `agribound:sam_refined` is False,
`agribound:sam_score` is NaN, and it counts in `n_low_coverage`. Input
polygons without area are never rejected.

- With `"trim"`, each mask is tested right after its trim, in the trim's
  score order. A rejected mask takes no area, so the masks with lower scores
  are trimmed as if its polygon had not been refined. A mask with nothing
  left after the trim counts in `n_failed`, not in `n_low_coverage`.
- With `"keep"`, each mask is tested as SAM returned it.

The value must be a number from 0 to 1 (both included); 0 turns the test off
(the 1.0.0 behaviour). Any other value, including `None`, `True` and NaN,
raises `ValueError` when the refinement starts. `sam_stats["min_coverage"]`
records the value used. Set it with
`engine_params={"sam_min_coverage": 0.7}` (CLI
`--engine-param sam_min_coverage=0.7`). As with `sam_overlaps`, it applies to
the `embedding` engine's own refinement and to `refine_boundaries` called
directly, and the `ensemble` engine accepts it in its `engine_params`.

!!! warning "Trade-off, measured on 2026-09-29"
    SAM 2 (`sam2-hiera-large`, Apple MPS). SAM ran once per input; its masks
    were then tested at each threshold, and the results area-filtered,
    smoothed and simplified as in the examples. The example 15 inputs are its
    1.0.0 crop polygons, refined on the Sentinel-2 composite. The example 14
    inputs are its 1.0.0 DINOv3 outputs without SAM, refined afterwards as
    example 13 does; example 14's own SAM runs refine before post-processing
    and are not shown. Example 13's input is example 20's Delineate-Anything
    output, as in the [gallery](../gallery.md).

    | Input | Measure | no SAM | 0 (1.0.0) | 0.5 (default) | 0.7 |
    |---|---|---|---|---|---|
    | Example 15, TESSERA / Google crop polygons (510 / 439 masks) | `n_low_coverage` | - | 0 / 0 | 48 / 95 | 93 / 167 |
    | | pivots less than half covered (of 29) | 0 / 0 | 8 / 14 | 1 / 2 | 0 / 0 |
    | | area lost against the input polygons | - | 15.7 % / 24.1 % | 7.4 % / 8.1 % | 2.9 % / 0.9 % |
    | Example 13, Delineate-Anything, Sentinel-2 2019 (67 masks) | `n_low_coverage` | - | 0 | 0 | 0 |
    | Example 14, DINOv3 NAIP / SPOT 2022, Lea County | F1 (in-sample, 227 reference fields) | 0.604 / 0.423 | 0.609 / 0.479 | 0.590 / 0.445 | 0.595 / 0.418 |
    | | reference fields less than half covered | 35 / 67 | 80 / 105 | 53 / 85 | 40 / 77 |

    Every example 13 mask covered at least 92 % of its polygon, so no
    threshold up to 0.9 changed that output. The polygons of examples 15 and
    14 often hold several fields. There the default mostly returns the input
    polygon where SAM would have left fields uncovered: 9 (TESSERA) and 12
    (Google) of the 29 pivots were still part of a polygon more than twice
    their area at 0.5, against 10 and 13 before SAM. In Lea County a rejected
    mask often matched one of its polygon's fields well, so the test also
    lowers F1 against 0; at 0.5 the NAIP F1 is below the value without SAM.

On inputs whose polygons often hold several fields, such as embedding
clusters, 0.7 left fewer fields uncovered than the default in these runs,
but its Lea County SPOT F1 is below the value without SAM. Check the effect
on your own reference data. Because the default changes the results, 1.0.1
does not reuse a 1.0.0 output of a run with `sam_refine`: it raises
`FileExistsError` until the output is recomputed with `overwrite=True` (see
[Output reuse](reproducibility.md#output-reuse)).

## Backends

`sam_backend` selects the implementation; `sam_model` overrides the default
model (legacy: `engine_params["sam_model"]`).

| `sam_backend` | Implementation | Default model | Accepted models |
|---|---|---|---|
| `sam2` (default) | `samgeo.SamGeo2(...).predictor` (SAM 2.0) | `facebook/sam2-hiera-large` | `facebook/sam2-hiera-{tiny,small,base-plus,large}` (or `tiny`, `small`, `base_plus`, `large`) |
| `sam2.1` | `sam2.SAM2ImagePredictor.from_pretrained` | `facebook/sam2.1-hiera-large` | `facebook/sam2.1-hiera-{tiny,small,base-plus,large}` (or the size aliases) |
| `sam3` ([untested](#sam-3-is-untested)) | `samgeo.SamGeo3(backend="meta", enable_inst_interactivity=True)`, instance box prompts | `facebook/sam3` | `facebook/sam3`, `facebook/sam3.1` |
| `sam3-hf` ([untested](#sam-3-is-untested)) | `transformers.Sam3TrackerModel` / `Sam3TrackerProcessor` | `facebook/sam3` | `facebook/sam3` |

Only single-object box prompts are used; SAM 3's concept-exemplar prompts
(which segment every object similar to the box) are never used.

### SAM 3 is untested

!!! warning "The SAM 3 backends are currently untested"
    Neither `sam3` nor `sam3-hf` has been run end to end with agribound 1.0.1:
    the `facebook/sam3` weights are gated, and no approved Hugging Face token
    was available when 1.0.0 and 1.0.1 were prepared. The tests cover their imports,
    platform checks and argument handling only. agribound logs a WARNING (also
    recorded in the provenance) whenever a SAM 3 backend is loaded. Check the
    refined polygons before relying on them, or use `sam2` (the default), which
    was run in the 1.0.0 and 1.0.1 examples.

### SAM 3 platform support

| | Linux + CUDA | Windows + CUDA | macOS |
|---|---|---|---|
| `sam3` (Meta `sam3` 0.1.4 via samgeo) | supported by Meta | only through the community [`triton-windows`](https://github.com/triton-lang/triton-windows) wheel; accepted by agribound but **not verified**, logs a WARNING | not supported (no CUDA) |
| `sam3-hf` (transformers 5.x) | CUDA recommended | CUDA recommended | imports (checked on macOS); not run end to end |

- `sam3` needs a CUDA GPU and `triton`, which the Meta package imports at import
  time (`sam3.model_builder` → `sam3_tracking_predictor` →
  `sam3_tracker_utils` → `edt.py`), even for single images. agribound checks
  both before loading the model and raises an actionable error otherwise.
  `pip install "agribound[sam3]"` installs segment-geospatial's SAM 3 extra
  and, on Windows, `triton-windows` (agribound declares it with
  `sys_platform == 'win32'`, because segment-geospatial 1.4.2 declares it
  under `sys_platform == "windows"`, a marker that never matches).
- `sam3-hf` has no triton dependency, so it is the option for macOS and for
  Windows without `triton-windows`. It needs `transformers>=5`. It has not yet
  been run end to end by agribound on any platform, because the weights are
  gated.
- Both need approved access to the gated `facebook/sam3` repository on
  Hugging Face (request access there, then `hf auth login` or `HF_TOKEN`);
  `facebook/sam3.1` is available only through the `sam3` backend.
- Offline nodes: set `SAM3_CHECKPOINT_PATH` to a downloaded checkpoint (Meta
  backend, read by samgeo), or pre-populate the Hugging Face cache with
  `agribound prefetch --engine <engine> --sam-refine --sam-backend <backend>` and set
  `HF_HUB_OFFLINE=1`.

## Refining an existing layer

`refine_boundaries(gdf, raster_path, config)` can be called on any polygons
and any raster (for example DINOv3 polygons on a NAIP composite, or
embedding-cluster polygons on a Sentinel-2 composite); `config.source` must
describe the raster so the right R, G, B bands are read. Polygons are
reprojected to the raster CRS; the raster must not be rotated or sheared.

## References

- Ravi, N., et al. (2025). SAM 2: Segment anything in images and videos.
  ICLR 2025. arXiv:2408.00714.
- Carion, N., et al. (2026). SAM 3: Segment anything with concepts. ICLR 2026.
  arXiv:2511.16719.
- Wu, Q., & Osco, L. P. (2023). samgeo. *JOSS* 8(89), 5663.
  <https://doi.org/10.21105/joss.05663>

The SAM 3 licence (clause 1.b.ii) asks publications to acknowledge the use of
SAM materials.
