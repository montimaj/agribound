# SAM Refinement

With `sam_refine=True` (CLI `--sam-refine`) the pipeline refines the
engine's polygons with a Segment Anything model: every polygon's bounding box
is given to SAM as a single-object box prompt, and the polygon is replaced by
the mask SAM returns. It is a post-processing stage
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

Skipped polygons keep their geometry. The boolean column
`agribound:sam_refined` marks refined polygons, the column
`agribound:sam_score` holds SAM's predicted IoU of each refined mask (NaN for
the other polygons), and `gdf.attrs["sam_stats"]`
(also in the provenance record) holds the backend, model, device, the
counts `n_total`, `n_refined`, `n_skipped_small`, `n_skipped_outside` and
`n_failed`, and the overlap mode with `n_overlap_trimmed` and
`overlap_trimmed_fraction` (see [Overlapping masks](#overlapping-masks)).
`crop_window_px` and `is_refinable` reproduce the gating test
exactly (the agent's `estimate_resolvability` tool uses them).

The size gate matters at coarse resolution: a 49 px side is 490 m at 10 m
and about 1.5 km at 30 m. In a Kenya Sentinel-2 smoke run of the pipeline,
631 of 634 fields were skipped as too small.

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
  A trimmed mask keeps its largest part. A mask with nothing left keeps the
  input geometry and counts in `n_failed`. So the refined output never
  overlaps more than the input polygons did, and SAM adds no overlap to an
  engine output that had none.
- `"keep"`: the masks are kept as SAM returned them, as in agribound 0.1.x,
  so refined polygons can overlap their neighbours.

Any other value raises `ValueError` when the refinement starts.
`sam_stats["overlaps"]` records the mode. `n_overlap_trimmed` counts the
trimmed masks, and `overlap_trimmed_fraction` is the share of the refined
mask area that was removed. The same parameter applies to the `embedding`
engine's own refinement and to `refine_boundaries` called directly (pass it
in `config.engine_params`).

!!! warning "Trade-off, measured once on a small area"
    Delineate-Anything (`large_v2`) + SAM 2 (`sam2-hiera-large`, Apple MPS),
    Sentinel-2 2023, Namoi test area (4 reference fields), 16 of 230 polygons
    refined, measured on 2026-09-28:

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
    Neither `sam3` nor `sam3-hf` has been run end to end with agribound 1.0.0:
    the `facebook/sam3` weights are gated, and no approved Hugging Face token
    was available when 1.0.0 was prepared. The tests cover their imports,
    platform checks and argument handling only. agribound logs a WARNING (also
    recorded in the provenance) whenever a SAM 3 backend is loaded. Check the
    refined polygons before relying on them, or use `sam2` (the default), which
    was run in the 1.0.0 examples.

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
