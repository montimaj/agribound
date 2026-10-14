# Delineation Engines

An engine turns the stage-A raster into field polygons. Nine engines are
registered in `agribound.registry.ENGINE_REGISTRY` (`agribound list-engines`,
`agribound.list_engines()`). The configuration is validated against the
registry: an engine that does not support the source, or `fine_tune=True` for
an engine that cannot be fine-tuned, raises `ValueError` before anything is
downloaded.

## Overview

"Label-free" means the engine runs without a checkpoint trained by the user.
"GPU recommended" is a speed recommendation; every engine also runs on CPU.

| Engine | Key | Approach | Label-free | Fine-tunable | GPU recommended | Sources | Extra |
|---|---|---|---|---|---|---|---|
| Delineate-Anything | `delineate-anything` | YOLO11-seg instance segmentation (Ultralytics) | yes (published weights) | yes | yes | all imagery sources and `local` | `delineate-anything` |
| Fields of The World | `ftw` | semantic segmentation (field / boundary / background) with ftw-tools checkpoints, polygonised | yes (published weights) | **no** | yes | `sentinel2`, `landsat`, `hls`, `local` | `ftw` |
| GeoAI | `geoai` | Mask R-CNN ResNet50-FPN instance segmentation (geoai-py) | **no** (no published field weights) | yes | yes | all imagery sources and `local` | `geoai` |
| DINOv3 | `dinov3` | DINOv3 ViT backbone + DPT head (geoai-py) | **no** (no published field weights) | yes | yes | all imagery sources and `local` | `dinov3` |
| Prithvi-EO-2.0 | `prithvi` | Prithvi-EO-2.0 ViT (terratorch): clustering of patch embeddings or a fine-tuned segmentation model | only in `mode="embed"` (and the `pca` baseline) | yes | yes | `sentinel2`, `landsat`, `hls`, `local` | `prithvi` (GFM environment) |
| Embedding clustering | `embedding` | K-means (or spectral) clustering of pre-computed embeddings or of a local feature raster | yes | no | no (CPU) | `google-embedding`, `tessera-embedding`, `local` (feature GeoTIFF) | `embedding` |
| SAM 2.1 automatic masks | `sam-auto` | Segment Anything 2.1 automatic mask generation (a grid of point prompts) on large overlapping tiles, partitioned by score into non-overlapping polygons; opt-in, never a default | yes (published weights) | no | yes | all imagery sources and `local` (R, G, B); per-source defaults for `naip` and `spot`; `sentinel2` (did not transfer) and `landsat` with a WARNING; `sentinel1` only with every setting given | `sam-auto` |
| Crop-sequence zones | `crop-sequence` | zones of uniform multi-year crop history from annual crop-type maps (CSB-like); not field instances | yes | no | no (CPU) | `cdl-sequence`, `aci-sequence`, `local` (stack of class codes) | `crop-sequence` |
| Ensemble | `ensemble` | intersection, union or pixel vote of several engines/models | depends on the members | no | yes | depends on the members | members' extras |

"All imagery sources" = `landsat`, `landsat-pan`, `sentinel2`, `hls`, `naip`,
`usgs-naip-plus`, `spot`, `spot-pan`, `sentinel1`, `umbra`, `capella`, `local`.

agribound has no superpixel engine (SNIC, SLIC). One was tested for 1.1 on
oil palm estate blocks and did not pass the rule fixed for it; see
[Superpixel segmentation](satellite-sources.md#superpixel-segmentation-snic-slic).

`sentinel1` composites hold radar backscatter, not optical reflectance. The
RGB engines (Delineate-Anything, GeoAI, DINOv3) read VV, VH and VV − VH as R,
G and B. GeoAI's and DINOv3's weights were pre-trained on optical images, so
for them these inputs are out of distribution. Delineate Anything v2's
training data, FBIS-73M, include Sentinel-1 patches (76,448 named
`*_S1_10m_256_custom` and 7,852 `*_SYN-S1_256_custom` among the 1,432,187 of
its training list), but how they render the radar bands is not documented, so
agribound's composites may still differ from them. Zero-shot F1 on such
composites was 0.18 and 0.39 on two 5 km smoke squares (default settings;
Lea County 2022; near Davis 2024, not held out) and, in a pilot on example
23's four squares (`max_detections=1000`), 20.5 km² inside the Davis smoke
square and a 415 km² box of Lea County that contains the Lea County one,
from 0.00 to 0.80: 0.80 on the orchard blocks of example 23's Madera
square, 0.455 near Davis and 0.173 in Lea County (2024), and 0.11, 0.06 and
0.00 on example 23's oil palm and olive squares (Twifo Praso, Úbeda, Oro).
The California and Spanish scores are not held out: FBIS-73M uses DWR and
SIGPAC polygons as labels there and holds Sentinel-1 patches that intersect
or lie near the Davis and Madera areas (see
[Sentinel-1](satellite-sources.md#sentinel-1-sentinel1), which
also gives the effect of `stretch_percentiles` and `merge_tile_pieces` on
these composites). In the development run of 1.1, example 23's Sentinel-1
runs gave the pilot's scores and Delineate-Anything on Sentinel-1 scored 0.35
and 0.45 in example 24's two Nevada squares; in example 26's run (2026-10-10)
it scored 0.005 to 0.32 at six radar sites (see the [gallery](../gallery.md)).
Expect to need fine-tuning with all three. The registry's
`source_notes` of these three engines say so, and the agent repeats them as
warnings. SAM refinement reads the composite in the same way; it has no
source note, and agribound does not fine-tune SAM. FTW and Prithvi need
optical reflectance and do not accept the source (see
[Sentinel-1](satellite-sources.md#sentinel-1-sentinel1)).

The open VHR SAR sources `umbra` and `capella` follow the same pattern: one
X-band amplitude band (the median of co-registered, level-normalised dates;
Umbra uncalibrated DN, Capella sigma0 dB + 50), read as R, G and B by
Delineate-Anything, GeoAI, DINOv3 and SAM; out of distribution for GeoAI and
DINOv3 (fine-tune on these composites); refused by FTW, Prithvi and the
embedding engine. Delineate-Anything needs
`engine_params={"processing_gsd_m": 3, "merge_tile_pieces": False}` on them.
In the Umbra prototype, the median scored F1 0.281, 0.007, 0.120 and 0.167
with the default settings and 0.420, 0.149, 0.257 and 0.317 with these
(reference fields down to 100 m²; one date with the defaults scored
0.00-0.08). With the fields of at least 2,500 m², the fixed recipe scored
0.439 near UC Davis (2024), 0.163 at Centerfield (2025), 0.257 near Purdue
(2024) and 0.360 on Mallorca (2026), against 0.426, 0.360, 0.240 and 0.460 for
orthophotos; Capella amplitude near Sevilla scored 0.50-0.54. None of these
areas is held out. In the held-out test of the fixed recipe (OpenET field
boundaries of at least 2,500 m²), F1 was 0.390 and 0.261 at two Umbra sites
(Texas, Ohio) and 0.274 at one Capella site (Vermont), against 0.284, 0.193
and 0.261 for NAIP + Delineate-Anything with its defaults and 0.403, 0.310 and
0.249 with the SAR settings (see
[Open VHR SAR](satellite-sources.md#open-vhr-sar-umbra-capella)). Example 26
repeats these runs; in its run of 2026-10-10 with the development code of
agribound 1.1 they scored 0.370, 0.261 and 0.274, against 0.282, 0.193 and
0.261 for NAIP with the defaults and 0.406, 0.310 and 0.249 with the SAR
settings, and 0.485 near UC Davis, 0.150 at Centerfield and 0.513 near Sevilla
(in-sample).

Every engine attaches `gdf.attrs["engine_meta"]` (backend, model key, weights
repository, revision and SHA-256 where applicable, thresholds, device, ...),
which the pipeline copies into the [provenance record](reproducibility.md).
There are no silent fallbacks to another model, band set or engine: a
configuration that cannot run raises an error that says what is missing, and
anything that degrades is logged at WARNING and recorded in `engine_meta`.

Every engine class implements `prefetch(config)`, which downloads its weights
for offline use (`agribound prefetch --engine <name>`, see
[HPC](hpc.md#prefetching-weights)).

---

## Delineate-Anything (`delineate-anything`)

YOLO11-seg instance segmentation trained on 0.25-10 m imagery. The weights come
from the Hugging Face repository `MykolaL/DelineateAnything` at **pinned
revisions**, and the SHA-256 of each file is checked before use:

| `engine_params["da_model"]` | File | Model | Default `conf_threshold` |
|---|---|---|---|
| `large_v2` (default) | `DelineateAnythingv2.pt` @ `369d0b4c` | Delineate Anything v2, YOLO11x-seg, trained on FBIS-73M | 0.15 |
| `large` | `DelineateAnything.pt` @ `029e9a94` | YOLO11x-seg, trained on FBIS-22M | 0.005 |
| `small` | `DelineateAnything-S.pt` @ `029e9a94` | YOLO11n-seg, trained on FBIS-22M | 0.005 |

The default confidences are those of the upstream sample configurations. The
aliases `"DelineateAnythingV2"`, `"DelineateAnything"` and
`"DelineateAnything-S"` are accepted; the legacy `model_size`
(`"large"`/`"small"`) selects the v1 models.

**Backends** (`engine_params["backend"]`, no automatic fallback between them):

- `"native"` (default): agribound's own tiled Ultralytics inference. It
  reproduces the reference pipeline's preprocessing: a scene-level per-band
  1-99 percentile stretch to uint8 (uint8 rasters are used unchanged), 512 px
  tiles below 4 m ground sampling distance (GSD), else 256 px tiles upsampled
  2× so the model input is always 512 × 512, 50 % tile overlap, BGR channel
  order for Ultralytics, FP16 on GPU/MPS. A detection is *tile-cut* when it
  reaches within 4 px (of the 512 px model input) of an interior tile edge,
  the strip the reference pipeline marks. Detections from all tiles are
  combined at polygon level: tile-cut pieces of one field that duplicate one
  another (IoU >= 0.3 or 80 % of the smaller piece; `merge_tile_pieces`,
  default True) are merged, so a field larger than a tile is rebuilt when its
  pieces overlap that much. Then greedy non-maximum suppression, and overlap
  resolution, which gives contested area to complete (not tile-cut)
  detections first, then to the higher confidence, so that the straight edge
  where a tile cut a piece is not cut into a field that another tile saw
  whole. Last, tile-cut pieces that meet along a tile edge without
  overlapping are joined when they pass the reference pipeline's seam test
  (`merge_tile_seams`, default True; a port of upstream commit 488e537 with
  its thresholds): they touch along at least 8 px and 30 % of the longer cut
  edge; the cut edges start and end within max(8 px, a quarter of the
  contact) of each other, unless the smaller piece is a strip at most 16 px
  deep; the tile centred on that edge did not see a boundary there; and the
  image looks alike on both sides. Pieces that fail the test stay separate, cut
  along the tile edge, and a piece with nothing detected beyond its tile
  edge still ends there (example 23: oil palm blocks of about 1 km on
  SPOT-Pan, with 768 m tiles). `min_field_area_m2` drops the raw detections
  smaller than it before these merges, and again after them, so a tile-cut
  piece smaller than the minimum area is not joined to its neighbour
  (upstream applies its minimum area only after its seam merge). Results
  are close to, but not identical with, the `reference` backend. Returns a `confidence` column.
- `"reference"`: runs the upstream Delineate-Anything pipeline
  (`methods.main.inference.execute`) in a subprocess from a checkout given by
  `engine_params["da_repo"]` or `AGRIBOUND_DA_REPO`. Needs the GDAL Python
  bindings (`osgeo`, conda-forge `gdal`) and `numba` (included in the
  `delineate-anything` extra), and a checkout at upstream commit 34eddf7 or
  later.
- `"ftw"`: `ftw_tools.inference.inference.run_instance_segmentation`. FTW's
  wrapper divides the first three bands by 3000, so only
  `reflectance_x10000` composites are accepted. `large_v2` needs an ftw-tools
  build whose model registry contains `DelineateAnythingV2` (ftw-baselines
  main at fa86d4a or later; not in ftw-tools 2.0.0b5).

**Parameters** (all optional; the full list is in the
[API reference](../api/engines.md#delineate-anything)):
`conf_threshold`, `batch_size` (4), `checkpoint_path` (fine-tuned weights;
set by the pipeline after fine-tuning), `super_resolution` (1, 2 or 4),
`tile_step` (0.5), `half`, `iou_threshold` (NMS IoU, 0.3), `max_detections`
(300), `dedup_iou` (0.3), `dedup_containment` (0.8), `merge_tile_pieces`
(True), `resolve_overlaps` (True), `merge_tile_seams` (native backend; True
when `resolve_overlaps` is True and `tile_step` is 0.5, which it needs, and
False otherwise), `processing_gsd_m` and `stretch_percentiles` (native
backend; see [Fields larger than the tiles](#fields-larger-than-the-tiles)),
`min_hole_area_m2` (reference backend, 2500 m²). A parameter that the
selected backend cannot honour raises `ValueError`, as does
`merge_tile_seams=True` with `resolve_overlaps=False` or another
`tile_step`.

!!! warning "Changed in 1.0.0"
    The confidence parameter is `conf_threshold`; the old names
    `confidence` and `minimal_confidence` now raise `ValueError`.

!!! note "Changed in 1.1.0: tile stitching of the native backend"
    The 4 px tile-cut strip, the complete-first overlap resolution and the
    seam merge change the native backend's polygons for the same
    configuration, so its outputs of agribound 1.0.1 or earlier are no
    longer reused (results version `delineate_anything` 2; see
    [Output reuse](reproducibility.md#output-reuse)). On the cached SPOT-Pan
    composites of examples 22 and 23 (released weights), the share of
    boundary length within 2 m of a tile edge fell from 14.0 % to 12.7 % at
    Twifo Praso and from 4.6 % to 2.5 % in the Cauvery delta (about 2 % by
    chance), and the Twifo split rate from 0.49 to 0.44; with the Higaturu
    model fine-tuned with the earlier recipe at Oro it fell from 20.9 % to
    13.7 %, and the split rate from 0.34 to 0.23 (the model of the current
    recipe gives 3.2 % and 0.25). The seam merge also joins some pieces of
    two fields, as the upstream configuration guide warns for fields that
    meet on a tile edge: of the joins that could be checked against the
    references (both pieces at least half in one reference field), 1 of 4 at
    Twifo and 6 of 18 with that earlier Higaturu model at Oro (0 of 1 with
    the current one). It also joins fields whose gap runs close to the edge
    but not on it: at Madera (example 23), in the development run of 1.1, 2
    of the 3 checkable joins of the SPOT-Pan output merged two orchard blocks
    whose road gap runs up to 21 m off the tile edge, at 2-3° to it (the one
    checkable join of the NAIP output also merged two fields). The test reads
    a strip 2 px wide along the edge (upstream's `seam_split_window_px` of 1)
    in the tile centred on it, and where one block's outline crosses the edge
    it finds no boundary there. `merge_tile_seams=False` turns the seam merge
    off.

**What the stitching still gets wrong.** The development run of 1.1 shows
these cases (see the [gallery](../gallery.md)):

- A field detected in some tiles and not in the next ones ends on the edge
  of the last tile that detected it, as do the blocks between the centre
  pivots of western Bahia (example 22: 10.1 % of the boundary there lies
  within 2 m of a tile edge, about 2 % by chance).
- `merge_tile_pieces` chains tile-cut detections across fields that the
  image separates. In example 25's Montana square 20 of the 83 polygons are
  larger than one 512 m tile, up to 359.6 ha (with the merging of tile
  pieces off, the largest is 78.4 ha); on the in-sample NORPALM square of
  example 23 one polygon of the fine-tuned model covers 7,634.7 ha.
- In the Nile Delta (example 22) a block of strip plots is drawn as one
  polygon: one tile returned a low-confidence outline of the whole block
  (0.29), which ends 2 px short of the tile's edge and so counts as tile-cut;
  it was merged with 18 tile-cut pieces, took their highest confidence
  (0.70) and then suppressed the 76 detections inside it.
- Where the tiles that see a centre pivot whole score it below the
  confidence threshold, only the pieces from the tiles that cut it are kept
  (example 13's input; the upstream pipeline gives the same pieces).

Candidate fixes were evaluated on the native Delineate-Anything outputs of
that run against rules set before the evaluation. Each lowered F1 by more
than 0.005 in some evaluation or put more boundary on tile edges in some
output, so none was adopted, and the results version stays 2.

`min_field_area_m2` is applied as an absolute area computed in the equal-area
EPSG:6933 by every backend. For rasters whose GSD lies more than 5 % outside
the 0.25-10 m training range (for example 15 m Landsat panchromatic, or 30 m
Landsat or HLS) a WARNING is logged and
`engine_meta["gsd_outside_training_range"]` is True. Panchromatic composites
(`spot-pan`, `landsat-pan`) have one band, which the engine reads as R, G and
B, a grey image. Sentinel-1 composites (`sentinel1`) are radar backscatter
(VV, VH and VV − VH as R, G and B); FBIS-73M, the training data of Delineate
Anything v2, includes Sentinel-1 patches whose rendering is not documented,
and zero-shot F1 on agribound's composites ranged from 0.00 to 0.80 (0.80
only on Madera's large orchard blocks, not held out). The GSD check sees
their 10 m pixels and logs no WARNING, although their resolution is about
20 × 22 m (see [Sentinel-1](satellite-sources.md#sentinel-1-sentinel1) for a
zero-shot test). The Umbra and Capella composites (`umbra`, `capella`) are
float32 X-band amplitude, so the scene stretch applies; read them with
`processing_gsd_m=3` and `merge_tile_pieces=False`, the recipe their scores
were measured with (see
[Open VHR SAR](satellite-sources.md#open-vhr-sar-umbra-capella)).
In the development run of agribound 1.1, example 20 ran Delineate Anything v2
as released on 2022 composites of San Juan County, New Mexico. Against the
1,355 DRI/NMOSE field boundaries (V18) of at least 2,500 m² in its box, which
were not used for training or fine-tuning in these runs (the model's training
set, FBIS-73M, has no patch in New Mexico in its public lists: the nearest
lies 90.6 km from the box, in Arizona and unlabelled, and the nearest
labelled patch 604.4 km away, checked on 2026-10-13), F1 was 0.13 on
Landsat (30 m), 0.29 on Sentinel-2 (10 m), 0.17 on SPOT 6/7 (6 m; a single
scene with valid data over 76.7 % of the box) and 0.39 on NAIP (1 m), and
recall rose from 0.07 to 0.34; see the [gallery](../gallery.md). The runs of
agribound 1.0.0 on 2018 composites, against the NMOSE WUCB polygons (the New
Mexico reference before V18), gave 0.15, 0.34, 0.33 and 0.43.

The Delineate-Anything model code and weights, and Ultralytics, are AGPL-3.0.

### Fields larger than the tiles

A tile of 512 px covers 768 m of 1.5 m imagery, so no tile sees the whole of
a larger field, and the native backend can only rebuild such a field from
pieces (see the stitching above). Two native-backend options change the input
the model sees. Neither is set by default, and a run that does not set them
keeps its results, cache keys and configuration hash.

- `processing_gsd_m` (metres): read the raster at this coarser pixel size, so
  a tile covers more ground (1536 m at 3 m). The raster is resampled once by
  average resampling (each new pixel is the area-weighted mean of the valid
  pixels it covers) and cached in the working directory; everything else then
  runs as on a raster of that pixel size, including the 4 m super-resolution
  rule and the check of the 0.25-10 m training range. The polygons are in the
  raster's CRS but are traced on the coarser pixels, and small fields and
  narrow gaps between fields blur. Values outside the 0.25-10 m training
  range, or finer than the raster's pixels by more than 1 %, raise
  `ValueError`; a value within 1 % of the raster's pixel size reads the raster
  as it is. `engine_meta` records `processing_gsd_m` and `processing` (pixel
  size, factor, size and path of the resampled copy; `None` when the raster
  is read as it is).
- `stretch_percentiles` (`[low, high]`, 0 ≤ low < high ≤ 100; default
  `[1, 99]`, upstream's): the percentiles of the scene stretch to uint8. A
  wider range stretches the values less, which matters where the scene has an
  even tone, such as closed canopy. uint8 rasters (NAIP, `sentinel1`
  composites) are not stretched, so the option has no effect on them
  (WARNING); the float32 `umbra` and `capella` composites are. On Sentinel-1, a float copy of the composite stretched at
  (0.1, 99.9) scored lower than the uint8 composite in 4 of 6 pilot areas
  (see [Sentinel-1](satellite-sources.md#sentinel-1-sentinel1)).

```python
gdf = agribound.delineate(
    study_area="area.geojson",
    source="spot-pan",
    year=2020,
    engine="delineate-anything",
    gee_project="my-project",
    engine_params={"processing_gsd_m": 3, "stretch_percentiles": [0.1, 99.9]},
)
```

```bash
agribound delineate --study-area area.geojson --source spot-pan --year 2020 \
    --engine delineate-anything --gee-project my-project \
    --engine-param processing_gsd_m=3 --engine-param 'stretch_percentiles=[0.1, 99.9]'
```

Fine-tuning cuts its chips from the same resampled raster, with the same
stretch (see [Fine-Tuning](fine-tuning.md#training-data)), and records the
input the chips gave the model: pixel size, super-resolution factor and
stretch. A run that gives a fine-tuned checkpoint another input (pixel sizes
more than 1 % apart, another factor or another stretch; a uint8 raster is not
stretched) logs a WARNING and records both in
`engine_meta["checkpoint_input_mismatch"]`. The options themselves are not
compared: a checkpoint fine-tuned on 1.5 m chips gets the WARNING on a 3 m
raster even without `processing_gsd_m`, and `processing_gsd_m` equal to the
raster's own pixel size changes nothing.

At Twifo Praso (example 23), 54 of the 55 oil palm blocks are longer than a
768 m tile. With the released weights on the cached SPOT-Pan composite,
scored as in the [gallery](../gallery.md) (IoU ≥ 0.5; precision among the
polygons that overlap a block, because the RSPO map does not cover every field
in the square):

| `engine_params` | Fields | Recall | Precision | F1 | Split | Merged | Boundary on tile edges (chance) |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 122 | 0.49 | 0.23 | 0.32 | 0.44 | 0.05 | 12.7 % (2.1 %) |
| `processing_gsd_m=3` | 132 | 0.56 | 0.26 | 0.35 | 0.36 | 0.18 | 3.1 % (1.0 %) |
| `stretch_percentiles=[0.1, 99.9]` | 112 | 0.65 | 0.35 | 0.46 | 0.40 | 0.20 | 8.7 % (2.1 %) |
| both | 134 | 0.71 | 0.32 | 0.44 | 0.35 | 0.05 | 2.7 % (1.0 %) |

"Boundary on tile edges" is the share of the boundary length within 2 m of an
interior edge of the run's own tiles (every 384 m at 1.5 m, every 768 m at
3 m); chance is the share of the area within 2 m of those edges. Precision
over all polygons was 0.22, 0.23, 0.32 and 0.29. The merge rates are those of
3, 10, 11 and 3 of the 55 blocks. At 1.5 m the 0.1-99.9 % stretch maps
886-1439 DN of this composite to 0-255, instead of 909-1199 DN; at 3 m the
backend runs 64 tiles instead of 225. Both values were picked while
investigating the tile-edge lines at this site, so the table shows what the
options do there, not how well they work elsewhere.

On NAIP (1 m) a tile of 512 px covers 512 m, less than a centre pivot (about
800 m across); at `processing_gsd_m=3` it covers 1,536 m. Delineate Anything
v2 as released with `processing_gsd_m=3` was tested by rules set before the
test (PREREG-DV3, 2026-10-12; scored on 2026-10-13; amended on 2026-10-13 for
a defect of its check step that changed no layer or score) on three Nevada
pivot squares and example 24's Mason Valley square: F1 rose by +0.04,
+0.15, +0.13 and +0.10 (the interval of the first includes 0). It gives
fewer, larger
polygons (24–39 % fewer, median area 1.5–5.1 times larger on these four
squares and Diamond Valley's), and the gain is mostly precision: the matched
fields changed by −4 to +6. It had failed a first development pilot in
Diamond Valley and passed a second, so the test was a third look. On example
24's Diamond Valley square it was not blind: example 26 had scored it there
with `merge_tile_pieces=False` (F1 0.26 to 0.30), and the option alone gives
0.26 to 0.31. Example 24's `--options` runs use it with Delineate-Anything
on NAIP.

## Fields of The World (`ftw`)

Runs an ftw-tools checkpoint on R, G, B and NIR and polygonises the predicted
field class. The default model is the ftw-tools `MODEL_REGISTRY` entry marked
`default`, which in ftw-tools 2.0.0b5 is `FTW_PRUE_EFNET_B5` (a PRUE U-Net
with an EfficientNet-B5 encoder, two input windows). List the models with
`agribound list-ftw-models` (`--all` includes legacy models) and choose one
with `engine_params["model"]`, or pass a local checkpoint with
`engine_params["checkpoint_path"]`. Instance-segmentation registry entries
(Delineate-Anything) are rejected; use the `delineate-anything` engine.

**Two-window models.** The number of windows follows the model (registry
`requires_window`, or `in_channels` of a checkpoint: 4 = one window, 8 =
two). Two-window models take `[R, G, B, NIR]` of an early-season window A
followed by the same bands of a late-season window B, the order that
ftw-tools' own inference input builder writes (FTW's training data layout
stacks the windows in the other order; in a live Beauce 2024 test, swapping
the order changed about 2 % of the predicted pixels). The window centres are FTW's
summer-crop start and end of season over the study-area bounding box (from
ftw-tools' crop calendar; the end moves to `year + 1` for southern-hemisphere
seasons). Each window is a median composite over `centre ± window_days`
(default 30) built by the source's composite builder with its own cache
entry. `engine_params["window_dates"]` (two `"YYYY-MM-DD"` centres) replaces
the crop calendar. If a window has no imagery the run fails with an error
that names `window_dates`, `window_days` and `allow_annual_fallback`;
`allow_annual_fallback=True` uses the annual composite for that window
(WARNING, recorded in `engine_meta`). For `source="local"` a two-window model
needs `stacked_windows=True` with bands 1-4 and 5-8 holding the two windows,
or `allow_annual_fallback=True`.

**Radiometry.** ftw-tools divides the input by 3000, i.e. it expects
Sentinel-2 L2A reflectance × 10000. Sentinel-2, Landsat and HLS composites
are on that scale and are used unchanged; Landsat and HLS are nevertheless
outside the Sentinel-2 training distribution (WARNING,
`engine_meta["out_of_distribution_source"] = True`). `local` rasters need
`engine_params["value_scale"]`.

**Polygonisation.** Prediction rasters in a geographic CRS, a CRS whose unit
is not the metre, or a Mercator/Web Mercator CRS are reprojected to the UTM
zone of the study-area centre before `polygonize`, so `simplify` and
`min_size` are in metres. `close_interiors` (default True) fills holes;
combining it with `erode_dilate` or `dilate_erode` needs ftw-baselines main
(ftw-tools 2.0.0b5 raises, and agribound raises `ValueError` before building
any input).

!!! note "macOS: use a `__main__` guard"
    ftw-tools' data-loader workers (`num_workers = config.n_workers`) use the
    `spawn` start method on macOS, which re-imports the main script. Scripts
    that run FTW (or the Delineate-Anything `ftw` backend) must put their code
    under `if __name__ == "__main__":`, otherwise the run crashes. Setting
    `n_workers=0` loads data in the main process.

FTW models cannot be fine-tuned in agribound: its one-composite-per-chip
training data does not match FTW's training layout. Train with ftw-baselines
(`ftw model fit -c <config.yaml>`) and pass the checkpoint with
`engine_params={"checkpoint_path": ...}`.

## GeoAI (`geoai`)

torchvision Mask R-CNN ResNet50-FPN (2 classes) run through geoai-py's
instance-segmentation workflow. **No field-boundary weights are published**
for geoai (as of 2026-09 the `giswqs/geoai` Hugging Face repository holds
building, car, ship, solar-panel, parking, water and wetland models, and
geoai's default detector weights detect buildings). The engine therefore
requires a checkpoint and never falls back to other weights:

- `fine_tune=True` with `reference_boundaries` (see [Fine-tuning](fine-tuning.md)), or
- `engine_params["checkpoint_path"]` (a 2-class, 3-channel Mask R-CNN state
  dict), or `repo_id` + `filename` (+ `revision`) for a file on Hugging Face.
  `checkpoint_path` and `repo_id` cannot both be set.

Input: canonical R, G, B with a scene-level 1-99 percentile stretch to uint8,
the same as the fine-tuning chips. Mask R-CNN resizes every image so its
shorter side is 800 px, so the inference window sets the apparent field size;
the window (`window_size`) therefore defaults to the training chip size
recorded next to the checkpoint (keep them equal). Mask R-CNN cannot detect a
field larger than the window as one instance, so fine-tuning sizes the chip
from the reference fields by default (1.25 × their 90th-percentile
bounding-box side, rounded up to a
multiple of 32 px and clamped to 256–1024 px; see
[Fine-tuning](fine-tuning.md#training-data)). geoai keeps partial detections
of a field from overlapping windows, so the engine joins instances split
along the window edges (`merge_window_seams`, default True): two instances
are joined when they meet across an interior window edge along at least
`seam_min_px` pixels (default 16) and at least half the shorter of their two
runs on that edge, with at most `seam_max_gap_px` background pixels (default
2) between them; those gaps are then filled. Instances that touch anywhere
else are not joined. `engine_meta` records `n_instances_merged_at_seams` and
`n_seam_gap_pixels_filled`. In example 12's NAIP run of agribound 1.0
(centre pivots of about 800 m at 1 m), F1 against the NMOSE WUCB polygons
(the New Mexico reference before V18) was 0.01 with 256 px chips, 0.25 with
1,024 px chips and 0.48 after joining (in-sample). Fine-tuned on the
DRI/NMOSE field boundaries (V18) in the development run of 1.1, with the same
chip rule and joining (370 instances joined), it scored 0.29 against V18
(in-sample), lower than the 1.0 model at both references (0.44 against V18,
0.48 against WUCB, where the 1.1 model scores 0.26). The NAIP composite, the
GeoAI code and settings and the seed were the same; the labels (755
training instances against 466), 16 more chips and the operating system
differed. Which of these lowered the score was not established, and the
spread between trainings with other seeds was not measured (see the
[gallery](../gallery.md)). Mask R-CNN
keeps at most 100 detections per window, and a `confidence_threshold` (default
0.5) below its internal score threshold of 0.05 acts as 0.05; for dense small
fields, fine-tune with a smaller `chip_size`. On Apple MPS the model runs on CPU
(WARNING): on MPS it reported Metal command-buffer errors and its detections
differed from CPU.

## DINOv3 (`dinov3`)

geoai-py's `DINOv3Segmenter`: a DINOv3 ViT backbone with a DPT decoder,
trained by agribound into background / field interior / field boundary.
**There are no published field-boundary weights**, so a fine-tuned
checkpoint (`fine_tune=True`, or `engine_params["checkpoint_path"]`) is
required.

- Backbone: SAT-493M ViT-L/16 (`giswqs/geoai` / `dinov3_vitl16_sat493m.pth`)
  built with `torch.hub` from `facebookresearch/dinov3` (or the local clone in
  `DINOV3_LOCATION`). SAT-493M weights exist only for ViT-L/16 and ViT-7B/16;
  other sizes (`dinov3_model="small"`/`"base"`) need `weights_path`.
- Fine-tuning defaults to **full fine-tuning** (about 303 M backbone
  parameters for ViT-L/16 plus the decoder). `use_lora=True` trains rank-4
  LoRA adapters on the attention `qkv` layers (about 0.39 M parameters) on a
  frozen backbone; `freeze_backbone=True` alone trains the decoder only. The
  recipe (class weights, the checkpoint metric, the boundary width) is
  described in [Fine-tuning](fine-tuning.md#engine-specific-settings).
- Decoder: geoai's DPT head predicts the classes on the 16 px patch grid of
  the ViT and upsamples them bilinearly, so a thin band of the boundary class
  is hard to learn: between two touching fields the band is
  2 × `boundary_erosion` pixels wide (8 px, half a patch, at the default of
  4 px; 4 px at agribound 1.0.1's 2 px). The default was chosen on 1.5 m
  imagery (see [Fine-tuning](fine-tuning.md#engine-specific-settings)).
- Grid: a raster with pixels more than 1 % larger than 10 m (Landsat and
  HLS, 30 m; Landsat PAN, 15 m) is read on a bilinear copy with 10 m pixels,
  written once to the cache, both for fine-tuning and at inference
  (`processing_gsd_m`: `"auto"`, the default; `"native"` reads the raster as
  it is; a number of metres finer than the raster's pixels sets the grid,
  and a coarser one raises `ValueError`). At 30 m a patch is 480 m across.
  The polygons are in the raster's CRS on the 10 m grid, and
  `engine_meta["processing"]` records the copy. Pixel sizes are in metres,
  also in a projected CRS in feet, and the copy holds the 10 m pixels that
  lie inside the raster. Without `processing_gsd_m`, inference reads the
  grid the checkpoint was trained on (`processing_gsd_m` in
  `<checkpoint>.agribound.json`, or in the checkpoint's hyper-parameters
  when that file is missing), so a checkpoint trained on a raster's own grid
  reads rasters as they are. Rasters of 10 m or finer (Sentinel-1,
  Sentinel-2, SPOT 6/7, NAIP) are read as they are unless `processing_gsd_m`
  gives a finer grid. Fine-tuned outputs made before the grid on coarser
  rasters are not reused (results version `dinov3_processing_grid`).
- Input: canonical R, G, B with a scene-level percentile stretch, as float
  `uint8 / 255`, which the model normalises with the mean and standard
  deviation its backbone was pre-trained with (for the SAT-493M weights mean
  (0.430, 0.411, 0.296) and standard deviation (0.213, 0.156, 0.143), from the
  [facebookresearch/dinov3 README](https://github.com/facebookresearch/dinov3#image-transforms)).
  The statistics are saved in the checkpoint and applied inside the model, so
  training and inference normalise alike (`engine_meta["input_normalization"]`).
  A checkpoint of agribound 1.0.1 or earlier was trained on unnormalised input
  and runs without normalisation.
- Inference: windows of the training chip size (a multiple of 16; read from
  `<checkpoint>.agribound.json` or, for a checkpoint fine-tuned after 1.0.1,
  from the checkpoint), capped at the raster size, half a window apart
  (`window_size`, `overlap`), with the
  input mirror-padded so that no window is zero-padded. agribound runs the
  sliding window itself: with `window_stitching="blend"` (default) each pixel
  gets the class with the highest average probability over the windows that
  cover it, each window weighted by a cosine taper that is 1 in its centre and
  falls towards its edges over the overlap (the weights of neighbouring
  windows sum to 1). `"crop"` keeps the prediction of the window whose centre
  part a pixel is in, the rule of geoai's `dinov3_segment_geotiff`, which
  agribound 1.0.1 used. Neighbouring windows disagree near their edges, so
  `"crop"` changes class along straight lines in the middle of each overlap:
  at Twifo Praso (example 23), the 1.0.1 segmentation changed class about
  25 times more often across those lines than elsewhere, and 66 % of the long straight
  north-south and east-west segments of its outline lay on them (about 0.9 %
  would by chance). With a model of the current recipe, 13.5 % of them lay
  on those lines with `"crop"` and 0.6 % with `"blend"`.
- Each field-interior region is grown back over the predicted boundary class
  by the training boundary width (`dilate_interior_px`; by default the
  `boundary_erosion` recorded with the checkpoint), so neighbouring polygons
  never overlap; right-angled convex corners lose k(k+1)/2 pixels (10 px at
  the default k = 4, 3 px at the 1.0.1 default k = 2).
- Fine-tuned models can predict a boundary band wider than the band they were
  trained on, and the default growth (the trained width) then leaves the
  outlines inside the fields: in example 24 a median 35 m inside on SPOT 6/7
  (6 m) in Diamond Valley, where boundary F1 is 0.07 at F1 0.50.
  `dilate_interior_px="half_band"` (opt-in) measures the band on each
  prediction, T (the median city-block distance in pixels from the boundary
  pixels next to background to the nearest interior pixel), and grows each
  interior by g = ceil(T/2) px, so that where a field meets background the
  outline falls in the middle of the band; where two fields meet, both
  interiors grow into the band between them until they meet or have grown g
  px. Without an interior pixel or a boundary pixel next to background, T is
  undefined and the trained width is used. `engine_meta` records
  `dilate_interior_rule`, `band_thickness_px` (T), `band_edge_px`,
  `dilate_fallback` and `trained_dilate_px` (only with the option); without
  it, outputs, configuration hashes and results versions are unchanged.
  Prithvi refuses `"half_band"`, and the value is checked when the
  configuration is built.

  In a pre-registered pilot on development squares (2026-10-12) the rule
  raised boundary F1 on 8 of 9 held-out runs, with F1 changes of −0.003 to
  +0.033. In the confirmatory test (PREREG-DV3: rules fixed on 2026-10-12,
  scored on 2026-10-13; three Nevada pivot squares in other basins, chosen
  before any model ran on them, with example 24's Diamond Valley models, so
  outside the basin they learned from) it raised boundary F1 on 5 of 6 runs,
  the fewest its rule allowed: on SPOT 6/7 from 0.07–0.09 to 0.28–0.33
  (T 20–25 px, g 10–13 px; F1 +0.000 to +0.052), on NAIP by −0.003 to +0.017
  (g 5–7 px; F1 −0.002 to +0.018). Without it, the SPOT outlines' boundary F1
  was about that of an image-blind grid of squares of the layer's median
  polygon area (0.07–0.08); with it, well above. PREREG-DV3 was amended on
  2026-10-13 for a defect of its check step that changed no layer or score.
  Limits:
  - it overshoots where the predicted boundary class covers the ground
    between fields (on a Diamond Valley development square, SPOT: g 13 px,
    outlines a median 14 m outside the fields) and falls short where the band
    is narrow (on a Mason Valley development square, SPOT: 8 m inside);
  - a width-matched rule (the growth set from the fine-tune's validation
    chips) placed those development SPOT outlines closer (boundary F1 0.27
    and 0.28 against 0.24 and 0.25) but raised boundary F1 on fewer runs (6
    of 9), so the pilot's rule chose `"half_band"`;
  - a model that predicts exactly its trained band of k px gets ceil(k/2),
    less than the default k;
  - on NAIP the change is small.

  On example 24's evaluation squares its effect was known before the test
  (an earlier sweep of the growth on the same rasters): Diamond Valley SPOT
  boundary F1 0.07 to 0.37 (g 12 px; F1 0.50 to 0.55), Mason Valley SPOT
  0.19 to 0.22 (g 5 px), Diamond Valley NAIP 0.60 to 0.62 (g 6 px), Mason
  Valley NAIP unchanged (g 4 px, the trained width). Example 24's `--options`
  runs use it (see [Evaluation](evaluation.md) and the
  [gallery](../gallery.md), entry 24e).

Offline nodes: run `agribound prefetch --engine dinov3` first, then set
`DINOV3_LOCATION` to the returned hub directory and `HF_HUB_OFFLINE=1`.

Licence: the DINOv3 weights are Meta's "DINO Materials" under the
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
(a custom licence, not OSI-approved; last updated 19 August 2025), which also
covers re-hosted copies such as the `giswqs/geoai` file above. Its clause
1.b.ii requires publications of research performed using DINO Materials to
acknowledge their use, and clause 1.b.i requires a copy of the licence to be
provided when the weights (or derivatives, such as fine-tuned checkpoints) are
redistributed.

## Prithvi-EO-2.0 (`prithvi`)

Prithvi-EO-2.0 (default `model_name="Prithvi-EO-2.0-300M-TL"`) built from the
terratorch backbone registry and run on single-date composites
(`num_frames=1`). It needs terratorch, which requires `lightning>=2.6` and
therefore cannot share an environment with ftw-tools 2.x; use
`environment-gfm.yml` or `pip install "agribound[all-gfm]"`.

| `engine_params["mode"]` | Label-free | What it does |
|---|---|---|
| `"embed"` (default without a checkpoint) | yes | Patch-token features of one encoder layer, interpolated to pixel resolution and clustered with K-means; 4-connected regions of one cluster become polygons. Clusters are land-cover segments, not field instances. |
| `"segment"` (default with `checkpoint_path`) | no | A Prithvi + UPerNet segmentation model fine-tuned by agribound (or any terratorch `SemanticSegmentationTask` checkpoint with the same bands, normalisation and classes: 1 field interior, 2 field boundary), run with terratorch's tiled inference. Interiors are grown over the boundary class, as for DINOv3. |
| `"pca"` | yes | Baseline without the ViT: K-means on the PCA of per-band z-scores of R, G, B, NIR. |

Inputs are Blue, Green, Red, narrow NIR, SWIR 1 and SWIR 2 as reflectance ×
10000, normalised with the Prithvi-EO-2.0 means and standard deviations. The
NIR input is `NIR_NARROW` where the source defines it (Sentinel-2 B8A, HLS B5)
and `NIR` (SR_B5) for Landsat, which is the broad TM/ETM+ NIR on Landsat 5/7.
`local` rasters need `engine_params["value_scale"]`
(`"reflectance_x10000"` or `"unit"`).

On Apple MPS, Prithvi + UPerNet runs only where the coarsest feature map is
1 px or a multiple of 6 px (for example 192 px tiles with `tile_size=192` and
`chip_size=192`); other sizes run on CPU with a WARNING.

## Embedding clustering (`embedding`)

Clusters pre-computed per-pixel embeddings (`google-embedding`, 64-D;
`tessera-embedding`, 128-D) and polygonises every connected region of every
cluster. No labels, weights or GPU are needed. Clusters are land-cover
segments, not field instances; non-cropland segments are removed only by the
area and LULC filters.

Defaults (`engine_params`): `use_pca=True`, `pca_components=16`,
`n_clusters="auto"` (silhouette over `k_candidates` 5, 10, 15, 20, 30, 50),
`clustering_method="kmeans"` (`KMeans(n_init=10)`, see the note below;
`"spectral"` is slower), sample sizes 100,000 / 50,000 / 5,000 for PCA,
clustering and silhouette, and `max_block_mb=256` (the raster is read in row
blocks, so memory is bounded). `matryoshka_depth` (4, 16, 32 or 64) clusters
a Matryoshka prefix of TESSERA v2 embeddings instead of PCA. Every random
choice is seeded from `config.seed`.

!!! note "Changed in 1.0.1: complete k-means restarts"
    With `clustering_method="kmeans"`, agribound 1.0.1 fits scikit-learn
    `KMeans(n_init=10)` on the clustering sample (50,000 pixels by default)
    whatever the raster size: ten complete restarts, of which the one with
    the lowest error (inertia) is kept. The silhouette selection of k
    (`n_clusters="auto"`) uses `KMeans(n_init=10)` too, on the first 5,000
    pixels of that sample. agribound 1.0.0 and 0.1.x used
    `MiniBatchKMeans(batch_size=10000, n_init=3)` when the raster had more
    than 100,000 valid pixels, `KMeans(n_init=5)` otherwise, and
    `MiniBatchKMeans(n_init=3, batch_size=5000)` for the silhouette
    selection. `engine_meta` records `clusterer` (`"KMeans"`),
    `kmeans_n_init` and `inertia`.

    The reason is that on large rasters the 1.0.0 result depended on the
    pixel sample. `MiniBatchKMeans` runs once from the best of its `n_init`
    starting points and stops early, so different samples can end in
    different solutions. With 31 seeds on each of two TESSERA rasters of
    example 15 (Pampas; the 1.0.0 raster and the 0.1.x mosaic; 2026-09-29)
    and k fixed at 5 (the 1.0.0 silhouette selection chose 5 at seed 42, but
    would have chosen 10 for 3 of these 62 samples), the 1.0.0 code reached
    the lower-error solution in 30 of 62 runs. In 18, including the default seed 42 on the 1.0.0 raster, it
    reached a solution with one bare-soil cluster fewer; the rest stopped
    early. The joined bare-soil cluster merges neighbouring bare fields into
    polygons of several hundred hectares: after the crop filter, 38 % of the
    area was in polygons over 200 ha, against 18 % with the lower-error
    solution. `KMeans(n_init=10)` reached the lower-error solution in 119 of
    120 new samples at k = 5 (`n_init=5`, the 1.0.0 setting for small
    rasters: 113 of 120). agribound 0.1.x used the same `MiniBatchKMeans`
    settings with unseeded samples, so its runs could differ from one another
    in the same way.

    Inertia of the final fit on the 50,000-pixel fit sample at seed 42
    (2026-09-29):

    | Raster | k | 1.0.0 `MiniBatchKMeans` | 1.0.1 `KMeans(n_init=10)` | Largest cluster's share of the sample, 1.0.0 → 1.0.1 |
    |---|---|---|---|---|
    | Pampas TESSERA (example 15) | 5 | 6,687,488 | 6,381,504 (−4.6 %) | 0.366 → 0.289 |
    | Pampas Google Satellite Embedding (example 15) | 5 | 2,942.7 | 2,733.2 (−7.1 %) | 0.432 → 0.384 |
    | India TESSERA (example 02) | 8 | 4,956,497 | 4,901,440 (−1.1 %) | 0.235 → 0.210 |
    | India Google Satellite Embedding (example 02) | 5 | 3,611.9 | 3,598.9 (−0.4 %) | 0.364 → 0.315 |

    The 1.0.1 example runs of 2026-09-29 recorded the same inertia
    (`engine_meta["inertia"]`) for the two Google fits at the table's
    precision (2,733.24 and 3,598.94), and 6,381,499.5 (Pampas) and
    4,901,444.5 (India) for the TESSERA fits, 4.5 below and 4.5 above the
    table (under 0.0001 %). In the Pampas cluster rasters of those runs, the
    largest cluster holds 29.0 % (TESSERA) and 38.5 % (Google) of the pixels.

    Also measured on 2026-09-29: on the Pampas TESSERA raster at k = 5, the
    error at seeds 42, 7 and 0 varied by 0.19 % in 1.0.1, against 3.6 % in
    1.0.0. On ten samples of the Pampas and India rasters, the silhouette
    selection chose k = 5, as in 1.0.0. Rasters with 100,000 valid pixels or
    fewer can change too, because `n_init` goes from 5 to 10. The restarts
    cost little: the final fit took about 0.2 s (`MiniBatchKMeans`:
    0.08-0.19 s) and the silhouette selection about 1.7 s (1.0.0: 1.15 s),
    while clustering the whole Pampas TESSERA raster took 97-108 s.
    scikit-learn computes the k-means sums in parallel, so a different number
    of CPU (OpenMP) threads can move a small share of pixels to another
    cluster (at most 0.23 % of the fit sample in these tests; see
    [Seeds](reproducibility.md#seeds)). To see how much a result depends on
    the sample, re-run with other values of `seed` (with `n_clusters="auto"`
    a new seed can also change k).

    Lower error is not better fields everywhere. In the 1.0.1 run of example
    15 on 2026-09-29 (seed 42, `KMeans(n_init=10)`, k = 5 with silhouette
    0.280), the TESSERA crop-filter layer has 18.2 % of its area in polygons
    over 200 ha (1.0.0: 38.4 %) and 1 polygon over 500 ha (1.0.0: 10); the
    largest is 568 ha (1.0.0: 1,447 ha; EPSG:6933). But one cluster, with
    29.0 % of the pixels and a mean October 2024 Sentinel-2 NDVI of 0.563
    (the greenest cluster has 0.827, the three others 0.26-0.28), forms one
    4-connected region of 21,452 ha in the cluster raster. The next largest
    region is 974 ha, and in 1.0.0 no region was larger than 3,321 ha. The
    default study-area rule (`aoi_selection="representative_point"`, see
    [Study-area selection](configuration.md#study-area-selection)) drops that
    region, because its representative point lies outside the study area,
    although 66 % of it (14,236 ha) is inside. Delineate-Anything outlines
    2,939 ha of fields on Sentinel-2 in that region, and 2,893 ha of them have
    no polygon in the 1.0.1 TESSERA output. The Google Satellite Embedding
    clusters of the same run have an 11,105 ha region whose representative
    point is inside the study area. It is kept as one polygon, which the crop
    filter then removes, so 1,881 of the 1,896 ha of Delineate-Anything fields
    in it have no crop-filter polygon (1.0.0: the largest Google region,
    3,266 ha, passed the crop filter as one polygon). The Google crop-filter
    layer has 52.7 % of its area in polygons over 200 ha (1.0.0: 50.9 %), so
    it shows no drop in large merged polygons.

    In the development run of 1.1 (2026-10-09), with the TESSERA store as
    converted again upstream (see
    [Satellite Sources](satellite-sources.md#embeddings)), the selection
    again chose k = 5 for TESSERA (silhouette 0.281); the crop-filter layer
    still has 18.2 % of its area in polygons over 200 ha and 1 polygon over
    500 ha (568 ha), and the large region is 21,421 ha, with about 2,857 ha
    of Delineate-Anything fields in it, 2,809 ha of them without a TESSERA
    polygon (Google: 1,880 of the 1,896 ha).

    1.0.1 does not reuse an existing output of a 1.0.0 embedding run: the
    `embedding` results version is now 2 (see
    [Output reuse](reproducibility.md#output-reuse)), so a 1.0.1 run with the
    same output path raises `FileExistsError` until you pass `overwrite=True`
    (CLI: `--overwrite`). Cluster rasters cached by 1.0.0 are computed again
    (cache version `embedding-clusters-v4`).

### Feature rasters (`source="local"`)

After 1.0.1, `engine="embedding"` also clusters a local GeoTIFF of per-pixel
features, one real-valued feature per band: for example SAR backscatter in dB,
polarimetric entropy and alpha angle, or temporal statistics. Every band is
clustered (`bands` does not select features), integer and float bands are
read as float32, and complex pixels (for example SAR SLC) are rejected:
write real-valued features first. A pixel is clustered when every band is
finite and the bands are not all zero; with a finite nodata value, a pixel
of a feature raster is skipped when **any** band equals it (embedding
pixels only when every band does, as before). The raster must be in map
geometry (features computed in radar geometry must be geocoded first).

Embedding dimensions share one scale; feature bands usually do not. Set
`engine_params["standardize"]` when their units differ:

| `standardize` | Per-band scaling before PCA and clustering |
|---|---|
| `"none"` (default) | none; the band with the largest numeric spread decides the clusters |
| `"zscore"` | mean and standard deviation of the sampled pixels |
| `"robust"` | median and interquartile range / 1.349 of the sampled pixels (less sensitive to outliers) |

The statistics come from the same seeded pixel sample that fits PCA and the
clusterer, and are recorded in `engine_meta["standardize"]` (method, centre
and scale per band; a band with zero spread keeps scale 1). A local raster
clustered without standardisation whose band standard deviations differ by
more than 10 times logs a WARNING. In a synthetic test (25 square fields of
1.44 ha on a road grid at 10 m; two feature bands in small units carry the
classes, a third band in large units is noise), the run without
standardisation scored F1 0.00 and the z-scored run 0.98: every field, plus
the road grid as one extra polygon. `"robust"` failed the same test (F1 0.00
to 0.13 over four seeds, where `"zscore"` scored 0.98 with each): the class
bands hold two values, so their interquartile range / 1.349 (0.445) is
larger than their standard deviation (0.30), and robust scaling left them
less weight than the noise band. Prefer `"zscore"`, and test `"robust"`
against a reference before you use it. Z-scoring gives every band the same
weight, so it also amplifies bands that carry only noise: leave such bands
out of the raster.

```python
import agribound

gdf = agribound.delineate(
    source="local",
    local_tif_path="sar_features.tif",  # float32, NaN nodata, map geometry
    engine="embedding",
    engine_params={"standardize": "zscore", "n_clusters": "auto"},
    output_path="fields.gpkg",
)
```

Clusters remain land-cover segments: neighbouring fields of one class that
touch become one polygon, as with embeddings. `n_clusters="auto"` (the
default) suits a few feature bands better than a k chosen for 64-128
embedding dimensions; examples 23 and 24 fix `n_clusters=20` for the
embedding sources. The LULC filter applies as for any source. A feature
raster needs a held-out test before it is used as a delineation route: in a
pilot (2026-10-08), K-means of z-scored Sentinel-1 GRD annual feature stacks
(k = 10, 20 and 30) scored F1 at most 0.10 near Davis and 0.01 in Lea County
(2024), where Delineate-Anything on the `sentinel1` composite scored 0.455
and 0.173 (see [Sentinel-1](satellite-sources.md#sentinel-1-sentinel1)).
`standardize="none"` (the default) is left out of the cluster cache
key and of `engine_meta["params"]`, so existing configurations keep their
cache entries and provenance records; an explicit `engine_params` entry
enters the configuration hash as given (see
[Output reuse](reproducibility.md#output-reuse)).

### SAM refinement inside the engine

With `sam_refine=True` the engine refines its own polygons on the
embedding raster and then needs `engine_params["sam_rgb_bands"]` (three
1-based embedding dimensions used as a pseudo-RGB image); without it the
engine raises before clustering. On a local feature raster without
`sam_rgb_bands`, SAM reads the bands that `bands` maps to R, G and B
(default 1, 2 and 3). The coverage check of SAM refinement
(`engine_params["sam_min_coverage"]`, default 0.5, added in 1.0.1) applies
here too, and cluster polygons often hold several fields. With the default,
a mask that covers less than half of its input polygon (after the overlap
trim) is not used, the polygon keeps its cluster geometry, and
`engine_meta["sam_stats"]["n_low_coverage"]` counts it (see
[SAM refinement](sam-refinement.md#masks-that-cover-too-little-of-the-polygon)).
To refine embedding polygons on optical imagery instead, call
`agribound.engines.samgeo_engine.refine_boundaries` with the optical raster
and a configuration for that source.

## SAM 2.1 automatic masks (`sam-auto`)

Segment Anything 2.1 (SAM 2.1 large; Ravi et al., 2025) run directly on the
composite. The automatic mask generator of Segment Anything (Kirillov et al.,
2023) is prompted with a regular grid of points on large overlapping tiles,
and agribound turns the masks of all tiles into one set of non-overlapping
polygons. The engine needs no field labels and is not fine-tuned. SAM
outlines every region it sees as one object, not only fields. The engine is
opt-in: it is never an engine default, and agribound's agent proposes it only
when it is named (see [Agent layer](agent.md#tools)). Example 28 runs it on
NAIP (`--no-sam-auto` drops that run; `--sam-auto-sentinel2` adds a
Sentinel-2 run), and examples 24 and 25 run it with `--sam-auto`.

```python
import agribound

fields = agribound.delineate(
    study_area="square.geojson",
    source="naip",
    year=2022,
    engine="sam-auto",  # the engine's NAIP setting; no engine_params needed
    gee_project="my-project",
)
```

Install it with `pip install "agribound[sam-auto]"` (also part of `all` and
`all-gfm`). This brings sam2 1.1.0, OpenCV and the Hugging Face hub client,
and not segment-geospatial, whose patool dependency is GPL-3. The checkpoint,
`facebook/sam2.1-hiera-large` (Apache-2.0), is 898 MB. It is downloaded on
first use at a pinned revision and checked by SHA-256; `agribound prefetch
--engine sam-auto` downloads it ahead of time. HPC tiling refuses the engine
in this version (see [HPC](hpc.md#engines-that-cannot-be-tiled)).

This engine delineates; [SAM refinement](sam-refinement.md)
(`sam_refine=True`) instead prompts SAM with the boxes of another engine's
polygons. The two do not combine: engine `sam-auto` refuses `sam_refine`
and the refinement stage's `sam_model` (its own model is
`engine_params["sam_model"]`).

### Method

1. **Image.** The R, G and B bands of the source. A raster that is not uint8
   gets one stretch for the whole scene: the 1st and 99th percentiles of each
   band. uint8 rasters, such as NAIP, are used as they are.
2. **Tiles.** Square tiles of `sam_tile_m` metres, overlapping by half. Each
   tile is resampled to SAM's 1,024 × 1,024 input.
3. **Masks.** `points_per_side`² point prompts per tile, three masks each.
   A mask is kept when its predicted IoU is above `pred_iou_thresh` and its
   stability is at least `stability_score_thresh`, and then by box
   non-maximum suppression at 0.7.
4. **Filters.** Dropped are masks cut by an interior tile edge (within
   2 px), masks under 16 px or the run's minimum area, and masks over
   2.59 km² or half a tile.
5. **Partition.** The masks of all tiles are painted into one label raster,
   highest score first. A mask keeps only the pixels no mask has claimed, and
   it is kept only if it claims at least half of its pixels. This resolves the
   duplicates from overlapping tiles and the nested masks.
6. **Polygons.** Each label becomes one polygon, its largest part. The
   pipeline's post-processing and crop filter (on by default) follow.

An object is whole in some tile only if it is at most about half a tile
across (`engine_meta["tiles"]["largest_whole_object_m"]`). Larger objects can
be cut in every tile and dropped.

### Settings per source

The four settings that depend on the source were chosen on 2026-10-12 on two
Nevada development squares. These are example 24's training squares in
Diamond Valley and Mason Valley, so the scores there are in-sample. The
choice was by mean F1 among 72 settings per source: settings within 0.01 of
the highest mean F1 tied, then the fewest SAM passes per km², then the
higher mean boundary F1. The NAIP, Sentinel-2 and Landsat settings have the
highest mean F1 of their 72. For SPOT 6/7 ten settings tied at the same
cost, and boundary F1 chose one of mean F1 0.578 (the highest was 0.587).
Values given in `engine_params` replace them. The last column is the
pre-registered evaluation below.

| Source | `sam_tile_m` | Points per side | Predicted IoU > | Stability ≥ | Prompt spacing | Status | Evaluation (paired F1 difference against Delineate-Anything v2) |
|---|---|---|---|---|---|---|---|
| `naip` | 2,048 m | 32 | 0.8 | 0.95 | 64 m | default | +0.017 (95 % interval +0.004 to +0.030) on 46 squares: better |
| `spot` | 6,144 m | 32 | 0.6 | 0.95 | 192 m | default | mean +0.256 over 4 areas: kept, no "better" claim |
| `sentinel2` | 5,120 m | 32 | 0.8 | 0.9 | 160 m | available, with a WARNING | **did not transfer**: −0.152 (−0.178 to −0.127) on 46 squares |
| `landsat` | 7,680 m | 32 | 0.6 | 0.8 | 240 m | available, with a WARNING | mean +0.126 over 4 areas: kept, no "better" claim |
| `sentinel1` | - | - | - | - | - | no default: give all four | not evaluated |
| other sources | 1,024 px | 32 | 0.8 | 0.95 | - | native scale, with a WARNING (not piloted) | not evaluated |

The Sentinel-2 setting was not changed or tried again after its result, and
its WARNING states the result and the cost. `usgs-naip-plus` shows the same
imagery as `naip`, but no setting was piloted on it, so it runs at native
scale. The other `engine_params` are shared by every source: `sam_model`
(SAM 2 or 2.1, default large), `sam_tile_overlap` 0.5, `points_per_batch` 64,
`box_nms_thresh` 0.7, `tile_edge_margin_px` 2, `min_mask_px` 16,
`max_field_area_m2` 2.59e6, `max_tile_fraction` 0.5, `partition_order`
(`"score"` or `"area_asc"`), `min_keep` 0.5 and `sam_rgb_bands`; the
pipeline's keys (`smooth_iterations`, `regularize`, `pivots`) are accepted
too. Unknown keys and `sam_refine` raise `ValueError` when the configuration
is built.

### Pre-registered evaluation

The evaluation was fixed before any `sam-auto` output existed on its areas
(PREREG-SAM, 2026-10-12). It ran on 2026-10-13 with SAM 2.1 large on an
Apple M2 Max (MPS). It compared `sam-auto` with Delineate-Anything v2 as
released, on the same composites, with each example's own configuration
and scoring: one-to-one matching at IoU ≥ 0.5. Paired 95 % intervals resample
1 km blocks (`agribound.evaluate.paired_bootstrap`, 2,000 resamples, seed 42).

- **NAIP and Sentinel-2.** The primary endpoint is the mean over the 46
  squares of example 25 without California, with a bootstrap over squares.
  California is left out because Delineate-Anything's training data hold
  California's DWR labels.
- **SPOT and Landsat.** The primary endpoint is the mean over four areas of
  examples 24, 12 and 20: Diamond Valley and Mason Valley (Nevada), and the
  Lea County and San Juan County boxes (New Mexico). As pre-registered, the
  Nevada pairs are scored without the crop filter and the New Mexico pairs
  with it, both arms of a pair alike. With all four pairs unfiltered, the
  means are +0.211 (SPOT) and +0.091 (Landsat), a secondary result.
- **The rule.** A source is kept if the mean is at least −0.05. It is called
  better only if the interval lies above 0, which is possible on the 46
  squares only.
- **Amendment.** Mason Valley's pairs had been scored against a crop-filter
  copy of its reference (176 of its 187 fields). On 2026-10-13, after the
  first scores and the endpoints had been read, both Nevada areas were
  scored again with the reference: an amendment of the pre-registration,
  recorded as a deviation from it. Diamond Valley's scores did not change.
  Mason Valley's corrected values were therefore partly known before they
  were computed; no outcome changed.

| Source | Areas | Mean F1 difference | 95 % interval | Outcome |
|---|---|---|---|---|
| NAIP | 46 squares of example 25 | +0.017 | +0.004 to +0.030 | better than Delineate-Anything v2 as released |
| Sentinel-2 | the same 46 squares | −0.152 | −0.178 to −0.127 | **did not transfer** |
| SPOT 6/7 | 4 Nevada and New Mexico areas | +0.256 | none (per area: +0.46, +0.08, +0.41, +0.06) | kept |
| Landsat (8/9; 7/8/9 on the San Juan box) | the same 4 areas | +0.126 | none (per area: +0.30, −0.04, +0.21, +0.04) | kept |

Read these with the following.

- **NAIP.** The difference is small, and both engines are low in absolute
  terms: mean F1 0.31 against 0.29 on the 46 squares. `sam-auto` scored higher
  on 32 squares and lower on 14. It drew fewer polygons (17,417 against
  23,275), with higher precision and lower recall. In New York (example 28's
  three squares, a secondary result whose first square is one of the 46; the
  evaluation read example 28's own runs, by an amendment made before any run)
  the differences were +0.02 to +0.04, each with a 1 km-block interval that
  includes 0 (the Black Dirt's lower end, −0.0015, at the edge of 0).
- **Sentinel-2.** `sam-auto` scored lower than Delineate-Anything on 45 of the
  46 squares. It drew 3,465 polygons against 13,549, and its mean recall was
  0.09 against 0.26. The setting prompts every 160 m, and 45 of the 46
  squares have a median reference field below 5 ha; whether the spacing is
  the cause was not tested, and within these squares the difference does not
  follow field size (post hoc: Spearman ρ −0.06). It was ahead in three
  Nevada and New Mexico areas with median fields of 7.7-48 ha and level in
  the fourth (1.1 ha), but well behind on Quebec's declared parcels (−0.40
  and −0.36) and in New York (−0.11 to −0.23), both secondary results.
- **SPOT.** It was evaluated only in the four Nevada and New Mexico areas,
  whose median reference fields are 1.1-48 ha. Its largest gains are where
  Delineate-Anything as released does worst on SPOT: F1 0.13 in Diamond
  Valley and 0.09 on the Lea box. In Mason Valley the 1 km-block interval
  just includes 0 (−0.003 to +0.161); its lower bound is at the edge of 0 by
  Monte Carlo error alone (20 other seeds of the same bootstrap give −0.003
  to +0.004). On the San Juan box (median field 1.1 ha) it was ahead by
  0.06, with both engines low (F1 0.23 and 0.17). On one New Mexico square of
  small fields with a winter-only SPOT composite, the pilot gave −0.13; that
  is development evidence.
- **Landsat.** At 30 m, Delineate-Anything runs outside its training range
  (0.25-10 m), and Mason Valley is negative (−0.04, interval −0.11 to +0.03).
- **Held out.** The Nevada areas lie 5.0 km (Diamond Valley) and 16.0 km
  (Mason Valley) from the squares the settings were chosen on, in the same
  basins: held out from the selection, not from the landscape.
  Delineate-Anything's scores on most areas were known before the
  evaluation, from the examples' own runs.
- **Delineate-Anything's training data.** In FBIS-73M's public lists
  (revision 9e4cdbd; its US- and Canadian-prefixed patches located by their
  own GeoTIFF headers, 2026-10-13), 128 labelled training patches intersect
  the California square, and no patch intersects any other area of the
  evaluation. Outside California, the nearest labelled training patch lies
  15.1 km from the Arizona square and 38.0 km or more from every other area,
  and unlabelled training patches lie 7.7 km from the Arizona square and
  10.5 km from the Wyoming square. Only the US- and Canadian-prefixed patches
  were located, and the public lists were published after training and
  changed on the day of the release, so they may not be exactly what the
  released weights saw. Whether SA-1B, SAM's training data, holds any of
  these images was not checked.
- **Against example 24's "-opt" runs** (a secondary comparison that decides
  nothing). `sam-auto` was scored against example 24's six NAIP and SPOT
  "-opt" runs (Delineate-Anything v2 or fine-tuned DINOv3 with the Diamond
  Valley options; see [DINOv3](#dinov3-dinov3)) on the same composites of the
  two Nevada squares. Both scores of 10 of the 12 pairs had been seen before,
  so these are reported results, not tests. Before the "-opt" runs' crop
  filter, the F1 difference ranged from −0.119 to +0.083, with one 1 km-block
  interval of six below 0 (Mason Valley NAIP against DINOv3) and none above;
  with both layers crop-filtered, from −0.086 to +0.092, with three intervals
  above 0 and one below.

References: the DRI field boundaries (Nevada, 2026-08-20), Desert Research
Institute; the OpenET agricultural field boundaries, version 2.1.1, Desert
Research Institute for the OpenET Consortium (2026), with the field
boundaries of their source agencies (on example 25's squares mostly Common
Land Units, USDA Farm Service Agency); and, in New Mexico, the DRI/NMOSE
field boundaries (V18), Desert Research Institute and New Mexico Office of
the State Engineer.

### Limits

- **Not only fields.** House lots, rangeland, roads and image seams come out
  as polygons too. On NAIP over the 46 squares, 21 % of `sam-auto`'s polygons
  overlap no reference field, against 22 % of Delineate-Anything's. The crop
  filter, on by default, removes only part of them.
- **Merged neighbours.** On SPOT, Sentinel-2 and Landsat, `sam-auto` merged
  neighbouring fields more often than Delineate-Anything in all 12 area
  pairs of Nevada and New Mexico. A field counts as merged when the polygon
  that covers most of it also covers at least a quarter of another field. In
  Diamond Valley on SPOT, 34 % of the fields were merged, against 4 %. On the
  46 squares it merged less often on average, on NAIP (41 % against 53 %)
  and on Sentinel-2 (46 % against 53 %; more often on 17 of the 46).
- **Small fields at 10 m.** See Sentinel-2 above.
- **Large objects.** At the default overlap, an object larger than about half
  a tile is cut in every tile and dropped: about 1.0 km on NAIP and
  2.5-3.7 km on SPOT, Sentinel-2 and Landsat at their settings.
- **Many tiles at 6-30 m.** The settings were chosen with one tile per 5 km
  square. In a pre-registered multi-tile development check (2026-10-12),
  three development squares were scored inside runs on 25 km composites of
  36-90 tiles. Sentinel-2's F1 changed by at most 0.02. SPOT and Landsat lost
  0.09 and 0.11 on one square (Diamond Valley, with the same images in both
  runs) and gained on the other two, which fails that check's rule (every
  change at least −0.05).
- **Devices.** These runs used Apple MPS. Output is identical on one device
  with the same versions, and masks differ between devices (MPS against CPU:
  mask IoU 0.59-0.97 for the same fields, measured in the SAM 2 refinement
  stage); no score on CUDA or CPU is reported. `engine_meta` records the
  device, and the device is not part of the configuration hash.

### Cost

The pipeline's `delineate` step includes model loading. It was measured back
to back with Delineate-Anything on the same composites of the Lea County box
(415 km²) and the San Juan County box (496 km²), on an Apple M2 Max (MPS),
with other processes running. The seconds per km² below divide by the areas
of the composites' export grids (about 430 and 526 km²), and the ratio is
the ratio of the summed seconds.

| Source | `sam-auto` s/km² | Delineate-Anything s/km² | Ratio |
|---|---|---|---|
| NAIP | 4.48 | 0.74 | 6.1 |
| SPOT 6/7 | 0.44 | 0.068 | 6.4 |
| Sentinel-2 | 0.59 | 0.035 | 17.0 |
| Landsat | 0.23 | 0.011 | 20.7 |

On the 5 km squares of example 25, against Delineate-Anything's runs in other
sessions, the ratio was 3.1 on NAIP and 2.1 on Sentinel-2.

### Provenance and licences

The output has one polygon per kept mask, with `sam_label`, `sam_iou` and
`sam_stability`. `gdf.attrs["engine_meta"]` records the following:

- the model and checkpoint (revision and SHA-256);
- the sam2 distribution and its origin;
- the torch, torchvision and OpenCV versions;
- the device;
- every setting and where it came from;
- the tile grid, the stretch, the counts of each step and the seconds of each
  stage.

SAM 2.1's code and weights are Meta's, under Apache-2.0. The `sam2` 1.1.0
package on PyPI is a third-party build of Meta's code: its 62 code,
configuration and licence files are byte-identical to Meta's commit 2b90b9f
(2024-12-15). The `sam-auto` extra pins it, because the engine calls
predictor internals checked with that version only.

## Crop-sequence zones (`crop-sequence`)

Zones of uniform multi-year crop history from a stack of annual crop-type
maps, made the way USDA NASS makes its Crop Sequence Boundaries (CSB; Hunt et
al. 2024). The input is not imagery: the `cdl-sequence` (USDA Cropland Data
Layer, conterminous US) and `aci-sequence` (AAFC Annual Crop Inventory,
Canada) sources download one band of class codes per year (see
[Crop-type map sequences](satellite-sources.md#crop-type-map-sequences-cdl-sequence-aci-sequence)),
and `source="local"` reads a stack of another series. The output is
crop-history zones, not field instances: `engine_meta["field_instances"]` is
False. The engine runs on the CPU and needs no model weights: 1,375 km² (a
mosaic of 6 × 6 copies of the pilot's most fragmented stack, the California
square) took 9.6 s of CPU on an Apple M2 Max (2026-10-10).

```python
import agribound

zones = agribound.delineate(
    study_area="square.geojson",
    source="cdl-sequence",  # or "aci-sequence" in Canada
    year=2023,  # the last map year
    crop_sequence_years=8,  # CDL 2016-2023 (the default window length)
    engine="crop-sequence",
    lulc_filter=False,  # the engine applies its own cultivated mask
    gee_project="my-project",
)
```

Install with `pip install "agribound[crop-sequence]"` (scikit-image and the
Earth Engine extra).

### Method

`engine_params["method"]="zones"` (the only method) is deterministic:

1. **Cultivated pixels**: in an agricultural class in more than half of the
   years (`min_crop_years="auto"`: `n // 2 + 1`, 5 of 8).
2. **Barriers**: pixels in a barrier class (developed land) in any year.
3. **Per-year focal mode**: the most frequent code in a disk of
   `smooth_radius_px` pixels (3, that is 90 m at 30 m), counting only
   cultivated pixels that have a code that year; ties go to the lowest code.
4. **Regions**: 8-connected regions of cultivated, non-barrier pixels whose
   smoothed sequences are equal. Codes are compared for equality only; no
   distance or mean of codes is taken, since codes are nominal.
5. **Merge**: regions under `min_region_m2` (1 ha: at most 11 pixels at 30 m)
   are merged, smallest first, into the 4-adjacent region with the most
   pixels (ties: the lowest region id, in raster-scan order); sizes are
   updated after each merge, and small regions without a neighbour are kept.
6. **Polygons**: each zone is traced (8-connected) and split into its parts,
   parts under `min_field_area_m2` are dropped, and the outlines are
   densified at one pixel, so that the pipeline's smoothing rounds only pixel
   corners.

| `engine_params` | Default | Evidence |
|---|---|---|
| `classes` | CDL: example 25's farmed classes, 1-61, 66-77, 176 and 204-254 (125 codes, with 37 Other Hay/Non Alfalfa and 176 Grass/Pasture, which also holds rangeland). ACI: the legend codes 120-199 (61 codes) | The only CDL list measured. The ACI list is the set of codes 120-199 in the 2009-2023 legends, plus 121 (Cropland) and 159 (Other Oilseeds), new in 2024 |
| `barrier_classes` | CDL 121-124 (developed); ACI 34 (Urban and Developed) | The crop-sequence pilot's setting |
| `min_crop_years` | `"auto"` | Equals Earth Engine's `mode` of the yearly agricultural indicator as the pilot ran it (4-of-8 ties are not cultivated) |
| `smooth_radius_px` | 3 | With ties to the lowest code, equals Earth Engine's `focalMode(3, "circle")` at every cultivated pixel and year of the pilot's 11 squares (radius 2 or 4: 83-98 % agreement) |
| `min_region_m2` | 10,000 (1 ha) | CSB's minimum of 1 ha |
| `merge_rule` | `"largest"` | The CSB release wording. The CSB code merges into the neighbour with the longest shared border (not measured); GDAL's sieve filter left 1.07-5.17 times the pilot's regions |

Other keys: `dataset` (`"cdl"` or `"aci"`: the default class lists for a
local stack), `years` (local stacks: the year of each band; default the
`AGRIBOUND_YEARS` tag, else the one year in each band description),
`value_scale` (`"categorical"`, required for `source="local"`) and
`attributes` (default True). `smooth_radius_px` may not exceed the stack's
pad (10 px). Unknown keys raise `ValueError`. A local stack needs
`engine_params={"value_scale": "categorical", "dataset": "cdl"}` (or explicit
`classes` and `barrier_classes`). With a study area, the local builder crops
the raster for this engine to the study area's bounding box plus 10 pixels
on every side, as far as the raster reaches, so focal windows of up to
10 pixels are complete at the edge. On the pilot's Iowa square, the
`cdl-sequence` stack run as `source="local"` with the square as study area
gave the zones of the `cdl-sequence` run (0 m² symmetric difference,
2026-10-10).

Output columns: `agribound:zone_id` (one zone can give several polygon
parts), `agribound:zone_px`, `agribound:n_zones_merged` and, per year,
`agribound:crop_<year>` (the zone's most frequent smoothed code; 0 = none).
`engine_meta` records the parameters, class lists, years, the counts of valid
pixels, distinct sequences, regions before and after the merge, merges and
small regions kept, the share of valid pixels whose sequence differs from all
eight neighbours (`unique_sequence_share_3x3`, smoothed and raw) and the
cached label raster. The label raster is cached, so a change of
`min_field_area_m2` or of the post-processing reuses it.

The other engines refuse rasters whose `AGRIBOUND_VALUE_SCALE` tag is
`categorical`, and so do SAM refinement, fine-tuning and ensembles; the
reflectance conversions (which take arrays) refuse the two sources and
`value_scale="categorical"`. `date_range`, `sam_refine`, batch exports and an
`export_crs` that is geographic or not in metres are refused for the two
sources.

### What the zones are and are not

- **Zones of uniform crop history.** Neighbouring fields with the same crop
  sequence fall into one zone. In the pilot's six CLU squares, in 18-67 % of
  the pairs of adjacent reference fields both fields had the same eight-year
  CDL sequence (Iowa 67 %, Virginia 52 %); in example 27's two Quebec squares,
  13 % and 23 % of the pairs of adjacent declared parcels had the same ACI
  class in every year.
- **Narrow and small fields are lost.** The focal mode has a 90 m radius and
  regions under 1 ha are merged, so fields narrower than about 60-90 m or
  smaller than 1 ha are absorbed. In the pilot, 35 % of the US reference
  fields were 0.25-1 ha, and the zones recalled 0.1 % of them.
- **Crop-map errors pass into the zones**: hay and pasture classes (CDL 37
  and 176) that alternate between years, tree-crop classes that change from
  year to year (California), and the maps' own errors.
- **Grid.** The zones depend on the grid: on a UTM and an EPSG:5070 grid
  their scores differed by at most 0.02, but their polygons agreed at F1
  0.23-0.72. Reproducing a run needs the same `export_crs`.
- **Not measured**: large-field landscapes (the Prairies, the Corn Belt),
  other crop-map series, and other windows than 2016-2023 (2017-2024 changed
  the pilot's F1 by −0.020 to +0.031).

### Measured limits

The ten US squares of the crop-sequence pilot are example 25's squares (5 km,
the densest farmed cell of each state); Quebec is example 27's
Bas-Saint-Laurent square. Zones: this engine through `agribound.delineate`
(2026-10-10; 2016-2023, default settings and post-processing,
`lulc_filter=False`). CSB 2016-2023 (the published release) and
Delineate-Anything v2 on a Sentinel-2 2024 composite with agribound's
defaults: the pilot (2026-10-09, research code). One-to-one matching at IoU
≥ 0.5 against every reference field (representative point inside the
square, at least 2,500 m²); BF60: boundary F at 60 m.

| Square | Reference (fields, median) | Zones F1 | Zones BF60 | CSB F1 | Delineate-Anything F1 |
|---|---|---|---|---|---|
| LA | CLU (1,132; 1.24 ha) | 0.054 | 0.886 | 0.068 | 0.225 |
| AZ | LCRVPD 95 % (832; 1.15 ha) | 0.070 | 0.884 | 0.113 | 0.264 |
| VA | CLU (936; 1.48 ha) | 0.009 | 0.682 | 0.007 | 0.022 |
| CA | CADWR 2023 provisional (796; 1.63 ha) | 0.074 | 0.894 | 0.090 | 0.649 |
| MT | CLU (684; 1.92 ha) | 0.004 | 0.740 | 0.002 | 0.011 |
| FL | FLSAID (231; 2.66 ha) | 0.057 | 0.773 | 0.113 | 0.364 |
| KY | CLU (444; 2.86 ha) | 0.078 | 0.867 | 0.111 | 0.330 |
| WY | WYSWP (152; 5.18 ha) | 0.021 | 0.650 | 0.033 | 0.151 |
| WI | CLU (870; 0.91 ha) | 0.021 | 0.868 | 0.016 | 0.075 |
| IA | CLU (669; 1.07 ha) | 0.052 | 0.639 | 0.066 | 0.102 |
| Quebec (ACI) | BDPPAD 2023 (526; 2.33 ha) | 0.199 | 0.939 | - | 0.315 |

Read these numbers with the following:

- Imagery is ahead at object level on all ten US squares, and the zones are
  about as good as the published CSB (CSB's F1 is the higher on seven of the
  ten, by a median of 0.014).
- The boundary F at 60 m does not separate the methods on these squares:
  lines drawn without regard to the fields reach a boundary precision of
  0.967 or more at 60 m on seven of the ten (see
  [Chance baselines](evaluation.md#chance-baselines)), and the pilot's blind
  grid reached at least CSB's boundary F on six. Report object scores and
  the baselines with it.
- CLU outlines are farm-programme units, and several references map
  irrigated land only, so a zone can legitimately split or merge them.
- With `smooth_iterations=0` and `simplify_tolerance=0` (the engine's pixel
  outlines) instead of the default post-processing, F1 changed by −0.005 to
  +0.001 and BF60 rose by 0.002 to 0.011.

Example 27 scores the zones against the parcels declared to La Financière
agricole du Québec for 2023 in two squares, beside Delineate-Anything v2 on a
Sentinel-2 composite of April-October 2023 (agribound 1.1.0, 2026-10-11):

| Square | Declared parcels | Zones F1 | Delineate-Anything F1 | Blind grids F1 | BF60 of all layers |
|---|---|---|---|---|---|
| Bas-Saint-Laurent (hay, cereals) | 490 | 0.197 | 0.486 | 0.017-0.018 | 0.88-0.93 |
| Montérégie (soybeans, corn) | 443 | 0.119 | 0.472 | 0.007 | 0.85-0.99 |

There the zones merged 66 % and 78 % of the parcels with another and split
about half, and the reference shifted by half its median width had a boundary
F at 60 m of 0.92 and 0.99.

### Crop history of other outlines

`agribound.attach_crop_history(fields, stack)` adds to any polygons, such as
Delineate-Anything outlines from imagery or a reference layer, the most
frequent class of each year of a crop-type map stack: `agribound:crop_<year>`
(ties go to the lowest code; 0 = no class), `agribound:crop_px` (the pixels
counted) and, with `shares=True`, `agribound:crop_share_<year>`. A pixel is
counted when its centre lies inside the polygon (`all_touched=True` counts
every pixel the polygon touches); a polygon without a pixel centre takes the
pixel under its representative point. The outlines and the CRS are not
changed, and `attrs["crop_history"]` holds the years, the stack's legend and
its attribution. The codes are read raw (not smoothed), the way NASS
attributes CSB.

```python
from agribound import AgriboundConfig, attach_crop_history, build_composite

stack = build_composite(
    AgriboundConfig(
        study_area="square.geojson",
        source="cdl-sequence",
        year=2023,
        engine="crop-sequence",
        lulc_filter=False,
        gee_project="my-project",
    )
)
fields = attach_crop_history(fields, stack)  # CDL 2016-2023 per field
legend = fields.attrs["crop_history"]["legend"]  # code -> class name
```

### Published Crop Sequence Boundaries (`query_csb`)

`agribound.query_csb(study_area, year)` reads the published CSB polygons of a
study area: the national release (rev23, public domain) whose eight-year
window ends in `year` (2015-2025, i.e. 2008-2015 to 2018-2025), with its
attributes (`CSBID`, `CSBACRES`, `CDL<year>`, state and county codes). It
reads the release zip on the NASS web site in place with GDAL
(`/vsizip//vsicurl/`), without storing it; the geodatabase in the zip is
compressed, so a query reads the zip up to the records it needs (195 s for a
5 km square in Iowa and 462 s for a 1 km box in Arizona, 2016-2023,
2026-10-10). For repeated queries, download the zip once
(`agribound.csb_query.csb_release(year)["url"]`) and pass it as `source`.
`clip=True` cuts the polygons to the study area; `output_path` also writes
them with a provenance record. CSB polygons are crop-history zones, at least
about 1 ha, and adjacent fields with the same crop sequence form one polygon,
so CSB is not a field-boundary reference.

### Large areas

On HPC tiles (see [HPC and large areas](hpc.md)), each tile builds its own
stack and zones. In the pilot, zones reached 4,080 m across inside a 5 km
square, more than the default halo of 1,000 m, so set `halo_m` to 5,000 m or
more for these sources.

Thanks to Jacob T. Stid (Michigan State University) and Jeremy Rapp, who
shared the crop-sequence recipe that this route follows.

## Ensemble (`ensemble`)

Runs several engines, or one engine with different models, on the same
composite and combines them. Members are given in
`engine_params["engines"]` as names or dicts
(`{"engine": ..., "engine_params": {...}, "label": ...}`); the default
members are `delineate-anything` and `ftw`. Every member, including the
defaults, is checked against the source when the configuration is validated
(so `engine="ensemble"` on `naip`, `sentinel1`, `umbra` or `capella` without
explicit members fails at once, listing members that support the source).

| `merge_strategy` | Rule |
|---|---|
| `"intersection"` (default) | Successive overlay intersections: the areas every member covers. Small slivers can appear where boundaries disagree; the area filter removes those below `min_field_area_m2`. |
| `"union"` | All polygons pooled; duplicates (IoU ≥ `union_iou_threshold` 0.3 or containment ≥ `union_containment_threshold` 0.8) fused. |
| `"vote"` | Members' polygons rasterised on the input grid; a pixel is kept when at least `min_votes` members cover it; kept pixels are polygonised. |

Vote rule: members that returned no polygons are left out (WARNING,
`vote_stats["empty_members"]`); for the `n` remaining members
`min_votes = max(min(2, n), ceil(vote_threshold × n))` (default
`vote_threshold=0.5`), i.e. at least two members must agree whenever two or
more have polygons, as in agribound 0.1.x. `engine_params["min_votes"]` sets it
directly. Adjacent fields that are both kept merge where they touch on the
pixel grid.

Output columns: `engine_count`; `ensemble:members` (intersection, union),
`ensemble:n_members` (union); `vote_count` (the **maximum** number of
agreeing members inside the polygon; in 0.1.x this column held the constant
`min_votes`), `vote_count_mean` and `min_votes` (vote).

Other parameters: `vote_resolution`, `on_member_error` (`"raise"` default,
or `"skip"`), `isolate_member_caches` (default True: each member caches in
its own sub-directory). Members receive only the `engine_params` of their own
spec and run with `sam_refine=False`; the pipeline refines the ensemble output.
The centre-pivot step (`engine_params["pivots"]`, see
[Configuration](configuration.md#post-processing)) is set in the ensemble's
own `engine_params` and runs on its output; in a member's `engine_params` it
raises `ValueError`. A `sam-auto` member's `engine_params` and a DINOv3
member's `dilate_interior_px` are checked when the configuration is built.
The ensemble cannot be fine-tuned; fine-tune each member in its own run and
pass its checkpoint in the member spec.

---

## SAM refinement

Box-prompted SAM refinement (`sam_refine=True`) is a separate stage that runs
after any engine except `embedding` (which refines its own polygons),
`crop-sequence` and `sam-auto` (which refuse it); see
[SAM refinement](sam-refinement.md).

## References

- Lavreniuk, M., et al. (2025). Delineate Anything: Resolution-agnostic field
  boundary delineation on satellite imagery. ECAI 2025. arXiv:2504.02534.
- Lavreniuk, M., et al. (2026). Delineate Anything v2: A global foundation
  model for field delineation. ECCV 2026 Workshops (ECCVW), GAIA workshop.
  arXiv:2607.19069.
- Kerner, H., et al. (2025). Fields of The World. *AAAI* 39(27), 28151-28159.
  <https://doi.org/10.1609/aaai.v39i27.35034>
- Muhawenayo, G., et al. (2026). PRUE: A practical recipe for field boundary
  segmentation at scale. arXiv:2603.27101 (the `FTW_PRUE_*` models).
- Wu, Q. (2026). GeoAI. *JOSS* 11(118), 9605.
  <https://doi.org/10.21105/joss.09605>
- He, K., et al. (2017). Mask R-CNN. ICCV, 2980-2988.
  <https://doi.org/10.1109/ICCV.2017.322>
- Siméoni, O., et al. (2025). DINOv3. arXiv:2508.10104.
- Kirillov, A., et al. (2023). Segment anything. ICCV, 3992-4003.
  <https://doi.org/10.1109/ICCV51070.2023.00371>
- Ravi, N., et al. (2025). SAM 2: Segment anything in images and videos.
  ICLR 2025. arXiv:2408.00714.
- Szwarcman, D., et al. (2026). Prithvi-EO-2.0. *IEEE TGRS* 64, 1-20.
  <https://doi.org/10.1109/TGRS.2025.3642610>
- Feng, Z., et al. (2026). TESSERA. CVPR 2026. arXiv:2506.20380.
- Brown, C. F., et al. (2025). AlphaEarth Foundations. arXiv:2507.22291.
- Hunt, K. A., Abernethy, J., Beeson, P. C., Bowman, M., Wallander, S., &
  Williams, R. (2024). Crop sequence boundaries using USDA National
  Agricultural Statistics Service historic cropland data layers. *Statistical
  Journal of the IAOS* 40, 237-246. <https://doi.org/10.3233/SJI-230078>
- Beeson, P. C., Daughtry, C. S. T., & Wallander, S. A. (2020). Estimates of
  conservation tillage practices using Landsat archive. *Remote Sensing*
  12(16), 2665. <https://doi.org/10.3390/rs12162665>

Full citations: [Citation & References](../citation.md).
