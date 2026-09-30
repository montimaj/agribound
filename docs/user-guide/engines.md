# Delineation Engines

An engine turns the stage-A raster into field polygons. Seven engines are
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
| Embedding clustering | `embedding` | K-means (or spectral) clustering of pre-computed embeddings | yes | no | no (CPU) | `google-embedding`, `tessera-embedding` | `embedding` |
| Ensemble | `ensemble` | intersection, union or pixel vote of several engines/models | depends on the members | no | yes | depends on the members | members' extras |

"All imagery sources" = `landsat`, `sentinel2`, `hls`, `naip`,
`usgs-naip-plus`, `spot`, `spot-pan`, `local`.

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
  order for Ultralytics, FP16 on GPU/MPS. Detections from all tiles are
  combined at polygon level: tile-cut pieces of one field are merged
  (`merge_tile_pieces`, default True, so fields larger than a tile are rebuilt),
  then greedy non-maximum suppression and overlap resolution. Results are close
  to, but not identical with, the `reference` backend. Returns a `confidence`
  column.
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
(True), `resolve_overlaps` (True), `min_hole_area_m2` (reference backend,
2500 m²). A parameter that the selected backend cannot honour raises
`ValueError`.

!!! warning "Changed in 1.0.0"
    The confidence parameter is `conf_threshold`; the old names
    `confidence` and `minimal_confidence` now raise `ValueError`.

`min_field_area_m2` is applied as an absolute area computed in the equal-area
EPSG:6933 by every backend. For rasters whose GSD lies more than 5 % outside
the 0.25-10 m training range (for example 30 m Landsat or HLS) a WARNING is
logged and `engine_meta["gsd_outside_training_range"]` is True. Example 20
runs Delineate Anything v2 as released on 2018 composites of San Juan County,
New Mexico. Against the NMOSE polygons, which were not used for training or
fine-tuning in these runs (whether the model's training set, FBIS-73M,
includes them was not checked), F1 was 0.15 on Landsat (30 m), 0.34 on
Sentinel-2 (10 m), 0.33 on SPOT 6/7 (6 m) and 0.43 on NAIP (1 m); see the
[gallery](../gallery.md).

The Delineate-Anything model code and weights, and Ultralytics, are AGPL-3.0.

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
`n_seam_gap_pixels_filled`. In example 12's NAIP run (centre pivots of
about 800 m at 1 m), F1 against NMOSE was 0.01 with 256 px chips, 0.25 with
1,024 px chips and 0.48 after joining (in-sample; see the
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
  frozen backbone; `freeze_backbone=True` alone trains the decoder only.
- Input: canonical R, G, B with a scene-level percentile stretch, as float
  `uint8 / 255`. geoai applies no mean/standard-deviation normalisation, so
  the backbone does not see the SAT-493M pre-training normalisation; this
  matters most with a frozen backbone.
- Inference window: the training chip size (a multiple of 16), capped at the
  raster size, with the input mirror-padded so that no window is zero-padded.
- Each field-interior region is grown back over the predicted boundary class
  by the training boundary width, so neighbouring polygons never overlap;
  right-angled convex corners lose k(k+1)/2 pixels (3 px at the default
  k = 2).

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

    1.0.1 does not reuse an existing output of a 1.0.0 embedding run: the
    `embedding` results version is now 2 (see
    [Output reuse](reproducibility.md#output-reuse)), so a 1.0.1 run with the
    same output path raises `FileExistsError` until you pass `overwrite=True`
    (CLI: `--overwrite`). Cluster rasters cached by 1.0.0 are computed again
    (cache version `embedding-clusters-v4`).

`engine="embedding"` accepts only the embedding sources; `source="local"` is
rejected. With `sam_refine=True` the engine refines its own polygons on the
embedding raster and then needs `engine_params["sam_rgb_bands"]` (three
1-based embedding dimensions used as a pseudo-RGB image); without it the
engine raises before clustering. The coverage check of SAM refinement
(`engine_params["sam_min_coverage"]`, default 0.5, added in 1.0.1) applies
here too, and cluster polygons often hold several fields. With the default,
a mask that covers less than half of its input polygon (after the overlap
trim) is not used, the polygon keeps its cluster geometry, and
`engine_meta["sam_stats"]["n_low_coverage"]` counts it (see
[SAM refinement](sam-refinement.md#masks-that-cover-too-little-of-the-polygon)).
To refine embedding polygons on optical imagery instead, call
`agribound.engines.samgeo_engine.refine_boundaries` with the optical raster
and a configuration for that source.

## Ensemble (`ensemble`)

Runs several engines, or one engine with different models, on the same
composite and combines them. Members are given in
`engine_params["engines"]` as names or dicts
(`{"engine": ..., "engine_params": {...}, "label": ...}`); the default
members are `delineate-anything` and `ftw`. Every member, including the
defaults, is checked against the source when the configuration is validated
(so `engine="ensemble"` on `naip` without explicit members fails at once,
listing members that support the source).

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
The ensemble cannot be fine-tuned; fine-tune each member in its own run and
pass its checkpoint in the member spec.

---

## SAM refinement

Box-prompted SAM refinement (`sam_refine=True`) is a separate stage that runs
after any engine except `embedding`; see [SAM refinement](sam-refinement.md).

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
- Szwarcman, D., et al. (2026). Prithvi-EO-2.0. *IEEE TGRS* 64, 1-20.
  <https://doi.org/10.1109/TGRS.2025.3642610>
- Feng, Z., et al. (2026). TESSERA. CVPR 2026. arXiv:2506.20380.
- Brown, C. F., et al. (2025). AlphaEarth Foundations. arXiv:2507.22291.

Full citations: [Citation & References](../citation.md).
