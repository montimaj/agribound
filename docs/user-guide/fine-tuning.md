# Fine-Tuning

With `fine_tune=True` and `reference_boundaries`, the pipeline trains the
engine on the reference polygons over the composite, then runs inference with
the new checkpoint (passed to the engine as `engine_params["checkpoint_path"]`
and recorded in the provenance record as `facts.fine_tuned_checkpoint`).

```python
gdf = agribound.delineate(
    study_area="area.geojson",
    source="naip",
    year=2022,
    engine="dinov3",
    reference_boundaries="reference_fields.gpkg",
    fine_tune=True,
    fine_tune_epochs=20,
    gee_project="my-project",
)
```

```bash
agribound delineate --study-area area.geojson --source naip --year 2022 --engine dinov3 \
    --reference reference_fields.gpkg --fine-tune --fine-tune-epochs 20 --gee-project my-project
```

## Which engines

| Engine | Fine-tunable | What is trained |
|---|---|---|
| `delineate-anything` | yes | Ultralytics YOLO11-seg from the selected Delineate-Anything weights (`da_model`, default `large_v2`) |
| `geoai` | yes (required: no published field weights) | Mask R-CNN ResNet50-FPN from the COCO weights |
| `dinov3` | yes (required: no published field weights) | DINOv3 + DPT; full fine-tuning by default, LoRA or decoder-only optional |
| `prithvi` | yes | Prithvi-EO-2.0 + UPerNet (terratorch), full fine-tuning by default, LoRA optional |
| `ftw` | **no** | Train with ftw-baselines (`ftw model fit -c <config.yaml>`) and pass `engine_params={"checkpoint_path": ...}` |
| `embedding` | no | No trainable weights; the reference is used for evaluation only |
| `ensemble` | no | Fine-tune each member in its own run and pass each member its checkpoint |

`fine_tune=True` with a non-fine-tunable engine raises `ValueError` with these
instructions; the engine is never replaced by another one. When
`fine_tune=True` the output is not evaluated against the same reference
(train on one area, evaluate on another).

## Training data

`agribound.engines.finetune._data` cuts chips and masks from the composite:

- RGB engines (Delineate-Anything, GeoAI, DINOv3) read the canonical R, G, B
  bands with the same scene-level 1-99 percentile stretch as at inference
  (uint8 rasters unchanged). Prithvi reads Blue, Green, Red, narrow NIR,
  SWIR 1, SWIR 2 as reflectance × 10000.
- Masks: 0 background, 1 field interior, 2 field boundary (reference pixels
  within `boundary_erosion` pixels of another polygon or background), 255
  where the image has no valid data. Instance masks give each reference
  polygon its own id, so touching fields stay separate.
- **The reference must be complete inside the chips that contain its
  polygons**: pixels outside every reference polygon are labelled background.

`engine_params` for the chips:

| Parameter | Default | Meaning |
|---|---|---|
| `chip_size` | 224 (Prithvi), 256 (DINOv3), 512 below 4 m GSD else 256 (Delineate-Anything); GeoAI: sized from the reference fields (1.25 × the 90th percentile of the fields' longer bounding-box sides, in pixels, rounded up to a multiple of 32 and clamped to 256–1024 px) | Chip size in pixels. GeoAI infers on windows of the chip size, and no window sees the whole of a field larger than a chip. The engine rejoins such a field's pieces only where they meet at a window edge (`merge_window_seams`, see [Engines](engines.md#geoai-geoai)), so agribound logs a WARNING when more than 10 % of the reference fields are larger than the chip. |
| `boundary_erosion` | 2 | Boundary width in pixels. |
| `min_label_fraction` | 0.01 | Minimum fraction of chip pixels inside reference polygons. |
| `min_valid_fraction` | 0.5 | Minimum fraction of valid image pixels. |
| `value_scale` | - | Required for Prithvi on `local` rasters. |

## Train/validation split

`assign_splits` is used by every trainer and is seeded from `config.seed`
(`agribound._repro.get_rng`):

- `fine_tune_split="block"` (default): chips are grouped into square blocks
  of `fine_tune_block_size_m` (default 5000 m) in an equal-area CRS, and whole
  blocks go to validation until about `fine_tune_val_split` (default 0.2) of
  the chips; this reduces the spatial leakage of a random chip split.
- `"random"`: a seeded permutation of the chips.
- `"column"`: groups from `fine_tune_split_column`, a column of the reference
  layer (majority value among the reference polygons in each chip).

At least one training and one validation chip are guaranteed when there are
two or more chips; otherwise a `ValueError` is raised.

## Engine-specific settings

**Delineate-Anything.** Labels are the reference polygons clipped to each
chip. Chips are upsampled by the same super-resolution factor as at inference
(2 at 4 m GSD or coarser), so `imgsz` is 512 when chip size × factor is 512
(a WARNING is logged and `imgsz_matches_model_input: false` recorded
otherwise). Settings: `seed=config.seed`, `deterministic=True`, `mosaic=0`,
AdamW with `lr0` 0.002 (`yolo_lr0`), flips, batch 16 (`yolo_batch`),
`epochs=fine_tune_epochs`; Ultralytics keeps `best.pt`. The 1.0.0 test suite
runs this trainer against a stub of Ultralytics. Example 12's NAIP runs
fine-tuned `large_v2` with Ultralytics 8.4.163 (650 chips of 512 px, 10
epochs); their in-sample scores, and those of the GeoAI and DINOv3 models
fine-tuned on the same reference, are in the [gallery](../gallery.md).

**GeoAI.** geoai's Mask R-CNN recipe run on agribound's own train/validation
chips (geoai's own trainer would re-split the chips randomly): SGD (lr
`learning_rate` 0.005, momentum 0.9, weight decay 5e-4), `StepLR`, batch 4,
the checkpoint with the best validation mask IoU (`best_model.pth`), early
stopping after `early_stopping_patience` (10) epochs. The chip size is saved
next to the checkpoint and becomes the default inference window. On Apple MPS
the model trains on CPU (WARNING), as at inference.
A WARNING is logged when the best validation IoU is below 0.1 (the model has
probably not learned the field boundaries, and its output may be artefacts)
or when there are fewer than 10 training chips (treat the model as a smoke
test). The warnings are also written to the `warnings` list of
`best_model.pth.agribound.json` and to the run's provenance record. They are
a floor for "learned something", not a quality target: a Namoi test run with
one training and one validation chip peaked at a validation IoU of 0.033 and
delineated a regular lattice of ovals.

**DINOv3.** `geoai.dinov3_finetune.train_dinov3_segmentation`: cross-entropy
with `ignore_index=255`, AdamW (`learning_rate` 1e-4, `weight_decay` 1e-4)
with cosine annealing, batch 4, early stopping on `val_loss`
(`early_stopping_patience` 10), best checkpoint by `val_loss`. Full
fine-tuning by default (`use_lora=False`, `freeze_backbone=False`; about 303 M
backbone parameters for ViT-L/16); `use_lora=True` (rank `lora_rank`, default
4; about 0.39 M adapter parameters) or `freeze_backbone=True` (decoder only).
`trainer_kwargs` is passed to the Lightning trainer. The trainable-parameter
counts of each run are written to `<checkpoint>.agribound.json`.
The returned checkpoint is the best one by `val_loss`. geoai-py also saves a
`last.ckpt`, which agribound deletes once the best checkpoint is known.
Checkpoints hold the optimizer state: a ViT-L/16 full fine-tuning checkpoint
is about 3.7 GB (the ~303 M float32 backbone weights plus AdamW's two moment
buffers). While training runs, both checkpoints exist, so the cache directory
needs about twice that in free space.

**Prithvi.** terratorch `SemanticSegmentationTask` with a Prithvi-EO-2.0
backbone (`model_name`, default `Prithvi-EO-2.0-300M-TL`), a UPerNet decoder
and three classes; AdamW (`learning_rate` 1e-4), batch 8, best checkpoint by
`val/loss`. `use_lora=True` adds LoRA on the query and value projections
(`lora_rank` default 16); `use_lora` together with `freeze_backbone` raises
`ValueError` (it would freeze the adapters too). `early_stopping_patience`
(no early stopping by default) adds a Lightning `EarlyStopping` callback on
`val/loss`, and `trainer_kwargs` is passed to the Lightning trainer. Needs the GFM environment. On
Apple MPS the default 224 px chips run on CPU; `chip_size=192` (and
`tile_size=192` at inference) runs on MPS.

## Caching

Each fine-tuning run gets its own cache directory keyed by the study area,
source, year/date range, compositing settings, engine, base model, a
fingerprint of the reference file (path, modification time, size), epochs,
split settings, seed, `bands`, the training-related `engine_params` (all but
`checkpoint_path` and the `sam_*` keys) and the engine's default chip-size
rule, so a changed default does not reuse a checkpoint trained on chips of
another size. A second call with the same inputs returns the cached
checkpoint without retraining (`finetune_manifest.json`).

## Large areas

`agribound tiles make` refuses `fine_tune: true` unless
`--allow-fine-tune-per-tile` is given (it would train one model per tile). Fine-tune once over the reference area and pass the checkpoint to the
tiles with `--engine-param checkpoint_path=<path>`; see
[HPC](hpc.md#fine-tuned-models).
