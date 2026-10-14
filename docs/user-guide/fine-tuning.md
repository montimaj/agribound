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
  (uint8 rasters unchanged). Delineate-Anything chips follow its
  `stretch_percentiles` and `processing_gsd_m`, as its native backend does at
  inference: they are cut from the raster resampled to `processing_gsd_m`
  (see [Fields larger than the tiles](engines.md#fields-larger-than-the-tiles)),
  whose pixel size then sets the default chip size, and stretched between
  those percentiles. Prithvi reads Blue, Green, Red, narrow NIR, SWIR 1,
  SWIR 2 as reflectance × 10000.
- Masks: 0 background, 1 field interior, 2 field boundary (reference pixels
  within `boundary_erosion` pixels of another polygon or background), 255
  where the image has no valid data. Instance masks give each reference
  polygon its own id, so touching fields stay separate.
- A pixel has no valid data when a band is not finite there, or when every
  band equals the raster's nodata value. Delineate-Anything chips also treat
  a pixel that is 0 in every band as nodata, as its native backend does at
  inference, and their image is 0 at every pixel without valid data, as in
  the backend's tiles, so the chips and the inference tiles have the same
  valid pixels (agribound 1.0.1 kept pixels that are 0 in every band valid in
  the chips).
- Chips are `chip_size` windows every `chip_stride` pixels. Delineate-Anything
  chips overlap by default, half a chip apart, as its inference tiles do, so
  every field less than half a chip across is whole in some chip; the chips of
  the other engines do not overlap.
- **The reference must be complete inside the chips that contain its
  polygons**: pixels outside every reference polygon are labelled background.

`engine_params` for the chips:

| Parameter | Default | Meaning |
|---|---|---|
| `chip_size` | 224 (Prithvi), 256 (DINOv3), 512 below 4 m GSD else 256 (Delineate-Anything); GeoAI: sized from the reference fields (1.25 × the 90th percentile of the fields' longer bounding-box sides, in pixels, rounded up to a multiple of 32 and clamped to 256–1024 px) | Chip size in pixels. GeoAI infers on windows of the chip size, and no window sees the whole of a field larger than a chip. The engine rejoins such a field's pieces only where they meet at a window edge (`merge_window_seams`, see [Engines](engines.md#geoai-geoai)), so agribound logs a WARNING when more than 10 % of the reference fields are larger than the chip. |
| `chip_stride` | half the chip size (Delineate-Anything), the chip size (the other engines) | Pixels between the origins of neighbouring chips, from 1 to the chip size. A stride equal to the chip size gives chips that do not overlap (agribound 1.0.1's chips for every engine); half the chip size gives about four times as many chips (see [Train/validation split](#trainvalidation-split)). |
| `boundary_erosion` | 4 (DINOv3), 2 (the other engines) | Width of the boundary class in pixels: the reference pixels within this city-block distance of another polygon or of background. DINOv3 and Prithvi train on these masks and grow their predicted interiors back by it. A field less than 2 × `boundary_erosion` + 1 pixels across has no interior pixel; agribound counts such fields in the chips of these two engines (`chips_meta.json`) and logs a WARNING when they are more than 10 % of the reference fields. For DINOv3's default see [Engine-specific settings](#engine-specific-settings). |
| `min_label_fraction` | 0.01 | Minimum fraction of chip pixels inside reference polygons. |
| `min_valid_fraction` | 0.5 | Minimum fraction of valid image pixels. |
| `value_scale` | - | Required for Prithvi on `local` rasters. |
| `processing_gsd_m` | - (Delineate-Anything); `"auto"` (DINOv3) | Delineate-Anything: cut the chips from the raster resampled to this pixel size, as at inference. DINOv3: the grid of a bilinear copy that its chips are cut from and its inference reads; `"auto"` is 10 m for rasters with pixels more than 1 % larger than 10 m (Landsat and HLS, 30 m; Landsat PAN, 15 m) and the raster's own grid otherwise, `"native"` the raster's own grid, a number of metres finer than the raster's pixels that grid (a coarser number raises `ValueError`). |
| `stretch_percentiles` | `[1, 99]` | Delineate-Anything: percentiles of the scene stretch, as at inference. |

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

The split is made on chip-size cells: the raster is cut into cells of
`chip_size` pixels from its origin, the cells that the chips cover are split
as above (`fine_tune_val_split` is then the fraction of these cells), and a
chip takes the split of its cells. A chip whose cells are in different splits
is dropped. Chips that overlap therefore never put the same pixel in both
splits, although a field can still be in both (its parts in cells of
different splits). When the chips do not overlap, each chip is its own cell
and no chip is dropped. For overlapping chips the `split` record of
`chips_meta.json` gives the number of chips in each split (`n_train`,
`n_val`), of cells (`n_train_cells`, `n_val_cells`) and of dropped chips
(`n_dropped_both_splits`).

Overlapping chips have two costs. Half a chip apart, there are about four
times as many chips as without overlap, so each epoch takes several times
longer and the chips need several times the disk: in example 23's Higaturu
area (Delineate-Anything, 512 px chips at 1.5 m) 110 chips qualified instead
of 27. That matters most for large training sets, such as example 01's
statewide Landsat fine-tuning or example 12's NAIP runs: 650 chips of 512 px
without overlap in agribound 1.0, 2,882 chips (2,298 for training, 584 for
validation, 103 dropped) half a chip apart in the development run of 1.1.
And the validation share of the chips falls below `fine_tune_val_split`,
because most chips cover more than one cell and are
dropped when their cells are in different splits: at Higaturu, with a random
split at 0.2, 9 of the 47 cells went to validation, but only 6 of the 77
chips kept are validation chips (33 were dropped), mostly chips that fill
exactly one validation cell. Check the counts in `chips_meta.json`. For a
very small area, `engine_params["chip_stride"]` equal to the chip size
(chips that do not overlap) keeps every chip, and the block split, whose
blocks are whole groups of cells, drops only the chips that cross the edge
between a training and a validation block.

At least one training and one validation chip are guaranteed when there are
two or more chips; otherwise a `ValueError` is raised, also when dropping the
chips that cover cells of both splits leaves one split without chips.

## Engine-specific settings

**Delineate-Anything.** Labels are the reference polygons clipped to each
chip, without slivers: a part that a side of the chip frame cuts from its
polygon (the polygon extends beyond that side), and that reaches at most
`yolo_sliver_depth_px` pixels of the model input (default 16) into the chip
from that side, gets no label, in either split; `yolo_sliver_depth_px=0`
keeps them. Such a strip shows little of its field, and its edge along the
frame is not in the image. 16 px is the depth at which upstream DelAnyFlow's
tile-seam merge treats a tile-cut piece as only a strip along the border
(`seam_max_strip_depth_px`, also used by agribound's native backend). Parts
that end at nodata pixels are kept. Chips are upsampled by the same
super-resolution factor as at inference
(2 at 4 m GSD or coarser), so `imgsz` is 512 when chip size × factor is 512
(a WARNING is logged and `imgsz_matches_model_input: false` recorded
otherwise). Settings: `seed=config.seed`, `deterministic=True`, `mosaic=0`,
AdamW with `lr0` 0.002 (`yolo_lr0`) and a bias warmup learning rate of 0
(`yolo_warmup_bias_lr`; what Ultralytics' `optimizer="auto"`, whose AdamW
choice the recipe follows, sets; agribound 1.0.1 used Ultralytics' default of
0.1, which applies only to a named optimizer), flips, batch 16 (`yolo_batch`)
with an optimizer step after every batch (`yolo_nbs`, Ultralytics' nominal
batch size, defaults to the batch size; see below),
`epochs=fine_tune_epochs`; Ultralytics keeps `best.pt`. Its training
metadata (`best.pt.agribound.json`) records `processing_gsd_m` and
`stretch_percentiles` of the chips and the input they gave the model (pixel
size `gsd_m`, super-resolution factor, `raster_dtype` and stretch). A run
that gives the checkpoint another input (pixel sizes more than 1 % apart,
another factor or another stretch; a uint8 raster is not stretched) logs a
WARNING and records both in `engine_meta["checkpoint_input_mismatch"]`; the
options themselves are not compared, so a checkpoint fine-tuned on 1.5 m
chips gets the WARNING on a 3 m raster without any option. The test suite runs
this trainer against a stub of Ultralytics. In the development run of 1.1,
example 12's NAIP runs fine-tuned `large_v2` with this recipe and Ultralytics
8.4.163 (2,882 chips of 512 px every 256 px, 10 epochs in 73 minutes on an
Apple M2 Max; `lr0` 0.002, batch 16, `nbs` 16); their in-sample scores, and
those of the GeoAI and DINOv3 models fine-tuned on the same reference, are in
the [gallery](../gallery.md).

!!! warning "Small training sets"
    Ultralytics steps the optimizer once every `round(nbs / batch)` batches
    (during its warmup, the first three epochs, it raises that number from
    1), and its default `nbs` is 64: with batches of 16, the 22 training
    chips of example 23's Higaturu area (two batches per epoch) got 11
    optimizer steps in 20 epochs (10 without the warmup's ramp). agribound sets `nbs` to the batch size (`yolo_nbs`), so the
    optimizer steps after every batch. The default `lr0` can still move the
    pretrained weights further than the run can recover. On the 62 training
    chips (512 px, SPOT 6/7 panchromatic) of an oil palm estate in example 23,
    with chips that did not overlap and `nbs` 64, the validation mask mAP50 at
    the default `lr0` was 0.02 after the first epoch (the checkpoint
    Ultralytics kept) and 0.00 from the third epoch on; with
    `engine_params={"yolo_lr0": 1e-4}` it was 0.47 after 20 epochs, against
    0.17 for the released weights (agribound 1.0.1's bias warmup scored 0.00
    at the default `lr0`). With the current recipe, the same estate gave 185
    training and 23 validation chips (overlapping chips; 86 dropped) and a
    validation mask mAP50 of 0.58 after 20 epochs in the development run of
    1.1. With a small training set, set a smaller `yolo_lr0` and compare the
    validation scores (in the Ultralytics `results.csv` of the run) with
    those of the released weights. A better validation score does not
    guarantee a better result elsewhere. In example 23 the model fine-tuned
    on that estate did worse than the released weights on another estate,
    69 km from the training blocks (centres 81 km apart), with the earlier
    recipe and stitching (F1 0.23 against 0.29, the 1.0 gallery), and better
    with the current ones (0.40 against 0.32; recall 0.53 against 0.49). In
    the development run of 1.1 the model fine-tuned near Madera scored below
    the released weights there (0.78 against 0.83), and in example 24 the
    model fine-tuned on SPOT 6/7 of Mason Valley's training square (17
    training chips) scored below them on its evaluation square (0.43 against
    0.51).

**Tile-frame bias.** A model fine-tuned on few chips can learn the frame of
its chips and draw detections along the frame of every inference tile. With
agribound's earlier recipe (chips that did not overlap, Ultralytics' `nbs` of
64 and every clipped part labelled), the model fine-tuned on example 23's
Higaturu area (22 training chips, `yolo_lr0=1e-4`, 20 epochs) did so at Oro,
13.6 km from the training square: on the same pixels, its per-tile
detections covered 53 % of the pixels within 16 px of an interior tile side
and 18 % of those 64-128 px inside the tile (the reference parcels: 47 % and
50 %), 22 % of the reference parcels were detected only in tiles that cut
them, and 13.7 % of the output boundary lay within 2 m of a tile edge (1.9 %
by chance). Trained again with
the current recipe and the same settings (71 training chips, 100 optimizer
steps), the coverage was 51 % and 54 %, 5 % of the parcels were detected only
in cutting tiles, and 3.2 % of the boundary lay on the tile grid. Recall at
IoU 0.5 rose from 0.27 to 0.42 and the precision among the predictions that
overlap a parcel from 0.25 to 0.49, but the merge rate rose from 0.20 to 0.49
(the Oro output of the development run of 1.1 has the same recall, precision,
merge rate and share on the tile grid):
the model outlines whole plantings, and joins neighbouring parcels that the
image does not separate, where the earlier model drew parts of parcels (the
median per-tile detection: 1.58 ha against 0.65 ha; the median parcel:
1.53 ha). With another seed (7), the earlier recipe gave a model that found
almost nothing (recall 0.03, 22 % of the boundary on the tile grid) and the
current one recall 0.52, precision 0.57 and a merge rate of 0.36. All four
runs used agribound's current inference. The recipe changes were made after
this bias was found at Oro, so Oro is not an independent test of them; their
values (half a chip, `nbs` equal to the batch, 16 px) were not tuned there.

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

**DINOv3.** agribound trains geoai's `DINOv3Segmenter` with its own Lightning
loop, that of geoai's `train_dinov3_segmentation` (which agribound 1.0.1
called): AdamW (`learning_rate` 1e-4, `weight_decay` 1e-4) with cosine
annealing, batch 4, at most `fine_tune_epochs` epochs. Full fine-tuning by
default (`use_lora=False`, `freeze_backbone=False`; about 303 M backbone
parameters for ViT-L/16); `use_lora=True` (rank `lora_rank`, default 4; about
0.39 M adapter parameters) or `freeze_backbone=True` (decoder only).
`trainer_kwargs` is passed to the Lightning trainer. The recipe
(`agribound.engines.finetune._dinov3.RECIPE_VERSION` 2):

- **Class weights.** Cross-entropy with `ignore_index=255`, weighted by class
  (`class_weights`, default `"inverse_frequency"`): a class with n of the N
  labelled pixels of the training masks gets the weight N / (3 n), so the
  three classes weigh about equally in the loss (the average weight of a
  labelled pixel is 1), and no weight is larger than `class_weight_cap`
  (default 10, so no class counts more than 30 times another).
  `class_weights=None` trains unweighted, as agribound 1.0.1 did; three
  numbers set the weights of background, interior and boundary.
- **Checkpoint.** The epoch with the highest mean IoU of the three classes
  over all validation chips (`monitor="val_miou"`, default; `"val_loss"` is
  agribound 1.0.1's choice), with early stopping after
  `early_stopping_patience` (10) epochs without improvement. Only this
  checkpoint is written.
- **Input normalisation** (`input_normalization`). The model normalises its
  input with the mean and standard deviation its backbone was pre-trained
  with: `"sat493m"` (the default, for geoai's weights), `"lvd1689m"` (Meta's
  web-image weights; the default when the file name of `weights_path`
  contains `lvd1689m`), a dict `{"mean": [...], "std": [...]}`, or `"none"`
  (agribound 1.0.1). The statistics are saved in the checkpoint, so inference
  applies the same normalisation.
- **Boundary width.** `boundary_erosion` is 4 px by default for DINOv3 (2 px
  in agribound 1.0.1 and for the other engines; see below). It is saved in
  the checkpoint too, and inference grows the predicted interiors back by it.
  A fine-tuned model can predict a wider band than it was trained on (8-12 px
  against 4 px for example 24's DINOv3 models), and the outlines then lie
  inside the fields; `dilate_interior_px="half_band"` (opt-in) grows each
  interior by half the band measured on the prediction instead (see
  [DINOv3](engines.md#dinov3-dinov3)).
- **Processing grid.** A raster with pixels more than 1 % larger than 10 m
  (Landsat and HLS, 30 m; Landsat PAN, 15 m) is read on a bilinear copy with
  10 m pixels, for the training chips and at inference (`processing_gsd_m`,
  default `"auto"`; `"native"` reads the raster as it is). The model
  predicts on the 16 px patch grid, which is 480 m at 30 m, wider than the
  shorter side of 47 % of example 14's training fields; and the 30 m
  composite of example 14's square held only 4 chips of 256 px (3 training,
  1 validation: one optimizer step per epoch, and `val_miou` of the one
  validation chip kept the model after 3 steps). On the 10 m copy the same
  composite gives 52 chips (42 training, 10 validation). In CPU runs with
  example 14's settings (seed 42) the Landsat 2024 model's field F1 against
  the 388 V18 polygons rose from 0.023 to 0.538; checkpointing by
  `val_loss` (0.107) or 80 px chips at 30 m (0.102) helped far less. Seeds
  0 and 1 gave 0.528 and 0.545, and HLS 2024 0.542; Landsat PAN was not
  tested. In example 14's run with agribound 1.1.0 (2026-10-11, MPS) the
  in-sample F1 was 0.534 on Landsat and 0.547 on HLS, against 0.014 for both
  at 30 m in the development run of 2026-10-09. Rasters of 10 m or finer are
  read as they are, with the same
  cache keys and results versions as before. Pixel sizes are in metres
  (also in a projected CRS in feet); the copy holds the 10 m pixels that lie
  inside the raster (where the raster's size times the factor is not a whole
  number, such as an odd size at Landsat PAN's factor of 1.5, a strip
  narrower than one pixel along its last column or row is left out).
  `<checkpoint>.agribound.json` and the checkpoint's hyper-parameters record
  `processing_gsd_m`, and inference reads the grid the checkpoint was
  trained on, also when the checkpoint is moved without that file. Outputs
  of such fine-tunings made before the grid are not reused (results version
  `dinov3_processing_grid`; see
  [Reproducibility](reproducibility.md#output-reuse)).

The class weights, the pixel counts they come from, the monitored metric and
its best value and epoch, and the settings above are written to
`<checkpoint>.agribound.json`. The checkpoint holds the optimizer state: a
ViT-L/16 full fine-tuning checkpoint is about 3.7 GB (the ~303 M float32
backbone weights plus AdamW's two moment buffers). When a later epoch is
better, its checkpoint is written before the previous one is deleted, so the
cache directory needs about twice that in free space.

**DINOv3 boundary width.** The default was chosen on the validation chips of
example 23's four training areas (SPOT 6/7 panchromatic at 1.5 m), never on
an evaluated square. DINOv3 was fine-tuned on each area with 2, 4, 8 and
12 px and example 23's other settings (20 epochs, a random 20 % of the chips
for validation, seed 42), and the fields of each model (its interior regions
grown back by the band width, at least 2,500 m²) were matched one to one to
the reference fields of the area's validation chips at IoU 0.5. The rule,
fixed before the widths could be compared, takes the width with the highest
F1 averaged over the four areas or, of the widths within 0.01 of it, the
narrowest, since a wider band leaves more small fields without an interior
pixel:

| Validation chips, mean of the four areas | agribound 1.0.1 | 2 px | 4 px | 8 px | 12 px |
|---|---|---|---|---|---|
| F1 | 0.421 | 0.588 | 0.641 | 0.649 | 0.647 |
| Merge rate | 0.77 | 0.43 | 0.39 | 0.30 | 0.26 |
| Split rate | 0.01 | 0.08 | 0.09 | 0.14 | 0.13 |

The agribound 1.0.1 column is that of example 23's models in the 1.0 gallery
(a 2 px band and none of the other changes); they predicted the boundary class
on at most 1.8 % of the pixels of these chips. The models of the current
recipe, in the development run of 1.1, predicted it on 10.6 % (Madera) to
22.4 % (Oro) of the pixels of the four evaluated squares (0.0-0.7 % with
agribound 1.0.1's). The width is in pixels: a field less than
2 × `boundary_erosion` + 1 pixels across (9 px at the default: 13.5 m at
1.5 m, 90 m at 10 m, 270 m at 30 m) has no interior pixel and cannot be
learned. On the composite grids of the eastern Lea County square of
examples 12 and 14 that is 5.6 % of the 393 DRI/NMOSE training polygons at
Sentinel-2's 10 m and 41.2 % at 30 m (with 2 px: 1.0 % and 13.7 %; with 1 px
at 30 m: 3.3 %). On coarse imagery, check the count in `chips_meta.json`
(and the WARNING above 10 %) and consider a narrower band:
`agribound.engines.finetune.widest_boundary_erosion(reference, raster_path)`
returns the widest band of at most 4 px (`max_px`) that leaves at most 10 %
of the reference fields without an interior pixel (`max_share`), from the
labels and the raster's grid alone, with the share for every band. Examples
12 and 14 set DINOv3's band per source with it, measured on the grid DINOv3
reads (`grid_gsd_m=dinov3_processing_gsd(raster)`): 4 px on every source,
Landsat and HLS on their 10 m processing grid (on their own 30 m grid it
would be 1 px).

**DINOv3 recipe at Twifo Praso.** The NORPALM model that example 23
fine-tuned with agribound 1.0.1 drew one polygon over 94 % of the Twifo Praso
square (its field class covered 97 % of the square's pixels), and with the
current inference it still draws one polygon over most of the square. Below,
its output in the 1.0 gallery and NORPALM models fine-tuned again with
example 23's settings and run on the same composite with the current
inference (55 reference blocks; precision among the predictions that overlap
a block, and merge and split rates, as in example 23):

| NORPALM model | Polygons | Blocks matched (IoU ≥ 0.5) | Precision | Merge rate | Split rate |
|---|---|---|---|---|---|
| agribound 1.0.1 (the 1.0 gallery) | 1 | 0 | 0.00 | 1.00 | 0.00 |
| Current recipe (as in the development run of 1.1) | 102 | 36 | 0.40 | 0.13 | 0.27 |
| Current recipe, seed 7 | 107 | 36 | 0.39 | 0.16 | 0.29 |
| Without class weights | 22 | 0 | 0.00 | 0.98 | 0.04 |
| Checkpoint by validation loss | 142 | 29 | 0.25 | 0.04 | 0.51 |
| Without input normalisation | 113 | 33 | 0.33 | 0.25 | 0.38 |

Without class weights the boundary class covered 2.4 % of the square (12 %
with them) and the blocks merged again; the validation loss kept an earlier
epoch (5 instead of 13), which split many more blocks. Many of the extra
boundaries of the current model follow tracks inside the reference blocks.
The changes were made after the merging was found at Twifo Praso, so these
numbers are not an independent test of them; no setting was tuned there,
although NORPALM models with 2, 4 and 8 px bands were scored at Twifo Praso
while the widths were compared on the validation chips (the rule above
picked the width; the 8 px model had the higher precision at Twifo Praso).

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
another size; for Delineate-Anything and DINOv3 it also includes the version
of the training recipe (`agribound.engines.finetune._yolo.RECIPE_VERSION`,
`agribound.engines.finetune._dinov3.RECIPE_VERSION`), so a run cached by an
earlier recipe is trained again. For DINOv3 on a raster it reads on a
processing grid (pixels coarser than 10 m by default) the key also holds
the grid, and so does the key of its chips; the keys of other rasters do
not change. A second call with the same
inputs returns the cached checkpoint without retraining
(`finetune_manifest.json`).

## Large areas

`agribound tiles make` refuses `fine_tune: true` unless
`--allow-fine-tune-per-tile` is given (it would train one model per tile). Fine-tune once over the reference area and pass the checkpoint to the
tiles with `--engine-param checkpoint_path=<path>`; see
[HPC](hpc.md#fine-tuned-models). A Delineate-Anything checkpoint fine-tuned
with `processing_gsd_m` or `stretch_percentiles` needs the same values at
inference (`--engine-param processing_gsd_m=3`, for example). A DINOv3
checkpoint records its processing grid in its hyper-parameters and in
`<checkpoint>.agribound.json`, and inference reads it without the setting.
