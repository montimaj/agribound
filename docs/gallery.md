# Example Gallery (agribound 1.1)

Field boundaries from the agribound example scripts, run at their default
settings except where an entry names an option (example 23 with
`--s2-single-date`, example 24 with `--options` and `--sam-auto`, example 25
with `--crop-sequence` and example 28 with `--sam-auto-sentinel2`), and
rendered for agribound 1.1.0; one entry comes from the pre-registered
evaluation of the `sam-auto` engine (SAM 2.1 automatic masks). Examples 01
and 12 were not run end to end (12's NAIP runs were); examples 05, 10, 17, 18
and 21 ran but have no image here. The 1.0 images, rendered from the
agribound 1.0.0 and 1.0.1 runs (example 23's from
the development version that followed 1.0.1), are kept in
[`assets/gallery_1.0`](https://github.com/montimaj/agribound/tree/main/assets/gallery_1.0)
with their `gallery_stats.json`, and the 0.1.x screenshots on the
[archived 0.1.x page](gallery-0.1x.md).

**How to read the images.** Red outlines are agribound output, cyan outlines
are reference polygons and orange outlines are fields that SAM 2 refined (the
SAM 2.1 automatic masks of the `sam-auto` engine are agribound output, in
red). In the crop-filter image of example 20, magenta outlines are the
polygons the crop filter removes; in the irrigated-only images of example 24,
hatched fields are left out of the score and grey predictions are not scored.
Each map is drawn on a composite from the run (the overview map of example 25
on Natural Earth boundaries instead, and the left panel of its Montana image
on the NAIP of 2005, exported for that image; in the 0.1.x comparison the
top-left panel is the 0.1.x README screenshot), named under the map: usually
the engine's input; for FTW, window A, the first of FTW's two season inputs;
for the SAM 2 refinement entries, the composite SAM 2 read; for the embedding
panels of the tree-crop entries, the Sentinel-2 composite of the year; for the
crop-history zones (examples 25 and 27), the image that the
Delineate-Anything panel beside them read. The imagery therefore shows the
acquisition period of that composite. Zoom panels and cropped windows show the
square with the most polygons of one layer or the most reference polygons
(named in each entry) among candidate squares half a side apart, not a random
sample of the study area; the Pampas windows (example 15), the Nevada and New
York squares (examples 24 and 28) and the squares of the SAM 2.1 entry follow
the rules their entries state. The number in a panel title counts every
polygon in that output, not only those inside the window shown. The inset
locates the study area (red dot) in its country or state; in the images with
several study areas (example 22, the first and last tree-crop images of
example 23, the squares image of example 25, examples 26 and 27, and the SAM
2.1 entry) it shows them numbered as the panels. India is drawn from the
Survey of India outline, all other boundaries from Natural Earth. Under each
image are the imagery, the model and its version, and the data credits; the
Copernicus credit is printed only under the images of examples 24, 26, 27
and 28. Every image drawn on, or made from, a Sentinel-1, Sentinel-2 or HLS
composite contains modified Copernicus Sentinel data of the years of those
composites.

**SPOT colours.** SPOT 6/7 multispectral composites are uncalibrated digital
numbers, and each of their bands is stretched separately, while the other
sources share one stretch across R, G and B. SPOT colours therefore cannot be
compared with those of the other sources: bare soil often looks mauve or
lavender (examples 03, 11, 14, 15 and 24, and the SAM 2.1 entry). A median of
a few SPOT scenes can also show straight, sudden colour steps. In the SPOT
panel of example 14 (12 images), a nearly vertical band about 100 m wide, a
little left of the
middle, darkens the lower third of the panel and the pivots it crosses there.
The band is in the composite the engine read, not added by the rendering.
SPOT-Pan panels are shown in grey.

**Radar.** Sentinel-1 panels (examples 24 and 26) show the composite the
engine read: VV, VH and VV − VH backscatter as red, green and blue. Umbra and
Capella panels (example 26) show X-band backscatter in grey.

**Crop filter.** Unless an entry says otherwise, the polygons pass a crop
filter at a threshold of 0.30: outside the conterminous US, Dynamic World (the
polygon mean of the year's median crop probability); in the conterminous US,
Annual NLCD (the share of cultivated-crop and pasture/hay pixels, classes 82
and 81). The minimum field area is given in each entry.

!!! note "What these images show"
    The images were rendered from these runs of the example scripts and one
    pre-registered evaluation (each named with its run root under `outputs/`,
    the folder `tools/make_gallery.py` reads; see
    [Contributing](contributing.md)):

    - the development run of agribound 1.1 on 2026-10-09/10
      (`outputs/examples_1.1`), whose outputs 1.1.0 reuses unchanged: every
      entry except those below, and the outputs that the runs below reused;
    - the targeted 1.1.0 re-runs of 2026-10-11 (`outputs/examples_1.1_final`):
      example 03 (made again; it corrects the HLS image count), example 14
      (Landsat and HLS made again on DINOv3's 10 m processing grid), example 23
      with `--s2-single-date` (its TESSERA runs made again), example 25 with
      `--crop-sequence` (the crop-history zones), example 27 (new) and example
      10 (no image);
    - example 26's run of 2026-10-10 (`outputs/examples_1.1_sar`);
    - example 24's re-scoring run of 2026-10-11 with 1.1.0
      (`outputs/examples_1.1_ex24`), which added the irrigated-only scores to
      the same model outputs (34 of its 36 outputs reused; the two TESSERA runs
      made again, with identical polygons); its all-fields scores equal those
      of the development run;
    - example 24's run of 2026-10-13 with `--options` and `--sam-auto`
      (`outputs/examples_1.1_ex24b`), which reused the 28 runs and 8
      fine-tunings of the re-scoring run unchanged and added 6 `sam-auto` runs
      (NAIP, SPOT 6/7 and Landsat in each basin) and 8 "-opt" runs: entries 24
      to 24e;
    - example 28's run of 2026-10-12/13 with `--sam-auto-sentinel2`
      (`outputs/examples_1.1_r3/ny`; 26 runs, 12 of them drawn): entries 28 to
      28c;
    - the pre-registered evaluation of `sam-auto` against Delineate Anything v2
      as released, whose runs were made on 2026-10-13
      (`outputs/examples_1.1_r3/sam_eval`): the `sam-auto` panels of the SAM
      2.1 entry, whose Delineate-Anything panels are outputs of examples 24
      and 25 named above. Its `sam-auto` layers on example 24's two squares
      equal those of example 24's run with `--sam-auto`, coordinate for
      coordinate.

    The Montana image of example 25 also draws the NAIP of 2005, exported from
    Earth Engine on 2026-10-12 by `tools/fetch_naip_history.py`
    (`outputs/examples_1.1_r3/montana`).

    Three evaluations ran on 2026-10-12/13, each with rules written and
    hashed before any of its runs; the footers name them:

    - **PREREG-SAM**, `sam-auto` against Delineate Anything v2 as released (the
      SAM 2.1 entry). It was amended once before any of its runs and again on
      2026-10-13, after its first scores and endpoints had been read: Mason
      Valley's pairs had been scored against a crop-filter copy of its
      reference (176 of its 187 fields), so both Nevada areas were scored
      again with the reference, a recorded deviation from its section 6.
      Mason Valley's scores changed; Diamond Valley's did not.
    - **PREREG-DV3**, the test of the Diamond Valley options (entry 24e). It was
      revised and amended before any run on its squares (2026-10-12, at
      17:22Z and 19:06Z) and amended again on 2026-10-13, for a defect of its
      check step found before any score (one-part MultiPolygons compared with
      Polygons), which changed no layer or score; the evaluation was then run
      again in full.
    - **PREREG-NY**, example 28's claim and panel rules (entries 28 to 28c),
      amended three times before example 28 ran: the implementation of the
      paired intervals, the `--sam-auto-sentinel2` runs, and the author's
      approval to show the Common Land Unit outlines. The round's post-check
      of the run also matched the crop-filter copies of the references; the
      author marked the step done, a recorded deviation that changed no
      number.

    The legends name agribound 1.1.0, except in the overview map of example
    25 (its legend explains the markers), in its Montana image (no model
    output) and in the 0.1.x comparison (no legend; its panel titles name
    1.1.0). The provenance records of the development run and of example 26's
    run name agribound 1.0.1, the version label their code snapshots still
    carried; the records hold the results versions and configuration keys of
    1.1, and 1.1.0 reuses these outputs. The outputs made on 2026-10-11 and
    later name 1.1.0.

    Each image shows what one configuration produces, not how accurate it is.
    Accuracy is reported only where a reference layer exists: examples 12, 13,
    14 and 20 against the DRI/NMOSE field boundaries (V18) in New Mexico (for
    14 and the fine-tuned models of 12 they are also the training labels, so
    those scores are in-sample); example 23 against RSPO, DWR / Land IQ and
    SIGPAC polygons and example 24 against the DRI field boundaries of
    Nevada, with fine-tuned models trained elsewhere; example 25 against the
    OpenET field boundaries v2.1.1 (in New Mexico against V18 and in Nevada
    against the DRI field boundaries); example 26 against OpenET, DWR / Land IQ,
    SIGPAC and Utah polygons and the DRI field boundaries of Nevada; example
    27 against the parcels declared in Quebec; example 28 against the OpenET
    field boundaries v2.1.1 in New York, all USDA FSA Common Land Units, with
    fine-tuned models trained on squares 5 to 15 km away (and, in the Black
    Dirt, near-sample models trained 1 km away, which are not held out); the
    SAM 2.1 entry against the DRI field boundaries in Diamond Valley and the
    OpenET field boundaries (all Common Land Units) in Tennessee, its
    endpoints also against example 25's other squares, Mason Valley's square
    and the V18 boxes of examples 12 and 20. Delineate Anything v2's own
    training data (FBIS-73M) hold labels
    that match the DWR / Land IQ and SIGPAC polygons: they cover the Madera
    square and 87 % of the Úbeda square of example 23, so its scores there are
    mostly on fields it has seen. Located patch by patch on 2026-10-13 from
    their own image headers, FBIS-73M's public lists hold US training labels
    only in California and Canadian training labels only in Ontario. Of
    example 25's 47 squares only California's holds any patch (128, all
    labelled training patches). The nearest patch lies 38.0 km from Mason
    Valley's square, 90.6 km from the San Juan box, 98.9 km from example 25's
    Nevada square, 212.4 km from its New Mexico square, 285.8 km from Diamond
    Valley's square, 543.0 km from the Lea box and 146.9 km or more from
    example 28's evaluation squares. Only the US and Canadian patches were
    located, and the lists were published after the model's training and
    changed on their release day, so they may not be exactly what the
    released weights saw.

The images are rendered at 3000 px by
[`tools/make_gallery.py`](https://github.com/montimaj/agribound/blob/main/tools/make_gallery.py)
from the run outputs. This page shows 1600 px WebP previews; click an image to
open the full-resolution PNG. The numbers below come from the run logs, the
evaluation metrics and summary files, and
[`assets/gallery_1.1/gallery_stats.json`](https://github.com/montimaj/agribound/blob/main/assets/gallery_1.1/gallery_stats.json).

## Models and versions

| Engine | Model and version used in these runs |
|---|---|
| Delineate-Anything | Delineate Anything v2, `large_v2` (YOLO11x-seg, trained on FBIS-73M): `DelineateAnythingv2.pt` from Hugging Face `MykolaL/DelineateAnything` at revision `369d0b4`, SHA-256 pinned; agribound's native implementation with ultralytics 8.4.163 and the tile stitching of agribound 1.1 (results version `delineate_anything` 2); confidence threshold 0.15; up to 1,000 detections per tile on Sentinel-1, Sentinel-2, Landsat and Landsat PAN in examples 23-28; fine-tuned with agribound 1.1's recipe in examples 12, 23, 24 and 28; in entry 24e also read on a 3 m grid (`processing_gsd_m=3`, an opt-in option of 1.1.0) |
| FTW | `FTW_PRUE_EFNET_B5` ("FTW v3: Standard, B5": PRUE U-Net with an EfficientNet-B5 encoder), ftw-baselines v3 checkpoint, SHA-256 pinned; run with ftw-tools 2.0.0b5 (a pre-release); no fine-tuning |
| SAM 2 (refinement) | `facebook/sam2-hiera-large` (SAM 2.0 Hiera-L, not 2.1), through segment-geospatial 1.4.2 and the `sam2` 1.1.0 package; one box prompt per field |
| SAM 2.1 automatic masks (`sam-auto`) | `facebook/sam2.1-hiera-large` (SAM 2.1 large) at revision `665f8e2`, SHA-256 checked, through the `sam2` 1.1.0 package, a third-party PyPI build whose 62 code, configuration and licence files equal Meta's commit `2b90b9f` (SAM 2.1 code and weights: Meta, Apache-2.0); a grid of point prompts on large overlapping tiles, with the engine's setting for the source: NAIP 2,048 m tiles, 32 points per side, predicted IoU > 0.8, stability ≥ 0.95; SPOT 6/7 6,144 m tiles, 32 points per side, predicted IoU > 0.6, stability ≥ 0.95; no field labels and no fine-tuning; run on Apple MPS (results version `sam_auto` 1) |
| Prithvi | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL` at revision `63adbd3`, terratorch 1.2.13 |
| GeoAI | torchvision Mask R-CNN ResNet50-FPN via geoai-py 0.43.1, fine-tuned on the reference polygons (no published field weights); chips sized from the reference fields; instances split at the inference-window edges joined |
| DINOv3 | `dinov3_vitl16` (ViT-L/16) with the SAT-493M weights `giswqs/geoai/dinov3_vitl16_sat493m.pth` at revision `aa2b25d`, geoai-py 0.43.1; full fine-tuning on the reference polygons (in examples 23, 24 and 28, on the labels of a training area near each study area, never on the evaluated square; no published field weights) with agribound 1.1's recipe (class weights, a 4 px boundary band, the checkpoint with the best validation mean IoU) and blended inference windows; rasters coarser than 10 m are read on a 10 m bilinear copy; in entry 24e also grown by half the predicted boundary band (`dilate_interior_px="half_band"`, an opt-in option of 1.1.0) |
| Embeddings | Google Satellite Embedding (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, AlphaEarth Foundations, 64-D) and TESSERA v1 (128-D, geotessera 0.10.2, read from the store as converted again upstream on 2026-10-06/08; see [Satellite Sources](user-guide/satellite-sources.md#embeddings)); PCA to 16 components, then scikit-learn `KMeans` with ten restarts (`n_init=10`); k chosen by silhouette score among 5, 10, 15, 20, 30 and 50 unless stated (in every automatic choice here the score was highest at k = 5, the smallest candidate, and smaller k were not tested). agribound 1.0.0 used `MiniBatchKMeans` on rasters of more than 100,000 valid pixels, as all of these are |
| Crop-sequence zones | agribound's `crop-sequence` engine (`agribound-crop-sequence-1`; no model): the USDA Cropland Data Layer (example 25) or the AAFC Annual Crop Inventory (example 27) of 2016-2023 at 30 m, cultivated in at least 5 of the 8 years, a per-year focal mode of radius 3 px, regions of equal crop sequence, regions under 1 ha merged |

**SAM size rule** (SAM 2 refinement, not `sam-auto`). SAM is prompted only
when a field's bounding box, padded on every side by 15 % of its size (a
factor of 1.3), is at least 64 pixels wide and 64 pixels tall. That is an
unpadded box of at least about 49 pixels
on each side (64 / 1.3 ≈ 49.2): about 1.5 km at 30 m, 490 m at 10 m, 295 m at
6 m and 49 m at 1 m. Fields below that keep their geometry. With the default
`sam_overlaps="trim"`, a refined mask cannot take area from a neighbouring
polygon. Since 1.0.1, a mask that covers less than half of its input polygon
after that trim (`sam_min_coverage`, default 0.5) is not used, and the polygon
keeps its input geometry.

**Delineate-Anything tiles.** agribound runs Delineate Anything v2 on tiles
that overlap by half and give the model 512 px each (768 m on SPOT-Pan, 512 m
on NAIP, 2.56 km on Sentinel-2, whose 256 px tiles are upsampled 2×), and
rebuilds a field larger than a tile from the pieces its tiles return (see
[Engines](user-guide/engines.md#delineate-anything-delineate-anything)).
Some images show where this goes wrong: straight cuts along tile edges
(examples 22 in western Bahia, 23 at Twifo Praso, 24 and 25), a block of
narrow fields drawn as one polygon (22, Nile Delta), joins of two fields
across a tile edge (23, Madera; outside the windows shown), large polygons
chained from many tile-cut pieces (25, Montana), and a pivot cut in two where
the model scored the pieces higher than the whole field (13). Candidate fixes
of the stitching were evaluated on the native Delineate-Anything outputs of
the development run against rules set before the evaluation. Each of them
lowered F1 by more than 0.005 in some evaluations or put more boundary on tile
edges in some outputs, so none was adopted, and the entries describe what is
visible.

---

## Lea County, New Mexico — DINOv3 fine-tuned, from 30 m to 1 m

**Example 14** · DINOv3 ViT-L/16 (SAT-493M weights, geoai-py 0.43.1), fully
fine-tuned on the 393 DRI/NMOSE field boundaries (V18) of the study area
separately for each source, then refined with SAM 2 · eastern Lea County, New
Mexico, north of Hobbs (the easternmost 1.3 km of the box, 7 % of its area, is
in Gaines County, Texas, where V18 has no polygons) · Landsat 8/9 and
Sentinel-2 of October 2024, SPOT 6/7 of 2023, NAIP of 2022 (2024, or the
latest year of a source without 2024 imagery over the box) · min. area
2,500 m² (5,000 m² for NAIP); Annual NLCD crop filter of each source's year ·
window: the 6 km square with the most V18 polygons among those half a side
apart in this map.

Panels: Landsat 30 m (3 images), Sentinel-2 10 m (7 images), SPOT 6/7 6 m
(12 images; its colour band is described under "SPOT colours" above) and NAIP
1 m (25 images). The V18 polygons (cyan) are also the training labels, so
these scores are in-sample, not an independent accuracy estimate. Against the
388 V18 polygons with their representative point in the box (one more is
smaller than 2,500 m² and dropped; one-to-one matching at IoU ≥ 0.5), the
fields and the in-sample F1 without and with SAM 2 are:

| Source | Fields (without / with SAM 2) | In-sample F1 without SAM 2 | In-sample F1 with SAM 2 |
|---|---|---|---|
| Landsat 30 m, read on a 10 m copy | 226 / 226 | 0.53 | 0.53 |
| Sentinel-2 10 m | 218 / 219 | 0.56 | 0.57 |
| SPOT 6/7 6 m | 259 / 259 | 0.58 | 0.59 |
| NAIP 1 m | 413 / 413 | 0.56 | 0.56 |

(HLS 30 m, not shown: 223 fields, in-sample F1 0.55 with and without SAM 2.)
agribound 1.1.0 reads rasters coarser than 10 m on a 10 m bilinear copy for
DINOv3, for the training chips and at inference (the footer says so for the
Landsat panel). On that grid the box holds 52 training chips of 256 px for
Landsat and for HLS (42 for training, 10 for validation), as for Sentinel-2.
Read at 30 m, as in the development run of 2026-10-09, it held 4 (3 and 1),
and F1 was 0.014 on both; the Landsat run shown here and the HLS run (not
shown) were made again with 1.1.0 on 2026-10-11, and the other three panels
are the development run's. From 10 m to 1 m the training data still change
with the pixel size (52, 134 and 2,354 chips), and NAIP uses a minimum area of
5,000 m².

SAM 2 changes in-sample F1 by +0.003 at most (Sentinel-2 0.564 to 0.567, SPOT
0.584 to 0.587). On NAIP, F1 is the same with and without it (226 matched
fields), and the mean IoU of the matched fields falls from 0.936 to 0.912. SAM
2 was prompted with 1,006 of the 1,016 NAIP polygons it received (10 were
below the size rule) and refined 759; 246 masks covered less than half of
their polygon, which keeps its DINOv3 outline (`sam_min_coverage`), and 1
failed. On Sentinel-2 it refined 180 of 295 polygons (115 below the size rule)
and on SPOT 230 of 406 (175 below; 1 mask covering too little). On Landsat and
HLS it changed nothing: read at 30 m, every polygon is below the size rule.
Without SAM 2, 13 of the 413 NAIP polygons hold two or more V18 fields (26
fields in all). The four checkpoints shown were trained with agribound 1.1's
DINOv3 recipe; the 1.0 image showed agribound 1.0.1's recipe, trained on the
NMOSE WUCB polygons and on imagery of 2022, so its scores cannot be compared
with these.

Reference: DRI/NMOSE field boundaries (V18), Desert Research Institute and New
Mexico Office of the State Engineer.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/NM_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/NM_example.webp" alt="Lea County — DINOv3 fine-tuned and SAM 2 on Landsat, Sentinel-2, SPOT and NAIP" width="800"></a>

---

## Lea County, New Mexico — Delineate-Anything v2, GeoAI and DINOv3 on NAIP 1 m

**Example 12** (NAIP runs only; the rest of example 12 was not run for 1.0
or 1.1) · eastern Lea County, New Mexico, north of Hobbs (the box of example
14) · NAIP 1 m, 2022 (25 images; the composite of example 14) · min. area
5,000 m²; Annual NLCD 2022 crop filter · window: the 6 km square with the most
V18 polygons among those half a side apart in this map (the candidates follow
each map's extent, so the window is not example 14's).

Delineate Anything v2 as released (top left; a run with example 12's
settings that example 12 itself does not include) and three models
fine-tuned on the 393 V18 polygons of the box with example 12's settings
(10 epochs, 5 km block split): Delineate Anything v2 `large_v2` (2,882 chips
of 512 px, half a chip apart: 2,298 for training and 584 for validation),
GeoAI Mask R-CNN (245 chips of 1,024 px: 196 and 49) and DINOv3 (2,354 chips
of 256 px: 1,846 and 508). The V18 polygons (cyan) are the fine-tuning labels,
so the scores of the fine-tuned models are in-sample, not independent
accuracy estimates. Against the 388 V18 polygons in the box (one-to-one
matching at IoU ≥ 0.5):

| Model | Fields | Precision | Recall | F1 | Mean IoU of matched fields | Predictions overlapping no reference polygon |
|---|---|---|---|---|---|---|
| Delineate Anything v2 as released | 608 | 0.28 | 0.44 | 0.35 | 0.86 | 189 |
| Delineate Anything v2 fine-tuned (in-sample) | 301 | 0.54 | 0.42 | 0.48 | 0.89 | 63 |
| GeoAI fine-tuned (in-sample) | 753 | 0.22 | 0.43 | 0.29 | 0.77 | 196 |
| DINOv3 fine-tuned (in-sample) | 415 | 0.53 | 0.57 | 0.55 | 0.92 | 79 |

Many of the predictions that overlap no reference polygon are fields the
registry does not include (V18 has no polygons in the Texas strip of the
box). Fine-tuning mainly reduced Delineate Anything v2's predictions outside
the registry (189 to 63) and raised its precision (0.28 to 0.54), at a small
cost in recall (0.44 to 0.42).

GeoAI's chips are sized from the reference fields, and the instances it
splits at the edges of its overlapping 1,024 px inference windows are joined
(`merge_window_seams`; 370 instances joined here); without these two engine
changes of 1.0.0 it did not produce whole fields (see
[Engines](user-guide/engines.md#geoai-geoai)). Its 1.1 output scores lower
than its 1.0 output at both references: F1 0.29 against 0.44 on V18, and 0.26
against 0.48 on the NMOSE WUCB polygons that 1.0 was trained and scored on.
The NAIP composite, GeoAI's code and settings and the seed were the same in
both runs; the labels (755 V18 training instances against 466 WUCB ones), 16
more chips and the operating system differed, and which of these lowered the
score was not established. The 1.1 model splits more fields (156 V18 fields
in two or more polygons, against 85), and the panel shows straight
north-south and east-west cuts through many pivots, often meeting at the
pivot centre; 88 % of the extra boundary inside the fields lies off the
512 px seams of the inference windows.

Reference: DRI/NMOSE field boundaries (V18), Desert Research Institute and New
Mexico Office of the State Engineer.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/NM_NAIP_models_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/NM_NAIP_models_example.webp" alt="Lea County — Delineate-Anything v2, GeoAI and DINOv3 on NAIP" width="800" loading="lazy"></a>

---

## San Juan County, New Mexico — Delineate-Anything v2 from 30 m to 1 m

**Example 20** (resolution comparison) · Delineate Anything v2 used as
released, on Landsat 7/8/9 (a median of the Landsat 7, 8 and 9 Collection 2
Level-2 collections; 30 m, 115 images), Sentinel-2 (10 m, 182 images), SPOT
6/7 (6 m, 1 image) and NAIP (1 m, 24 images), all 2022 annual composites ·
the study area of example 20 · min. area 2,500 m²; Annual NLCD 2022 crop
filter · window: the 2.5 km square with the most reference fields.

NAIP is flown every two years here and Earth Engine has no NAIP after 2022,
so 2022 is the year closest to the main run's 2024 that all four sources
cover. Against the 1,355 V18 polygons of at least 2,500 m² in the box (350 of
the 1,705 are smaller and dropped; one-to-one matching at IoU ≥ 0.5; the
polygons were not used for training or fine-tuning in these runs):

| Source | Fields in the box | Precision | Recall | F1 | Boundary F1 (10 m) |
|---|---|---|---|---|---|
| Landsat 30 m | 133 | 0.72 | 0.07 | 0.13 | 0.13 |
| Sentinel-2 10 m | 407 | 0.62 | 0.19 | 0.29 | 0.36 |
| SPOT 6/7 6 m | 321 | 0.45 | 0.11 | 0.17 | 0.22 |
| NAIP 1 m | 995 | 0.46 | 0.34 | 0.39 | 0.54 |

(Fields in the box: the predictions whose representative point lies inside
the evaluation box; here every polygon of the four outputs does, so the panel
titles give the same counts.)

Recall rises about fivefold from 30 m to 1 m and boundary F1 about fourfold.
The 2022 SPOT composite is a single scene (2022-11-07) with valid data over
76.7 % of the box: 158 of the 1,355 reference fields lie entirely in the
western strip without data (west of about 108.19° W) and count as missed, and
5 more lie partly in it. On the 1,192 fields inside its valid area SPOT scores
F1 0.19, against 0.28 for Sentinel-2; on the same fields the 1.0 output from
the SPOT composite of 2018 (7 scenes) scores 0.25. In 2023 no SPOT scene
covers the box; the footprints of the two scenes of 2021 cover all of it
(their valid data were not checked). Landsat is outside Delineate Anything
v2's 0.25–10 m training range (agribound warns and records it). The crop
filter removed 35 % of the output polygons on Landsat (it kept 133 of 205),
40 % on Sentinel-2 (407 of 674), 32 % on SPOT (321 of 469) and 63 % on NAIP
(995 of 2,664). V18 records no imagery date per field, so fields
that changed between the reference's date and the imagery count as errors.

Reference: DRI/NMOSE field boundaries (V18), Desert Research Institute and New
Mexico Office of the State Engineer.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/San_Juan_resolution_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/San_Juan_resolution_example.webp" alt="San Juan County — Delineate-Anything v2 on Landsat, Sentinel-2, SPOT and NAIP" width="800" loading="lazy"></a>

---

## San Juan County, New Mexico — FTW and Delineate-Anything v2 with and without the crop filter

**Example 20** (crop-filter comparison) · the pre-trained FTW
(`FTW_PRUE_EFNET_B5`; windows 10 Mar–9 May 2024, 27 images, shown in the top
panels, and 19 Sep–18 Nov 2024, 29 images) and Delineate Anything v2 (annual
2024 composite, 184 images, bottom), both used as released, each with the
Annual NLCD 2024 crop filter on (left) and off (right) · min. area 2,500 m² ·
window: the 3 km square with the most reference fields.

With the filter off, the polygons it would remove are outlined in magenta. The
filter kept 445 of FTW's 1,526 polygons and 357 of Delineate Anything v2's
591. Against the 1,355 V18 polygons (one-to-one matching at IoU ≥ 0.5; the
polygons were not used for training or fine-tuning in these runs):

| Engine | Crop filter | Fields | Precision | Recall | F1 | Predictions overlapping no reference polygon |
|---|---|---|---|---|---|---|
| FTW | on | 443 | 0.35 | 0.12 | 0.17 | 57 |
| FTW | off | 1,524 | 0.10 | 0.12 | 0.11 | 1,066 |
| Delineate Anything v2 | on | 357 | 0.64 | 0.17 | 0.27 | 8 |
| Delineate Anything v2 | off | 591 | 0.40 | 0.18 | 0.25 | 203 |

(Fields in the box, by representative point: 2 of FTW's polygons lie outside
it.) Of the 1,081 polygons the filter removes from FTW's output, 1,009 overlap
no reference polygon (195 of 234 for Delineate Anything v2); in this window
most of them lie on the bare slopes north of the valley and along the river.
For both engines the filter raises precision and lowers recall by less than
0.01. It also removes some polygons that do overlap reference fields (a few are
visible here).

Reference: DRI/NMOSE field boundaries (V18), Desert Research Institute and New
Mexico Office of the State Engineer.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/San_Juan_crop_filter_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/San_Juan_crop_filter_example.webp" alt="San Juan County — FTW and Delineate-Anything v2 with and without the crop filter" width="800" loading="lazy"></a>

---

## Pampas, Argentina — embeddings + SAM 2 vs Delineate-Anything v2 on Sentinel-2 and SPOT

**Example 15** · Pergamino partido, Buenos Aires Province, east of the city ·
label-free (no training and no reference data) · min. area 5,000 m²; Dynamic
World crop filter of each input's year (2024; 2023 for SPOT) · window: zoom 1
of the next entry, the 4 km square with the most centre pivots.

- Top: Google Satellite Embedding and TESSERA v1 clusters of 2024, after the
  crop filter and SAM 2 on a Sentinel-2 composite of October 2024, with parts
  over 50 ha kept unrefined (1,986 and 2,154 fields; see the next entry).
- Bottom left: Delineate Anything v2 on the same October 2024 Sentinel-2
  composite (8 images): 2,744 fields (the crop filter kept 2,744 of 3,213).
- Bottom right: Delineate Anything v2 on SPOT 6/7 (6 m), 2023 (6 images;
  AIRBUS/SPOT6_7 ends on 2023-11-15): 3,199 fields (kept 3,199 of 3,648).

The four layers come from the development run; the Google layers are those of
1.0.1, the TESSERA layers changed with the TESSERA store (below) and the
Delineate-Anything layers with the tile stitching of 1.1. On Sentinel-2,
Delineate-Anything outlines most pivots in the window as fields of their own
and splits a few along tone changes inside the circle. On SPOT it leaves at
least one faint pivot inside a larger rectangular field and breaks the
two-tone pivot at the top left into pieces. The two inputs differ in date,
season and resolution (SPOT: a 2023 median of 6 images at 6 m; Sentinel-2: an
October 2024 median of 8 images at 10 m) and in crop-filter year (2023 and
2024); which of these differences causes the different outlines was not
tested. Before SAM 2, 8 of the 14 pivots in the window have a TESSERA polygon
of their own (IoU ≥ 0.8) and 7 a Google one; the others have a rougher polygon
(IoU 0.5–0.8), are part of larger polygons or, for one Google pivot, have
almost no polygon (see zoom 1 in the next entry). SAM 2 neither joins pieces
nor splits merged polygons (it returns one polygon per input polygon). The
pivots here are 40.5–56.8 ha, close to 50 ha, so some are refined (orange) and
others keep their cluster outline (red). The TESSERA outlines lie on the pivot
edges of the Sentinel-2 composite: the TESSERA v1 store was converted again
upstream on 2026-10-06/08, which moved 91 % of this example's TESSERA raster
one 10 m pixel west or north-west and left 9 % in place (see
[Satellite Sources](user-guide/satellite-sources.md#embeddings)). There is no
reference layer here; the panels compare outlines, not accuracy.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Pampas_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Pampas_example.webp" alt="Pampas — embeddings with SAM 2 vs Delineate-Anything v2 on Sentinel-2 and SPOT" width="800" loading="lazy"></a>

---

## Pampas, Argentina — Google Satellite Embedding and TESSERA, whole study area and three zooms

**Example 15** (steps 1–4) · Label-free: no training and no reference data ·
Pergamino partido, Buenos Aires Province, east of the city · min. area
5,000 m²; Dynamic World 2024 crop filter · the whole study area (a pentagon
with a bounding box of about 28 × 31 km), then three 4 km windows (yellow
squares 1–3).

Google Satellite Embedding (top) and TESSERA v1 (bottom) embeddings of 2024
are clustered (k = 5) and kept where they pass the crop filter (left): it kept
1,986 of 2,307 Google and 2,154 of 2,310 TESSERA polygons. SAM 2 then refines
them on a Sentinel-2 composite of October 2024, 8 images (right), in the
example's split variant: parts over 50 ha are kept unrefined (205 Google and
284 TESSERA polygons; neither crop layer has multi-part polygons, so nothing
was split). Of the other 1,781 and 1,870 polygons, SAM 2 was prompted for 236
and 298 and refined 208 and 287. The other 28 and 11 masks covered less than
half of their input polygon, so those polygons keep their input geometry;
1,545 and 1,572 polygons were below the size rule and were not prompted.
SAM's overlap trim sees only the polygons it is given, so the example then
trims the refined masks where they overlap the kept polygons (167 Google and
189 TESSERA masks). The 5,000 m² filter that follows removed no polygon:
1,986 and 2,154 remain, 208 and 287 of them refined (orange). The script
smooths and simplifies the polygons of 50 ha or less again after SAM; the
larger ones keep their crop-filter outline.

Each crop layer leaves a large group of fields without a polygon. In the
TESSERA cluster raster, one cluster forms a connected region of 21,421 ha. Its
representative point lies outside the study area, so the study-area rule
drops it. In the Google cluster raster, a connected region of 11,105 ha is
kept as one polygon, which the crop filter then removes. Delineate-Anything
outlines 2,857 ha (TESSERA region) and 1,896 ha (Google region) of fields on
Sentinel-2 in these regions; 2,809 ha and 1,880 ha of them are covered by no
crop-filter polygon of that embedding (see
[Engines](user-guide/engines.md#embedding-clustering-embedding)).

Why the split: SAM returns one object for each box prompt, and the refined
polygon replaces the whole input polygon when its mask covers at least
`sam_min_coverage` = 0.5 of it. Refining every polygon
(`fields_*_crop_sam2-s2_2024.gpkg`) removed 4.9 % of the Google and 6.2 % of
the TESSERA crop-filter area (EPSG:6933 sums); in that run, 70 Google and 36
TESSERA masks covered less than half of their polygon and were not used. Of
29 centre pivots located in the composite and checked by eye, 4 (Google) and
2 (TESSERA) were then less than half covered by any polygon. Three of the
Google ones had been part of a cluster polygon more than twice their area (2.4
to 7.1 times); the fourth lay inside the 11,105 ha polygon that the crop filter
removed, so it had almost no polygon already before SAM (0.4 % covered). Of
the two TESSERA pivots, one (2.0 % covered) had been part of a polygon 2.7
times its area and the other (44.9 % covered) of one 1.8 times its area. With
the split, 1 Google pivot and no TESSERA pivot is less than half covered: the
Google one is that pivot inside the removed 11,105 ha polygon (0.2 % covered;
see [SAM Refinement](user-guide/sam-refinement.md#polygons-that-cover-several-fields)).
The split layers cover 0.8 % (Google) and 0.1 % (TESSERA) less ground than the
crop-filter polygons. They keep multi-field polygons as the clustering drew
them: 6 of the 29 pivots are inside a polygon more than twice their area in
the TESSERA layer, 9 in the Google layer, all of these polygons unrefined.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Pampas_SAM2_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Pampas_SAM2_example.webp" alt="Pampas — Google Satellite Embedding and TESSERA clusters before and after SAM 2, whole study area" width="800" loading="lazy"></a>

### Zoom 1: centre pivots

The 4 km square with the most centre pivots (14 of the 29). The TESSERA
clusters give 8 of them a polygon of their own (IoU ≥ 0.8) and 2 a rougher one
(IoU 0.5–0.8). The other four, among them the two-tone pivot at the top left,
are part of one 568.7 ha polygon, 10 to 14 times the area of each, which stays
unrefined. The Google clusters give 7 pivots a polygon of their own and 2 a
rougher one. Three at the left are part of one 137.2 ha polygon (2.4 to 2.6
times the area of each), and one at the bottom right shares an 89.3 ha
polygon with the pivot above it (IoU 0.39). The dark-green pivot at the left
edge has almost no Google polygon (0.4 % covered): it lay inside the
11,105 ha cluster polygon that the crop filter removed. SAM 2 redraws the
outlines of the pivots it refines (orange) along their edges and can leave
holes along within-field variation. Refining every polygon leaves the four
TESSERA pivots of the 568.7 ha polygon 93–98 % covered, because that polygon
keeps its input geometry.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Pampas_SAM2_zoom1_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Pampas_SAM2_zoom1_example.webp" alt="Pampas — zoom 1, centre pivots: Google and TESSERA clusters before and after SAM 2" width="800" loading="lazy"></a>

### Zoom 2: centre pivots, south-east

A 4 km square around the south-east pivot group (8 of the 29 pivots), most
of them bare in October 2024. One of them is part of a TESSERA polygon more
than twice its area (115.1 ha, 2.2 times) and two are part of such Google
polygons (2,639.3 ha, 45 times, and 125.0 ha, 2.4 times). Three more TESSERA
pivots are in polygons of 105.2 ha (two pivots share it) and 109.7 ha, 1.8 to
1.9 times their area, and one more Google pivot is in a 94.4 ha polygon,
1.7 times its area. All of these polygons are over 50 ha and stay unrefined.
Refining every polygon left 1 TESSERA pivot (44.9 % covered) and 1 Google
pivot (5.6 %) of the 8 less than half covered.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Pampas_SAM2_zoom2_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Pampas_SAM2_zoom2_example.webp" alt="Pampas — zoom 2, south-east centre pivots: Google and TESSERA clusters before and after SAM 2" width="800" loading="lazy"></a>

### Zoom 3: large merged polygons

This 4 km square (inside the study area, clear of zooms 1 and 2, with no
checked pivot) is the one with the largest combined TESSERA and Google share
of its area in crop-filter polygons over 200 ha; the rule picks the same
square on the 1.1 layers as on the 1.0.1 layers. 53.0 % of it is in four
TESSERA polygons of 248.0 to 415.1 ha, each holding 7 to 12 Delineate-Anything
Sentinel-2 fields of 5 ha or more (counting fields with at least 80 % of their
area inside it), and 77.4 % in three Google polygons over 200 ha. The largest
Google one, 2,639.3 ha, holds 88 such fields and also reaches into zoom 2. Both
embeddings draw one outline around blocks of bare paddocks whose boundaries
show in the composite. These polygons are over 50 ha, so the split variant
leaves them unrefined. How many such merges the TESSERA clusters make depends
on the k-means solution: the whole 1.1 TESSERA crop layer has 1 polygon over
500 ha (568.7 ha, in zoom 1), against 10 in 1.0.0 (see
[Engines](user-guide/engines.md#embedding-clustering-embedding)).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Pampas_SAM2_zoom3_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Pampas_SAM2_zoom3_example.webp" alt="Pampas — zoom 3, large merged polygons: Google and TESSERA clusters before and after SAM 2" width="800" loading="lazy"></a>

### Compared with the 0.1.x README image

The agribound 0.1.x README showed this example as a wide screenshot (top
left): about 23 × 18 km, rotated, with outlines about 63 m wide, of the 0.1.x
layer with SAM 2 on three TESSERA dimensions and polygons over 50 ha
unrefined (its caption said SAM 2 on Sentinel-2). Drawn in the same frame with
the same line width, the 0.1.x layer (top right, 3,089 fields) and the two
1.1.0 split layers (bottom: the gallery layer, and SAM 2 on three TESSERA
dimensions as in 0.1.x; 2,154 fields each) look much alike: at this scale
single pixels, small fragments and merged fields are hard to see, which is why
the 4 km zooms above look rougher than the 0.1.x image. By their polygon
sizes, the 1.1 TESSERA clusters are closer to the 0.1.x ones than the 1.0.0
clusters were. In the crop-filter layers, 15.5 % (0.1.x), 38.4 % (1.0.0) and
18.2 % (1.1) of the area is in polygons over 200 ha, and the largest polygon
is 566, 1,447 and 568 ha (EPSG:6933); 7, 10 and 6 of the 29 pivots are part of
a polygon more than twice their area. The cluster labels were not compared
pixel by pixel (see
[Engines](user-guide/engines.md#embedding-clustering-embedding)). The 1.1
TESSERA raster was read from the reconverted store and lies within about 1 m
of the Sentinel-2 composite; the rasters of agribound 1.0.0 and 1.0.1 lay
9–10 m east or south of it in most tiles (see
[Satellite Sources](user-guide/satellite-sources.md#embeddings)). Unlike the
other images, it has no inset, and its footer names the imagery but not the
models; the 1.1.0 layers use the models and versions of the entries above. The
image is rendered by `tools/make_gallery_pampas_0.1x.py`.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Pampas_0.1x_comparison_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Pampas_0.1x_comparison_example.webp" alt="Pampas — the 0.1.x README image next to the 0.1.x and 1.1.0 layers drawn in the same frame" width="800" loading="lazy"></a>

---

## India, West Bengal — FTW on Sentinel-2 vs Delineate-Anything on SPOT-Pan

**Example 02** · Label-free · Nadia District, West Bengal, between Nabadwip
and Krishnanagar (study area 88.35–88.50° E, 23.35–23.50° N; about 95 % in
Nadia, the north-western corner west of the Bhagirathi in Purba Bardhaman) ·
min. area 100 m² · the same 1 km window in both panels (in Krishnagar-I
block, centre 23.39° N, 88.42° E; the square with the most Delineate-Anything
polygons), four years apart.

- Left: FTW on Sentinel-2 2024, with two season windows, 4 May–3 Jul 2024
  (6 images, shown) and 25 Nov 2024–24 Jan 2025 (18 images); Dynamic World
  2024 crop filter. The window is about 100 × 100 Sentinel-2 pixels.
- Right: Delineate Anything v2 on SPOT 6/7 panchromatic (1.5 m), 2020
  (4 images; restricted SPOT access); Dynamic World 2020 crop filter.

FTW produced 63,485 polygons. Of these, 62,110 fall inside the study area (the
composite covers its bounding box), 49,800 pass the 100 m² filter, and the
crop filter kept 20,765 of those 49,800. Their median area is 0.04 ha, about
four Sentinel-2 pixels. In this window they do not follow the field edges
visible in the SPOT-Pan image. Delineate Anything v2 produced 78,472 polygons
(76,782 inside the study area, 76,547 after the 100 m² filter); the crop
filter kept 54,938 of 76,547, with a median area of 0.12 ha. Both layers come
from the development run: FTW's is identical to the 1.0.0 output, and
Delineate-Anything's changed with the tile stitching of 1.1. There is no
reference data here; neither output was evaluated. The example also clusters
Google Satellite Embedding and TESSERA embeddings (not shown; Google with
k = 5 chosen automatically, TESSERA with a fixed k = 8): the crop filter kept
1,465 of 9,628 Google and 11,937 of 82,178 TESSERA polygons (3,850.2 and
5,823.2 ha). The TESSERA polygons differ from 1.0.1's (11,722 of 79,712)
because the TESSERA v1 store was converted again upstream (see
[Satellite Sources](user-guide/satellite-sources.md#embeddings)).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/India_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/India_example.webp" alt="India — FTW on Sentinel-2 vs Delineate-Anything on SPOT-Pan" width="800" loading="lazy"></a>

---

## Global South — Delineate-Anything v2 on SPOT 6/7 panchromatic, six landscapes

**Example 22** · Label-free · six study areas, each a 3 km square in its UTM
zone (6 km in western Bahia, where the pivots are about 1 km across) · min.
area 100 m² · no crop filter on the maps (see below) · in each panel, the
densest of the 2 km squares whose centres lie on a 1 km grid (four per 3 km
area), at about 2 m per pixel of the full-size image; in western Bahia, the
whole 6 km study area (about 6 m per pixel).

Delineate Anything v2 as released on SPOT 6/7 panchromatic 1.5 m composites
(restricted SPOT access), the median of one calendar year's scenes with at
most 15 % cloud cover. The years were chosen so that 2 to 6 scenes cover each
square and none covers only part of it (image counts in the footer include
selected scenes with no pixels in the square).

| # | Study area | Year | Fields | Median (ha) | Crop filter kept |
|---|---|---|---:|---:|---:|
| 1 | Cauvery Delta, Tamil Nadu, India | 2018 | 3,056 | 0.19 | 2,908 |
| 2 | Hetao irrigation district, Inner Mongolia, China | 2021 | 4,611 | 0.11 | 3,793 |
| 3 | Agrelo, Mendoza, Argentina | 2019 | 354 | 1.33 | 333 |
| 4 | Mwea irrigation scheme, Kenya | 2020 | 1,404 | 0.41 | 1,319 |
| 5 | Nile Delta near Tanta, Egypt | 2020 | 1,465 | 0.21 | 1,356 |
| 6 | Luís Eduardo Magalhães, western Bahia, Brazil | 2018 | 310 | 1.27 | 46 |

The fields range from small paddies and strip plots to vineyard blocks and
centre pivots about 1 km across. The model follows the bunds of the Cauvery
Delta paddies and the Mwea tenant strips and outlines the Mendoza vineyard
blocks along their windbreaks. In Hetao it draws many polygons smaller than
the canal-grid blocks.

In the Nile Delta, one polygon of 15.0 ha at the top left of the square covers
a block of strip plots that the image shows clearly. It is made by the tile
stitching: one tile returned a single low-confidence outline of the whole
block (0.29), the stitching merged it with 18 tile-cut pieces from
neighbouring tiles into one polygon, which took the highest confidence among
them (0.70, that of a strip of 0.16 ha), and the 76 detections nested inside
it were then dropped as duplicates of it. The block outline ended 2 px short
of its tile's edge, which the stitching of 1.1 counts as cut by the tile
(within 4 px); with the 1.0 rule it counted as whole, and the strips were
kept. A polygon of 99.2 ha, mostly west of the square (8 % of it inside),
covers another block of strip plots between two drains in the same way: it is
merged from 114 detections of 21 tiles, and the 1.0 stitching made it too
(about 105 ha).

In western Bahia, of the 13 pivots wholly inside the panel the model splits
eight (four into quarters, two into many pieces along their sector lines, one
into rings and one into halves along an airstrip) and outlines five whole; its
two largest polygons there (161 and 151 ha) are blocks between the pivots.
10.1 % of the Bahia boundary lies within 2 m of an edge of the 768 m tiles
(about 2 % by chance) and 9.0 % runs along one. In the engine's own output
(before post-processing), 94 % of the boundary that runs along a tile edge
lies on polygons that are not round, and more than half on 36 blocks of 20 ha
or more between and around the pivots; the sector lines through the pivots
are not on the tile grid. Such a block is detected in some tiles and not in
the next ones (where most of its detections score below the confidence
threshold of 0.15), so its outline ends on the edge of the last tile that
detected it.

There is no reference data here, so these are outlines, not accuracy. The
crop filter (Dynamic World of each year, mean `crops` probability at least
0.3) runs as a separate step in the example. It kept 46 of the 310 Bahia
polygons: over the other 264 the mean Dynamic World crops probability is
below 0.3. The maps therefore show the unfiltered polygons. The inset is a
world map with the six study areas numbered as the panels.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Global_South_SPOT_Pan_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Global_South_SPOT_Pan_example.webp" alt="Global South — Delineate-Anything v2 on SPOT 6/7 panchromatic in six farming landscapes" width="800" loading="lazy"></a>

---

## Tree crops — Delineate-Anything v2 on SPOT 6/7 panchromatic, four landscapes

**Example 23** · Released weights (no fine-tuning) · four study areas, squares
of 3 to 5 km in their UTM zones · min. area 2,500 m² · no crop filter on the
maps (see below) · in each panel, the square with the most reference polygons
(2 km at Twifo Praso and Madera, 1 km at Oro and Úbeda).

Delineate Anything v2 as released on SPOT 6/7 panchromatic 1.5 m composites
(restricted SPOT access) of one year per study area (1, 2, 11 and 2 scenes),
with the reference polygons in cyan: the RSPO GeoRSPO concession maps
(member-declared, published in September 2026) for an industrial oil palm
estate and for oil palm smallholders, the DWR / Land IQ crop map of water year
2022 for almond and pistachio orchards, and SIGPAC parcels ("recintos") of the
2025 campaign for olive groves. Fields match at IoU ≥ 0.5. Precision counts
only the predictions that overlap a reference polygon, because the RSPO maps
of Twifo Praso and Oro do not map every field in their squares; in Madera and
Úbeda, whose references map all fields, the scores are those of the tree-crop
fields.

Only at Twifo Praso and Oro has the model seen none of the fields. Delineate
Anything v2 was trained on FBIS-73M, which has no patches in Ghana or Papua New
Guinea, but whose training patches cover the Madera square, with field labels
that match the DWR / Land IQ polygons (121 of the 122 reference fields appear
as training labels at IoU ≥ 0.5), and 87 % of the Úbeda square, with labels
that follow SIGPAC (166 of the 225 recintos). This was checked against the
public FBIS-73M patch list, images and labels (patch footprints and label
polygons); the dataset does not name its sources.

| # | Study area | Year | Reference (tree crops) | Fields | Recall | Precision | F1 | Crop filter kept: default / tree crops |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | Twifo Praso, Ghana: oil palm estate | 2020 | 55 blocks, median 39.9 ha | 122 | 0.49 | 0.23 | 0.32 | 0 / 122 |
| 2 | Oro Province, Papua New Guinea: oil palm smallholders | 2021 | 302 parcels, median 1.5 ha | 1 | 0.003 | 1 of 1 | 0.007 | 0 / 1 |
| 3 | Madera County, California: almonds and pistachios | 2022 | 112 of 122 fields, median 16.6 ha | 145 | 0.79 | 0.87 | 0.83 | 144 / 144 |
| 4 | Úbeda, Jaén, Spain: olive groves | 2023 | 212 of 225 recintos, median 3.2 ha | 57 | 0.04 | 0.17 | 0.06 | 16 / 24 |

Where the trees grow in blocks separated by roads, the released model finds
the blocks: 79 % of the Madera orchards (88 of 112), and 49 % of the Twifo
estate blocks (27 of 55). At Twifo it also splits 24 of the 55 blocks (split
rate 0.44), so most of its other polygons lie on reference blocks. 54 of the
55 blocks are longer than the engine's 768 m inference tiles, and the
stitching of 1.1 joins pieces that meet at a tile edge only when they pass its
seam test (6 joins in this output); 12.7 % of the boundary length still lies
within 2 m of a tile edge (2.1 % by chance; 14.0 % with agribound 1.0.1's
stitching). At Madera the same test joins whole orchard blocks: three of the
six joins in this output merge two separate blocks. The test looks for a
boundary in a strip 2 px wide along the tile edge, in the tile centred on that
edge; there the road between the two blocks runs up to 21 m off the edge, at
2–3° to it, and one block's outline crosses the edge, so the test finds no
boundary. The five fields involved lose their match, and the tree-crop F1 is
0.83, against 0.86 with agribound 1.0.1's stitching (88 and 93 matched
orchards); the seams of these joins lie outside the windows of the tree-crop
images. Where the parcels are stands of trees among other trees, the model
finds almost nothing: one polygon at Oro, where the parcels show the planting
grid of the palms. At Úbeda the recintos follow cadastral lines that cross
uniform groves; after dropping recintos under 2,500 m² and joining touching
recintos of the same land use (46 tree-crop groves of 57) recall is 0.22 and
precision 0.30.

Not shown: the example also runs Delineate Anything v2 on Sentinel-1 (F1 0.11,
0, 0.80 and 0.06 at the four sites), and, with `--s2-single-date` (the 1.1.0
re-run), on the Sentinel-2 composite of the clearest single date of each year
(F1 0.10, 0, 0.84 and 0.06, against 0.24, 0, 0.80 and 0.16 on the year's
median composite).

The crop filter runs as a separate step in the example, twice. The default
rule (Dynamic World outside the conterminous US, NLCD inside) removes every
polygon of panels 1 and 2 (Twifo Praso and Oro): it also removes all 55 Twifo
and all 302 Oro reference polygons, and keeps 46 of the 225 Úbeda recintos.
`lulc_tree_crops=True` (Dynamic World `crops` + `trees`) keeps all these
polygons and all the Twifo and Oro reference polygons, and 108 of the Úbeda
recintos. In Madera the filter uses NLCD, whose cultivated-crops class
includes orchards, and keeps all 122 reference fields with either rule. The
maps therefore show the unfiltered polygons. The inset is a world map with the
four study areas numbered as the panels.

Thanks to Jacob Abramowitz (The University of Alabama in Huntsville), who
asked about tree crops and pointed to the RSPO concession maps, and who raised
the question of date mixing in tropical composites and shared his SNIC script
(see
[Dates in a composite](user-guide/satellite-sources.md#dates-in-a-composite)).
The RSPO polygons are member-declared, are provided "for informational and
illustrative communication purposes only" (RSPO Disclaimer for Map
Publication) and are not redistributed here; the example downloads them from
RSPO.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Tree_Crops_SPOT_Pan_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Tree_Crops_SPOT_Pan_example.webp" alt="Tree crops — Delineate-Anything v2 on SPOT 6/7 panchromatic: oil palm estate, oil palm smallholders, almond and pistachio orchards, olive groves" width="800" loading="lazy"></a>

---

## Twifo Praso, Ghana — oil palm estate blocks from SPOT-Pan 1.5 m to Landsat PAN 15 m

**Example 23** · Reference: the estate's RSPO GeoRSPO blocks (55 with their
representative point in the 5 km square, median 39.9 ha) · 2020 for every
source · min. area 2,500 m² · no crop filter on the maps · the 2 km square
with the most reference blocks.

| Panel | Fields | Recall | Precision | F1 | Mean IoU | Boundary F1 (10 m) | Merged | Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Delineate-Anything v2, SPOT-Pan | 122 | 0.49 | 0.23 | 0.32 | 0.74 | 0.56 | 0.05 | 0.44 |
| + SAM 2 | 122 | 0.42 | 0.22 | 0.29 | 0.80 | 0.46 | 0.02 | 0.44 |
| Fine-tuned (NORPALM), SPOT-Pan | 104 | 0.53 | 0.33 | 0.40 | 0.84 | 0.57 | 0.38 | 0.33 |
| Delineate-Anything v2, Sentinel-2 | 70 | 0.27 | 0.22 | 0.24 | 0.73 | 0.42 | 0.15 | 0.25 |
| Delineate-Anything v2, Landsat PAN | 17 | 0.05 | 0.18 | 0.08 | 0.88 | 0.15 | 0.02 | 0.05 |
| FTW, Sentinel-2 | 0 | 0 | - | 0 | - | 0 | - | - |
| Google embedding, k = 20 | 1,185 | 0 | 0 | 0 | - | 0.17 | 0.07 | 0.76 |
| TESSERA, k = 20 | 702 | 0 | 0 | 0 | - | 0.17 | 0.02 | 0.11 |

Precision counts the predictions that overlap a reference block (the square
also holds a village and forest outside the estate). A block is merged when
the prediction covering most of it also covers a quarter of another block,
and split when two predictions each cover a tenth of it. The blocks are
separated by gaps of about 4 m along the roads, sharp on SPOT-Pan and less
than a pixel wide on Sentinel-2 and Landsat PAN. The SPOT-Pan composite of
2020 is a single scene. On it the released model follows the roads, but 24 of
the 55 blocks come out in pieces (split rate 0.44). Many are cut along the
straight edges of the 768 m inference tiles (every 384 m), because a block of
about 1 km is longer than a tile: the stitching made 6 seam joins across tile
edges here (1 of the 4 joins that could be checked against the blocks joins
two blocks), and 12.7 % of the boundary still lies within 2 m of a tile
edge. SAM 2 refined 100 of the 122 polygons (5 were below the size rule, and
17 masks covered less than half of their polygon); it raised the mean IoU of
the matched blocks and lowered recall. The fine-tuned model was trained on the
109 blocks of the NORPALM estate, 69 km from the Twifo square (centres 81 km
apart), on SPOT-Pan of the same year (185 training and 23 validation chips,
20 epochs, `yolo_lr0=1e-4`; validation mask mAP50 0.58 after the 20th epoch);
on Twifo it finds 29 blocks, against the released model's 27, and merges 21 of
the 55 with a neighbour. The Landsat 8 PAN composite (3 images) shows the road
grid, but the model draws only 17 polygons. FTW predicted no field pixels from
its two Sentinel-2 windows (2019-12-16 to 2020-02-14 and 2020-07-16 to
2020-09-14, 12 and 7 images, set to clear months because oil palm has no crop
season), and the embedding clusters are land-cover segments that match no
block; the TESSERA segments come from the 1.1.0 re-run (702; the development
run made 703). Against the estate outline (the blocks joined), the 122 SPOT-Pan
polygons of the released model cover 73 % of the estate's area in the square
and put 3.5 % of their area (60.9 ha) outside it; the 115 with their
representative point in the estate put 0.2 % of their area outside it. The
default crop filter removes every polygon of every panel;
`lulc_tree_crops=True` keeps all but three Google and three TESSERA segments.
The embedding panels are drawn on the Sentinel-2 composite of the same year.
Not shown: on the clearest single Sentinel-2 date of 2020 (2020-08-04), the
released model draws 8 polygons (F1 0.10, against 0.24 on the year's median).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Tree_Crops_Twifo_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Tree_Crops_Twifo_example.webp" alt="Twifo Praso, Ghana — oil palm estate blocks: Delineate-Anything v2 on SPOT-Pan, Sentinel-2 and Landsat PAN, SAM 2, a fine-tuned model, FTW and embeddings" width="800" loading="lazy"></a>

---

## Oro Province, Papua New Guinea — oil palm smallholder parcels, released and fine-tuned models

**Example 23** · Reference: RSPO GeoRSPO parcels of the Higaturu scheme (302
with their representative point in the 3 km square, median 1.5 ha; they
cover about half of it) · 2021 · min. area 2,500 m² · no crop filter on the
maps · the 1.5 km square with the most reference parcels, clipped to the
imagery (1.50 × 1.43 km).

The released Delineate Anything v2 made one detection on the SPOT-Pan
composite of the square (81 inference tiles), and none on Sentinel-2 or
Landsat PAN; FTW predicted no field pixels, and Sentinel-1 (not shown) gave
one unmatched polygon. Fine-tuned on parcels of the same scheme in a 6 km
square 13.6 km from the Oro square (centres 19.5 km apart; SPOT-Pan of March
2021; 71 training chips, which hold parts of 437 of the 564 parcels in the
square, and 6 validation chips; 20 epochs, `yolo_lr0=1e-4`), it drew 272
polygons and matched 126 of the 302 parcels (recall 0.42; precision 0.49, or
0.50 among the predictions with at least half of their area within 20 m of a
parcel; mean IoU of the matches 0.66; boundary F1 0.53 at 10 m). It outlines
whole plantings and joins neighbouring parcels (merge rate 0.49). 3.2 % of its
boundary length lies within 2 m of a tile edge (1.9 % by chance), and the long
straight north-south and east-west lines along tile edges that the 1.0 image
showed are not in this one. The fine-tuning recipe and the tile stitching that
made this output were changed after those lines were found here (see
[Fine-tuning](user-guide/fine-tuning.md#engine-specific-settings)), so Oro is
not an independent test of them; the fine-tuned model was also added to the
example after the released model's result here, and kept after its own Oro
score was seen. Its training parcels, digitised by the same RSPO member, share
no parcel with the Oro reference. The Google Satellite Embedding clusters (814
segments) match 8 % of the parcels. Precision cannot count the palm stands
that the reference leaves out: about half of the square is not mapped. Both
crop-filter rules were applied: the default one keeps none of the 302
reference parcels, `lulc_tree_crops=True` keeps all of them, and 271 of the
fine-tuned model's 272 polygons. The embedding panel is drawn on the
Sentinel-2 composite of the same year.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Tree_Crops_Oro_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Tree_Crops_Oro_example.webp" alt="Oro Province, Papua New Guinea — oil palm smallholder parcels: released and fine-tuned Delineate-Anything v2 on SPOT-Pan, and Google embedding clusters" width="800" loading="lazy"></a>

---

## Madera County, California — almond and pistachio orchards from NAIP 1 m to Landsat PAN 15 m

**Example 23** · Not unseen fields: Delineate Anything v2's training data
(FBIS-73M) cover this square, with labels that match 121 of its 122 reference
fields · Reference: the DWR / Land IQ crop map of water year 2022
(122 fields with their representative point in the 5 km square, 112 of them
almond or pistachio orchards, median 16.6 ha) · 2022 for every source · scores
of the orchards · min. area 2,500 m² · no crop filter on the maps (the default
filter, NLCD here, removes at most 2 polygons of a layer shown) · the 2.5 km
square with the most reference fields, clipped to the imagery
(2.40 × 2.50 km).

| Panel | Fields | Recall | Precision | F1 | Mean IoU | Boundary F1 (10 m) | Merged |
|---|---:|---:|---:|---:|---:|---:|---:|
| Delineate-Anything v2, NAIP | 135 | 0.69 | 0.81 | 0.74 | 0.87 | 0.84 | 0.43 |
| Delineate-Anything v2, SPOT-Pan | 145 | 0.79 | 0.87 | 0.83 | 0.90 | 0.89 | 0.31 |
| Delineate-Anything v2, Sentinel-2 | 113 | 0.71 | 0.93 | 0.80 | 0.85 | 0.71 | 0.26 |
| Delineate-Anything v2, Landsat PAN | 120 | 0.73 | 0.89 | 0.80 | 0.81 | 0.52 | 0.30 |
| FTW, Sentinel-2 | 110 | 0.54 | 0.74 | 0.63 | 0.73 | 0.12 | 0.38 |
| Google embedding, k = 20 | 322 | 0.54 | 0.21 | 0.30 | 0.72 | 0.27 | 0.39 |

The orchards are blocks of 16.6 ha (median) separated by roads, and every
approach shown finds most of them: Delineate Anything v2 at all four
resolutions, even 15 m Landsat PAN, which is outside its 0.25-10 m training
range (Landsat 8 and 9, 29 images), and FTW. SAM 2 on the SPOT-Pan polygons
(not shown) refined 121 of the 145 and left F1 at 0.83; TESSERA clusters (not
shown) matched 52 of the 112 orchards (recall 0.46, F1 0.15), and Delineate
Anything v2 on Sentinel-1 (not shown, 103 polygons) scored 0.80. The reference
splits some blocks along narrow tracks (the diagonal cyan lines) that no
output draws, so a prediction over such a block covers parts of two reference
fields (merge rates of 0.26 to 0.43 for Delineate-Anything). Three of the six
tile seam joins of the SPOT-Pan output merge two separate orchard blocks (see
the first tree-crop entry), and one of the two joins of the NAIP output does;
none of these seams lies in this window. The embedding panel is drawn on the
Sentinel-2 composite of the same year.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Tree_Crops_Madera_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Tree_Crops_Madera_example.webp" alt="Madera County, California — almond and pistachio orchards: Delineate-Anything v2 on NAIP, SPOT-Pan, Sentinel-2 and Landsat PAN, FTW and Google embedding clusters" width="800" loading="lazy"></a>

---

## Tree crops — Delineate-Anything v2, released and fine-tuned, against DINOv3 fine-tuned

**Example 23** · SPOT 6/7 panchromatic 1.5 m · fine-tuned models trained near
each study area, never on the evaluated square · in each row, the square with
the most reference polygons (2 km at Twifo Praso and Madera, 1 km at Oro and
Úbeda), the same for the three models.

Delineate Anything v2 predicts each field as an object. DINOv3 (a ViT-L/16
backbone pre-trained on SAT-493M satellite imagery, with a DPT head) labels
each pixel as background, field interior or field boundary, and the fields
are its interior regions, grown back by the boundary band. It has no
published field-boundary weights, so it was fine-tuned only; Delineate
Anything v2 is shown as released and fine-tuned. Both were fine-tuned on the
same labels and SPOT-Pan composite of a training area: the NORPALM estate for
Twifo Praso, Higaturu scheme parcels for Oro, a 5 km square of DWR / Land IQ
fields near Cressey, 41.9 km from the Madera square (centres 48.3 km apart),
and a 4 km square of SIGPAC recintos around Ibros, 10.8 km from the Úbeda
square (centres 16.1 km apart). The last two squares were chosen by a fixed
rule from the labels and the imagery before any model was trained on them.
Each model used agribound 1.1's recipe with at most 20 epochs, and
`yolo_lr0=1e-4` for Delineate-Anything (the default is 0.002), set on the
NORPALM validation chips with the earlier recipe: Delineate-Anything on 512 px
chips half a chip apart, with a random 20 % of chip-size cells for validation
and the chips that straddle both splits dropped (185, 71, 80 and 50 training
chips at the four training areas); DINOv3 with full fine-tuning, class weights,
a 4 px boundary band and the checkpoint with the best validation mean IoU
(238, 94, 134 and 79 training chips). These recipes, and the tile stitching
that made these outputs, were changed after the 1.0 results at these sites:
DINOv3's after its merging at Twifo Praso, Delineate-Anything's after the
tile-edge lines at Oro. These two sites are therefore not an independent test
of them; the Delineate-Anything fine-tuning for Oro was also added after the
released model's Oro result and kept after its own Oro score was seen (see the
Oro entry).

| Study area | Model | Fields | Recall | Precision | F1 | Mean IoU | Boundary F1 (10 m) | Merged | Split |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Twifo Praso | Delineate-Anything v2, released | 122 | 0.49 | 0.23 | 0.32 | 0.74 | 0.56 | 0.05 | 0.44 |
| | Delineate-Anything v2, fine-tuned | 104 | 0.53 | 0.33 | 0.40 | 0.84 | 0.57 | 0.38 | 0.33 |
| | DINOv3, fine-tuned | 102 | 0.65 | 0.40 | 0.50 | 0.85 | 0.59 | 0.13 | 0.27 |
| 2 Oro | Delineate-Anything v2, released | 1 | 0.003 | 1 of 1 | 0.007 | 0.78 | 0.007 | 0.003 | 0 |
| | Delineate-Anything v2, fine-tuned | 272 | 0.42 | 0.49 | 0.45 | 0.66 | 0.53 | 0.49 | 0.25 |
| | DINOv3, fine-tuned | 255 | 0.46 | 0.56 | 0.50 | 0.68 | 0.62 | 0.44 | 0.10 |
| 3 Madera | Delineate-Anything v2, released | 145 | 0.79 | 0.87 | 0.83 | 0.90 | 0.89 | 0.31 | 0.02 |
| | Delineate-Anything v2, fine-tuned | 178 | 0.84 | 0.73 | 0.78 | 0.87 | 0.85 | 0.25 | 0.11 |
| | DINOv3, fine-tuned | 142 | 0.87 | 0.96 | 0.91 | 0.88 | 0.87 | 0.21 | 0.01 |
| 4 Úbeda (recintos) | Delineate-Anything v2, released | 57 | 0.04 | 0.17 | 0.06 | 0.73 | 0.34 | 0.81 | 0.08 |
| | Delineate-Anything v2, fine-tuned | 866 | 0.34 | 0.09 | 0.14 | 0.65 | 0.50 | 0.33 | 0.56 |
| | DINOv3, fine-tuned | 91 | 0.12 | 0.36 | 0.18 | 0.65 | 0.47 | 0.88 | 0.08 |

Scores as in the first tree-crop entry (precision among the predictions that
overlap a reference polygon; at Madera and Úbeda, the tree-crop fields). At
Madera and Úbeda, Delineate Anything v2 has seen most of the evaluated fields
in its own training data (FBIS-73M covers the Madera square and 87 % of the
Úbeda square); DINOv3, whose backbone was pre-trained without labels, has not.
At Twifo Praso and Oro neither has.

DINOv3 has the highest F1 of the three models at all four sites: 0.50 at Twifo
Praso, 0.50 at Oro, 0.91 at Madera and 0.18 on the Úbeda recintos, against
0.40, 0.45, 0.83 and 0.14 for the better Delineate Anything v2 model of each
site (fine-tuned at Twifo Praso, Oro and Úbeda, released at Madera). On the
Úbeda groves dissolved by land use it scores 0.24, against 0.25 for the
released Delineate Anything v2. Its merge rates are 0.13 at Twifo Praso, 0.21
at Madera, 0.44 at Oro and 0.88 at Úbeda, where the recintos follow cadastral
lines across uniform groves (with agribound 1.0.1's recipe they were 0.80 to
1.00 at every site). Its boundary class covers 3.5 % (NORPALM) to 30.6 %
(Ibros) of the pixels of its training masks and 10.6 % (Madera) to 22.4 %
(Oro) of its predictions. On its own training areas, which are in-sample and
not shown, DINOv3 draws 77 polygons for the 109 NORPALM blocks (the largest of
6,415.4 ha) and 407 for the 451 Cressey fields; there the fine-tuned Delineate
Anything v2 draws one polygon of 7,634.7 ha, which the stitching chained from
about 860 tile-cut detections.

Fine-tuning Delineate Anything v2 raises F1 at Twifo Praso (0.32 to 0.40), Oro
(0.007 to 0.45) and Úbeda (0.06 to 0.14) and lowers it at Madera (0.83 to
0.78), whose training orchards are smaller (median 3.7 ha against 16.6 ha). At
Úbeda the fine-tuned model draws 866 polygons with a precision of 0.09; its
validation mask mAP50 on the Ibros chips was 0.000 after the first epoch and
0.271 after the 20th.

The Ibros training composite is one scene (8 July 2023) and the Cressey one is
the 8 May 2022 scene over 87 % of its square, while the Madera composite is a
median of 11 scenes, 9 of them from January to March. On the 8 May scene of
the Madera square the three models score F1 0.86 (released), 0.83
(fine-tuned) and 0.92 (DINOv3): the fine-tuned Delineate Anything v2 stays
below the released one there too, so the difference in season does not
explain that gap.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Tree_Crops_DINOv3_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Tree_Crops_DINOv3_example.webp" alt="Tree crops — Delineate-Anything v2 released and fine-tuned, and DINOv3 fine-tuned, on SPOT 6/7 panchromatic at four study areas" width="800" loading="lazy"></a>

---

## Nevada, Diamond Valley — centre pivots against the DRI field boundaries

**Example 24** · Diamond Valley (hydrographic basin 153, Eureka County,
Nevada) · the 5 km evaluation square chosen by a fixed rule from the
inventory alone (the square with the most DRI-mapped fields of the basin) ·
reference: DRI field boundaries (Nevada, 2026-08-20), Desert Research
Institute: the 74 DRI-mapped fields with their representative point in the
square (median 48.2 ha; none under 2,500 m²) · min. area 2,500 m²; no crop
filter on the maps · window: the evaluation square.

Each panel is drawn on its own input: NAIP 1 m of 2022 (4 images) at top, the
SPOT 6/7 6 m median of April to October 2023 (1 image) at bottom left, and the
Sentinel-1 10 m median of April to October 2024 (36 descending acquisitions)
at bottom right: 2024, or the latest year of a source without 2024 imagery
over the square. Top left is Delineate Anything v2 as released on NAIP; the
other panels are the runs of example 24 with the highest F1 among its other
NAIP runs, its SPOT runs and its Sentinel-1 run, counting the `sam-auto` runs
of `--sam-auto` and not the "-opt" runs of `--options` (entry 24e). SAM 2.1
automatic masks (`sam-auto`) take the NAIP slot, with F1 0.387 against 0.380
for DINOv3 fine-tuned, and the SPOT slot, with 0.589 against 0.500 for DINOv3
fine-tuned (the rule ranks the point estimates; in both slots the two 95 %
intervals overlap: 0.328–0.430 against 0.335–0.428, and 0.533–0.634 against
0.407–0.587).
`sam-auto` uses no field labels and no fine-tuning (SAM 2.1 as released). Its
settings per source were chosen on example 24's two training squares (by
mean F1 among 72 settings per source; settings within 0.01 of the highest
counted as tied, then the fewest SAM passes per km² and the higher boundary
F1 decided), so its scores there are in-sample. One of them lies 5.0 km
south of this square: this square is held out from that choice, not from the
landscape. Its two layers
here are, coordinate for coordinate, those of the SAM evaluation (the SAM 2.1
entry below). One-to-one matching at IoU ≥ 0.5 against all 74 fields, with
95 % bootstrap intervals (200 resamples of the reference fields, stratified
by irrigation system):

| Panel | Fields | Precision | Recall | F1 [95 %] | Boundary F1 (10 m) |
|---|---:|---:|---:|---|---:|
| NAIP 2022, Delineate Anything v2 as released | 216 | 0.17 | 0.50 | 0.26 [0.21–0.30] | 0.57 |
| NAIP 2022, `sam-auto` | 112 | 0.32 | 0.49 | 0.39 [0.33–0.43] | 0.58 |
| SPOT 6/7 2023, `sam-auto` | 55 | 0.69 | 0.51 | 0.59 [0.53–0.63] | 0.58 |
| Sentinel-1 2024, Delineate Anything v2 as released | 82 | 0.33 | 0.36 | 0.35 [0.26–0.42] | 0.39 |

The square holds two kinds of mapped field. 39 of the 74 are centre pivots
(median 49.9 ha, about 800 m across), and in 2022 they were the only fields
irrigated (IrrMapper share of the field at least 0.5). The other 35 (16
sprinkler, 18 with uncontrolled flood or no apparent irrigation equipment and
1 with controlled flood) were not irrigated in 2022. The NAIP runs find the
pivots (35 of 39 for Delineate Anything v2 as released, 36 of 39 for
`sam-auto`: recall 0.90 and 0.92) and miss most of the other fields (2 and
none of the 35); on SPOT 6/7, `sam-auto` finds 38 of the 39 pivots and none
of the other fields. Of the 216 polygons of Delineate Anything v2 on NAIP, 67
have their representative point in a pivot, 71 in one of the other 35 fields
and 78 on ground the inventory does not map (192 ha); of `sam-auto`'s 112 on
NAIP, 47, 26 and 39 (240 ha), and of its 55 on SPOT 6/7, 41, 2 and 12
(56 ha). None of the polygons on unmapped ground matches a field. On SPOT 6/7,
`sam-auto` merges more fields with a neighbour than Delineate Anything v2 as
released (merge rate 0.34 against 0.04: the share of fields whose main
polygon also covers at least a quarter of another field). The irrigated-only
entry below scores the same runs against the irrigated fields only.

The reference also limits what any outline can score here. Of the 74 fields,
38 are circles (intersection over union at least 0.9 with a circle of equal
area; one of the 39 pivots counted by irrigation type, 49.8 ha, scores 0.85),
and 19 are the land around a pivot mapped as one field, 13 of them in several
pieces. For 4 of these 19 the largest piece is 27–36 % of the field, so no
outline of a single piece can reach the intersection over union of 0.5 that a
match needs: outlines that do not join separate pieces can match at most 70 of
the 74 fields (recall 0.95; F1 0.97 with no false positive). The other 15 need
an outline that follows the field around its pivot or, for the 9 in several
pieces, an outline of their largest piece (51–88 % of the field). These counts
come from the reference alone. In Mason Valley 1 of the 187 fields lies around
a pivot (2 pieces, the larger 81 % of the field), and no field is out of reach
of a single outline.

The top-left panel shows straight cuts through some pivots along the edges of
the engine's 512 m tiles (the pivots are larger than a tile): 10.2 % of its
boundary lies within 2 m of a tile edge (the tiles overlap by half, so an edge
runs every 256 m), against 3.0 % for the same grid shifted off the edges.
DINOv3 fine-tuned on SPOT 6/7 (F1 0.50; the bottom left of entry 24e, not
drawn here) puts its polygons inside the field edges (the median outline
lies 35 m inside the reference outline, and 93 % of the matched pairs more
than 10 m inside), so its boundary F1 is 0.07. The model predicts a boundary
band two to three times as wide as the band it was trained on (8–12 px
against 4 px), inference grows the field interiors back by 4 px only, and at
6 m the rest exceeds the 10 m tolerance of boundary F1. A first test of three growth
rules on this square, set before they were run, found none that met its bar,
and the default growth is unchanged in 1.1.0. A later rule, opt-in in 1.1.0
(`dilate_interior_px="half_band"`: grow by half the band measured on the
prediction), passed a pre-registered development pilot and a test on three
other Nevada squares (entry 24e). On this raster it grows the interiors by
12 px, and boundary F1 rises to 0.37 and F1 to 0.55. That result had been
known before the test, from the first test's sweep of the growth on this
raster, so it is shown here, not tested.

On the Sentinel-1 image the pivots are grey on blue-violet ground; the red
outlines are faint there at preview scale and clear in the full-resolution
image. The inventory records no imagery date per field; NAIP and SPOT are 2
and 1 years older than the Sentinel-1 composite, and fields that changed in
between count as errors. The nearest patch of the public lists of FBIS-73M,
Delineate Anything v2's training data, lies 285.8 km from this square, in
California (see the note at the top of this page). Entry
24e shows this square with the opt-in options of 1.1.0 that passed their
tests, and the SAM 2.1 entry compares `sam-auto` with Delineate Anything v2
as released on its SPOT 6/7 composite.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Nevada_Diamond_Valley_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Nevada_Diamond_Valley_example.webp" alt="Diamond Valley, Nevada — centre pivots: Delineate-Anything v2 as released on NAIP and Sentinel-1, and SAM 2.1 automatic masks (sam-auto) on NAIP and SPOT 6/7, against the DRI field boundaries" width="800" loading="lazy"></a>

---

## Nevada, Mason Valley — flood and sprinkler irrigation against the DRI field boundaries

**Example 24** · Mason Valley (basin 108, Lyon County, Nevada) · the
evaluation square chosen by the same rule · reference: the 187 DRI-mapped
fields with their representative point in the square (median 7.7 ha; 105
sprinkler, 39 controlled flood, 30 uncontrolled flood or no apparent
irrigation equipment, 9 retired, 4 pivot) · min. area 2,500 m²; no crop filter
on the maps · window: the evaluation square.

Each panel is drawn on its own input: NAIP of 2022 (2 images) at top, the SPOT
6/7 median of April to October 2023 (6 images) at bottom left and the
Sentinel-1 median of April to October 2024 (34 ascending acquisitions) at
bottom right. The panels are chosen as in Diamond Valley: Delineate Anything
v2 as released on NAIP, then the best other NAIP run (DINOv3 fine-tuned, on a
training square 16.0 km north, which 179 DRI fields reach into; `sam-auto`
on NAIP scores 0.62), the best SPOT run (`sam-auto`) and the Sentinel-1 run. On
SPOT, `sam-auto`'s F1 (0.60) is higher than that of Delineate Anything v2 as
released (0.51; the two 95 % intervals overlap, and the paired 1 km-block
interval of the SAM evaluation, −0.003 to +0.161, just includes 0) and those
of the models fine-tuned on the training square (0.43 for Delineate Anything
v2 and 0.45 for DINOv3, which learned from 17 and 7 training chips).
`sam-auto`'s settings were chosen on that training square and Diamond
Valley's, so this square is held out from that choice, not from the
landscape.

| Panel | Fields | Precision | Recall | F1 [95 %] | Boundary F1 (10 m) |
|---|---:|---:|---:|---|---:|
| NAIP 2022, Delineate Anything v2 as released | 296 | 0.46 | 0.73 | 0.57 [0.51–0.62] | 0.73 |
| NAIP 2022, DINOv3 fine-tuned | 217 | 0.69 | 0.80 | 0.74 [0.69–0.78] | 0.76 |
| SPOT 6/7 2023, `sam-auto` | 125 | 0.74 | 0.50 | 0.60 [0.54–0.65] | 0.64 |
| Sentinel-1 2024, Delineate Anything v2 as released | 113 | 0.60 | 0.36 | 0.45 [0.39–0.51] | 0.42 |

The fields are smaller than in Diamond Valley (median 7.7 ha against 48.2 ha),
and 148 of the 187 were irrigated in 2022. Delineate Anything v2 as released
finds 115 of those 148 on NAIP (recall 0.78) and 22 of the 39 that were not
irrigated (0.56). `sam-auto` on SPOT 6/7 finds few of the smallest fields:
recall 0.04 (1 of 27) under 2 ha, 0.41 (34 of 83) at 2–10 ha and 0.75 (58 of
77) at 10–50 ha. Sentinel-1 scores below the NAIP runs here (0.45 against 0.57
and 0.74) and above Delineate Anything v2's NAIP run in Diamond Valley (0.35
against 0.26), where 39 of the 74 fields are large pivots. The caveats of
Diamond Valley about the reference and the years apply here too. The nearest
patch of FBIS-73M's public lists lies 38.0 km from this square, a labelled
training patch in California.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Nevada_Mason_Valley_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Nevada_Mason_Valley_example.webp" alt="Mason Valley, Nevada — flood- and sprinkler-irrigated fields: Delineate-Anything v2 as released and fine-tuned DINOv3 on NAIP, SAM 2.1 automatic masks (sam-auto) on SPOT 6/7 and Delineate-Anything v2 on Sentinel-1, against the DRI field boundaries" width="800" loading="lazy"></a>

---

## Nevada, irrigated fields only — Diamond Valley and Mason Valley

**Example 24** (irrigated-only score; entries 24c and 24d) · the runs,
panels, squares and backgrounds of the two entries above, scored against the
DRI-mapped fields irrigated in the year of each panel's imagery.

Every run of example 24 is also scored against the reference fields that were
irrigated in the year of its imagery (IrrMapper share of the field at least
0.5): in Diamond Valley 39 of the 74 in 2022, 40 in 2023 and 34 in 2024; in
Mason Valley 148 of the 187, 173 and 159. The other fields of the square are
left out of the reference (hatched), and a prediction whose representative
point lies in one of them is not scored (outlined in grey on a lighter grey
casing and filled with a translucent grey; red outlines are drawn over grey
ones where they coincide). Every other prediction is scored as in the
all-fields score, with the same matching (one-to-one at IoU ≥ 0.5) and
bootstrap. This is the `ignore` option of `agribound.evaluate` (see
[Evaluation](user-guide/evaluation.md#ignoring-predictions-on-excluded-ground)).
The panel titles give the predictions scored and the polygons of the output.

The score is not a score on irrigated land alone: predictions on ground the
inventory does not map (corners between pivots, rangeland, roads) still count
as false positives, and a prediction that covers an irrigated field and a
left-out one is scored or not by where its representative point lies. Its
recall can differ from that of the irrigated fields in the all-fields score,
which comes from one matching of all fields and all predictions; in these
runs the two are equal, so only precision and F1 change. Ranked by this
score, the runs of each basin give the same four panels.

| Basin, panel (year of the imagery) | Reference fields scored + left out | Predictions scored | Matched | F1, irrigated only [95 %] | F1, all fields |
|---|---|---|---:|---|---:|
| Diamond Valley, NAIP, Delineate Anything v2 (2022) | 39 + 35 | 145 of 216 | 35 | 0.38 [0.32–0.44] | 0.26 |
| Diamond Valley, NAIP, `sam-auto` (2022) | 39 + 35 | 86 of 112 | 36 | 0.58 [0.50–0.64] | 0.39 |
| Diamond Valley, SPOT 6/7, `sam-auto` (2023) | 40 + 34 | 53 of 55 | 38 | 0.82 [0.74–0.87] | 0.59 |
| Diamond Valley, Sentinel-1, Delineate Anything v2 (2024) | 34 + 40 | 55 of 82 | 26 | 0.58 [0.48–0.67] | 0.35 |
| Mason Valley, NAIP, Delineate Anything v2 (2022) | 148 + 39 | 250 of 296 | 115 | 0.58 [0.52–0.63] | 0.57 |
| Mason Valley, NAIP, DINOv3 fine-tuned (2022) | 148 + 39 | 182 of 217 | 127 | 0.77 [0.73–0.81] | 0.74 |
| Mason Valley, SPOT 6/7, `sam-auto` (2023) | 173 + 14 | 123 of 125 | 91 | 0.61 [0.54–0.67] | 0.60 |
| Mason Valley, Sentinel-1, Delineate Anything v2 (2024) | 159 + 28 | 106 of 113 | 62 | 0.47 [0.39–0.53] | 0.45 |

In Diamond Valley the fields left out are the 35 fields that are not pivots in
2022, 34 of them in 2023 (one was irrigated that year) and, in 2024, those 35
and 5 pivots that were not irrigated. The predictions that drop out of the
score are those on these fields (71 of the 216 Delineate-Anything polygons on
NAIP and 26 of the 112 `sam-auto` polygons), and the F1 of the two NAIP runs
rises from 0.26 to 0.38 and from 0.39 to 0.58. Their precision stays low (0.24
and 0.42): the 78 and 39 predictions on unmapped ground stay in the score as
false positives. Among Diamond Valley's panels the highest F1 in this score
is 0.82, for `sam-auto` on SPOT 6/7, which matches 38 of the 40 fields
irrigated in 2023 with 53 scored predictions; among all of example 24's
Diamond Valley runs it is 0.87 [0.81–0.92], for `sam-auto` on Landsat 2024
(not drawn: the panels have no Landsat slot), which matches 33 of the 34
fields irrigated in 2024 with 42 scored predictions. In Mason Valley most
fields were irrigated in each year, so few predictions drop out (46, 35, 2 and
7 in the four panels) and F1 changes by +0.01 to +0.03.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Nevada_Diamond_Valley_irrigated_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Nevada_Diamond_Valley_irrigated_example.webp" alt="Diamond Valley, Nevada — irrigated fields only: the fields not irrigated in each panel's year hatched and the predictions on them not scored" width="800" loading="lazy"></a>

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Nevada_Mason_Valley_irrigated_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Nevada_Mason_Valley_irrigated_example.webp" alt="Mason Valley, Nevada — irrigated fields only: the fields not irrigated in each panel's year hatched and the predictions on them not scored" width="800" loading="lazy"></a>

---

## Nevada, Diamond Valley — opt-in options of 1.1.0, before and after

**Example 24** (the "-opt" runs of `--options`; entry 24e) · Diamond Valley's
evaluation square, reference and backgrounds as in the Diamond Valley entry
above (the 74 DRI-mapped fields of the DRI field boundaries (Nevada,
2026-08-20), Desert Research Institute) · min. area 2,500 m² · window: the
evaluation square · panels fixed in PREREG-DV3 before the test.

The left panels are Delineate Anything v2 as released on NAIP 1 m of 2022 (4
images), the top left of the Diamond Valley entry, and DINOv3 fine-tuned on
the SPOT 6/7 6 m median of April to October 2023 (1 image), the SPOT run
fixed for this panel before the test (in the Diamond Valley entry `sam-auto`
now scores higher on SPOT 6/7 and takes that slot). The right panels are the
same runs with the opt-in options of agribound 1.1.0 that passed a
pre-registered test on three other Nevada squares (below), applied in this
order:

- **top right:** the NAIP image read on a 3 m grid
  (`engine_params={"processing_gsd_m": 3}`), centre pivots cut into pieces
  rejoined and round polygons snapped to circles (`"pivots": "snap"`; here 25
  pivots, 5 of them rejoined from 11 pieces), then the crop filter (Annual
  NLCD 2022; a polygon is kept when at least 30 % of it is cultivated crops or
  pasture/hay);
- **bottom right:** the field interiors grown by half the boundary band
  measured on the model's own prediction (`"dilate_interior_px": "half_band"`:
  12 px on this raster, against the 4 px the model was trained with), pivots
  snapped (17, none rejoined), then the crop filter (Annual NLCD 2023).

| Panel | Polygons | Precision | Recall | F1 [95 %] | F1, irrigated only [95 %] | Boundary F1 (10 m) |
|---|---:|---:|---:|---|---|---:|
| NAIP 2022, Delineate Anything v2 as released | 216 | 0.17 | 0.50 | 0.26 [0.21–0.30] | 0.38 [0.32–0.44] | 0.57 |
| NAIP 2022, 3 m grid, pivots snapped, crop filter | 87 | 0.39 | 0.46 | 0.42 [0.34–0.50] | 0.61 [0.49–0.74] | 0.61 |
| *Change, right − left [1 km blocks, 95 %]* | | | | +0.17 [+0.09 to +0.24] | +0.23 [+0.13 to +0.34] | +0.03 [−0.01 to +0.08] |
| SPOT 6/7 2023, DINOv3 fine-tuned as in 1.1 | 46 | 0.65 | 0.41 | 0.50 [0.41–0.59] | 0.71 [0.59–0.85] | 0.07 |
| SPOT 6/7 2023, half-band growth, pivots snapped, crop filter | 41 | 0.78 | 0.43 | 0.56 [0.48–0.64] | 0.79 [0.68–0.89] | 0.40 |
| *Change, right − left [1 km blocks, 95 %]* | | | | +0.06 [−0.02 to +0.13] | +0.08 [−0.03 to +0.18] | +0.33 [+0.28 to +0.38] |

The panels' intervals are those of the entries above (200 resamples of the
reference fields); the intervals of the changes come from a paired bootstrap
of 1 km blocks of the square (2,000 resamples), which also resamples the
predictions on ground the inventory does not map.

**The tests.** Each option was tested on its own, by rules fixed before the
test (PREREG-DV3, 2026-10-12), on three 5 km squares in other Nevada basins
(Fish Lake Valley, Middle Reese River Valley and Pine Forest Valley), chosen
from the inventory and image metadata alone before any model ran on them,
51–161 km from the nearest square used before. The four run types fixed for
the test (Delineate Anything v2 as released on NAIP and on Sentinel-1, DINOv3
fine-tuned on NAIP and on SPOT 6/7) ran there with this example's settings
and models. The DINOv3 models learned from Diamond Valley's training square,
so on those squares they ran outside the basin they learned from. The nearest
patches of FBIS-73M's public lists, Delineate Anything v2's training data,
lie 17.9, 235.2 and 129.9 km from the three squares, in California.

- *Pivot snapping* (`pivots="snap"`; also tested on this square and on Mason
  Valley's) lowered F1 on none of the 20 runs tested and raised it on 6, where
  it rejoined pivots cut into pieces: by +0.003 to +0.018 for Delineate
  Anything v2 on NAIP and by +0.005 and +0.015 for Sentinel-1. F1 did not
  change on the other 14, including all four Mason Valley runs. Boundary F1
  changed by −0.015 to +0.030; it fell slightly on the inset DINOv3 outlines
  on SPOT, because a circle fitted to an inset outline stays inset.
- *Half-band growth* (`dilate_interior_px="half_band"`) raised boundary F1 on
  5 of the 6 DINOv3 runs, the fewest its rule allowed: on SPOT 6/7 from
  0.07–0.09 to 0.28–0.33 (growth 10–13 px), on NAIP by +0.010 and +0.017, and
  on one NAIP run it fell by 0.003. F1 changed by −0.002 to +0.052. A
  development pilot chose this rule, over one that sets the growth from the
  fine-tune's validation chips, by the number of runs it improved; the other
  rule had placed that pilot's SPOT outlines closer to the reference.
- *The 3 m grid* (`processing_gsd_m=3`, Delineate Anything v2 on NAIP; also
  tested on Mason Valley's square) raised F1 by +0.04, +0.15 and +0.13 on the
  three squares and by +0.10 in Mason Valley; the interval of the first
  includes 0. It gives fewer, larger polygons (24–39 % fewer, median area
  1.5–5.1 times larger on the five squares), and the gain is mostly
  precision: the matched fields changed by −4 to +6.
- *The crop filter* raised F1 on all 12 optical runs tested, by +0.006 to
  +0.137, through precision: it also removes matched fields that Annual NLCD
  does not map as crops or pasture in the imagery year (6–10 on the NAIP runs
  of one square, where boundary F1 fell by 0.07–0.09). It is not applied to
  Sentinel-1, where a development pilot found that it removes pivots fallow in
  the radar's year.

The 3 m grid and the crop filter each failed a first development pilot and
passed a second before this test, so the test was a third look at both.

**On this square.** Only the pivot snapping was tested blind here (F1 +0.018
on NAIP, unchanged on the other three panels). The other effects were partly
known before the test: example 26 had scored Delineate Anything v2 on NAIP at
3 m, but with `merge_tile_pieces=False`, at F1 0.30 against 0.26 (the option
itself gives 0.31); an earlier sweep of the DINOv3 growth on this SPOT raster
had given, at 12 px, the boundary F1 of 0.37 and the F1 of 0.55 that the
option gives; and the crop-filtered scores follow from example 24's
`summary.csv`. The right-hand panels combine options that were tested one at
a time, so they are reported results, not tests. Image-blind baselines are
low: a grid of squares of the left panels' median polygon areas scores F1
0.00 (NAIP) and 0.12 (SPOT, cells of about 600 m; mean of 36 offsets), and
the reference shifted by 397 m scores 0. The boundary F1 of the DINOv3 SPOT
outlines without the growth option (0.07) is below that of their blind grid
(0.09); with it (0.37, before the pivot step and the crop filter) it is well
above (0.08). PREREG-DV3 was amended on 2026-10-13 for a defect of its check
step that changed no layer or score (see the note at the top of this page).

**Mason Valley** (no figure; example 24's "-opt" runs on its evaluation
square: Delineate Anything v2 and DINOv3 on NAIP, Delineate Anything v2 on
SPOT 6/7 and on Sentinel-1): F1 rises by +0.11 for Delineate Anything v2 on
NAIP (0.57 to 0.68: 3 m grid and crop filter) and by +0.03 for DINOv3 on NAIP
(0.74 to 0.77: the growth option measures 4 px there, the trained width, and
the pivot step finds no pivot, so the change is the crop filter's), falls by
0.004 for Delineate Anything v2 on SPOT (0.513 to 0.510, the crop filter;
interval −0.018 to +0.010) and does not change for Sentinel-1 (its pivot
option finds no pivot).

**The three test squares** (the same combined runs; reported, not tested):
Delineate Anything v2 on NAIP goes from F1 0.48, 0.31 and 0.28 to 0.64, 0.50
and 0.46; DINOv3 on NAIP from 0.38, 0.33 and 0.18 to 0.52, 0.40 and 0.23;
DINOv3 on SPOT from 0.47, 0.47 and 0.48 to 0.61, 0.53 and 0.51, with boundary
F1 from 0.07–0.09 to 0.31–0.39; Sentinel-1 from 0.30, 0.39 and 0.38 to 0.30,
0.41 and 0.38. On the second square the crop filter lowers the NAIP runs'
boundary F1 (Delineate Anything v2: 0.61 to 0.51) while their F1 rises. The
SAM 2.1 entry below sets `sam-auto` against these "-opt" runs, as a secondary
comparison.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Nevada_Diamond_Valley_options_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Nevada_Diamond_Valley_options_example.webp" alt="Diamond Valley, Nevada — Delineate-Anything v2 on NAIP and fine-tuned DINOv3 on SPOT 6/7 before and after the opt-in options of 1.1.0 (3 m grid, half-band growth, pivot snapping, crop filter), against the DRI field boundaries" width="800" loading="lazy"></a>

---

## 47 US states — Delineate-Anything v2 on NAIP against the OpenET field boundaries

**Example 25** · one 5 km square in each of the 47 states of the OpenET
agricultural field boundaries v2.1.1 (the conterminous US except Alabama),
chosen by a fixed rule before any model ran (the square with the most farmed
fields, CDL 2024) · Delineate Anything v2 as released on NAIP 1 m of the
latest year whose images cover the square (2021 in 1 state, 2022 in 16, 2023
in 30) · min. area 2,500 m²; no crop filter · reference: the square's fields
of at least 2,500 m².

Each marker is the centre of a state's square, drawn far larger than the
square and coloured by object F1 (one-to-one at IoU ≥ 0.5); its shape gives
the reference: USDA FSA Common Land Units (CLU), a state or federal agency
inventory, or DRI's field boundaries (in New Mexico the DRI/NMOSE field
boundaries, V18; in Nevada the DRI field boundaries of 2026-08-20; in Oregon
the DRI field delineation of the OpenET file). F1 has a median of 0.31 over
the 47 squares and ranges from 0.01 (Montana) to 0.70 (California); pooled
over the squares, precision is 0.30, recall 0.32 and F1 0.31. The same model
on Sentinel-2 2024 (not shown) pools to F1 0.29 (median 0.33). One square per
state shows how the released model does across the country's farming
landscapes; it is not an accuracy estimate for the states, and the rule
favours squares of many small farmed fields. The same squares and composites
were the NAIP and Sentinel-2 endpoints of the pre-registered evaluation of
`sam-auto` (the SAM 2.1 entry below); example 25's `--sam-auto` runs the same
engine on them.

- The references map different things: a CLU is a farm-programme unit, which
  can hold several fields that the image separates, and the reverse. Pooled
  NAIP F1 is 0.27 in the 35 squares whose reference is all CLU and 0.42 in the
  9 squares with none (0.37 without California).
- Per-square F1 follows the elongation of the reference fields (Spearman
  ρ = −0.49) more than their size (ρ = 0.09).
- In Montana and North Dakota (0.01 and 0.04) most reference fields are CLU
  strips about 50 m wide. The 2005 NAIP shows crop and fallow strips of that
  width and orientation in both squares; by the 2023 NAIP both are farmed in
  blocks of many strips, and few of the CLU lines show an edge in the image
  (the Montana square in 2005 and 2023: below). The edges along the CLU lines
  are the weakest of the 47 squares, and the best variant of the tile
  stitching that was tested reached F1 0.03 and 0.05.
- California (0.70): FBIS-73M, Delineate Anything v2's training data, holds the
  California DWR / Land IQ crop maps as labels, and 128 of its labelled
  training patches intersect this square, so the model is not held out here
  (whether its labels match these very fields was not checked). No other
  square holds an FBIS-73M patch: the nearest labelled training patch lies
  15.1 km from the Arizona square, 102.0 km from the Nevada square and
  169.3 km or more from each of the other 44, and the nearest unlabelled
  patches 7.7 and 10.5 km from the Arizona and Wyoming squares (public lists
  as published, checked on 2026-10-13; see the note at the top of this page).

Reference: OpenET agricultural field boundaries, version 2.1.1, Desert
Research Institute for the OpenET Consortium (2026), with the field boundaries
of their source agencies; New Mexico: DRI/NMOSE field boundaries (V18), Desert
Research Institute and New Mexico Office of the State Engineer; Nevada: DRI
field boundaries (Nevada, 2026-08-20), Desert Research Institute.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Multistate_OpenET_overview_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Multistate_OpenET_overview_example.webp" alt="47 US states — object F1 of Delineate-Anything v2 on NAIP in one 5 km square per state against the OpenET field boundaries" width="800" loading="lazy"></a>

---

## 47 US states — four squares, Delineate-Anything v2 and crop-history zones on NAIP

**Example 25** (with `--crop-sequence`, the 1.1.0 re-run) · squares 1–4 of
the map above, in increasing NAIP F1: the lowest, the two that cut the ranking
into thirds, the highest (ranks 1, 16, 32 and 47 of 47) · in each, the 2.5 km
square with the most reference fields among nine candidates half a side apart
in the 5 km square · NAIP 1 m: 2023 for squares 1–3 (5, 4 and 3 images), 2022
for square 4 (2 images) · reference: OpenET field boundaries v2.1.1 (1 and 2:
USDA FSA Common Land Units; 3: Idaho Department of Water Resources, Eastern
Snake River Plain irrigated lands, 8 % of the fields CLU; 4: California
Department of Water Resources, Statewide Crop Mapping, with Land IQ).

Left: Delineate Anything v2 as released. Right: crop-history zones of the USDA
Cropland Data Layer 2016–2023 (`cdl-sequence` and `crop-sequence`, made the
way USDA NASS makes its Crop Sequence Boundaries), drawn on the same NAIP
image; their input is the 30 m CDL stack. The zones are not field outlines:
neighbouring fields with the same crop sequence share a zone, and narrow or
small fields are lost.

| Square | Delineate Anything v2: fields, F1 | CDL zones: zones, F1 |
|---|---|---|
| 1 Glacier County, Montana | 83, 0.01 | 288, 0.004 |
| 2 Hampshire County, Massachusetts | 589, 0.27 | 280, 0.06 |
| 3 Twin Falls County, Idaho | 728, 0.37 | 293, 0.11 |
| 4 Fresno County, California | 761, 0.70 | 603, 0.07 |

(Counts: the whole output; F1: the whole 5 km square, one-to-one at
IoU ≥ 0.5.) Over the 47 squares the zones' F1 has a median of 0.08 (from 0.004
in Montana to 0.26 in Indiana), against 0.31 for Delineate Anything v2 on NAIP.
In Montana most reference fields are north-south CLU strips about 50 m wide,
with the width and orientation of the crop and fallow strips of the 2005 NAIP.
By the 2023 NAIP the square is farmed in blocks of many strips, so a
delineation of exactly what the image shows would score at most about F1
0.01-0.08 against this reference (next image): the score measures how far
the reference is from the ground more than how well the model delineates the
fields that exist. Delineate Anything v2 draws the blocks, as large polygons
across the strips: 20 of its 83 polygons are larger than one 512 m tile
(26.2 ha), up to 359.6 ha, and five of these reach into this window. The
stitching chains them from many tile-cut pieces (with the merging of tile
pieces off, the largest polygon is 78.4 ha), but they do not set the score:
no variant of the stitching reached an F1 above 0.03 there. The square is
the most subdivided in Montana, because the rule favours many small farmed
fields: 86 % of its fields are strips, against 23 % of Montana's fields.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Multistate_OpenET_squares_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Multistate_OpenET_squares_example.webp" alt="47 US states — four squares from the lowest to the highest F1: Delineate-Anything v2 and CDL crop-history zones on NAIP against the OpenET field boundaries" width="800" loading="lazy"></a>

---

## 47 US states — Montana: the CLU on NAIP 2005 and 2023

**Example 25** (no model output) · square 1 of the map above, Glacier County,
Montana, the lowest NAIP F1 of the 47 (0.01) · the whole 5 km square · left:
NAIP 2005 (six images of 28 July 2005, red, green and blue only; each 2 m
pixel is the mean of the native 1 m pixels), exported from Earth Engine by
`tools/fetch_naip_history.py` because agribound's `naip` source reads only
4-band images; right: the NAIP 2023 composite that Delineate Anything v2 read
in example 25 (1 m, five images of 22 July 2023) · reference: the square's
684 OpenET fields of at least 2,500 m², all USDA FSA Common Land Units (CLU),
the fields example 25 scores against.

Of the 684 fields, 590 (86 %) are north-south strips, a median 49 m wide and
401 m long. The NAIP of 2004 and 2005 shows crop and fallow strips of that
width and orientation, but the CLU lines do not follow their edges closely:
they lie a median 7-8 m from the strip edges, and 15-19 m in one block in the
north-west, where they run inside the strips. By 2023 the square is farmed in
blocks of many strips. A test that compares the image on the two sides of
each line, with no model, finds contrast across 10 % of the 1,163 CLU lines
that two fields share on the 2023 composite, where 5 % would show it by
chance; 3 of the 684 fields have a visible edge on every shared side, and 3
more share no side with another field. A delineation of exactly what the
2023 image shows would therefore score at most about F1 0.01-0.08 against
this reference, over the seven contrast thresholds tested (0.017 at the
threshold that gives the 10 % above); Delineate Anything v2 scores 0.01 on
this composite (0.02 on Sentinel-2 2024).

The square is not typical of Montana: example 25's rule takes the cell with
the most farmed fields, and in Montana that is the most subdivided cell, with
86 % of its fields strips, against 23 % of Montana's fields and 6 % of its
field area. On two development squares 11 and 16 km away (edge to edge; the
next cells by the same rule; in-sample for any choice made on them), the
variants tried scored pooled F1 0.017-0.037 against the CLU: Sentinel-2
composites of four parts of the 2024 season, SAM 2.1 automatic masks on NAIP
and on Sentinel-2 (a prototype, with settings not tuned on Montana), and
Delineate Anything v2 fine-tuned on a third square's CLU. Example 25's own
NAIP and Sentinel-2 settings scored 0.031 and 0.041 there; the only variant
above its comparator, SAM 2.1 on NAIP, matched 3 more of the 966 fields than
Delineate Anything v2. The same released model scored 0.21 and 0.11 on the two
squares' NAIP of 2005, against 0.015 and 0.052 on their NAIP of 2023. A grid
of north-south strips of the median width and length of each square's
reference fields (49 and 52 m by 401 m), drawn without any image, scored a
mean F1 of 0.083 and 0.044 over 36 positions of the grid (0.23 and 0.12 at the
best position; a check made after the scores were known). North Dakota's
square (Adams County, F1 0.04) shows the same change in the test without a
model: contrast across 62 % of its shared CLU lines in the 2005 NAIP and 17 %
in 2023. These measurements and the development-square runs were made for this
release on 2026-10-12, after the scores were known; they changed nothing in
example 25.

Reference: OpenET agricultural field boundaries, version 2.1.1, Desert
Research Institute for the OpenET Consortium (2026), here the Common Land
Units of the USDA Farm Service Agency. NAIP: USDA Farm Service Agency.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Multistate_OpenET_Montana_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Multistate_OpenET_Montana_example.webp" alt="Montana, Glacier County — the CLU reference fields of example 25 on NAIP 2005 and on the NAIP 2023 composite, no model output" width="800" loading="lazy"></a>

---

## SAM 2.1 automatic masks against Delineate-Anything v2

**Pre-registered SAM evaluation** (PREREG-SAM; panels fixed by a rule before
the evaluation ran) · top: example 24's Diamond Valley evaluation square
(Nevada), the SPOT 6/7 6 m median of April to October 2023 (1 image;
restricted SPOT access) · bottom: example 25's Tennessee square (Crockett
County), the square at the median of the 47 paired NAIP F1 differences, NAIP
1 m of 18 October 2023 (2 images) · left: Delineate Anything v2 as released;
right: `sam-auto` (SAM 2.1 large, with the engine's own setting for the
source) · each pair on the same composite, with the example's own
configuration and no crop filter · window: the 5 km evaluation square ·
reference: the DRI field boundaries (top, 74 fields) and the OpenET field
boundaries v2.1.1 (bottom, USDA FSA Common Land Units, 515 fields of at least
2,500 m²).

`sam-auto` runs Meta's SAM 2.1 automatic mask generation, a grid of point
prompts, on large overlapping tiles, and turns the masks of all tiles into
one set of non-overlapping polygons. It uses no field labels and no
fine-tuning, and it outlines every region it sees as one object, not only
fields. The scores below are each example's own (one-to-one matching at
IoU ≥ 0.5); the paired difference is `sam-auto` minus Delineate Anything v2,
with a 95 % interval over 2,000 resamples of the square's 25 1 km blocks.

| Panel | Fields | Precision | Recall | F1 | Paired difference [95 %, 1 km blocks] |
|---|---:|---:|---:|---:|---|
| SPOT 6/7 2023, Diamond Valley, Delineate Anything v2 | 133 | 0.10 | 0.18 | 0.13 | |
| SPOT 6/7 2023, Diamond Valley, `sam-auto` | 55 | 0.69 | 0.51 | 0.59 | +0.46 [+0.35, +0.58] |
| NAIP 2023, Tennessee, Delineate Anything v2 | 335 | 0.44 | 0.28 | 0.34 | |
| NAIP 2023, Tennessee, `sam-auto` | 317 | 0.48 | 0.29 | 0.36 | +0.02 [−0.01, +0.05] |

The bottom row is the typical NAIP square: 23 of the 47 differences are
smaller and 23 larger. The top row shows where `sam-auto` gained most on
SPOT 6/7: there Delineate Anything v2 as released matches 13 of the 74
fields, and `sam-auto` 38.

Results on the pre-registered endpoints, each a mean of paired F1
differences (`sam-auto` minus Delineate Anything v2 as released):

- **NAIP.** Over the 46 squares of example 25 without California, F1 was
  0.017 higher (95 % interval 0.004 to 0.030), so better by the rule. Both
  engines are near F1 0.3 there (mean 0.31 against 0.29), and `sam-auto`
  scored higher on 32 squares and lower on 14. California is left out because
  Delineate Anything v2's training data hold its DWR labels.
- **Sentinel-2.** On the same squares F1 was 0.152 lower (0.127 to 0.178
  lower), so it did not transfer. `sam-auto` drew about a quarter as many
  polygons (3,465 against 13,549) and found fewer than one field in ten (mean
  recall 0.09 against 0.26).
- **SPOT 6/7.** Over four Nevada and New Mexico areas (Diamond Valley and
  Mason Valley, and the Lea County and San Juan County boxes of examples 12
  and 20; median fields 1.1–48 ha) the mean was +0.26, with every area
  positive; in Mason Valley the 1 km-block interval only just includes 0
  (lower end −0.003, within Monte Carlo error of 0). As pre-registered, the
  Nevada pairs are without the crop filter and the New Mexico pairs with it;
  with all four unfiltered the mean is +0.21 (a secondary result).
- **Landsat** (8/9; 7/8/9 on the San Juan box). Over the same areas, with the
  same filter states, the mean was +0.13 (+0.09 with all four unfiltered, a
  secondary result). Mason Valley is −0.04. At 30 m Delineate Anything v2
  runs outside its training range (0.25–10 m).
- **Four areas** give no interval and support no claim that `sam-auto` is
  better on SPOT 6/7 or Landsat.
- **Secondary results** (none decides anything): on Sentinel-2, −0.40 and
  −0.36 on example 27's two Quebec squares and −0.11 to −0.23 on example 28's
  three New York squares; on NAIP in New York, +0.02 to +0.04, every 1
  km-block interval including 0 (the Black Dirt's lower end, −0.0015, at the
  edge of 0).
- **Amendment.** Mason Valley's pairs had been scored against a crop-filter
  copy of its reference (176 of its 187 fields). Both Nevada areas were
  scored again on 2026-10-13, after the first scores and the endpoints had
  been read: an amendment of the pre-registration, recorded as a deviation
  from it. Diamond Valley's scores, the top row's, did not change; Mason
  Valley's new scores were partly known before they were computed, since its
  scores against the copy had been seen.

Against example 24's "-opt" runs (entry 24e), a secondary comparison that
decides nothing: on example 24's two evaluation squares, `sam-auto` was
scored against the six NAIP and SPOT "-opt" runs, in two filter states. Before
their crop filter the F1 difference (`sam-auto` minus "-opt") ranged from
−0.119 to +0.083, with one of six 1 km-block intervals below 0 (Mason Valley,
NAIP, against DINOv3) and none above (Mason Valley SPOT's lower bound,
−0.003, is within Monte Carlo error of 0); with both layers crop-filtered it
ranged from −0.086 to +0.092, with three intervals above 0 (Diamond Valley
NAIP against Delineate Anything v2, Diamond Valley SPOT against DINOv3, Mason
Valley SPOT against Delineate Anything v2) and one below (Mason Valley NAIP
against DINOv3). Both scores of 10 of the 12 pairs had been seen before, and
the two squares lie 5.0 and 16.0 km from the training squares on which
`sam-auto`'s settings were chosen and DINOv3 was fine-tuned, so these are
reported results, not tests.

Cost, measured back to back on the same composites of two New Mexico boxes
of 415 and 496 km² (composite grids of about 430 and 526 km²; Apple M2 Max,
MPS; the pipeline's delineate step, model loading included): 6.1 times
Delineate Anything v2's GPU time on NAIP, 6.4 on SPOT 6/7, 17.0 on
Sentinel-2 and 20.7 on Landsat. On example 25's 5 km squares, against
Delineate Anything v2's runs in other sessions, the ratio was 3.1 on NAIP and
2.1 on Sentinel-2.

Limits:

- **Non-field land.** SAM outlines it too: on NAIP over the 46 squares, 21 %
  of `sam-auto`'s polygons overlap no reference field, against 22 % of
  Delineate Anything v2's. The crop filter, on by default, removes only part
  of them.
- **Merged fields.** On SPOT 6/7, Sentinel-2 and Landsat, `sam-auto` merged
  neighbouring fields more often than Delineate Anything v2 in all 12 area
  pairs of Nevada and New Mexico; in Diamond Valley on SPOT 6/7, 34 % of the
  fields were merged with a neighbour, against 4 %. On the 46 squares it
  merged less often on average, on NAIP (41 % against 53 %) and on
  Sentinel-2 (46 % against 53 %; more often on 17 of the 46).
- **Small fields.** On Sentinel-2 `sam-auto` is behind on 45 of the 46
  squares, whose median fields are 0.6–5.2 ha, but within them its difference
  does not follow field size (post hoc). On SPOT 6/7 the development pilot's
  loss on small fields was not borne out: on the San Juan box (median field
  1.1 ha) `sam-auto` was ahead by 0.06.
- **Large objects.** At the default tile overlap an object larger than about
  half a tile is cut in every tile and dropped: about 1.0 km on NAIP and
  2.5–3.7 km at 6–30 m.
- **Devices.** These runs used Apple MPS. A run gives the same polygons on the
  same device with the same versions, and masks differ between devices (MPS
  against CPU: mask IoU 0.59–0.97 for the same fields, measured in the SAM 2
  refinement stage).
- **Where the settings were chosen.** The settings come from example 24's two
  training squares (in-sample there), 5.0 km south of the Diamond Valley
  evaluation square and 16.0 km north of Mason Valley's, so the top row is
  held out from that choice but not from the landscape. Delineate Anything
  v2's scores on both squares were known before the evaluation (examples 24
  and 25's own runs). FBIS-73M's public lists hold no patch within 800 km of
  the Tennessee square.

References: DRI field boundaries (Nevada, 2026-08-20), Desert Research
Institute; OpenET agricultural field boundaries, version 2.1.1, Desert
Research Institute for the OpenET Consortium (2026), with the field
boundaries of their source agencies (in Tennessee the Common Land Units of
the USDA Farm Service Agency); and, for the endpoints' New Mexico areas, the
DRI/NMOSE field boundaries (V18), Desert Research Institute and New Mexico
Office of the State Engineer. NAIP: USDA Farm Service Agency. SAM 2.1: Meta,
Apache-2.0; the `sam2` 1.1.0 package on PyPI is a third-party build whose 62
code, configuration and licence files equal Meta's commit `2b90b9f`. No panel
uses Sentinel data.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/SAM2_automatic_masks_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/SAM2_automatic_masks_example.webp" alt="SAM 2.1 automatic masks (sam-auto) against Delineate-Anything v2 as released: Diamond Valley, Nevada, on SPOT 6/7 2023 and Crockett County, Tennessee, on NAIP 2023" width="800" loading="lazy"></a>

---

## Radar — Sentinel-1, Umbra and Capella at three held-out sites

**Example 26** · Delineate Anything v2 as released on radar composites,
beside NAIP read with the same settings · three held-out sites, at which the
Umbra and Capella recipe and these settings were fixed before they were
scored: 1 Texas A&M AgriLife farm, Brazos Bottom, Burleson County, Texas;
2 Intel Ohio One site, Licking County, Ohio; 3 Orleans County, Vermont · no
crop filter · reference: OpenET field boundaries v2.1.1 (here USDA FSA Common
Land Units) of at least 2,500 m² in each site's evaluation area, except those
whose crop code of the radar year is a non-agricultural class of the Cropland
Data Layer: 136, 77 and 112 fields · in each panel, the 2 km square with the
most reference fields.

Per site, left to right: Sentinel-1 (C-band, 10 m pixels, about 20 m
resolution; the median of one orbit pass over the site's window: 12, 13 and 13
acquisitions), the Umbra (sites 1 and 2: 6 and 4 collects) or Capella
composite (site 3: 2 collects) of the same window (X-band, 1 m, a median of
co-registered collects), and NAIP of the year closest to the radar (2022, 2023
and 2023). Delineate Anything v2 reads the Umbra, Capella and NAIP composites
with the settings agribound documents for Umbra and Capella
(`processing_gsd_m=3`, `merge_tile_pieces=False`), and Sentinel-1 with up to
1,000 detections per tile. Fields in the whole output and F1 on the site's
evaluation area (one-to-one at IoU ≥ 0.5):

| Site (radar year) | Sentinel-1 | Umbra or Capella | NAIP, same settings | NAIP, defaults (not shown) | Sentinel-2 (not shown) |
|---|---|---|---|---|---|
| 1 Texas A&M (2024) | 28, 0.16 | 248, 0.37 | 210, 0.41 | 0.28 | 0.35 |
| 2 Intel Ohio (2024) | 14, 0.13 | 257, 0.26 | 152, 0.31 | 0.19 | 0.39 |
| 3 Orleans County (2022) | 3, 0.03 | 166, 0.27 | 171, 0.25 | 0.26 | 0.28 |

The Umbra and Capella composites come within 0.05 of NAIP read with the same
settings (−0.04, −0.05 and +0.03) and score above NAIP read with
Delineate-Anything's defaults; their lead over NAIP with the defaults comes
mostly from the settings. Sentinel-1 finds few fields (3 to 28 polygons;
recall 0.02 to 0.10): its resolution blurs fields of a few hectares (median
4.9, 5.7 and 3.5 ha). The references are farm-programme units of the USDA FSA
delivery (last geometry change 2020-08-17 in Texas, 2021-04-21 in Ohio and
Vermont), not a map of the imagery season; the Texas A&M area holds research
plots and the Intel Ohio area a construction site. With 77 to 136 fields per
site, the F1 intervals are about ±0.07. Credits: SAR data: Umbra Lab, Inc., CC
BY 4.0, modified; © Capella Space Corp, CC BY 4.0, modified; contains modified
Copernicus Sentinel data 2022 and 2024; OpenET agricultural field boundaries,
version 2.1.1, Desert Research Institute for the OpenET Consortium (2026), with
the field boundaries of their source agencies.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/SAR_held_out_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/SAR_held_out_example.webp" alt="Radar, held-out sites — Delineate-Anything v2 on Sentinel-1, Umbra and Capella, and on NAIP with the same settings" width="800" loading="lazy"></a>

---

## Radar — Sentinel-1, Umbra and Capella at three in-sample sites

**Example 26** · as above, at the three sites on which the Umbra and Capella
recipe and settings were chosen (in-sample): 1 UC Davis, Yolo and Solano
counties, California (Umbra, 6 collects); 2 Sevilla, Andalusia, Spain
(Capella, 2 collects of January 2023); 3 Centerfield, Sanpete County, Utah
(Umbra, 4 collects) · references: 1 DWR / Land IQ Statewide Crop Mapping 2024
(provisional): 235 fields; 2 SIGPAC recintos of 2025 (field land uses, at
least 2,500 m²): 1,128; 3 Utah Water Related Land Use 2025: 262 · in each
panel, the square with the most reference fields (2 km; 1.5 km at Sevilla).

| Site (radar year) | Sentinel-1 | Umbra or Capella | Optical control |
|---|---|---|---|
| 1 UC Davis (2024) | 152, 0.32 | 346, 0.49 | NAIP 2022, same settings: 217, 0.48 |
| 2 Sevilla (2023) | 8, 0.005 | 1,079, 0.51 | Sentinel-2, January 2023: 586, 0.40 |
| 3 Centerfield (2025) | 172, 0.29 | 59, 0.15 | NAIP 2021, same settings: 151, 0.43 |

(Fields in the whole output, F1 on the evaluation area. Not shown: NAIP with
the defaults, 0.43 at UC Davis and 0.35 at Centerfield; Sentinel-2, 0.48 and
0.29.) Spain has no NAIP, so at Sevilla the optical control is Sentinel-2 of
the same January (10 images), against recintos with a median of 0.8 ha: radar
and optical are not compared at a matching resolution there. Centerfield's
2025 Umbra products are single-look at 1 m, so its composite stays speckled.
These are not held-out tests: the recipe and settings were chosen on these
areas, and FBIS-73M holds the DWR / Land IQ and SIGPAC labels and Sentinel-1
patches, two of them inside the UC Davis area. Diamond Valley, the seventh
site of example 26 (Sentinel-1 only), is the bottom-right panel of the Diamond
Valley entry (F1 0.35, as in example 24). Credits: as above, with Copernicus
Sentinel data 2023–2025; DWR / Land IQ (California Natural Resources Agency
Open Data); SIGPAC, Fondo Español de Garantía Agraria (FEGA), CC BY 4.0,
modified; Utah Division of Water Resources.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/SAR_in_sample_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/SAR_in_sample_example.webp" alt="Radar, in-sample sites — Delineate-Anything v2 on Sentinel-1, Umbra and Capella, and on NAIP or Sentinel-2" width="800" loading="lazy"></a>

---

## Quebec — crop-history zones from the Annual Crop Inventory against declared parcels

**Example 27** · two 5 km squares in Quebec: 1 Bas-Saint-Laurent,
2 Montérégie · reference: the parcels declared to La Financière agricole du
Québec for 2023 (BDPPAD 2023; parcels with a production declared for 2023, at
least 2,500 m²): 490 and 443, median 2.4 ha · no crop filter · in each panel,
the 2.5 km square with the most reference parcels among nine candidates.

Left: crop-history zones of the AAFC Annual Crop Inventory 2016–2023
(`aci-sequence` and `crop-sequence`, the method of the zones of the 47-state
squares above), drawn on the Sentinel-2 composite of April to October 2023
(18 and 10 images), the input of the panel beside them. Right: Delineate
Anything v2 as released on that composite, with up to 1,000 detections per
tile.

| Square | Zones: polygons, F1 | Delineate Anything v2: fields, F1 | Blind grid, F1 | Shifted reference, F1 |
|---|---|---|---:|---:|
| 1 Bas-Saint-Laurent | 488, 0.20 | 456, 0.49 | 0.018 | 0.09 |
| 2 Montérégie | 403, 0.12 | 535, 0.47 | 0.007 | 0.12 |

(Counts: the whole output; F1: the whole 5 km square, one-to-one at
IoU ≥ 0.5.) The zones merge neighbouring parcels with the same crop history
(merge rates 0.66 and 0.78, against 0.42 and 0.36 for Delineate Anything v2):
82 of 638 and 128 of 546 adjacent declared parcel pairs have the same ACI class
in every year 2016–2023. Example 27 also scores two layers that use no image:
a grid of square cells the size of the zones' median (152 m and 149 m),
averaged over 36 offsets, and the reference shifted by 54 m and 40 m east and
north. At a 60 m tolerance every one of these baselines reaches a boundary F of
0.85 to 0.99, in the range of the zones (0.92 and 0.88) and of Delineate
Anything v2 (0.93 and 0.91), so at that tolerance boundary F does not tell
either from a layer that uses no image. The crop-sequence route follows the
recipe that Jacob T. Stid (Michigan State University) and Jeremy Rapp shared.
Credits: BDPPAD, La Financière agricole du Québec, CC BY 4.0 (Données
Québec), modified; ACI: Agriculture and Agri-Food Canada, Annual Crop
Inventory 2016–2023, Open Government Licence – Canada (2.0); contains
modified Copernicus Sentinel data 2023.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Quebec_crop_sequence_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Quebec_crop_sequence_example.webp" alt="Quebec — crop-history zones from the AAFC Annual Crop Inventory and Delineate-Anything v2 on Sentinel-2 against the declared parcels" width="800" loading="lazy"></a>

---

## New York, Black Dirt — vegetable and onion strips on muck against the Common Land Units

**Example 28** (entry 28) · the Black Dirt, Orange County, New York (towns of
Goshen and Wawayanda), example 25's New York square: the 5 km square with
the most farmed fields of the county, chosen by a fixed rule before any model
ran · reference: the OpenET agricultural field boundaries v2.1.1, in New York
all USDA FSA Common Land Units (CLU): the 668 fields of at least 2,500 m²
whose representative point lies in the square (median 1.70 ha; 305
strip-like, with a Polsby-Popper score below 0.3, about 37 m wide) · min.
area 2,500 m²; no crop filter on the maps · window: the evaluation square ·
panels by the rule of PREREG-NY, fixed before the run.

Each panel is drawn on its own input: the NAIP 1 m mosaic of 2022 (two images
of 4 July 2022 on top of two of 6 November 2021) in the top row and at bottom
right, and the Sentinel-2 10 m median of April to October 2024 (19 images) at
bottom left. The panels follow the rule: Delineate Anything v2 as released on
NAIP (top left); the NAIP run with the highest F1 among Delineate Anything v2
with SAM 2 refinement and Delineate Anything v2 and DINOv3 fine-tuned on the
region's training square (top right: SAM 2 refinement, orange where SAM 2
refined a polygon, 492 of the 565); the higher F1 of Delineate Anything v2
and FTW on Sentinel-2 (bottom left: Delineate Anything v2); and `sam-auto`
(SAM 2.1 automatic masks, with its NAIP setting chosen on two Nevada squares,
not on any New York square) on NAIP (bottom right). One-to-one matching at
IoU ≥ 0.5 against all 668 fields, with 95 % bootstrap intervals (200
resamples of the reference fields, stratified by land use in 2024):

| Panel | Fields | F1 [95 %] | Merge rate | Strip-like fields found | Blind grid F1 | Shifted reference F1 |
|---|---:|---|---:|---|---:|---:|
| NAIP 2022, Delineate Anything v2 as released | 585 | 0.15 [0.12–0.18] | 0.75 | 7 of 305 | 0.002 | 0.70 |
| NAIP 2022, Delineate Anything v2 with SAM 2 refinement | 565 | 0.16 [0.13–0.19] | 0.70 | 11 of 305 | 0.003 | 0.70 |
| Sentinel-2 2024, Delineate Anything v2 as released | 266 | 0.11 [0.08–0.14] | 0.82 | 1 of 305 | 0.006 | 0.70 |
| NAIP 2022, `sam-auto` | 365 | 0.18 [0.15–0.21] | 0.64 | 9 of 305 | 0.005 | 0.70 |

Merge rate: the share of reference fields whose main prediction also covers
at least a quarter of another field. The two baselines use no image: square
cells of the run's median polygon area over the square (mean of 36
positions), and the reference itself moved 24 m east and 24 m north.

No run in this square is above chance by the rule fixed before the run (the
lower bound of F1 above both baselines), the near-sample models below
included: the moved reference still matches 470 of the 668 fields. Most
fields are long and narrow (their minimum rotated rectangles a median 49 m by
365 m); 385 of them run north-east, along the shift, and strips of one width
repeat side by side, so a moved field still overlaps itself or its
neighbour (measured after the scores were known). The released model finds 7
of the 305 strip-like fields and 85 of the 363 compact ones, and for 75 % of
the fields the prediction that covers most of a field also covers a quarter
of another: its polygons span several strips. The released runs on NAIP and
Sentinel-2 reproduce example 25's New York scores (0.147 and 0.111) polygon
for polygon, the check fixed before the run (those two scores were known).

Fine-tuning. Delineate Anything v2 and DINOv3 fine-tuned on the NAIP of a
training square 5 km west (pasture and forest; 310 CLU labels reach into
it), never on the evaluated fields, scored 0.09 and 0.14: Delineate Anything
v2 below the released model (both paired 95 % intervals of the F1 difference
below 0), DINOv3 not clearly different. Fine-tuned instead on a square of
Pine Island muck strips 1 km south-west, flown the same day (680 CLU labels),
they scored 0.30 and 0.28 and found 82 and 57 of the 305 strips; by the rule
both are above the released model (both paired intervals above 0), but these
two runs are near-sample, not a held-out test.

The scores measure agreement with CLUs (farm-programme units), not with
visible fields; one square per region is not an accuracy estimate for the
county. FBIS-73M, Delineate Anything v2's training data, holds no field label
in New York, and its nearest patches lie 146.9 to 422.5 km from the three
evaluation squares (checked on 2026-10-13 against its public lists, which
were published after training and may not be exactly what the released
weights saw).

Reference: OpenET agricultural field boundaries, version 2.1.1, Desert
Research Institute for the OpenET Consortium (2026), with the field
boundaries of their source agency: Common Land Unit (CLU), USDA Farm Service
Agency. NAIP: USDA Farm Service Agency. The Sentinel-2 panel contains
modified Copernicus Sentinel data (2024). SAM 2.1: Meta, Apache-2.0; the
`sam2` 1.1.0 package on PyPI is a third-party build whose 62 code,
configuration and licence files equal Meta's commit `2b90b9f`.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/New_York_Black_Dirt_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/New_York_Black_Dirt_example.webp" alt="Black Dirt, Orange County, New York — Delineate-Anything v2 with and without SAM 2 refinement and sam-auto on NAIP, Delineate-Anything v2 on Sentinel-2, against the Common Land Units of the OpenET field boundaries" width="800" loading="lazy"></a>

---

## New York, Finger Lakes — dairy rotations against the Common Land Units

**Example 28** (entry 28b) · the Finger Lakes, Cayuga County, New York
(towns of Fleming and Springport, between Auburn and Union Springs): corn,
alfalfa and soybeans of dairy farms, with hedgerows and some contour strips ·
the 5 km square chosen by the same rule · reference: the 482 CLU fields of
the square (median 2.49 ha; 105 strip-like) · min. area 2,500 m²; no crop
filter on the maps · window: the evaluation square.

Each panel is drawn on its own input: the NAIP mosaic of 2022 (four images of
15 July 2022 on top of four of 24 August 2021) and, at bottom left, the
Sentinel-2 median of April to October 2024 (27 images). The panels follow the
rule of entry 28; top right is DINOv3 fine-tuned on the NAIP of a training
square 15 km south (505 CLU labels reach into it), never on the evaluated
fields.

| Panel | Fields | F1 [95 %] | Merge rate | Strip-like fields found | Blind grid F1 | Shifted reference F1 |
|---|---:|---|---:|---|---:|---:|
| NAIP 2022, Delineate Anything v2 as released | 392 | 0.34 [0.30–0.37] | 0.70 | 11 of 105 | 0.015 | 0.087 |
| NAIP 2022, DINOv3 fine-tuned | 321 | 0.43 [0.40–0.47] | 0.70 | 16 of 105 | 0.032 | 0.087 |
| Sentinel-2 2024, Delineate Anything v2 as released | 305 | 0.35 [0.31–0.39] | 0.70 | 4 of 105 | 0.032 | 0.087 |
| NAIP 2022, `sam-auto` | 364 | 0.36 [0.32–0.39] | 0.54 | 16 of 105 | 0.020 | 0.087 |

Every run of this square is above chance by the rule (the reference moved
59 m east and north matches 42 of the 482 fields). DINOv3 fine-tuned scores
above the released model (F1 difference +0.095; paired 95 % intervals 0.060
to 0.128 over reference fields and 0.055 to 0.138 over 1 km blocks).
Delineate Anything v2 fine-tuned the same way (+0.007), SAM 2 refinement
(−0.006) and `sam-auto` (+0.023) show no clear difference, and FTW on
Sentinel-2 scores below Delineate Anything v2 there (0.26 against 0.35).
Recall rises with field size: the released model on NAIP finds 5 of the 83
fields under 1 ha and 14 of the 30 of 10 ha or more; it finds 11 of the 105
strip-like fields and 137 of the 377 compact ones.

The scores measure agreement with CLUs (farm-programme units), not with
visible fields; one square per region is not an accuracy estimate for the
county.

Reference and credits as in entry 28. The Sentinel-2 panel contains modified
Copernicus Sentinel data (2024).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/New_York_Finger_Lakes_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/New_York_Finger_Lakes_example.webp" alt="Finger Lakes, Cayuga County, New York — Delineate-Anything v2, fine-tuned DINOv3 and sam-auto on NAIP, Delineate-Anything v2 on Sentinel-2, against the Common Land Units of the OpenET field boundaries" width="800" loading="lazy"></a>

---

## New York, North Country — corn, hay and pasture against the Common Land Units

**Example 28** (entry 28c) · the North Country, Jefferson County, New York
(town of Ellisburg, near Belleville and Pierrepont Manor), on the Lake
Ontario plain · the 5 km square chosen by the same rule · reference: the 281
CLU fields of the square (median 3.98 ha; 29 strip-like) · min. area
2,500 m²; no crop filter on the maps · window: the evaluation square.

Each panel is drawn on its own input: the NAIP mosaic of 2022 (two images of
14 October 2022 on top of three of 20 September 2021) and, at bottom left,
the Sentinel-2 median of April to October 2024 (24 images). The panels follow
the rule of entry 28; top right is Delineate Anything v2 fine-tuned on the
NAIP of a training square 10 km north (293 CLU labels reach into it), which
edges out DINOv3 fine-tuned the same way (F1 0.433 against 0.432).

| Panel | Fields | F1 [95 %] | Merge rate | Strip-like fields found | Blind grid F1 | Shifted reference F1 |
|---|---:|---|---:|---|---:|---:|
| NAIP 2022, Delineate Anything v2 as released | 272 | 0.37 [0.32–0.41] | 0.63 | 3 of 29 | 0.007 | 0.036 |
| NAIP 2022, Delineate Anything v2 fine-tuned | 296 | 0.43 [0.38–0.49] | 0.56 | 3 of 29 | 0.016 | 0.036 |
| Sentinel-2 2024, Delineate Anything v2 as released | 211 | 0.46 [0.40–0.51] | 0.60 | 3 of 29 | 0.020 | 0.036 |
| NAIP 2022, `sam-auto` | 244 | 0.40 [0.35–0.46] | 0.60 | 1 of 29 | 0.013 | 0.036 |

Every run of this square is above chance by the rule (the reference moved
82 m east and north matches 10 of the 281 fields). Both fine-tuned models
score above the released model on NAIP: Delineate Anything v2 by +0.061
(paired 95 % intervals 0.014 to 0.109 over fields and 0.018 to 0.110 over
1 km blocks), DINOv3 by +0.060 (0.022 to 0.096 and 0.024 to 0.096); SAM 2
refinement (−0.012) and `sam-auto` (+0.031) show no clear difference. The
released model on NAIP finds 27 of the 43 fields of 10 ha or more and 5 of
the 40 under 1 ha. The NAIP of 2022 is from 14 October; no rule compares it
with the Sentinel-2 season median.

The scores measure agreement with CLUs (farm-programme units), not with
visible fields; one square per region is not an accuracy estimate for the
county.

Reference and credits as in entry 28. The Sentinel-2 panel contains modified
Copernicus Sentinel data (2024).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/New_York_North_Country_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/New_York_North_Country_example.webp" alt="North Country, Jefferson County, New York — Delineate-Anything v2 released and fine-tuned and sam-auto on NAIP, Delineate-Anything v2 on Sentinel-2, against the Common Land Units of the OpenET field boundaries" width="800" loading="lazy"></a>

---

## France, Beauce — FTW on Sentinel-2

**Example 04** · FTW on Sentinel-2 2023 · the Beauce, Eure-et-Loir, just east
of Bonneval and north-east of Châteaudun · min. area 5,000 m²; Dynamic World
2023 crop filter · FTW read two season windows chosen from the FTW crop
calendar, 27 Feb–28 Apr (4 images, shown) and 7 Aug–6 Oct 2023 (4 images).

665 fields (median 10.0 ha); the crop filter kept all of them. The output is
identical to that of agribound 1.0.0.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/France_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/France_example.webp" alt="France — FTW on Sentinel-2" width="800" loading="lazy"></a>

---

## Kenya, Kakamega — FTW on Sentinel-2 at four minimum areas

**Example 06** · FTW on Sentinel-2 2023 · Kakamega County, Western Kenya,
mainly Malava sub-county, an area of small farms (field sizes in this box were
not measured) · Dynamic World 2023 crop filter · FTW windows
15 Aug–14 Oct 2023 (21 images, shown) and 11 Jan–11 Mar 2024 (8 images) ·
four `min_field_area_m2` values in the same 1 km window (the most polygons at
100 m²; outlines drawn with a white halo).

Fields: 2,404 at 100 m², 1,578 at 500 m², 1,029 at 1,000 m² and 344 at
2,500 m². The minimum area also applies when FTW's prediction is polygonised,
not only as a filter afterwards. The crop filter removes most of the FTW
polygons: it kept 2,404 of 24,616 at 100 m² (10 %) and 344 of 1,987 at
2,500 m² (17 %). Why they are removed was not measured, and there is no
reference layer here. Before relying on the filter in a landscape like this,
compare the output with and without it (`lulc_filter`,
`lulc_crop_threshold`).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Kenya_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Kenya_example.webp" alt="Kenya — FTW at four minimum-area thresholds" width="800" loading="lazy"></a>

---

## California, USA — Delineate-Anything on NAIP

**Example 07** · Delineate Anything v2 on NAIP 1 m, 2022 (9 scenes, mosaicked,
from Earth Engine) · western San Joaquin Valley, Fresno County, near Five
Points · min. area 10,000 m²; Annual NLCD 2022 crop filter.

322 fields (median 22.9 ha); the crop filter kept all of them.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Central_Valley_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Central_Valley_example.webp" alt="California — Delineate-Anything on NAIP" width="800" loading="lazy"></a>

---

## California, USA — Delineate-Anything on USGS NAIP Plus (no Earth Engine)

**Example 16** · Delineate Anything v2 on USGS NAIP Plus (0.6 m, 2022; 8
source rasters exported from the USGS ImageServer) · the same study area as
example 07 · min. area 10,000 m²; no crop filter.

297 fields (median 22.4 ha). This source path needs no Earth Engine, so the
example runs without the crop filter, which reads its land-cover data from
Earth Engine.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/USGS_NAIP_Plus_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/USGS_NAIP_Plus_example.webp" alt="California — Delineate-Anything on USGS NAIP Plus" width="800" loading="lazy"></a>

---

## Australia, Murray-Darling Basin — Prithvi on HLS vs Delineate-Anything v2 on SPOT

**Example 03** · Narrabri Shire, New South Wales, just north of Narrabri ·
min. area 5,000 m²; Dynamic World crop filter · the Prithvi runs use the GFM
environment · made again with agribound 1.1.0 on 2026-10-11.

- Left: Prithvi-EO-2.0 `mode="embed"`: K-means on the patch tokens of the last
  encoder layer. The tokens are 16 pixels, 480 m, apart at 30 m and are
  interpolated to pixels. It merges neighbouring fields into 51 polygons
  (median 132 ha).
- Middle: `mode="pca"`, a baseline that uses no Prithvi weights: K-means on the
  PCA of per-band z-scores of R, G, B and NIR. It splits fields along
  within-field variation into 1,339 fragments (median 1.4 ha).
- Right: Delineate Anything v2 ("DA v2") used as released, on SPOT 6/7 (6 m),
  2023 (3 images; SPOT has no scene over this box in 2022): 420 fields that
  follow the rectangular field edges.

The left and middle panels use HLS 2022 (30 m; 77 images, 20 HLSL30 and 57
HLSS30, all of tile T55JGG). Up to the development run of 1.1, agribound
counted 298 images here: the count included 221 HLSS30 images of the MGRS
tiles at 180°, which hold no pixel here (see
[HLS](user-guide/satellite-sources.md#hls-hls)). The re-run has the same
composite pixels and the same three outputs as the development run. The
right panel is a year later. There is no reference layer here, so the panels
compare outlines, not accuracy.

Neither mode delineates field instances; both cluster pixels into land-cover
segments, and neither was trained or fine-tuned on field boundaries in these
runs. The embed mode's over-merging, first seen in 0.1.x, remains with the
corrected HLS radiometry of 1.0. Prithvi's supervised `mode="segment"` needs
fine-tuning on reference polygons (`fine_tune=True`) and was not run here.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Australia_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Australia_example.webp" alt="Australia — Prithvi embed mode and a PCA baseline on HLS, and Delineate-Anything v2 on SPOT" width="800" loading="lazy"></a>

---

## North China Plain — Delineate-Anything on SPOT

**Example 08** · Delineate Anything v2 on SPOT 6/7 (6 m), 2023 (5 scenes;
restricted SPOT access) · Hengshui (Jizhou District), Hebei · min. area
3,000 m²; Dynamic World 2023 crop filter · window: the 4 km square with the
most Delineate-Anything polygons.

4,503 strip fields in the study area (median 0.61 ha); the crop filter kept
4,503 of 4,543.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/China_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/China_example.webp" alt="China — Delineate-Anything on SPOT" width="800" loading="lazy"></a>

---

## Spain, Andalusia — Delineate-Anything, FTW and their vote merge

**Example 09** · Delineate Anything v2 and FTW on Sentinel-2 2024, combined
afterwards from the saved outputs · Seville province, east of Carmona (the
3 km window is at the study area's east edge, in the Carmona and Fuentes de
Andalucía municipalities; the most Delineate-Anything polygons) · min. area
2,500 m²; Dynamic World 2024 crop filter · Delineate-Anything read the annual
composite (66 images, shown under all three panels); FTW read 12 Mar–11 May
and 11 Sep–10 Nov 2024 (4 images each).

Delineate-Anything returns 1,505 fields (median 4.4 ha) and FTW 469 (median
6.5 ha). The vote merge keeps the pixels that both engines cover, on a 10 m
grid, which gives it its stair-stepped outlines. Neighbouring fields that are
both kept fuse into one polygon: 790 polygons, 589 after the 2,500 m² filter.
The example also writes the intersection (1,405) and union (690) merges,
which are computed on the vector polygons. The straight cuts along tile edges
that crossed fields in the 1.0 image of this window (1.7 and 1.3 km long) are
gone with the tile stitching of 1.1.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Spain_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Spain_example.webp" alt="Spain — Delineate-Anything, FTW and vote merge" width="800" loading="lazy"></a>

---

## Mississippi Alluvial Plain, USA — Delineate-Anything on SPOT, 2021–2023

**Example 11** · Delineate Anything v2 on SPOT 6/7 (6 m) for 2021, 2022 and
2023, each panel on its own year's composite (restricted SPOT access) · near
Greenville, Washington County, Mississippi (about 6 % of the study area, in
its north-western corner, is in Chicot County, Arkansas, by US Census county
boundaries; the 4 km window, the square with the most 2023 polygons, centre
33.42° N, 90.93° W, is in Mississippi) · min. area 10,000 m²; Annual NLCD
crop filter of each year.

1,970, 1,253 and 1,728 fields. The 2022 composite has only 2 SPOT scenes
(16 in 2021, 12 in 2023). That year has the fewest polygons and the largest
median area (8.4 ha, against 4.2 and 5.1 ha), while the total delineated area
is similar; these runs do not test whether the scene count is the cause.
Year-to-year agreement (one-to-one matching at IoU ≥ 0.5) is F1 0.51 from
2021 to 2022 and 0.57 from 2022 to 2023; this compares predictions with each
other and says nothing about accuracy.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/MAP_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/MAP_example.webp" alt="Mississippi Alluvial Plain — Delineate-Anything on SPOT, 2021 to 2023" width="800" loading="lazy"></a>

---

## San Juan County, New Mexico — evaluation against the DRI/NMOSE field boundaries (V18)

**Example 20** · Delineate Anything v2 on Sentinel-2 2024 (184 images) ·
San Juan County, New Mexico (Farmington–Bloomfield area) · min. area
2,500 m²; Annual NLCD 2024 crop filter · evaluated against the DRI/NMOSE
field boundaries (V18) of at least 2,500 m² (cyan; 350 of the 1,705 polygons
in the box are smaller and dropped), which were not used for training or
fine-tuning in these runs (Delineate Anything v2 is used as released; its
training set, FBIS-73M, has no patch in New Mexico in its public lists: the
nearest lies 90.6 km from the box, in Arizona and unlabelled, and the nearest
labelled patch 604.4 km away, checked on 2026-10-13).

All 357 predictions lie inside the evaluation box (by representative point),
against 1,355 reference fields. One-to-one matching at IoU ≥ 0.5 gives
precision 0.64, recall 0.17 and F1 0.27
([95 % bootstrap intervals](user-guide/evaluation.md) from 200 resamples of
reference fields within sub-basins: F1 0.24–0.29, recall 0.15–0.19; they
treat fields as independent, so they are too narrow if errors are spatially
clustered); the mean IoU of matched fields is 0.80. Area-weighted recall is
0.55 and area-weighted precision 0.87. Recall rises with field size: it is
0.01 for fields of 0.2–0.5 ha, 0.51 at 10–20 ha, 0.68 at 20–50 ha and 0.78 at
50–100 ha. F1 broadly follows but dips at 5–10 ha (0.30, against 0.37 at
2–5 ha). By V18 sub-basin, the Middle San Juan (107 reference fields, median
4.6 ha) reaches F1 0.42, the Upper San Juan (689, median 1.1 ha) 0.31 and the
Animas (559, median 0.9 ha) 0.16. Two caveats: V18 records no imagery date per
field, so fields that changed between the reference's date and the imagery
count as errors, and the crop filter removed 234 of 591 polygons before the
evaluation. This output holds the pivot that the SAM 2 entry below shows cut
in two.

Reference: DRI/NMOSE field boundaries (V18), Desert Research Institute and New
Mexico Office of the State Engineer.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/San_Juan_evaluation_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/San_Juan_evaluation_example.webp" alt="San Juan County — Delineate-Anything vs the DRI/NMOSE field boundaries (V18)" width="800" loading="lazy"></a>

---

## SAM 2 refinement — example 20's output before and after

**Example 13** · SAM 2 box-prompted refinement of example 20's
Delineate-Anything output (Sentinel-2 2024). The default input of this
example is example 12's output, which was not run for 1.0 or 1.1 (only 12's
NAIP runs were) · window: the 3 km square with the most SAM-refined fields, a
group of centre pivots.

At 10 m, only 72 of the 357 fields pass the SAM size rule (all 8.3 ha or
larger, median 32.7 ha); the other 285 are not refined. After SAM, the example
smooths (Chaikin ×3) and simplifies all 357 polygons again, although example
20's output was already smoothed, so the before/after numbers include this
second smoothing. Against the 1,355 V18 polygons, object F1 goes from 0.265 to
0.262 (227 and 224 matches), and the mean boundary distance falls from 14.2 m
to 13.2 m, with SAM and the second smoothing together. SAM replaces each
prompted polygon with its own mask and never merges or deletes polygons
(357 in, 357 out), so Delineate-Anything's splits inside pivots remain; with
`sam_overlaps="trim"` a refined mask also cannot take area from a
neighbouring polygon (12 masks were trimmed). No mask covered less than half
of its polygon, so the coverage test (`sam_min_coverage` = 0.5) rejected none.
Examples 13 and 20 evaluate the same 357 polygons.

At the lower left of the window a pivot is cut along a straight north-south
line, and its north-west part has no polygon. The line is the edge of the
engine's tiles (2.56 km on this composite, overlapping by half). The tiles that
see the whole pivot score it below the confidence threshold of 0.15 (about
0.08 and 0.07), while the tiles that cut it score the two pieces higher (0.21
and 0.18), so only the pieces are kept. They cover 26 % and 33 % of the pivot;
the seam test does not join them, because their cut edges do not end at the
same places along the tile edge, and no detection at or above 0.15 covers the
north-west part. The upstream Delineate-Anything pipeline gives the same two
pieces. SAM 2 refines neither of them.

Reference: DRI/NMOSE field boundaries (V18), Desert Research Institute and New
Mexico Office of the State Engineer.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/SAM2_refinement_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/SAM2_refinement_example.webp" alt="SAM 2 refinement before and after" width="800" loading="lazy"></a>

---

## HPC tiling — four tiles merged

**Example 19** · the `agribound.hpc` workflow (make tiles, run, merge) run
locally on a 4.5 × 4.5 km study area in the Beauce, Eure-et-Loir, east of
Bonneval (inside example 04's study area) · Delineate Anything v2 on
Sentinel-2 2024 composites, one per tile · min. area 2,500 m²; Dynamic World
2024 crop filter.

The study area is cut along the UTM 31N grid into four core tiles of up to
2.5 km (dashed); their outer edges follow the study area's longitude/latitude
box, so they lean slightly in this UTM view. Each core is delineated with a
1 km halo around it. The merge keeps each polygon only in the tile that owns
its representative point: 1,161 tile polygons become 293. No polygon reached
a halo edge, so fields that cross the core edges come out whole; one field was
delineated by two tiles and appears twice (the pair is not marked).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/HPC_tiling_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/HPC_tiling_example.webp" alt="HPC tiling — four tiles merged" width="800" loading="lazy"></a>
