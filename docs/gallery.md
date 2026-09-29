# Example Gallery (agribound 1.0)

Field boundaries from the agribound 1.0.0 example scripts, run at their
default settings on 2026-09-28 and 29. The exception is example 13, which
refines example 20's output (see its entry); examples 01 and 12 were not run
end to end (12's NAIP runs were). The 0.1.x screenshots are kept on the
[archived 0.1.x page](gallery-0.1x.md).

**How to read the images.** Red outlines are agribound output, cyan outlines
are reference polygons and orange outlines are fields that SAM refined. Each
map is drawn on a composite from the run, named under the map: usually the
engine's input; for FTW, window A, the first of FTW's two season inputs; for
the SAM entries, the composite SAM read. The imagery therefore shows the
acquisition period of that composite. Zoom panels and cropped windows show the
square with the most polygons of one layer (named in each entry), not a random
sample of the study area. The number in a panel title counts every polygon in
that output, not only those inside the window shown. The inset locates the
study area (red dot) in its country; India is drawn from the Survey of India
outline, all other boundaries from Natural Earth. Under each image are the
imagery, the model and its version.

**SPOT colours.** SPOT 6/7 multispectral composites are uncalibrated digital
numbers, and each of their bands is stretched separately, while the other
sources share one stretch across R, G and B. SPOT colours therefore cannot be
compared with those of the other sources: bare soil often looks mauve or
lavender (examples 03, 11, 14 and 15). A median of a few SPOT scenes can also
show straight, sudden colour steps. The SPOT panel of example 14 (7 images)
has a nearly vertical one about halfway across that shows several pivots in
two tones. The step is in the composite the engine read, not added by the
rendering. SPOT-Pan panels are shown in grey.

**Crop filter.** Unless an entry says otherwise, the polygons pass a crop
filter at a threshold of 0.30: outside the conterminous US, Dynamic World (the
polygon mean of the year's median crop probability); in the conterminous US,
Annual NLCD (the share of cultivated-crop and pasture/hay pixels, classes 82
and 81). The minimum field area is given in each entry.

!!! note "What these images show"
    Each image shows what one configuration produces, not how accurate it is.
    Accuracy is reported only where a reference layer exists (examples 12, 13,
    14 and 20, all against the NMOSE polygons in New Mexico; for 14 and for the
    fine-tuned models of 12 they are also the training labels, so those scores
    are in-sample).

The images are rendered at 3000 px by
[`tools/make_gallery.py`](https://github.com/montimaj/agribound/blob/main/tools/make_gallery.py)
from the run outputs. This page shows 1600 px WebP previews; click an image to
open the full-resolution PNG. The numbers below come from the run logs, the
evaluation metrics files and
[`assets/gallery_1.0/gallery_stats.json`](https://github.com/montimaj/agribound/blob/main/assets/gallery_1.0/gallery_stats.json).

## Models and versions

| Engine | Model and version used in these runs |
|---|---|
| Delineate-Anything | Delineate Anything v2, `large_v2` (YOLO11x-seg, trained on FBIS-73M): `DelineateAnythingv2.pt` from Hugging Face `MykolaL/DelineateAnything` at revision `369d0b4`, SHA-256 pinned; agribound's native implementation with ultralytics 8.4.163; confidence threshold 0.15 |
| FTW | `FTW_PRUE_EFNET_B5` ("FTW v3: Standard, B5": PRUE U-Net with an EfficientNet-B5 encoder), ftw-baselines v3 checkpoint, SHA-256 pinned; run with ftw-tools 2.0.0b5 (a pre-release); no fine-tuning |
| SAM 2 | `facebook/sam2-hiera-large` (SAM 2.0 Hiera-L, not 2.1), through segment-geospatial 1.4.2 and the `sam2` 1.1.0 package; one box prompt per field |
| Prithvi | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL` at revision `63adbd3`, terratorch 1.2.13 |
| GeoAI | torchvision Mask R-CNN ResNet50-FPN via geoai-py 0.43.1, fine-tuned on the reference polygons (no published field weights); chips sized from the reference fields; instances split at the inference-window edges joined |
| DINOv3 | `dinov3_vitl16` (ViT-L/16) with the SAT-493M weights `giswqs/geoai/dinov3_vitl16_sat493m.pth` at revision `aa2b25d`, geoai-py 0.43.1; full fine-tuning on the reference polygons (no published field weights) |
| Embeddings | Google Satellite Embedding (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, AlphaEarth Foundations, 64-D) and TESSERA v1 (128-D, geotessera 0.10.2); PCA to 16 components, then MiniBatchKMeans (k chosen by silhouette score among 5, 10, 15, 20, 30 and 50 unless stated; in every automatic choice here the score was highest at k = 5, the smallest candidate, and smaller k were not tested) |

**SAM size rule.** SAM is prompted only when a field's bounding box, padded
on every side by 15 % of its size (a factor of 1.3), is at least 64 pixels
wide and 64 pixels tall. That is an unpadded box of at least about 49 pixels
on each side (64 / 1.3 ≈ 49.2): about 490 m at 10 m, 295 m at 6 m and 49 m at
1 m. Fields below that keep their geometry. With the default
`sam_overlaps="trim"`, a refined mask cannot take area from a neighbouring
polygon.

---

## Lea County, New Mexico — DINOv3 fine-tuned, from 30 m to 1 m

**Example 14** · DINOv3 ViT-L/16 (SAT-493M weights, geoai-py 0.43.1), fully
fine-tuned on the NMOSE polygons of the study area separately for each source,
then refined with SAM 2 · eastern Lea County, New Mexico, north of Hobbs (the
easternmost 1.3 km of the box, 7 % of its area, is in Gaines County, Texas,
where NMOSE has no polygons) · 2022 · min. area 2,500 m² (5,000 m² for NAIP);
Annual NLCD 2022 crop filter · window: the 6 km square with the most
reference fields.

Panels: Landsat 30 m and Sentinel-2 10 m (October composites, 3 images each),
SPOT 6/7 6 m (annual, 7 images; its colour step is described under "SPOT
colours" above) and NAIP 1 m (25 images). The
NMOSE polygons (cyan) are also the training labels, so these scores are
in-sample, not an independent accuracy estimate. Against the 227 NMOSE
polygons in the box (one-to-one matching at IoU ≥ 0.5), the fields and the
in-sample F1 without and with SAM 2 are:

| Source | Fields (without / with SAM 2) | In-sample F1 without SAM 2 | In-sample F1 with SAM 2 |
|---|---|---|---|
| Landsat 30 m | 31 / 31 | 0.06 | 0.09 |
| Sentinel-2 10 m | 116 / 114 | 0.38 | 0.39 |
| SPOT 6/7 6 m | 137 / 135 | 0.42 | 0.48 |
| NAIP 1 m | 190 / 181 | 0.60 | 0.60 |

(HLS 30 m, not shown: in-sample F1 0.11 with and without SAM 2.) The training
data change with resolution as well: the box holds 4 training chips of 256
pixels at 30 m (3 for training, 1 for validation), 49 at 10 m, 126 at 6 m and
2,014 at 1 m, so this comparison changes the amount of training data along
with the pixel size (and NAIP uses a minimum area of 5,000 m² rather than
2,500 m²). In-sample F1 rises most with SAM 2 at 6 m (0.42 to 0.48); at 1 m
SAM 2 raises the mean IoU of matched fields from 0.88 to 0.91, while in-sample
F1 stays at 0.60 (0.604 without SAM 2 and 0.598 with it; 126 and 122 matched
fields).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/NM_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/NM_example.webp" alt="Lea County — DINOv3 fine-tuned and SAM 2 on Landsat, Sentinel-2, SPOT and NAIP" width="800"></a>

---

## Lea County, New Mexico — Delineate-Anything v2, GeoAI and DINOv3 on NAIP 1 m

**Example 12** (NAIP runs only; the rest of example 12 was not run for
1.0.0) · eastern Lea County, New Mexico, north of Hobbs (the box of example 14)
· NAIP 1 m, 2022 (25 images) · min. area 5,000 m²; Annual NLCD 2022 crop
filter · window: the 6 km square with the most reference fields.

Delineate Anything v2 as released (top left; a run with example 12's
settings that example 12 itself does not include) and three models
fine-tuned on the 230 NMOSE polygons of the box with example 12's settings
(10 epochs, 5 km block split): Delineate Anything v2 `large_v2` (650 chips of
512 px), GeoAI Mask R-CNN (229 chips of 1,024 px) and DINOv3 (2,014 chips of
256 px). The NMOSE polygons (cyan) are the fine-tuning labels, so the scores of
the fine-tuned models are in-sample, not independent accuracy estimates.
Against the 227 NMOSE polygons in the box (one-to-one matching at IoU ≥ 0.5):

| Model | Fields | Precision | Recall | F1 | Mean IoU of matched fields | Predictions overlapping no reference polygon |
|---|---|---|---|---|---|---|
| Delineate Anything v2 as released | 635 | 0.23 | 0.64 | 0.34 | 0.86 | 261 |
| Delineate Anything v2 fine-tuned (in-sample) | 374 | 0.35 | 0.58 | 0.44 | 0.85 | 86 |
| GeoAI fine-tuned (in-sample) | 447 | 0.36 | 0.72 | 0.48 | 0.84 | 135 |
| DINOv3 fine-tuned (in-sample) | 195 | 0.64 | 0.55 | 0.59 | 0.90 | 53 |

Many of the predictions that overlap no reference polygon are fields the
registry does not include (NMOSE has no polygons in the Texas strip of the
box). Fine-tuning mainly reduced Delineate Anything v2's predictions outside
the registry (261 to 86) at a small cost in recall.

GeoAI needed two engine changes before it produced whole fields; both are in
1.0.0. With the earlier fixed 256 px chips, every GeoAI polygon was at most one
256 m inference window (the pivots here are about 800 m across), and in-sample
F1 was 0.01 (2,902 polygons). Chips sized from the reference fields (the
1.0.0 default, here 1,024 px, 1 km; `engine_params["chip_size"]` overrides)
raised F1 to 0.25 (723 polygons), with straight cuts left where a field
crossed the edges of the overlapping inference windows. Joining the pieces
split at those edges (`engine_params["merge_window_seams"]`, on by default:
448 instances joined, gaps of up to 2 px along the edges filled) gave the
result shown (F1 0.48). A few short cuts remain.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/NM_NAIP_models_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/NM_NAIP_models_example.webp" alt="Lea County — Delineate-Anything v2, GeoAI and DINOv3 on NAIP" width="800" loading="lazy"></a>

---

## San Juan County, New Mexico — Delineate-Anything v2 from 30 m to 1 m

**Example 20** (resolution comparison) · Delineate Anything v2 used as
released, on Landsat 7/8 (a median of the Landsat 7 and 8 Collection 2 Level-2
collections; 30 m, 75 images), Sentinel-2 (10 m, 188 images), SPOT 6/7 (6 m,
7 images) and NAIP (1 m, 24 images), all 2018 annual composites · the study
area of example 20 · min. area 2,500 m²; Annual NLCD 2018 crop filter ·
window: the 2.5 km square with the most reference fields.

2018 is the year closest to the reference's 2016 imagery that all four sources
cover (NAIP is flown every two years here). Against the 944 NMOSE polygons
(one-to-one matching at IoU ≥ 0.5; the polygons were not used for training or
fine-tuning in these runs):

| Source | Fields in the box | Precision | Recall | F1 | Boundary F1 (10 m) |
|---|---|---|---|---|---|
| Landsat 30 m | 116 | 0.68 | 0.08 | 0.15 | 0.14 |
| Sentinel-2 10 m | 420 | 0.55 | 0.24 | 0.34 | 0.41 |
| SPOT 6/7 6 m | 494 | 0.48 | 0.25 | 0.33 | 0.43 |
| NAIP 1 m | 986 | 0.42 | 0.44 | 0.43 | 0.59 |

(Fields in the box: the predictions whose representative point lies inside
the evaluation box, as in the crop-filter table below. The panel titles count
each whole output, 117, 421, 495 and 987 polygons: in each run one polygon
lies across the box's south edge (the west edge for NAIP), with its
representative point just outside.)

Recall rises about fivefold from 30 m to 1 m and boundary F1 about fourfold;
SPOT at 6 m is within 0.01 of Sentinel-2 on object F1 (0.33 and 0.34) and
slightly higher on boundary F1 (0.43 and 0.41). Landsat is outside Delineate
Anything v2's 0.25–10 m training range (agribound warns and records it). The
crop filter removed 43 % of the output polygons on Landsat (it kept 117 of
204), 41 % on Sentinel-2 (421 of 713), 50 % on SPOT (495 of 996) and 64 % on
NAIP (987 of 2,709). The reference was digitised from 2016 NAIP, so changes by
2018 count as errors.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/San_Juan_resolution_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/San_Juan_resolution_example.webp" alt="San Juan County — Delineate-Anything v2 on Landsat, Sentinel-2, SPOT and NAIP" width="800" loading="lazy"></a>

---

## San Juan County, New Mexico — FTW and Delineate-Anything v2 with and without the crop filter

**Example 20** (crop-filter comparison) · the pre-trained FTW
(`FTW_PRUE_EFNET_B5`; windows 11 Mar–10 May 2019, 22 images, shown in the top
panels, and 20 Sep–19 Nov 2019, 37 images) and Delineate Anything v2 (annual
2019 composite, 171 images, bottom), both used as released, each with the
Annual NLCD 2019 crop filter on (left) and off (right) · min. area 2,500 m² ·
window: the 3 km square with the most reference fields.

With the filter off, the polygons it would remove are outlined in magenta. The
filter kept 414 of FTW's 1,377 polygons and 380 of Delineate Anything v2's
562. Against the 944 NMOSE polygons (one-to-one matching at IoU ≥ 0.5; the
polygons were not used for training or fine-tuning in these runs):

| Engine | Crop filter | Fields | Precision | Recall | F1 | Predictions overlapping no reference polygon |
|---|---|---|---|---|---|---|
| FTW | on | 414 | 0.39 | 0.17 | 0.24 | 63 |
| FTW | off | 1,377 | 0.12 | 0.18 | 0.15 | 976 |
| Delineate Anything v2 | on | 379 | 0.58 | 0.23 | 0.33 | 12 |
| Delineate Anything v2 | off | 561 | 0.40 | 0.24 | 0.30 | 174 |

(Fields in the box, by representative point.) Of the 963 polygons the filter
removes from FTW's output, 913 overlap no reference polygon (162 of 182 for
Delineate Anything v2); in this window they are mostly in the riparian strip
along the river and in the built-up area. For both engines the
filter raises precision and lowers recall by less than 0.01. It also removes
some polygons that do overlap reference fields (a few are visible here).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/San_Juan_crop_filter_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/San_Juan_crop_filter_example.webp" alt="San Juan County — FTW and Delineate-Anything v2 with and without the crop filter" width="800" loading="lazy"></a>

---

## Pampas, Argentina — embeddings + SAM 2 vs Delineate-Anything v2 on Sentinel-2 and SPOT

**Example 15** · Pergamino partido, Buenos Aires Province, east of the city ·
label-free (no training and no reference data) · min. area 5,000 m²; Dynamic
World crop filter of each input's year (2024; 2023 for SPOT) · window: the
most centre pivots among the SAM-refined TESSERA fields.

- Top: Google Satellite Embedding and TESSERA v1 clusters of 2024, after the
  crop filter and SAM 2 on a Sentinel-2 composite of October 2024 (2,366 and
  1,918 fields; see the next entry).
- Bottom left: Delineate Anything v2 on the same October 2024 Sentinel-2
  composite (8 images): 2,818 fields (the crop filter kept 2,818 of 3,297).
- Bottom right: Delineate Anything v2 on SPOT 6/7 (6 m), 2023 (6 images;
  AIRBUS/SPOT6_7 ends on 2023-11-15): 3,306 fields (kept 3,306 of 3,761).

On Sentinel-2, Delineate-Anything outlines most pivots in the window as fields
of their own and splits a few along tone changes inside the circle. On SPOT it
leaves at least one faint pivot inside a larger rectangular field and breaks
the two-tone pivot at the top left into pieces. The two inputs differ in date,
season and resolution (SPOT: a 2023 median of 6 images at 6 m; Sentinel-2: an
October 2024 median of 8 images at 10 m) and in crop-filter year (2023 and
2024); which of these differences causes the different outlines was not
tested. The TESSERA clusters already follow most pivot circles before SAM 2
(see the next entry), while the Google clusters split several pivots, and
SAM 2 does not join the pieces (it returns one polygon per input polygon).
There is no reference layer here; the
panels compare outlines, not accuracy.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Pampas_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Pampas_example.webp" alt="Pampas — embeddings with SAM 2 vs Delineate-Anything v2 on Sentinel-2 and SPOT" width="800" loading="lazy"></a>

---

## Pampas, Argentina — Google Satellite Embedding and TESSERA, before and after SAM 2

**Example 15** (steps 1–4) · Label-free: no training and no reference data · Pergamino
partido, Buenos Aires Province, east of the city · min. area 5,000 m²;
Dynamic World 2024 crop filter · window: the most centre pivots among the
SAM-refined TESSERA fields.

Google Satellite Embedding (top) and TESSERA v1 (bottom) embeddings of 2024
are clustered (k = 5) and kept where they pass the crop filter (left); SAM 2
then refines them on a Sentinel-2 composite of October 2024, 8 images
(right). The crop filter kept 2,372 of 2,846 Google and 1,923 of 2,461
TESSERA polygons. SAM 2 was prompted for 442 Google and 511 TESSERA polygons
and refined 439 and 510 of them. The other 3 and 1 keep their input geometry:
one Google mask was empty, and the overlap trim removed 2 Google masks and
1 TESSERA mask entirely. The remaining 1,930 and 1,412 polygons were below the
size rule and were not prompted. After the 5,000 m² filter, 2,366 and 1,918
polygons remain; the refined fields are outlined in orange. The script smooths
and simplifies all polygons again after SAM. On the pivots, the TESSERA
clusters follow most circles, while the Google clusters split several of
them. SAM 2 redraws many pivot outlines along their edges; it returns one
polygon per input polygon, so it neither splits a field in two nor joins the
pieces of a split pivot, and it can leave holes inside a field along
within-field variation.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Pampas_SAM2_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Pampas_SAM2_example.webp" alt="Pampas — Google Satellite Embedding and TESSERA clusters before and after SAM 2" width="800" loading="lazy"></a>

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
visible in the SPOT-Pan image. Delineate Anything v2 produced 80,045 polygons
(78,340 inside the study area, 78,083 after the 100 m² filter); the crop
filter kept 55,995 of 78,083, with a median area of 0.12 ha. There is no
reference data here; neither output was evaluated. The example also clusters
Google Satellite Embedding and TESSERA embeddings (1,230 and 8,818 polygons
after the crop filter; Google with k = 5 chosen automatically, TESSERA with a
fixed k = 8; not shown).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/India_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/India_example.webp" alt="India — FTW on Sentinel-2 vs Delineate-Anything on SPOT-Pan" width="800" loading="lazy"></a>

---

## France, Beauce — FTW on Sentinel-2

**Example 04** · FTW on Sentinel-2 2023 · the Beauce, Eure-et-Loir, just east
of Bonneval and north-east of Châteaudun · min. area 5,000 m²; Dynamic World
2023 crop filter · FTW read two season windows chosen from the FTW crop
calendar, 27 Feb–28 Apr (4 images, shown) and 7 Aug–6 Oct 2023 (4 images).

665 fields (median 10.0 ha); the crop filter kept all of them.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/France_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/France_example.webp" alt="France — FTW on Sentinel-2" width="800" loading="lazy"></a>

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

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Kenya_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Kenya_example.webp" alt="Kenya — FTW at four minimum-area thresholds" width="800" loading="lazy"></a>

---

## California, USA — Delineate-Anything on NAIP

**Example 07** · Delineate Anything v2 on NAIP 1 m, 2022 (9 scenes, mosaicked,
from Earth Engine) · western San Joaquin Valley, Fresno County, near Five
Points · min. area 10,000 m²; Annual NLCD 2022 crop filter.

328 fields (median 22.6 ha); the crop filter kept all of them.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Central_Valley_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Central_Valley_example.webp" alt="California — Delineate-Anything on NAIP" width="800" loading="lazy"></a>

---

## California, USA — Delineate-Anything on USGS NAIP Plus (no Earth Engine)

**Example 16** · Delineate Anything v2 on USGS NAIP Plus (0.6 m, 2022; 8
source rasters exported from the USGS ImageServer) · the same study area as
example 07 · min. area 10,000 m²; no crop filter.

299 fields (median 22.1 ha). This source path needs no Earth Engine, so the
example runs without the crop filter, which reads its land-cover data from
Earth Engine.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/USGS_NAIP_Plus_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/USGS_NAIP_Plus_example.webp" alt="California — Delineate-Anything on USGS NAIP Plus" width="800" loading="lazy"></a>

---

## Australia, Murray-Darling Basin — Prithvi on HLS vs Delineate-Anything v2 on SPOT

**Example 03** · Narrabri Shire, New South Wales, just north of Narrabri ·
min. area 5,000 m²; Dynamic World crop filter · the Prithvi runs use the GFM
environment.

- Left: Prithvi-EO-2.0 `mode="embed"`: K-means on the patch tokens of the last
  encoder layer. The tokens are 16 pixels, 480 m, apart at 30 m and are
  interpolated to pixels. It merges neighbouring fields into 51 polygons
  (median 132 ha).
- Middle: `mode="pca"`, a baseline that uses no Prithvi weights: K-means on the
  PCA of per-band z-scores of R, G, B and NIR. It splits fields along
  within-field variation into 1,339 fragments (median 1.4 ha).
- Right: Delineate Anything v2 ("DA v2") used as released, on SPOT 6/7 (6 m),
  2023 (3 images; SPOT has no scene over this box in 2022): 427 fields that
  follow the rectangular field edges.

The left and middle panels use HLS 2022 (30 m, 298 scenes); the right panel
is a year later. There is no reference layer here, so the panels compare
outlines, not accuracy.

Neither mode delineates field instances; both cluster pixels into land-cover
segments, and neither was trained or fine-tuned on field boundaries in these
runs. The embed mode's over-merging, first seen in 0.1.x, remains with the
corrected HLS radiometry of 1.0. Prithvi's supervised `mode="segment"` needs
fine-tuning on reference polygons (`fine_tune=True`) and was not run here.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Australia_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Australia_example.webp" alt="Australia — Prithvi embed mode and a PCA baseline on HLS, and Delineate-Anything v2 on SPOT" width="800" loading="lazy"></a>

---

## North China Plain — Delineate-Anything on SPOT

**Example 08** · Delineate Anything v2 on SPOT 6/7 (6 m), 2023 (5 scenes;
restricted SPOT access) · Hengshui (Jizhou District), Hebei · min. area
3,000 m²; Dynamic World 2023 crop filter · window: the 4 km square with the
most Delineate-Anything polygons.

4,517 strip fields in the study area (median 0.60 ha); the crop filter kept
4,517 of 4,560.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/China_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/China_example.webp" alt="China — Delineate-Anything on SPOT" width="800" loading="lazy"></a>

---

## Spain, Andalusia — Delineate-Anything, FTW and their vote merge

**Example 09** · Delineate Anything v2 and FTW on Sentinel-2 2024, combined
afterwards from the saved outputs · Seville province, east of Carmona (the
3 km window is at the study area's east edge, in the Carmona and Fuentes de
Andalucía municipalities; the most Delineate-Anything polygons) · min. area
2,500 m²; Dynamic World 2024 crop filter · Delineate-Anything read the annual
composite (66 images, shown); FTW read 12 Mar–11 May and 11 Sep–10 Nov 2024
(4 images each).

Delineate-Anything returns 1,544 fields (median 4.2 ha) and FTW 469 (median
6.5 ha). The vote merge keeps the pixels that both engines cover, on a 10 m
grid, which gives it its stair-stepped outlines. Neighbouring fields that are
both kept fuse into one polygon: 793 polygons, 591 after the 2,500 m² filter.
The example also writes the intersection (1,446) and union (695) merges,
which are computed on the vector polygons.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Spain_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Spain_example.webp" alt="Spain — Delineate-Anything, FTW and vote merge" width="800" loading="lazy"></a>

---

## Mississippi Alluvial Plain, USA — Delineate-Anything on SPOT, 2021–2023

**Example 11** · Delineate Anything v2 on SPOT 6/7 (6 m) for 2021, 2022 and
2023, each panel on its own year's composite (restricted SPOT access) · near
Greenville, Washington County, Mississippi (about 6 % of the study area, in
its north-western corner, is in Chicot County, Arkansas, by US Census county
boundaries; the 4 km window, the square with the most 2023 polygons, centre
33.42° N, 90.93° W, is in Mississippi) · min. area 10,000 m²; Annual NLCD
crop filter of each year.

2,006, 1,284 and 1,764 fields. The 2022 composite has only 2 SPOT scenes
(16 in 2021, 12 in 2023). That year has the fewest polygons and the largest
median area (7.8 ha, against 4.2 and 5.0 ha), while the total delineated area
is similar; these runs do not test whether the scene count is the cause.
Year-to-year agreement (one-to-one matching at IoU ≥ 0.5) is F1 0.49 from
2021 to 2022 and 0.55 from 2022 to 2023; this compares predictions with each
other and says nothing about accuracy.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/MAP_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/MAP_example.webp" alt="Mississippi Alluvial Plain — Delineate-Anything on SPOT, 2021 to 2023" width="800" loading="lazy"></a>

---

## San Juan County, New Mexico — evaluation against NMOSE

**Example 20** · Delineate Anything v2 on Sentinel-2 2019 (171 images) ·
San Juan County, New Mexico (Farmington–Bloomfield area) · min. area
2,500 m²; Annual NLCD 2019 crop filter · evaluated against the NMOSE WUCB
polygons (cyan), which were not used for training or fine-tuning in these
runs (Delineate Anything v2 is used as released; whether its training set,
FBIS-73M, includes them was not checked).

Of the 380 predictions, 379 lie inside the evaluation box (by representative
point; the panel title counts all 380, and the other one lies across the
box's south edge), against 944 reference fields. One-to-one matching at
IoU ≥ 0.5 gives precision 0.58, recall 0.23 and F1 0.33
([95 % bootstrap intervals](user-guide/evaluation.md) from 200 resamples of
reference fields within sub-basins: F1 0.31–0.36, recall 0.21–0.25; they
treat fields as independent, so they are too narrow if errors are spatially
clustered); the mean IoU of matched fields is 0.80. Area-weighted recall is
0.61 and area-weighted precision 0.84. Recall rises with field size and then
levels off: it is 0.01 for fields of 0.2–0.5 ha and about 0.7 for fields above
10 ha. F1 broadly follows but dips at 5–10 ha (0.32, against 0.41 at 2–5 ha)
and at 50–100 ha (0.59, against 0.69 at 20–50 ha). By sub-basin, NIIP (the
Navajo Indian Irrigation Project; 97 reference fields, all 18 ha or larger)
reaches F1 0.72, and the Animas River (324 reference fields, median 0.83 ha)
F1 0.18. Two caveats: the reference was digitised from 2016 NAIP, so changes
by 2019 count as errors, and the crop filter removed 182 of 562 polygons
before the evaluation.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/San_Juan_evaluation_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/San_Juan_evaluation_example.webp" alt="San Juan County — Delineate-Anything vs NMOSE reference" width="800" loading="lazy"></a>

---

## SAM 2 refinement — example 20's output before and after

**Example 13** · SAM 2 box-prompted refinement of example 20's
Delineate-Anything output (Sentinel-2 2019). The default input of this
example is example 12's output, which was not run for 1.0.0 · window: the
most SAM-refined fields, the centre pivots of the Navajo Indian Irrigation
Project, San Juan County.

At 10 m, only 67 of the 380 fields pass the SAM size rule (all 19 ha or
larger); the other 313 are not refined. After SAM, the example smooths
(Chaikin ×3) and simplifies all 380 polygons again, although example 20's
output was already smoothed, so the before/after numbers include this second
smoothing. Against the NMOSE polygons, object F1 is unchanged (0.332, 220
matches before and after), and the mean boundary distance falls from 13.4 m
to 12.5 m, with SAM and the second smoothing together. SAM replaces each
prompted polygon with its own mask and never merges or deletes polygons
(380 in, 380 out), so Delineate-Anything's splits inside pivots remain; with
`sam_overlaps="trim"` a refined mask also cannot take area from a
neighbouring polygon (14 masks were trimmed). (Example 13 evaluates all 380
polygons; example 20 evaluates the 379 inside its box.)

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/SAM2_refinement_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/SAM2_refinement_example.webp" alt="SAM 2 refinement before and after" width="800" loading="lazy"></a>

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
its representative point: 1,177 tile polygons become 302. No polygon reached
a halo edge, so fields that cross the core edges come out whole; one field was
delineated by two tiles and appears twice (the pair is not marked).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/HPC_tiling_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/HPC_tiling_example.webp" alt="HPC tiling — four tiles merged" width="800" loading="lazy"></a>
