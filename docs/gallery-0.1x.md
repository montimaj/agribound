# Example Gallery (agribound 0.1.x, archived)

!!! info "Archived page"
    This is the agribound 0.1.x gallery, with the same images as published.
    Some captions were edited for 1.0.0. FTW model names now read "pre-trained
    model", because 0.1.x used the same default FTW model (FTW_PRUE_EFNET_B5)
    for every region. Years that did not match the 0.1.x example scripts were
    removed. The Prithvi remark about PCA and ViT output was qualified. The
    page is kept for reference only. The current gallery, rendered from the
    1.0.1 example runs, is [Example Gallery (1.0)](gallery.md). The images
    below are stored under `assets/gallery_0.1x/`.

Visual results from agribound 0.1.x example scripts across different regions, satellites, and engines.

!!! warning "Produced with agribound 0.1.x"
    These screenshots were made with agribound 0.1.x and have **not** been
    regenerated with agribound 1.0. Several 0.1.x defects can affect them: FTW
    received two copies of the annual composite instead of two season
    windows; Delineate-Anything on non-Sentinel-2 sources could run a
    fallback with swapped red and blue channels; HLS and Landsat inputs were
    on the wrong radiometric scale; composites were clipped to the study area;
    and caches ignored the study area and year. See the
    [list of affected results](https://github.com/montimaj/agribound/blob/main/CHANGELOG.md#results-produced-with-agribound--013-that-are-affected).
    The captions describe the 0.1.x configuration of each example (as
    corrected above); the example scripts have since been updated for 1.0,
    and the years they use may differ.

!!! note
    The satellite basemap in these screenshots may not correspond to the same acquisition date as the imagery used for delineation. Field boundaries and crop patterns may differ between the basemap and the analysis period.

---

## New Mexico, USA — DINOv3 + SAM2 on NAIP

**Example 14** · NAIP (1 m) · DINOv3 (SAT-493M) fine-tuned on NMOSE reference boundaries · LULC crop filter (NLCD) · SAM2 per-field refinement · Eastern Lea County (2020). Note: Fields in Texas bordering New Mexico are also present.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/NM_example.png" alt="New Mexico — DINOv3 + SAM2 on NAIP" width="700">

---

## Pampas, Argentina — TESSERA + LULC + SAM2

**Example 15** · Fully automated (no training, no reference data) · TESSERA (128-D) embedding clustering · LULC crop filter (Dynamic World) · SAM2 refinement on TESSERA embedding dimensions (see the correction below) · Pergamino (2024).

!!! note "Correction (agribound 1.0.1)"
    The 0.1.x README captioned this screenshot "SAM2 boundary refinement on
    Sentinel-2". Matched against the saved 0.1.x outputs of example 15, the red
    outlines are the example's last variant instead: SAM2 on three TESSERA
    embedding dimensions used as a pseudo-RGB image, after splitting
    multi-part polygons, with polygons over 50 ha kept unrefined
    (`fields_sam2_tessera_improved_2024.gpkg`). SAM2 changed little of this
    layer: the unrefined polygons over 50 ha hold 66 % of its area. The view is
    about 23 km wide with thick outlines, so single pixels and small
    fragments are not visible. The same layer from the 1.0.1 run is
    `fields_tessera_crop_sam2-tessera-split_2024.gpkg`; the 1.0 gallery shows both
    layers in this screenshot's frame
    ([Compared with the 0.1.x README image](gallery.md#compared-with-the-01x-readme-image)).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/Pampas_example.png" alt="Pampas — TESSERA + LULC + SAM2" width="700">

---

## India, West Bengal — FTW on Sentinel-2

**Example 02** · Sentinel-2 (10 m) · FTW pre-trained model · Nadia District, West Bengal — smallholder rice paddies (2024).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/India_example.png" alt="India — FTW on Sentinel-2" width="600">

---

## France, Beauce — FTW on Sentinel-2

**Example 04** · Sentinel-2 (10 m) · FTW pre-trained model · Large-field cereal agriculture in the Beauce plain (2023).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/France_example.png" alt="France — FTW on Sentinel-2" width="700">

---

## Kenya — FTW on Sentinel-2

**Example 06** · Sentinel-2 (10 m) · FTW pre-trained model · Central Kenya smallholder fields with `min_area` tuning.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/Kenya_example.png" alt="Kenya — FTW on Sentinel-2" width="600">

---

## California, USA — Delineate-Anything on NAIP

**Example 07** · NAIP (1 m) · Delineate-Anything (YOLO) · Central Valley large commercial agriculture (2022).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/Central_Valley_example.png" alt="California — DA on NAIP" width="700">

---

## Australia, Murray-Darling Basin — Prithvi PCA on HLS

**Example 03** · HLS (30 m) · Prithvi PCA baseline · Large-scale irrigated agriculture in the Murray-Darling Basin (2022). PCA mode runs without a GPU. In 0.1.x the ViT embedding mode tended to over-merge fields into very few large polygons; its HLS input was then on the wrong radiometric scale (fixed in 1.0.0), so that observation may not carry over.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/Australia_example.png" alt="Australia — Prithvi PCA on HLS" width="700">

---

## North China Plain — Delineate-Anything on SPOT

**Example 08** · SPOT 6/7 (6 m) · Delineate-Anything · Smallholder wheat/maize fields. Restricted SPOT access.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/China_example.png" alt="China — DA on SPOT" width="700">

---

## Spain, Andalusia — Ensemble (DA + FTW)

**Example 09** · Sentinel-2 (10 m) · Multi-engine vote-merge of Delineate-Anything and FTW · Olive groves and cereal fields.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/Spain_example.png" alt="Spain — Ensemble" width="700">

---

## Mississippi Alluvial Plain, USA — Delineate-Anything on SPOT

**Example 11** · SPOT 6/7 (6 m) · Delineate-Anything · Row-crop agriculture with cross-year stability analysis (2021–2023). Restricted SPOT access.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/MAP_example.png" alt="Mississippi Alluvial Plain — DA on SPOT" width="700">
