# Example Gallery

Visual results from agribound's example scripts across different regions, satellites, and engines.
!!! note
    The satellite basemap in these screenshots may not correspond to the same acquisition date as the imagery used for delineation. Field boundaries and crop patterns may differ between the basemap and the analysis period.

---

## New Mexico, USA — DINOv3 + SAM2 on NAIP

**Example 14** · NAIP (1 m) · DINOv3 (SAT-493M) fine-tuned on NMOSE reference boundaries · LULC crop filter (NLCD) · SAM2 per-field refinement · Eastern Lea County (2020). Note: Fields in Texas bordering New Mexico are also present.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/NM_example.png" alt="New Mexico — DINOv3 + SAM2 on NAIP" width="700">

---

## Pampas, Argentina — TESSERA + LULC + SAM2

**Example 15** · Fully automated (no training, no reference data) · TESSERA (128-D) embedding clustering · LULC crop filter (Dynamic World) · SAM2 refinement on Sentinel-2 · Pergamino (2024).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/Pampas_example.png" alt="Pampas — TESSERA + LULC + SAM2" width="700">

---

## India, West Bengal — FTW on Sentinel-2

**Example 02** · Sentinel-2 (10 m) · FTW supervised model for India · Nadia District, West Bengal — smallholder rice paddies (2024).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/India_example.png" alt="India — FTW on Sentinel-2" width="600">

---

## France, Beauce — FTW on Sentinel-2

**Example 04** · Sentinel-2 (10 m) · FTW pre-trained model for France · Large-field cereal agriculture in the Beauce plain (2023).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/France_example.png" alt="France — FTW on Sentinel-2" width="700">

---

## Kenya — FTW on Sentinel-2

**Example 06** · Sentinel-2 (10 m) · FTW supervised model · Central Kenya smallholder fields with `min_area` tuning (2024).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/Kenya_example.png" alt="Kenya — FTW on Sentinel-2" width="600">

---

## California, USA — Delineate-Anything on NAIP

**Example 07** · NAIP (1 m) · Delineate-Anything (YOLO) · Central Valley large commercial agriculture (2022).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/Central_Valley_example.png" alt="California — DA on NAIP" width="700">

---

## Australia, Murray-Darling Basin — Prithvi PCA on HLS

**Example 03** · HLS (30 m) · Prithvi PCA baseline · Large-scale irrigated agriculture in the Murray-Darling Basin (2022). PCA mode produces realistic field boundaries without a GPU. The ViT embedding mode (without fine-tuning) tends to over-merge fields into very few large polygons — fine-tuning on reference boundaries is recommended for ViT.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/Australia_example.png" alt="Australia — Prithvi PCA on HLS" width="700">

---

## North China Plain — Delineate-Anything on SPOT

**Example 08** · SPOT 6/7 (6 m) · Delineate-Anything · Smallholder wheat/maize fields (2023). Restricted SPOT access.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/China_example.png" alt="China — DA on SPOT" width="700">

---

## Spain, Andalusia — Ensemble (DA + FTW)

**Example 09** · Sentinel-2 (10 m) · Multi-engine vote-merge of Delineate-Anything and FTW · Olive groves and cereal fields (2023).

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/Spain_example.png" alt="Spain — Ensemble" width="700">

---

## Mississippi Alluvial Plain, USA — Delineate-Anything on SPOT

**Example 11** · SPOT 6/7 (6 m) · Delineate-Anything · Row-crop agriculture with cross-year stability analysis (2021–2023). Restricted SPOT access.

<img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/MAP_example.png" alt="Mississippi Alluvial Plain — DA on SPOT" width="700">
