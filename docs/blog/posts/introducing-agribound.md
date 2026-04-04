---
date: 2026-04-04
authors:
  - montimaj
categories:
  - Release
  - Announcements
  - Community
tags:
  - field-boundaries
  - satellite-imagery
  - geospatial-ai
  - google-earth-engine
  - usgs-naip-plus
  - community-contribution
---

# Introducing Agribound: Unified Field Boundary Delineation from Satellite Imagery

We are excited to announce the public release of **agribound**, a Python package that unifies seven complementary approaches to agricultural field boundary delineation into a single, reproducible pipeline. Whether you are working with 1 m NAIP imagery over U.S. croplands or 10 m Sentinel-2 composites anywhere in the world, agribound lets you go from raw satellite data to clean, vectorized field boundary polygons in a single function call.

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/Argribounds_Lea_County.png" alt="Agribound field boundaries over Lea County, NM" width="700">
  <figcaption>Agribound-delineated field boundaries over Lea County, New Mexico (DINOv3 fine-tuned + SAM2 on NAIP). Image credit: Jayden Atkin, DRI.</figcaption>
</figure>

<!-- more -->

## The Problem

Agricultural field boundary delineation is essential for crop monitoring, yield estimation, water resource management, and precision agriculture. However, the landscape of available tools and models is fragmented:

- **Object detection** approaches (YOLO-based) are fast but may miss irregular shapes.
- **Semantic segmentation** models (FTW, UNet) generalize well but require post-processing to extract individual field instances.
- **Foundation models** (DINOv3, Prithvi-EO-2.0) offer powerful learned representations but need fine-tuning for best results.
- **Embedding-based methods** (Google Satellite Embeddings, TESSERA) enable unsupervised delineation without any labeled data but require clustering and refinement.

Each approach has its own data format expectations, preprocessing requirements, and output conventions. Researchers and practitioners end up maintaining dozens of ad hoc scripts to stitch these workflows together.

## What Agribound Provides

Agribound wraps all of these into a unified pipeline:

```
Satellite composite --> [Optional fine-tuning] --> Delineation engine --> Post-processing --> LULC crop filter --> Export
```

A minimal example:

```python
import agribound

gdf = agribound.delineate(
    study_area="fields.geojson",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
)
```

This single call handles GEE authentication, cloud-free composite generation, engine inference, polygon smoothing and simplification, LULC-based non-agricultural polygon removal, and export to a GeoDataFrame with area, perimeter, and provenance metadata.

### Seven Delineation Engines

| Engine | Approach | GPU Required |
|---|---|---|
| Delineate-Anything | YOLO instance segmentation | Recommended |
| Fields of The World (FTW) | Semantic segmentation (14+ models) | Yes |
| GeoAI Field Boundary | Mask R-CNN | No |
| DINOv3 | Satellite-pretrained ViT + DPT head | Yes |
| Prithvi-EO-2.0 | NASA/IBM ViT foundation model | Recommended |
| Embedding | Unsupervised clustering | No |
| Ensemble | Multi-engine consensus | Depends |

### Nine Satellite Sources

Agribound supports Landsat, Sentinel-2, HLS, NAIP, USGS NAIP Plus, SPOT 6/7, local GeoTIFFs, and pre-computed embedding datasets (Google Satellite Embeddings, TESSERA) -- all through a consistent interface. The USGS NAIP Plus source provides the same NAIP imagery available on GEE but acquired directly from the [USGS USGSNAIPPlus ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer), enabling high-resolution field delineation without GEE authentication.

### Automatic LULC Crop Filtering

A key differentiator: unlike other tools that detect *all* visual boundaries (including roads, water bodies, forests, and buildings), agribound **automatically removes non-agricultural polygons** using the best available LULC dataset for your study area:

- **CONUS:** USGS NLCD (1985--2024, 30 m)
- **Global (2015+):** Google Dynamic World (10 m)
- **Global (pre-2015):** Copernicus C3S Land Cover (300 m)

This is enabled by default and requires no configuration.

## Example Results

### Supervised: DINOv3 + SAM2 on NAIP (New Mexico, USA)

DINOv3 fine-tuned on NMOSE reference boundaries with LULC filtering and SAM2 per-field refinement on 1 m NAIP imagery. The resulting boundary linework approaches the quality of manual human digitization:

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/NM_field.png" alt="Center-pivot irrigated fields in New Mexico" width="700">
  <figcaption>Center-pivot irrigated fields in eastern New Mexico delineated by DINOv3 (fine-tuned) + SAM2 on NAIP — the boundary linework approaches the quality of manual human digitization. Image credit: Jayden Atkin, DRI.</figcaption>
</figure>

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/NM_example.png" alt="DINOv3 + SAM2 on NAIP" width="700">
  <figcaption>Blue = predicted boundaries, Yellow = NMOSE reference boundaries. Eastern Lea County, 2020.</figcaption>
</figure>

### Unsupervised: TESSERA + LULC + SAM2 (Pampas, Argentina)

Fully automated -- no training data, no reference boundaries. TESSERA embedding clustering with Dynamic World crop filtering and SAM2 refinement:

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/Pampas_example.png" alt="TESSERA + LULC + SAM2" width="700">
  <figcaption>Pergamino, Argentina, 2024. Delineated without any labeled data.</figcaption>
</figure>

## Getting Started

Install agribound:

```bash
pip install agribound
```

For GPU engines and GEE support:

```bash
pip install agribound[gee,delineate-anything]
```

Check out the [Quickstart tutorial](../../user-guide/quickstart.md) for a complete walkthrough, or browse the [Gallery](../../gallery.md) for results across nine regions, five satellites, and all engines.

## Community Recognition

Agribound was [highlighted by the TESSERA team](https://geotessera.org/blog/2026-04-01-agribound-field-boundaries) at the University of Cambridge's Centre for Earth Observation for its integration of TESSERA embeddings into an end-to-end field boundary delineation pipeline.

The [launch announcement on LinkedIn](https://www.linkedin.com/posts/sayantanmajumdar_opensource-opensource-activity-7444461276329836544-n2PF) received over 500 reactions and 55 reposts from the geospatial and remote sensing community within two days of its release.

## First Community Contribution

We are thrilled to highlight agribound's **first community contribution** from **[Jeremy Rapp](https://espp.msu.edu/directory/rapp-jeremy.html)** at the Department of Earth and Environmental Sciences, Michigan State University. Jeremy contributed [Example 16](https://github.com/montimaj/agribound/blob/main/examples/16_usa_usgs_naip_plus.py), which adds support for the **USGS NAIP Plus ImageServer** -- the same NAIP imagery available on GEE but acquired directly from the [USGS USGSNAIPPlus ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer) -- as a non-GEE high-resolution imagery source.

This example demonstrates agribound's local-raster acquisition path: the AOI is queried directly from the USGS ImageServer, exported to a local GeoTIFF, and then passed into the Delineate-Anything engine pipeline -- all **without requiring Google Earth Engine authentication**. This is a significant addition for users who want to work with 1 m NAIP imagery but do not have GEE access or prefer a purely local workflow.

Jeremy's contribution showcases exactly the kind of community-driven extension we hoped agribound would enable: identifying a new data source, integrating it into the existing pipeline, and providing a complete working example. Thank you, Jeremy, for this excellent contribution!

## What's Next

- A paper submission to the *Journal of Open Source Software* is in preparation.
- We are expanding engine support and adding new embedding datasets as they become available.
- Community contributions are welcome -- see the [Contributing guide](../../contributing.md).

## Citation

If you find agribound useful, please cite:

> Majumdar, S., Huntington, J. L., ReVelle, P., Nozari, S., Smith, R. G., Hasan, M. F., Bromley, M., Atkin, J., Rapp, J., Jensen, E. R., Ketchum, D., & Roy, S. (2026). *Agribound: Unified agricultural field boundary delineation from satellite imagery using geospatial foundation models, pre-trained segmentation, and embeddings* [Software]. Zenodo. [https://doi.org/10.5281/zenodo.19229665](https://doi.org/10.5281/zenodo.19229665)

> Majumdar, S., Huntington, J. L., ReVelle, P., Nozari, S., Smith, R. G., Hasan, M. F., Bromley, M., Atkin, J., Rapp, J., Jensen, E. R., Ketchum, D., & Roy, S. (2026). *Agribound: Unified agricultural field boundary delineation from satellite imagery using geospatial foundation models, pre-trained segmentation, and embeddings*. In prep. for *Journal of Open Source Software*.

Give the [repo](https://github.com/montimaj/agribound) a star if you find it useful!
