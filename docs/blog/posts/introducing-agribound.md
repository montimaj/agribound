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

!!! warning "Historical post"
    This post describes agribound 0.1.x and is kept as it was published,
    except for these changes made in 1.0.0: a comparison with other tools was
    removed; the statement that the New Mexico linework "approaches the
    quality of manual human digitization", for which no evaluation was made,
    was removed; the gallery sentence now links to the archived 0.1.x
    gallery and no longer says it covered nine regions, five satellites and
    all engines (it had ten study areas and no GeoAI example); the manuscript
    title in the citation was updated (see the note there); and links were
    updated: the images now point to `assets/gallery_0.1x/`, and the TESSERA
    post is named and linked through the TESSERA blog index.
    Several statements in it no longer hold for agribound 1.0.0, and some
    0.1.x results were affected by defects fixed in 1.0.0 (see the
    [v1.0.0 release post](v1.0.0-release.md), the
    [migration guide](../../migration-1.0.md) and the
    [CHANGELOG](https://github.com/montimaj/agribound/blob/main/CHANGELOG.md#results-produced-with-agribound--013-that-are-affected)).

We are excited to announce the public release of **agribound**, a Python package that unifies seven complementary approaches to agricultural field boundary delineation into a single, reproducible pipeline. Whether you are working with 1 m NAIP imagery over U.S. croplands or 10 m Sentinel-2 composites anywhere in the world, agribound lets you go from raw satellite data to clean, vectorized field boundary polygons in a single function call.

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/Argribounds_Lea_County.png" alt="Agribound field boundaries over Lea County, NM" width="700">
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

The delineation engines detect visual boundaries of all kinds (including roads, water bodies, forests, and buildings), so agribound **automatically removes non-agricultural polygons** using the best available LULC dataset for your study area:

- **CONUS:** USGS NLCD (1985--2024, 30 m)
- **Global (2015+):** Google Dynamic World (10 m)
- **Global (pre-2015):** Copernicus C3S Land Cover (300 m)

This is enabled by default and requires no configuration.

## Example Results

### Supervised: DINOv3 + SAM2 on NAIP (New Mexico, USA)

DINOv3 fine-tuned on NMOSE reference boundaries with LULC filtering and SAM2 per-field refinement on 1 m NAIP imagery:

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/NM_field.png" alt="Center-pivot irrigated fields in New Mexico" width="700">
  <figcaption>Center-pivot irrigated fields in eastern New Mexico delineated by DINOv3 (fine-tuned) + SAM2 on NAIP. Image credit: Jayden Atkin, DRI.</figcaption>
</figure>

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/NM_example.png" alt="DINOv3 + SAM2 on NAIP" width="700">
  <figcaption>Blue = predicted boundaries, Yellow = NMOSE reference boundaries. Eastern Lea County, 2020.</figcaption>
</figure>

### Unsupervised: TESSERA + LULC + SAM2 (Pampas, Argentina)

Fully automated -- no training data, no reference boundaries. TESSERA embedding clustering with Dynamic World crop filtering and SAM2 refinement:

<figure markdown="span">
  <img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_0.1x/Pampas_example.png" alt="TESSERA + LULC + SAM2" width="700">
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

Check out the [Quickstart tutorial](../../user-guide/quickstart.md) for a complete walkthrough, or browse the [archived 0.1.x Gallery](../../gallery-0.1x.md) for the results shown at the time: ten study areas; NAIP, Sentinel-2, HLS and SPOT 6/7 imagery and TESSERA embeddings; and every engine except GeoAI, which had no gallery example. The [current gallery](../../gallery.md) shows the 1.0.0 runs.

## Community Recognition

Agribound was highlighted by the TESSERA team, in the post "Agribound: agricultural field boundary delineation" (1 April 2026) on the [TESSERA blog](https://geotessera.org/blog/), at the University of Cambridge's Centre for Earth Observation for its integration of TESSERA embeddings into an end-to-end field boundary delineation pipeline.

The [launch announcement on LinkedIn](https://www.linkedin.com/posts/sayantanmajumdar_opensource-opensource-activity-7444461276329836544-n2PF) received over 500 reactions and 55 reposts from the geospatial and remote sensing community within two days of its release.

## First Community Contribution

We are thrilled to highlight agribound's **first community contribution** from **[Jeremy Rapp](https://espp.msu.edu/directory/rapp-jeremy.html)** at the Department of Earth and Environmental Sciences, Michigan State University. Jeremy contributed [Example 16](https://github.com/montimaj/agribound/blob/main/examples/16_usa_usgs_naip_plus.py), which adds support for the **USGS NAIP Plus ImageServer** -- the same NAIP imagery available on GEE but acquired directly from the [USGS USGSNAIPPlus ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer) -- as a non-GEE high-resolution imagery source.

This example demonstrates agribound's local-raster acquisition path: the AOI is queried directly from the USGS ImageServer, exported to a local GeoTIFF, and then passed into the Delineate-Anything engine pipeline -- all **without requiring Google Earth Engine authentication**. This is a significant addition for users who want to work with 1 m NAIP imagery but do not have GEE access or prefer a purely local workflow.

Jeremy's contribution showcases exactly the kind of community-driven extension we hoped agribound would enable: identifying a new data source, integrating it into the existing pipeline, and providing a complete working example. Thank you, Jeremy, for this excellent contribution!

## What's Next

- A paper submission to *Remote Sensing of Environment* is in preparation.
- We are expanding engine support and adding new embedding datasets as they become available.
- Community contributions are welcome -- see the [Contributing guide](../../contributing.md).

## Citation

!!! note "Updated citation"
    The manuscript title given in the original version of this post has been
    superseded. The entries below are the current ones; the
    [Citation page](../../citation.md) is kept up to date.

If you find agribound useful, please cite:

> Majumdar, S., Rapp, J., Huntington, J. L., ReVelle, P., Nozari, S., Smith, R. G., Hasan, M. F., Bromley, M., Atkin, J., Jensen, E. R., Ketchum, D., & Roy, S. (2026). *Agribound: Unified agricultural field boundary delineation from satellite imagery using geospatial foundation models, pre-trained segmentation, and embeddings* [Software]. Zenodo. [https://doi.org/10.5281/zenodo.19229665](https://doi.org/10.5281/zenodo.19229665)

> Majumdar, S., Rapp, J., Huntington, J. L., ReVelle, P., Nozari, S., Smith, R. G., Hasan, M. F., Bromley, M., Atkin, J., Jensen, E. R., Ketchum, D., & Roy, S. (2026). *Measuring what geospatial AI delivers for policy-grade agricultural field boundaries*. In prep. for *Remote Sensing of Environment*.

Give the [repo](https://github.com/montimaj/agribound) a star if you find it useful!
