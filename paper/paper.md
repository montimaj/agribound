---
title: 'Agribound: Unified agricultural field boundary delineation from satellite imagery using geospatial foundation models, pre-trained segmentation, and embeddings'
tags:
  - Python
  - agriculture
  - field boundaries
  - remote sensing
  - deep learning
  - Google Earth Engine
  - land use land cover
authors:
  - name: Sayantan Majumdar
    orcid: 0000-0002-1037-0481
    affiliation: 1
    corresponding: true
  - name: Justin L. Huntington
    orcid: 0009-0008-2006-2969
    affiliation: 1
  - name: Peter ReVelle
    orcid: 0000-0001-8266-3753
    affiliation: 1
  - name: Soheil Nozari
    orcid: 0000-0002-0098-5837
    affiliation: 2
  - name: Ryan G. Smith
    orcid: 0000-0002-3747-6868
    affiliation: 2
  - name: M. F. Hasan
    orcid: 0000-0002-9520-983X
    affiliation: 2
  - name: Matt Bromley
    orcid: 0000-0002-2169-3307
    affiliation: 1
  - name: Jayden Atkin
    affiliation: 1
  - name: Eric R. Jensen
    orcid: 0000-0003-4208-5041
    affiliation: 1
  - name: David Ketchum
    orcid: 0000-0003-0871-9055
    affiliation: 3
  - name: Samapriya Roy
    affiliation: 1
affiliations:
  - name: Division of Hydrologic Sciences, Desert Research Institute, Reno, NV, USA
    index: 1
  - name: Department of Civil and Environmental Engineering, Colorado State University, Fort Collins, CO, USA
    index: 2
  - name: Montana Climate Office, W.A. Franke College of Forestry and Conservation, University of Montana, Missoula, MT, USA
    index: 3
date: 28 March 2026
bibliography: paper.bib
---

<!-- TODO: This paper is a work in progress. -->
<!-- TODO: Add architecture diagram (Figure 1) -->
<!-- TODO: Add benchmark results table comparing engines across study sites -->
<!-- TODO: Add LULC filtering before/after comparison figure -->
<!-- TODO: Review and update all references after final submission -->

# Summary

Accurate agricultural field boundary delineation from satellite imagery is essential for crop monitoring, yield estimation, water resource management, and land use policy. While several deep learning models have been developed for this task --- including object detection [@lavreniuk2025], semantic segmentation [@kerner2025], and vision transformer approaches [@simeoni2025; @szwarcman2024] --- each exists in its own ecosystem with incompatible input formats, band ordering conventions, and output schemas. Researchers working on field boundary mapping must navigate multiple repositories, write custom preprocessing scripts for each model, and manually clean outputs that include non-agricultural areas.

Agribound addresses this fragmentation by providing a unified Python package that wraps seven complementary delineation engines behind a single `agribound.delineate()` interface. It handles the full pipeline from satellite composite generation through Google Earth Engine [@gorelick2017] to vectorized, post-processed field boundary polygons, supporting eight satellite sources spanning 1984 to the present.

# Statement of Need

Field boundary delineation is a critical input for agricultural applications worldwide, yet the current landscape of tools is fragmented. Models such as Delineate-Anything [@lavreniuk2025], Fields of The World [@kerner2025], GeoAI [@wu2026geoai], DINOv3 [@simeoni2025], and Prithvi-EO-2.0 [@szwarcman2024] each require different satellite bands, resolutions, and preprocessing steps. Running multiple models for comparison or ensembling requires extensive boilerplate code. Furthermore, all existing models detect *visual* boundaries in imagery --- they segment any visually distinct region (roads, water bodies, forests, parking lots) without distinguishing agricultural fields from other land uses.

Agribound fills this gap with three key contributions:

1. **Unified multi-engine interface.** A single function call or CLI command runs any of seven engines on any of eight satellite sources, handling band extraction, resolution matching, and format conversion automatically.

2. **Automatic LULC crop filtering.** Agribound is the first field boundary package to automatically remove non-agricultural polygons from the output. It leverages three land-use/land-cover datasets --- USGS Annual NLCD for the contiguous United States (1985--2024, 30 m) [@dewitz2024], Google Dynamic World globally (2015--present, 10 m) [@brown2022], and Copernicus C3S Land Cover for pre-2015 global coverage (1992--2022, 300 m) [@defourny2023] --- selecting the appropriate dataset based on study area location and year. All zonal statistics are computed server-side on Google Earth Engine, requiring no raster downloads.

3. **Zero-training delineation pipeline.** By combining embedding-based unsupervised clustering [@brown2025; @feng2025] with Dynamic World crop probability filtering and SAM2 [@ravi2024] boundary refinement on Sentinel-2 imagery, Agribound enables fully automated field boundary delineation without any human-labeled reference data or model training.

# State of the Field

Several open-source tools address parts of the field boundary delineation problem, but none provide a unified end-to-end solution:

- **Delineate-Anything** [@lavreniuk2025] uses YOLO-based instance segmentation for resolution-agnostic boundary detection, but requires manual satellite preprocessing and produces boundaries for all land cover types without semantic filtering.
- **Fields of The World** [@kerner2025] provides a 25-country semantic segmentation benchmark with pre-trained models, but is limited to Sentinel-2 input with specific band ordering and temporal window requirements.
- **GeoAI** [@wu2026geoai] offers a Mask R-CNN pipeline for field delineation, but is tightly coupled to its training data format and does not support multi-source ensembling.
- **DINOv3** [@simeoni2025] provides powerful vision transformer features suitable for segmentation, but requires fine-tuning on reference boundaries for meaningful results.
- **Prithvi-EO-2.0** [@szwarcman2024] is a NASA/IBM geospatial foundation model with strong multi-temporal capabilities, but requires TerraTorch for fine-tuning and has limited out-of-the-box field boundary support.
- **samgeo** [@wu2023samgeo] wraps Meta's Segment Anything Model for geospatial applications but operates as a general segmentation tool without field-boundary-specific post-processing.

None of these tools integrate satellite composite generation, support multiple imagery sources, provide automatic LULC-based crop filtering, or enable multi-engine ensembling. Agribound unifies all of these capabilities in a single package.

# Design and Implementation

## Pipeline Architecture

Agribound's pipeline consists of the following steps:

1. **Composite building** --- cloud-free annual composites from GEE or local GeoTIFFs, with configurable date ranges, compositing methods (median, greenest pixel, max NDVI), and cloud masking
2. **Optional fine-tuning** --- automatic fine-tuning of engines on user-provided reference boundaries, with per-source checkpoint isolation and early stopping
3. **Delineation** --- one of seven engines produces raw field polygons from the composite
4. **Post-processing** --- cross-tile polygon merging, area filtering, Chaikin corner-cutting smoothing, Ramer-Douglas-Peucker simplification, and optional regularization
5. **LULC crop filtering** --- server-side removal of non-agricultural polygons with per-field crop fraction metadata
6. **Export** --- fiboa-compliant GeoPackage, GeoJSON, or GeoParquet output

<!-- TODO: Insert Figure 1 (architecture diagram) here -->

## Delineation Engines

The seven supported engines span complementary approaches:

| Engine | Approach | Input |
|--------|----------|-------|
| Delineate-Anything | YOLO instance segmentation | RGB |
| Fields of The World | Semantic segmentation (14+ models) | RGBN |
| GeoAI | Mask R-CNN instance segmentation | RGB |
| DINOv3 | Vision transformer (SAT-493M satellite-pretrained) + DPT decoder | RGB |
| Prithvi-EO-2.0 | Foundation model + UPerNet | RGBN |
| Embedding | Unsupervised K-means clustering | Embeddings |
| Ensemble | Multi-engine consensus (vote/union/intersection) | Multiple |

Each engine automatically extracts and reorders the bands it needs from the full multi-band composite via canonical band mappings. SAM2 [@ravi2024] boundary refinement can be applied per-source for pixel-accurate edges, using each sensor's native raster at its own resolution.

## LULC Crop Filtering

The LULC filtering feature addresses a fundamental limitation shared by all existing field boundary models: they detect boundaries, not fields. A model trained on satellite imagery will segment any visually distinct region, producing false positives in non-agricultural areas.

Agribound computes, for each detected polygon, the mean crop probability (Dynamic World) or crop class fraction (NLCD, C3S) using GEE's `reduceRegions` operation. Polygons below a configurable threshold (default 0.3) are discarded. The crop fraction is preserved as a `lulc:crop_fraction` column in the output, enabling downstream analysis of crop confidence.

| Region | Dataset | Resolution | Years |
|--------|---------|-----------|-------|
| CONUS | USGS Annual NLCD | 30 m | 1985--2024 |
| Global | Google Dynamic World | 10 m | 2015--present |
| Global (pre-2015) | Copernicus C3S | 300 m | 1992--2022 |

All three datasets use nearest-year matching when the exact target year is unavailable. Processing is batched (1000 polygons per GEE call) to handle large study areas without exceeding API limits.

<!-- TODO: Insert LULC filtering before/after comparison figure here -->

## Source-Aware Caching

Intermediate files (RGB inputs, segmentation outputs, SAM2 crops) are tagged with the source name to prevent cache collisions when running multiple satellite sensors through the same working directory. Fine-tuning checkpoints and training data are similarly isolated per engine and source, enabling checkpoint reuse across years without cross-contamination.

# Usage

```python
import agribound

# Basic delineation with automatic LULC filtering (default)
gdf = agribound.delineate(
    study_area="area.geojson",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-project",
)

# Disable LULC filtering for non-agricultural use cases
gdf = agribound.delineate(..., lulc_filter=False)

# Adjust crop threshold for arid regions
gdf = agribound.delineate(..., lulc_crop_threshold=0.2)
```

Fifteen example scripts and Jupyter notebooks demonstrate workflows spanning six continents, eight satellite sources, and all delineation engines.

# Research Enabled by Agribound

<!-- TODO: Add specific research applications once available -->

Agribound enables several research workflows that were previously impractical:

- **Multi-decadal field boundary time series.** The 40-year Landsat example (1985--2025) with automatic NLCD crop filtering produces annual field boundary maps for long-term agricultural change analysis across the contiguous United States.
- **Multi-source ensemble delineation.** Running the same engine (e.g., DINOv3) across Sentinel-2, Landsat, HLS, NAIP, and SPOT with per-source SAM2 refinement and majority-vote merging produces boundaries that leverage the strengths of each sensor.
- **Zero-reference-data field mapping.** The automated pipeline combines Google Satellite Embedding clusters, Dynamic World crop filtering, and SAM2 boundary refinement on Sentinel-2 to delineate crop fields anywhere in the world without human-labeled training data or model training.
- **Global agricultural water use estimation.** At the Desert Research Institute, Agribound supports operational water use mapping for the New Mexico Office of the State Engineer, where accurate field boundaries are required for irrigation water accounting across Lea County and other agricultural regions.

# Research Impact Statement

Agribound's primary impact is reducing the barrier to entry for agricultural field boundary mapping from satellite imagery. By unifying seven engines behind a single interface and providing automatic LULC crop filtering, it transforms a multi-week research engineering task (setting up models, preprocessing imagery, cleaning non-agricultural outputs) into a single function call. The LULC filtering feature --- the first of its kind in any field boundary package --- directly addresses the practical reality that delineation models produce false positives in non-agricultural areas, a problem that has been largely ignored in the literature because benchmarks typically use pre-selected agricultural study areas.

# Broader Applicability

While Agribound is designed for agricultural field boundaries, its architecture generalizes to other polygon delineation tasks:

- **Urban parcel mapping** --- the same engines detect building and lot boundaries; disabling `lulc_filter` retains all detected polygons.
- **Forest stand delineation** --- embedding-based clustering with appropriate LULC class selection (e.g., C3S tree cover classes) could delineate forest management units.
- **Wetland boundary mapping** --- Dynamic World's `flooded_vegetation` and `water` bands could replace the crop probability filter for wetland applications.
- **General remote sensing segmentation** --- the `source="local"` mode accepts any GeoTIFF, making the engines usable for non-GEE workflows including drone imagery and commercial satellite data.

The `lulc_filter` architecture is designed to be extensible: adding a new LULC dataset requires implementing a single function that returns a binary or probability image and a filter threshold.

# Limitations

- **GEE dependency for LULC filtering.** The automatic crop filter requires Google Earth Engine authentication, which may not be available in all computing environments. The filter fails gracefully (polygons are retained without filtering) and can be disabled with `lulc_filter=False`.
- **Dynamic World accuracy in arid regions.** Dynamic World's crop probability estimates can be low for irrigated agriculture in arid landscapes where green vegetation is sparse outside the growing season. Users in these regions should lower the `lulc_crop_threshold` (e.g., to 0.2) or use seasonal date ranges.
- **Engine-specific GPU requirements.** Most delineation engines require a GPU for practical runtimes. The embedding engine and GeoAI (which falls back to CPU on Apple Silicon) are exceptions.
- **SPOT access restrictions.** SPOT 6/7 imagery on GEE is restricted to authorized users under a data-sharing agreement.
- **No sub-field delineation.** Agribound targets field-level boundaries (>2500 m² by default) and is not designed for within-field zone mapping or crop row detection.
- **C3S resolution.** For pre-2015 global coverage, the C3S Land Cover product at 300 m resolution may miss small fields, particularly in smallholder agriculture regions.

# Author Contributions

<!-- TODO: Finalize after all authors review -->

**S. Majumdar:** Conceptualization, software architecture, implementation, documentation, testing, writing --- original draft. **J. L. Huntington:** Supervision, funding acquisition, domain expertise (water resources). **P. ReVelle:** Domain expertise (agricultural remote sensing), testing. **S. Nozari:** Domain expertise (irrigation water use), testing. **R. G. Smith:** Domain expertise (water resources), testing. **M. F. Hasan:** Testing, feedback. **M. Bromley:** Domain expertise (water resources), funding acquisition. **J. Atkin:** Testing, feedback. **E. R. Jensen:** Domain expertise (remote sensing), testing. **D. Ketchum:** Domain expertise (agricultural remote sensing), testing. **S. Roy:** GEE data integration, LULC dataset curation.

# AI Usage Disclosure

Portions of this software were developed with the assistance of AI coding tools, including Anthropic's Claude. AI was used to accelerate code scaffolding, documentation drafting, and test generation. All AI-generated code was reviewed, tested, and validated by the human authors. The scientific methodology, architectural decisions, algorithm selection, and domain-specific implementations reflect the expertise and judgment of the authors.

# Acknowledgments

This work was supported by the New Mexico Office of the State Engineer, the Google Satellite Embeddings Dataset Small Grants Program, the U.S. Army Corps of Engineers, the U.S. Department of Treasury/State of Nevada, the USGS and NASA Landsat Science Team, the USGS Water Resources Research Institute, the Desert Research Institute Maki Endowment, and the Windward Fund. Access to SPOT 6/7 imagery on Google Earth Engine was provided through the Google Trusted Tester opportunity.

# References
