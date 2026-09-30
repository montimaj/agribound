# Agribound

**Agricultural field boundary delineation from satellite imagery** with
published segmentation models, geospatial foundation models and satellite
embeddings, through one configuration and one pipeline.

[![Release](https://img.shields.io/badge/release-v1.0.1-green.svg)](https://github.com/montimaj/agribound/releases)
[![PyPI version](https://img.shields.io/pypi/v/agribound)](https://pypi.org/project/agribound/)
[![Downloads](https://static.pepy.tech/badge/agribound/month)](https://pepy.tech/projects/agribound)
[![CI](https://github.com/montimaj/agribound/actions/workflows/ci.yml/badge.svg)](https://github.com/montimaj/agribound/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://montimaj.github.io/agribound)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](https://earthengine.google.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/montimaj/agribound/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19229665.svg)](https://doi.org/10.5281/zenodo.19229665)
[![GitHub stars](https://img.shields.io/github/stars/montimaj/agribound)](https://github.com/montimaj/agribound/stargazers)

Agribound runs a composite → delineation → post-processing → crop-filter →
export pipeline over ten sources (Landsat, Sentinel-2, HLS, NAIP and SPOT 6/7
composites on Google Earth Engine, USGS NAIP Plus, local GeoTIFFs, and Google
Satellite Embedding and TESSERA embeddings) with seven engines
(Delineate-Anything, Fields of The World, GeoAI Mask R-CNN, DINOv3,
Prithvi-EO-2.0, embedding clustering and ensembles). Every run is seeded,
cached under content-addressed names and documented by a provenance record;
evaluation, tiling for HPC clusters and an optional human-confirmed agent
layer are included.

!!! warning "Upgrading from 0.1.x"
    Version 1.0.0 fixes defects that affected results produced with agribound
    0.1.x (for example FTW season windows, Landsat/HLS radiometry, caches that
    ignored the study area and year, and silent engine fallbacks). See the
    [migration guide](migration-1.0.md) and the
    [list of affected results](https://github.com/montimaj/agribound/blob/main/CHANGELOG.md#results-produced-with-agribound--013-that-are-affected).

## How it works

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/agribound_workflow_1.0.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/agribound_workflow_1.0.png" alt="The agribound 1.0 workflow: an optional agent layer with a human confirmation gate and a deterministic entry point above a six-stage pipeline from ten imagery and embedding sources to field boundaries" width="900"></a>

*The agribound 1.0 workflow (select the image for full resolution): a six-stage pipeline from ten imagery and embedding sources (0.3–30 m, 1984–present) to field boundaries, with a deterministic entry point and an optional, human-confirmed agent layer above it.*

1. **Composite.** Earth Engine builds a median or greenest-pixel (max-NDVI) composite for a year or a date window and exports it on a UTM grid. NAIP is mosaicked, and only Landsat, Sentinel-2 and HLS are cloud-masked and scaled to reflectance ×10 000. USGS NAIP Plus, TESSERA and local GeoTIFF inputs are read without Earth Engine.
2. **Fine-tuning (optional).** Full (Delineate-Anything, GeoAI, DINOv3, Prithvi) or LoRA (DINOv3, Prithvi) fine-tuning on reference boundaries, validated by default on a spatially blocked split (5 km blocks). GeoAI, DINOv3 and Prithvi's UPerNet mode need a checkpoint, from fine-tuning or supplied by the user.
3. **Delineation.** One of seven engines, coloured by family: task-specific segmentation, geospatial foundation model, label-free embedding clustering and multi-engine ensemble.
4. **Refine and post-process.** Optional SAM refinement (SAM 2, 2.1 or 3; the SAM 3 backends are [untested](user-guide/sam-refinement.md#sam-3-is-untested)), then study-area selection, merging, minimum-area filtering, smoothing and simplification.
5. **LULC crop filter.** Removes polygons whose crop fraction is below 0.3, computed on Earth Engine or locally on a downloaded crop raster. Annual NLCD, Dynamic World or C3S Land Cover is selected by coverage and year; CDL (CONUS only) is used on request.
6. **Export.** GeoParquet (fiboa-style columns), GeoPackage or GeoJSON, with per-field area, perimeter, compactness and crop fraction, plus a `provenance.json` record.

Around the pipeline:

- **Entry point.** `delineate()` and `agribound delineate --config` run the six stages directly. Every run is seeded and uses a content-addressed cache, and `provenance.json` is written by default.
- **Agent layer (optional).** A language model, reached through the Claude API or an MCP host (or a local Anthropic-compatible server via `base_url`), investigates with typed read-only tools and proposes one configuration. It runs only after you confirm that exact plan at the human gate, with an approval bound to the plan's hash and used once. At most one plan runs per session, and the session then stops (see [Agent layer](user-guide/agent.md)).
- **Scale out and evaluate.** `agribound tiles make`, `run` and `merge` split a large study area into tiles that run as Slurm array jobs (see [HPC and large areas](user-guide/hpc.md)). `evaluate()` scores results against reference boundaries with object-level and area-weighted metrics (see [Evaluation](user-guide/evaluation.md)).

## Quick install

```bash
pip install "agribound[gee,delineate-anything]"
```

Two environments are needed for the full stack because FTW and Prithvi
require incompatible `lightning` versions; see [Installation](installation.md).

## Quickstart

```python
import agribound

gdf = agribound.delineate(
    study_area="area.geojson",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
)
```

The result is a GeoDataFrame of field polygons with area, perimeter,
compactness and provenance columns, written to `fields_sentinel2_2024.gpkg`
with a `.provenance.json` record next to it. See the
[Quickstart](user-guide/quickstart.md).

## LULC crop filter

Engines delineate visual boundaries, which include roads, water bodies, forest
and built-up areas. The LULC filter, on by default, removes polygons whose crop
fraction in a land-cover dataset is below 0.3:

- **Annual NLCD** (1985-2025, 30 m, classes 81/82) where at least 90 % of the
  area has NLCD data (conterminous US);
- otherwise **Dynamic World** (10 m, annual median crop probability) for 2016
  up to the last complete year;
- otherwise **C3S Land Cover** (2000-2022, 300 m, classes 10, 11, 12, 20, 30);
- **CDL** (`cultivated`, 2013-2023) on request.

It reads the datasets from Earth Engine for every source and raises by default
when it fails. See [LULC crop filter](user-guide/satellite-sources.md#lulc-crop-filter).

## Example results

From the agribound 1.0.1 example runs (the San Juan County map shows 1.0.0
outputs, which 1.0.1 reuses unchanged). Each map is drawn on a composite from
the run, named under the map: usually the engine's input; for FTW, its window
A; for the SAM-refined embedding panels, the Sentinel-2 composite SAM 2 read.
Select an image for the full-resolution file; see the [Gallery](gallery.md)
for all regions and engines.

**From 30 m to 1 m (San Juan County, New Mexico).** Delineate Anything v2,
used as released, on Landsat, Sentinel-2, SPOT 6/7 and NAIP of 2018 against
the 944 NMOSE polygons (cyan; not used for training or fine-tuning in these
runs): object F1 (IoU ≥ 0.5) 0.15, 0.34, 0.33 and 0.43.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/San_Juan_resolution_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/San_Juan_resolution_example.webp" alt="Delineate-Anything v2 from 30 m to 1 m" width="800"></a>

**Supervised: DINOv3 fine-tuned + SAM 2 (eastern Lea County, New Mexico).**
In-sample F1 against the training polygons: 0.06 (Landsat), 0.38
(Sentinel-2), 0.45 (SPOT) and 0.59 (NAIP).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/NM_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/NM_example.webp" alt="DINOv3 fine-tuned and SAM 2 on four sources" width="800"></a>

**Label-free: embeddings + SAM 2 vs Delineate-Anything v2 (Pampas,
Argentina).** No reference data or training; centre pivots near Pergamino.
Orange = refined by SAM 2 (parts over 50 ha kept unrefined). The embedding
panels come from the agribound 1.0.1 run of 2026-09-29; the Delineate-Anything
panels are the 1.0.0 outputs, which that run reused. The [gallery](gallery.md)
adds the whole study area and three zoomed windows.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Pampas_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Pampas_example.webp" alt="Embeddings with SAM 2 vs Delineate-Anything v2" width="800"></a>

## Documentation

| Section | Content |
|---|---|
| [Installation](installation.md) | environments, extras, Apple-silicon notes |
| [Migrating to 1.0](migration-1.0.md) | every breaking change, old vs new |
| [Quickstart](user-guide/quickstart.md) | Python and CLI in five minutes |
| [Satellite sources](user-guide/satellite-sources.md) | coverage, resolution, value scales, masking, LULC filter |
| [Engines](user-guide/engines.md) and [SAM refinement](user-guide/sam-refinement.md) | what each engine does, weights, parameters, limits |
| [Configuration](user-guide/configuration.md) and [CLI](user-guide/cli.md) | every field and command |
| [Fine-tuning](user-guide/fine-tuning.md) | training on reference boundaries |
| [Evaluation](user-guide/evaluation.md) | metric definitions |
| [Reproducibility](user-guide/reproducibility.md) | seeds, cache keys, provenance, output reuse |
| [HPC and large areas](user-guide/hpc.md) | tiling, two-phase runs, Earth Engine quotas, NSF ACCESS |
| [Agent layer](user-guide/agent.md) | human-confirmed planning, MCP server |
| [FTW polygon query](user-guide/ftw-query.md) and [GEE setup](user-guide/gee-setup.md) | published FTW polygons by area; Earth Engine credentials and project |
| [API reference](api/pipeline.md) | generated from the docstrings |
| [Gallery](gallery.md) | maps from the 1.0.0 and 1.0.1 example runs |

## License

Agribound is released under the [Apache 2.0 License](https://github.com/montimaj/agribound/blob/main/LICENSE).
The Delineate-Anything model code and weights and Ultralytics are AGPL-3.0;
check the licences of the models and datasets you use.
