# Agribound

**Agricultural field boundary delineation from satellite imagery** with
published segmentation models, geospatial and general-purpose foundation
models and satellite embeddings, through one configuration and one pipeline.

[![Release](https://img.shields.io/badge/release-v1.1.0-green.svg)](https://github.com/montimaj/agribound/releases)
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
export pipeline over sixteen sources (Landsat, Landsat panchromatic, Sentinel-2,
HLS, NAIP, SPOT 6/7 and Sentinel-1 SAR composites on Google Earth Engine, USGS
NAIP Plus, the Umbra and Capella open SAR archives on AWS, local GeoTIFFs,
Google Satellite Embedding and TESSERA embeddings, and stacks of annual USDA CDL
and AAFC ACI crop-type maps on Google Earth Engine) with nine engines
(Delineate-Anything, Fields of The World, GeoAI Mask R-CNN, DINOv3,
Prithvi-EO-2.0, SAM 2.1 automatic masks, embedding clustering, crop-sequence
zones and ensembles). Every run is seeded, cached under content-addressed
names and documented by a provenance record; evaluation, tiling for HPC
clusters and an optional human-confirmed agent layer are included.

!!! warning "Upgrading from 0.1.x"
    Version 1.0.0 fixes defects that affected results produced with agribound
    0.1.x (for example FTW season windows, Landsat/HLS radiometry, caches that
    ignored the study area and year, and silent engine fallbacks). See the
    [migration guide](migration-1.0.md) and the
    [list of affected results](https://github.com/montimaj/agribound/blob/main/CHANGELOG.md#results-produced-with-agribound--013-that-are-affected).

## How it works

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/agribound_workflow_1.0.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/agribound_workflow_1.0.png" alt="The agribound 1.1 workflow: an optional agent layer with a human confirmation gate and a deterministic entry point above a six-stage pipeline from sixteen imagery, embedding and crop-type map sources to field boundaries" width="900"></a>

*The agribound 1.1 workflow (select the image for full resolution): a six-stage pipeline from sixteen imagery, embedding and crop-type map sources (0.16–30 m, 1984–present) to field boundaries, with a deterministic entry point and an optional, human-confirmed agent layer above it.*

1. **Composite.** Earth Engine builds a median or greenest-pixel (max-NDVI) composite for a year or a date window and exports it on a UTM grid. NAIP is mosaicked, and only Landsat, Sentinel-2 and HLS are cloud-masked and scaled to reflectance ×10 000 (Landsat panchromatic is cloud-masked but kept as TOA reflectance; Sentinel-1 radar backscatter is edge-masked and scaled from dB to 8 bits). USGS NAIP Plus, Umbra and Capella SAR (median amplitude of co-registered collects of one geometry, built locally from the AWS open data), TESSERA and local GeoTIFF inputs are read without Earth Engine. The USDA CDL and AAFC ACI crop-type maps are downloaded as class codes, one band per year, resampled to 30 m by Earth Engine (never averaged).
2. **Fine-tuning (optional).** Full (Delineate-Anything, GeoAI, DINOv3, Prithvi) or LoRA (DINOv3, Prithvi) fine-tuning on reference boundaries, validated by default on a spatially blocked split (5 km blocks). GeoAI, DINOv3 and Prithvi's UPerNet mode need a checkpoint, from fine-tuning or supplied by the user.
3. **Delineation.** One of nine engines, coloured by family: task-specific segmentation, foundation model (geospatial: DINOv3, Prithvi; general-purpose: SAM 2.1), label-free embedding clustering, crop-history zones (zones of uniform multi-year crop history from crop-type maps, not field instances) and multi-engine ensemble. SAM 2.1 automatic masks (`sam-auto`) are opt-in: never an engine default.
4. **Refine and post-process.** Optional SAM refinement (SAM 2, 2.1 or 3; the SAM 3 backends are [untested](user-guide/sam-refinement.md#sam-3-is-untested)), then study-area selection, merging, minimum-area filtering, smoothing and simplification, and optional centre-pivot snapping (opt-in, `engine_params["pivots"]`).
5. **LULC crop filter.** Removes polygons whose crop fraction is below 0.3, computed on Earth Engine or locally on a downloaded crop raster. Annual NLCD, Dynamic World or C3S Land Cover is selected by coverage and year; CDL (CONUS only) is used on request. Dynamic World files plantations and many orchards under trees, so for tree crops set `lulc_tree_crops=True`, which, with Dynamic World or C3S, counts tree cover (forest included) as crop; NLCD and CDL are unchanged.
6. **Export.** GeoParquet (fiboa-style columns), GeoPackage or GeoJSON, with per-field area, perimeter, compactness and crop fraction, plus a `provenance.json` record.

Around the pipeline:

- **Entry point.** `delineate()` and `agribound delineate --config` run the six stages directly. Every run is seeded and uses a content-addressed cache, and `provenance.json` is written by default.
- **Agent layer (optional).** A language model, reached through the Claude API or an MCP host (or a local Anthropic-compatible server via `base_url`), investigates with typed read-only tools and proposes one configuration. It runs only after you confirm that exact plan at the human gate, with an approval bound to the plan's hash and used once. At most one plan runs per session, and the session then stops (see [Agent layer](user-guide/agent.md)).
- **Scale out and evaluate.** `agribound tiles make`, `run` and `merge` split a large study area into tiles that run as Slurm array jobs (see [HPC and large areas](user-guide/hpc.md)). `evaluate()` scores results against reference boundaries with object-level and area-weighted metrics (see [Evaluation](user-guide/evaluation.md)).

## Quick install

```bash
pip install "agribound[gee,delineate-anything]"
pip install "agribound[gee,sam-auto]"   # SAM 2.1 automatic masks (engine "sam-auto")
```

The `sam-auto` extra installs `sam2` 1.1.0, a third-party PyPI build of
Meta's SAM 2 code (Apache-2.0), pinned, with OpenCV and the Hugging Face hub
client, and not segment-geospatial. Two environments are needed for the full
stack because FTW and Prithvi require incompatible `lightning` versions; see
[Installation](installation.md).

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

Dynamic World counts plantations and orchards as trees, so the default filter
can remove tree crops where it uses Dynamic World: of 95 oil-palm blocks in
Ghana it kept none for 2020. `lulc_tree_crops=True` counts tree cover as crop
and kept all 95; with Dynamic World or C3S the filter then also keeps forest
(NLCD and CDL are unchanged). See
[Tree crops](user-guide/satellite-sources.md#tree-crops).

## Example results

From the agribound 1.1 example runs: the development run of 2026-10-09/10,
whose outputs 1.1.0 reuses unchanged, the targeted 1.1.0 re-runs of
2026-10-11, example 26's run of 2026-10-10, example 24's re-scoring run of
2026-10-11 and its run with `--options` and `--sam-auto` of 2026-10-13, and
example 28's run of 2026-10-12/13; the gallery adds the pre-registered
evaluation of `sam-auto` (2026-10-13). Each map is drawn on a composite from
the run, named under the
map: usually the engine's input; for FTW, its window A; for the SAM-refined
embedding panels, the Sentinel-2 composite SAM 2 read. Select an image for
the full-resolution file; see the [Gallery](gallery.md) for all regions and
engines.

**From 30 m to 1 m (San Juan County, New Mexico).** Delineate Anything v2,
used as released, on Landsat, Sentinel-2, SPOT 6/7 and NAIP of 2022 against
the 1,355 DRI/NMOSE field boundaries (V18) of at least 2,500 m² in the box
(cyan; not used for training or fine-tuning in these runs): object F1
(IoU ≥ 0.5) 0.13, 0.29, 0.17 and 0.39. The 2022 SPOT composite is a single
scene with valid data over 76.7 % of the box.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/San_Juan_resolution_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/San_Juan_resolution_example.webp" alt="Delineate-Anything v2 from 30 m to 1 m" width="800"></a>

Credits: DRI/NMOSE field boundaries (V18), Desert Research Institute and New
Mexico Office of the State Engineer; contains modified Copernicus Sentinel
data 2022.

**Supervised: DINOv3 fine-tuned + SAM 2 (eastern Lea County, New Mexico).**
In-sample F1 against the training polygons (DRI/NMOSE V18): 0.53 (Landsat
2024, read on a 10 m copy), 0.57 (Sentinel-2 2024), 0.59 (SPOT 2023) and 0.56
(NAIP 2022).

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/NM_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/NM_example.webp" alt="DINOv3 fine-tuned and SAM 2 on four sources" width="800"></a>

Credits: DRI/NMOSE field boundaries (V18), Desert Research Institute and New
Mexico Office of the State Engineer; contains modified Copernicus Sentinel
data 2024.

**Label-free: embeddings + SAM 2 vs Delineate-Anything v2 (Pampas,
Argentina).** No reference data or training; centre pivots near Pergamino.
Orange = refined by SAM 2 (parts over 50 ha kept unrefined). The four panels
come from the development run; the TESSERA layers were read from the TESSERA
v1 store as converted again upstream on 2026-10-06/08. The
[gallery](gallery.md) adds the whole study area and three zoomed windows.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Pampas_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Pampas_example.webp" alt="Embeddings with SAM 2 vs Delineate-Anything v2" width="800"></a>

Credits: contains modified Copernicus Sentinel data 2024.

**Tree crops (Ghana, Papua New Guinea, California, Spain).** Delineate Anything
v2 as released on SPOT 6/7 panchromatic against RSPO GeoRSPO, DWR / Land IQ
and SIGPAC polygons (cyan): object F1 (IoU ≥ 0.5; precision among the
predictions that overlap a reference polygon) 0.32 for oil palm estate
blocks and one polygon among 302 oil palm smallholder parcels (126 matched
after fine-tuning on parcels of the same scheme in a square 13.6 km from the
Oro square, a run added after the released model's result there and kept
after its own score was seen); 0.83 for almond and pistachio blocks, whose
fields are in the model's training data (FBIS-73M). DINOv3, fine-tuned on the
same labels near each site (never on the evaluated squares), has the highest
F1 of the three models at all four sites (0.50, 0.50, 0.91 and, on the Úbeda
recintos, 0.18), with merge rates of 0.13 to 0.88. The default crop filter
removes every oil palm polygon; `lulc_tree_crops=True` keeps all but one of
the oil palm Delineate-Anything polygons and all but 7 of the 2,701 oil palm
embedding segments. The [gallery](gallery.md) compares the sources and
engines.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/Tree_Crops_SPOT_Pan_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.1/preview/Tree_Crops_SPOT_Pan_example.webp" alt="Tree crops — Delineate-Anything v2 on SPOT 6/7 panchromatic" width="800"></a>

**More in the [gallery](gallery.md).** Nevada (example 24): centre pivots and
flood- and sprinkler-irrigated fields against the DRI field boundaries, scored
on all mapped fields and on the fields irrigated in the year of the imagery;
with `--sam-auto`, SAM 2.1 automatic masks take three of the eight panels by
F1 (ranked by point estimate; in each of the three slots the two 95 %
intervals overlap). Their settings were chosen 5.0 and 16.0 km away in the
same basins, so these squares are held out from that choice, not from the
landscape.
[Diamond Valley options](gallery.md#nevada-diamond-valley-opt-in-options-of-110-before-and-after)
(example 24 with `--options`): the opt-in options of 1.1.0 that passed a
pre-registered test on three other Nevada squares, before and after (F1 0.26
to 0.42 for Delineate Anything v2 on NAIP; their effects on this square were
partly known before the test, so these are reported results, not tests; the
test's rules were revised and amended before any run and amended again
before any score, for a defect of its check step that changed no layer or
score).
[SAM 2.1 automatic masks](gallery.md#sam-21-automatic-masks-against-delineate-anything-v2)
(`sam-auto`): in a pre-registered evaluation against Delineate Anything v2 as
released, F1 0.017 higher on NAIP over 46 of example 25's squares (95 %
interval 0.004 to 0.030; both near F1 0.3); on Sentinel-2 it did not transfer
(0.152 lower); on SPOT 6/7 and Landsat it was kept over four Nevada and New
Mexico areas, with no "better" claim. Its settings were chosen on example
24's two training squares, so its scores there are in-sample. 47 states
(example 25): Delineate Anything v2 on NAIP in one 5 km square per state
against the OpenET field boundaries v2.1.1, in New Mexico the DRI/NMOSE field
boundaries (V18) and in Nevada the DRI field boundaries (median F1 0.31);
[Montana](gallery.md#47-us-states-montana-the-clu-on-naip-2005-and-2023),
the lowest square (F1 0.01): its reference fields are Common Land Unit strips
of the 2005 NAIP that the 2023 NAIP mostly does not show, so the score
measures how far the reference is from the ground more than how well the
model delineates the fields that exist (measured after the scores were
known; nothing in example 25 changed). Radar
(example 26): Sentinel-1, Umbra and Capella beside NAIP read with the same
settings. Quebec (example 27): crop-history zones of the AAFC Annual Crop
Inventory beside Delineate Anything v2 on Sentinel-2.
[New York](gallery.md#new-york-black-dirt-vegetable-and-onion-strips-on-muck-against-the-common-land-units)
(example 28): Delineate Anything v2, SAM 2.1 automatic masks and fine-tuned
models against the Common Land Units of the OpenET field boundaries in three
landscapes (Black Dirt muck strips, Finger Lakes dairy rotations, North
Country corn, hay and pasture).

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
| [Gallery](gallery.md) | maps from the agribound 1.1 example runs |

## License

Agribound is released under the [Apache 2.0 License](https://github.com/montimaj/agribound/blob/main/LICENSE).
The Delineate-Anything model code and weights and Ultralytics are AGPL-3.0;
check the licences of the models and datasets you use.
