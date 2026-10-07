# agribound

**Agricultural field boundary delineation from satellite imagery**

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

---

## Overview

Agribound runs published field-boundary models, geospatial foundation models
and embedding-based methods on satellite and aerial imagery through one
configuration and one pipeline: composite → optional fine-tuning → delineation
→ optional SAM refinement → study-area selection → post-processing → LULC crop
filter → export. It supports eleven sources (Landsat, Landsat panchromatic,
Sentinel-2, HLS, NAIP and SPOT 6/7 composites built on Google Earth Engine;
USGS NAIP Plus without Earth Engine; local GeoTIFFs; Google Satellite Embedding
and TESSERA embeddings) and seven engines (Delineate-Anything, Fields of The
World, GeoAI Mask R-CNN, DINOv3, Prithvi-EO-2.0, embedding clustering and
ensembles).

Every run is seeded, caches its intermediates under content-addressed names,
and writes a provenance record (configuration and hash, package versions,
device, step timings, model weights, counts, warnings) next to its output.
The package also provides object-level evaluation against reference
boundaries, tiling of large regions for HPC clusters, a query helper for the
published Fields of The World polygons, and an optional agent layer that
proposes one run for a human to approve.

> **Upgrading from 0.1.x?** Version 1.0.0 fixes defects that affected results
> produced with agribound 0.1.x, among them FTW season windows that were
> copies of the annual composite, Landsat/HLS inputs on the wrong radiometric
> scale, a silent Delineate-Anything fallback with swapped red/blue channels,
> and caches that ignored the study area and year. See
> [the affected results](https://github.com/montimaj/agribound/blob/main/CHANGELOG.md#results-produced-with-agribound--013-that-are-affected)
> and the [migration guide](https://montimaj.github.io/agribound/migration-1.0/).

## How It Works

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/agribound_workflow_1.0.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/agribound_workflow_1.0.png" alt="The agribound 1.0 workflow: an optional agent layer with a human confirmation gate and a deterministic entry point above a six-stage pipeline from eleven imagery and embedding sources to field boundaries" width="900"></a>

*The agribound 1.0 workflow (select the image for full resolution): a six-stage pipeline from eleven imagery and embedding sources (0.3–30 m, 1984–present) to field boundaries, with a deterministic entry point and an optional, human-confirmed agent layer above it.*

1. **Composite.** Earth Engine builds a median or greenest-pixel (max-NDVI) composite for a year or a date window and exports it on a UTM grid. NAIP is mosaicked, and only Landsat, Sentinel-2 and HLS are cloud-masked and scaled to reflectance ×10 000 (Landsat panchromatic is cloud-masked but kept as TOA reflectance). USGS NAIP Plus, TESSERA and local GeoTIFF inputs are read without Earth Engine.
2. **Fine-tuning (optional).** Full (Delineate-Anything, GeoAI, DINOv3, Prithvi) or LoRA (DINOv3, Prithvi) fine-tuning on reference boundaries, validated by default on a spatially blocked split (5 km blocks). GeoAI, DINOv3 and Prithvi's UPerNet mode need a checkpoint, from fine-tuning or supplied by the user.
3. **Delineation.** One of seven engines, coloured by family: task-specific segmentation, geospatial foundation model, label-free embedding clustering and multi-engine ensemble.
4. **Refine and post-process.** Optional SAM refinement (SAM 2, 2.1 or 3; the SAM 3 backends are untested), then study-area selection, merging, minimum-area filtering, smoothing and simplification.
5. **LULC crop filter.** Removes polygons whose crop fraction is below 0.3, computed on Earth Engine or locally on a downloaded crop raster. Annual NLCD, Dynamic World or C3S Land Cover is selected by coverage and year; CDL (CONUS only) is used on request. Dynamic World files plantations and many orchards under trees, so for tree crops set `lulc_tree_crops=True`, which, with Dynamic World or C3S, counts tree cover (forest included) as crop; NLCD and CDL are unchanged.
6. **Export.** GeoParquet (fiboa-style columns), GeoPackage or GeoJSON, with per-field area, perimeter, compactness and crop fraction, plus a `provenance.json` record.

Around the pipeline:

- **Entry point.** `delineate()` and `agribound delineate --config` run the six stages directly. Every run is seeded and uses a content-addressed cache, and `provenance.json` is written by default.
- **Agent layer (optional).** A language model, reached through the Claude API or an MCP host (or a local Anthropic-compatible server via `base_url`), investigates with typed read-only tools and proposes one configuration. It runs only after you confirm that exact plan at the human gate, with an approval bound to the plan's hash and used once. At most one plan runs per session, and the session then stops (see [Agent layer](https://montimaj.github.io/agribound/user-guide/agent/)).
- **Scale out and evaluate.** `agribound tiles make`, `run` and `merge` split a large study area into tiles that run as Slurm array jobs (see [HPC and large areas](https://montimaj.github.io/agribound/user-guide/hpc/)). `evaluate()` scores results against reference boundaries with object-level and area-weighted metrics (see [Evaluation](https://montimaj.github.io/agribound/user-guide/evaluation/)).

## Results

From the agribound 1.0.1 example runs (the San Juan County map shows 1.0.0
outputs, which 1.0.1 reuses unchanged; the tree-crop map comes from the
development version that follows 1.0.1). Each map is drawn on a composite from
the run, named under the map with the model and its version: usually the
engine's input; for FTW, its window A; for the SAM-refined embedding panels
(Pampas, top), the Sentinel-2 composite SAM 2 read. Select an image for the
full-resolution file. See the
[gallery](https://montimaj.github.io/agribound/gallery/) for more regions and
engines.

### From 30 m to 1 m: Delineate-Anything v2 against a reference registry (San Juan County, New Mexico, USA)

Example 20: Delineate Anything v2, used as released, on Landsat, Sentinel-2,
SPOT 6/7 and NAIP of 2018, each evaluated against the 944 NMOSE WUCB polygons
(cyan; not used for training or fine-tuning in these runs). Object F1
(IoU ≥ 0.5) is 0.15 at 30 m, 0.34 at 10 m, 0.33 at 6 m and 0.43 at 1 m;
recall rises from 0.08 to 0.44.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/San_Juan_resolution_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/San_Juan_resolution_example.webp" alt="Delineate-Anything v2 on Landsat, Sentinel-2, SPOT and NAIP — San Juan County, New Mexico" width="800"></a>

### Supervised: DINOv3 fine-tuned + SAM 2 on four sources (eastern Lea County, New Mexico, USA)

Example 14: DINOv3 (SAT-493M weights) fine-tuned on the NMOSE polygons for
each source and refined with SAM 2. In-sample F1 (the polygons are also the
training labels): 0.06 on Landsat, 0.38 on Sentinel-2, 0.45 on SPOT and 0.59
on NAIP; at 30 m the box also holds only 4 training chips, against 2,014 at
1 m.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/NM_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/NM_example.webp" alt="DINOv3 fine-tuned and SAM 2 on Landsat, Sentinel-2, SPOT and NAIP — eastern Lea County, New Mexico" width="800"></a>

### Label-free: embeddings + SAM 2 vs Delineate-Anything v2 (Pampas, Argentina)

Example 15, no reference data or training: Google Satellite Embedding and
TESSERA v1 clusters of 2024, crop-filtered and refined with SAM 2 on
Sentinel-2 (top; parts over 50 ha kept unrefined), against Delineate Anything
v2 on the same Sentinel-2 composite and on SPOT 6/7 2023 (bottom); centre
pivots near Pergamino. Orange = refined by SAM 2. The embedding panels come
from the agribound 1.0.1 run of 2026-09-29; the Delineate-Anything panels are
the 1.0.0 outputs, which that run reused. The
[gallery](https://montimaj.github.io/agribound/gallery/) adds the whole study
area and three zoomed windows.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Pampas_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Pampas_example.webp" alt="Embeddings with SAM 2 vs Delineate-Anything v2 on Sentinel-2 and SPOT — Pampas, Argentina" width="800"></a>

### Tree crops: oil palm, orchards and olive groves against reference boundaries (Ghana, Papua New Guinea, California, Spain)

Example 23: Delineate Anything v2 as released on SPOT 6/7 panchromatic
1.5 m, against reference polygons (cyan): the RSPO GeoRSPO concession maps
(an oil palm estate at Twifo Praso and smallholder parcels in Oro Province),
the DWR / Land IQ 2022 crop map (Madera County) and SIGPAC parcels (Úbeda).
Object F1 (IoU ≥ 0.5; precision counts only the predictions that overlap a
reference polygon, as in the gallery) is 0.29 for the estate blocks, about
half of which come out in pieces: many are cut along the edges of the
engine's 768 m inference tiles (the blocks are about 1 km long), and about as
many along roads or tracks that run through the blocks. Among 302 smallholder
parcels it draws one polygon. A model fine-tuned on parcels of the same scheme,
in a square 13.6 km from the Oro square (centres 19.5 km apart), matches 68;
that fine-tuning was added after the released model's Oro result and kept after
its own Oro score was seen. For the almond and pistachio blocks F1 is 0.86, on
fields the model has seen: its training data (FBIS-73M) cover the Madera square
and 87 % of the Úbeda square, with labels that match 121 of the 122 DWR / Land
IQ fields and 166 of the 225 SIGPAC recintos, and have no patches in Ghana or
Papua New Guinea. Fine-tuned on the same labels near each site, DINOv3
(ViT-L/16 pre-trained on satellite imagery, with agribound's default recipe)
merges neighbouring fields (merge rates 0.80 to 1.00), because it almost never
predicts the field-boundary class. Oro is the only site where it beats both
Delineate Anything v2 models (F1 0.23, against 0.21 for the fine-tuned one).
The default LULC crop filter removes every polygon at both oil palm sites,
because Dynamic World counts the palms as trees; `lulc_tree_crops=True` keeps
every Delineate-Anything polygon there and all but 8 of their 2,720 embedding
segments.
The [gallery](https://montimaj.github.io/agribound/gallery/) compares the
sources and engines at Twifo Praso, Oro and Madera.

<a href="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/Tree_Crops_SPOT_Pan_example.png"><img src="https://raw.githubusercontent.com/montimaj/agribound/main/assets/gallery_1.0/preview/Tree_Crops_SPOT_Pan_example.webp" alt="Tree crops — Delineate-Anything v2 on SPOT 6/7 panchromatic in Ghana, Papua New Guinea, California and Spain" width="800"></a>

## Satellite Sources

From `agribound.registry` (`agribound list-sources`); facts checked against the
Earth Engine catalogue and the providers on 2026-09-26 to 2026-09-28.

| Source | Key | Export resolution | Years | Coverage | Value scale | Earth Engine |
|---|---|---|---|---|---|---|
| Sentinel-2 L2A (harmonized) | `sentinel2` | 10 m | 2017-present | global (2017-2018 L2A not global) | reflectance × 10000 | yes |
| Landsat 5/7/8/9 C2 L2 | `landsat` | 30 m | 1984-present | global | reflectance × 10000 | yes |
| Landsat 7/8/9 C2 TOA panchromatic | `landsat-pan` | 15 m | 1999-present | global; by default Landsat 8/9, or Landsat 7 for windows before 2013-03-18 (`landsat_pan_missions`) | TOA reflectance | yes |
| HLS v2.0 (L30 + S30) | `hls` | 30 m | 2013-present (S30 2015-) | global land | reflectance × 10000 | yes |
| NAIP | `naip` | 1 m (`naip_resolution_m`; native 0.6 m in most states since 2018) | 2002-2023 | conterminous US | uint8 | yes |
| USGS NAIP Plus ImageServer | `usgs-naip-plus` | 0.3-0.6 m (finest selected footprint) | 2012-2023, **latest vintage per state only** | US states and territories | uint8 | no |
| SPOT 6/7 multispectral | `spot` | 6 m | 2012-2023 | global, **restricted** | uncalibrated DN | yes |
| SPOT 6/7 panchromatic | `spot-pan` | 1.5 m | 2012-2023 | global, **restricted** | uncalibrated DN | yes |
| Local GeoTIFF | `local` | the file's | any | user-provided | unknown | no |
| Google Satellite Embedding (AlphaEarth) | `google-embedding` | 10 m, 64-D | 2017-2025 | global land | embedding | default backend; `source_coop` backend needs none |
| TESSERA | `tessera-embedding` | 10 m, 128-D | v1 2017-2025 (near-global 2024), v1.1 2015-2025, v2 beta | depends on version | embedding | no |

Composites cover the study area's bounding box in the UTM zone of its centroid
(`export_crs`) without polygon masking. The LULC crop filter reads Earth Engine
for **every** source. Details: [Satellite sources](https://montimaj.github.io/agribound/user-guide/satellite-sources/).

## Delineation Engines

| Engine | Key | Approach | Label-free | Fine-tunable | Default weights / model | GPU |
|---|---|---|---|---|---|---|
| Delineate-Anything | `delineate-anything` | YOLO11-seg instance segmentation | yes | yes | `large_v2` (Delineate Anything v2) from `MykolaL/DelineateAnything`, pinned revision, SHA-256 checked | recommended |
| Fields of The World | `ftw` | semantic segmentation with ftw-tools checkpoints, polygonised | yes | **no** | the ftw-tools registry default, `FTW_PRUE_EFNET_B5` in ftw-tools 2.0.0b5 (two crop-calendar season windows) | recommended |
| GeoAI | `geoai` | Mask R-CNN instance segmentation (geoai-py) | **no**: no published field weights; needs fine-tuning or `checkpoint_path` | yes | COCO Mask R-CNN as the fine-tuning start | recommended (CPU on Apple MPS) |
| DINOv3 | `dinov3` | DINOv3 ViT + DPT head (geoai-py) | **no**: needs fine-tuning or `checkpoint_path` | yes (full by default, LoRA optional) | SAT-493M ViT-L/16 backbone | recommended |
| Prithvi-EO-2.0 | `prithvi` | Prithvi-EO-2.0 ViT (terratorch): `embed` clustering, `pca` baseline, or fine-tuned `segment` | only `embed`/`pca` | yes | `Prithvi-EO-2.0-300M-TL` | recommended |
| Embedding | `embedding` | K-means clustering of pre-computed embeddings | yes | no | - | no (CPU) |
| Ensemble | `ensemble` | intersection, union or pixel vote of several engines/models | depends on members | no | members `delineate-anything` + `ftw` | depends |

Engine/source support is checked when the configuration is created; there are
no silent fallbacks to another model, band set or engine. GeoAI fine-tuning
sizes its chips from the reference fields by default, and GeoAI joins the
instances that a field was split into at the edges of its inference windows
(see [Fine-tuning](https://montimaj.github.io/agribound/user-guide/fine-tuning/)).
Optional **SAM refinement** (`sam_refine=True`) runs after any engine except
`embedding` (which refines its own polygons, given
`engine_params["sam_rgb_bands"]`), with backends `sam2` (default), `sam2.1`,
`sam3` (Meta; CUDA + triton, Linux) and `sam3-hf` (transformers); **both SAM 3
backends are currently untested** (they have not been run end to end, because
the `facebook/sam3` weights are gated; agribound logs a WARNING when one is
loaded). Fields whose padded bounding box is under
`sam_min_crop_px` (64 px) are not refined. Details:
[Engines](https://montimaj.github.io/agribound/user-guide/engines/),
[SAM refinement](https://montimaj.github.io/agribound/user-guide/sam-refinement/).

## Installation

Python **>= 3.12**. terratorch (Prithvi) requires `lightning>=2.6` and
ftw-tools 2.0.0b5 (FTW) requires `lightning<2.6`, so the full stack needs **two
environments**:

| Environment | File | Extra | Includes | Excludes |
|---|---|---|---|---|
| core | `environment.yml` | `agribound[all]` | GEE, Delineate-Anything, FTW, GeoAI, DINOv3, SAM 2, TESSERA, agent | Prithvi, SAM 3 |
| GFM | `environment-gfm.yml` | `agribound[all-gfm]` | GEE, Delineate-Anything, GeoAI, DINOv3, Prithvi, SAM 2, TESSERA, agent | FTW, SAM 3 |

```bash
git clone https://github.com/montimaj/agribound.git && cd agribound
conda env create -f environment.yml          # or environment-gfm.yml; both install -e .
conda activate agribound                     # (agribound-gfm)
```

or with pip, in a fresh Python 3.12 environment:

```bash
pip install "agribound[all]"        # core
pip install "agribound[all-gfm]"    # separate environment, for Prithvi
pip install "agribound[gee,delineate-anything]"   # or only what you need
```

| Extra | For |
|---|---|
| `gee` | Earth Engine sources, the default Google-embedding backend, the LULC filter |
| `delineate-anything` | Delineate-Anything (`numba` is used only by its reference backend) |
| `ftw` | FTW (`ftw-tools>=2.0.0b5,<3`, a pre-release on PyPI) |
| `geoai`, `dinov3` | GeoAI, DINOv3 (`geoai-py>=0.43.1`) |
| `prithvi` | Prithvi (`terratorch[peft]`; conflicts with `ftw`) |
| `samgeo` | SAM 2 / 2.1 refinement |
| `sam3` | SAM 3 Meta backend (CUDA; `triton-windows` on Windows); **untested** (see SAM refinement) |
| `tessera`, `embedding` | TESSERA (`geotessera>=0.10.2,<0.11`); both embedding sources |
| `agent` | the agent layer and MCP server (`anthropic`, `mcp`) |
| `docs`, `dev` | documentation; tests and linting |

The GDAL Python bindings (`osgeo`, conda-forge `gdal`) are needed only by the
Delineate-Anything reference backend. See
[Installation](https://montimaj.github.io/agribound/installation/).

## Quick Start (Python)

```python
import agribound

gdf = agribound.delineate(
    study_area="my_region.geojson",  # or "bbox:minx,miny,maxx,maxy", WKT, GEE asset
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
    output_path="fields.gpkg",
)
print(len(gdf), gdf.attrs["run_id"])  # fields.gpkg + fields.gpkg.provenance.json
```

## Quick Start (CLI)

```bash
agribound delineate \
    --study-area my_region.geojson \
    --source sentinel2 \
    --year 2024 \
    --engine delineate-anything \
    --gee-project my-gee-project \
    --output fields.gpkg
```

## Configuration

Every option is a field of `AgriboundConfig`; YAML files use the same names,
and unknown keys are rejected.

```yaml
# config.yml
study_area: my_region.geojson
source: sentinel2
year: 2024
engine: delineate-anything
gee_project: my-gee-project

composite_method: median        # median | greenest (max_ndvi is an alias)
cloud_cover_max: 20
export_crs: utm                 # UTM zone of the study-area centroid

aoi_selection: representative_point
min_field_area_m2: 2500         # m²
simplify_tolerance: 2.0         # metres
lulc_filter: true
lulc_crop_threshold: 0.3
lulc_on_error: raise            # raise | warn

engine_params:
  da_model: large_v2
  conf_threshold: 0.15          # the old name 'confidence' now raises

output_path: fields.gpkg
seed: 42
```

```bash
agribound delineate --config config.yml                      # YAML supplies every value
agribound delineate --config config.yml --year 2023 -o fields_2023.gpkg   # explicit flags override it
agribound delineate --config config.yml --dry-run            # print the resolved YAML
```

Reference: [Configuration](https://montimaj.github.io/agribound/user-guide/configuration/),
[CLI](https://montimaj.github.io/agribound/user-guide/cli/).

## Reproducibility

- **Seed**: `seed` (default 42) seeds Python, NumPy, torch and Lightning; the
  fine-tuning split and every sample are derived from it.
- **Cache keys**: intermediates are named by a key over the study-area
  geometry, source, year, date range and compositing settings, so runs with
  different settings never reuse each other's files, even in one `cache_dir`.
- **Provenance**: `<output>.provenance.json` holds the configuration and its
  hash, versions, platform, device, step timings, peak memory, engine metadata
  (for example weight revisions and SHA-256), stage counts and warnings.
- **Output reuse**: an existing output is returned only if its provenance
  record reports success with the same configuration hash, study-area
  fingerprint and results versions; otherwise `delineate()` raises
  `FileExistsError` (`overwrite=True` re-runs). A record of agribound 1.0.0
  or earlier has no study-area fingerprint; when nothing else differs, its
  output is still reused.

See [Reproducibility](https://montimaj.github.io/agribound/user-guide/reproducibility/).

## Evaluation

```python
from agribound.evaluate import evaluate

metrics = evaluate(
    pred_gdf,
    ref_gdf,
    iou_threshold=0.5,
    boundary_tolerance_m=10,
    strata="county",
    size_bins="auto",
    bootstrap=1000,
)
```

One-to-one IoU matching (default; `matching="many_to_one"` uses the 0.1.x
matching rule, although geometry repair and other 1.0.0 changes can still
change the numbers slightly) gives precision, recall and F1; also mean IoU, area-weighted precision/recall,
over- and under-segmentation (Persello & Bruzzone, 2010), Hausdorff and mean
boundary distances, boundary precision/recall/F1 and coverage within a
tolerance, per-stratum and per-size-class metrics, and percentile bootstrap
intervals. `agribound evaluate -p pred.gpkg -r ref.gpkg` does the same from
the command line. Definitions:
[Evaluation](https://montimaj.github.io/agribound/user-guide/evaluation/).

## Large Areas and HPC

`agribound tiles` cuts a region into tiles with halos and runs each tile as an
independent, idempotent job; the composite stage can run on nodes with
internet access and the delineation stage on offline GPU nodes; `tiles merge`
keeps each polygon in the tile that owns its representative point.
[`examples/hpc/`](https://github.com/montimaj/agribound/blob/main/examples/hpc/README.md) has Slurm scripts, profiles for NSF
ACCESS systems and Earth Engine throttling rules;
[`examples/regions/`](https://github.com/montimaj/agribound/blob/main/examples/regions/README.md) defines 16 regions. The region
files name no Earth Engine project: pass your own with `--gee-project` (or
`GEE_PROJECT`, gcloud, or a service-account key); the scripts check it with
`agribound tiles gee-project` before they run or submit anything. See
[HPC](https://montimaj.github.io/agribound/user-guide/hpc/).

## Agent Layer (optional)

`pip install "agribound[agent]"` adds a planning assistant with a deliberately
low level of autonomy: a language model investigates with **read-only tools**
and proposes **one** configuration; a **human confirms the exact plan** (typed
`yes`, bound to the plan's SHA-256 hash, single use); **at most one** approved
plan runs; and the session **stops** after the run or a denial. There is no
automatic re-run or re-tuning. Every turn and tool call is written to a JSON
transcript.

```python
import agribound

result = agribound.agent(
    "Delineate fields in this AOI for 2024 with a label-free approach",
    study_area="fields.geojson",
    gee_project="my-gee-project",
    dry_run=True,  # propose only; run the plan YAML with `agribound delineate --config`
)
print(result.report)
```

```bash
agribound agent "Delineate fields in this AOI for 2024" --study-area fields.geojson --dry-run
agribound mcp serve                   # the same tools for MCP hosts (Claude Desktop, Claude Code, ...)
agribound mcp serve --allow-execute   # also execute_plan, confirmed by an MCP elicitation
claude mcp add agribound -- /path/to/env/bin/agribound mcp serve     # Claude Code
```

The default model is `claude-opus-5` (`--model` or `AGRIBOUND_AGENT_MODEL`
changes it); `--base-url` points the backend at an Anthropic-compatible
local endpoint such as Ollama or vLLM (not tested with agribound). The
`streamable-http` MCP transport has no authentication, so it is refused with
`--allow-execute` or a non-loopback `--host` unless
`--allow-unauthenticated-http` is given. See
[Agent layer](https://montimaj.github.io/agribound/user-guide/agent/).

## Query Published FTW Polygons

`query_ftw` retrieves the already-published Fields of The World polygons for
an area of interest (it does not run inference; the polygons are model
predictions, not ground truth). The default layout (`by-admin-conf`) holds
2024 and 2025 with a `confidence` column, which is null for all of New Mexico
and 99.7 % of New South Wales in the published files, so `min_confidence`
cannot select reliable polygons there.

```python
import agribound as ab

ftw = ab.query_ftw(
    study_area="bbox:-106.80,34.60,-106.75,34.65",
    year=2024,
    clip=True,
    output_path="ftw_2024.parquet",
)
```

```bash
agribound query-ftw --study-area "bbox:-106.80,34.60,-106.75,34.65" --year 2024 -o ftw_2024.parquet
```

With `clip=True` (the default), polygons that cross the AOI boundary are
clipped, their `metrics:area` and `metrics:perimeter` are recomputed from the
clipped geometry, and the column `agribound:clipped` marks them; `clip=False`
returns the whole published polygons with their published metrics. With an
output path, the query parameters, backend and counts are also written to
`<output>.provenance.json`.

See [FTW polygon query](https://montimaj.github.io/agribound/user-guide/ftw-query/).

## Project Structure

```
agribound/
├── agribound/                  # Main package
│   ├── __init__.py             # Public API (delineate, build_composite, evaluate, list_*, query_ftw, agent)
│   ├── _version.py             # Version string
│   ├── _cache.py               # Content-addressed cache keys
│   ├── _repro.py               # Seeding, seeded generators, version capture, run IDs
│   ├── auth.py                 # Earth Engine authentication
│   ├── cli.py                  # Click CLI (delineate, composite, prefetch, evaluate, ...)
│   ├── config.py               # AgriboundConfig dataclass and validation
│   ├── evaluate.py             # Object-level evaluation
│   ├── ftw_arrow.py            # PyArrow reader of the published FTW polygons
│   ├── ftw_query.py            # query_ftw()
│   ├── pipeline.py             # delineate() and build_composite()
│   ├── provenance.py           # Provenance records and configuration hash
│   ├── registry.py             # Source and engine registries
│   ├── visualize.py            # Interactive maps (leafmap/folium)
│   ├── agent/                  # Optional agent layer: tools, plans, gate, session, MCP server, backends
│   ├── clients/                # USGS NAIP Plus ImageServer client
│   ├── composites/             # Composite builders (base, gee, usgs, local + embeddings, dynamic_world)
│   ├── engines/                # Engines, SAM refinement (samgeo_engine), finetune/ package
│   ├── hpc/                    # Tiling, regions, `agribound tiles`
│   ├── io/                     # Raster, vector and CRS helpers
│   └── postprocess/            # Polygonize, merge, filter, simplify/smooth, regularize, LULC filter
├── assets/                     # Figures linked by URL from the README and docs (see assets/README.md):
│   ├── gallery_1.0/            #   rendered from the 1.0.0 and 1.0.1 example runs (tools/make_gallery*.py); WebP previews in preview/
│   ├── gallery_0.1x/           #   archived 0.1.x screenshots
│   └── agribound_workflow_1.0.*  # workflow diagram (tools/make_workflow_diagram.py)
├── docs/                       # MkDocs documentation (user guide, API reference, gallery, blog)
├── examples/                   # Example scripts 01-23, notebooks/ (generated from the scripts), hpc/, regions/
├── paper/                      # Manuscript materials (not included in the PyPI distribution)
├── tests/                      # Pytest suite (unit/, integration/)
├── tools/                      # Maintainer scripts: make_gallery.py, make_gallery_pampas_0.1x.py, make_workflow_diagram.py, sync_notebooks.py
├── CHANGELOG.md
├── CITATION.cff                # Citation metadata
├── CONTRIBUTING.md             # Developer guide
├── DISCLAIMER.md               # Software disclaimer
├── LICENSE                     # Apache 2.0
├── MANIFEST.in                 # Source distribution inclusions/exclusions
├── environment.yml             # Core conda environment
├── environment-gfm.yml         # GFM (Prithvi) conda environment
├── mkdocs.yml                  # MkDocs site configuration
├── pyproject.toml              # Build configuration, dependencies, extras
└── README.md
```

## Examples

Example scripts and notebooks are in [`examples/`](https://github.com/montimaj/agribound/tree/main/examples/); see the
[examples README](https://github.com/montimaj/agribound/blob/main/examples/README.md).

| Script | Notebook | Description |
|---|---|---|
| [01_new_mexico_landsat_timeseries.py](https://github.com/montimaj/agribound/blob/main/examples/01_new_mexico_landsat_timeseries.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/01_new_mexico_landsat_timeseries.ipynb) | Landsat time series over New Mexico with a Delineate-Anything model fine-tuned on NMOSE polygons |
| [02_india_ganges_sentinel2.py](https://github.com/montimaj/agribound/blob/main/examples/02_india_ganges_sentinel2.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/02_india_ganges_sentinel2.ipynb) | Four label-free approaches (FTW, Google and TESSERA embeddings, SPOT pan with Delineate-Anything) in Nadia, West Bengal |
| [03_australia_murray_darling_hls.py](https://github.com/montimaj/agribound/blob/main/examples/03_australia_murray_darling_hls.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/03_australia_murray_darling_hls.ipynb) | Prithvi `embed` and `pca` modes on HLS in the Murray-Darling Basin, compared with Delineate-Anything v2 on SPOT |
| [04_france_beauce_sentinel2.py](https://github.com/montimaj/agribound/blob/main/examples/04_france_beauce_sentinel2.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/04_france_beauce_sentinel2.ipynb) | FTW with crop-calendar season windows in the Beauce |
| [05_pampas_embeddings.py](https://github.com/montimaj/agribound/blob/main/examples/05_pampas_embeddings.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/05_pampas_embeddings.ipynb) | CPU-only embedding clustering (Google, TESSERA) in the Pampas |
| [06_kenya_smallholder_ftw.py](https://github.com/montimaj/agribound/blob/main/examples/06_kenya_smallholder_ftw.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/06_kenya_smallholder_ftw.ipynb) | FTW on smallholder fields in Kakamega with four minimum-area thresholds |
| [07_usa_naip_high_res.py](https://github.com/montimaj/agribound/blob/main/examples/07_usa_naip_high_res.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/07_usa_naip_high_res.ipynb) | Delineate-Anything on 1 m NAIP in the Central Valley |
| [08_china_north_plain_spot.py](https://github.com/montimaj/agribound/blob/main/examples/08_china_north_plain_spot.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/08_china_north_plain_spot.ipynb) | Delineate-Anything on SPOT 6/7 in the North China Plain (restricted source) |
| [09_ensemble_comparison.py](https://github.com/montimaj/agribound/blob/main/examples/09_ensemble_comparison.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/09_ensemble_comparison.ipynb) | Intersection, union and vote ensembles of Delineate-Anything and FTW (Andalusia) |
| [10_local_tif_quickstart.py](https://github.com/montimaj/agribound/blob/main/examples/10_local_tif_quickstart.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/10_local_tif_quickstart.ipynb) | Local GeoTIFF without Earth Engine |
| [11_mississippi_alluvial_plain_spot.py](https://github.com/montimaj/agribound/blob/main/examples/11_mississippi_alluvial_plain_spot.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/11_mississippi_alluvial_plain_spot.ipynb) | SPOT 6/7 series 2021-2023 with year-to-year agreement (restricted source) |
| [12_new_mexico_ensemble_timeseries.py](https://github.com/montimaj/agribound/blob/main/examples/12_new_mexico_ensemble_timeseries.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/12_new_mexico_ensemble_timeseries.ipynb) | Per-source multi-model ensembles with SAM 2, eastern Lea County, 2022 |
| [13_sam2_refine_dinov3.py](https://github.com/montimaj/agribound/blob/main/examples/13_sam2_refine_dinov3.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/13_sam2_refine_dinov3.ipynb) | Stand-alone SAM refinement of existing DINOv3 boundaries |
| [14_dinov3_sam2_ensemble.py](https://github.com/montimaj/agribound/blob/main/examples/14_dinov3_sam2_ensemble.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/14_dinov3_sam2_ensemble.ipynb) | Fine-tuned DINOv3 with and without SAM 2 on five sources |
| [15_pampas_semi_supervised.py](https://github.com/montimaj/agribound/blob/main/examples/15_pampas_semi_supervised.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/15_pampas_semi_supervised.ipynb) | Label-free chain: embeddings → LULC filter → SAM 2, compared with Delineate-Anything v2 on Sentinel-2 and SPOT |
| [16_usa_usgs_naip_plus.py](https://github.com/montimaj/agribound/blob/main/examples/16_usa_usgs_naip_plus.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/16_usa_usgs_naip_plus.ipynb) | USGS NAIP Plus without Earth Engine (contributed by Jeremy Rapp) |
| [17_query_published_ftw_polygons.py](https://github.com/montimaj/agribound/blob/main/examples/17_query_published_ftw_polygons.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/17_query_published_ftw_polygons.ipynb) | Offline `query_ftw` demo with a local tile store (contributed by Jeremy Rapp) |
| [18_agent_orchestration.py](https://github.com/montimaj/agribound/blob/main/examples/18_agent_orchestration.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/18_agent_orchestration.ipynb) | Agent tools and a dry-run plan with the human confirmation gate |
| [19_hpc_tiling.py](https://github.com/montimaj/agribound/blob/main/examples/19_hpc_tiling.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/19_hpc_tiling.ipynb) | Tiling, two-phase runs and merging with `agribound.hpc` |
| [20_stratified_evaluation.py](https://github.com/montimaj/agribound/blob/main/examples/20_stratified_evaluation.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/20_stratified_evaluation.ipynb) | Stratified, size-class and boundary evaluation against NMOSE with bootstrap intervals; overall object and boundary metrics on Landsat, Sentinel-2, SPOT and NAIP of 2018 and with and without the crop filter |
| [21_published_ftw_audit.py](https://github.com/montimaj/agribound/blob/main/examples/21_published_ftw_audit.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/21_published_ftw_audit.ipynb) | Published FTW polygons evaluated against NMOSE |
| [22_global_south_spot_pan.py](https://github.com/montimaj/agribound/blob/main/examples/22_global_south_spot_pan.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/22_global_south_spot_pan.ipynb) | Delineate-Anything v2 on SPOT 6/7 panchromatic (1.5 m, restricted) in six farming landscapes of the Global South |
| [23_tree_crops.py](https://github.com/montimaj/agribound/blob/main/examples/23_tree_crops.py) | [notebook](https://github.com/montimaj/agribound/blob/main/examples/notebooks/23_tree_crops.ipynb) | Tree crops (an oil palm estate, oil palm smallholders, almond and pistachio orchards, olive groves) against RSPO, DWR / Land IQ and SIGPAC reference boundaries: Delineate-Anything v2 (released and fine-tuned), DINOv3 (fine-tuned), SAM 2, FTW and embeddings on SPOT-Pan, NAIP, Landsat PAN and Sentinel-2, and the crop filter with and without `lulc_tree_crops` |

## Google Earth Engine Authentication

Earth Engine is needed for the Landsat, Sentinel-2, HLS, NAIP and SPOT
composites, for Google embeddings with the default backend, for GEE-asset
study areas, and for the **LULC crop filter** (on by default, for every
source). `local`, `usgs-naip-plus` and `tessera-embedding` runs need no Earth
Engine only with `lulc_filter=False`.

```bash
earthengine authenticate                       # once
agribound auth --project YOUR_GEE_PROJECT      # check
```

Credentials are tried in this order: `gee_service_account_key`
(`--gee-service-account-key`), `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`, stored
`earthengine authenticate` credentials, Application Default Credentials
(`GOOGLE_APPLICATION_CREDENTIALS`). Batch jobs (Slurm, no TTY) never fall back
to an interactive browser prompt; they raise with instructions. See
[GEE setup](https://montimaj.github.io/agribound/user-guide/gee-setup/).

## SPOT Access

The SPOT 6/7 collection in Google Earth Engine (`AIRBUS/SPOT6_7`) is
**restricted** and not publicly available; access is limited to select Earth
Engine users. In agribound this source is for internal use at the Desert
Research Institute (DRI). External users who need SPOT-based field boundaries
can contact the package author to request processing.

## Apple Silicon (MPS)

Observed with torch 2.10 on Apple MPS during the 1.0.0 checks:
Delineate-Anything (FP16), FTW, DINOv3 and Prithvi `embed` mode ran on MPS.
GeoAI's Mask R-CNN always runs on CPU (on MPS it reported Metal command-buffer
errors and its detections differed from CPU). Prithvi + UPerNet runs on MPS
only for compatible sizes (for example 192 px chips and tiles) and otherwise
falls back to CPU with a WARNING. SAM masks differ between MPS and CPU. The
Meta SAM 3 backend needs CUDA; use `sam_backend="sam3-hf"` on macOS (both SAM 3
backends are untested). Scripts
that run FTW need an `if __name__ == "__main__":` guard (spawned data-loader
workers).

## Citation

If you use agribound in your research, please cite:

> Majumdar, S., Rapp, J., Huntington, J. L., ReVelle, P., Nozari, S., Smith, R. G., Hasan, M. F., Bromley, M., Atkin, J., Jensen, E. R., Ketchum, D., Abramowitz, J. C., & Roy, S. (2026). *Agribound: Unified agricultural field boundary delineation from satellite imagery using geospatial foundation models, pre-trained segmentation, and embeddings* [Software]. _Zenodo_. https://doi.org/10.5281/zenodo.19229665

> Majumdar, S., Rapp, J., Huntington, J. L., ReVelle, P., Nozari, S., Smith, R. G., Hasan, M. F., Bromley, M., Atkin, J., Jensen, E. R., Ketchum, D., & Roy, S. (2026). *Measuring what geospatial AI delivers for policy-grade agricultural field boundaries*. In prep. for _Remote Sensing of Environment_.

Please also cite the underlying engines, models and data as appropriate:

- **Delineate-Anything**: Lavreniuk, M., Kussul, N., Shelestov, A., Yailymov, B., Salii, Y., Kuzin, V., & Szantoi, Z. (2025). Delineate Anything: Resolution-agnostic field boundary delineation on satellite imagery. European Conference on Artificial Intelligence (ECAI 2025). *arXiv:2504.02534*. https://doi.org/10.48550/arXiv.2504.02534
- **Delineate-Anything v2** (default model): Lavreniuk, M., Kussul, N., Shelestov, A., Salii, Y., Kuzin, V., Wang, C. J. L.-X., & Szantoi, Z. (2026). Delineate Anything v2: A global foundation model for field delineation. European Conference on Computer Vision Workshops (ECCVW 2026), GAIA workshop. *arXiv:2607.19069*. https://doi.org/10.48550/arXiv.2607.19069
- **Fields of The World (FTW)**: Kerner, H., Chaudhari, S., Ghosh, A., Robinson, C., Ahmad, A., Choi, E., Jacobs, N., Holmes, C., Mohr, M., Dodhia, R., Lavista Ferres, J. M., & Marcus, J. (2025). Fields of The World: A machine learning benchmark dataset for global agricultural field boundary segmentation. *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(27), 28151–28159. https://doi.org/10.1609/aaai.v39i27.35034
- **FTW PRUE models** (default FTW model): Muhawenayo, G., Robinson, C., Khanal, S., Fang, Z., Corley, I., Wollam, A., Gao, T., Strnad, L., Avery, R., Estes, L., Tárano, A. M., Jacobs, N., & Kerner, H. (2026). PRUE: A practical recipe for field boundary segmentation at scale. *arXiv:2603.27101*. https://doi.org/10.48550/arXiv.2603.27101
- **Published FTW polygons**: Robinson, C., et al. (2026). The first global agricultural field boundary map at 10m resolution. *arXiv:2605.11055* (preprint). https://doi.org/10.48550/arXiv.2605.11055
- **GeoAI**: Wu, Q. (2026). GeoAI: A Python package for integrating artificial intelligence with geospatial data analysis and visualization. *Journal of Open Source Software*, 11(118), 9605. https://doi.org/10.21105/joss.09605
- **DINOv3**: Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., ... Bojanowski, P. (2025). DINOv3. *arXiv:2508.10104*. https://doi.org/10.48550/arXiv.2508.10104
- **Prithvi-EO-2.0**: Szwarcman, D., Roy, S., Fraccaro, P., et al. (2026). Prithvi-EO-2.0: A versatile multitemporal foundation model for Earth observation applications. *IEEE Transactions on Geoscience and Remote Sensing*, 64, 1–20. https://doi.org/10.1109/TGRS.2025.3642610
- **TerraTorch**: Gomes, C., Blumenstiel, B., de Sousa Almeida, J. L., et al. (2025). TerraTorch: The geospatial foundation models toolkit. *IGARSS 2025*, 6364–6368. https://doi.org/10.1109/IGARSS55030.2025.11243570
- **TESSERA**: Feng, Z., Atzberger, C., Jaffer, S., Knezevic, J., Sormunen, S., Young, R., Lisaius, M. C., Immitzer, M., Jackson, T., Ball, J., Coomes, D. A., Madhavapeddy, A., Blake, A., & Keshav, S. (2026). TESSERA: Temporal embeddings of surface spectra for Earth representation and analysis. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 34818–34831. arXiv:2506.20380
- **Google Satellite Embeddings (AlphaEarth)**: Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., Rucklidge, W. J., Samsikova, M., Zhang, C., Shelhamer, E., Lahera, E., Wiles, O., Ilyushchenko, S., Gorelick, N., Zhang, L. L., Alj, S., Schechter, E., Askay, S., Guinan, O., Moore, R., Boukouvalas, A., & Kohli, P. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. *arXiv:2507.22291*. https://doi.org/10.48550/arXiv.2507.22291
- **SAM 2**: Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., & Feichtenhofer, C. (2025). SAM 2: Segment anything in images and videos. *ICLR 2025*. arXiv:2408.00714
- **SAM 3**: Carion, N., Gustafson, L., Hu, Y.-T., et al. (2026). SAM 3: Segment anything with concepts. *ICLR 2026*. arXiv:2511.16719
- **SamGeo**: Wu, Q., & Osco, L. P. (2023). samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM). *Journal of Open Source Software*, 8(89), 5663. https://doi.org/10.21105/joss.05663
- **SAM for Remote Sensing**: Osco, L. P., Wu, Q., de Lemos, E. L., Gonçalves, W. N., Ramos, A. P. M., Li, J., & Marcato Junior, J. (2023). The Segment Anything Model (SAM) for remote sensing applications: From zero to one shot. *International Journal of Applied Earth Observation and Geoinformation*, 124, 103540. https://doi.org/10.1016/j.jag.2023.103540
- **geemap**: Wu, Q. (2020). geemap: A Python package for interactive mapping with Google Earth Engine. *Journal of Open Source Software*, 5(51), 2305. https://doi.org/10.21105/joss.02305
- **Google Earth Engine**: Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18–27. https://doi.org/10.1016/j.rse.2017.06.031
- **Awesome GEE Community Catalog**: Roy, S., Majumdar, S., & Swetnam, T. (2025). samapriya/awesome-gee-community-datasets: Community Catalog (3.9.0). Zenodo. https://doi.org/10.5281/zenodo.17641528

More references (HLS, Dynamic World, NLCD, C3S, Cloud Score+, evaluation
methods, ACCESS) are listed in the
[documentation](https://montimaj.github.io/agribound/citation/).

## License

This project is licensed under the [Apache License 2.0](https://github.com/montimaj/agribound/blob/main/LICENSE). The
Delineate-Anything model code and weights, and Ultralytics, are AGPL-3.0.
The DINOv3 weights used by the `dinov3` engine are under the
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
(custom, not OSI-approved), whose clause 1.b.ii requires publications to
acknowledge the use of the DINO Materials; the SAM 3 licence has a similar
clause. Check the licences of the models and datasets you use.

## Acknowledgments

Agribound builds on the work of many open-source projects and research teams:

- The **Ultralytics** team for the YOLO ecosystem
- **Mykola Lavreniuk** and co-authors for Delineate-Anything
- **Meta AI Research** for the Segment Anything models and DINOv3
- The **Fields of The World** consortium and Hannah Kerner's group at Arizona State University
- **Qiusheng Wu** for the GeoAI and samgeo Python packages
- **NASA** and **IBM Research** for the Prithvi geospatial foundation model and TerraTorch
- **Google DeepMind** for AlphaEarth satellite embeddings
- **Feng et al.** for the TESSERA foundation model embeddings
- The **Google Earth Engine** team for planetary-scale geospatial computing
- The **fiboa** community for the field boundary schema standard
- The **TorchGeo** team for geospatial deep learning data loaders and utilities
- The **Desert Research Institute (DRI)** for supporting this research
- **Jacob Abramowitz** (The University of Alabama in Huntsville) for asking about tree crops and pointing to the RSPO concession maps (example 23)
- The **Roundtable on Sustainable Palm Oil (RSPO)** (GeoRSPO concession maps), the **California Department of Water Resources** and **Land IQ** (Statewide Crop Mapping) and the **Fondo Español de Garantía Agraria** (SIGPAC) for the reference boundaries of example 23

## Funding

This work was supported by multiple funding sources. The **New Mexico Office of the State Engineer (NMOSE)** provided reference field boundary data and supported the development of agricultural water use mapping in New Mexico. The **Google Satellite Embeddings Dataset Small Grants Program** enabled the integration of pre-computed satellite embeddings for unsupervised field boundary delineation. Access to the **SPOT 6 and 7 archive on Google Earth Engine** was provided through the Google Trusted Tester opportunity. Additional support was provided by the **U.S. Army Corps of Engineers** and **The U.S. Department of Treasury/State of Nevada**. This work was also supported by the **NASA Water Resources Applications Program**, the **United States Geological Survey (USGS)** and **NASA Landsat Science Team**, the **USGS Water Resources Research Institute**, the **Desert Research Institute Maki Endowment**, and the **Windward Fund**.

## AI Usage Disclosure

Portions of this software were developed with the assistance of AI coding tools, including Anthropic's Claude. AI was used to accelerate code scaffolding, documentation drafting, and test generation. All AI-generated code was reviewed, tested, and validated by the human authors. The scientific methodology, architectural decisions, algorithm selection, and domain-specific implementations reflect the expertise and judgment of the authors.
