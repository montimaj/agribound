# Installation

## Requirements

- **Python >= 3.12** (ftw-tools 2.x, geoai-py >= 0.41, geotessera >= 0.8 and
  torchgeo 0.10 require it). Python 3.12 and 3.13 are tested in CI.
- GDAL, PROJ and GEOS for rasterio, geopandas, pyproj and shapely. The PyPI
  wheels of these packages bundle the libraries; conda-forge is recommended
  because it keeps them consistent. The GDAL Python bindings (`osgeo`, from
  the conda-forge `gdal` package) are needed only by the Delineate-Anything
  `backend="reference"` option.

## Two environments

terratorch (needed by the **Prithvi** engine) requires `lightning>=2.6`, while
ftw-tools 2.0.0b5 (needed by the **FTW** engine) requires `lightning<2.6`, so
they cannot be installed in one environment. agribound therefore ships two
environment files and two "everything" extras:

| Environment | File | Extra | Contains | Excludes |
|---|---|---|---|---|
| core | `environment.yml` (env name `agribound`) | `agribound[all]` | GEE, Delineate-Anything, FTW, GeoAI, DINOv3, SAM 2, SAM 2.1 automatic masks, TESSERA, crop-sequence, agent | Prithvi, SAM 3 |
| GFM | `environment-gfm.yml` (env name `agribound-gfm`) | `agribound[all-gfm]` | GEE, Delineate-Anything, GeoAI, DINOv3, Prithvi, SAM 2, SAM 2.1 automatic masks, TESSERA, crop-sequence, agent | FTW, SAM 3 |

Both environment files install agribound in editable mode from the
repository root (`pip install -e .[all,dev]` or `.[all-gfm,dev]`), so create
them from a clone:

```bash
git clone https://github.com/montimaj/agribound.git
cd agribound
conda env create -f environment.yml          # core
conda activate agribound
# or
conda env create -f environment-gfm.yml      # Prithvi / terratorch
conda activate agribound-gfm
```

Without conda, into a fresh Python 3.12 environment:

```bash
pip install "agribound[all]"        # core
pip install "agribound[all-gfm]"    # in a separate environment, for Prithvi
```

!!! note "ftw-tools is a pre-release"
    ftw-tools 2.x is published on PyPI only as pre-releases (2.0.0b5 as of
    2026-09). The `ftw` extra requests `ftw-tools>=2.0.0b5,<3`, and the
    explicit pre-release lower bound lets pip select it. ftw-tools 1.4.x (the
    latest stable release) is not compatible with agribound.

## Extras

| Extra | Installs | Needed for |
|---|---|---|
| `gee` | `earthengine-api>=1.7.45,<2`, `geemap>=0.37`, `geedim>=2.0,<3` | Earth Engine sources, `google-embedding` (default backend), the LULC filter |
| `delineate-anything` | `ultralytics>=8.4.80,<8.5`, `opencv-python`, `huggingface-hub`, `psutil`, `numba>=0.58` | the `delineate-anything` engine (`numba` only for the reference backend) |
| `ftw` | `ftw-tools>=2.0.0b5,<3`, `torch>=2.4`, `torchgeo>=0.9`, `segmentation-models-pytorch>=0.5` | the `ftw` engine; the Delineate-Anything `ftw` backend |
| `geoai` | `geoai-py>=0.43.1` | the `geoai` engine |
| `dinov3` | `geoai-py>=0.43.1` | the `dinov3` engine |
| `prithvi` | `terratorch[peft]>=1.2.13,<1.3` | the `prithvi` engine (conflicts with `ftw`) |
| `samgeo` | `segment-geospatial[samgeo2]>=1.4.2` | SAM refinement with `sam2`/`sam2.1` |
| `sam-auto` | `sam2==1.1.0`, `huggingface-hub>=0.34,<2`, `opencv-python>=4.8` | the `sam-auto` engine (SAM 2.1 automatic masks; see below); no segment-geospatial |
| `sam3` | `segment-geospatial[samgeo3]>=1.4.2`, `triton-windows` on Windows | SAM refinement with the Meta `sam3` backend (CUDA; see [SAM refinement](user-guide/sam-refinement.md#sam-3-platform-support)); **untested** |
| `tessera` | `geotessera>=0.10.2,<0.11` | `tessera-embedding` |
| `embedding` | `agribound[tessera,gee]` | the `embedding` engine on both embedding sources |
| `crop-sequence` | `agribound[gee]`, `scikit-image>=0.24` | the `crop-sequence` engine and the `cdl-sequence` and `aci-sequence` sources |
| `agent` | `anthropic>=1.8,<2`, `mcp>=2.2,<3` | the [agent layer](user-guide/agent.md) and MCP server |
| `all` | `gee,delineate-anything,ftw,geoai,dinov3,samgeo,sam-auto,tessera,agent,crop-sequence` | everything except `prithvi` and `sam3` |
| `all-gfm` | `gee,delineate-anything,geoai,dinov3,prithvi,samgeo,sam-auto,tessera,agent,crop-sequence` | everything except `ftw` and `sam3` |
| `docs`, `dev` | MkDocs toolchain; pytest, pytest-cov, pytest-timeout, ruff | building the docs; running the tests |

The `sam3-hf` SAM backend (also **untested**) needs `transformers>=5`
(installed by the `sam3` extra, or install it directly). The Google-embedding
`source_coop` backend needs only the core dependencies and network access to
`data.source.coop`.

The `sam-auto` extra installs the [SAM 2.1 automatic-mask
engine](user-guide/engines.md#sam-21-automatic-masks-sam-auto) without
segment-geospatial, whose patool dependency is GPL-3. `sam2` 1.1.0 on PyPI is
a third-party build of Meta's Apache-2.0 SAM 2 code: its 62 code,
configuration and licence files are byte-identical to Meta's commit 2b90b9f
(2024-12-15), and `engine_meta` records a SHA-256 of the installed files.
The extra pins that version, because the engine calls predictor internals
checked with it only; sam2 brings torch and torchvision. The checkpoint,
`facebook/sam2.1-hiera-large` (898 MB), is downloaded from Hugging Face on
first use at a pinned revision and checked by SHA-256; `agribound prefetch
--engine sam-auto` downloads it ahead of time (for example for offline
nodes; see [HPC](user-guide/hpc.md#prefetching-weights)). CI's engine job
installs the extra and checks that segment-geospatial is not installed with
it.

The core install (`pip install agribound`) covers configuration, local
rasters, post-processing, evaluation, the registries and the CLI. A missing
optional dependency raises an `ImportError` with the install command when the
feature that needs it is used.

## Verifying the installation

```bash
agribound --version            # 1.1.0
agribound list-engines
agribound list-sources
agribound list-ftw-models      # needs the ftw extra
```

## Apple silicon (MPS)

Measured with torch 2.10 on Apple MPS during the 1.0.0 checks:

- Delineate-Anything (native backend, FP16), FTW, DINOv3 and Prithvi `embed`
  mode ran on MPS.
- GeoAI's Mask R-CNN always runs on CPU (WARNING): on MPS it reported Metal
  command-buffer errors and its detections differed from CPU and between runs.
- Prithvi + UPerNet (`segment` mode and fine-tuning) runs on MPS only for
  compatible input sizes, for example 192 px tiles and chips; the default
  224 px falls back to CPU with a WARNING.
- SAM masks differ between MPS and CPU (IoU 0.59-0.97 on a Sentinel-2 test
  crop). The `sam-auto` engine ran on MPS in its evaluation (2026-10-13);
  its output is identical on one device with the same versions and differs
  between devices.
- The Meta SAM 3 backend needs CUDA and is not available on macOS; use
  `sam_backend="sam3-hf"`. Both SAM 3 backends are **untested** in 1.1.0 (see
  [SAM refinement](user-guide/sam-refinement.md#sam-3-is-untested)).
- Scripts that run FTW must use an `if __name__ == "__main__":` guard (the
  data-loader workers use the `spawn` start method).

## Development install

```bash
git clone https://github.com/montimaj/agribound.git
cd agribound
conda env create -f environment.yml      # installs -e .[all,dev]
conda activate agribound
pip install -e ".[docs]"                 # optional: documentation toolchain
python -m pytest -m "not gpu and not gee and not slow and not network"
```

## Troubleshooting

- **Dependency conflicts** (for example around `lightning`): use a fresh
  environment from one of the two environment files, and do not install the
  `ftw` and `prithvi` extras together.
- **Check which pip is active** after activating an environment:
  `which pip` (Linux/macOS) or `where pip` (Windows), and `pip --version`.
- **GPU wheels**: current PyPI Linux torch wheels are CUDA 13 builds, which
  need a recent NVIDIA driver and do not support Volta (V100) GPUs; on such
  systems install matching `torch`/`torchvision` wheels from a CUDA 12 index
  (see `examples/hpc/README.md`).
