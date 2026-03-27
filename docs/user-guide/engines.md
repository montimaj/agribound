# Delineation Engines

Agribound provides six delineation engines, each suited to different use cases, satellite sources, and hardware configurations.

## Engine Comparison

| Engine | Key | Approach | Strengths | GPU Required | Reference |
|---|---|---|---|---|---|
| Delineate-Anything | `delineate-anything` | YOLO instance segmentation (2 model variants) | Fast; resolution-agnostic (1--10 m+); routes through FTW for S2 with native MPS support | Recommended | [Lavreniuk et al. (2025)](https://arxiv.org/abs/2504.02534) |
| Fields of The World | `ftw` | Semantic segmentation (14+ models: EfficientNet-B3/B5/B7, UNet, UPerNet) | Strong generalization; 25-country training set; bi-temporal input (planting + harvest); all models via `list_ftw_models()` | Yes | [Kerner et al. (2024)](https://fieldsofthe.world/) |
| GeoAI Field Boundary | `geoai` | Esri GeoAI segmentation model | Easy to use; ArcGIS-compatible | No | [Esri GeoAI](https://github.com/Esri/geoai-py) |
| Prithvi-EO-2.0 | `prithvi` | NASA/IBM geospatial foundation model with TerraTorch fine-tuning | State-of-the-art foundation model; multi-temporal | Yes | [Jakubik et al. (2024)](https://huggingface.co/ibm-nasa-geospatial) |
| Embedding | `embedding` | Unsupervised clustering of pre-computed embeddings | No GPU needed; no labeled data required | No | [Aung et al. (2024)](https://sites.research.google/gr/open-buildings/) |
| Ensemble | `ensemble` | Multi-engine or multi-model consensus (vote / union / intersection) | Best accuracy; supports running same engine with different models | Depends on engines | -- |

## Engine Details

### Delineate-Anything

Instance segmentation based on Ultralytics YOLO (DelineateAnything and DelineateAnything-S), trained on the FBIS-22M dataset. Resolution-agnostic: works across 1 m (NAIP) to 10 m+ (Sentinel-2) imagery.

For **Sentinel-2**, DA automatically routes through FTW's built-in instance segmentation with proper S2 preprocessing (`/3000` normalization) and native MPS (Apple GPU) support. For all other sensors, the standalone DA pipeline with sensor-agnostic percentile normalization is used.

```bash
pip install agribound[delineate-anything]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `naip`, `spot`, `local`

**Fine-tuning**: Supported (YOLO). Chips are converted to PNG with percentile-normalized uint8 RGB.

**Reference**: arXiv:2504.02534

### Fields of The World (FTW)

Semantic segmentation using EfficientNet-B3/B5/B7, UNet, UPerNet, and DeepLabV3+ architectures. Ships with 14+ pre-trained models covering 25 countries. All models are available via `agribound.list_ftw_models()`. Produces field interior and boundary masks that are then polygonized.

```bash
pip install agribound[ftw]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `local`

**Fine-tuning**: Not yet supported (requires paired temporal windows). Pre-trained weights are used directly.

**Reference**: Fields of The World (FTW) dataset

### GeoAI Field Delineator

Mask R-CNN instance segmentation from the `geoai-py` package. Includes built-in NDVI computation for enhanced multi-spectral input.

```bash
pip install agribound[geoai]
```

**Supported sources**: `sentinel2`, `naip`, `local`

**Reference**: geoai-py package

### Prithvi-EO-2.0

NASA/IBM foundation model (Vision Transformer) fine-tuned for Earth observation. Uses terratorch for segmentation with a UPerNet decoder. Requires 4-band input (R, G, B, NIR).

```bash
pip install agribound[prithvi]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `local`

**Reference**: NASA/IBM Prithvi-EO-2.0

### Embedding Clustering

Unsupervised approach using K-means or spectral clustering on pre-computed pixel embeddings. Does not require a GPU. Designed for use with the Google Satellite Embedding V1 and TESSERA embedding datasets.

```bash
pip install agribound[geoai]
```

**Supported sources**: `google-embedding`, `tessera-embedding`

**Reference**: TESSERA (CVPR 2026) + Google AlphaEarth

### Ensemble

Combines outputs from multiple engines using majority vote or polygon intersection. Runs the specified constituent engines and merges their results to improve robustness.

```bash
pip install agribound[all]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `naip`, `spot`, `local`

## When to Use Each Engine

| Scenario | Recommended Engine |
|---|---|
| High-resolution imagery (1--6 m), NAIP or SPOT | `delineate-anything` |
| Sentinel-2 in a country covered by FTW pre-trained models | `ftw` |
| General-purpose Sentinel-2 or NAIP with NDVI | `geoai` |
| Multi-temporal Landsat/HLS analysis | `prithvi` |
| No GPU available, pre-computed embeddings exist | `embedding` |
| Maximum accuracy, multiple engines available | `ensemble` |

## GPU Requirements

All engines except `embedding` require a CUDA-capable GPU for inference. The `device` configuration parameter controls hardware selection:

```python
config = AgriboundConfig(
    device="auto",  # auto-detect: cuda > mps > cpu
    ...
)
```

Supported values: `auto`, `cuda`, `cpu`, `mps` (Apple Silicon).

!!! warning
    Running GPU-required engines on CPU is technically possible but will be extremely slow for anything beyond small test areas.
