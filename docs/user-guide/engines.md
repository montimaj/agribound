# Delineation Engines

Agribound provides six delineation engines, each suited to different use cases, satellite sources, and hardware configurations.

## Engine Comparison

| Engine | Approach | GPU | Input Bands | Best For |
|---|---|---|---|---|
| `delineate-anything` | YOLO instance segmentation | Required | R, G, B | High-res imagery (NAIP, SPOT), resolution-agnostic (1 m--10 m+) |
| `ftw` | Semantic segmentation (UNet/UPerNet/DeepLabV3+) | Required | R, G, B | Sentinel-2, 25-country pre-trained models |
| `geoai` | Mask R-CNN instance segmentation | Required | R, G, B | Sentinel-2, NAIP, built-in NDVI support |
| `prithvi` | Foundation model (ViT) + segmentation | Required | R, G, B, NIR | HLS, Landsat, multi-temporal analysis |
| `embedding` | Unsupervised K-means/spectral clustering | Not required | Embeddings | Pre-computed embedding datasets, no GPU needed |
| `ensemble` | Multi-engine consensus (majority vote) | Required | Varies | Combining strengths of multiple engines |

## Engine Details

### Delineate-Anything

Instance segmentation based on Ultralytics YOLO, trained on global agricultural field boundaries. Resolution-agnostic: works across 1 m (NAIP) to 10 m+ (Sentinel-2) imagery.

```bash
pip install agribound[delineate-anything]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `naip`, `spot`, `local`

**Reference**: arXiv:2504.02534

### Fields of The World (FTW)

Semantic segmentation using architectures like UNet, UPerNet, and DeepLabV3+. Ships with 16+ pre-trained models covering 25 countries. Produces field interior and boundary masks that are then polygonized.

```bash
pip install agribound[ftw]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `local`

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
