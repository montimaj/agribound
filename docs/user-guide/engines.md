# Delineation Engines

Agribound provides seven delineation engines, each suited to different use cases, satellite sources, and hardware configurations.

## Engine Comparison

| Engine | Key | Approach | Strengths | GPU Required | Reference |
|---|---|---|---|---|---|
| Delineate-Anything | `delineate-anything` | YOLO instance segmentation (2 model variants) | Fast; resolution-agnostic (1--10 m+); routes through FTW for S2 with native MPS support | Recommended | [Lavreniuk et al. (2025)](https://arxiv.org/abs/2504.02534) |
| Fields of The World | `ftw` | Semantic segmentation (14+ models: EfficientNet-B3/B5/B7, UNet, UPerNet) | Strong generalization; 25-country training set; bi-temporal input (planting + harvest); all models via `list_ftw_models()` | Yes | [Kerner et al. (2024)](https://fieldsofthe.world/) |
| GeoAI Field Boundary | `geoai` | Mask R-CNN instance segmentation | Easy to use; built-in NDVI support; auto-falls back to CPU on Apple Silicon (MPS) | No | [Wu (2026)](https://github.com/opengeos/geoai) |
| DINOv3 | `dinov3` | DINOv2/v3 ViT backbone + DPT segmentation head | Powerful ViT features; LoRA fine-tuning; resolution-agnostic | Yes | [Siméoni et al. (2025)](https://arxiv.org/abs/2508.10104) |
| Prithvi-EO-2.0 | `prithvi` | NASA/IBM geospatial foundation model with TerraTorch fine-tuning | State-of-the-art foundation model; multi-temporal | Yes | [Jakubik et al. (2024)](https://huggingface.co/ibm-nasa-geospatial) |
| Embedding | `embedding` | Unsupervised clustering of pre-computed embeddings | No GPU needed; no labeled data required | No | [Brown et al. (2025)](https://arxiv.org/abs/2507.22291), [Feng et al. (2025)](https://arxiv.org/abs/2506.20380) |
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

!!! warning "Apple Silicon (MPS)"
    Mask R-CNN is unstable on Apple Silicon GPUs via MPS (Metal Performance Shaders). Metal command buffer errors cause crashes during both training and inference. Agribound automatically detects MPS and falls back to CPU for all GeoAI operations. All other engines (FTW, Delineate-Anything, Prithvi) work correctly on MPS.

### DINOv3

DINOv2/v3 Vision Transformer backbone with a DPT (Dense Prediction Transformer) segmentation head. Uses LoRA-efficient fine-tuning with a frozen backbone for fast adaptation on reference boundaries. Resolution-agnostic — works across all satellite sources.

```bash
pip install agribound[geoai]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `naip`, `spot`, `local`

**Requires fine-tuning**: Yes — DINOv3 requires fine-tuning on reference boundaries to produce meaningful field segmentation. Set `fine_tune=True` with `reference_boundaries`.

**Reference**: Siméoni et al. (2025), DINOv3

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

**Reference**: [Google AlphaEarth](https://arxiv.org/abs/2507.22291), [TESSERA (Feng et al.)](https://arxiv.org/abs/2506.20380)

### Ensemble

Combines outputs from multiple engines using majority vote or polygon intersection. Runs the specified constituent engines and merges their results to improve robustness.

```bash
pip install agribound[all]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `naip`, `spot`, `local`

## SAM2 Boundary Refinement

SAM2 is not a standalone engine — it is an optional **post-processing step** that refines field boundaries. Each polygon's bounding box is fed to SAM2 as a prompt, and SAM2 produces a pixel-accurate mask that replaces the original geometry.

```bash
pip install agribound[samgeo]
```

**Recommended usage:** Apply SAM2 to the **final ensemble output** rather than per-engine, since refinement scales linearly with the number of polygons. For large study areas (thousands of fields), use `sam_model="tiny"` for faster processing.

For single-engine runs, enable via `engine_params`:

```python
gdf = agribound.delineate(
    ...,
    engine_params={"sam_refine": True, "sam_model": "tiny"},
)
```

For ensemble workflows, call `refine_boundaries()` directly on the merged result:

```python
from agribound.engines.samgeo_engine import refine_boundaries

gdf = refine_boundaries(ensemble_gdf, raster_path, config)
```

SAM2 model variants: `"tiny"`, `"small"`, `"base_plus"`, `"large"` (default). Batch size configurable via `engine_params["sam_batch_size"]` (default 100).

**Reference**: [Wu & Osco (2023)](https://doi.org/10.21105/joss.05663), [Ravi et al. (2024)](https://arxiv.org/abs/2408.00714)

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
