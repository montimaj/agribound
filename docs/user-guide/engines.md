# Delineation Engines

Agribound provides seven delineation engines, each suited to different use cases, satellite sources, and hardware configurations.

## Engine Comparison

| Engine | Key | Approach | Strengths | GPU Required | Reference |
|---|---|---|---|---|---|
| Delineate-Anything | `delineate-anything` | YOLO instance segmentation (2 model variants) | Fast; resolution-agnostic (1--10 m+); routes through FTW for S2 with native MPS support | Recommended | [Lavreniuk et al. (2025)](https://arxiv.org/abs/2504.02534) |
| Fields of The World | `ftw` | Semantic segmentation (14+ models: EfficientNet-B3/B5/B7, UNet, UPerNet) | Strong generalization; 25-country training set; bi-temporal input (planting + harvest); all models via `list_ftw_models()` | Yes | [Kerner et al. (2025)](https://fieldsofthe.world/) |
| GeoAI Field Boundary | `geoai` | Mask R-CNN instance segmentation | Built-in NDVI support; auto-falls back to CPU on Apple Silicon (MPS). **Without fine-tuning on region-specific reference data, GeoAI typically does not delineate any fields** | No | [Wu (2026)](https://github.com/opengeos/geoai) |
| DINOv3 | `dinov3` | DINOv3 ViT backbone (SAT-493M satellite-pretrained) + DPT segmentation head | Satellite-native ViT features pretrained on 493M satellite images; LoRA fine-tuning; resolution-agnostic | Yes | [Siméoni et al. (2025)](https://arxiv.org/abs/2508.10104) |
| Prithvi-EO-2.0 | `prithvi` | NASA/IBM ViT foundation model (embed / PCA / segment modes) | 1024-D ViT embeddings from 6 HLS bands; PCA baseline for comparison. **ViT embed mode requires fine-tuning for good results** | Recommended (embed); No (PCA) | [Szwarcman et al. (2024)](https://arxiv.org/abs/2412.02732) |
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

### GeoAI Field Boundary

Mask R-CNN instance segmentation from the `geoai-py` package. Includes built-in NDVI computation for enhanced multi-spectral input.

```bash
pip install agribound[geoai]
```

**Supported sources**: `sentinel2`, `naip`, `local`

**Reference**: geoai-py package

!!! warning "Apple Silicon (MPS)"
    Mask R-CNN is unstable on Apple Silicon GPUs via MPS (Metal Performance Shaders). Metal command buffer errors cause crashes during both training and inference. Agribound automatically detects MPS and falls back to CPU for all GeoAI operations. All other engines (FTW, Delineate-Anything, Prithvi) work correctly on MPS.

### DINOv3

DINOv3 Vision Transformer backbone with a DPT (Dense Prediction Transformer) segmentation head. Uses LoRA-efficient fine-tuning with a frozen backbone for fast adaptation on reference boundaries. Resolution-agnostic — works across all satellite sources.

```bash
pip install agribound[geoai]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `naip`, `spot`, `local`

**Requires fine-tuning**: Yes — DINOv3 requires fine-tuning on reference boundaries to produce meaningful field segmentation. Set `fine_tune=True` with `reference_boundaries`.

**Reference**: Siméoni et al. (2025), DINOv3

### Prithvi-EO-2.0

NASA/IBM foundation model (300M-parameter Vision Transformer) pretrained on HLS imagery with masked autoencoders. Supports three modes:

- **`embed`** (default) — Extracts 1024-D ViT encoder embeddings from 224×224 patches, then K-means clusters them to delineate fields. Uses all 6 HLS bands (Blue, Green, Red, NIR, SWIR1, SWIR2) with Prithvi's pre-training normalization. GPU recommended. **Without fine-tuning, ViT embeddings tend to produce very few, over-merged fields.** Fine-tuning on reference boundaries is recommended for production use.
- **`pca`** — Lightweight baseline that clusters PCA-reduced spectral bands (R, G, B, NIR) without running the ViT encoder. No GPU or `transformers` needed. Useful for comparison.
- **`segment`** — Fine-tuned UPerNet decoder via terratorch. Requires a checkpoint from fine-tuning on reference boundaries.

```bash
pip install agribound[prithvi]
```

```python
# ViT embedding mode (default)
agribound.delineate(..., engine="prithvi", engine_params={"mode": "embed"})

# PCA baseline
agribound.delineate(..., engine="prithvi", engine_params={"mode": "pca"})

# Fine-tuned segmentation
agribound.delineate(..., engine="prithvi",
                    engine_params={"mode": "segment", "checkpoint_path": "..."})
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `local`

**Reference**: [Szwarcman et al. (2024), Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732)

### Embedding Clustering

Unsupervised approach using K-means or spectral clustering on pre-computed pixel embeddings. Does not require a GPU. Designed for use with the Google Satellite Embedding V1 and TESSERA embedding datasets.

```bash
pip install agribound                # Google Embeddings (no extra deps)
pip install agribound[tessera]       # TESSERA Embeddings
```

**Supported sources**: `google-embedding`, `tessera-embedding`

**Reference**: [Google AlphaEarth](https://arxiv.org/abs/2507.22291), [TESSERA (Feng et al.)](https://arxiv.org/abs/2506.20380)

### Ensemble

Combines outputs from multiple engines using majority vote or polygon intersection. Runs the specified constituent engines and merges their results to improve robustness.

```bash
pip install agribound[all]
```

**Supported sources**: `landsat`, `sentinel2`, `hls`, `naip`, `spot`, `local`

!!! warning "When to use ensembles"
    Ensembles work best when **multiple models run on the same sensor data**. Each architecture has different biases, and vote-merging cancels out individual errors because every model sees the same pixels.

    Ensembles across **different sensors** (e.g., Sentinel-2 + Landsat + NAIP) do not work well due to resolution mismatch (1 m vs 30 m polygons), temporal mismatch (different overpass dates), and spatial alignment errors. For multi-sensor analysis, compare per-source results independently rather than merging them.

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
| Fine-tuning on reference boundaries (any sensor) | `dinov3` (SAT-493M) |
| Multi-temporal Landsat/HLS analysis (6 bands) | `prithvi` (embed mode) |
| No GPU, no reference data, global coverage | `embedding` + LULC filter |
| Maximum accuracy, multiple engines on same sensor | `ensemble` |

## Recommended Workflows

| Situation | Approach | Example |
|---|---|---|
| **Reference boundaries available** | DINOv3 + SAM2 per source | Example 14 |
| **No reference boundaries** | Embedding clustering + LULC filter + SAM2 | Example 15 |
| **Multi-model ensemble** | All engines on same sensor, majority vote | Example 12 |
| **Multi-year time series** | Single engine per year, fine-tune once | Example 01 |
| **Quick local test** | Delineate-Anything on local GeoTIFF | Example 10 |

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
