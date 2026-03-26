# Fine-Tuning

Agribound supports fine-tuning delineation engines on user-provided reference field boundaries. This can significantly improve accuracy for specific regions or crop types not well represented in the default pre-trained models.

## Overview

The fine-tuning workflow:

1. Reference boundaries are rasterized into 3-class segmentation masks (background, field interior, field boundary).
2. The raster and masks are chipped into training patches.
3. Patches are split into train/validation sets.
4. The selected engine is fine-tuned on the training data.
5. The fine-tuned model is used for inference on the full study area.

## Providing Reference Boundaries

Reference boundaries must be a vector file (Shapefile, GeoPackage, GeoJSON, or GeoParquet) containing field boundary polygons:

```python
from agribound import AgriboundConfig, delineate

config = AgriboundConfig(
    study_area="area.geojson",
    source="sentinel2",
    year=2024,
    engine="ftw",
    gee_project="my-project",
    reference_boundaries="reference_fields.gpkg",
    fine_tune=True,
    fine_tune_epochs=20,
    fine_tune_val_split=0.2,
)

gdf = delineate(config=config, study_area=config.study_area)
```

Via CLI:

```bash
agribound delineate \
    --study-area area.geojson \
    --source sentinel2 \
    --engine ftw \
    --gee-project my-project \
    --reference reference_fields.gpkg \
    --fine-tune
```

!!! note
    When `--reference` is provided without `--fine-tune`, agribound evaluates the delineation results against the reference boundaries instead of fine-tuning.

## Engine-Specific Fine-Tuning

### FTW (Default for Fine-Tuning)

FTW uses PyTorch Lightning for training. It is the default and most robust fine-tuning pipeline.

- Architecture: UNet, UPerNet, or DeepLabV3+
- Input: 3-band (R, G, B) chips
- Framework: torchgeo + Lightning

```python
config = AgriboundConfig(
    engine="ftw",
    fine_tune=True,
    fine_tune_epochs=30,
    engine_params={
        "ftw_model": "unet-s2-rgb",
        "batch_size": 8,
    },
    ...
)
```

### Delineate-Anything

Fine-tunes the YOLO instance segmentation model. Training data is converted to YOLO segmentation format automatically.

```python
config = AgriboundConfig(
    engine="delineate-anything",
    fine_tune=True,
    engine_params={
        "model_size": "small",  # or "large"
        "chip_size": 256,
    },
    ...
)
```

### GeoAI

Fine-tunes the Mask R-CNN model from the geoai-py package.

```python
config = AgriboundConfig(
    engine="geoai",
    fine_tune=True,
    engine_params={
        "batch_size": 4,
    },
    ...
)
```

### Prithvi

Fine-tunes the Prithvi-EO-2.0 foundation model with a UPerNet decoder via terratorch. Requires 4-band input (R, G, B, NIR).

```python
config = AgriboundConfig(
    engine="prithvi",
    fine_tune=True,
    engine_params={
        "model_name": "Prithvi-EO-2.0-300M-TL",
        "batch_size": 4,
    },
    ...
)
```

## Configuration Options

| Parameter | Default | Description |
|---|---|---|
| `reference_boundaries` | None | Path to vector file with reference field polygons. |
| `fine_tune` | False | Enable fine-tuning before inference. |
| `fine_tune_epochs` | 20 | Number of training epochs. |
| `fine_tune_val_split` | 0.2 | Fraction of data reserved for validation. |
| `engine_params.chip_size` | 256 | Patch size for chipping (pixels). |
| `engine_params.batch_size` | 4-8 | Training batch size (engine-dependent). |

## Fallback Behavior

If the selected engine does not support fine-tuning (e.g., `embedding` or `ensemble`), agribound automatically falls back to the best-supported engine for the satellite source:

| Source | Fallback Engine |
|---|---|
| `sentinel2`, `hls`, `landsat`, `local` | `ftw` |
| `naip`, `spot` | `delineate-anything` |

A warning is logged when a fallback occurs.

## Tips

- Provide at least 50--100 reference field polygons for meaningful fine-tuning.
- Use polygons that are representative of the study area's field sizes and shapes.
- Start with 10--20 epochs and increase if validation loss is still decreasing.
- Fine-tuned checkpoints are saved in the `.agribound_cache/checkpoints/` directory next to the output path.
