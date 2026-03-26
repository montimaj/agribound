# Satellite Sources

Agribound supports multiple satellite imagery sources for building annual composites. Each source is accessed through Google Earth Engine or loaded from local files.

## Source Overview

| Source | Name | Resolution | Coverage | GEE Collection | Requires GEE |
|---|---|---|---|---|---|
| `landsat` | Landsat 8/9 | 30 m | Global, 1985--present | `LANDSAT/LC08/C02/T1_L2` + `LANDSAT/LC09/C02/T1_L2` | Yes |
| `sentinel2` | Sentinel-2 L2A | 10 m | Global, 2017--present | `COPERNICUS/S2_SR_HARMONIZED` | Yes |
| `hls` | Harmonized Landsat-Sentinel | 30 m | Global, 2013--present | `NASA/HLS/HLSL30/v002` + `NASA/HLS/HLSS30/v002` | Yes |
| `naip` | NAIP | 1 m | Continental US, ~2-3 year cycle | `USDA/NAIP/DOQQ` | Yes |
| `spot` | SPOT 6/7 | 6 m | Global (restricted), 2012--2023 | `AIRBUS/SPOT6_7` | Yes |
| `local` | Local GeoTIFF | Varies | User-provided | N/A | No |
| `google-embedding` | Google Satellite Embedding V1 | 10 m | Global, 2017--2025 | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | No |
| `tessera-embedding` | TESSERA Embeddings | 10 m | Global, 2017--2024 | N/A | No |

!!! note "SPOT Access"
    SPOT 6/7 imagery (`AIRBUS/SPOT6_7`) is restricted to select GEE users and is for internal DRI use only. External users who need SPOT-based field boundaries should contact the package author.

## Downloaded Bands

All spectral bands are downloaded for each source. Engines automatically select the bands they need using the canonical band mapping.

| Source | All Bands | Canonical R | Canonical G | Canonical B | Canonical NIR |
|---|---|---|---|---|---|
| `landsat` | SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7 | SR_B4 | SR_B3 | SR_B2 | SR_B5 |
| `sentinel2` | B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 | B4 | B3 | B2 | B8 |
| `hls` | B1, B2, B3, B4, B5, B6, B7 | B4 | B3 | B2 | B5 |
| `naip` | R, G, B, N | R | G | B | N |
| `spot` | R, G, B | R | G | B | -- |

For `local` sources, band mapping can be overridden via the `bands` configuration parameter.

## Compositing Methods

When building an annual composite from a time series of scenes, agribound supports three methods:

### Median Composite (`median`)

Takes the per-pixel median value across all cloud-free scenes in the date range. This is the default method and produces stable, artifact-free composites.

```python
config = AgriboundConfig(
    source="sentinel2",
    composite_method="median",
    year=2024,
    ...
)
```

### Greenest Pixel (`greenest`)

Selects the pixel from the scene with the highest NDVI value. Best for capturing peak vegetation in agricultural regions.

```python
config = AgriboundConfig(
    source="sentinel2",
    composite_method="greenest",
    year=2024,
    ...
)
```

### Max NDVI (`max_ndvi`)

Computes the per-pixel maximum NDVI across all scenes. Similar to greenest pixel but operates on the NDVI band directly rather than selecting full scenes.

```python
config = AgriboundConfig(
    source="sentinel2",
    composite_method="max_ndvi",
    year=2024,
    ...
)
```

## Cloud Masking

Cloud masking is applied automatically for each GEE-based source using source-specific quality bands. The `cloud_cover_max` parameter (default 20) filters out scenes with excessive cloud cover before compositing.

```python
config = AgriboundConfig(
    source="sentinel2",
    cloud_cover_max=10,  # stricter filtering
    ...
)
```

- **Landsat**: Uses the `QA_PIXEL` band (CFMask algorithm).
- **Sentinel-2**: Uses the `QA60` band and `SCL` (Scene Classification Layer).
- **HLS**: Uses the `Fmask` quality band.
- **NAIP**: No cloud masking (clear-sky acquisitions).
- **SPOT**: Uses the `CLOUD_MASK` band.

## Date Range Configuration

By default, agribound uses the full calendar year specified by the `year` parameter. To restrict compositing to a specific growing season or date window:

```python
config = AgriboundConfig(
    source="sentinel2",
    year=2024,
    date_range=("2024-04-01", "2024-09-30"),  # growing season only
    ...
)
```

!!! tip
    Restricting to the growing season often produces better delineation results because field boundaries are most visible when crops are actively growing.

## Embedding Sources

The `google-embedding` and `tessera-embedding` sources provide pre-computed pixel-level embeddings rather than raw imagery. These are used exclusively with the `embedding` engine and do not require GEE authentication.

- **Google Satellite Embedding V1**: 64-dimensional embeddings at 10 m resolution, derived from high-resolution satellite imagery.
- **TESSERA Embeddings**: 128-dimensional embeddings at 10 m resolution, from the TESSERA model (CVPR 2026).
