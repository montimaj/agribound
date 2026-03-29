# Satellite Sources

Agribound supports multiple satellite imagery sources for building annual composites. Each source is accessed through Google Earth Engine or loaded from local files.

## Source Overview

All spectral bands are downloaded for each sensor. Engines automatically extract and reorder the bands they need via canonical band mappings (e.g., FTW expects R, G, B, NIR as bands 1--4 matching its `B04, B03, B02, B08` training order, so agribound extracts those from the full composite before passing to FTW).

| Source | Key | Resolution | Bands Downloaded | GEE Collection | Notes |
|---|---|---|---|---|---|
| Sentinel-2 | `sentinel2` | 10 m | B1--B12, B8A (12 bands) | `COPERNICUS/S2_SR_HARMONIZED` | Default source; L2A surface reflectance |
| Landsat | `landsat` | 30 m | SR_B2--SR_B7 (6 bands) | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | Long time-series; L5/7 bands harmonized to L8/9 naming |
| HLS | `hls` | 30 m | B1--B7 (7 bands) | `NASA/HLS/HLSL30/v002`, `NASA/HLS/HLSS30/v002` | Harmonized Landsat+Sentinel-2 |
| NAIP | `naip` | 1 m | R, G, B, N (4 bands) | `USDA/NAIP/DOQQ` | 4-band (RGBN); best for small fields. **Very slow over large areas** |
| SPOT 6/7 | `spot` | 1.5 m | R, G, B (3 bands) | Restricted -- see below | Restricted GEE collection. **Very slow over large areas** |
| Local GeoTIFF | `local` | Any | All bands | N/A | Bring your own imagery via `--local-tif` |
| Google Embeddings | `google-embedding` | 10 m | 64-D embeddings | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | Pre-computed satellite embeddings |
| TESSERA Embeddings | `tessera-embedding` | 10 m | 128-D embeddings | N/A | TESSERA foundation model embeddings |

!!! note "SPOT Access"
    SPOT 6/7 imagery (`AIRBUS/SPOT6_7`) is restricted to select GEE users and is for internal DRI use only. External users who need SPOT-based field boundaries should contact the package author.

!!! warning "High-Resolution Source Performance"
    NAIP (1 m) and SPOT (1.5 m) produce rasters that are 100–900x larger in pixel count than Sentinel-2 (10 m) for the same study area. Composite download, fine-tuning, and inference on these sources over county-scale or larger areas can take hours even on GPU. Consider subsetting the study area for high-resolution sources, or use Sentinel-2/Landsat for large-area mapping.

## Canonical Band Mappings

Each engine extracts the bands it needs from the full composite using these canonical mappings:

| Source | Canonical R | Canonical G | Canonical B | Canonical NIR |
|---|---|---|---|---|
| `landsat` | SR_B4 | SR_B3 | SR_B2 | SR_B5 |
| `sentinel2` | B4 | B3 | B2 | B8 |
| `hls` | B4 | B3 | B2 | B5 |
| `naip` | R | G | B | N |
| `spot` | R | G | B | -- |

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
- **TESSERA Embeddings**: 128-dimensional embeddings at 10 m resolution, from the TESSERA foundation model ([Feng et al., 2025](https://arxiv.org/abs/2506.20380)). Coverage varies by region and year (2017–2025) — not all areas have data for every year. See [geotessera](https://github.com/ucam-eo/geotessera) for coverage details and the download tool.
