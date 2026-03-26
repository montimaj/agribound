# Installation

## Requirements

- Python >= 3.10
- GDAL Python bindings (`osgeo`), PROJ, and GEOS (for rasterio, fiona, geopandas, and geedim)

## Recommended: Conda + pip

Geospatial packages like rasterio, fiona, and geopandas depend on system-level C libraries (GDAL, PROJ, GEOS) that can be difficult to install via pip alone. The `gdal` conda package also provides the GDAL Python bindings (`osgeo`) required by `geedim` for downloading satellite composites from Google Earth Engine.

```bash
conda create -n agribound python=3.12 gdal rasterio geopandas fiona shapely pyproj -c conda-forge
conda activate agribound
pip install agribound
```

!!! warning "GDAL Python bindings"
    Installing `libgdal` or `libgdal-core` alone is **not** sufficient. You need the full `gdal` conda package to get the `osgeo` Python module. Without it, GEE composite downloads will fail with `No module named 'osgeo'`.

This ensures all binary dependencies are properly resolved. You can then add optional extras via pip as described below.

## Alternative: pip Only

If you have system GDAL with Python bindings already available:

```bash
pip install agribound
```

## Optional Extras

Agribound uses optional dependency groups to keep the base install lightweight. Install only what you need:

### Google Earth Engine

Required for all GEE-based satellite sources (Landsat, Sentinel-2, HLS, NAIP, SPOT):

```bash
pip install agribound[gee]
```

### Delineate-Anything

YOLO-based instance segmentation engine. Works across resolutions from 1 m to 10 m+:

```bash
pip install agribound[delineate-anything]
```

### Fields of The World (FTW)

Semantic segmentation with 16+ pre-trained models covering 25 countries:

```bash
pip install agribound[ftw]
```

### GeoAI

Mask R-CNN instance segmentation with built-in NDVI support:

```bash
pip install agribound[geoai]
```

### Prithvi

NASA/IBM Prithvi-EO-2.0 foundation model with UPerNet decoder:

```bash
pip install agribound[prithvi]
```

### TESSERA

TESSERA embedding-based delineation:

```bash
pip install agribound[tessera]
```

### Everything

Install all engine extras and GEE support:

```bash
pip install agribound[all]
```

### Documentation

To build the docs locally:

```bash
pip install agribound[docs]
```

### Development

For testing and linting:

```bash
pip install agribound[dev]
```

## Development Install from Source

Clone the repository and install in editable mode with development dependencies:

```bash
conda create -n agribound-dev python=3.12 gdal rasterio geopandas fiona shapely pyproj -c conda-forge
conda activate agribound-dev
git clone https://github.com/montimaj/agribound.git
cd agribound
pip install -e ".[all,dev,docs]"
```

## Verifying the Installation

```bash
agribound --version
agribound list-engines
agribound list-sources
```
