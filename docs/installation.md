# Installation

## Requirements

- Python >= 3.10
- GDAL, PROJ, and GEOS (for rasterio, fiona, and geopandas)

## Recommended: Conda + pip

Geospatial packages like rasterio, fiona, and geopandas depend on system-level C libraries (GDAL, PROJ, GEOS) that can be difficult to install via pip alone. We recommend creating a conda environment first, then installing agribound with pip:

```bash
conda create -n agribound python=3.12 rasterio geopandas fiona shapely pyproj -c conda-forge
conda activate agribound
pip install agribound
```

This ensures all binary dependencies are properly resolved. You can then add optional extras via pip as described below.

## Alternative: pip Only

If you have a working GDAL installation or prefer a pure pip workflow:

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
conda create -n agribound-dev python=3.12 rasterio geopandas fiona shapely pyproj -c conda-forge
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
