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

Alternatively, you can use the provided [`environment.yml`](https://github.com/montimaj/agribound/blob/main/environment.yml) for a one-step setup. Download it from the repository (or clone the repo) and run:

```bash
conda env create -f environment.yml
conda activate agribound
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

Semantic segmentation with 14+ pre-trained models covering 25 countries:

```bash
pip install agribound[ftw]
```

!!! note "Delineate-Anything on Sentinel-2"
    For Sentinel-2, the `delineate-anything` engine routes through FTW's built-in instance segmentation with proper S2 preprocessing and native MPS (Apple GPU) support. This requires the **development version** of ftw-baselines (not yet on PyPI):

    ```bash
    git clone https://github.com/fieldsoftheworld/ftw-baselines.git
    pip install -e ftw-baselines
    ```

    Without this, DA on Sentinel-2 will error. DA on all other sensors (Landsat, NAIP, HLS, SPOT, local) works without this step. Once `ftw-tools` v2.0+ is released on PyPI, this extra install step will no longer be needed.

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

# Required for DA instance segmentation on Sentinel-2
git clone https://github.com/fieldsoftheworld/ftw-baselines.git ../ftw-baselines
pip install -e ../ftw-baselines
```

## Verifying the Installation

```bash
agribound --version
agribound list-engines
agribound list-sources
```

## Troubleshooting

### Dependency Conflicts

If you see errors like `pip's dependency resolver does not currently take into account all the packages that are installed`, this typically means packages in your environment have
conflicting version requirements. Common culprits include `fsspec`, `lightning`, and
`pytorch-lightning`.

**Solution: use a fresh conda environment.** Installing into a clean environment avoids
inheriting conflicting packages from a base or shared environment:

```bash
conda create -n agribound python=3.12 gdal rasterio geopandas fiona shapely pyproj -c conda-forge
conda activate agribound
pip install "agribound[all]"
```

!!! tip "Verify which pip is active"
    After activating the conda environment, confirm that `pip` points to the
    environment and not a system-level installation:

    ```bash
    which pip        # Linux/macOS
    where pip        # Windows
    pip --version
    ```

    The path should include the name of your conda environment (e.g.,
    `.../envs/agribound/bin/pip`).

### Conflicts with ftw-baselines (dev)

The development version of `ftw-baselines` (`ftw-tools` v2.x beta) may have different
dependency constraints (e.g., `lightning<2.6`) that conflict with other engines such as
Prithvi (which requires `lightning>=2.6` via `terratorch`). If you need both:

1. Install agribound first, then install ftw-baselines:

    ```bash
    pip install "agribound[all]"
    git clone https://github.com/fieldsoftheworld/ftw-baselines.git
    pip install -e ftw-baselines
    ```

2. If conflicts persist, consider installing only the extras you need rather than `[all]`:

    ```bash
    pip install "agribound[ftw,delineate-anything,gee]"
    ```

### Installing from Source as a Last Resort

If dependency issues persist, build and install directly from the repository:

```bash
conda create -n agribound-dev python=3.12 gdal rasterio geopandas fiona shapely pyproj -c conda-forge
conda activate agribound-dev
git clone https://github.com/montimaj/agribound.git
cd agribound
pip install -e ".[all,dev]"
```
